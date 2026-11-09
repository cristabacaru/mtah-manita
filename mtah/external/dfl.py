"""DFL / IDSSE open dataset (Bassek et al. 2025, CC BY 4.0) -> mtah.schema.Game.

Seven Bundesliga / 2. Bundesliga matches with full-pitch tracking at 25 fps and per-frame provider
flags on the ball:  BallStatus (1 alive, 0 dead)  and  BallPossession (1 home, 2 guest).
Those flags are the ground truth: D when dead, otherwise the owning team.

Download (raw files are not redistributed; data/ is gitignored):
    python -m mtah.external.dfl --download
Build the cached Games and print the GT report:
    python -m mtah.external.dfl

The parser streams the positions XML with lxml.iterparse (kloppy builds Python objects per frame
and needs far more memory and time on 600 MB files). kloppy can be used to cross-check one match.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from lxml import etree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from mtah.schema import Game, HOME, AWAY, possession_summary, runs, speed_from_xy  # noqa: E402

from mtah import config  # noqa: E402

RAW = str(config.EXTERNAL.DFL_DIR)
CACHE = str(config.EXTERNAL.GAMES_DIR)
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
SECTIONS = ("firstHalf", "secondHalf")
# kloppy._providers.sportec.get_IDSSE_url gives springernature ndownloader links, which sit behind
# a bot challenge; the figshare API serves the same file ids directly.
API = "https://api.figshare.com/v2/file/download/{}"
DEBOUNCE = 0   # frames; H/A runs shorter than this inside a live stretch are merged (0 = off)
SNAP_WINDOW = 25   # frames; a live->D flip this soon after the ball leaves the pitch is moved
                   # back to the crossing frame (our convention; DFL flags lag it by ~3-4 frames)


def download():
    from kloppy._providers.sportec import get_IDSSE_url
    os.makedirs(RAW, exist_ok=True)
    for m in MATCHES:
        for kind in ("meta", "tracking"):
            path = os.path.join(RAW, "{}_{}.xml".format(m, kind))
            if os.path.exists(path) and os.path.getsize(path) > 0:
                continue
            fid = get_IDSSE_url(m, kind).rsplit("/", 1)[1]
            subprocess.check_call(["curl", "-sSL", "--retry", "3", "-o", path + ".part",
                                   API.format(fid)])
            os.replace(path + ".part", path)
            print(path, os.path.getsize(path))


def read_meta(match):
    root = etree.parse(os.path.join(RAW, "{}_meta.xml".format(match))).getroot()
    env = root.find(".//Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    teams = {}
    for t in root.iter("Team"):
        role = t.get("Role")
        teams[t.get("TeamId")] = {"role": role, "name": t.get("TeamName"),
                                  "players": [p.get("PersonId") for p in t.iter("Player")]}
    return pitch, teams


def match_info(match):
    """Competition, season, matchday, title and result from the <General> element."""
    root = etree.parse(os.path.join(RAW, "{}_meta.xml".format(match))).getroot()
    gen = root.find(".//General")
    keys = ("CompetitionName", "Season", "MatchDay", "MatchTitle", "Result", "KickoffTime")
    return {k[0].lower() + k[1:]: gen.get(k) for k in keys} if gen is not None else {}


# DFL/Sportec tracking: origin at the centre, y bottom-to-top (kloppy
# SportecTrackingDataCoordinateSystem). The ATD / Metrica format is origin top-left, y top-to-bottom,
# so y is mirrored: y_atd = W/2 - Y. (Before 2026-11-07 the cache used Y + W/2: a mirror image.)
def read_positions(match):
    """{(section, team_id, person_id): (N int array, X, Y, Z|None, poss|None, status|None)}."""
    path = os.path.join(RAW, "{}_tracking.xml".format(match))
    out = {}
    for _, fs in etree.iterparse(path, events=("end",), tag="FrameSet", huge_tree=True):
        sec, team, person = fs.get("GameSection"), fs.get("TeamId"), fs.get("PersonId")
        frames = fs.findall("Frame")
        N = np.fromiter((int(f.get("N")) for f in frames), np.int64, len(frames))
        X = np.fromiter((float(f.get("X")) for f in frames), np.float32, len(frames))
        Y = np.fromiter((float(f.get("Y")) for f in frames), np.float32, len(frames))
        Z = poss = status = None
        if team.upper() == "BALL":
            Z = np.fromiter((float(f.get("Z", "nan")) for f in frames), np.float32, len(frames))
            poss = np.fromiter((int(f.get("BallPossession", "0")) for f in frames), np.int8,
                               len(frames))
            status = np.fromiter((int(f.get("BallStatus", "0")) for f in frames), np.int8,
                                 len(frames))
        out[(sec, team, person)] = (N, X, Y, Z, poss, status)
        fs.clear()
        while fs.getprevious() is not None:
            del fs.getparent()[0]
    return out


def debounce(labels, k):
    """Merge H/A runs shorter than k frames that sit between two runs of the other... any
    live label: they are flag flicker, not possessions."""
    if k <= 0:
        return labels, 0
    out = labels.copy()
    s, e, v = runs(out)
    n = 0
    for i in range(1, len(s) - 1):
        if v[i] in "HA" and e[i] - s[i] < k and v[i - 1] in "HA":
            out[s[i]:e[i]] = v[i - 1]
            v[i] = v[i - 1]
            n += 1
    return out, n


def snap_out_starts(labels, ball, pitch, window=SNAP_WINDOW):
    """Move each live->D change back to the frame the ball left the pitch, if that happened in
    the `window` frames before it. Same rule as the Metrica converter (WORKLOG step 6)."""
    L, W = pitch
    bx, by = ball[:, 0], ball[:, 1]
    out = (bx < 0) | (bx > L) | (by < 0) | (by > W)
    labels = labels.copy()
    s, e, v = runs(labels)
    moved = []
    for i in range(1, len(s)):
        if v[i] != "D" or v[i - 1] not in ("H", "A"):
            continue
        t = s[i]
        lo = max(s[i - 1], t - window)
        w = np.flatnonzero(out[lo:t + 1])
        if len(w) == 0:
            continue
        # first frame of the last outside stretch that runs into t
        k = lo + w[-1]
        while k - 1 >= lo and out[k - 1]:
            k -= 1
        if k < t:
            labels[k:t] = "D"
            moved.append(int(t - k))
    return labels, moved


def load(match, debounce_frames=DEBOUNCE, snap=True):
    pitch, teams = read_meta(match)
    L, W = pitch
    pos = read_positions(match)
    home_id = next(t for t, d in teams.items() if d["role"] == "home")
    away_id = next(t for t, d in teams.items() if d["role"] == "guest")
    ball_keys = {k[0]: k for k in pos if k[1].upper() == "BALL"}

    # frame index: half 1 frames, one X separator frame, half 2 frames
    halves, base = [], {}
    cursor = 0
    for sec in SECTIONS:
        N = pos[ball_keys[sec]][0]
        lo, hi = int(N.min()), int(N.max())
        if cursor:
            cursor += 1
        base[sec] = cursor - lo
        halves.append((cursor, cursor + hi - lo + 1))
        cursor += hi - lo + 1
    n = cursor

    persons = [(home_id, p) for p in teams[home_id]["players"]] + \
              [(away_id, p) for p in teams[away_id]["players"]]
    seen = {(k[1], k[2]) for k in pos}
    persons = [p for p in persons if p in seen]   # players who never took the field are dropped
    slot = {p: i for i, p in enumerate(persons)}
    team = np.array([HOME if t == home_id else AWAY for t, _ in persons], np.int8)

    xy = np.full((n, len(persons), 2), np.nan, np.float32)
    ball = np.full((n, 3), np.nan, np.float32)
    labels = np.full(n, "X", "<U1")
    for (sec, tid, pid), (N, X, Y, Z, poss, status) in pos.items():
        if sec not in base:
            continue
        idx = N + base[sec]
        if tid.upper() == "BALL":
            ball[idx, 0], ball[idx, 1], ball[idx, 2] = X + L / 2, W / 2 - Y, Z
            lab = np.where(status == 1, np.where(poss == 1, "H", np.where(poss == 2, "A", "D")),
                           "D")
            labels[idx] = lab
        elif (tid, pid) in slot:
            s = slot[(tid, pid)]
            xy[idx, s, 0], xy[idx, s, 1] = X + L / 2, W / 2 - Y

    # anything inside a half the ball frameset did not cover stays X -> make it D (no evidence)
    for a, b in halves:
        seg = labels[a:b]
        seg[seg == "X"] = "D"
    labels, merged = debounce(labels, debounce_frames)
    moved = []
    if snap:
        labels, moved = snap_out_starts(labels, ball, pitch)

    players = np.empty((n, len(persons), 3), np.float32)
    players[..., :2] = xy
    players[..., 2] = speed_from_xy(xy, halves)

    a, b = halves[0]
    with np.errstate(all="ignore"):
        hx = np.nanmean(xy[a:b, team == HOME, 0])
        ax = np.nanmean(xy[a:b, team == AWAY, 0])
    g = Game(name="dfl_{}".format(match), players=players, team=team, ball=ball, halves=halves,
             pitch=(L, W), home_attacks_right_first_half=bool(hx < ax), labels=labels,
             meta={"source": "DFL/IDSSE open data (Bassek et al. 2025), match {}".format(match),
                   "licence": "CC BY 4.0",
                   "home": teams[home_id]["name"], "away": teams[away_id]["name"],
                   "match": match_info(match),
                   "labels": "provider flags BallStatus / BallPossession",
                   "debounce_frames": debounce_frames, "debounced_runs": merged,
                   "snapped_out_starts": len(moved),
                   "snap_median_frames": float(np.median(moved)) if moved else 0.0,
                   "clean": True, "ball_missing_when_dead": False,
                   "y_axis": "top_to_bottom"})
    g.validate()
    return g


def flag_checks(g):
    """Independent sanity checks of the flags against the tracking."""
    live = np.isin(g.labels, ["H", "A"])
    bx, by = g.ball[:, 0], g.ball[:, 1]
    d = np.hypot(g.players[..., 0] - bx[:, None], g.players[..., 1] - by[:, None])
    with np.errstate(all="ignore"):
        nearest = np.nanargmin(np.where(np.isfinite(d), d, np.inf), axis=1)
        dmin = np.nanmin(d, axis=1)
    near_team = g.team[nearest]
    close = live & (dmin < 1.5)
    agree = float(((near_team == HOME) == (g.labels == "H"))[close].mean())
    L, W = g.pitch
    outside = (bx < -0.5) | (bx > L + 0.5) | (by < -0.5) | (by > W + 0.5)
    return {"nearest_within_1.5m_agrees": round(agree, 3),
            "live_with_ball_outside": round(float((outside & live).sum() / live.sum()), 4),
            "dead_with_ball_outside": round(float((outside & (g.labels == "D")).sum()
                                                  / max(1, (g.labels == "D").sum())), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--match", nargs="+", default=list(MATCHES))
    ap.add_argument("--debounce", type=int, default=DEBOUNCE)
    args = ap.parse_args()
    if args.download:
        download()
        return
    report = {}
    for m in args.match:
        g = load(m, args.debounce)
        g.save(os.path.join(CACHE, g.name))
        s = possession_summary(g.labels)
        s.update(flag_checks(g))
        s["orientation"] = [round(x, 3) for x in g.orientation_check()]
        report[g.name] = s
        print(g.name, g.meta["home"], "-", g.meta["away"], json.dumps(s), flush=True)
    with open(os.path.join(str(config.EXTERNAL.P2_DIR), "dfl_gt_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
