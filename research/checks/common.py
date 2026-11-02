"""Shared loaders for the verification checks. Nothing here prints frame-level values.

The raw EPTS text is parsed the way starter/baseline.py does it ("<frame>:<p>;<p>;...:<ball>",
each group "x,y,speed", NaN where missing) and cached as .npz in a cache folder OUTSIDE the repo,
so each check costs one parse rather than one per run.

    MTAH_CACHE=/path/to/scratch python research/checks/<check>.py
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, asset_paths, half_video_blocks, source_frames  # noqa: E402

# Params
CACHE = os.environ.get("MTAH_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "mtah2027")
L, W = GAMES[GAME]["pitch"]
FPS = GAMES[GAME]["fps"]
HOME, AWAY = 0, 1


def graded_mask(n=None):
    n = n or source_frames("tactical")
    g = np.zeros(n, dtype=bool)
    for _, start, stop in half_video_blocks(GAME):
        g[start:stop + 1] = True
    return g


def half_masks(n=None):
    n = n or source_frames("tactical")
    out = []
    for _, start, stop in half_video_blocks(GAME):
        m = np.zeros(n, dtype=bool)
        m[start:stop + 1] = True
        out.append(m)
    return out


def slot_teams(source="tactical"):
    """Per slot: 0 home, 1 away (first Team in the XML is home), plus position_type strings."""
    root = ET.parse(asset_paths(source=source)["atd_meta"]).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    teams, ptypes = [], []
    for p in root.iter("Player"):
        teams.append(HOME if p.get("teamId") == team_ids[0] else AWAY)
        ptype = "?"
        for pp in p.iter("ProviderParameter"):
            if pp.findtext("Name") == "position_type":
                ptype = pp.findtext("Value")
        ptypes.append(ptype)
    return np.array(teams, dtype=np.int8), ptypes


def _parse(raw_path, n, n_slots):
    px = np.full((n, n_slots), np.nan, dtype=np.float32)
    py = np.full((n, n_slots), np.nan, dtype=np.float32)
    ps = np.full((n, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < n:
                continue
            b = parts[2].split(",")
            if b[0][:1] != "N":
                ball[f] = float(b[0]), float(b[1])
            for s, grp in enumerate(parts[1].split(";")):
                if grp[:1] == "N":
                    continue
                q = grp.split(",")
                px[f, s], py[f, s] = float(q[0]), float(q[1])
                if q[2][:1] != "N":
                    ps[f, s] = float(q[2])
    return px, py, ps, ball


def load_atd(source="tactical"):
    """dict: x, y (metres, n x slots), speed (m/s), bx, by (metres), team, ptype, n."""
    team, ptype = slot_teams(source)
    n = source_frames(source)
    cache = os.path.join(CACHE, "atd_{}.npz".format(source))
    if os.path.exists(cache):
        z = np.load(cache)
        px, py, ps, ball = z["px"], z["py"], z["ps"], z["ball"]
    else:
        px, py, ps, ball = _parse(asset_paths(source=source)["atd_raw"], n, len(team))
        os.makedirs(CACHE, exist_ok=True)
        np.savez(cache, px=px, py=py, ps=ps, ball=ball)
    return {"x": px * L, "y": py * W, "speed": ps, "bx": ball[:, 0] * L, "by": ball[:, 1] * W,
            "team": team, "ptype": ptype, "n": n}


def runs(mask):
    """(starts, ends_exclusive) of maximal True runs."""
    m = np.concatenate(([False], np.asarray(mask, bool), [False]))
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def load_tagging():
    from load_smart_tagging import load
    return load()


def jaccard(a, b):
    u = np.sum(a | b)
    return np.sum(a & b) / u if u else float("nan")


def rolling_mean(a, half_window):
    """Centred mean over +/- half_window frames, NaN-aware."""
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, a, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(a))
    lo, hi = np.maximum(0, i - half_window), np.minimum(len(a), i + half_window + 1)
    cnt = k[hi] - k[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (s[hi] - s[lo]) / cnt, np.nan)


def camera_step():
    """Per-frame movement (m) of the tactical camera aim: image centre back-projected to the pitch."""
    from load_homography import load as load_h
    H, _, valid = load_h()
    Hinv = np.linalg.inv(np.where(valid[:, None, None], H, np.eye(3)))
    v = Hinv @ np.array([960.0, 540.0, 1.0])
    aim = np.stack([v[:, 0] / v[:, 2] * L, v[:, 1] / v[:, 2] * W], 1)
    aim[~valid] = np.nan
    step = np.full(len(aim), np.nan)
    step[1:] = np.linalg.norm(np.diff(aim, axis=0), axis=1)
    return step


def nearest_team(d):
    """Per frame: team of the tracked object nearest the ball, -1 where the ball is missing."""
    vis = np.isfinite(d["bx"])
    dist = np.hypot(d["x"] - d["bx"][:, None], d["y"] - d["by"][:, None])
    dist[~np.isfinite(dist)] = np.inf
    out = np.full(d["n"], -1, dtype=np.int8)
    out[vis] = d["team"][np.argmin(dist[vis], axis=1)]
    return out


def tagging_masks(n, code_filter=None):
    """Boolean frame mask covered by Smart Tagging instances whose code passes code_filter."""
    m = np.zeros(n, dtype=bool)
    for ev in load_tagging():
        if code_filter is None or code_filter(ev["code"].strip().upper()):
            m[max(0, ev["start_frame"]):min(n, ev["end_frame"])] = True
    return m
