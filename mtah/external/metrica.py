"""Metrica public sample games 1 and 2 -> mtah.schema.Game (clean tracking + event-derived GT).

    python -m mtah.external.metrica            # builds $MTAH_CACHE/p2/games/metrica{1,2}_{events,control}/

Two label conventions are produced (WORKLOG step 6, open question on uncontrolled touches):
  events   tools/events_to_labels.py as-is: any RECOVERY / BALL LOST changes possession.
  control  same, then every short spell (< CONTROL_MIN) that follows the other team's spell and
           contains no deliberate action by its team (PASS, SHOT, or a BALL LOST that is a failed
           pass) is given back to the previous team: a header or a deflection is not control.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import sample_data  # noqa: E402
import events_to_labels  # noqa: E402

from mtah.schema import Game, HOME, AWAY, possession_summary, runs, speed_from_xy  # noqa: E402

from mtah import config  # noqa: E402

CACHE = str(config.EXTERNAL.GAMES_DIR)
CONTROL_MIN = 25          # frames (1 s)
UNCONTROLLED_LOSS = ("HEAD", "CLEARANCE")   # BALL LOST subtypes that are a touch, not a pass


def _deliberate(ev):
    if ev.type in ("PASS", "SHOT"):
        return True
    if ev.type == "BALL LOST":
        return not any(w in ev.subtype for w in UNCONTROLLED_LOSS)
    return False


def require_control(labels, events, min_frames=CONTROL_MIN):
    """Return labels with uncontrolled short spells merged into the previous team's spell."""
    out = labels.copy()
    starts, stops, vals = runs(out)
    by_start = events.sort_values("start")
    ev_frames = by_start.start.values.astype(int) - 1      # label row = frame - 1
    changed = 0
    for i in range(1, len(starts)):
        team = vals[i]
        prev = vals[i - 1]
        if team not in ("H", "A") or prev not in ("H", "A") or prev == team:
            continue
        s, e = starts[i], stops[i]
        if e - s >= min_frames:
            continue
        lo, hi = np.searchsorted(ev_frames, [s, e])
        if any(_deliberate(ev) for ev in by_start.iloc[lo:hi].itertuples() if ev.team == team):
            continue
        out[s:e] = prev
        vals[i] = prev  # so a following spell compares against the merged team
        changed += 1
    return out, changed


def load(game, convention="events"):
    track = sample_data.tracking(game)
    n = len(track["frame"])
    L, W = sample_data.PITCH
    home = sample_data.to_metres(track["home"])
    away = sample_data.to_metres(track["away"])
    xy = np.concatenate([home, away], axis=1)
    team = np.r_[np.full(home.shape[1], HOME), np.full(away.shape[1], AWAY)].astype(np.int8)

    halves = []
    for p in sorted(np.unique(track["period"])):
        idx = np.flatnonzero(track["period"] == p)
        halves.append((int(idx[0]), int(idx[-1]) + 1))

    players = np.empty((n, xy.shape[1], 3), np.float32)
    players[..., :2] = xy
    players[..., 2] = speed_from_xy(xy, halves)
    ball = np.full((n, 3), np.nan, np.float32)
    ball[:, :2] = sample_data.to_metres(track["ball"])

    labels, stoppages, notes = events_to_labels.convert(game)
    changed = 0
    if convention == "control":
        labels, changed = require_control(labels, sample_data.events(game))
    elif convention != "events":
        raise ValueError(convention)

    # attack direction from the first-half centroids
    a, b = halves[0]
    with np.errstate(all="ignore"):
        hx = np.nanmean(home[a:b, :, 0])
        ax = np.nanmean(away[a:b, :, 0])
    g = Game(name="metrica{}_{}".format(game, convention), players=players, team=team, ball=ball,
             halves=halves, pitch=(L, W), home_attacks_right_first_half=bool(hx < ax),
             labels=labels,
             meta={"source": "metrica-sports/sample-data Sample_Game_{}".format(game),
                   "licence": "public sample data, no formal licence; acknowledged",
                   "labels": "tools/events_to_labels.py (human event log)",
                   "convention": convention, "control_merged_spells": changed,
                   "clean": True, "ball_missing_when_dead": True})
    g.validate()
    return g, stoppages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", type=int, nargs="+", default=list(sample_data.GAMES))
    args = ap.parse_args()
    for game in args.game:
        for conv in ("events", "control"):
            g, stoppages = load(game, conv)
            folder = g.save(os.path.join(CACHE, g.name))
            stoppages.to_csv(os.path.join(folder, "stoppages.csv"), index=False)
            print(g.name, "orientation", [round(x, 3) for x in g.orientation_check()],
                  "merged", g.meta["control_merged_spells"])
            print("   ", json.dumps(possession_summary(g.labels)))


if __name__ == "__main__":
    main()
