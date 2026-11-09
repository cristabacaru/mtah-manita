"""Check the labels derived from a sample match against its clean tracking.

The converter (tools/events_to_labels.py) only reads the event log. The sample tracking is clean
and independent of it, so it can test the labels: if a rule in the converter is wrong, the ball
and the players will disagree with it in the same way every time that rule fires. Each check
flags the frames where they disagree; only those need looking at.

    python tools/events_to_labels.py          # first
    python tools/check_labels.py              # games 1 and 2
    python tools/check_labels.py --game 2 --show 20

Writes data/validation/game<N>/game<N>_flags.csv (frame, check, detail) and prints a summary.

The checks
----------
    out_crossing         a ball-out / shot-out / goal stoppage should start on the frame the ball
                         leaves the pitch. Reports the offset; flags more than OUT_TOLERANCE.
    restart_position     a throw-in is taken on a touchline, a corner at a corner, a goal kick in
                         the goal area, a kickoff on the centre spot, a penalty on a penalty spot.
    restart_team         the player nearest the ball at a restart belongs to the restarting team.
    possession_agrees    inside a possession, the player nearest a visible ball should mostly be
                         from the team in possession. Flags possessions below MIN_AGREEMENT.
    ball_moving_while_dead   the ball is tracked, on the pitch and travelling fast for a second
                         or more during D: the stoppage probably starts too early or ends too late.
    ball_missing_while_live  the sample data removes the ball while it is out of play, so a long
                         gap during H/A suggests a stoppage the log did not record.
    inferred             places where the converter had to guess (from game<N>_notes.csv).

Last, the official baseline is run on the clean tracking and scored against the labels. It should
score well above its competition result -- clean tracking, and a ball that vanishes exactly when
play stops -- so a low score here means the labels and the tracking do not line up.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_data  # noqa: E402

sys.path.insert(0, os.path.join(sample_data.ROOT, "evaluation"))
sys.path.insert(0, os.path.join(sample_data.ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
from baseline import smooth, HOME, AWAY, NOBODY  # noqa: E402

# Params
LENGTH, WIDTH = sample_data.PITCH
FPS = sample_data.FPS

OUT_WINDOW = 75            # frames either side of a stoppage start to look for the crossing
OUT_TOLERANCE = 12         # frames; ~0.5 s
RESTART_WINDOW = 15        # frames either side of a restart to find a tracked ball
THROW_IN_MAX = 2.5         # metres from a touchline
CORNER_MAX = 3.0           # metres from a corner
GOAL_AREA = (5.5 + 1.5, 9.16 + 1.5)  # metres: depth from goal line, half-width; +1.5 m slack
SPOT_MAX = 2.0             # metres from the centre or a penalty spot
MIN_POSSESSION = 2 * FPS   # possessions shorter than this are not checked for agreement
MIN_VISIBLE = FPS          # ...nor ones with fewer visible-ball frames than this
MIN_AGREEMENT = 0.5
MOVING_SPEED = 3.0         # m/s
MOVING_FRAMES = FPS        # a second of fast ball movement during D
MISSING_FRAMES = 4 * FPS   # four seconds without a ball during H/A
BASELINE_WINDOW = 50       # the published floor's smoothing


def runs(mask):
    """[(start, stop)] half-open index ranges where mask is True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def nearest_teams(track):
    """(nearest team code per frame, HOME/AWAY/NOBODY; ball in metres)."""
    ball = sample_data.to_metres(track["ball"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows when the ball is missing
        home = np.nanmin(np.linalg.norm(sample_data.to_metres(track["home"]) - ball[:, None],
                                        axis=2), axis=1)
        away = np.nanmin(np.linalg.norm(sample_data.to_metres(track["away"]) - ball[:, None],
                                        axis=2), axis=1)
    codes = np.full(len(ball), NOBODY, dtype=np.int8)
    seen = np.isfinite(home) & np.isfinite(away)
    codes[seen] = np.where(home[seen] <= away[seen], HOME, AWAY)
    return codes, ball


def nearest_visible(visible, index, window):
    """Index of the tracked-ball frame closest to index within +/- window, or None."""
    low, high = max(0, index - window), min(len(visible), index + window + 1)
    candidates = np.flatnonzero(visible[low:high]) + low
    if not len(candidates):
        return None
    return int(candidates[np.argmin(np.abs(candidates - index))])


def restart_ok(kind, x, y):
    """(ok, detail) for the ball position at a restart, in metres. None when nothing to check."""
    if "THROW IN" in kind:
        distance = min(y, WIDTH - y)
        return distance <= THROW_IN_MAX, "{:.1f} m from a touchline".format(distance)
    if "CORNER" in kind:
        distance = min(np.hypot(x - cx, y - cy) for cx in (0, LENGTH) for cy in (0, WIDTH))
        return distance <= CORNER_MAX, "{:.1f} m from a corner".format(distance)
    if kind in ("goal kick", "recovery"):
        depth, offset = min(x, LENGTH - x), abs(y - WIDTH / 2)
        ok = depth <= GOAL_AREA[0] and offset <= GOAL_AREA[1]
        return ok, "{:.1f} m from the goal line, {:.1f} m off centre".format(depth, offset)
    if "KICK OFF" in kind:
        distance = np.hypot(x - LENGTH / 2, y - WIDTH / 2)
        return distance <= SPOT_MAX, "{:.1f} m from the centre spot".format(distance)
    if "PENALTY" in kind:
        distance = min(np.hypot(x - spot, y - WIDTH / 2) for spot in (11.0, LENGTH - 11.0))
        return distance <= SPOT_MAX, "{:.1f} m from a penalty spot".format(distance)
    return None, ""


def check(game):
    """(flags DataFrame, report lines) for one game."""
    stem = os.path.join(sample_data.game_dir(game), "game{}_".format(game))
    labels = pd.read_csv(stem + "gt.csv")["label"].values.astype("<U1")
    stoppages = pd.read_csv(stem + "stoppages.csv").fillna("")
    notes = pd.read_csv(stem + "notes.csv")
    track = sample_data.tracking(game)

    nearest, ball = nearest_teams(track)
    visible = np.isfinite(ball[:, 0])
    on_pitch = visible & (ball[:, 0] >= 0) & (ball[:, 0] <= LENGTH) \
        & (ball[:, 1] >= 0) & (ball[:, 1] <= WIDTH)
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS

    flags, report = [], []

    def flag(index, name, detail):
        flags.append({"frame": int(index) + 1, "check": name, "detail": detail})

    # out_crossing
    offsets = []
    for stop in stoppages[stoppages.cause.isin(["ball out", "shot out", "goal"])].itertuples():
        index = int(stop.start) - 1
        low, high = max(1, index - OUT_WINDOW), min(len(ball), index + OUT_WINDOW)
        window = np.arange(low, high)
        crossings = window[~on_pitch[window] & on_pitch[window - 1]]
        if not len(crossings):
            flag(index, "out_crossing", "{}: no crossing within +/-{} frames".format(
                stop.cause, OUT_WINDOW))
            continue
        offset = int(crossings[np.argmin(np.abs(crossings - index))] - index)
        offsets.append(offset)
        if abs(offset) > OUT_TOLERANCE:
            flag(index, "out_crossing", "{}: ball leaves the pitch {:+d} frames from D "
                 "start".format(stop.cause, offset))
    if offsets:
        offsets = np.array(offsets)
        report.append("out_crossing: {} measured, D start vs ball leaving the pitch: median {:+.0f}"
                      " frames, |offset| <= {} in {:.0%}".format(
                          len(offsets), np.median(offsets), OUT_TOLERANCE,
                          np.mean(np.abs(offsets) <= OUT_TOLERANCE)))

    # restart_position and restart_team
    checked = passed = team_checked = team_passed = 0
    for stop in stoppages.itertuples():
        if stop.restart in ("end of period", "inferred") or not stop.restart_team:
            continue
        index = int(stop.end) + 1 - 1  # restart frame, as a row
        where = nearest_visible(visible, index, RESTART_WINDOW)
        if where is None:
            flag(index, "restart_position", "{}: ball not tracked".format(stop.restart))
            continue
        ok, detail = restart_ok(stop.restart, *ball[where])
        if ok is not None:
            checked += 1
            passed += ok
            if not ok:
                flag(index, "restart_position", "{}: {}".format(stop.restart, detail))
        if "KICK OFF" not in stop.restart and nearest[where] != NOBODY:
            team_checked += 1
            team = "H" if nearest[where] == HOME else "A"
            team_passed += team == stop.restart_team
            if team != stop.restart_team:
                flag(index, "restart_team", "{} for {}, nearest player is {}".format(
                    stop.restart, stop.restart_team, team))
    report.append("restart_position: {}/{} restarts where the rules say".format(passed, checked))
    report.append("restart_team: {}/{} restarts with the nearest player on the restarting "
                  "team".format(team_passed, team_checked))

    # possession_agrees
    label_code = np.where(labels == "H", HOME, np.where(labels == "A", AWAY, NOBODY))
    live_seen = (label_code != NOBODY) & (nearest != NOBODY)
    report.append("possession_agrees: nearest player on the possessing team on {:.1%} of "
                  "live frames with a tracked ball".format(
                      np.mean(nearest[live_seen] == label_code[live_seen])))
    for team in "HA":
        for start, stop in runs(labels == team):
            seen = live_seen[start:stop]
            if stop - start < MIN_POSSESSION or seen.sum() < MIN_VISIBLE:
                continue
            agreement = np.mean(nearest[start:stop][seen] == label_code[start:stop][seen])
            if agreement < MIN_AGREEMENT:
                flag(start, "possession_agrees", "{} for {:.1f} s, nearest player agrees on "
                     "{:.0%}".format(team, (stop - start) / float(FPS), agreement))

    # ball_moving_while_dead
    moving = (labels == "D") & on_pitch & (np.nan_to_num(speed) > MOVING_SPEED)
    for start, stop in runs(moving):
        if stop - start >= MOVING_FRAMES:
            flag(start, "ball_moving_while_dead", "{:.1f} s at {:.0f} m/s on the pitch".format(
                (stop - start) / float(FPS), np.nanmean(speed[start:stop])))

    # ball_missing_while_live
    for start, stop in runs(np.isin(labels, ["H", "A"]) & ~visible):
        if stop - start >= MISSING_FRAMES:
            flag(start, "ball_missing_while_live", "{:.1f} s without a ball during {}".format(
                (stop - start) / float(FPS), labels[start]))

    for note in notes.itertuples():
        flag(int(note.frame) - 1, "inferred", note.note)

    # the baseline on clean tracking
    gt_codes = pm.label_codes(labels)
    for window in (0, BASELINE_WINDOW):
        result = pm.evaluate(gt_codes, pm.label_codes(smooth(nearest, window)))
        report.append("baseline (nearest player, smooth +/-{}) on clean tracking: macroF1 {:.4f}"
                      "  spellF1 {:.4f}  possessions {} vs {}".format(
                          window, result["macro_f1"], result["spell_f1_mean"],
                          result["n_pred_spells"], result["n_gt_spells"]))

    flags = pd.DataFrame(flags, columns=["frame", "check", "detail"]).sort_values(
        "frame", kind="stable")
    flags.to_csv(stem + "flags.csv", index=False)
    return flags, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", type=int, nargs="+", default=list(sample_data.GAMES))
    parser.add_argument("--show", type=int, default=5, help="flags to print per check")
    args = parser.parse_args()

    for game in args.game:
        flags, report = check(game)
        print("=" * 78)
        print("game {}".format(game))
        for line in report:
            print("  " + line)
        print("\n  flags: {} total  {}".format(len(flags), flags.check.value_counts().to_dict()))
        for name, group in flags.groupby("check", sort=False):
            print("\n  {} ({})".format(name, len(group)))
            for row in group.head(args.show).itertuples():
                print("    frame {:>6}  {}".format(row.frame, row.detail))
        print("\n  -> {}".format(os.path.relpath(
            os.path.join(sample_data.game_dir(game), "game{}_flags.csv".format(game)),
            sample_data.ROOT)))
