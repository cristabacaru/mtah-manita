"""Turn a sample match's human event log into frame-by-frame H/A/D labels.

The output is a ground truth for a validation match, in exactly the competition format
(`frame,label`, one row per tracking frame, H/A/D/X), so the official scorer can grade any
pipeline against it:

    python tools/events_to_labels.py                 # games 1 and 2
    python tools/events_to_labels.py --game 2
    python evaluation/run_evaluation.py --gt data/validation/game2/game2_gt.csv --results <dir>

Written per game, under data/validation/game<N>/ (gitignored):

    game<N>_gt.csv          the labels
    game<N>_gt_meta.json    row count, for the scorer's sanity check
    game<N>_stoppages.csv   every dead-ball stretch: start, end, cause, restart type and team
    game<N>_notes.csv       places where the converter had to infer something the log left out

The rules
---------
A state machine walks the events in time order. The state is H, A or D. The conventions come from
watching the public 5 minutes of the competition ground truth (WORKLOG.md, step 3) and from the
Metrica event definitions (documentation/events-definitions.pdf in the sample data):

    * A possession is held until the other team takes CONTROL: a PASS, SHOT, RECOVERY or BALL
      LOST by that team (BALL LOST asserts the team had the ball when the event starts). Loose
      balls, challenges and deflections change nothing, so the last team keeps it.
    * The ball goes DEAD at:
        BALL OUT end frame                    the ball crosses a line
        SHOT ...-OUT / ...-GOAL end frame     wide, or a goal
      (these three are then moved to the frame the clean tracking sees the ball leave the
      pitch, because the logged end frame runs a few frames late)
        FAULT RECEIVED                        "the referee immediately awards a free kick"
        BALL LOST OFFSIDE / FORCED end frame  the referee stops play
        BALL LOST HAND BALL / END HALF        the referee stops play
    * While dead, everything is ignored until a RESTART: a SET PIECE, a goal kick (logged as
      RECOVERY with no subtype plus PASS GOAL KICK), or a BALL LOST GOAL KICK. The restarting
      team has the ball from that frame.
    * Before the first kickoff of the match: X (not graded, like the competition). Before the
      kickoff of a later period: D (graded, like the competition's second half). After an END
      HALF: D to the end of the period. After the last event of a period with no END HALF: X,
      because the log does not say when the whistle came.
    * The first frame of each later period is X, so the scorer treats the periods as separate
      runs the way it treats the competition's two halves (there is no halftime gap in the
      sample tracking).

Where the log is incomplete, the converter infers and writes a note:
    * a restart while play is live (a set piece with no stoppage logged before it): the stoppage
      is assumed to start at the last fault challenge since the previous event, or else at the
      previous event's end frame;
    * an in-play action while dead (a pass with no restart logged before it): play is assumed to
      restart there.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample_data  # noqa: E402

# Params
# Actions that share a frame are applied in the order the analyst logged them (SET PIECE before
# its PASS, CHALLENGE before FAULT RECEIVED), and within one event in the order actions_of
# returns them (a BALL LOST OFFSIDE is "had it, then stopped").
KILL_AT_END = ("OFFSIDE", "FORCED")        # BALL LOST subtypes: dead at the end frame
KILL_AT_START = ("HAND BALL", "END HALF")  # BALL LOST subtypes: dead at the start frame
SHOT_KILLS = ("OUT", "GOAL")               # SHOT subtypes ending in these are dead at end frame
IN_PLAY_TYPES = ("PASS", "SHOT")           # actions that cannot happen while the ball is dead

# The logged end frame of a BALL OUT is a few frames after the ball actually leaves the pitch
# (median 4-5 frames on games 1 and 2, see WORKLOG.md). The competition ground truth starts D on
# the crossing frame, so those stoppages are moved to where the clean tracking sees the ball
# leave the pitch, searching this many frames before and after the logged frame.
OUT_CAUSES = ("ball out", "shot out", "goal")
CROSSING_BEFORE, CROSSING_AFTER = 50, 25


def has(subtype, words):
    return any(word in subtype for word in words)


def crossing_shift(on_pitch, frame, earliest):
    """Frames to move a ball-out stoppage so it starts where the ball leaves the pitch, or 0.

    A crossing is a frame where the ball is off the pitch or untracked right after a frame where
    it was tracked on the pitch. The nearest one to the logged frame wins, never earlier than the
    event's own start (the last touch).
    """
    low = max(earliest, frame - CROSSING_BEFORE, 2)
    high = min(len(on_pitch), frame + CROSSING_AFTER)
    candidates = np.arange(low, high + 1)  # frame numbers; row = frame - 1
    candidates = candidates[~on_pitch[candidates - 1] & on_pitch[candidates - 2]]
    if not len(candidates):
        return 0
    return int(candidates[np.argmin(np.abs(candidates - frame))] - frame)


def actions_of(event):
    """[(frame, kind, cause)] for one event, in the order they apply.

    kind is 'control', 'kill' or 'restart'; 'either' is a RECOVERY with no subtype, which
    restarts when dead and controls when live.
    """
    kind_of = event.type
    sub = event.subtype
    out = []
    if kind_of == "SET PIECE":
        out.append((event.start, "restart", "set piece: " + sub))
    elif kind_of == "RECOVERY":
        if sub == "":
            out.append((event.start, "either", "recovery"))
        else:
            out.append((event.start, "control", "recovery " + sub))
    elif kind_of == "PASS":
        if "GOAL KICK" in sub:
            out.append((event.start, "restart", "goal kick"))
        else:
            out.append((event.start, "control", "pass"))
    elif kind_of == "SHOT":
        out.append((event.start, "control", "shot"))
        if sub.endswith(SHOT_KILLS) or "GOAL" in sub.split("-"):
            cause = "goal" if "GOAL" in sub else "shot out"
            out.append((event.end, "kill", cause))
    elif kind_of == "BALL LOST":
        if "GOAL KICK" in sub:
            out.append((event.start, "restart", "goal kick"))
        else:
            out.append((event.start, "control", "ball lost"))
        if has(sub, KILL_AT_END):
            out.append((event.end, "kill", sub.lower()))
        elif has(sub, KILL_AT_START):
            out.append((event.start, "kill", sub.lower()))
    elif kind_of == "BALL OUT":
        out.append((event.end, "kill", "goal" if "GOAL" in sub else "ball out"))
    elif kind_of == "FAULT RECEIVED":
        out.append((event.start, "kill", "foul"))
    return out


def convert(game):
    """(labels, stoppages, notes) for one game. labels is an array of H/A/D/X, row = frame - 1."""
    events = sample_data.events(game)
    track = sample_data.tracking(game)
    n_frames = len(track["frame"])
    labels = np.full(n_frames, "X", dtype="<U1")
    stoppages, notes = [], []
    ball = track["ball"]
    on_pitch = np.isfinite(ball[:, 0]) & np.all((ball >= 0) & (ball <= 1), axis=1)

    # A bare RECOVERY logged right before the same team's GOAL KICK is the keeper fetching the
    # ball, sometimes half a minute before the kick. The kick is the restart, not the fetch.
    goal_kick = events.type.isin(["PASS", "BALL LOST"]) & events.subtype.str.contains("GOAL KICK")
    fetches = set(events.index[:-1][
        (events.type.values[:-1] == "RECOVERY") & (events.subtype.values[:-1] == "")
        & goal_kick.values[1:] & (events.team.values[1:] == events.team.values[:-1])])

    for period in sorted(np.unique(track["period"])):
        frames = track["frame"][track["period"] == period]
        first, last = int(frames[0]), int(frames[-1])
        in_period = events[events.period == period]

        actions = []
        for event in in_period.itertuples():
            for step, (frame, kind, cause) in enumerate(actions_of(event)):
                if event.Index in fetches:
                    kind = "control"  # ignored while dead; the goal kick restarts
                shift = 0
                if kind == "kill" and cause in OUT_CAUSES:
                    shift = crossing_shift(on_pitch, int(frame), int(event.start))
                actions.append((int(frame) + shift, event.Index, step, kind, cause, event.team,
                                shift))
        actions.sort()

        # The state before the kickoff: X before the match starts, D before a later period.
        state = "X" if period == 1 else "D"
        changes = [(first, state)]
        dead_since, dead_cause = (first, "before kickoff") if state == "D" else (None, None)
        dead_shift = 0
        last_end = first
        ended = False

        def set_state(frame, new):
            if changes and changes[-1][0] == frame:
                changes[-1] = (frame, new)
            else:
                changes.append((frame, new))

        for frame, row, _, kind, cause, team, shift in actions:
            event = in_period.loc[row]
            live = state in ("H", "A")

            if live and kind == "kill":
                state, dead_since, dead_cause, dead_shift = "D", frame, cause, shift
                set_state(frame, "D")
                ended = ended or cause == "end half" or "end half" in cause
            elif live and kind in ("control", "either"):
                if team != state:
                    state = team
                    set_state(frame, team)
            elif live and kind == "restart" and changes[-1] == (frame, team):
                pass  # the second event of one restart (RECOVERY, then PASS GOAL KICK)
            elif live and kind == "restart":
                # A restart nobody stopped play for: infer where the stoppage began.
                faults = in_period[(in_period.type == "CHALLENGE")
                                   & in_period.subtype.str.contains("FAULT")
                                   & (in_period.start >= last_end) & (in_period.start < frame)]
                start = int(faults.start.max()) if len(faults) else last_end
                start = min(start, frame - 1)  # an event can end after the restart is logged
                notes.append({"frame": frame, "note": "restart while live ({}); stoppage "
                              "assumed from {}".format(cause, start)})
                stoppages.append({"start": start, "end": frame - 1, "cause": "inferred",
                                  "restart": cause, "restart_team": team, "start_shift": 0})
                set_state(start, "D")
                state = team
                set_state(frame, team)
            elif not live and kind == "either" and dead_cause not in OUT_CAUSES:
                pass  # a bare RECOVERY restarts only a goal kick, i.e. after the ball went out
            elif not live and kind in ("restart", "either"):
                if state == "D":
                    stoppages.append({"start": dead_since, "end": frame - 1, "cause": dead_cause,
                                      "restart": cause, "restart_team": team,
                                      "start_shift": dead_shift})
                state = team
                set_state(frame, team)
            elif not live and kind == "control" and event.type in IN_PLAY_TYPES \
                    and state != "X" and not ended:
                notes.append({"frame": frame, "note": "{} while dead; restart assumed "
                              "here".format(cause)})
                stoppages.append({"start": dead_since, "end": frame - 1, "cause": dead_cause,
                                  "restart": "inferred", "restart_team": team,
                                  "start_shift": dead_shift})
                state = team
                set_state(frame, team)
            if kind != "kill":
                last_end = max(last_end, int(event.end))

        # After the last event: dead to the end if the half was whistled, unknown otherwise.
        if state == "D":
            stoppages.append({"start": dead_since, "end": last, "cause": dead_cause,
                              "restart": "end of period", "restart_team": "",
                              "start_shift": dead_shift})
        else:
            set_state(last_end + 1, "X")
            notes.append({"frame": last_end + 1, "note": "period {} has no END HALF; X from "
                          "the last event to the end".format(period)})

        bounds = [frame for frame, _ in changes] + [last + 1]
        for (frame, label), stop in zip(changes, bounds[1:]):
            labels[frame - 1:stop - 1] = label
        if period > 1:
            labels[first - 1] = "X"  # separates the periods into two scored runs

    return labels, pd.DataFrame(stoppages), pd.DataFrame(notes, columns=["frame", "note"])


def write(game, labels, stoppages, notes):
    folder = sample_data.game_dir(game)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    stem = os.path.join(folder, "game{}_".format(game))
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(
        stem + "gt.csv", index=False)
    with open(stem + "gt_meta.json", "w") as handle:
        json.dump({"game": "metrica-sample-{}".format(game), "n_video_frames": len(labels),
                   "source": "derived from human event data by tools/events_to_labels.py"},
                  handle, indent=2)
    stoppages.to_csv(stem + "stoppages.csv", index=False)
    notes.to_csv(stem + "notes.csv", index=False)
    return stem + "gt.csv"


def summary(game, labels, stoppages, notes):
    graded = labels[labels != "X"]
    spells = []
    for start in range(len(labels)):
        if labels[start] in "HA" and (start == 0 or labels[start - 1] != labels[start]):
            spells.append(start)
    lengths = np.diff(np.flatnonzero(np.concatenate(
        ([True], labels[1:] != labels[:-1], [True]))))
    run_labels = labels[np.flatnonzero(np.concatenate(([True], labels[1:] != labels[:-1])))]
    possession_lengths = lengths[np.isin(run_labels, ["H", "A"])] / float(sample_data.FPS)

    print("game {}: {} frames, {} graded  (H {:.1%}  A {:.1%}  D {:.1%})".format(
        game, len(labels), len(graded), np.mean(graded == "H"), np.mean(graded == "A"),
        np.mean(graded == "D")))
    print("  possessions {}  mean {:.1f} s  median {:.1f} s  ({:.1f} per minute of play)".format(
        len(possession_lengths), possession_lengths.mean(), np.median(possession_lengths),
        len(possession_lengths) / (len(graded) / float(sample_data.FPS) / 60.0)))
    print("  stoppages {} by cause: {}".format(
        len(stoppages), stoppages.cause.value_counts().to_dict()))
    out = stoppages[stoppages.cause.isin(OUT_CAUSES)]
    moved = out.start_shift[out.start_shift != 0]
    print("  ball-out starts moved to the tracking crossing: {} of {}, median {:+.0f} frames".format(
        len(moved), len(out), moved.median() if len(moved) else 0))
    print("  inferred by the converter: {}".format(len(notes)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--game", type=int, nargs="+", default=list(sample_data.GAMES))
    args = parser.parse_args()
    for game in args.game:
        labels, stoppages, notes = convert(game)
        path = write(game, labels, stoppages, notes)
        summary(game, labels, stoppages, notes)
        print("  -> {}\n".format(os.path.relpath(path, sample_data.ROOT)))
