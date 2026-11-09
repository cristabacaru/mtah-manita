"""Summarise the public Metrica sample data we use to build validation matches.

The sample data is three anonymised matches with tracking and human-logged events, from the same
provider as the competition data. It is public, not competition data, so it can be inspected
freely. Download it once (see WORKLOG.md):

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git \
        data/external/metrica-sample-data

    python tools/inspect_sample_data.py

Prints, per match: event counts by type and subtype, the tracking length and halves, how often the
ball is present, how the ball gaps are distributed, and one worked example of a ball going out of
play next to the events around it.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Params
sys.path.insert(0, ROOT)
from mtah import config  # noqa: E402

SAMPLE_DIR = str(config.EXTERNAL.METRICA_DIR)
CSV_GAMES = (1, 2)  # games 1 and 2 are CSV; game 3 is EPTS tracking + JSON events

# The worked example: game 2, the first ball out (events at 504-534) and the throw-in at 672.
EXAMPLE_GAME, EXAMPLE_FIRST, EXAMPLE_LAST, EXAMPLE_STEP = 2, 490, 700, 15


def csv_paths(game):
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(game))
    stem = os.path.join(folder, "Sample_Game_{}_".format(game))
    return stem + "RawEventsData.csv", stem + "RawTrackingData_Home_Team.csv"


def read_tracking(path):
    """The tracking CSV with its two extra header rows skipped. Ball is the last column pair."""
    table = pd.read_csv(path, skiprows=2)
    return table.rename(columns={table.columns[-2]: "ball_x", table.columns[-1]: "ball_y"})


def gap_lengths(missing):
    """Lengths of every run of consecutive True values."""
    edges = np.diff(np.concatenate(([0], missing.astype(int), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def summarise_csv_game(game):
    events_path, tracking_path = csv_paths(game)
    events = pd.read_csv(events_path)
    tracking = read_tracking(tracking_path)

    print("=" * 78)
    print("Game {}: {} events".format(game, len(events)))
    print(events.groupby(["Type", "Subtype"], dropna=False).size().to_string())

    print("\ntracking: {} frames ({}..{}) at 25 fps".format(
        len(tracking), tracking.Frame.min(), tracking.Frame.max()))
    for period in sorted(tracking.Period.unique()):
        frames = tracking.Frame[tracking.Period == period]
        print("  period {}: frames {}..{}".format(period, frames.min(), frames.max()))

    missing = tracking.ball_x.isna().values
    lengths = gap_lengths(missing)
    print("ball present on {:.1%} of frames".format(1 - missing.mean()))
    print("ball gaps: {}, median {} frames, p90 {} frames, longest {} frames".format(
        len(lengths), int(np.median(lengths)), int(np.percentile(lengths, 90)),
        int(lengths.max())))


def show_example():
    events_path, tracking_path = csv_paths(EXAMPLE_GAME)
    events = pd.read_csv(events_path)
    tracking = read_tracking(tracking_path)

    print("=" * 78)
    print("Worked example, game {}, frames {}..{}".format(
        EXAMPLE_GAME, EXAMPLE_FIRST, EXAMPLE_LAST))
    window = events[(events["Start Frame"] >= EXAMPLE_FIRST - 60)
                    & (events["Start Frame"] <= EXAMPLE_LAST)]
    print(window[["Team", "Type", "Subtype", "Start Frame", "End Frame", "From",
                  "End X", "End Y"]].to_string(index=False))
    rows = tracking[(tracking.Frame >= EXAMPLE_FIRST) & (tracking.Frame <= EXAMPLE_LAST)
                    & (tracking.Frame % EXAMPLE_STEP == 0)]
    print("\nball position every {} frames:".format(EXAMPLE_STEP))
    print(rows[["Frame", "ball_x", "ball_y"]].to_string(index=False))


def summarise_json_game():
    path = os.path.join(SAMPLE_DIR, "Sample_Game_3", "Sample_Game_3_events.json")
    events = json.load(open(path))["data"]
    print("=" * 78)
    print("Game 3: {} events (JSON; tracking is FIFA/EPTS like the competition ATD)".format(
        len(events)))
    counts = pd.Series([e["type"]["name"] for e in events]).value_counts()
    print(counts.to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    if not os.path.isdir(SAMPLE_DIR):
        raise SystemExit("sample data not found in {} -- see WORKLOG.md".format(SAMPLE_DIR))
    for game in CSV_GAMES:
        summarise_csv_game(game)
    summarise_json_game()
    show_example()
