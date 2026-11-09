"""Load the public Metrica sample matches (games 1 and 2, CSV format) as plain arrays.

Shared by tools/events_to_labels.py and tools/check_labels.py. The sample data is public, not
competition data; see WORKLOG.md for how to download it.

Conventions match the competition data: 25 fps, coordinates normalised 0-1, and tracking frame
`f` becomes label row `f - 1`, exactly like the ATD's frame_id.
"""

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
VALIDATION_DIR = str(config.VALIDATION_DIR)
GAMES = (1, 2)
FPS = 25
PITCH = (105.0, 68.0)  # metres, from the sample-data README
TEAMS = {"Home": "H", "Away": "A"}


def _stem(game):
    return os.path.join(SAMPLE_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_".format(game))


def game_dir(game):
    """Where the validation files for one game are written."""
    return os.path.join(VALIDATION_DIR, "game{}".format(game))


def events(game):
    """The event log as a DataFrame with short lower-case column names.

    team is 'H' or 'A', subtype is '' rather than NaN, and rows are sorted by start frame (stable,
    so events logged on the same frame keep the order the analyst wrote them in).
    """
    table = pd.read_csv(_stem(game) + "RawEventsData.csv")
    table = table.rename(columns={
        "Team": "team", "Type": "type", "Subtype": "subtype", "Period": "period",
        "Start Frame": "start", "End Frame": "end", "Start X": "start_x", "Start Y": "start_y",
        "End X": "end_x", "End Y": "end_y"})
    table["team"] = table["team"].map(TEAMS)
    table["subtype"] = table["subtype"].fillna("")
    table = table.sort_values("start", kind="stable").reset_index(drop=True)
    return table[["team", "type", "subtype", "period", "start", "end",
                  "start_x", "start_y", "end_x", "end_y"]]


def _read_side(path):
    """(period, frame, players (n, k, 2), ball (n, 2)) from one team's tracking CSV."""
    table = pd.read_csv(path, skiprows=2)
    values = table.values
    period = values[:, 0].astype(int)
    frame = values[:, 1].astype(int)
    xy = values[:, 3:].astype(np.float64)  # pairs of columns: players..., then the ball last
    players = xy[:, :-2].reshape(len(table), -1, 2)
    ball = xy[:, -2:]
    return period, frame, players, ball


def tracking(game):
    """Clean tracking for both teams.

    Returns a dict of numpy arrays, one row per tracking frame:
        frame (n,)          tracking frame number, starting at 1
        period (n,)
        ball (n, 2)         normalised; NaN when the ball is not tracked
        home, away (n, k, 2) normalised; NaN for players not on the pitch
    """
    period, frame, home, ball = _read_side(_stem(game) + "RawTrackingData_Home_Team.csv")
    away_period, away_frame, away, _ = _read_side(_stem(game) + "RawTrackingData_Away_Team.csv")
    if not np.array_equal(frame, away_frame):
        raise ValueError("game {}: home and away tracking frames differ".format(game))
    if not np.array_equal(frame, np.arange(1, len(frame) + 1)):
        raise ValueError("game {}: tracking frames are not 1..n".format(game))
    return {"frame": frame, "period": period, "ball": ball, "home": home, "away": away}


def to_metres(xy):
    """Normalised coordinates -> metres, on any array whose last axis is (x, y)."""
    return xy * np.array(PITCH)
