"""Line features for p_dead (spec section 3: last_line_dist_m, next_line_dist_m, ball_out_m).

"Line" means the four outer lines only: both touchlines and both goal lines, never the halfway line
(decision (e), 2026-11-07). The pitch runs from 0 to L in x and 0 to W in y, with (L, W) from
game["meta"]["pitch"]; no pitch size is written in this file.

Each feature is a float64 array of shape (n_frames,), NaN outside the halves:
- last_line_dist_m / next_line_dist_m: inside a gap, the distance inside the pitch to the nearest
  line (0 on or past a line) of the last sighting before the gap / the next one after it; where
  the ball is visible, the current ball's distance (decision (a)). A gap touching a half edge has no sighting on that side inside the half,
  so that side is NaN (decision (b)); nothing outside the half is used.
- ball_out_m: how far a visible ball is outside the pitch, straight-line distance to the pitch
  rectangle (decision (d)), capped at OUT_CAP_M; NaN while the ball is missing (decision (c)).
- *_missing: 1.0 where the matching feature is NaN inside a half, 0.0 elsewhere inside a half.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from mtah import config
from mtah.deadball.util import ball_missing, check_game, halves, runs


def _pitch(game: Mapping) -> tuple[float, float]:
    """(L, W) in metres from game["meta"]["pitch"]; ValueError unless both are positive."""
    pitch = game["meta"].get("pitch")
    if pitch is None or len(pitch) != 2 or not all(np.isfinite(pitch)) or min(pitch) <= 0:
        raise ValueError(f"game['meta']['pitch'] must be (L, W) in metres, both > 0, got {pitch!r}")
    return float(pitch[0]), float(pitch[1])


def _ball(game: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """(xy (n, 2) float64 with both coordinates NaN where the ball is missing, missing (n,) bool)."""
    check_game(game, need=("ball_xy",))
    xy = np.array(game["ball_xy"], dtype=np.float64)       # a copy: the input is never modified
    missing = ball_missing(game)
    xy[missing] = np.nan                                    # one NaN coordinate = ball missing
    return xy, missing


def _line_dist(xy: np.ndarray, L: float, W: float) -> np.ndarray:
    """(n,) distance inside the pitch to the nearest outer line: max(0, min(x, L - x, y, W - y)).

    0 on or past a line (decision of 2026-11-07, option 1): a ball last seen 3 m behind the goal
    line must not look like a ball 3 m inside the pitch. "< 2 m" is then exactly c1's near_line
    (research/checks/c1_ball_gaps.py:27-28). Unchanged when the ball is mirrored (x -> L - x and/or
    y -> W - y). NaN where xy is NaN.
    """
    x, y = xy[:, 0], xy[:, 1]
    return np.maximum(0.0, np.min(np.stack([x, L - x, y, W - y]), axis=0))   # NaN stays NaN


def _last_next(game: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """(last_line_dist_m, next_line_dist_m), each (n,) float64; see the module docstring."""
    xy, missing = _ball(game)
    dist = _line_dist(xy, *_pitch(game))
    n = len(dist)
    last, nxt = np.full(n, np.nan), np.full(n, np.nan)
    for half in halves(game):
        d = dist[half]
        h_last, h_next = d.copy(), d.copy()                 # visible frames: current distance
        for start, stop in zip(*runs(missing[half])):
            h_last[start:stop] = d[start - 1] if start > 0 else np.nan     # gap opens the half
            h_next[start:stop] = d[stop] if stop < len(d) else np.nan      # gap closes the half
        last[half], nxt[half] = h_last, h_next
    return last, nxt


def last_line_dist_m(game: Mapping) -> np.ndarray:
    """Metres from the nearest line of the last sighting before the gap (current ball if visible)."""
    return _last_next(game)[0]


def next_line_dist_m(game: Mapping) -> np.ndarray:
    """Metres from the nearest line of the next sighting after the gap (current ball if visible)."""
    return _last_next(game)[1]


def ball_out_m(game: Mapping) -> np.ndarray:
    """Metres a visible ball is outside the pitch: min(OUT_CAP_M, hypot(dx, dy)); 0 inside the pitch.

    dx = max(0, -x, x - L), dy = max(0, -y, y - W). NaN while the ball is missing.
    """
    xy, _ = _ball(game)
    L, W = _pitch(game)
    x, y = xy[:, 0], xy[:, 1]
    dx = np.maximum(0.0, np.maximum(-x, x - L))
    dy = np.maximum(0.0, np.maximum(-y, y - W))
    out_all = np.minimum(config.DEADBALL.OUT_CAP_M, np.hypot(dx, dy))   # NaN stays NaN
    out = np.full(len(x), np.nan)
    for half in halves(game):
        out[half] = out_all[half]
    return out


def _missing_indicator(feature: np.ndarray, game: Mapping) -> np.ndarray:
    """1.0 where feature is NaN inside a half, 0.0 elsewhere inside a half, NaN outside the halves."""
    out = np.full(len(feature), np.nan)
    for half in halves(game):
        out[half] = np.isnan(feature[half]).astype(np.float64)
    return out


def last_line_missing(game: Mapping) -> np.ndarray:
    """1.0 where last_line_dist_m is NaN inside a half (a gap that opens the half)."""
    return _missing_indicator(last_line_dist_m(game), game)


def next_line_missing(game: Mapping) -> np.ndarray:
    """1.0 where next_line_dist_m is NaN inside a half (a gap that closes the half)."""
    return _missing_indicator(next_line_dist_m(game), game)


def ball_out_missing(game: Mapping) -> np.ndarray:
    """1.0 where ball_out_m is NaN inside a half, i.e. where the ball is missing."""
    return _missing_indicator(ball_out_m(game), game)
