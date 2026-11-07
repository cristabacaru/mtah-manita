"""Ball-gap features for p_dead (spec section 3, the three gap rows).

A "gap" is a run of frames where the ball is missing (NaN). Each feature is a float64 array of
shape (n_frames,):
- NaN outside the halves;
- 0 inside a half where the ball is visible;
- log(1 + seconds) inside a gap, so long gaps keep their order without dominating the fit.

Gaps are cut at half edges. When a gap touches a half edge, the edge counts as the missing
sighting (decision of 2026-11-07, option a): the value is then a lower bound, and nothing outside
the half is used. Seconds come from game["meta"]["fps"].
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from mtah.deadball.util import ball_missing, check_game, halves, runs


def _gap_frames(game: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame counts (length, since, until) per frame: NaN outside halves, 0 where visible.

    In a gap of k frames, the i-th missing frame (i = 1..k) has length k, since i, until k + 1 - i,
    i.e. the frames back to the last sighting and forward to the next one.
    """
    check_game(game, need=("ball_xy",))
    n = int(game["meta"]["n_frames"])
    length, since, until = (np.full(n, np.nan, dtype=np.float64) for _ in range(3))
    missing = ball_missing(game)
    for half in halves(game):
        size = half.stop - half.start
        h_len, h_since, h_until = np.zeros(size), np.zeros(size), np.zeros(size)
        for start, stop in zip(*runs(missing[half])):
            k = stop - start
            i = np.arange(1, k + 1, dtype=np.float64)
            h_len[start:stop] = k
            h_since[start:stop] = i
            h_until[start:stop] = k + 1 - i
        length[half], since[half], until[half] = h_len, h_since, h_until
    return length, since, until


def _log_seconds(frames: np.ndarray, game: Mapping) -> np.ndarray:
    """log(1 + frames / fps); NaN stays NaN, 0 stays 0."""
    return np.log1p(frames / float(game["meta"]["fps"]))


def gap_len_s(game: Mapping) -> np.ndarray:
    """log(1 + length in s) of the ball-missing run containing t; 0 where visible; NaN outside halves."""
    return _log_seconds(_gap_frames(game)[0], game)


def t_since_seen_s(game: Mapping) -> np.ndarray:
    """log(1 + s since the ball was last visible, or since the half started); 0 where visible."""
    return _log_seconds(_gap_frames(game)[1], game)


def t_until_seen_s(game: Mapping) -> np.ndarray:
    """log(1 + s until the ball is next visible, or until the half ends); 0 where visible."""
    return _log_seconds(_gap_frames(game)[2], game)
