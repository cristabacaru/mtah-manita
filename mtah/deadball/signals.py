"""Ball-gap and motion features for p_dead (spec section 3: the gap rows, speed_abs, speed_rel).

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

from mtah import config
from mtah.deadball.util import ball_missing, check_game, halves, runs

_MEDIAN_CHUNK = 4096   # rows per nanmedian call in speed_rel: bounds memory, not a model setting


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


# --- motion (spec section 3, rows speed_abs, speed_rel, *_missing; "Motion details") -----------------
#
# Per-frame team aggregate only: the mean over the players tracked in that frame, never one column
# followed over time (guide section 5). Windows are round(seconds x fps) frames, stop at the half
# edge and shrink there; NaN frames inside a window are skipped.

def _frame_mean_speed(game: Mapping) -> np.ndarray:
    """(n,) mean players_speed over the players with a speed in each frame; NaN if there is none.

    Sum / count instead of np.nanmean, so a frame with nobody tracked gives NaN without a warning.
    """
    speed = np.asarray(game["players_speed"], dtype=np.float64)   # stored as float32 (data.py:130)
    ok = np.isfinite(speed)
    count = ok.sum(axis=1)
    total = np.where(ok, speed, 0.0).sum(axis=1)
    out = np.full(len(speed), np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _centred_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Mean of the finite values of x over [t - k, t + k], window cut at both ends of x; NaN if none."""
    ok = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(len(x))
    lo, hi = np.maximum(0, t - k), np.minimum(len(x), t + k + 1)
    n_ok = count[hi] - count[lo]
    out = np.full(len(x), np.nan, dtype=np.float64)
    np.divide(total[hi] - total[lo], n_ok, out=out, where=n_ok > 0)
    return out


def _trailing_median(x: np.ndarray, w: int) -> np.ndarray:
    """Median of the finite values of x over [t - w + 1, t], cut at the start of x; NaN where x[t] is NaN.

    The window includes t, so wherever x[t] is finite it has a finite value and np.nanmedian never
    sees an all-NaN row (no "All-NaN slice" warning).
    """
    out = np.full(len(x), np.nan, dtype=np.float64)
    rows = np.flatnonzero(np.isfinite(x))
    if len(rows) == 0:
        return out
    padded = np.concatenate((np.full(w - 1, np.nan), x))
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)   # row t = x[t - w + 1 : t + 1]
    for i in range(0, len(rows), _MEDIAN_CHUNK):
        chunk = rows[i:i + _MEDIAN_CHUNK]
        out[chunk] = np.nanmedian(windows[chunk], axis=1)
    return out


def _speed_features(game: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """(speed_abs, speed_rel), each (n,) float64, NaN outside halves and where no player is tracked."""
    check_game(game, need=("players_speed",))
    n = int(game["meta"]["n_frames"])
    fps = float(game["meta"]["fps"])
    k = int(round(config.DEADBALL.SPEED_SMOOTH_HALF_S * fps))
    w = max(1, int(round(config.DEADBALL.REL_WIN_S * fps)))
    raw = _frame_mean_speed(game)
    speed_abs, speed_rel = np.full(n, np.nan), np.full(n, np.nan)
    for half in halves(game):
        a = _centred_mean(raw[half], k)
        speed_abs[half] = a
        speed_rel[half] = a / np.maximum(config.DEADBALL.REL_FLOOR, _trailing_median(a, w))
    return speed_abs, speed_rel


def speed_abs(game: Mapping) -> np.ndarray:
    """m/s: mean speed of the tracked players, smoothed over +/- SPEED_SMOOTH_HALF_S inside the half."""
    return _speed_features(game)[0]


def speed_rel(game: Mapping) -> np.ndarray:
    """speed_abs / max(REL_FLOOR, its median over the trailing REL_WIN_S, t included, inside the half)."""
    return _speed_features(game)[1]


def speed_missing(game: Mapping) -> np.ndarray:
    """1.0 where speed_abs is NaN inside a half, 0.0 elsewhere inside a half, NaN outside the halves.

    The features themselves stay NaN there; the combiner fills them with the training median.
    """
    a = speed_abs(game)
    out = np.full(len(a), np.nan, dtype=np.float64)
    for half in halves(game):
        out[half] = np.isnan(a[half]).astype(np.float64)
    return out
