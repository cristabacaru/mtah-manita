"""Helpers shared by every dead-ball file.

The input contract (guide section 5) is checked here once, so each feature can assume it.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np


def check_game(game: Mapping, need: tuple[str, ...] = ()) -> None:
    """Raise ValueError unless `game` follows the shared schema for the keys this caller needs.

    Checked: meta has fps, n_frames and halves; halves are (first, last) inclusive, ordered,
    non-overlapping and inside [0, n_frames); each needed array has n_frames rows.
    """
    meta = game.get("meta") if isinstance(game, Mapping) else None
    if not isinstance(meta, Mapping):
        raise ValueError("game must be a mapping with a 'meta' mapping (guide section 5)")
    for key in ("fps", "n_frames", "halves"):
        if key not in meta:
            raise ValueError(f"game['meta'] has no {key!r}")
    n = int(meta["n_frames"])
    if not meta["fps"] or meta["fps"] <= 0:
        raise ValueError("game['meta']['fps'] must be positive")
    prev_last = -1
    for half in meta["halves"]:
        if len(half) != 2:
            raise ValueError(f"each half must be (first, last), got {half!r}")
        first, last = int(half[0]), int(half[1])
        if not (prev_last < first <= last < n):
            raise ValueError(f"halves must be ordered, inclusive and inside [0, {n}): {meta['halves']!r}")
        prev_last = last
    for key in need:
        if key not in game:
            raise ValueError(f"game has no {key!r}")
        if len(game[key]) != n:
            raise ValueError(f"game[{key!r}] has {len(game[key])} rows, meta says n_frames = {n}")


def halves(game: Mapping) -> Iterator[slice]:
    """One slice per half (inclusive bounds turned into a slice). Features never cross them."""
    for first, last in game["meta"]["halves"]:
        yield slice(int(first), int(last) + 1)


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, stops) of the maximal True runs in a 1-D mask; stops are exclusive."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def ball_missing(game: Mapping) -> np.ndarray:
    """(n,) bool: True where either ball coordinate is NaN."""
    return ~np.isfinite(np.asarray(game["ball_xy"], dtype=np.float64)).all(axis=1)


def save_npy(path: Path, array: np.ndarray) -> str:
    """Write `array` to `path` atomically (temp file, then rename). Returns the file's sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()
