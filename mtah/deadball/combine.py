"""p_dead: probability that the ball is dead, one value per frame (spec sections 2-3).

Today this is the v0 placeholder (README module contract): dead where the ball has been missing
for at least DEADBALL.STUB_GAP_S inside one half. The logistic combiner replaces it in step 6.

Output: $MTAH_CACHE/<game>/p_dead.npy, float64 (n,), in [EPS, 1 - EPS] inside halves, NaN outside.

    python -m mtah.deadball.combine --game BAR-ATH

Prints aggregates only (shares, counts, sha256), never frame-level values.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from mtah import config
from mtah.config import DEADBALL
from mtah.deadball.util import ball_missing, check_game, halves, runs, save_npy


def stub_p_dead(game: Mapping, gap_s: float = DEADBALL.STUB_GAP_S, eps: float = DEADBALL.EPS) -> np.ndarray:
    """v0: 1 - eps on ball-missing runs of >= gap_s seconds within a half, eps elsewhere in it.

    Runs are cut at half edges, so a gap through half-time counts as two shorter gaps.
    """
    check_game(game, need=("ball_xy",))
    meta = game["meta"]
    min_frames = math.ceil(gap_s * meta["fps"] - 1e-9)   # length/fps >= gap_s, robust to float error
    missing = ball_missing(game)
    p = np.full(int(meta["n_frames"]), np.nan, dtype=np.float64)
    for half in halves(game):
        dead = np.zeros(half.stop - half.start, dtype=bool)
        for start, stop in zip(*runs(missing[half])):
            if stop - start >= min_frames:
                dead[start:stop] = True
        p[half] = np.where(dead, 1.0 - eps, eps)
    return p


def summarise(p: np.ndarray, game: Mapping) -> dict:
    """Aggregates for the log: graded frames, share called dead (p > 0.5), number of dead runs."""
    graded = dead_runs = called_dead = 0
    for half in halves(game):
        dead = p[half] > 0.5
        graded += dead.size
        called_dead += int(dead.sum())
        dead_runs += len(runs(dead)[0])
    return {"graded_frames": graded, "dead_share": called_dead / graded if graded else float("nan"),
            "dead_runs": dead_runs}


def run(game_name: str, game: Mapping | None = None, out_dir: Path | None = None) -> dict:
    """Compute p_dead for one game and write it. `game` and `out_dir` are for tests."""
    if game is None:
        try:
            from mtah.data import load   # P1, not pushed yet: fail with a clear message
        except ImportError as err:
            raise SystemExit("mtah.data is not available yet (P1 is writing it). "
                             "Tests run without it: python tests/deadball/run_all.py") from err
        game = load(game_name)
    p = stub_p_dead(game)
    out = Path(out_dir) if out_dir is not None else config.game_cache(game_name)
    sha = save_npy(out / "p_dead.npy", p)
    return {**summarise(p, game), "path": str(out / "p_dead.npy"), "sha256": sha}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write p_dead.npy for one game (v0 placeholder).")
    parser.add_argument("--game", default=config.GAME)
    args = parser.parse_args()
    info = run(args.game)
    print("p_dead (v0 placeholder, gap >= {} s) for {}".format(DEADBALL.STUB_GAP_S, args.game))
    print("  graded frames {graded_frames}, called dead {dead_share:.1%}, dead runs {dead_runs}".format(**info))
    print("  wrote {path}\n  sha256 {sha256}".format(**info))


if __name__ == "__main__":
    main()
