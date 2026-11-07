"""Step 1: v0 placeholder p_dead and the input check.

    python tests/deadball/test_combine.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from harness import make_game, run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.config import DEADBALL
from mtah.deadball.combine import run, stub_p_dead
from mtah.deadball.util import check_game

EPS = DEADBALL.EPS
DEAD, ALIVE = 1 - EPS, EPS


def test_shape_dtype_and_nan_outside_halves():
    game = make_game()
    p = stub_p_dead(game)
    assert p.shape == (1000,) and p.dtype == np.float64
    inside = np.zeros(1000, bool)
    inside[100:450] = inside[550:900] = True          # halves are inclusive: (100, 449), (550, 899)
    assert np.isnan(p[~inside]).all()
    assert set(np.unique(p[inside])) <= {ALIVE, DEAD}


def test_threshold_is_three_seconds_at_25_fps():
    game = make_game(missing=[(150, 225), (300, 374)])   # 75 frames = 3.00 s, 74 frames = 2.96 s
    p = stub_p_dead(game)
    assert (p[150:225] == DEAD).all()
    assert (p[300:374] == ALIVE).all()
    assert p[149] == ALIVE and p[225] == ALIVE


def test_threshold_follows_fps():
    game = make_game(fps=30, missing=[(150, 240), (300, 389)])   # 90 frames = 3 s, 89 < 3 s at 30 fps
    p = stub_p_dead(game)
    assert (p[150:240] == DEAD).all() and (p[300:389] == ALIVE).all()


def test_gap_through_half_time_is_split():
    # 30 frames at the end of half 1 + the break + 51 frames at the start of half 2: each piece < 3 s
    game = make_game(missing=[(420, 601)])
    p = stub_p_dead(game)
    assert (p[420:450] == ALIVE).all() and (p[550:601] == ALIVE).all()
    assert np.isnan(p[450:550]).all()


def test_long_gap_at_half_start_counts_inside_half_only():
    game = make_game(missing=[(40, 180)])               # 60 frames before the half, 80 inside it
    p = stub_p_dead(game)
    assert (p[100:180] == DEAD).all()                    # 80 frames = 3.2 s inside the half


def test_write_is_deterministic_and_stays_in_out_dir():
    game = make_game(missing=[(150, 260)])
    with tempfile.TemporaryDirectory() as tmp:
        a = run("FAKE", game=game, out_dir=Path(tmp) / "a")
        b = run("FAKE", game=game, out_dir=Path(tmp) / "b")
        assert a["sha256"] == b["sha256"]
        assert np.array_equal(np.load(a["path"]), stub_p_dead(game), equal_nan=True)
        assert sorted(p.name for p in Path(tmp, "a").iterdir()) == ["p_dead.npy"]
    assert a["dead_runs"] == 1 and a["graded_frames"] == 700


def test_check_game_rejects_bad_input():
    bad_cases = []
    g = make_game(); del g["meta"]["fps"]; bad_cases.append(g)
    g = make_game(); g["meta"]["halves"] = [(550, 899), (100, 449)]; bad_cases.append(g)   # unordered
    g = make_game(); g["meta"]["halves"] = [(100, 1000)]; bad_cases.append(g)              # past n
    g = make_game(); g["ball_xy"] = g["ball_xy"][:-1]; bad_cases.append(g)                 # wrong length
    for game in bad_cases:
        try:
            check_game(game, need=("ball_xy",))
        except ValueError:
            continue
        raise AssertionError("check_game accepted a bad game")


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
