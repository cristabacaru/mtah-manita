"""Step 2: ball-gap features (spec section 5, "Gap" bullet, plus edge cases).

    python tests/deadball/test_signals.py
"""
import copy
import sys

import numpy as np

from harness import make_game, run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.deadball.signals import gap_len_s, t_since_seen_s, t_until_seen_s

FEATURES = (gap_len_s, t_since_seen_s, t_until_seen_s)
INSIDE = np.zeros(1000, bool)
INSIDE[100:450] = INSIDE[550:900] = True       # make_game halves are inclusive: (100, 449), (550, 899)


# --- spec section 5, "Gap" -------------------------------------------------------------------------

def test_spec_gap_len_is_log5_inside_and_0_outside():
    game = make_game(missing=[(200, 300)])                 # 100 NaN frames at 25 fps = 4 s
    g = gap_len_s(game)
    assert np.allclose(g[200:300], np.log(5))
    outside_gap = INSIDE.copy(); outside_gap[200:300] = False
    assert (g[outside_gap] == 0).all()


def test_spec_since_rises_from_log104_to_log5_strictly():
    game = make_game(missing=[(200, 300)])
    s = t_since_seen_s(game)[200:300]
    assert np.isclose(s[0], np.log(1.04)) and np.isclose(s[-1], np.log(5))
    assert (np.diff(s) > 0).all()


def test_spec_gap_crossing_half_edge_is_split():
    # 30 frames at the end of half 1, the break, 51 frames at the start of half 2
    game = make_game(missing=[(420, 601)])
    g, s, u = (f(game) for f in FEATURES)
    assert np.allclose(g[420:450], np.log1p(30 / 25)) and np.allclose(g[550:601], np.log1p(51 / 25))
    assert np.isnan(g[450:550]).all() and np.isnan(s[450:550]).all() and np.isnan(u[450:550]).all()


# --- edge cases added in step 2 ----------------------------------------------------------------------

def test_shape_dtype_and_nan_exactly_outside_halves():
    game = make_game(missing=[(200, 300)])
    for f in FEATURES:
        x = f(game)
        assert x.shape == (1000,) and x.dtype == np.float64
        assert np.isnan(x[~INSIDE]).all() and np.isfinite(x[INSIDE]).all()


def test_until_falls_from_log5_to_log104_and_mirrors_since():
    game = make_game(missing=[(200, 300)])
    s, u = t_since_seen_s(game)[200:300], t_until_seen_s(game)[200:300]
    assert np.isclose(u[0], np.log(5)) and np.isclose(u[-1], np.log(1.04))
    assert np.allclose(u, s[::-1])


def test_half_edge_counts_as_sighting():
    # decision (a): a gap touching a half edge counts from / to that edge, never past it
    game = make_game(missing=[(40, 150), (420, 450)])      # starts before half 1; ends at its last frame
    s, u = t_since_seen_s(game), t_until_seen_s(game)
    assert np.isclose(s[100], np.log1p(1 / 25)) and np.isclose(u[100], np.log1p(50 / 25))
    assert np.isclose(u[449], np.log1p(1 / 25)) and np.isclose(s[449], np.log1p(30 / 25))


def test_durations_follow_fps():
    game = make_game(fps=30, missing=[(200, 320)])         # 120 frames at 30 fps = 4 s
    assert np.allclose(gap_len_s(game)[200:320], np.log(5))
    assert np.isclose(t_since_seen_s(game)[200], np.log1p(1 / 30))


def test_one_frame_gap():
    game = make_game(missing=[(200, 201)])
    for f in FEATURES:
        x = f(game)
        assert np.isclose(x[200], np.log1p(1 / 25)) and x[199] == 0 and x[201] == 0


def test_neighbouring_gaps_stay_separate():
    game = make_game(missing=[(200, 210), (211, 261)])     # one visible frame between them
    g, s = gap_len_s(game), t_since_seen_s(game)
    assert np.allclose(g[200:210], np.log1p(10 / 25)) and np.allclose(g[211:261], np.log1p(50 / 25))
    assert g[210] == 0 and np.isclose(s[211], np.log1p(1 / 25))


def test_half_with_ball_never_visible():
    game = make_game(missing=[(550, 900)])                 # all 350 frames of half 2
    g, s, u = (f(game) for f in FEATURES)
    assert np.allclose(g[550:900], np.log1p(350 / 25))
    assert np.isclose(s[550], np.log1p(1 / 25)) and np.isclose(u[899], np.log1p(1 / 25))


def test_one_nan_coordinate_counts_as_missing():
    game = make_game()
    game["ball_xy"][200, 1] = np.nan
    assert np.isclose(gap_len_s(game)[200], np.log1p(1 / 25))


def test_input_is_not_modified():
    game = make_game(missing=[(200, 300)])
    before = copy.deepcopy(game)
    for f in FEATURES:
        f(game)
    assert np.array_equal(game["ball_xy"], before["ball_xy"], equal_nan=True)
    assert game["meta"] == before["meta"]


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
