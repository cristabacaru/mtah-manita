"""Step 4: line features (spec section 5, "Line" bullet, "Units and orientation", plus edge cases).

    python tests/deadball/test_lines.py
"""
import copy
import sys
import warnings

import numpy as np

from harness import make_game, run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah import config
from mtah.deadball.lines import (ball_out_m, ball_out_missing, last_line_dist_m, last_line_missing,
                                 next_line_dist_m, next_line_missing)
from mtah.deadball.signals import gap_len_s

FEATURES = (last_line_dist_m, next_line_dist_m, ball_out_m)
INDICATORS = (last_line_missing, next_line_missing, ball_out_missing)
INSIDE = np.zeros(1000, bool)
INSIDE[100:450] = INSIDE[550:900] = True       # make_game halves are inclusive: (100, 449), (550, 899)
L, W = 105.0, 68.0                             # make_game's default pitch


def _centre_game(**kw):
    """make_game with the ball parked at the centre spot (line distance W/2 = 34) on every frame."""
    game = make_game(**kw)
    pl, pw = game["meta"]["pitch"]
    game["ball_xy"][:] = [pl / 2, pw / 2]
    for start, stop in kw.get("missing", ()):
        game["ball_xy"][start:stop] = np.nan
    return game


# --- spec section 5, "Line" -------------------------------------------------------------------------

def test_spec_ball_out_values():
    game = _centre_game()
    game["ball_xy"][200] = [L + 2, W / 2]
    game["ball_xy"][201] = [L / 2, W / 2]
    game["ball_xy"][202] = [-10, W / 2]
    out = ball_out_m(game)
    assert np.isclose(out[200], 2.0) and out[201] == 0.0 and out[202] == config.DEADBALL.OUT_CAP_M


def test_spec_last_seen_at_y_half_metre_holds_for_whole_gap():
    game = _centre_game(missing=[(200, 300)])
    game["ball_xy"][199] = [L / 2, 0.5]
    last = last_line_dist_m(game)
    assert np.allclose(last[200:300], 0.5)


# --- spec section 5, "Units and orientation" --------------------------------------------------------

def test_spec_mirrored_x_gives_identical_features():
    game = make_game(missing=[(200, 300), (420, 601)])
    game["ball_xy"][150] = [L + 3, 10]                     # one ball outside, so ball_out is not all 0
    mirrored = copy.deepcopy(game)
    mirrored["ball_xy"][:, 0] = L - mirrored["ball_xy"][:, 0]
    rotated = copy.deepcopy(mirrored)
    rotated["ball_xy"][:, 1] = W - rotated["ball_xy"][:, 1]   # 180 degree turn, the BAR-ATH flip
    for f in FEATURES:
        a, b, c = f(game), f(mirrored), f(rotated)
        assert np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-12)
        assert np.allclose(a, c, equal_nan=True, rtol=0, atol=1e-12)


def test_spec_pitch_100_by_60_moves_lines():
    game = _centre_game(pitch=(100.0, 60.0))
    game["ball_xy"][200] = [102.0, 30.0]                  # 2 m past the goal line of a 100 m pitch
    game["ball_xy"][201] = [50.0, 59.0]                   # 1 m inside the far touchline (W = 60)
    assert np.isclose(ball_out_m(game)[200], 2.0)
    assert np.isclose(last_line_dist_m(game)[201], 1.0)
    assert np.isclose(last_line_dist_m(game)[300], 30.0)  # centre spot: W/2 = 30


# --- step 4 decisions and edge cases ----------------------------------------------------------------

def test_shape_dtype_and_nan_exactly_outside_halves():
    game = make_game()                                    # ball visible everywhere
    for f in FEATURES + INDICATORS:
        x = f(game)
        assert x.shape == (1000,) and x.dtype == np.float64
        assert np.isnan(x[~INSIDE]).all() and np.isfinite(x[INSIDE]).all()
    for f in INDICATORS:
        assert (f(game)[INSIDE] == 0).all()


def test_visible_ball_gives_current_distance():
    # decision (a): visible frames carry the current ball's line distance, both features
    game = _centre_game()
    game["ball_xy"][200] = [3.0, 30.0]
    last, nxt = last_line_dist_m(game), next_line_dist_m(game)
    assert last[200] == 3.0 and nxt[200] == 3.0 and np.allclose(last[INSIDE & (np.arange(1000) != 200)], 34.0)


def test_next_takes_first_sighting_after_gap():
    game = _centre_game(missing=[(200, 300)])
    game["ball_xy"][199] = [L / 2, 0.5]
    game["ball_xy"][300] = [L - 1.5, W / 2]
    nxt = next_line_dist_m(game)
    assert np.allclose(nxt[200:300], 1.5) and np.isclose(nxt[199], 0.5) and np.isclose(nxt[300], 1.5)


def test_gap_at_half_edges_gives_nan_and_indicator():
    # decision (b): no sighting on that side inside the half -> NaN on that side only
    game = _centre_game(missing=[(40, 150), (420, 601)])  # opens half 1; crosses half-time
    game["ball_xy"][150] = [1.0, 30.0]                    # first sighting in half 1
    game["ball_xy"][419] = [L / 2, 2.0]                   # last sighting before the long gap
    game["ball_xy"][601] = [L / 2, W - 4.0]               # first sighting in half 2
    last, nxt = last_line_dist_m(game), next_line_dist_m(game)
    lm, nm = last_line_missing(game), next_line_missing(game)
    assert np.isnan(last[100:150]).all() and np.allclose(nxt[100:150], 1.0)
    assert np.allclose(last[420:450], 2.0) and np.isnan(nxt[420:450]).all()
    assert np.isnan(last[550:601]).all() and np.allclose(nxt[550:601], 4.0)
    expect_lm = np.zeros(1000, bool); expect_lm[100:150] = expect_lm[550:601] = True
    expect_nm = np.zeros(1000, bool); expect_nm[420:450] = True
    assert np.array_equal(lm[INSIDE] == 1, expect_lm[INSIDE]) and np.array_equal(nm[INSIDE] == 1, expect_nm[INSIDE])


def test_half_with_ball_never_visible():
    game = _centre_game(missing=[(550, 900)])
    for f in (last_line_dist_m, next_line_dist_m, ball_out_m):
        assert np.isnan(f(game)[550:900]).all()
    for f in INDICATORS:
        assert (f(game)[550:900] == 1).all()


def test_ball_out_nan_while_missing_and_indicator_matches_gaps():
    # decision (c): ball_out_m is NaN in a gap; ball_out_missing = 1 exactly where gap_len_s > 0
    game = make_game(missing=[(200, 300), (420, 601)])
    out, om = ball_out_m(game), ball_out_missing(game)
    assert np.isnan(out[200:300]).all()
    assert np.array_equal(om[INSIDE] == 1, gap_len_s(game)[INSIDE] > 0)


def test_ball_out_corner_is_straight_line_distance():
    # decision (d): 0.6 m past the goal line and 0.8 m past the touchline -> 1.0 m
    game = _centre_game()
    game["ball_xy"][200] = [L + 0.6, -0.8]
    game["ball_xy"][201] = [L + 3.0, W + 4.0]             # sqrt(9 + 16) = 5 = OUT_CAP_M
    game["ball_xy"][202] = [L + 30.0, W + 40.0]           # 50 m out -> capped
    out = ball_out_m(game)
    assert np.isclose(out[200], 1.0) and np.isclose(out[201], 5.0)
    assert out[202] == config.DEADBALL.OUT_CAP_M


def test_halfway_line_is_not_a_line():
    # decision (e): a ball on the halfway line, far from the outer lines
    game = _centre_game()
    game["ball_xy"][200] = [L / 2, 20.0]
    assert np.isclose(last_line_dist_m(game)[200], 20.0)


def test_ball_on_or_past_a_line_has_distance_0():
    # option 1 (2026-11-07): outside the pitch counts as "at the line", never as far from it
    game = _centre_game(missing=[(201, 250)])
    game["ball_xy"][197] = [-2.0, W / 2]                  # 2 m behind a goal line
    game["ball_xy"][198] = [L / 2, W + 30.0]              # 30 m past a touchline
    game["ball_xy"][199] = [L / 2, 0.0]                   # on the line
    game["ball_xy"][200] = [L + 3.0, W / 2]               # last sighting before the gap: 3 m out
    last = last_line_dist_m(game)
    assert (last[197:201] == 0.0).all() and (last[201:250] == 0.0).all()
    assert np.isclose(ball_out_m(game)[200], 3.0)         # how far out stays in ball_out_m


def test_one_nan_coordinate_counts_as_missing():
    game = _centre_game()
    game["ball_xy"][200, 1] = np.nan
    game["ball_xy"][199] = [L / 2, 1.0]
    assert np.isnan(ball_out_m(game)[200]) and np.isclose(last_line_dist_m(game)[200], 1.0)


def test_neighbouring_gaps_use_their_own_sightings():
    game = _centre_game(missing=[(200, 210), (211, 261)])
    game["ball_xy"][199] = [L / 2, 1.0]
    game["ball_xy"][210] = [L / 2, 2.0]                   # the single visible frame between them
    game["ball_xy"][261] = [L / 2, 3.0]
    last, nxt = last_line_dist_m(game), next_line_dist_m(game)
    assert np.allclose(last[200:210], 1.0) and np.allclose(nxt[200:210], 2.0)
    assert np.allclose(last[211:261], 2.0) and np.allclose(nxt[211:261], 3.0)


def test_bad_pitch_is_rejected():
    for pitch in (None, [105.0], [0.0, 68.0], [105.0, np.nan]):
        game = make_game()
        game["meta"]["pitch"] = pitch
        try:
            ball_out_m(game)
        except ValueError:
            continue
        raise AssertionError(f"pitch {pitch!r} was accepted")


def test_no_numpy_warning_and_input_not_modified():
    game = make_game(missing=[(40, 150), (550, 900)])
    game["ball_xy"] = game["ball_xy"].astype(np.float32)  # as stored (mtah/data.py:128)
    game["ball_xy"][300, 0] = np.nan                       # one NaN coordinate
    before = copy.deepcopy(game)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for f in FEATURES + INDICATORS:
            assert f(game).dtype == np.float64
    assert np.array_equal(game["ball_xy"], before["ball_xy"], equal_nan=True)
    assert game["ball_xy"].dtype == np.float32 and game["meta"] == before["meta"]


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
