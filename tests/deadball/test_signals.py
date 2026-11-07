"""Steps 2-3: ball-gap and motion features (spec section 5, "Gap" and "Motion" bullets, fps = 30,
plus edge cases).

    python tests/deadball/test_signals.py
"""
import copy
import sys
import warnings

import numpy as np

from harness import make_game, run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.deadball.signals import (gap_len_s, speed_abs, speed_missing, speed_rel, t_since_seen_s,
                                   t_until_seen_s)

FEATURES = (gap_len_s, t_since_seen_s, t_until_seen_s)
MOTION = (speed_abs, speed_rel, speed_missing)
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



# --- step 3: motion (spec section 5, "Motion" bullet and "fps = 30") ------------------------------

def _step_game(fps, before_s=20, after_s=20, fast=3.0, slow=0.5):
    """One half of before_s + after_s seconds: every player at `fast` m/s, then at `slow` m/s."""
    n = (before_s + after_s) * fps
    game = make_game(n=n, fps=fps, halves=((0, n - 1),))
    change = before_s * fps
    game["players_speed"][:change] = fast
    game["players_speed"][change:] = slow
    return game, change


def test_spec_step_crosses_midpoint_within_half_s_and_rel_drops():
    for fps in (25, 30):                                    # spec: "fps = 30 keeps durations in seconds"
        game, change = _step_game(fps)
        a, r = speed_abs(game), speed_rel(game)
        first_below = int(np.flatnonzero(a < 1.75)[0])
        assert abs(first_below - change) <= 0.5 * fps, (fps, first_below, change)
        assert r[change + round(0.5 * fps)] < 0.3          # just after: smoothing window fully slow
        assert r[change + 4 * fps] < 0.3                    # trailing 10 s still mostly fast
        assert np.isclose(r[change + 6 * fps], 1.0)         # trailing median has turned slow


def test_spec_all_players_nan_gives_nan_and_indicator_1():
    game = make_game()
    game["players_speed"][:] = np.nan
    a, r, m = (f(game) for f in MOTION)                     # never an exception
    assert np.isnan(a).all() and np.isnan(r).all()
    assert (m[INSIDE] == 1).all() and np.isnan(m[~INSIDE]).all()


def test_motion_shape_dtype_and_nan_exactly_outside_halves():
    game = make_game()
    for f in MOTION:
        x = f(game)
        assert x.shape == (1000,) and x.dtype == np.float64
        assert np.isnan(x[~INSIDE]).all() and np.isfinite(x[INSIDE]).all()
    assert (speed_missing(game)[INSIDE] == 0).all()


def test_no_smoothing_across_half_time_or_into_outside_frames():
    # frames outside the halves are 11 m/s: any leak would move the edge values
    game = make_game()
    game["players_speed"][:] = 11.0
    game["players_speed"][100:450] = 3.0
    game["players_speed"][550:900] = 0.5
    a, r = speed_abs(game), speed_rel(game)
    assert np.allclose(a[100:450], 3.0) and np.allclose(a[550:900], 0.5)
    assert np.allclose(r[INSIDE], 1.0)                     # half 2 does not divide by half 1's speed


def test_frame_mean_skips_nan_players():
    # decision (d): one tracked player is enough; NaN players are left out of the frame mean
    game = make_game()
    game["players_speed"][:] = [1.0, 2.0, 3.0, np.nan]
    game["players_speed"][300] = [np.nan, np.nan, np.nan, 4.0]
    a = speed_abs(game)
    assert np.allclose(a[100:280], 2.0) and np.isfinite(a[INSIDE]).all()


def test_nan_frames_inside_window_are_skipped():
    # decision (c): NaN only where the whole +/- 12-frame window is NaN (25 fps, 0.5 s)
    game = make_game()
    game["players_speed"][:] = 2.0
    game["players_speed"][200:261] = np.nan                # 61 frames with nobody tracked
    a, r, m = (f(game) for f in MOTION)
    nan_expected = np.zeros(1000, bool); nan_expected[212:249] = True
    assert np.array_equal(np.isnan(a) & INSIDE, nan_expected)
    assert np.array_equal(np.isnan(r) & INSIDE, nan_expected)   # speed_rel defined where speed_abs is
    assert np.array_equal(m[INSIDE] == 1, nan_expected[INSIDE])
    assert np.allclose(a[200:212], 2.0) and np.allclose(r[200:212], 1.0)


def test_rel_floor_and_start_of_half():
    # decisions (b) and (f): at the first frame the window is t alone, so speed_rel = 1;
    # REL_FLOOR (0.1 m/s) keeps a standing team finite: 0 -> 0, 0.05 -> 0.5
    game = make_game()
    game["players_speed"][:] = 0.05
    game["players_speed"][550:900] = 0.0
    r = speed_rel(game)
    assert np.allclose(r[100:450], 0.5) and np.all(r[550:900] == 0.0)
    game["players_speed"][:] = 2.0
    assert speed_rel(game)[100] == 1.0 and speed_rel(game)[550] == 1.0


def test_motion_raises_no_numpy_warning():
    game = make_game()
    game["players_speed"][550:900] = np.nan                # a half with nobody tracked
    game["players_speed"][200:261] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for f in MOTION:
            f(game)


def test_motion_accepts_float32_and_does_not_modify_input():
    game = make_game()
    game["players_speed"] = game["players_speed"].astype(np.float32)   # as stored (mtah/data.py:130)
    before = copy.deepcopy(game)
    for f in MOTION:
        assert f(game).dtype == np.float64
    assert np.array_equal(game["players_speed"], before["players_speed"], equal_nan=True)
    assert game["players_speed"].dtype == np.float32 and game["meta"] == before["meta"]


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
