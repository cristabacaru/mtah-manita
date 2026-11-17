"""Synthetic tests for mtah/ball.py, mtah/features.py and mtah/model.py (P4). No match data needed.

    python tests/test_possession.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MTAH_CACHE"] = tempfile.mkdtemp(prefix="mtah_test_")   # before importing config

from mtah import ball, features, model  # noqa: E402
from mtah.config import POSSESSION  # noqa: E402

FPS = 25
L, W = 105.0, 68.0


def make_game(n=1000, halves=((50, 499), (520, 979)), seed=0):
    """Two players per team. In each half H dribbles for the first 40%, then A for the rest.

    The ball sits 0.5 m ahead of the dribbler; the other team stands 10 m away.
    """
    rng = np.random.default_rng(seed)
    pxy = np.full((n, 4, 2), np.nan)
    bxy = np.full((n, 2), np.nan)
    labels = np.full(n, "X", dtype="<U1")
    for first, last in halves:
        t = np.arange(first, last + 1)
        m = len(t)
        x = 20 + 60 * (t - first) / m                    # everyone runs along x at ~3 m/s
        y = np.full(m, 30.0)
        h_owns = (t - first) < 0.4 * m
        pxy[t, 0] = np.column_stack([x, y])              # H dribbler
        pxy[t, 1] = np.column_stack([x - 15, y + 10])    # H other
        pxy[t, 2] = np.column_stack([x, y + 10])         # A, near in x, 10 m off in y
        pxy[t, 3] = np.column_stack([x - 5, y - 15])     # A other
        owner = np.where(h_owns, 0, 2)
        bxy[t] = pxy[t, owner] + [0.5, 0.0]
        # A takes over: put A's player 2 on the ball in the second part
        pxy[t[~h_owns], 0] += [0.0, 10.0]                # H dribbler moves away
        pxy[t[~h_owns], 2] -= [0.0, 10.0]
        bxy[t[~h_owns]] = pxy[t[~h_owns], 2] + [0.5, 0.0]
        labels[t] = np.where(h_owns, "H", "A")
    bxy += rng.normal(0, 0.05, bxy.shape)
    game = {"ball_xy": bxy, "players_xy": pxy, "players_speed": np.full((n, 4), 3.0),
            "players_team": np.array(["H", "H", "A", "A"]),
            "meta": {"fps": FPS, "n_frames": n, "halves": [list(h) for h in halves], "pitch": [L, W]}}
    return game, labels


def line_game(n=400, gap=(100, 120)):
    t = np.arange(n) / FPS
    bxy = np.column_stack([10 + 5 * t, 20 + 2 * t])      # 5.4 m/s straight line
    truth = bxy.copy()
    bxy[gap[0]:gap[1]] = np.nan
    game = {"ball_xy": bxy, "meta": {"fps": FPS, "n_frames": n, "halves": [[0, n - 1]], "pitch": [L, W]}}
    return game, truth


# ------------------------------------------------------------------------------------------------
def test_bridge_fills_short_gap_on_a_line():
    game, truth = line_game(gap=(100, 120))              # 0.8 s gap
    b = ball.bridge(game)
    assert b["bridged"][100:120].all() and not b["bridged"][:100].any()
    err = np.hypot(*(b["xy"] - truth).T)
    assert err.max() < 0.05, err.max()
    v = np.hypot(*b["vel"][50:350].T)
    assert np.allclose(v, np.hypot(5, 2), atol=0.05), (v.min(), v.max())


def test_bridge_leaves_long_and_edge_gaps():
    long_gap = int(POSSESSION.BRIDGE_MAX_S * FPS) + 1
    game, _ = line_game(gap=(100, 100 + long_gap))
    b = ball.bridge(game)
    assert np.isnan(b["xy"][100:100 + long_gap]).all() and not b["bridged"].any()
    game, _ = line_game(gap=(0, 10))                     # touches the half start
    b = ball.bridge(game)
    assert np.isnan(b["xy"][:10]).all() and np.isfinite(b["xy"][10:]).all()


def test_features_shape_nan_outside_halves():
    game, _ = make_game()
    X, names = features.compute(game)
    assert X.shape == (1000, len(names)) and len(set(names)) == len(names)
    assert np.isnan(X[:50]).all() and np.isnan(X[500:520]).all() and np.isnan(X[980:]).all()
    assert np.isfinite(X[60:490, names.index("d_diff")]).all()


def test_control_and_last_team():
    game, labels = make_game()
    X, names = features.compute(game)
    last = X[:, names.index("ctrl_last")]
    inside = labels != "X"
    agree = (np.where(last > 0, "H", np.where(last < 0, "A", "?")) == labels)[inside].mean()
    assert agree > 0.95, agree


def test_windows_stay_inside_half():
    game, _ = make_game()
    X, names = features.compute(game)
    # first frame of half 2: no past inside the half, so the past window is NaN, not half-1 data
    assert np.isnan(X[520, names.index("ctrl_past_1s")])
    assert np.isnan(X[499, names.index("ctrl_fut_1s")])


def test_mirror_is_an_involution_and_flips_sides():
    game, _ = make_game()
    twice = features.mirror(features.mirror(game))
    assert np.allclose(np.nan_to_num(twice["players_xy"]), np.nan_to_num(game["players_xy"]))
    assert (twice["players_team"] == game["players_team"]).all()
    X, names = features.compute(game)
    Xm, _ = features.compute(features.mirror(game))
    j = names.index("d_diff")
    ok = np.isfinite(X[:, j])
    assert np.allclose(Xm[ok, j], -X[ok, j], atol=1e-6)
    j = names.index("ctrl_last")
    assert np.array_equal(np.nan_to_num(Xm[:, j]), -np.nan_to_num(X[:, j]))


def test_v0_and_oracle_scoring():
    game, labels = make_game()
    p = model.finish(model.v0_p_home(game), game)
    assert np.isnan(p[:50]).all() and np.isfinite(p[50:500]).all()
    assert np.nanmin(p) >= POSSESSION.EPS and np.nanmax(p) <= 1 - POSSESSION.EPS
    m = model.frame_metrics(p, labels)
    assert m["acc"] > 0.95, m
    perfect = np.where(labels == "H", 0.99, 0.01)
    pred = model.oracle_d_labels(perfect, labels, game)
    assert (pred == labels).all()
    assert model.spell_scores(pred, labels)["spellF1"] == 1.0


def test_isotonic_monotone_and_calibrates():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p ** 2).astype(float)   # true P(y) = p^2: miscalibrated input
    c = model.fit_isotonic(p, y)
    out = model.apply_isotonic(np.linspace(0, 1, 101), c)
    assert np.all(np.diff(out) >= -1e-12)
    assert abs(model.apply_isotonic(np.array([0.7]), c)[0] - 0.49) < 0.1


def test_deterministic():
    game, _ = make_game()
    a, _ = features.compute(game)
    b, _ = features.compute(game)
    assert np.array_equal(a, b, equal_nan=True)


def test_touch_detected_on_a_kick():
    """Ball rolls at 2 m/s, an A player at the ball kicks it to 15 m/s: one A touch at the kick."""
    n, kick = 300, 150
    t = np.arange(n) / FPS
    x = np.where(np.arange(n) < kick, 20 + 2 * t, 20 + 2 * t[kick] + 15 * (t - t[kick]))
    bxy = np.column_stack([x, np.full(n, 30.0)])
    pxy = np.full((n, 2, 2), np.nan)
    pxy[:, 0] = [x[kick], 30.5]                        # A player standing where the kick happens
    pxy[:, 1] = [x[kick] - 30, 40.0]                   # H player far away
    game = {"ball_xy": bxy, "players_xy": pxy, "players_speed": np.zeros((n, 2)),
            "players_team": np.array(["A", "H"]),
            "meta": {"fps": FPS, "n_frames": n, "halves": [[0, n - 1]], "pitch": [L, W]}}
    touch = features._frame_quantities(game)["touch"]
    hits = np.flatnonzero(touch != 0)
    assert len(hits) == 1 and touch[hits[0]] == -1 and abs(hits[0] - kick) <= 3, hits
    X, names = features.compute(game)
    assert X[kick + 20, names.index("touch_last")] == -1
    Xm, _ = features.compute(features.mirror(game))
    assert Xm[kick + 20, names.index("touch_last")] == 1     # A's touch is H's touch in the mirror


def test_frame_weights():
    fps = 25
    lab = np.array(["H"] * 100 + ["A"] * 30 + ["H"] * 100 + ["D"] * 50 + ["A"] * 200)
    w = model.frame_weights(lab, fps)
    assert w[0] == 1 and w[99] == POSSESSION.TURNOVER_WEIGHT          # 1 s before a turnover
    assert w[115] == max(POSSESSION.TURNOVER_WEIGHT, POSSESSION.SHORT_SPELL_WEIGHT)  # 1.2 s spell
    assert w[230] == 1 and w[400] == 1                                 # D -> A is not a direct turnover


def test_turn_target_and_auc():
    lab = np.array(["H"] * 100 + ["A"] * 100 + ["D"] * 50 + ["H"] * 100)
    y = model.turn_target(lab, 25)
    k = round(POSSESSION.TURN_WINDOW_S * 25)
    assert y[100 - k:100 + k].all() and y.sum() == 2 * k          # only the direct H -> A change
    rng = np.random.default_rng(0)
    yy = rng.uniform(size=2000) < 0.3
    assert model.auc(yy.astype(float), yy) == 1.0
    assert abs(model.auc(rng.uniform(size=2000), yy) - 0.5) < 0.05


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as err:  # noqa: BLE001 (report every failure, then exit 1)
                failed += 1
                print(f"FAIL  {name}: {type(err).__name__}: {err}")
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
