"""Step 6, the logistic combiner in combine.py (spec section 5, "Combiner"). Synthetic, no match data.

    python tests/deadball/test_regression.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from harness import make_game, run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.config import DEADBALL
from mtah.deadball import combine as C
from mtah.deadball.evaluate import pm

INSIDE = np.zeros(1000, bool)
INSIDE[100:450] = INSIDE[550:900] = True       # make_game halves (100, 449), (550, 899)


def gt_for(missing):
    """X outside halves, D where the ball is missing, H elsewhere."""
    c = np.full(1000, pm.X, dtype=np.int8)
    c[INSIDE] = pm.H
    for a, b in missing:
        c[a:b] = np.where(INSIDE[a:b], pm.D, c[a:b])
    return c


def item(source, missing, seed=0):
    g = make_game(missing=missing, seed=seed)
    X, names = C.features(g)
    return {"copy": source + "__x", "source": source, "game": g, "X": X, "codes": gt_for(missing), "names": names}


def test_spec_separable_toy_in_eps_range_and_deterministic():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-3, 0.5, 500), rng.normal(3, 0.5, 500)])[:, None]
    y = np.r_[np.zeros(500), np.ones(500)]
    m1, m2 = C.fit(x, y, ["x"]), C.fit(x, y, ["x"])
    assert m1 == m2 and m1["converged"] and m1["w"][0] > 0
    g = {"meta": {"n_frames": 1000, "halves": [[0, 999]]}}
    p = C.predict(m1, g, X=x)
    assert (p >= DEADBALL.EPS).all() and (p <= 1 - DEADBALL.EPS).all()
    assert p[:500].max() < 0.5 < p[500:].min()


def test_nan_filled_with_training_median_and_constant_column_ok():
    x = np.array([[0.0, 1.0], [np.nan, 1.0], [2.0, 1.0], [4.0, 1.0]])
    m = C.fit(x, np.array([0, 0, 1, 1.0]), ["a", "const"])
    assert m["median"][0] == 2.0 and m["std"][1] == 1.0
    g = {"meta": {"n_frames": 4, "halves": [[0, 3]]}}
    assert np.isfinite(C.predict(m, g, X=x)).all()


def test_l2_shrinks_the_weights():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, 1))
    y = (x[:, 0] + rng.normal(scale=0.5, size=400) > 0).astype(float)
    assert abs(C.fit(x, y, ["x"], l2=1000.0)["w"][0]) < abs(C.fit(x, y, ["x"], l2=0.0)["w"][0])


def test_training_rows_are_graded_frames_only():
    it = item("g1", [(200, 300)])
    X, y = C.training_rows(it["game"], it["codes"], it["X"])
    assert len(X) == INSIDE.sum() and y.sum() == 100


def test_predict_nan_outside_halves_and_learns_the_gap():
    items = [item(f"g{i}", [(150 + 20 * i, 300 + 20 * i), (600, 700 + 10 * i)], seed=i) for i in range(3)]
    model = C.fit_on(items)
    assert model["train_sources"] == ["g0", "g1", "g2"]
    p = C.predict(model, items[0]["game"], items[0]["X"])
    assert np.isnan(p[~INSIDE]).all() and np.isfinite(p[INSIDE]).all()
    assert p[200:290].mean() > 0.5 > p[380:440].mean()          # inside the gap vs ball visible


def test_leave_one_match_out_writes_every_copy_and_is_deterministic():
    items = [item(f"g{i}", [(150 + 20 * i, 300 + 20 * i)], seed=i) for i in range(3)]
    with tempfile.TemporaryDirectory() as d:
        a = C.leave_one_match_out(items, out_root=Path(d) / "a")
        b = C.leave_one_match_out(items, out_root=Path(d) / "b")
        assert a == b and sorted(a) == ["g0__x", "g1__x", "g2__x"]
        assert (Path(d) / "a" / "g1__x" / "p_dead.npy").is_file()


def test_run_with_model_writes_p_dead_and_features():
    items = [item(f"g{i}", [(150 + 20 * i, 300 + 20 * i)], seed=i) for i in range(2)]
    model = C.fit_on(items)
    with tempfile.TemporaryDirectory() as d:
        info = C.run("FAKE", game=items[0]["game"], out_dir=Path(d), model=model)
        assert sorted(p.name for p in Path(d).iterdir()) == ["deadball_features.npz", "p_dead.npy"]
        z = np.load(Path(d) / "deadball_features.npz")
        assert z["X"].shape == (1000, len(model["names"])) and list(z["names"]) == model["names"]
        assert info["graded_frames"] == 700


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
