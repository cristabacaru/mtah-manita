"""p_dead: probability that the ball is dead, one value per frame (spec sections 2-3).

Default: step 6, the logistic combiner (spec section 3, "Combiner"). The features of signals.py and
lines.py, NaN filled with the training median, standardised, then an L2 logistic regression
(DEADBALL.L2) fitted on the graded frames of P2's suite games (all noise levels), label = GT D. The
game being scored is never in its own training set. --stub writes the v0 placeholder instead: dead
where the ball has been missing for at least DEADBALL.STUB_GAP_S inside one half.

Output, in $MTAH_CACHE/<game>/: p_dead.npy, float64 (n,), in [EPS, 1 - EPS] inside halves, NaN
outside; deadball_features.npz (X (n, k), names) for P5; deadball_model.json (the fitted weights).

    python -m mtah.deadball.combine --game BAR-ATH        # fit on the whole suite, write p_dead
    python -m mtah.deadball.combine --suite               # leave-one-match-out p_dead for every suite copy
    python -m mtah.deadball.evaluate --suite              # then score those
    python -m mtah.deadball.combine --game BAR-ATH --stub # v0 placeholder

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


# --- step 6: logistic regression combiner (spec section 3, "Combiner") ---------------------------------

def feature_functions() -> dict:
    """name -> function(game) -> (n,) float64, NaN outside halves. Order = column order of X."""
    from mtah.deadball import lines, signals
    return {f.__name__: f for f in (
        signals.gap_len_s, signals.t_since_seen_s, signals.t_until_seen_s,
        lines.last_line_dist_m, lines.next_line_dist_m, lines.ball_out_m,
        signals.speed_abs, signals.speed_rel,
        lines.last_line_missing, lines.next_line_missing, lines.ball_out_missing, signals.speed_missing,
    )}


def features(game: Mapping) -> tuple[np.ndarray, list[str]]:
    """(X (n, k) float64, names): every spec section 3 feature of one game, NaN as each feature leaves it."""
    funcs = feature_functions()
    return np.column_stack([f(game) for f in funcs.values()]).astype(np.float64), list(funcs)


def _inside(game: Mapping) -> np.ndarray:
    m = np.zeros(int(game["meta"]["n_frames"]), dtype=bool)
    for half in halves(game):
        m[half] = True
    return m


def fit(X: np.ndarray, y: np.ndarray, names: list[str], l2: float = DEADBALL.L2) -> dict:
    """L2 logistic regression on rows X (m, k) with labels y (m,) in {0, 1} (1 = dead).

    NaN is filled with the training median (spec `*_missing` row), columns are standardised with
    the training mean and std, the bias is not penalised. Objective: sum of log-losses
    + l2 / 2 * |w|^2, minimised with scipy L-BFGS-B (deterministic; scikit-learn is not installed).
    """
    from scipy.optimize import minimize

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
        raise ValueError("fit needs X (m, k) and y (m,) with m > 0")
    with np.errstate(all="ignore"):
        median = np.nanmedian(X, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    Z = np.where(np.isnan(X), median, X)
    mean, std = Z.mean(axis=0), Z.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    Z = (Z - mean) / std
    k = Z.shape[1]

    def loss(theta):
        w, b = theta[:k], theta[k]
        z = Z @ w + b
        p = 0.5 * (1.0 + np.tanh(0.5 * z))                     # sigmoid without overflow
        value = np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (w @ w)
        r = p - y
        return value, np.concatenate([Z.T @ r + l2 * w, [r.sum()]])

    result = minimize(loss, np.zeros(k + 1), jac=True, method="L-BFGS-B", options={"maxiter": 1000})
    return {"names": list(names), "median": median.tolist(), "mean": mean.tolist(), "std": std.tolist(),
            "w": result.x[:k].tolist(), "b": float(result.x[k]), "l2": l2, "n_rows": int(len(y)),
            "dead_share": float(y.mean()), "converged": bool(result.success)}


def predict(model: Mapping, game: Mapping, X: np.ndarray | None = None,
            eps: float = DEADBALL.EPS) -> np.ndarray:
    """p_dead (n,) float64 for one game: model output clipped to [eps, 1 - eps] inside halves, NaN outside."""
    if X is None:
        X, names = features(game)
        if names != list(model["names"]):
            raise ValueError(f"model features {model['names']} != game features {names}")
    Z = np.where(np.isnan(X), np.asarray(model["median"]), X)
    z = ((Z - np.asarray(model["mean"])) / np.asarray(model["std"])) @ np.asarray(model["w"]) + model["b"]
    p = np.full(len(X), np.nan)
    inside = _inside(game)
    p[inside] = np.clip(0.5 * (1.0 + np.tanh(0.5 * z[inside])), eps, 1.0 - eps)
    return p


def training_rows(game: Mapping, gt_codes: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X rows, y) for the graded frames of one external game: y = 1 where the GT says D."""
    from mtah.deadball.evaluate import pm
    use = _inside(game) & (gt_codes != pm.X)
    return X[use], (gt_codes[use] == pm.D).astype(np.float64)


def suite_data(levels=None) -> list[dict]:
    """P2's suite games with their features and GT: [{copy, source, level, game, X, codes}]."""
    from mtah.data import load
    from mtah.deadball.evaluate import read_gt, read_suite
    out = []
    for row in read_suite(levels=levels):
        game = load(row["copy"])
        X, names = features(game)
        codes = read_gt(row["gt"])
        if len(codes) != len(X):
            raise ValueError(f"{row['copy']}: GT has {len(codes)} frames, the game {len(X)}")
        out.append({**row, "game": game, "X": X, "codes": codes, "names": names})
    return out


def fit_on(items: list[dict]) -> dict:
    """One model on the graded frames of every item (all noise levels of every listed game)."""
    parts = [training_rows(it["game"], it["codes"], it["X"]) for it in items]
    model = fit(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), items[0]["names"])
    model["train_sources"] = sorted({it["source"] for it in items})
    return model


def leave_one_match_out(items: list[dict], out_root: Path | None = None) -> dict:
    """For every source match: fit on the others (all levels), write p_dead.npy for its copies."""
    written = {}
    for source in sorted({it["source"] for it in items}):
        model = fit_on([it for it in items if it["source"] != source])
        for it in (i for i in items if i["source"] == source):
            p = predict(model, it["game"], it["X"])
            folder = (Path(out_root) / it["copy"]) if out_root is not None else config.game_cache(it["copy"])
            written[it["copy"]] = save_npy(folder / "p_dead.npy", p)
    return written


def run(game_name: str, game: Mapping | None = None, out_dir: Path | None = None,
        model: Mapping | None = None) -> dict:
    """Compute p_dead for one game and write it. `game` and `out_dir` are for tests.

    model=None writes the v0 placeholder; with a fitted model it also writes deadball_features.npz.
    """
    if game is None:
        from mtah.data import load
        game = load(game_name)
    out = Path(out_dir) if out_dir is not None else config.game_cache(game_name)
    if model is None:
        p = stub_p_dead(game)
    else:
        X, names = features(game)
        p = predict(model, game, X)
        np.savez(out / "deadball_features.npz", X=X, names=np.array(names))
    sha = save_npy(out / "p_dead.npy", p)
    return {**summarise(p, game), "path": str(out / "p_dead.npy"), "sha256": sha}


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="Write p_dead.npy (logistic combiner, or the v0 placeholder).")
    parser.add_argument("--game", default=config.GAME)
    parser.add_argument("--stub", action="store_true", help="write the v0 placeholder instead")
    parser.add_argument("--suite", action="store_true",
                        help="leave-one-match-out on P2's suite: write held-out p_dead.npy for every copy")
    args = parser.parse_args()

    if args.suite:
        items = suite_data()
        written = leave_one_match_out(items)
        print(f"p_dead (logistic, leave-one-match-out) written for {len(written)} suite copies "
              f"({len({i['source'] for i in items})} folds). Score: python -m mtah.deadball.evaluate --suite")
        return
    if args.stub:
        info = run(args.game)
        print("p_dead (v0 placeholder, gap >= {} s) for {}".format(DEADBALL.STUB_GAP_S, args.game))
    else:
        if not config.EXTERNAL.SUITE_MANIFEST.is_file():
            raise SystemExit(f"no {config.EXTERNAL.SUITE_MANIFEST}: build P2's suite first "
                             "(mtah/external/README.md), or use --stub")
        items = suite_data()
        own = {i["source"] for i in items if i["copy"] == args.game or i["source"] == args.game}
        model = fit_on([i for i in items if i["source"] not in own])   # never train on the game itself
        out = config.game_cache(args.game)
        out.mkdir(parents=True, exist_ok=True)
        (out / "deadball_model.json").write_text(json.dumps(model, indent=1, sort_keys=True) + "\n")
        info = run(args.game, model=model)
        print(f"p_dead (logistic, trained on {model['n_rows']} frames of {len(model['train_sources'])} "
              f"external games, converged {model['converged']}) for {args.game}")
        print("  weights (standardised): " + ", ".join(f"{n} {w:+.2f}" for n, w in zip(model["names"], model["w"])))
    print("  graded frames {graded_frames}, called dead {dead_share:.1%}, dead runs {dead_runs}".format(**info))
    print("  wrote {path}\n  sha256 {sha256}".format(**info))


if __name__ == "__main__":
    main()
