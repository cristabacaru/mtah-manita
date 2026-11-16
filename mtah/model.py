"""p_home[t] = P(H in possession | ball alive), the P4 output for the decoder (README module contract).

    python -m mtah.model --train                 # leave-one-match-out on the external suite, report,
                                                 # out-of-fold p_home.npy for every suite game,
                                                 # then the final model on all of them
    python -m mtah.model --game BAR-ATH          # p_home.npy with the final model (trains it if missing)
    python -m mtah.model --game BAR-ATH --v0     # rule only: last controlling team, no training
    python -m mtah.model --ablate                # LOMO with each feature group removed (writes p4/ablation.json only)

Output: $MTAH_CACHE/<game>/p_home.npy, float64 (n,), in [EPS, 1 - EPS] inside the halves, NaN outside.

Training data: the suite (config.EXTERNAL: 9 external games x 3 noise levels), alive frames only
(labels H / A), every TRAIN_STRIDE-th frame, mirrored copies added (features.mirror). Folds: one per
source match, its three noise levels together, so nothing of the held-out match is seen in training.
Calibration: an isotonic map from raw to calibrated probability, fitted on out-of-fold predictions.
No BAR-ATH labels exist and none are used; BAR-ATH is only ever predicted.

Evaluation (all out of fold, external games only):
  * frame metrics on alive frames: log-loss, accuracy, ECE;
  * oracle-D spellF1: D copied from the GT, H / A from p_home through a 2-state Viterbi with a fixed
    switch cost. This isolates the team model from p_dead; the full pipeline is scored by P5.
Prints aggregates only.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from mtah import config, data, features
from mtah.config import EXTERNAL, POSSESSION
from mtah.deadball.util import halves, save_npy

REPO = config.REPO_ROOT
sys.path.insert(0, str(REPO / "evaluation"))
import possession_metrics as pm  # noqa: E402  (official scorer, unedited)

MODEL_PATH = POSSESSION.MODEL_DIR / "p_home_lgbm.txt"
CALIB_PATH = POSSESSION.MODEL_DIR / "p_home_isotonic.json"
REPORT_PATH = POSSESSION.MODEL_DIR / "report.json"
ORACLE_SWITCH = 8.0        # fixed 2-state switch cost (nats) of the oracle-D evaluation only


# ------------------------------------------------------------------------------------------------
# v0: rule
# ------------------------------------------------------------------------------------------------
def v0_p_home(game) -> np.ndarray:
    """Last controlling team carried forward; 0.5 before the first control event of a half."""
    X, names = features.compute(game)
    last = X[:, names.index("ctrl_last")]
    c = POSSESSION.V0_CONF
    return np.where(np.isnan(last), np.nan, np.where(last > 0, c, np.where(last < 0, 1 - c, 0.5)))


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
def suite_games() -> list[tuple[str, str]]:
    """(cache name, fold) for every suite game; fold = the source match."""
    return [(f"{g}__{lvl}", g) for g in EXTERNAL.SUITE_GAMES for lvl in EXTERNAL.LEVELS]


def load_labels(name: str) -> np.ndarray:
    path = config.game_cache(name) / "labels.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: build the suite first (mtah/external/README.md)")
    return np.load(path)


def game_rows(name: str) -> dict:
    """Features of one game and its mirror, labels, and which rows are used for training."""
    game = data.load(name, mmap=False)
    labels = load_labels(name)
    X, names = features.compute(game)
    Xm, _ = (features.compute(features.mirror(game)) if POSSESSION.MIRROR or POSSESSION.TTA
             else (None, None))
    alive = np.isin(labels, ("H", "A"))
    train = np.zeros(len(labels), dtype=bool)
    train[np.flatnonzero(alive)[::POSSESSION.TRAIN_STRIDE]] = True
    return {"name": name, "game": game, "labels": labels, "X": X, "Xm": Xm, "names": names,
            "alive": alive, "train": train}


def training_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for r in rows:
        Xs.append(r["X"][r["train"]])
        ys.append(r["labels"][r["train"]] == "H")
        if POSSESSION.MIRROR:
            Xs.append(r["Xm"][r["train"]])
            ys.append(r["labels"][r["train"]] == "A")        # H and A swap in the mirror
    return np.vstack(Xs), np.concatenate(ys).astype(np.float64)


# ------------------------------------------------------------------------------------------------
# model
# ------------------------------------------------------------------------------------------------
def fit(rows: list[dict]):
    import lightgbm as lgb
    X, y = training_matrix(rows)
    params = dict(POSSESSION.LGBM, seed=POSSESSION.SEED)
    ds = lgb.Dataset(X, y, feature_name=rows[0]["names"], free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=POSSESSION.N_ROUNDS)


def predict_raw(booster, X: np.ndarray, Xm: np.ndarray | None) -> np.ndarray:
    """Raw P(H) on every frame; with TTA, averaged with 1 - P(H) of the mirrored frame."""
    p = booster.predict(X)
    if POSSESSION.TTA and Xm is not None:
        p = 0.5 * (p + 1.0 - booster.predict(Xm))
    return p


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> dict:
    """Increasing map raw -> calibrated, as knots for np.interp (blocks of the PAV solution)."""
    from scipy.optimize import isotonic_regression
    order = np.argsort(p, kind="stable")
    res = isotonic_regression(y[order], increasing=True)
    fitted, ps = res.x, p[order]
    starts = res.blocks[:-1]
    stops = res.blocks[1:]
    knots_x = np.array([ps[a:b].mean() for a, b in zip(starts, stops)])
    knots_y = fitted[starts]
    return {"x": knots_x.tolist(), "y": knots_y.tolist()}


def apply_isotonic(p: np.ndarray, calib: dict | None) -> np.ndarray:
    if not calib:
        return p
    return np.interp(p, calib["x"], calib["y"])


def finish(p: np.ndarray, game) -> np.ndarray:
    """Clip to [EPS, 1 - EPS] inside the halves, NaN outside."""
    out = np.full(int(game["meta"]["n_frames"]), np.nan)
    for half in halves(game):
        out[half] = np.clip(np.nan_to_num(p[half], nan=0.5), POSSESSION.EPS, 1 - POSSESSION.EPS)
    return out


# ------------------------------------------------------------------------------------------------
# evaluation
# ------------------------------------------------------------------------------------------------
def frame_metrics(p: np.ndarray, labels: np.ndarray, bins: int = 10) -> dict:
    alive = np.isin(labels, ("H", "A"))
    y = (labels[alive] == "H").astype(np.float64)
    q = np.clip(p[alive], 1e-6, 1 - 1e-6)
    logloss = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
    acc = float(np.mean((q > 0.5) == (y > 0.5)))
    idx = np.minimum((q * bins).astype(int), bins - 1)
    ece = sum(abs(q[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean()
              for k in range(bins) if (idx == k).any())
    return {"logloss": logloss, "acc": acc, "ece": float(ece)}


def oracle_d_labels(p: np.ndarray, labels: np.ndarray, game, switch: float = ORACLE_SWITCH) -> np.ndarray:
    """D from the GT; H / A from p through a Viterbi with switch cost `switch`, per half."""
    from mtah.decode import viterbi
    big = 1e9
    q = np.clip(np.nan_to_num(p, nan=0.5), 1e-6, 1 - 1e-6)
    dead = labels == "D"
    logp = np.column_stack([np.log(q), np.log(1 - q), np.full(len(q), -big)])
    logp[dead] = (-big, -big, 0.0)
    out = np.full(len(labels), "X", dtype="<U1")
    for half in halves(game):
        out[half] = np.array(["H", "A", "D"])[viterbi(logp[half], switch_team=switch, switch_dead=0.0)]
    return out


def spell_scores(pred: np.ndarray, labels: np.ndarray) -> dict:
    r = pm.evaluate(pm.label_codes(labels), pm.label_codes(pred))
    return {"spellF1": float(r["spell_f1_mean"]), "macroF1": float(r["macro_f1"])}


def evaluate_game(p: np.ndarray, r: dict) -> dict:
    return {**frame_metrics(p, r["labels"]), **spell_scores(oracle_d_labels(p, r["labels"], r["game"]), r["labels"])}


# ------------------------------------------------------------------------------------------------
# commands
# ------------------------------------------------------------------------------------------------
def load_rows() -> list[dict]:
    rows = []
    for name, fold in suite_games():
        r = game_rows(name)
        r["fold"] = fold
        rows.append(r)
        print(f"  features {name}: {r['X'].shape[1]} columns, {int(r['train'].sum())} training rows")
    return rows


def lomo(rows: list[dict], t0: float) -> dict:
    """Out-of-fold calibrated p_home per game: model and isotonic map both fitted without its match."""
    raw = {}
    for f in sorted({r["fold"] for r in rows}):
        booster = fit([r for r in rows if r["fold"] != f])
        for r in rows:
            if r["fold"] == f:
                raw[r["name"]] = predict_raw(booster, r["X"], r["Xm"])
        print(f"  fold {f} done ({time.time() - t0:.0f} s)")
    out = {}
    for r in rows:
        calib = (fit_isotonic(*oof_xy(raw, [o for o in rows if o["fold"] != r["fold"]]))
                 if POSSESSION.CALIBRATE else None)
        out[r["name"]] = finish(apply_isotonic(raw[r["name"]], calib), r["game"])
    return {"p": out, "raw": raw}


def oof_xy(raw: dict, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    p = np.concatenate([raw[r["name"]][r["train"]] for r in rows])
    y = np.concatenate([(r["labels"][r["train"]] == "H") for r in rows]).astype(np.float64)
    return p, y


def without(rows: list[dict], group: str) -> list[dict]:
    """Copies of the rows with one feature group's columns removed."""
    keep = [i for i, c in enumerate(rows[0]["names"]) if features.group_of(c) != group]
    names = [rows[0]["names"][i] for i in keep]
    return [{**r, "X": r["X"][:, keep], "Xm": None if r["Xm"] is None else r["Xm"][:, keep],
             "names": names} for r in rows]


def paired_delta(a: dict, b: dict, rows: list[dict], key: str = "spellF1", n_boot: int = 2000) -> dict:
    """Mean of b - a over game x level, with a 90% bootstrap CI over matches (the folds).

    Matches, not game-levels, are resampled: the three noise levels of one match are not independent.
    """
    folds = sorted({r["fold"] for r in rows})
    per_fold = np.array([np.mean([b[r["name"]][key] - a[r["name"]][key] for r in rows if r["fold"] == f])
                         for f in folds])
    rng = np.random.default_rng(POSSESSION.SEED)
    boots = per_fold[rng.integers(0, len(folds), (n_boot, len(folds)))].mean(axis=1)
    lo, hi = np.percentile(boots, [5, 95])
    return {"delta": float(per_fold.mean()), "ci90": [float(lo), float(hi)]}


def ablate() -> dict:
    """Full model, then one feature group removed at a time. Writes only p4/ablation.json."""
    t0 = time.time()
    rows = load_rows()
    full = lomo(rows, t0)["p"]
    base = {r["name"]: evaluate_game(full[r["name"]], r) for r in rows}
    out = {"full": summarise({n: {"model": s, "v0": s} for n, s in base.items()})["model/all"], "drop": {}}
    for group in features.GROUPS:
        sub = without(rows, group)
        p = lomo(sub, t0)["p"]
        scores = {r["name"]: evaluate_game(p[r["name"]], r) for r in rows}
        out["drop"][group] = {
            "n_cols_removed": len(rows[0]["names"]) - len(sub[0]["names"]),
            "mean": summarise({n: {"model": s, "v0": s} for n, s in scores.items()})["model/all"],
            "spellF1": paired_delta(base, scores, rows),
            "logloss": paired_delta(base, scores, rows, key="logloss"),
        }
        del sub
    path = POSSESSION.MODEL_DIR / "ablation.json"
    POSSESSION.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    f = out["full"]
    print(f"\n  ablation (oracle-D, LOMO, mean over 9 matches x 3 levels); full: spellF1 {f['spellF1']:.4f}, "
          f"logloss {f['logloss']:.4f}")
    print(f"  {'drop':<11}{'cols':>5}{'spellF1':>9}{'delta':>9}  {'90% CI over matches':<20}{'d logloss':>10}")
    for g, d in out["drop"].items():
        s = d["spellF1"]
        print(f"  {g:<11}{d['n_cols_removed']:>5}{d['mean']['spellF1']:>9.4f}{s['delta']:>+9.4f}  "
              f"[{s['ci90'][0]:+.4f}, {s['ci90'][1]:+.4f}]  {d['logloss']['delta']:>+9.4f}")
    print(f"  wrote {path} ({time.time() - t0:.0f} s)")
    return out


def train(log_change: str | None = None) -> dict:
    t0 = time.time()
    rows = load_rows()
    res = lomo(rows, t0)
    raw = res["raw"]
    report = {"games": {}, "settings": {k: v for k, v in vars(POSSESSION).items()
                                         if k.isupper() and k != "MODEL_DIR"}}
    for r in rows:
        p = res["p"][r["name"]]
        p_v0 = v0_p_home(r["game"])
        report["games"][r["name"]] = {"model": evaluate_game(p, r), "v0": evaluate_game(finish(p_v0, r["game"]), r)}
        save_npy(config.game_cache(r["name"]) / "p_home.npy", p)      # honest input for P5's tuning

    # final model on every suite game
    POSSESSION.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster = fit(rows)
    booster.save_model(str(MODEL_PATH))
    calib = fit_isotonic(*oof_xy(raw, rows)) if POSSESSION.CALIBRATE else None
    CALIB_PATH.write_text(json.dumps(calib))
    imp = booster.feature_importance(importance_type="gain")
    report["importance"] = dict(sorted(zip(rows[0]["names"], imp.tolist()), key=lambda kv: -kv[1]))
    report["summary"] = summarise(report["games"])
    report["seconds"] = round(time.time() - t0, 1)
    REPORT_PATH.write_text(json.dumps(report, indent=1, default=float))
    print_summary(report)
    if log_change:
        log_experiment(report, log_change)
    return report


def summarise(per_game: dict) -> dict:
    out = {}
    for variant in ("model", "v0"):
        for level in EXTERNAL.LEVELS + ("all",):
            sel = [g[variant] for n, g in per_game.items() if level == "all" or n.endswith("__" + level)]
            out[f"{variant}/{level}"] = {k: float(np.mean([s[k] for s in sel])) for k in sel[0]}
    return out


def print_summary(report: dict) -> None:
    print("\n  out of fold, mean over games (oracle-D: D from GT, team from p_home)")
    print(f"  {'variant/level':<16}{'spellF1':>9}{'macroF1':>9}{'acc':>8}{'logloss':>9}{'ece':>8}")
    for k, s in report["summary"].items():
        print(f"  {k:<16}{s['spellF1']:>9.4f}{s['macroF1']:>9.4f}{s['acc']:>8.4f}{s['logloss']:>9.4f}{s['ece']:>8.4f}")
    top = list(report["importance"])[:10]
    print("  top features (gain):", ", ".join(top))
    print(f"  wrote {MODEL_PATH}\n        {CALIB_PATH}\n        {REPORT_PATH}  ({report['seconds']} s)")


def log_experiment(report: dict, change: str) -> None:
    s = report["summary"]["model/all"]
    row = {"date": dt.date.today().isoformat(), "owner": "Bianca",
           "change": f"p_home: {change} (ORACLE-D, not full pipeline)",
           "spellF1_mean": f"{s['spellF1']:.4f}", "macroF1_mean": f"{s['macroF1']:.4f}",
           "games": len(EXTERNAL.SUITE_GAMES), "noise_levels": "/".join(EXTERNAL.LEVELS),
           "notes": f"LOMO, acc {s['acc']:.4f}, logloss {s['logloss']:.4f}, ece {s['ece']:.4f}"}
    with open(config.EXPERIMENTS_CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=list(row)).writerow(row)
    print(f"  logged in {config.EXPERIMENTS_CSV.name}")


def predict(name: str, v0: bool = False) -> dict:
    game = data.load(name, mmap=False)
    if v0:
        p = finish(v0_p_home(game), game)
        how = f"v0 rule (last control, conf {POSSESSION.V0_CONF})"
    else:
        if name in {g for g, _ in suite_games()}:
            raise SystemExit(f"{name} is a training game: its out-of-fold p_home comes from --train")
        if not MODEL_PATH.is_file():
            print("no trained model yet: training it first (deterministic)")
            train()
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(MODEL_PATH))
        X, _ = features.compute(game)
        Xm = features.compute(features.mirror(game))[0] if POSSESSION.TTA else None
        calib = json.loads(CALIB_PATH.read_text()) if POSSESSION.CALIBRATE else None
        p = finish(apply_isotonic(predict_raw(booster, X, Xm), calib), game)
        how = "LightGBM " + hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
    sha = save_npy(config.game_cache(name) / "p_home.npy", p)
    inside = np.isfinite(p)
    return {"how": how, "graded": int(inside.sum()), "share_h": float((p[inside] > 0.5).mean()),
            "mean": float(p[inside].mean()), "sha256": sha, "path": str(config.game_cache(name) / "p_home.npy")}


def main() -> None:
    ap = argparse.ArgumentParser(description="p_home = P(H in possession | ball alive) (P4).")
    ap.add_argument("--game", default=None, help="write p_home.npy for this game")
    ap.add_argument("--train", action="store_true", help="LOMO on the suite + final model")
    ap.add_argument("--v0", action="store_true", help="rule only (last controlling team)")
    ap.add_argument("--ablate", action="store_true", help="LOMO with each feature group removed")
    ap.add_argument("--log", metavar="CHANGE", help="with --train: append a row to experiments.csv")
    args = ap.parse_args()
    config.set_seeds()
    if args.ablate:
        ablate()
        return
    if args.train:
        train(args.log)
    if args.game or not args.train:
        info = predict(args.game or config.GAME, v0=args.v0)
        print(f"p_home for {args.game or config.GAME}: {info['how']}")
        print(f"  graded frames {info['graded']}, p > 0.5 on {info['share_h']:.1%}, mean {info['mean']:.3f}")
        print(f"  wrote {info['path']}\n  sha256 {info['sha256']}")


if __name__ == "__main__":
    main()
