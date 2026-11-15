"""Tune decoder parameters on the external validation suite, never the public GT.

    python -m mtah.optuna_tune --trials 50

The suite must have exported p_dead.npy and p_home.npy arrays in the shared cache for every
selected game/noise level. The objective is the official mean possession spell-F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    import optuna
except ImportError:  # Optuna is an optional dependency for inference, not for tuning.
    optuna = None

from mtah import config
from mtah.data import load as load_game
from mtah.decode import decode_probabilities

ROOT = config.REPO_ROOT
EVALUATION_DIR = ROOT / "evaluation"
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))
import possession_metrics as pm  # noqa: E402


def load_validation_examples(levels: Sequence[str] | None = None) -> list[dict]:
    """Load external-suite predictions inputs and their GT from the manifest."""
    manifest_path = config.EXTERNAL.SUITE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"external validation suite is missing: {manifest_path}; "
            "prepare it with the P2 suite workflow before tuning"
        )
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("games"), list):
        raise ValueError(f"{manifest_path} must contain a 'games' list")

    selected_levels = set(levels) if levels else None
    examples = []
    missing = []
    for row in manifest["games"]:
        if selected_levels is not None and row.get("level") not in selected_levels:
            continue
        name = str(row["name"])
        cache = config.game_cache(name)
        p_dead_path, p_home_path = cache / "p_dead.npy", cache / "p_home.npy"
        gt_path = (config.VALIDATION_DIR / row["gt"]).resolve()
        try:
            gt_path.relative_to(config.VALIDATION_DIR.resolve())
        except ValueError as exc:
            raise ValueError(f"validation GT path escapes validation directory: {gt_path}") from exc

        needed = [path for path in (cache / "meta.json", p_dead_path, p_home_path, gt_path)
                  if not path.is_file()]
        if needed:
            missing.extend(f"{name}: {path}" for path in needed)
            continue

        game = load_game(name)
        p_dead = np.load(p_dead_path, mmap_mode="r", allow_pickle=False)
        p_home = np.load(p_home_path, mmap_mode="r", allow_pickle=False)
        gt = pm.label_codes(pm.read_label_csv(gt_path))
        if p_dead.shape != (len(gt),) or p_home.shape != (len(gt),):
            raise ValueError(
                f"{name}: probability arrays and GT must all have shape ({len(gt)},)"
            )
        halves = [(int(a), int(b) + 1) for a, b in game["meta"]["halves"]]
        examples.append({
            "name": name,
            "p_dead": p_dead,
            "p_home": p_home,
            "halves": halves,
            "gt": gt,
            "anchors": _read_json_if_present(cache / "anchors.json"),
            "tags": _read_json_if_present(cache / "tags.json"),
            "fps": float(game["meta"]["fps"]),
        })

    if missing:
        details = "\n  ".join(missing[:20])
        more = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise FileNotFoundError(
            "external tuning inputs are incomplete; generate all missing p_dead/p_home outputs "
            f"and export data caches first:\n  {details}{more}"
        )
    if not examples:
        raise ValueError(f"no external validation games selected from {manifest_path}")
    return examples


def _read_json_if_present(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def _params_for_trial(trial, fps: float) -> dict:
    return {
        "emission_scale": trial.suggest_float("emission_scale", 0.02, 0.8, log=True),
        "switch_team": trial.suggest_float("switch_team", 1.0, 30.0, log=True),
        "switch_dead": trial.suggest_float("switch_dead", 1.0, 40.0, log=True),
        "min_live_frames": max(1, round(trial.suggest_float("min_live_s", 0.2, 1.5) * fps)),
        "min_dead_frames": max(1, round(trial.suggest_float("min_dead_s", 0.5, 4.0) * fps)),
        "anchor_bonus": trial.suggest_float("anchor_bonus", 0.0, 20.0),
        "tag_bonus": trial.suggest_float("tag_bonus", 0.0, 20.0),
    }


def make_objective(examples: Sequence[Mapping]):
    """Return a trial objective: mean official spellF1 over all supplied external games."""
    if not examples:
        raise ValueError("at least one external validation example is required")

    def objective(trial) -> float:
        fps = float(examples[0]["fps"])
        params = _params_for_trial(trial, fps)
        scores = []
        for example in examples:
            game_params = dict(params)
            game_params["min_live_frames"] = max(
                1, round(params["min_live_frames"] * float(example["fps"]) / fps)
            )
            game_params["min_dead_frames"] = max(
                1, round(params["min_dead_frames"] * float(example["fps"]) / fps)
            )
            pred = decode_probabilities(
                example["p_dead"],
                example["p_home"],
                half_blocks=example["halves"],
                anchors=example["anchors"],
                tags=example["tags"],
                **game_params,
            )
            pred_codes = pm.label_codes(pred)
            scores.append(
                pm.spell_scores(
                    example["gt"], pred_codes, pm.valid_runs(example["gt"])
                )["mean_f1"]
            )
        return float(np.mean(scores))

    return objective


def run_tuning(
    examples: Sequence[Mapping], trials: int = 50, seed: int = config.SEED
):
    if optuna is None:
        raise RuntimeError(
            "Optuna is required for tuning; install it in the active environment "
            "with `python -m pip install optuna`."
        )
    if trials < 1:
        raise ValueError("trials must be at least 1")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(make_objective(examples), n_trials=trials)
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--levels", nargs="+", choices=config.EXTERNAL.LEVELS)
    parser.add_argument(
        "--out",
        type=Path,
        default=config.REPORTS_DIR / "decoder_optuna_best.json",
        help="Where to save the best validation result (default: outside the repository cache)",
    )
    args = parser.parse_args()

    examples = load_validation_examples(args.levels)
    print(f"tuning on {len(examples)} external validation copies", flush=True)
    study = run_tuning(examples, trials=args.trials, seed=args.seed)
    result = {
        "objective": "mean official spell_f1_mean over selected external validation copies",
        "games": [example["name"] for example in examples],
        "trials": args.trials,
        "seed": args.seed,
        "best_spell_f1_mean": float(study.best_value),
        "best_params": study.best_params,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"best external spellF1 {study.best_value:.4f}")
    print(study.best_params)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
