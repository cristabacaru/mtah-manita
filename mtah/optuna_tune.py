"""Optuna tuning scaffold for the possession decoder.

This is intentionally lightweight: it defines the search space and calls the decoder under a
plausible objective, while leaving the exact external validation dataset to be plugged in by the
team that owns the data layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import optuna  # type: ignore
except ImportError: 
    optuna = None

from mtah.decode import viterbi_min_duration, viterbi_restart_bonus


def make_objective(logp: np.ndarray, set_piece_mask: np.ndarray | None = None, labels: np.ndarray | None = None):
    """Create the objective used by the tuning run.

    If external labels are not available, a proxy metric keeps the scaffold runnable and keeps
    the search anchored to football-like possession counts and durations.
    """
    if set_piece_mask is None:
        set_piece_mask = np.zeros(logp.shape[0], dtype=bool)
    if labels is not None and labels.shape[0] != logp.shape[0]:
        raise ValueError("labels must align with logp length")

    def objective(trial):
        if optuna is None:
            raise RuntimeError("optuna is required for the tuning scaffold; install it with `pip install optuna`.")

        emission_scale = trial.suggest_float("emission_scale", 0.02, 0.8)
        switch_team = trial.suggest_float("switch_team", 2.0, 30.0)
        switch_dead = trial.suggest_float("switch_dead", 2.0, 40.0)
        min_live = trial.suggest_int("min_live_frames", 4, 40)
        min_dead = trial.suggest_int("min_dead_frames", 10, 120)
        restart_bonus = trial.suggest_float("restart_bonus", 0.0, 20.0)

        scaled = logp * emission_scale
        decoded = viterbi_min_duration(
            scaled,
            min_len=(min_live, min_live, min_dead),
            switch_team=switch_team,
            switch_dead=switch_dead,
        )

        if labels is not None:
            score = float(np.mean(decoded == labels))
            return score

        # A lightweight proxy objective: prefer a realistic spell count and a stable mean spell
        # length, without looking at the public validation window.
        changes = np.flatnonzero(decoded[:-1] != decoded[1:]) + 1
        spell_count = int(changes.size + 1 if decoded.size else 0)
        if decoded.size and changes.size:
            durations = np.diff(np.r_[0, changes, decoded.size])
        else:
            durations = np.array([decoded.size], dtype=float) if decoded.size else np.array([0.0])
        target_count = 350.0
        target_mean = 10.0
        score = -abs(spell_count - target_count) / target_count
        score -= abs(np.mean(durations) / 25.0 - target_mean) / target_mean
        if set_piece_mask is not None and set_piece_mask.any():
            score += 0.1 * float(np.mean(restart_bonus * set_piece_mask)) / max(1.0, np.mean(set_piece_mask))
        return float(score)

    return objective


def run_tuning(logp: np.ndarray, set_piece_mask: np.ndarray | None = None, labels: np.ndarray | None = None, trials: int = 25):
    objective = make_objective(logp, set_piece_mask=set_piece_mask, labels=labels)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logp", type=str, default="", help="Optional .npy file with shape (n, 3) log probabilities.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.logp:
        logp = np.load(Path(args.logp), allow_pickle=False)
    else:
        logp = rng.normal(size=(3000, 3))
    set_piece_mask = np.zeros(logp.shape[0], dtype=bool)
    set_piece_mask[200:250] = True
    study = run_tuning(logp, set_piece_mask=set_piece_mask, trials=args.trials)
    print(study.best_trial.value)
    print(study.best_params)
