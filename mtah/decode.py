"""Viterbi-style sequence decoders for possession labelling.

This module keeps the core decoding logic separate from the feature-generation code and
embeds the expected plan for future HSMM and restart-aware upgrades.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:  # Optional integration with the challenge helper data.
    from games import GAME, half_video_blocks  # type: ignore
except Exception:  # pragma: no cover - import is optional for synthetic usage.
    GAME = None
    half_video_blocks = None  # type: ignore

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"], dtype="<U1")


def _transition_matrix(switch_team: float = 12.0, switch_dead: float = 10.0) -> np.ndarray:
    """Static transition cost matrix used by the baseline Viterbi decoder."""
    return np.array(
        [
            [0.0, -float(switch_team), -float(switch_dead)],
            [-float(switch_team), 0.0, -float(switch_dead)],
            [-float(switch_dead), -float(switch_dead), 0.0],
        ],
        dtype=np.float64,
    )


def viterbi(logp: np.ndarray, switch_team: float = 12.0, switch_dead: float = 10.0) -> np.ndarray:
    """Most-likely state path under a fixed 3-state HMM.

    Parameters
    ----------
    logp:
        Array with shape (n_frames, 3) and columns corresponding to H, A, D.
    switch_team:
        Penalty for moving between H and A.
    switch_dead:
        Penalty for entering or leaving D.
    """
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != 3:
        raise ValueError("logp must have shape (n_frames, 3)")
    if logp.shape[0] == 0:
        return np.empty(0, dtype=np.int8)

    trans = _transition_matrix(switch_team=switch_team, switch_dead=switch_dead)
    back = np.empty((logp.shape[0], 3), dtype=np.int8)
    score = logp[0].copy()

    for t in range(1, logp.shape[0]):
        cand = score[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]

    path = np.empty(logp.shape[0], dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(logp.shape[0] - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi_min_duration(
    logp: np.ndarray,
    min_len: Sequence[int] = (12, 12, 50),
    switch_team: float = 12.0,
    switch_dead: float = 10.0,
    cost: np.ndarray | None = None,
    transition_costs: np.ndarray | None = None,
) -> np.ndarray:
    """Viterbi with explicit minimum frame duration per state.

    A state can only transition away after it has persisted for at least ``min_len[state]``
    frames. This is a compact encoding of a semi-Markov / HSMM penalty without a full
    expanded-state model for every possible duration.

    The implementation expands each state into a small chain of sub-states, where the number
    of allowed same-state continuations is exactly the minimum duration. Once the minimum is
    satisfied, the chain may either keep the same state or switch to another one.
    """
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != 3:
        raise ValueError("logp must have shape (n_frames, 3)")
    if logp.shape[0] == 0:
        return np.empty(0, dtype=np.int8)

    min_len = tuple(int(v) for v in min_len)
    if len(min_len) != 3:
        raise ValueError("min_len must be a 3-tuple for (H, A, D)")
    if any(v < 1 for v in min_len):
        raise ValueError("minimum duration for each state must be >= 1 frame")

    if cost is not None and transition_costs is None:
        transition_costs = cost
    if transition_costs is None:
        transition_costs = _transition_matrix(switch_team=switch_team, switch_dead=switch_dead)
    else:
        transition_costs = np.asarray(transition_costs, dtype=np.float64)
        if transition_costs.shape != (3, 3):
            raise ValueError("transition_costs must have shape (3, 3)")

    expanded = [(state, age) for state in range(3) for age in range(min_len[state])]
    exp_index = {key: idx for idx, key in enumerate(expanded)}
    label_for = [state for state, _ in expanded]
    n_states = len(expanded)

    trans = np.full((n_states, n_states), -np.inf, dtype=np.float64)
    for i, (state, age) in enumerate(expanded):
        limit = min_len[state] - 1

        if age < limit:
            target = exp_index[(state, age + 1)]
            trans[i, target] = transition_costs[state, state]
        else:
            trans[i, i] = transition_costs[state, state]
            for target_state in range(3):
                if target_state == state:
                    continue
                target = exp_index[(target_state, 0)]
                trans[i, target] = transition_costs[state, target_state]

    score = np.full((logp.shape[0], n_states), -np.inf, dtype=np.float64)
    back = np.full((logp.shape[0], n_states), -1, dtype=np.int32)

    for state in range(3):
        score[0, exp_index[(state, 0)]] = logp[0, state]

    for t in range(1, logp.shape[0]):
        for j in range(n_states):
            target_state = label_for[j]
            cand = score[t - 1] + trans[:, j] + logp[t, target_state]
            best_prev = int(np.argmax(cand))
            score[t, j] = cand[best_prev]
            back[t, j] = best_prev

    final_state = int(np.argmax(score[-1]))
    path = np.empty(logp.shape[0], dtype=np.int8)
    current = final_state
    for t in range(logp.shape[0] - 1, -1, -1):
        path[t] = label_for[current]
        if t == 0:
            break
        current = back[t, current]
    return path


def viterbi_restart_bonus(
    logp: np.ndarray,
    bonus: float | Sequence[float] = 0.0,
    switch_team: float = 12.0,
    switch_dead: float = 10.0,
    set_piece_mask: Sequence[bool] | np.ndarray | None = None,
    restart_bonus: float | Sequence[float] | None = None,
) -> np.ndarray:
    """Viterbi with frame-local dead->live restart bonuses.

    When a restart / set-piece window is active on a given frame, switching from D into H or A
    is made cheaper by adding ``bonus`` to the transition cost. This keeps the state model
    compatible with the usual Viterbi interface while allowing time-varying costs.
    """
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != 3:
        raise ValueError("logp must have shape (n_frames, 3)")
    if logp.shape[0] == 0:
        return np.empty(0, dtype=np.int8)

    n_frames = logp.shape[0]
    if set_piece_mask is None:
        set_piece_mask = np.zeros(n_frames, dtype=bool)
    set_piece_mask = np.asarray(set_piece_mask, dtype=bool)
    if set_piece_mask.shape[0] != n_frames:
        raise ValueError("set_piece_mask must have one boolean per frame")

    if restart_bonus is not None:
        bonus = restart_bonus
    bonus_arr = np.asarray(bonus, dtype=np.float64)
    if bonus_arr.ndim == 0:
        bonus_arr = np.full(n_frames, float(bonus_arr), dtype=np.float64)
    elif bonus_arr.shape[0] != n_frames:
        raise ValueError("bonus must be a scalar or one value per frame")

    trans = _transition_matrix(switch_team=switch_team, switch_dead=switch_dead)
    back = np.empty((n_frames, 3), dtype=np.int8)
    score = logp[0].copy()

    for t in range(1, n_frames):
        current = trans.copy()
        if set_piece_mask[t]:
            current[D, H] += bonus_arr[t]
            current[D, A] += bonus_arr[t]
        cand = score[:, None] + current
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(
    logp: np.ndarray,
    half_blocks: Iterable[tuple[int, int, int]] | None = None,
    decoder=viterbi,
    **kwargs,
) -> np.ndarray:
    """Apply a decoder independently to each half block, leaving out-of-range frames as X."""
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != 3:
        raise ValueError("logp must have shape (n_frames, 3)")

    if half_blocks is None:
        if half_video_blocks is None:
            raise ValueError("half_blocks must be provided when the games helper is unavailable")
        half_blocks = half_video_blocks(GAME)

    labels = np.full(logp.shape[0], "X", dtype="<U1")
    for _, first, last in half_blocks:
        labels[first:last + 1] = LETTERS[decoder(logp[first:last + 1], **kwargs)]
    return labels
