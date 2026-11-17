"""Decode per-frame possession probabilities into H/A/D labels.

    python -m mtah.decode --game BAR-ATH

Inputs and output live in $MTAH_CACHE/<game>/. Probabilities are produced by P3/P4; this module
combines them with restart/tag context, decodes each half independently, and writes labels.npy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from mtah import config

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"], dtype="<U1")


def _transition_matrix(
    switch_team: float = config.DECODER.SWITCH_TEAM,
    switch_dead: float = config.DECODER.SWITCH_DEAD,
) -> np.ndarray:
    return np.array(
        [
            [0.0, -float(switch_team), -float(switch_dead)],
            [-float(switch_team), 0.0, -float(switch_dead)],
            [-float(switch_dead), -float(switch_dead), 0.0],
        ],
        dtype=np.float64,
    )


def _validate_logp(logp: np.ndarray) -> np.ndarray:
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != 3:
        raise ValueError("logp must have shape (n_frames, 3) in H, A, D column order")
    return logp


def viterbi(
    logp: np.ndarray,
    switch_team: float = config.DECODER.SWITCH_TEAM,
    switch_dead: float = config.DECODER.SWITCH_DEAD,
    transition_bonus: np.ndarray | None = None,
) -> np.ndarray:
    """Most-likely H/A/D path with fixed transition costs and optional per-frame bonuses."""
    logp = _validate_logp(logp)
    n_frames = len(logp)
    if n_frames == 0:
        return np.empty(0, dtype=np.int8)

    bonus = _validate_transition_bonus(transition_bonus, n_frames)
    trans = _transition_matrix(switch_team, switch_dead)
    back = np.empty((n_frames, 3), dtype=np.int8)
    score = logp[0].copy()

    for t in range(1, n_frames):
        cand = score[:, None] + trans + bonus[t]
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + logp[t]

    path = np.empty(n_frames, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n_frames - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _validate_transition_bonus(bonus: np.ndarray | None, n_frames: int) -> np.ndarray:
    if bonus is None:
        return np.zeros((n_frames, 3, 3), dtype=np.float64)
    bonus = np.asarray(bonus, dtype=np.float64)
    if bonus.shape != (n_frames, 3, 3):
        raise ValueError("transition_bonus must have shape (n_frames, 3, 3)")
    if not np.all(np.isfinite(bonus)):
        raise ValueError("transition_bonus must contain only finite values")
    return bonus


def viterbi_min_duration(
    logp: np.ndarray,
    min_len: Sequence[int] = (
        round(config.DECODER.MIN_LIVE_S * config.FPS),
        round(config.DECODER.MIN_LIVE_S * config.FPS),
        round(config.DECODER.MIN_DEAD_S * config.FPS),
    ),
    switch_team: float = config.DECODER.SWITCH_TEAM,
    switch_dead: float = config.DECODER.SWITCH_DEAD,
    cost: np.ndarray | None = None,
    transition_costs: np.ndarray | None = None,
    transition_bonus: np.ndarray | None = None,
) -> np.ndarray:
    """Viterbi with strict minimum state durations and optional time-varying transition bonuses.

    ``min_len`` gives minimum run lengths in frames for H, A, and D. The transition bonus, when
    supplied, is additive to the transition score at the destination frame and has shape
    ``(n_frames, 3, 3)`` (from-state, to-state).
    """
    logp = _validate_logp(logp)
    n_frames = len(logp)
    if n_frames == 0:
        return np.empty(0, dtype=np.int8)

    min_len = tuple(int(v) for v in min_len)
    if len(min_len) != 3 or any(v < 1 for v in min_len):
        raise ValueError("min_len must contain three positive frame counts for H, A, D")

    if cost is not None and transition_costs is None:
        transition_costs = cost
    trans = _transition_matrix(switch_team, switch_dead) if transition_costs is None else \
        np.asarray(transition_costs, dtype=np.float64)
    if trans.shape != (3, 3):
        raise ValueError("transition_costs must have shape (3, 3)")

    bonus = _validate_transition_bonus(transition_bonus, n_frames)
    offsets = np.cumsum((0,) + min_len[:-1]).astype(int)
    n_expanded = sum(min_len)
    pointer_dtype = np.int16 if n_expanded <= np.iinfo(np.int16).max else np.int32
    back = np.full((n_frames, n_expanded), -1, dtype=pointer_dtype)
    score = np.full(n_expanded, -np.inf, dtype=np.float64)
    state_for = np.empty(n_expanded, dtype=np.int8)

    for state, duration in enumerate(min_len):
        state_for[offsets[state]:offsets[state] + duration] = state
        score[offsets[state]] = logp[0, state]

    for t in range(1, n_frames):
        next_score = np.full(n_expanded, -np.inf, dtype=np.float64)
        frame_trans = trans + bonus[t]
        mature_indices = np.array(
            [offsets[s] + min_len[s] - 1 for s in range(3)], dtype=int
        )
        mature_scores = score[mature_indices]

        for state, duration in enumerate(min_len):
            start = offsets[state]
            mature = start + duration - 1

            if duration == 1:
                candidates = mature_scores + frame_trans[:, state]
                previous = int(np.argmax(candidates))
                next_score[mature] = candidates[previous] + logp[t, state]
                back[t, mature] = mature_indices[previous]
                continue

            # A new run can only begin from a different state's mature substate.
            candidates = mature_scores + frame_trans[:, state]
            candidates[state] = -np.inf
            previous = int(np.argmax(candidates))
            next_score[start] = candidates[previous] + logp[t, state]
            back[t, start] = mature_indices[previous]

            # Immature runs are forced to continue.
            for age in range(1, duration - 1):
                current = start + age
                prev = current - 1
                next_score[current] = score[prev] + frame_trans[state, state] + logp[t, state]
                back[t, current] = prev

            # Mature when reaching the minimum; thereafter stay mature until switching.
            mature_candidates = np.array(
                [score[mature], score[mature - 1]], dtype=np.float64
            ) + frame_trans[state, state]
            choice = int(np.argmax(mature_candidates))
            previous = mature if choice == 0 else mature - 1
            next_score[mature] = mature_candidates[choice] + logp[t, state]
            back[t, mature] = previous

        score = next_score

    current = int(np.argmax(score))
    path = np.empty(n_frames, dtype=np.int8)
    for t in range(n_frames - 1, -1, -1):
        path[t] = state_for[current]
        if t:
            current = int(back[t, current])
    return path


def viterbi_restart_bonus(
    logp: np.ndarray,
    bonus: float | Sequence[float] = 0.0,
    switch_team: float = config.DECODER.SWITCH_TEAM,
    switch_dead: float = config.DECODER.SWITCH_DEAD,
    set_piece_mask: Sequence[bool] | np.ndarray | None = None,
    restart_bonus: float | Sequence[float] | None = None,
) -> np.ndarray:
    """Viterbi with a positive bonus making D->H/A switches cheaper in restart windows."""
    logp = _validate_logp(logp)
    n_frames = len(logp)
    if not n_frames:
        return np.empty(0, dtype=np.int8)
    if restart_bonus is not None:
        bonus = restart_bonus
    values = np.asarray(bonus, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(n_frames, float(values))
    if values.shape != (n_frames,):
        raise ValueError("bonus must be a scalar or one value per frame")
    mask = np.zeros(n_frames, dtype=bool) if set_piece_mask is None else \
        np.asarray(set_piece_mask, dtype=bool)
    if mask.shape != (n_frames,):
        raise ValueError("set_piece_mask must have one boolean per frame")

    transition_bonus = np.zeros((n_frames, 3, 3), dtype=np.float64)
    transition_bonus[mask, D, H] = values[mask]
    transition_bonus[mask, D, A] = values[mask]
    return viterbi(logp, switch_team, switch_dead, transition_bonus)


def probabilities_to_logp(
    p_dead: np.ndarray, p_home: np.ndarray, emission_scale: float = 0.1,
    home_scale: float | None = None,
) -> np.ndarray:
    """Convert P(dead) and P(home | live) into scaled H/A/D log emissions.

    home_scale (p4-finetuning) weights the H-vs-A term log P(home | live) separately from the
    alive-vs-dead terms; None keeps the single emission_scale for both (previous behaviour).
    """
    p_dead = np.asarray(p_dead, dtype=np.float64)
    p_home = np.asarray(p_home, dtype=np.float64)
    if p_dead.ndim != 1 or p_home.shape != p_dead.shape:
        raise ValueError("p_dead and p_home must be aligned 1-D arrays")
    if not math.isfinite(emission_scale) or emission_scale <= 0:
        raise ValueError("emission_scale must be a positive finite number")
    if not (np.isfinite(p_dead).all() and np.isfinite(p_home).all()):
        raise ValueError("p_dead and p_home must be finite for the frames being decoded")
    if np.any((p_dead < 0) | (p_dead > 1)):
        raise ValueError("finite p_dead values must be in [0, 1]")
    if np.any((p_home < 0) | (p_home > 1)):
        raise ValueError("finite p_home values must be in [0, 1]")

    eps = 1e-8
    dead = np.clip(p_dead, eps, 1.0 - eps)
    home_live = np.clip(p_home, eps, 1.0 - eps)
    team_scale = emission_scale if home_scale is None else float(home_scale)
    if not math.isfinite(team_scale) or team_scale <= 0:
        raise ValueError("home_scale must be a positive finite number")
    logp = np.empty((len(dead), 3), dtype=np.float64)
    logp[:, H] = emission_scale * np.log1p(-dead) + team_scale * np.log(home_live)
    logp[:, A] = emission_scale * np.log1p(-dead) + team_scale * np.log1p(-home_live)
    logp[:, D] = emission_scale * np.log(dead)
    return logp


def restart_transition_bonus(
    n_frames: int,
    anchors: Iterable[Mapping] = (),
    tags: Iterable[Mapping] = (),
    anchor_bonus: float = 8.0,
    tag_bonus: float = 4.0,
) -> np.ndarray:
    """Build D->live transition bonuses from anchor windows and SET PIECES tag intervals."""
    if anchor_bonus < 0 or tag_bonus < 0:
        raise ValueError("restart bonuses must be non-negative")
    bonus = np.zeros((n_frames, 3, 3), dtype=np.float64)

    def add_window(start: int, stop: int, side: str | None, value: float) -> None:
        if not 0 <= start <= stop <= n_frames:
            raise ValueError(f"restart window [{start}, {stop}) is outside the frame array")
        if side == "H":
            bonus[start:stop, D, H] += value
        elif side == "A":
            bonus[start:stop, D, A] += value
        elif side is None:
            bonus[start:stop, D, H] += value
            bonus[start:stop, D, A] += value
        else:
            raise ValueError(f"restart team must be 'H', 'A', or null, got {side!r}")

    for anchor in anchors:
        window = anchor.get("window")
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError(f"invalid anchor window: {window!r}")
        confidence = float(anchor.get("confidence", 1.0))
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"anchor confidence must be in [0, 1], got {confidence}")
        add_window(int(window[0]), int(window[1]), anchor.get("team"), anchor_bonus * confidence)

    for tag in tags:
        if str(tag.get("code", "")).strip().upper() != "SET PIECES":
            continue
        add_window(
            int(tag["start_frame"]),
            int(tag["end_frame"]),
            tag.get("possession_team"),
            tag_bonus,
        )
    return bonus


def snap_restart_boundaries(
    labels: np.ndarray,
    anchors: Iterable[Mapping],
    half_blocks: Iterable[tuple[int, int]],
    min_confidence: float = config.DECODER.SNAP_MIN_CONFIDENCE,
) -> np.ndarray:
    """Snap D->H/A boundaries to nearby anchor frames, without reordering boundaries."""
    out = np.asarray(labels, dtype="<U1").copy()
    if out.ndim != 1:
        raise ValueError("labels must be a 1-D array")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    anchors = list(anchors)
    used: set[int] = set()

    for start, stop in half_blocks:
        edges = np.flatnonzero(out[start + 1:stop] != out[start:stop - 1]) + start + 1
        for i, boundary in enumerate(edges):
            if out[boundary - 1] != "D" or out[boundary] not in ("H", "A"):
                continue
            lower = int(edges[i - 1] + 1) if i else start
            upper = int(edges[i + 1] - 1) if i + 1 < len(edges) else stop - 1
            candidates = []
            for index, anchor in enumerate(anchors):
                if index in used or float(anchor.get("confidence", 0.0)) < min_confidence:
                    continue
                window = anchor.get("window")
                if not isinstance(window, (list, tuple)) or len(window) != 2:
                    continue
                frame = int(anchor.get("frame", -1))
                if not (int(window[0]) <= boundary < int(window[1])):
                    continue
                if not lower <= frame <= upper or not start <= frame < stop:
                    continue
                side = anchor.get("team")
                if side is not None and side != out[boundary]:
                    continue
                candidates.append((abs(frame - boundary), -float(anchor["confidence"]), frame, index))
            if not candidates:
                continue

            _, _, snapped, index = min(candidates)
            used.add(index)
            if snapped < boundary:
                out[snapped:boundary] = out[boundary]
            elif snapped > boundary:
                out[boundary:snapped] = "D"
    return out


def merge_short_live_spells(
    labels: np.ndarray,
    half_blocks: Iterable[tuple[int, int]],
    min_frames: int = round(config.DECODER.SHORT_LIVE_SPELL_S * config.FPS),
) -> np.ndarray:
    """Merge a short H/A flicker only when the same live team is on both sides."""
    if min_frames < 1:
        raise ValueError("min_frames must be positive")
    out = np.asarray(labels, dtype="<U1").copy()
    if out.ndim != 1:
        raise ValueError("labels must be a 1-D array")

    for start, stop in half_blocks:
        changed = True
        while changed:
            changed = False
            local = out[start:stop]
            edges = np.flatnonzero(local[1:] != local[:-1]) + 1
            starts = np.r_[0, edges]
            stops = np.r_[edges, len(local)]
            for i in range(1, len(starts) - 1):
                current = local[starts[i]]
                if (current in ("H", "A") and stops[i] - starts[i] < min_frames
                        and local[starts[i - 1]] == local[stops[i]] and current != local[stops[i]]
                        and local[stops[i]] in ("H", "A")):
                    out[start + starts[i]:start + stops[i]] = local[starts[i - 1]]
                    changed = True
                    break
    return out


def decode_probabilities(
    p_dead: np.ndarray,
    p_home: np.ndarray,
    half_blocks: Iterable[tuple[int, int]] | None = None,
    anchors: Iterable[Mapping] = (),
    tags: Iterable[Mapping] = (),
    emission_scale: float = config.DECODER.EMISSION_SCALE,
    switch_team: float = config.DECODER.SWITCH_TEAM,
    switch_dead: float = config.DECODER.SWITCH_DEAD,
    min_live_frames: int = round(config.DECODER.MIN_LIVE_S * config.FPS),
    min_dead_frames: int = round(config.DECODER.MIN_DEAD_S * config.FPS),
    anchor_bonus: float = config.DECODER.ANCHOR_BONUS,
    tag_bonus: float = config.DECODER.TAG_BONUS,
    snap_min_confidence: float = config.DECODER.SNAP_MIN_CONFIDENCE,
    short_live_spell_frames: int = round(config.DECODER.SHORT_LIVE_SPELL_S * config.FPS),
    home_scale: float | None = config.DECODER.HOME_SCALE,
    p_turn: np.ndarray | None = None,
    turn_bonus: float = config.DECODER.TURN_BONUS,
) -> np.ndarray:
    """Decode probability arrays into H/A/D/X labels, resetting state at every half boundary.

    p4-finetuning: p_turn (P(direct H <-> A change near t), from mtah.model) makes an H <-> A switch
    cheaper by turn_bonus * p_turn[t]; with p_turn None or turn_bonus 0 nothing changes.
    """
    p_dead = np.asarray(p_dead, dtype=np.float64)
    p_home = np.asarray(p_home, dtype=np.float64)
    if p_dead.ndim != 1 or p_home.shape != p_dead.shape:
        raise ValueError("p_dead and p_home must be aligned 1-D arrays")
    n_frames = len(p_dead)
    if half_blocks is None:
        half_blocks = [(a, b + 1) for _, a, b in config.HALF_BLOCKS]
    half_blocks = [(int(a), int(b)) for a, b in half_blocks]
    for start, stop in half_blocks:
        if not 0 <= start < stop <= n_frames:
            raise ValueError(f"invalid half block [{start}, {stop}) for {n_frames} frames")

    label_array = np.full(n_frames, "X", dtype="<U1")
    bonus_all = restart_transition_bonus(
        n_frames, anchors=anchors, tags=tags,
        anchor_bonus=anchor_bonus, tag_bonus=tag_bonus,
    )

    if p_turn is not None and turn_bonus:
        if turn_bonus < 0:
            raise ValueError("turn_bonus must be non-negative")
        turn = np.nan_to_num(np.asarray(p_turn, dtype=np.float64), nan=0.0)
        if turn.shape != p_dead.shape:
            raise ValueError("p_turn must be aligned with p_dead")
        bonus_all[:, H, A] += turn_bonus * turn
        bonus_all[:, A, H] += turn_bonus * turn

    for start, stop in half_blocks:
        dead = p_dead[start:stop]
        home = p_home[start:stop]
        if not (np.isfinite(dead).all() and np.isfinite(home).all()):
            raise ValueError(f"p_dead/p_home has missing values inside half [{start}, {stop})")
        logp = probabilities_to_logp(dead, home, emission_scale=emission_scale, home_scale=home_scale)
        path = viterbi_min_duration(
            logp,
            min_len=(min_live_frames, min_live_frames, min_dead_frames),
            switch_team=switch_team,
            switch_dead=switch_dead,
            transition_bonus=bonus_all[start:stop],
        )
        label_array[start:stop] = LETTERS[path]

    label_array = snap_restart_boundaries(
        label_array, anchors, half_blocks, min_confidence=snap_min_confidence
    )
    return merge_short_live_spells(label_array, half_blocks, min_frames=short_live_spell_frames)


def decode(
    logp: np.ndarray,
    half_blocks: Iterable[tuple[int, int, int]] | None = None,
    decoder=viterbi,
    **kwargs,
) -> np.ndarray:
    """Apply a decoder independently to inclusive (half, first, last) blocks, leaving X elsewhere."""
    logp = _validate_logp(logp)
    if half_blocks is None:
        half_blocks = config.HALF_BLOCKS
    labels = np.full(len(logp), "X", dtype="<U1")
    for _, first, last in half_blocks:
        if not 0 <= first <= last < len(logp):
            raise ValueError(f"invalid half block [{first}, {last}]")
        labels[first:last + 1] = LETTERS[decoder(logp[first:last + 1], **kwargs)]
    return labels


def _read_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    return value


def run(game: str = config.GAME, **overrides) -> dict:
    """Load teammate outputs, decode them, and write labels.npy to the shared game cache."""
    from mtah.data import load as load_game

    folder = config.game_cache(game)
    game_data = load_game(game)
    meta = game_data["meta"]
    p_dead_path, p_home_path = folder / "p_dead.npy", folder / "p_home.npy"
    missing = [path for path in (p_dead_path, p_home_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("decoder inputs missing: " + ", ".join(map(str, missing)))

    p_dead = np.load(p_dead_path, mmap_mode="r", allow_pickle=False)
    p_home = np.load(p_home_path, mmap_mode="r", allow_pickle=False)
    tags_path, anchors_path = folder / "tags.json", folder / "anchors.json"
    if game == config.GAME and (not tags_path.is_file() or not anchors_path.is_file()):
        raise FileNotFoundError(
            f"target-game decoder requires both {tags_path.name} and {anchors_path.name}"
        )
    tags = _read_json(tags_path)
    anchors = _read_json(anchors_path)
    turn_path = folder / "p_turn.npy"                     # p4-finetuning, optional
    if turn_path.is_file():
        overrides.setdefault("p_turn", np.load(turn_path, allow_pickle=False))
    half_blocks = [(int(a), int(b) + 1) for a, b in meta["halves"]]
    labels = decode_probabilities(
        p_dead,
        p_home,
        half_blocks=half_blocks,
        anchors=anchors,
        tags=tags,
        **overrides,
    )

    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "labels.npy"
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, labels, allow_pickle=False)
    os.replace(tmp, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    counts = {label: int(np.count_nonzero(labels == label)) for label in ("H", "A", "D", "X")}
    return {"labels": labels, "path": out, "sha256": digest, "counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode p_dead and p_home into match labels.")
    parser.add_argument("--game", default=config.GAME)
    parser.add_argument("--emission-scale", type=float, default=config.DECODER.EMISSION_SCALE)
    parser.add_argument("--switch-team", type=float, default=config.DECODER.SWITCH_TEAM)
    parser.add_argument("--switch-dead", type=float, default=config.DECODER.SWITCH_DEAD)
    parser.add_argument(
        "--min-live-frames", type=int,
        default=round(config.DECODER.MIN_LIVE_S * config.FPS),
    )
    parser.add_argument(
        "--min-dead-frames", type=int,
        default=round(config.DECODER.MIN_DEAD_S * config.FPS),
    )
    parser.add_argument("--anchor-bonus", type=float, default=config.DECODER.ANCHOR_BONUS)
    parser.add_argument("--tag-bonus", type=float, default=config.DECODER.TAG_BONUS)
    args = parser.parse_args()
    info = run(
        args.game,
        emission_scale=args.emission_scale,
        switch_team=args.switch_team,
        switch_dead=args.switch_dead,
        min_live_frames=args.min_live_frames,
        min_dead_frames=args.min_dead_frames,
        anchor_bonus=args.anchor_bonus,
        tag_bonus=args.tag_bonus,
    )
    print(f"labels for {args.game}: {info['counts']}")
    print(f"wrote {info['path']}\nsha256 {info['sha256']}")


if __name__ == "__main__":
    main()
