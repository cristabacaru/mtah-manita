"""Restart detection from tracking -> anchors.json (P3b).

    python -m mtah.deadball.restarts --game BAR-ATH

A restart is the moment the ball is played again after a stoppage: kick-off, throw-in, corner, goal
kick, free kick. That is where the label switches from D back to H or A, so the decoder (P5) can use
it to place the end of a dead stretch. Decisions 2026-11-07:
  R1  frame = the moment the ball is played (first frame faster than RESTARTS.BALL_PLAYED_MPS)
  R2  tracking only, so it runs and can be scored on any game (no Smart Tagging, no p_dead)
  R3  restarts whose kind is unclear are kept, as type "unknown", with lower confidence
  R4  independent of p_dead

Output: $MTAH_CACHE/<game>/anchors.json, a list sorted by frame:

    {"frame": int,                 the moment the ball is played (video frame for BAR-ATH)
     "window": [lo, hi],           half-open, frame +/- RESTARTS.WINDOW_S, clipped to the half
     "type": "kickoff" | "throw_in" | "corner" | "goal_kick" | "unknown",
     "team": "H" | "A" | null,     the restarting team
     "confidence": float,          0.25-1.0, a heuristic score, NOT a calibrated probability
     "half": int,                  1-based; anchors exist only inside the halves
     "source": "gap_end" | "still_ball" | "kickoff_formation",
     "slow_before": bool,          players' mean speed < SLOW_MPS over SLOW_BEFORE_S before the candidate
     "ball_played": bool}          the ball passed BALL_PLAYED_MPS within SEARCH_S

How (all per half, guide section 5 feature conventions; H attacks towards x = L):
1. Candidates, three independent templates (research/checks/c5_restarts.py):
   - gap_end: the ball reappears after being missing >= GAP_MIN_S;
   - still_ball: the ball starts moving after being still >= STILL_MIN_S, the players were slow
     before (slow_before) and the ball is at a restart spot (type not "unknown");
   - kickoff_formation: both teams in their own half, ball at the centre spot, players slow, held
     >= KICKOFF_HOLD_S, then movement.
2. Restart frame: the smoothed ball speed first passes BALL_PLAYED_MPS within SEARCH_S; the frame is
   then the first nearby frame (+/- 0.2 s) from which the ball moves that fast to the next frame
   (_played_frame). A kick-off formation whose ball is played away from the centre mark is dropped.
3. Type from where the ball is at that frame (corner arc, touchline, goal area, centre spot).
4. Team: corners and goal kicks from the Laws (H attacks x = L, so a corner at the x = L end is H's and
   a goal kick from the x = 0 goal area is H's); otherwise the team of the player nearest the ball.
5. Confidence: recognised type 0.75 + 0.125 * (slow_before + ball_played), so 0.75-1.0; "unknown"
   0.25 + 0.125 * (slow_before + ball_played), so 0.25-0.5. Every recognised anchor ranks above every
   unknown one. Heuristic, not calibrated: filter with mask(..., min_confidence=...).
6. Candidates closer than MERGE_S are one restart; the highest confidence wins.

Why option B (decision 2026-11-08; evidence from research/checks/c9_restarts_vs_tags.py, BAR-ATH):
Smart Tagging is not a full answer key (59 tagged set pieces, a match has about 110 restarts), but how
often anchors fall within +/- 5 s of a tagged set piece, against 35 % for a random graded frame, sorts
the templates: recognised types 59-100 %, gap_end "unknown" 35 % (chance), still_ball "unknown" 30 %
(below chance; 160 candidates). So still_ball unknowns are dropped, gap_end unknowns are kept with low confidence
(R3: some may be free kicks, which are mostly untagged). Using BAR-ATH's automatic tags for this is
allowed (guide section 2, algorithmic pseudo-labels); the real check is on P2's external games.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from mtah import config
from mtah.config import RESTARTS as R
from mtah.deadball.util import check_game, halves, runs

FILENAME = "anchors.json"
TYPES = ("kickoff", "throw_in", "corner", "goal_kick", "unknown")
SOURCE_RANK = {"kickoff_formation": 0, "gap_end": 1, "still_ball": 2}   # tie-break when merging


# =================================================================================================
# geometry (any pitch; origin at a corner, H attacks towards x = L: guide section 5)
# =================================================================================================
def classify(x: float, y: float, pitch) -> str:
    """Restart type from the ball position. Order matters: a corner is also near a touchline."""
    L, W = pitch
    if not (np.isfinite(x) and np.isfinite(y)):
        return "unknown"
    cx, cy = min(x, L - x), min(y, W - y)          # distance to nearest goal line / touchline (< 0 past it)
    if math.hypot(max(cx, 0.0), max(cy, 0.0)) <= R.CORNER_R_M:
        return "corner"
    if cy <= R.THROW_IN_M:
        return "throw_in"
    if cx <= R.GOAL_AREA_DEPTH_M and abs(y - W / 2) <= R.GOAL_AREA_HALF_WIDTH_M:
        return "goal_kick"
    if math.hypot(x - L / 2, y - W / 2) <= R.CENTRE_R_M:
        return "kickoff"
    return "unknown"


def laws_team(kind: str, x: float, pitch) -> str | None:
    """Restarting team fixed by the Laws, given that H attacks towards x = L. None if not fixed."""
    L = pitch[0]
    if kind == "corner":          # taken by the attacking team at the end it attacks
        return "H" if x > L / 2 else "A"
    if kind == "goal_kick":       # taken by the defending team from its own goal area
        return "H" if x < L / 2 else "A"
    return None


def nearest_team(players_xy_t: np.ndarray, team: np.ndarray, ball_t: np.ndarray) -> str | None:
    """Team of the tracked player nearest the ball on one frame; None if nobody or no ball."""
    if not np.isfinite(ball_t).all():
        return None
    d = np.hypot(players_xy_t[:, 0] - ball_t[0], players_xy_t[:, 1] - ball_t[1])
    if not np.isfinite(d).any():
        return None
    return str(team[int(np.nanargmin(d))])


# =================================================================================================
# per-half signals
# =================================================================================================
def _team_mean_speed(speed: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN frames -> NaN, expected
        return np.nanmean(np.asarray(speed, dtype=np.float64), axis=1)


def _ball_speed(ball: np.ndarray, fps: float) -> np.ndarray:
    from mtah.data import position_speed
    return position_speed(ball[:, None, :], fps, max_speed=np.inf)[:, 0]   # balls are faster than people


def _gap_end_starts(visible: np.ndarray, fps: float) -> list[int]:
    min_len = math.ceil(R.GAP_MIN_S * fps - 1e-9)
    return [int(stop) for start, stop in zip(*runs(~visible))
            if stop - start >= min_len and stop < len(visible)]


def _still_ball_starts(ball: np.ndarray, fps: float) -> list[int]:
    lag = max(1, round(R.STILL_MIN_S * fps))
    still = np.zeros(len(ball), dtype=bool)
    if len(ball) > lag:
        with np.errstate(invalid="ignore"):
            drift = np.hypot(*(ball[lag:] - ball[:-lag]).T)
        still[lag:] = drift < R.STILL_DRIFT_M                    # NaN -> False
    return [int(stop) for _, stop in zip(*runs(still)) if stop < len(ball)]


def _kickoff_starts(ball, pxy, team, mean_speed, pitch, fps) -> list[int]:
    L, W = pitch
    x = pxy[..., 0]
    tracked = np.isfinite(x)
    with np.errstate(invalid="ignore"):
        own = np.where(team[None, :] == "H", x <= L / 2 + R.KICKOFF_TOL_M, x >= L / 2 - R.KICKOFF_TOL_M)
        violators = (tracked & ~own).sum(axis=1)
        ball_ok = ~np.isfinite(ball[:, 0]) | (np.hypot(ball[:, 0] - L / 2, ball[:, 1] - W / 2) <= R.CENTRE_R_M)
        cond = ((tracked.sum(axis=1) >= R.KICKOFF_MIN_PLAYERS) & (violators <= R.KICKOFF_MAX_VIOLATORS)
                & ball_ok & (mean_speed < R.SLOW_MPS))
    starts, stops = runs(cond)
    merged: list[list[int]] = []
    for a, b in zip(starts, stops):                              # join interruptions under 1 s (c5)
        if merged and a - merged[-1][1] < fps:
            merged[-1][1] = int(b)
        else:
            merged.append([int(a), int(b)])
    hold = R.KICKOFF_HOLD_S * fps
    return [b for a, b in merged if b - a >= hold and b < len(cond)]


# =================================================================================================
# detection
# =================================================================================================
def _played_frame(start, ball, bspeed, fps) -> tuple[int, bool]:
    """(frame the ball is played, found?) for a candidate starting at `start` (half-relative).

    The smoothed speed (+/- k frames) first passes BALL_PLAYED_MPS at some t, up to k frames before or
    after the kick. The kick is then the first frame in [t - k, t + k] from which the ball moves faster
    than BALL_PLAYED_MPS to the next frame. Frames where the ball is missing never qualify, so after a
    gap the frame is at the earliest the first frame the ball is seen again.
    """
    search = slice(start, min(len(ball), start + int(round(R.SEARCH_S * fps))))
    fast = np.flatnonzero(bspeed[search] > R.BALL_PLAYED_MPS)
    if not len(fast):
        return start, False
    t = start + int(fast[0])
    k = max(1, int(round(config.SPEED_HALF_WINDOW_S * fps)))
    lo, hi = max(0, t - k), min(len(ball) - 1, t + k)
    with np.errstate(invalid="ignore"):
        step = np.hypot(*(ball[lo + 1:hi + 1] - ball[lo:hi]).T) * fps    # step[i]: frame lo+i -> lo+i+1
    moving = np.flatnonzero(step > R.BALL_PLAYED_MPS)
    return (lo + int(moving[0]) if len(moving) else t), True


def _anchor(start, source, ball, bspeed, pxy, team, mean_speed, pitch, fps, first_frame) -> dict | None:
    """One candidate -> anchor fields; None if the candidate contradicts its own template."""
    frame, played = _played_frame(start, ball, bspeed, fps)
    bx, by = ball[frame]
    kind = classify(bx, by, pitch)
    if source == "kickoff_formation":
        if np.isfinite(bx) and kind != "kickoff":
            return None                  # Laws: a kick-off is taken from the centre mark
        kind = "kickoff"
    team_out = laws_team(kind, bx, pitch) or nearest_team(pxy[frame], team, ball[frame])

    before = mean_speed[max(0, start - int(round(R.SLOW_BEFORE_S * fps))):start]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        slow = bool(np.isfinite(before).any() and np.nanmean(before) < R.SLOW_MPS)
    # Option B (2026-11-08): a recognised type is the strongest evidence, so every recognised anchor
    # scores above every unknown one; slow_before and ball_played add within the band.
    base = R.CONF_UNKNOWN if kind == "unknown" else R.CONF_KNOWN
    confidence = base + R.CONF_STEP * (slow + played)
    return {"frame": first_frame + frame, "type": kind, "team": team_out,
            "confidence": round(float(confidence), 4), "source": source,
            "slow_before": slow, "ball_played": played}


def _merge(cands: list[dict], fps: float) -> list[dict]:
    gap = R.MERGE_S * fps
    groups: list[list[dict]] = []
    for c in sorted(cands, key=lambda c: c["frame"]):
        if groups and c["frame"] - groups[-1][0]["frame"] < gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    best = lambda c: (-c["confidence"], c["type"] == "unknown", SOURCE_RANK[c["source"]], c["frame"])
    return [min(g, key=best) for g in groups]


def detect(game: Mapping, filter_still: bool = True) -> list[dict]:
    """Anchors for one game in the shared schema (mtah.data.load), any game.

    filter_still=False keeps every still_ball candidate, for research/checks/c9 only.
    """
    check_game(game, need=("ball_xy", "players_xy", "players_speed"))
    meta = game["meta"]
    fps, pitch = float(meta["fps"]), tuple(float(v) for v in meta["pitch"])
    team = np.asarray(game["players_team"])
    w = int(round(R.WINDOW_S * fps))
    out = []
    for number, h in enumerate(halves(game), start=1):
        ball = np.asarray(game["ball_xy"][h], dtype=np.float64)
        pxy = np.asarray(game["players_xy"][h], dtype=np.float64)
        mean_speed = _team_mean_speed(game["players_speed"][h])
        bspeed = _ball_speed(ball, fps)
        visible = np.isfinite(ball).all(axis=1)
        args = (ball, bspeed, pxy, team, mean_speed, pitch, fps, h.start)
        # Option B (2026-11-08, research/checks/c9_restarts_vs_tags.py): a still ball counts only if
        # play had stopped (slow_before) AND it sits at a restart spot (type recognised). Otherwise it is
        # near a tagged set piece no more often than chance: a player holding the ball in live play.
        # gap_end unknowns are kept (R3): a >= GAP_MIN_S gap is itself a stoppage.
        still = [a for a in (_anchor(s, "still_ball", *args) for s in _still_ball_starts(ball, fps))
                 if a is not None and (not filter_still or (a["slow_before"] and a["type"] != "unknown"))]
        cands = ([_anchor(s, "gap_end", *args) for s in _gap_end_starts(visible, fps)]
                 + still
                 + [_anchor(s, "kickoff_formation", *args)
                    for s in _kickoff_starts(ball, pxy, team, mean_speed, pitch, fps)])
        cands = [a for a in cands if a is not None]
        for a in _merge(cands, fps):
            a["window"] = [max(h.start, a["frame"] - w), min(h.stop, a["frame"] + w + 1)]
            a["half"] = number
            out.append(a)
    return sorted(out, key=lambda a: a["frame"])


# =================================================================================================
# files and helpers for the decoder
# =================================================================================================
def save(anchors: list[dict], path: Path) -> str:
    """Write anchors.json atomically and deterministically; return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(anchors, indent=1, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(game: str = config.GAME) -> list[dict]:
    path = config.game_cache(game) / FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {FILENAME} for {game!r}: python -m mtah.deadball.restarts --game {game}")
    return json.loads(path.read_text(encoding="utf-8"))


def mask(anchors: Iterable[Mapping], n_frames: int, min_confidence: float = 0.0,
         type: str | None = None) -> np.ndarray:
    """(n_frames,) bool, True inside the window of every matching anchor (decode.py set_piece_mask)."""
    m = np.zeros(int(n_frames), dtype=bool)
    for a in anchors:
        if a["confidence"] >= min_confidence and (type is None or a["type"] == type):
            m[a["window"][0]:a["window"][1]] = True
    return m


def summary(anchors: list[dict]) -> str:
    """Aggregates only: counts by type, source, team and confidence."""
    lines = [f"{len(anchors)} anchors"]
    for kind in TYPES:
        sel = [a for a in anchors if a["type"] == kind]
        teams = Counter(a["team"] for a in sel)
        lines.append(f"  {kind:<10} {len(sel):>4}   H {teams['H']:>3}  A {teams['A']:>3}  none {teams[None]:>3}")
    lines.append("by source: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(a["source"] for a in anchors).items())))
    lines.append("confidence: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(a["confidence"] for a in anchors).items())))
    lines.append("by half: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(a["half"] for a in anchors).items())))
    return "\n".join(lines)


def run(game: str) -> dict:
    from mtah.data import load as load_game
    anchors = detect(load_game(game))
    path = config.game_cache(game) / FILENAME
    return {"anchors": anchors, "path": str(path), "sha256": save(anchors, path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write anchors.json (restarts detected from tracking).")
    parser.add_argument("--game", default=config.GAME)
    info = run(parser.parse_args().game)
    print(summary(info["anchors"]))
    print(f"wrote {info['path']}\nsha256 {info['sha256']}")


if __name__ == "__main__":
    main()
