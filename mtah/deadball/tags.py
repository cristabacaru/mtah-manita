"""Smart Tagging -> tags.json: every tagged phase and set piece on video frames, team as H/A (P3b).

    python -m mtah.deadball.tags --game BAR-ATH

Output: $MTAH_CACHE/<game>/tags.json, a list sorted by (start_frame, end_frame, code), one entry per
Smart Tagging instance, none dropped:

    {"start_frame": int,          first frame of the phase
     "end_frame": int,            first frame AFTER it (half-open, as starter/load_smart_tagging.py)
     "code": str,                 e.g. "SET PIECES", "BUILD UP" (upper case, stripped)
     "type": str | null,          the Type tag, set pieces only, e.g. "Throw-in"
     "team": "H" | "A" | null,    the raw Team tag as a side; null for "N/A"
     "possession_team": "H" | "A" | null,   team in possession: team, flipped for the codes in
                                  config.RESTARTS.TAG_DEFENDING_CODES (their Team is the defender)
     "half": int | null}          1-based half the phase overlaps most; null if it overlaps none

Facts behind it (docs/deadball_SPEC.md section 6, decisions 2026-11-07):
- the XML clock is seconds from frame 0 of the tactical video (starter/load_smart_tagging.py:16-19);
- club names are the same as in the tracking XML, so H/A comes from mtah.data meta, not a constant;
- Smart Tagging exists for the target game only. Other games have none, so this stage refuses them.

Only the decoder (P5) reads this file. mask() turns a selection of tags into a per-frame boolean,
the form mtah/decode.py takes (set_piece_mask).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from mtah import config
from mtah.config import RESTARTS

FILENAME = "tags.json"
NO_TEAM = ("N/A", "", None)


def _official_events(game: str) -> list[dict]:
    """The official loader's output: [{code, start_frame, end_frame, start_s, end_s, tags}]."""
    starter = str(config.REPO_ROOT / "starter")
    if starter not in sys.path:
        sys.path.insert(0, starter)
    from load_smart_tagging import load   # official, unedited

    return load(game=game)


def _side(team_name, home: str, away: str):
    if team_name in NO_TEAM:
        return None
    if team_name == home:
        return "H"
    if team_name == away:
        return "A"
    raise ValueError(f"Team tag {team_name!r} is neither {home!r} (H) nor {away!r} (A)")


def _half(start: int, end: int, halves) -> int | None:
    """1-based index of the half with the largest overlap with [start, end); None if no overlap."""
    best, best_overlap = None, 0
    for i, (first, last) in enumerate(halves, start=1):
        overlap = min(end, last + 1) - max(start, first)
        if overlap > best_overlap:
            best, best_overlap = i, overlap
    return best


def convert(events: Iterable[Mapping], meta: Mapping) -> list[dict]:
    """Official loader events -> tags.json entries (see module docstring), sorted."""
    home, away = meta["home_name"], meta["away_name"]
    n = int(meta["n_frames"])
    defending = {c.strip().upper() for c in RESTARTS.TAG_DEFENDING_CODES}
    flip = {"H": "A", "A": "H", None: None}
    out = []
    for e in events:
        start, end = int(e["start_frame"]), int(e["end_frame"])
        if not 0 <= start < end <= n:
            raise ValueError(f"tag frames [{start}, {end}) not inside the video [0, {n})")
        code = (e["code"] or "").strip().upper()
        team = _side(e["tags"].get("Team"), home, away)
        out.append({
            "start_frame": start,
            "end_frame": end,
            "code": code,
            "type": e["tags"].get("Type") or None,
            "team": team,
            "possession_team": flip[team] if code in defending else team,
            "half": _half(start, end, meta["halves"]),
        })
    out.sort(key=lambda t: (t["start_frame"], t["end_frame"], t["code"]))
    return out


def save(tags: list[dict], path: Path) -> str:
    """Write tags.json atomically and deterministically; return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(tags, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load(game: str = config.GAME) -> list[dict]:
    """tags.json of one game, as written by this stage."""
    path = config.game_cache(game) / FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"no {FILENAME} for {game!r}: python -m mtah.deadball.tags --game {game}")
    return json.loads(path.read_text(encoding="utf-8"))


def mask(tags: Iterable[Mapping], n_frames: int, code: str | None = None, type: str | None = None,
         team: str | None = None) -> np.ndarray:
    """(n_frames,) bool, True on frames covered by the tags that match every given filter.

    Example, every set piece:  mask(load(), n, code="SET PIECES")
    """
    m = np.zeros(int(n_frames), dtype=bool)
    for t in tags:
        if code is not None and t["code"] != code.strip().upper():
            continue
        if type is not None and t["type"] != type:
            continue
        if team is not None and t["possession_team"] != team:
            continue
        m[t["start_frame"]:t["end_frame"]] = True
    return m


def summary(tags: list[dict]) -> str:
    """Aggregates only: counts by code and side, set-piece types, tags outside the halves."""
    lines = [f"{len(tags)} tags"]
    by_code = Counter(t["code"] for t in tags)
    for code, count in sorted(by_code.items(), key=lambda kv: -kv[1]):
        sides = Counter(t["possession_team"] for t in tags if t["code"] == code)
        flipped = " (team flipped)" if code in RESTARTS.TAG_DEFENDING_CODES else ""
        lines.append(f"  {code:<30} {count:>4}   possession H {sides['H']:>3}  A {sides['A']:>3}  "
                     f"none {sides[None]:>3}{flipped}")
    types = Counter(t["type"] for t in tags if t["type"])
    lines.append("set-piece types: " + ", ".join(f"{k} {v}" for k, v in sorted(types.items())))
    lines.append(f"no team (N/A): {sum(t['team'] is None for t in tags)}   "
                 f"outside every half: {sum(t['half'] is None for t in tags)}")
    return "\n".join(lines)


def run(game: str) -> dict:
    if game != config.GAME:
        raise SystemExit(f"Smart Tagging exists for {config.GAME} only, not {game!r} "
                         "(external games have none: README module contract).")
    from mtah.data import load as load_game

    meta = load_game(game)["meta"]
    tags = convert(_official_events(game), meta)
    path = config.game_cache(game) / FILENAME
    return {"tags": tags, "path": str(path), "sha256": save(tags, path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write tags.json (Smart Tagging on video frames).")
    parser.add_argument("--game", default=config.GAME)
    info = run(parser.parse_args().game)
    print(summary(info["tags"]))
    print(f"wrote {info['path']}\nsha256 {info['sha256']}")


if __name__ == "__main__":
    main()
