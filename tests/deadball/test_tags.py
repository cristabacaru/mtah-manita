"""tags.py: Smart Tagging -> tags.json (synthetic events, no match data needed).

    python tests/deadball/test_tags.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from harness import run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.config import RESTARTS
from mtah.deadball import tags as T

META = {"home_name": "Home FC", "away_name": "Away FC", "n_frames": 1000,
        "halves": [[100, 449], [550, 899]]}


def ev(code, start, end, team="Home FC", **extra):
    return {"code": code, "start_frame": start, "end_frame": end, "start_s": start / 25,
            "end_s": end / 25, "tags": {"Team": team, **extra}}


def test_team_maps_to_side_and_na_to_null():
    out = T.convert([ev("BUILD UP", 200, 250), ev("BUILD UP", 300, 350, team="Away FC"),
                     ev("SET PIECES", 400, 420, team="N/A", Type="Throw-in")], META)
    assert [t["team"] for t in out] == ["H", "A", None]
    assert [t["possession_team"] for t in out] == ["H", "A", None]
    assert out[2]["type"] == "Throw-in" and out[0]["type"] is None


def test_defending_codes_flip_possession_team_only():
    for code in RESTARTS.TAG_DEFENDING_CODES:
        t = T.convert([ev(code, 200, 250, team="Home FC")], META)[0]
        assert t["team"] == "H" and t["possession_team"] == "A", code
    t = T.convert([ev("DEFENSIVE TRANSITION", 200, 250, team="N/A")], META)[0]
    assert t["team"] is None and t["possession_team"] is None


def test_code_is_normalised():
    t = T.convert([ev("  defending in middle third ", 200, 250)], META)[0]
    assert t["code"] == "DEFENDING IN MIDDLE THIRD" and t["possession_team"] == "A"


def test_half_is_largest_overlap_or_null():
    out = T.convert([ev("A", 120, 130), ev("A", 600, 610), ev("A", 460, 540),   # between halves
                     ev("A", 440, 560),                                           # 10 in h1, 10 in h2
                     ev("A", 430, 600)], META)                                    # 20 in h1, 50 in h2
    got = {(t["start_frame"], t["end_frame"]): t["half"] for t in out}
    assert got[(120, 130)] == 1 and got[(600, 610)] == 2 and got[(460, 540)] is None
    assert got[(440, 560)] == 1      # tie: the earlier half
    assert got[(430, 600)] == 2


def test_end_frame_is_exclusive_in_mask():
    tags = T.convert([ev("SET PIECES", 200, 203, Type="Corner Kick")], META)
    m = T.mask(tags, 1000, code="set pieces")
    assert m.sum() == 3 and m[200] and m[202] and not m[203] and not m[199]


def test_mask_filters_combine():
    tags = T.convert([ev("SET PIECES", 200, 210, Type="Throw-in"),
                      ev("SET PIECES", 300, 310, team="Away FC", Type="Corner Kick"),
                      ev("BUILD UP", 400, 410)], META)
    assert T.mask(tags, 1000).sum() == 30
    assert T.mask(tags, 1000, code="SET PIECES").sum() == 20
    assert T.mask(tags, 1000, code="SET PIECES", type="Corner Kick").sum() == 10
    assert T.mask(tags, 1000, team="H").sum() == 20


def test_sorted_and_nothing_dropped():
    events = [ev("B", 500, 510), ev("A", 200, 210), ev("C", 200, 205), ev("A", 960, 999)]
    out = T.convert(events, META)
    assert len(out) == 4
    assert [(t["start_frame"], t["end_frame"]) for t in out] == [(200, 205), (200, 210), (500, 510), (960, 999)]


def test_unknown_team_and_bad_frames_raise():
    for bad in ([ev("A", 200, 210, team="Someone Else")], [ev("A", 210, 210)], [ev("A", 990, 1001)],
                [ev("A", -1, 10)]):
        try:
            T.convert(bad, META)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_save_is_deterministic_and_round_trips():
    tags = T.convert([ev("BUILD UP", 200, 250), ev("SET PIECES", 400, 420, team="N/A", Type="Throw-in")], META)
    with tempfile.TemporaryDirectory() as d:
        a = T.save(tags, Path(d) / "tags.json")
        b = T.save(tags, Path(d) / "tags.json")
        assert a == b
        assert json.loads((Path(d) / "tags.json").read_text()) == tags
        assert not list(Path(d).glob("*.tmp"))


def test_other_games_are_refused():
    try:
        T.run("some_external_game")
    except SystemExit:
        return
    raise AssertionError("tags.run must refuse a game without Smart Tagging")


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
