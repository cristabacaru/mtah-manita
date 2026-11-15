"""evaluate_restarts.py: anchors against true labels (synthetic, no match data needed).

    python tests/deadball/test_evaluate_restarts.py
"""
import sys

import numpy as np

from harness import run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.deadball import evaluate_restarts as E
from mtah.deadball import restarts
from test_restarts import game, kick_from

H, A, D, X = E.pm.H, E.pm.A, E.pm.D, E.pm.X


def codes_of(*spans, n=1000):
    """[(start, stop, code), ...] over an all-X array."""
    c = np.full(n, X, dtype=np.int8)
    for start, stop, code in spans:
        c[start:stop] = code
    return c


def anchor(frame, team="H", source="gap_end", kind="throw_in", confidence=1.0):
    return {"frame": frame, "team": team, "source": source, "type": kind, "confidence": confidence}


def test_true_restarts_are_d_to_alive_only():
    c = codes_of((100, 200, H), (200, 250, D), (250, 300, A), (300, 320, H), (320, 330, D), (330, 400, H))
    assert E.true_restarts(c) == [(250, "A"), (330, "H")]


def test_true_restarts_never_cross_halftime():
    c = codes_of((100, 200, D), (300, 400, H))          # D, X break, H: no restart
    assert E.true_restarts(c) == []


def test_match_is_one_to_one_closest_first():
    assert sorted(E.match([100, 104], [103], 10)) == [(1, 0)]
    assert sorted(E.match([100, 200], [101, 203], 5)) == [(0, 0), (1, 1)]
    assert E.match([100], [200], 25) == []


def test_perfect_anchors():
    c = codes_of((100, 200, D), (200, 300, H), (300, 350, D), (350, 450, A))
    a = [anchor(200, "H"), anchor(350, "A")]
    r = E.score_game(a, a, c, 25)
    for tol in r["by_tol"].values():
        assert tol["recall"] == tol["precision"] == tol["f1"] == 1.0
    assert r["offsets"] == [0, 0] and r["team_right"] == r["team_total"] == 2


def test_late_anchor_counts_only_at_the_loose_tolerance():
    c = codes_of((100, 200, D), (200, 400, H))
    r = E.score_game([anchor(250)], [anchor(250)], c, 25)       # 2 s late
    assert r["by_tol"][1.0]["recall"] == 0 and r["by_tol"][3.0]["recall"] == 1
    assert r["offsets"] == [50]


def test_false_anchor_lowers_precision_and_null_team_is_not_scored():
    c = codes_of((100, 200, D), (200, 800, H))
    a = [anchor(200, None), anchor(600, "A")]
    r = E.score_game(a, a, c, 25)
    assert r["by_tol"][1.0]["precision"] == 0.5 and r["by_tol"][1.0]["recall"] == 1
    assert r["team_total"] == 0


def test_groups_and_chance():
    c = codes_of((100, 200, D), (200, 800, H))
    cands = [anchor(200), anchor(600, source="still_ball", kind="unknown")]
    r = E.score_game(cands[:1], cands, c, 25)
    assert r["groups"] == {"gap_end/throw_in": [1, 1], "still_ball/unknown": [0, 1]}
    assert r["chance_total"] == 700 and r["chance_hits"] == 51  # +/- 25 frames around one restart


def test_no_true_restarts_is_nan_not_a_crash():
    c = codes_of((100, 800, H))
    r = E.score_game([anchor(200)], [anchor(200)], c, 25)
    assert np.isnan(r["by_tol"][1.0]["recall"]) and r["by_tol"][1.0]["precision"] == 0


def test_end_to_end_on_a_synthetic_game():
    g = kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    g["ball_xy"][200:300] = np.nan
    g["players_xy"][:, 0] = [40.0, 1.0]
    c = codes_of((100, 300, D), (300, 450, H), (550, 900, H))
    r = E.score_game(restarts.detect(g), restarts.detect(g, drop_weak=False), c, 25)
    assert r["n_true"] == 1 and r["by_tol"][1.0]["recall"] == 1 and r["offsets"] == [0]
    s = E.summarise_level([r])
    assert s["team_accuracy"] == 1.0 and s["n_games"] == 1
    row = E.experiments_row({"n_games": 1, "levels": {"target": s}}, "test", date="2026-11-09")
    assert row.startswith("2026-11-09,P3b,test,,,1,target,")


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
