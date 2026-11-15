"""restarts.py: restart anchors from tracking (synthetic games, no match data needed).

    python tests/deadball/test_restarts.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from harness import run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.config import RESTARTS as R
from mtah.deadball import restarts as Rm

PITCH = (105.0, 68.0)
L, W = PITCH


def game(n=1000, fps=25, halves=((100, 449), (550, 899)), slow=True):
    """22 players standing in their own halves (H at x < L/2), ball visible and rolling mid-pitch."""
    rng = np.random.default_rng(0)
    hx = rng.uniform(10, 45, 11)
    ax = rng.uniform(60, 95, 11)
    ys = rng.uniform(10, 58, 22)
    pxy = np.empty((n, 22, 2))
    pxy[:, :11, 0], pxy[:, 11:, 0] = hx, ax
    pxy[:, :, 1] = ys
    speed = np.full((n, 22), 0.5 if slow else 3.0)
    t = np.arange(n) / fps
    ball = np.column_stack([30 + 2.0 * np.sin(t), 20 + 0 * t])      # slow-rolling ball, never "still"
    return {"ball_xy": ball, "players_xy": pxy, "players_speed": speed,
            "players_team": np.array(["H"] * 11 + ["A"] * 11),
            "meta": {"fps": fps, "pitch": list(PITCH), "halves": [list(h) for h in halves], "n_frames": n}}


def kick_from(g, frame, at, velocity=(15.0, 0.0)):
    """Ball sits at `at` until `frame`, then flies off at `velocity` m/s."""
    fps = g["meta"]["fps"]
    n = g["meta"]["n_frames"]
    t = (np.arange(n) - frame) / fps
    moving = t >= 0
    g["ball_xy"][:, 0] = np.where(moving, at[0] + velocity[0] * t, at[0])
    g["ball_xy"][:, 1] = np.where(moving, at[1] + velocity[1] * t, at[1])
    return g


# --- geometry -------------------------------------------------------------------------------------

def test_classify_zones():
    assert Rm.classify(1.0, 1.0, PITCH) == "corner"
    assert Rm.classify(L - 0.5, W + 0.5, PITCH) == "corner"            # just past the corner flag
    assert Rm.classify(40.0, 0.5, PITCH) == "throw_in"
    assert Rm.classify(40.0, -1.0, PITCH) == "throw_in"                # past the touchline
    assert Rm.classify(5.0, W / 2 + 3, PITCH) == "goal_kick"
    assert Rm.classify(L - 5.0, W / 2, PITCH) == "goal_kick"
    assert Rm.classify(L / 2 + 1, W / 2, PITCH) == "kickoff"
    assert Rm.classify(40.0, 30.0, PITCH) == "unknown"
    assert Rm.classify(np.nan, 30.0, PITCH) == "unknown"


def test_classify_follows_pitch_size():
    assert Rm.classify(50.0, 30.0, (100.0, 60.0)) == "kickoff"
    assert Rm.classify(50.0, 30.0, PITCH) == "unknown"                  # 4.7 m from (52.5, 34): too far


def test_laws_team_uses_h_attacks_plus_x():
    assert Rm.laws_team("corner", L - 1, PITCH) == "H" and Rm.laws_team("corner", 1, PITCH) == "A"
    assert Rm.laws_team("goal_kick", 5, PITCH) == "H" and Rm.laws_team("goal_kick", L - 5, PITCH) == "A"
    assert Rm.laws_team("throw_in", 40, PITCH) is None


# --- detection --------------------------------------------------------------------------------------

def test_gap_end_throw_in():
    g = kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    g["ball_xy"][200:300] = np.nan                                      # 4 s missing, reappears moving
    g["players_xy"][:, 0] = [40.0, 1.0]                                 # an H player at the throw
    a = [x for x in Rm.detect(g) if x["source"] == "gap_end"]
    assert len(a) == 1, a
    a = a[0]
    assert a["frame"] == 300 and a["type"] == "throw_in" and a["team"] == "H"
    assert a["slow_before"] and a["ball_played"] and a["confidence"] == R.CONFIDENCE[(True, True, True)]
    assert a["half"] == 1 and a["window"] == [275, 326]


def test_restart_frame_is_when_ball_is_played():
    g = kick_from(game(), 320, at=(5.0, W / 2))                          # still in the goal area, kicked at 320
    g["ball_xy"][200:260] = np.nan                                      # reappears still at 260
    a = [x for x in Rm.detect(g) if x["type"] == "goal_kick"]
    assert a and abs(a[0]["frame"] - 320) <= 3, a                        # within the 0.2 s speed window
    assert a[0]["team"] == "H"                                          # x < L/2: H's own goal area


def test_still_ball_unknown_is_dropped_known_is_kept():
    mid = kick_from(game(), 300, at=(40.0, 30.0))                       # still mid-pitch, then kicked
    assert not [x for x in Rm.detect(mid) if x["source"] == "still_ball"]
    assert [x for x in Rm.detect(mid, drop_weak=False) if x["source"] == "still_ball"]
    line = kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    kept = [x for x in Rm.detect(line) if x["source"] == "still_ball"]
    assert kept and kept[0]["type"] == "throw_in"


def test_still_ball_needs_players_slow_before():
    fast = kick_from(game(slow=False), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    assert not [x for x in Rm.detect(fast) if x["source"] == "still_ball"]


def test_confidence_is_the_evidence_group_table():
    # recognised + slow + played
    g = kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    g["ball_xy"][200:300] = np.nan
    a = [x for x in Rm.detect(g) if x["source"] == "gap_end"][0]
    assert a["confidence"] == R.CONFIDENCE[(True, True, True)]
    # unknown spot + slow + played
    g = kick_from(game(), 300, at=(40.0, 30.0))
    g["ball_xy"][200:300] = np.nan
    a = [x for x in Rm.detect(g) if x["source"] == "gap_end"][0]
    assert a["type"] == "unknown" and a["confidence"] == R.CONFIDENCE[(False, True, True)]
    # recognised + not slow + played
    g = kick_from(game(slow=False), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    g["ball_xy"][200:300] = np.nan
    a = [x for x in Rm.detect(g) if x["source"] == "gap_end"][0]
    assert a["confidence"] == R.CONFIDENCE[(True, False, True)]


def test_weak_groups_are_dropped_but_kept_with_drop_weak_false():
    # unknown spot, players fast: hits about chance on P2's suite -> dropped
    g = kick_from(game(slow=False), 300, at=(40.0, 30.0))
    g["ball_xy"][200:300] = np.nan
    assert not [x for x in Rm.detect(g) if x["source"] == "gap_end"]
    kept = [x for x in Rm.detect(g, drop_weak=False) if x["source"] == "gap_end"]
    assert kept and kept[0]["confidence"] == 0
    # the ball never moves after reappearing: no ball_played -> dropped
    g = game()
    g["ball_xy"][:] = [40.0, 0.5]
    g["ball_xy"][200:300] = np.nan
    assert not Rm.detect(g)
    assert all(x["confidence"] == 0 for x in Rm.detect(g, drop_weak=False))
    # every kept anchor has a listed confidence
    for x in Rm.detect(kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))):
        assert x["confidence"] in R.CONFIDENCE.values()


def test_kickoff_formation():
    g = kick_from(game(), 300, at=(L / 2, W / 2))
    a = [x for x in Rm.detect(g) if x["type"] == "kickoff"]
    assert a and abs(a[0]["frame"] - 300) <= 3, a


def test_anchors_only_inside_halves_and_merged():
    g = kick_from(game(), 470, at=(40.0, 0.5), velocity=(0.0, 12.0))    # restart in the break
    g["ball_xy"][420:470] = np.nan
    assert all(100 <= a["frame"] <= 449 or 550 <= a["frame"] <= 899 for a in Rm.detect(g))
    both = kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    both["ball_xy"][200:290] = both["ball_xy"][299]                     # still, then missing, then played
    both["ball_xy"][290:300] = np.nan
    frames = [a["frame"] for a in Rm.detect(both)]
    assert all(b - a >= R.MERGE_S * 25 for a, b in zip(frames, frames[1:])), frames


def test_window_clipped_to_half():
    g = kick_from(game(), 105, at=(40.0, 0.5), velocity=(0.0, 12.0))
    g["ball_xy"][20:105] = np.nan
    for a in Rm.detect(g):
        assert 100 <= a["window"][0] <= a["frame"] < a["window"][1] <= 450


def test_durations_follow_fps():
    g = kick_from(game(n=1200, fps=30, halves=((120, 539), (660, 1079))), 360, at=(40.0, 0.5),
                  velocity=(0.0, 12.0))
    g["ball_xy"][250:360] = np.nan                                      # 110 frames = 3.7 s at 30 fps
    a = [x for x in Rm.detect(g) if x["source"] == "gap_end"]
    assert a and a[0]["frame"] == 360 and a[0]["window"] == [330, 391]


def test_save_deterministic_and_mask():
    anchors = Rm.detect(kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0)))
    with tempfile.TemporaryDirectory() as d:
        assert Rm.save(anchors, Path(d) / "a.json") == Rm.save(anchors, Path(d) / "a.json")
    m = Rm.mask(anchors, 1000)
    assert m.sum() == sum(a["window"][1] - a["window"][0] for a in anchors)   # windows do not overlap here
    assert Rm.mask(anchors, 1000, min_confidence=1.01).sum() == 0
    assert Rm.mask(anchors, 1000, type="corner").sum() == 0


def test_input_not_modified():
    g = kick_from(game(), 300, at=(40.0, 0.5), velocity=(0.0, 12.0))
    before = {k: np.array(v, copy=True) for k, v in g.items() if k != "meta"}
    Rm.detect(g)
    for k, v in before.items():
        assert np.array_equal(g[k], v, equal_nan=v.dtype.kind == "f"), k


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
