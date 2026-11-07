"""Synthetic tests for mtah/data.py. No match data needed.

    python tests/test_data.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MTAH_CACHE"] = tempfile.mkdtemp(prefix="mtah_test_")   # before importing config

from mtah import config, data  # noqa: E402


def test_speed_constant_velocity():
    fps = 25
    t = np.arange(100) / fps
    xy = np.stack([3.0 * t, 4.0 * t], axis=-1)[:, None, :]          # 5 m/s, one player
    sp = data.position_speed(xy, fps)
    k = round(config.SPEED_HALF_WINDOW_S * fps)
    assert np.all(np.isnan(sp[:k])) and np.all(np.isnan(sp[-k:])), "edges must be NaN"
    assert np.allclose(sp[k:-k], 5.0), sp[k:-k]


def test_speed_nan_gap_and_fps_independence():
    for fps in (10, 25, 30):
        t = np.arange(200) / fps
        xy = np.stack([2.0 * t, 0 * t], axis=-1)[:, None, :]
        xy[50, 0] = np.nan
        sp = data.position_speed(xy, fps)
        k = max(1, round(config.SPEED_HALF_WINDOW_S * fps))
        assert np.all(np.isnan(sp[[50 - k, 50, 50 + k], 0])), "a missing end or empty slot must give NaN"
        good = np.isfinite(sp[:, 0])
        assert np.allclose(sp[good, 0], 2.0), (fps, sp[good, 0])


def test_speed_cap():
    fps = 25
    x = np.zeros(60)
    x[30:] = 40.0                                     # slot jumps 40 m in one frame
    xy = np.stack([x, np.zeros(60)], axis=-1)[:, None, :]
    sp = data.position_speed(xy, fps)
    k = round(config.SPEED_HALF_WINDOW_S * fps)
    assert np.all(np.isnan(sp[30 - k:30 + k, 0])), "frames whose window spans the jump must be NaN"
    assert np.allclose(sp[k:30 - k, 0], 0) and np.allclose(sp[30 + k:-k, 0], 0)
    # a real 11 m/s sprint is kept
    t = np.arange(60) / fps
    sprint = np.stack([11.0 * t, 0 * t], axis=-1)[:, None, :]
    assert np.allclose(data.position_speed(sprint, fps)[k:-k], 11.0)


def test_rotation():
    pitch = (100.0, 60.0)
    xy = np.array([[[10.0, 5.0]], [[10.0, 5.0]], [[np.nan, np.nan]]])
    flip = np.array([False, True, True])
    out = data.rotate_halves(xy, flip, pitch)
    assert np.allclose(out[0, 0], [10, 5]) and np.allclose(out[1, 0], [90, 55])
    assert np.all(np.isnan(out[2])), "NaN must stay NaN"
    # rotation keeps distances, so speed before and after is the same
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(0, 0.1, (60, 3, 2)), axis=0) + 30
    flip = np.zeros(60, bool)
    flip[30:] = True
    a = data.position_speed(walk, 25)
    b = data.position_speed(data.rotate_halves(walk, np.ones(60, bool), pitch), 25)
    assert np.allclose(a, b, equal_nan=True)


def test_frame_half():
    idx = data.frame_half(20, [(2, 7), (12, 17)])
    assert idx.tolist() == [0] * 12 + [1] * 8, idx.tolist()


def _toy(n=50, p=4):
    rng = np.random.default_rng(1)
    pxy = rng.uniform(0, 50, (n, p, 2))
    pxy[10:15, 2] = np.nan
    ball = rng.uniform(0, 50, (n, 2))
    ball[20:30] = np.nan
    arrays = {"ball_xy": ball, "players_xy": pxy, "players_speed": data.position_speed(pxy, 25),
              "players_team": np.array(["H", "H", "A", "A"])}
    meta = {"pitch": [105.0, 68.0], "fps": 25, "halves": [[0, 24], [25, 49]], "n_frames": n}
    return arrays, meta


def test_save_load_roundtrip():
    arrays, meta = _toy()
    folder = data.save("toy", arrays, meta)
    back = data.load("toy")
    for k in ("ball_xy", "players_xy", "players_speed"):
        assert np.allclose(back[k], arrays[k].astype(np.float32), equal_nan=True), k
    assert back["players_team"].tolist() == ["H", "H", "A", "A"]
    assert back["meta"]["halves"] == [[0, 24], [25, 49]] and back["meta"]["game"] == "toy"
    assert json.loads((folder / "meta.json").read_text())["schema_version"] == data.SCHEMA_VERSION


def test_validate_rejects_bad_input():
    arrays, meta = _toy()
    bad_cases = [
        dict(arrays, ball_xy=arrays["ball_xy"][:-1]),                   # wrong length
        dict(arrays, players_team=np.array(["H", "H", "A", "X"])),     # bad team code
        dict(arrays, players_speed=np.nan_to_num(arrays["players_speed"])),  # 0 where slot empty
    ]
    for bad in bad_cases:
        try:
            data.validate(bad, meta)
        except ValueError:
            continue
        raise AssertionError("validate accepted bad input")
    centred = dict(arrays, players_xy=arrays["players_xy"] - np.array([52.5, 34.0]))
    centred["players_speed"] = data.position_speed(centred["players_xy"], 25)
    normalised = dict(arrays, players_xy=arrays["players_xy"] / 105.0 * 3)   # wrong units
    for bad in (centred,):
        try:
            data.validate(bad, meta)
        except ValueError as err:
            assert "origin" in str(err)
            continue
        raise AssertionError("validate accepted a centre origin")
    data.validate(normalised, meta)   # small values stay inside the pitch: not this check's job
    try:
        data.validate(arrays, dict(meta, halves=[[0, 60]]))
    except ValueError:
        pass
    else:
        raise AssertionError("validate accepted halves outside the recording")


def test_load_missing_game():
    try:
        data.load("no_such_game")
    except FileNotFoundError:
        return
    raise AssertionError("load of a missing game must raise FileNotFoundError")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"{len(tests)} passed")
