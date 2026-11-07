"""Tiny test runner (pytest is not installed) and a fake-game builder in the shared schema."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))      # so `import mtah` works when a test is run as a script


def make_game(n=1000, fps=25, halves=((100, 449), (550, 899)), missing=(), pitch=(105.0, 68.0),
              n_players=4, seed=0):
    """A fake game in the guide section 5 schema. `missing`: [(start, stop_exclusive)] of NaN ball."""
    rng = np.random.default_rng(seed)
    L, W = pitch
    ball = np.column_stack([rng.uniform(0, L, n), rng.uniform(0, W, n)])
    for start, stop in missing:
        ball[start:stop] = np.nan
    return {
        "ball_xy": ball,
        "players_xy": np.stack([rng.uniform(0, L, (n, n_players)), rng.uniform(0, W, (n, n_players))], axis=-1),
        "players_speed": rng.uniform(0, 4, (n, n_players)),
        "players_team": np.array(["H", "A"] * (n_players // 2)),
        "meta": {"fps": fps, "pitch": list(pitch), "halves": [list(h) for h in halves], "n_frames": n},
    }


def run_module(module: ModuleType) -> int:
    """Run every test_* function of a module, print PASS/FAIL per test, return the failure count."""
    failed = 0
    short = Path(module.__file__).stem
    for name in sorted(n for n in dir(module) if n.startswith("test_")):
        try:
            getattr(module, name)()
            print(f"PASS  {short}.{name}")
        except Exception:
            failed += 1
            print(f"FAIL  {short}.{name}")
            traceback.print_exc()
    return failed
