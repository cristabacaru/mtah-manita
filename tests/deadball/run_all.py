"""Run every tests/deadball/test_*.py; exit 1 if any test fails.

    python tests/deadball/run_all.py
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import run_module  # noqa: E402

failed = sum(run_module(importlib.import_module(p.stem)) for p in sorted(HERE.glob("test_*.py")))
print("ALL PASSED" if not failed else f"{failed} FAILED")
sys.exit(1 if failed else 0)
