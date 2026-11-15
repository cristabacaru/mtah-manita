"""Tests for mtah/submit.py. No match data needed.

    python tests/test_submit.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mtah import config, submit  # noqa: E402

N = config.N_FRAMES


def graded_mask():
    g = np.zeros(N, dtype=bool)
    for _, first, last in config.HALF_BLOCKS:
        g[first:last + 1] = True
    return g


def good_labels():
    lab = np.full(N, "X", dtype="<U1")
    g = graded_mask()
    lab[g] = np.resize(np.array(["H"] * 50 + ["D"] * 30 + ["A"] * 50), g.sum())
    return lab


def expect_error(labels, text):
    try:
        submit.prepare(labels)
    except ValueError as err:
        assert text in str(err), err
        return
    raise AssertionError(f"prepare accepted bad labels ({text})")


def test_x_forced_outside_halves():
    lab = good_labels()
    lab[~graded_mask()] = "D"
    out, fixes = submit.prepare(lab)
    assert (out[~graded_mask()] == "X").all() and fixes["forced_x_outside"] == int((~graded_mask()).sum())
    assert np.array_equal(out[graded_mask()], lab[graded_mask()])


def test_graded_x_is_refused():
    lab = good_labels()
    lab[np.flatnonzero(graded_mask())[10]] = "X"
    expect_error(lab, "graded frames are X")


def test_bad_length_and_bad_label_are_refused():
    expect_error(good_labels()[:-1], "shape")
    lab = good_labels()
    lab[np.flatnonzero(graded_mask())[0]] = "Q"
    expect_error(lab, "labels outside")


def test_file_format():
    data = submit.csv_bytes(np.array(["X", "H", "D"]))
    assert data == b"frame,label\n0,X\n1,H\n2,D\n"


def test_written_file_passes_the_official_validator_and_is_deterministic():
    lab, _ = submit.prepare(good_labels())
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "Manita.csv"
        a = submit.write(lab, path)
        b = submit.write(lab, path)
        assert a == b
        assert (Path(d) / "Manita.csv.sha256").read_text() == f"{a}  Manita.csv\n"
        assert not list(Path(d).glob("*.tmp"))
        passed, report = submit.validate(path)
        assert passed, report
        assert np.array_equal(submit.load_labels(path), lab)          # csv round trip
        np.save(Path(d) / "labels.npy", lab)
        assert np.array_equal(submit.load_labels(Path(d) / "labels.npy"), lab)


def test_validator_failure_is_reported():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "bad.csv"
        path.write_text("frame,label\n0,H\n")
        passed, _ = submit.validate(path)
        assert not passed


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("ok  ", test.__name__)
    print(f"{len(tests)} passed")
