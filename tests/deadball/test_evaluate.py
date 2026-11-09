"""Step 5: evaluate.py (spec section 4, section 5 "Calibration" and "Metrics", plus edge cases).

    python tests/deadball/test_evaluate.py

All labels are synthetic. Files go to a temporary folder, never to $MTAH_CACHE.
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

from harness import run_module  # noqa: E402  (puts the repo root on sys.path)
from mtah.config import DEADBALL
from mtah.deadball import evaluate as ev

pm = ev.pm
H, A, D, X = pm.H, pm.A, pm.D, pm.X


def _gt(n=3000, halves=((100, 1400), (1600, 2900)), dead=((100, 200), (500, 700), (1000, 1100),
                                                         (1600, 1700), (2000, 2300), (2800, 2900))):
    """Synthetic GT codes: X outside halves (stop exclusive), D on `dead`, H/A alternating elsewhere."""
    codes = np.full(n, X, dtype=np.int8)
    for start, stop in halves:
        codes[start:stop] = H
        codes[start + (stop - start) // 2:stop] = A
    for start, stop in dead:
        codes[start:stop] = D
    return codes


def _p(codes):
    """GT as a probability: 1 on D, 0 on H/A, NaN on X."""
    return np.where(codes == X, np.nan, (codes == D).astype(np.float64))


def _write_gt(codes, path):
    labels = np.array(pm.LABELS)[codes]
    path.write_text("frame,label\n" + "".join(f"{i},{lab}\n" for i, lab in enumerate(labels)))
    return path


# --- spec section 4 "done when" and section 5 "Metrics" ---------------------------------------------

def test_spec_gt_against_itself_is_perfect():
    r = ev.score_copy(_gt(), _p(_gt()))
    assert r["alive_dead_f1"] == 1.0 and r["d_interval"]["mean_f1"] == 1.0 and r["ece"] == 0.0
    assert all(x == 0 for x in r["_onset"] + r["_end"])
    assert ev.self_test(_gt()) == []


def test_spec_d_shifted_25_late_gives_plus_25():
    r = ev.score_copy(_gt(), _p(ev.shift_d(_gt(), 25)))
    assert r["onset"]["median"] == 25.0 and r["end"]["median"] == 25.0


def test_spec_one_extra_d_lowers_precision_only():
    pred = _gt().copy()
    pred[300:400] = D                                      # a stoppage that is not in the GT
    r = ev.score_copy(_gt(), _p(pred))
    for s in r["d_interval"]["per_iou"].values():
        assert s["precision"] < 1.0 and s["recall"] == 1.0
    assert r["d_interval"]["n_pred"] == r["d_interval"]["n_gt"] + 1


# --- spec section 5 "Calibration" -----------------------------------------------------------------

def _calibration_gt_and_p(transform):
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 100_000)
    dead = rng.uniform(0, 1, 100_000) < p                  # Bernoulli(p), seed 0
    codes = np.where(dead, D, H).astype(np.int8)
    return codes, transform(p)


def test_spec_ece_small_when_calibrated():
    codes, p = _calibration_gt_and_p(lambda p: p)
    assert ev.ece(p, codes) < 0.02


def test_spec_ece_large_when_miscalibrated():
    codes, p = _calibration_gt_and_p(lambda p: 0.8 * p + 0.2)
    assert ev.ece(p, codes) > 0.05


def test_ece_weights_bins_by_frame_count():
    # 90 alive frames at p = 0.05 (gap 0.05) and 10 alive frames at p = 0.95 (gap 0.95):
    # weighted (90 * 0.05 + 10 * 0.95) / 100 = 0.14; an unweighted mean of bins would give 0.5
    codes = np.full(100, H, dtype=np.int8)
    p = np.r_[np.full(90, 0.05), np.full(10, 0.95)]
    assert np.isclose(ev.ece(p, codes), 0.14)


# --- decisions and edge cases -----------------------------------------------------------------------

def test_hard_call_is_strictly_above_threshold():
    gt = _gt()
    p = _p(gt)
    p[(gt != X) & (gt != D)] = DEADBALL.HARD_CALL          # exactly 0.5 on every alive frame -> alive
    assert ev.score_copy(gt, p)["alive_dead_f1"] == 1.0


def test_all_dead_prediction_scores_zero_alive_dead_f1():
    gt = _gt()
    r = ev.score_copy(gt, np.where(gt == X, np.nan, 1.0))
    assert r["alive_dead_f1"] == 0.0


def test_alive_dead_f1_matches_official_by_hand():
    gt = _gt()
    pred = gt.copy()
    pred[300:400] = D                                      # 100 alive frames called dead
    tp = int(((gt == H) | (gt == A)).sum()) - 100
    expected = 2 * tp / (2 * tp + 100)                     # no false positive, 100 false negatives
    assert np.isclose(ev.score_copy(gt, _p(pred))["alive_dead_f1"], expected)


def test_half_edge_side_left_out_only_when_both_on_edge():
    # decision (a): GT D runs (100,200) and (1600,1700) open a half; (2800,2900) closes half 2
    gt = _gt()
    r = ev.score_copy(gt, _p(gt))
    assert r["onset"]["n"] == 4 and r["end"]["n"] == 5     # 6 GT D runs, 2 onsets and 1 end on an edge
    pred = gt.copy()
    pred[100:110] = H                                      # predicted D starts 10 frames into half 1
    r = ev.score_copy(gt, _p(pred))
    assert r["onset"]["n"] == 5 and sorted(r["_onset"])[-1] == 10


def test_halves_are_scored_separately():
    # D at the end of half 1 and the start of half 2: two runs, never one across half-time
    gt = _gt(dead=((1300, 1400), (1600, 1700)))
    r = ev.score_copy(gt, _p(gt))
    assert r["d_interval"]["n_gt"] == 2 and len(r["halves"]) == 2
    assert [h["d_interval"]["n_gt"] for h in r["halves"]] == [1, 1]


def test_matcher_pairs_same_count_as_official():
    rng = np.random.default_rng(1)
    for _ in range(50):
        gt_d = [(s, s + int(rng.integers(5, 60)), D) for s in sorted(rng.choice(2000, 15, replace=False))]
        pred_d = [(s, s + int(rng.integers(5, 60)), D) for s in sorted(rng.choice(2000, 15, replace=False))]
        overlaps = pm.spell_overlaps(gt_d, pred_d)
        for t in (0.1, 0.3, 0.5):
            assert len(ev.matched_pairs(overlaps, t)) == pm.match_spells(overlaps, 15, 15, t)[0]


def test_bad_p_dead_is_rejected():
    gt = _gt()
    for bad in (np.full(len(gt) - 1, 0.5), np.where(gt == X, np.nan, 1.2), np.full(len(gt), np.nan)):
        try:
            ev.score_copy(gt, bad)
        except ValueError:
            continue
        raise AssertionError("bad p_dead accepted")


def test_nan_outside_halves_is_allowed_and_ignored():
    gt = _gt()
    p = _p(gt)
    p[gt == X] = 0.9                                       # whatever is outside the halves never counts
    assert ev.score_copy(gt, p)["alive_dead_f1"] == 1.0 and ev.ece(p, gt) == 0.0


def test_no_paired_runs_gives_none_not_warning():
    gt = _gt()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = ev.score_copy(gt, np.where(gt == X, np.nan, 0.0))   # never predicts D
    assert r["onset"] == {"n": 0, "median": None, "p90_abs": None}
    assert r["d_interval"]["mean_f1"] == 0.0


def test_gt_csv_round_trip_and_hole_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_gt(_gt(), Path(tmp) / "g_gt.csv")
        assert np.array_equal(ev.read_gt(path), _gt())
        lines = path.read_text().splitlines()
        (Path(tmp) / "hole.csv").write_text("\n".join(lines[:10] + lines[11:]) + "\n")
        try:
            ev.read_gt(Path(tmp) / "hole.csv")
        except ValueError:
            return
        raise AssertionError("a missing frame was accepted")


def _fake_copies(tmp, games, levels=("light", "heavy"), shifts=(0, 25)):
    """Write GT CSVs and p_dead.npy for fake copies <game>__<level> under tmp; return suite-style entries."""
    entries = []
    for game in games:
        for level, shift in zip(levels, shifts):
            name = f"{game}__{level}"
            gt_path = _write_gt(_gt(), tmp / f"{name}_gt.csv")
            (tmp / "cache" / name).mkdir(parents=True)
            np.save(tmp / "cache" / name / "p_dead.npy", _p(ev.shift_d(_gt(), shift)))
            entries.append({"name": name, "source": game, "level": level, "fold": game, "gt": str(gt_path),
                            "halves": [[0, 1]]})     # wrong on purpose: halves must never be read from here
    return entries


def test_manifest_end_to_end_two_providers():
    # the --all path on fake copies: 2 games x 2 noise copies, providers DFL and Metrica
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entries = _fake_copies(tmp, ("g1", "g2"))
        rows = ["copy,gt,provider"] + [f"{e['name']},{e['gt']},{'DFL' if e['source'] == 'g1' else 'Metrica'}"
                                       for e in entries]
        (tmp / "manifest.csv").write_text("\n".join(rows) + "\n")
        s = ev.evaluate_manifest(tmp / "manifest.csv", cache_dir=tmp / "cache", out_dir=tmp / "reports")
        # a CSV list does not say which copies share a match: each GT file counts as one game here;
        # only --suite (P2's manifest, field `source`) counts matches
        assert s["n_copies"] == 4 and s["n_games"] == 4 and list(s["levels"]) == ["all"]
        assert sorted(s["levels"]["all"]["by_provider"]) == ["DFL", "Metrica"]
        assert len(list((tmp / "reports").glob("*.json"))) == 4


def test_provider_comes_from_p2_config():
    # decision (a) of 2026-11-09: P2's own lists, never a guess from the name
    from mtah import config
    assert ev.provider_of("dfl_" + config.EXTERNAL.DFL_MATCHES[0]) == "DFL"
    assert ev.provider_of("metrica1_events") == "Metrica"
    for unknown in ("dfl_NOTAGAME", "statsbomb_x"):
        try:
            ev.provider_of(unknown)
        except ValueError:
            continue
        raise AssertionError(f"{unknown!r} got a provider")


def test_suite_scores_per_level_and_ignores_manifest_halves():
    from mtah import config
    dfl, met = "dfl_" + config.EXTERNAL.DFL_MATCHES[0], "metrica1_events"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entries = _fake_copies(tmp, (dfl, met))           # light = exact GT, heavy = 25 frames late
        (tmp / "suite.json").write_text(json.dumps({"games": entries}))
        rows = ev.read_suite(tmp / "suite.json")
        assert {r["provider"] for r in rows} == {"DFL", "Metrica"}
        s = ev.evaluate_rows(rows, cache_dir=tmp / "cache", out_dir=tmp / "reports")
        assert list(s["levels"]) == ["heavy", "light"] and s["n_games"] == 2 and s["n_copies"] == 4
        light, heavy = s["levels"]["light"], s["levels"]["heavy"]
        assert light["alive_dead_f1"] == 1.0 and light["d_interval_f1"] == 1.0 and light["ece"] == 0.0
        assert heavy["by_provider"]["DFL"]["onset"]["median"] == 25.0 and heavy["d_interval_f1"] < 1.0
        only = ev.read_suite(tmp / "suite.json", levels=["heavy"])
        assert [r["level"] for r in only] == ["heavy", "heavy"]


def test_missing_p_dead_is_listed_before_scoring():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entries = _fake_copies(tmp, ("metrica1_events",))
        (tmp / "cache" / entries[1]["name"] / "p_dead.npy").unlink()
        (tmp / "suite.json").write_text(json.dumps({"games": entries}))
        try:
            ev.evaluate_rows(ev.read_suite(tmp / "suite.json"), cache_dir=tmp / "cache", out_dir=tmp / "r")
        except FileNotFoundError as err:
            assert entries[1]["name"] in str(err) and entries[0]["name"] not in str(err)
            assert not (tmp / "r").exists()                # nothing scored or written
            return
        raise AssertionError("missing p_dead.npy not reported")


def test_experiments_row_uses_score_suite_layout():
    # decision (b) of 2026-11-09: games = matches, noise_levels = level names, one block per level
    import csv as _csv
    from mtah import config
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entries = _fake_copies(tmp, ("dfl_" + config.EXTERNAL.DFL_MATCHES[0], "metrica2_events"))
        (tmp / "suite.json").write_text(json.dumps({"games": entries}))
        s = ev.evaluate_rows(ev.read_suite(tmp / "suite.json"), cache_dir=tmp / "cache", out_dir=tmp / "r")
        row = next(_csv.reader([ev.experiments_row(s, "v0 stub", date="2026-11-09")]))
        header = (config.REPO_ROOT / "experiments.csv").read_text().splitlines()[0].split(",")
        assert len(row) == len(header)
        rec = dict(zip(header, row))
        assert rec["owner"] == "P3a" and rec["spellF1_mean"] == "" and rec["macroF1_mean"] == ""
        assert rec["games"] == "2" and rec["noise_levels"] == "heavy/light"
        blocks = rec["notes"].split("; ")
        assert blocks[0].startswith("heavy alive_dead_f1=") and blocks[1].startswith("light alive_dead_f1=1.000")
        assert "DFL_onset_med=+25f" in blocks[0] and "Metrica_onset_med=+0f" in blocks[1]


def test_manifest_with_wrong_columns_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "m.csv"
        bad.write_text("name,gt\nx,y\n")
        try:
            ev.read_manifest(bad)
        except ValueError:
            return
        raise AssertionError("wrong manifest columns accepted")


def test_report_is_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        r = ev.score_copy(_gt(), _p(ev.shift_d(_gt(), 25)))
        a = ev.write_report(r, "a", tmp).read_text()
        b = ev.write_report(ev.score_copy(_gt(), _p(ev.shift_d(_gt(), 25))), "b", tmp).read_text()
        assert a == b and "_onset" not in json.loads(a)


if __name__ == "__main__":
    sys.exit(1 if run_module(sys.modules[__name__]) else 0)
