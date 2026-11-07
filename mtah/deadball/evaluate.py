"""Score p_dead against frame labels (spec section 4). No bootstrap yet (decision (b), 2026-11-07).

For one degraded copy of an external game: the copy's p_dead.npy against that game's frame,label
GT (guide section 4). Everything is computed inside the GT's valid runs (the halves), never across
half-time, with the official scorer's own functions (evaluation/possession_metrics.py, imported,
never edited):
- alive_dead_f1: p_dead > HARD_CALL is called D, anything else alive (H), then the official F1.
- D-interval F1: D runs from the official segments(), matched one-to-one with the official
  spell_overlaps + match_spells, F1 averaged over the official SPELL_IOUS.
- boundary offset: D runs paired at IoU >= OFFSET_IOU by the same greedy order; signed error in
  frames, prediction minus GT (late = positive), onset and end, median and p90. A side where both
  runs sit on the half edge is left out: it is 0 by construction (decision (a)).
- ECE: ECE_BINS equal-width bins over graded frames, weighted by frame count.

    python -m mtah.deadball.evaluate --game <copy> --gt <gt.csv> [--provider DFL]
    python -m mtah.deadball.evaluate --self-test --gt <gt.csv>
    python -m mtah.deadball.evaluate --all <manifest.csv> --change "what changed"

The manifest is a CSV with columns copy,gt,provider: one row per degraded copy, gt relative to the
repo root or absolute. It is written once P2 gives the layout (decision (d)); nothing is guessed.
--all prints a ready-to-paste experiments.csv row (scores in notes, decision (c)); it never writes
experiments.csv. Prints aggregates only, never frame-level values.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from mtah import config
from mtah.config import DEADBALL

_EVAL_DIR = str(config.REPO_ROOT / "evaluation")
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
import possession_metrics as pm  # noqa: E402  (official scorer, wrapped not edited)

MANIFEST_COLUMNS = ("copy", "gt", "provider")


# --- inputs ----------------------------------------------------------------------------------------

def read_gt(path: str | Path) -> np.ndarray:
    """GT codes (n,) int8 in the official encoding (H, A, D, X = 0, 1, 2, 3); frames must be 0..n-1."""
    frame = pm.read_label_csv(path)
    if not np.array_equal(frame["frame"].to_numpy(), np.arange(len(frame))):
        raise ValueError(f"{path}: frame column must be 0..n-1 with no holes")
    return pm.label_codes(frame)


def check_p_dead(p_dead: np.ndarray, gt_codes: np.ndarray) -> np.ndarray:
    """float64 copy of p_dead; ValueError unless it has the GT's length and lies in [0, 1] on graded frames."""
    p = np.asarray(p_dead, dtype=np.float64)
    if p.shape != gt_codes.shape:
        raise ValueError(f"p_dead has shape {p.shape}, GT has {gt_codes.shape}")
    graded = p[gt_codes != pm.X]
    if not np.isfinite(graded).all():
        raise ValueError("p_dead is NaN or infinite on a graded frame")
    if (graded < 0).any() or (graded > 1).any():
        raise ValueError("p_dead is outside [0, 1] on a graded frame")
    return p


def load_p_dead(copy: str, cache_dir: str | Path | None = None) -> np.ndarray:
    """$MTAH_CACHE/<copy>/p_dead.npy (or cache_dir/<copy>/p_dead.npy)."""
    folder = Path(cache_dir) / copy if cache_dir is not None else config.game_cache(copy)
    return np.load(folder / "p_dead.npy")


# --- pieces ----------------------------------------------------------------------------------------

def hard_codes(p: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray:
    """Official codes: D where p > HARD_CALL inside the runs, H elsewhere inside them, X outside."""
    codes = np.full(len(p), pm.X, dtype=np.int8)
    for start, stop in runs:
        codes[start:stop] = np.where(p[start:stop] > DEADBALL.HARD_CALL, pm.D, pm.H)
    return codes


def d_runs(codes: np.ndarray, runs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """[(start, stop, D)] maximal D runs inside each valid run (official segments, stop exclusive)."""
    return [seg for seg in pm.segments(codes, runs) if seg[2] == pm.D]


def matched_pairs(overlaps: list, threshold: float) -> list[tuple[int, int]]:
    """[(gt_index, pred_index)] in the official greedy order (pm.matched_gt_flags), at IoU >= threshold."""
    pairs, used_gt, used_pred = [], set(), set()
    for iou, gt_index, pred_index in overlaps:
        if iou < threshold:
            break
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        pairs.append((gt_index, pred_index))
    return pairs


def d_interval_scores(gt_d: list, pred_d: list) -> dict:
    """Precision / recall / F1 of D runs at each official SPELL_IOUS, and their mean F1 (as spell_scores)."""
    overlaps = pm.spell_overlaps(gt_d, pred_d)
    per_iou, f1s = {}, []
    for threshold in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(overlaps, len(gt_d), len(pred_d), threshold)
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_iou[str(threshold)] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    return {"per_iou": per_iou, "mean_f1": float(np.mean(f1s)), "n_gt": len(gt_d), "n_pred": len(pred_d)}


def boundary_offsets(gt_d: list, pred_d: list, runs: list) -> tuple[list[int], list[int]]:
    """(onset errors, end errors) in frames, prediction minus GT, for D runs paired at OFFSET_IOU.

    A side where the GT run and the predicted run both sit on the half edge is left out (it is 0 by
    construction, decision (a)). If only one of them sits on the edge, the error is real and kept.
    """
    overlaps = pm.spell_overlaps(gt_d, pred_d)
    onset, end = [], []
    for gi, pj in matched_pairs(overlaps, DEADBALL.OFFSET_IOU):
        gs, ge, _ = gt_d[gi]
        ps, pe, _ = pred_d[pj]
        run_start, run_stop = next((a, b) for a, b in runs if a <= gs < b)
        if not (gs == run_start and ps == run_start):
            onset.append(int(ps - gs))
        if not (ge == run_stop and pe == run_stop):
            end.append(int(pe - ge))
    return onset, end


def summarise(offsets: list[int]) -> dict:
    """{n, median, p90 of |error|}; median and p90 are None when there is nothing to summarise."""
    if not offsets:
        return {"n": 0, "median": None, "p90_abs": None}
    a = np.asarray(offsets, dtype=np.float64)
    return {"n": len(a), "median": float(np.median(a)), "p90_abs": float(np.percentile(np.abs(a), 90))}


def ece(p: np.ndarray, gt_codes: np.ndarray) -> float:
    """Expected calibration error over graded frames: ECE_BINS equal-width bins, weighted by count."""
    graded = gt_codes != pm.X
    prob, dead = p[graded], (gt_codes[graded] == pm.D).astype(np.float64)
    if len(prob) == 0:
        raise ValueError("no graded frames")
    k = DEADBALL.ECE_BINS
    idx = np.minimum((prob * k).astype(np.int64), k - 1)          # p = 1.0 goes in the last bin
    count = np.bincount(idx, minlength=k).astype(np.float64)
    sum_p = np.bincount(idx, weights=prob, minlength=k)
    sum_y = np.bincount(idx, weights=dead, minlength=k)
    used = count > 0
    return float(np.sum(np.abs(sum_p[used] - sum_y[used])) / len(prob))


# --- one copy ------------------------------------------------------------------------------------

def _score_runs(gt_codes: np.ndarray, p: np.ndarray, runs: list) -> dict:
    """All four scores restricted to `runs` (frames outside them count as X on both sides)."""
    gt = np.full(len(gt_codes), pm.X, dtype=np.int8)
    for start, stop in runs:
        gt[start:stop] = gt_codes[start:stop]
    pred = hard_codes(p, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    gt_d, pred_d = d_runs(gt, runs), d_runs(pred, runs)
    onset, end = boundary_offsets(gt_d, pred_d, runs)
    return {
        "alive_dead_f1": float(pm.alive_dead_f1(cm3)),
        "d_interval": d_interval_scores(gt_d, pred_d),
        "onset": summarise(onset), "end": summarise(end),
        "ece": ece(p, gt),
        "_onset": onset, "_end": end,           # raw per-run errors, kept for pooling, not printed
    }


def score_copy(gt_codes: np.ndarray, p_dead: np.ndarray) -> dict:
    """Scores over the whole game (halves kept apart) and per half."""
    p = check_p_dead(p_dead, gt_codes)
    runs = pm.valid_runs(gt_codes)
    if not runs:
        raise ValueError("the GT has no graded frames")
    out = _score_runs(gt_codes, p, runs)
    out["halves"] = [{k: v for k, v in _score_runs(gt_codes, p, [r]).items() if not k.startswith("_")}
                     for r in runs]
    return out


def public(report: dict) -> dict:
    """The report without the raw per-run lists (for JSON and printing)."""
    return {k: v for k, v in report.items() if not k.startswith("_")}


def write_report(report: dict, name: str, out_dir: str | Path | None = None) -> Path:
    """JSON report to out_dir/<name>.json (default $MTAH_CACHE/reports/deadball/). Sorted keys: deterministic."""
    folder = Path(out_dir) if out_dir is not None else config.REPORTS_DIR / "deadball"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(public(report), indent=2, sort_keys=True) + "\n")
    return path


# --- self-test: the ruler measures itself ----------------------------------------------------------

def shift_d(gt_codes: np.ndarray, frames: int) -> np.ndarray:
    """GT codes with every label moved `frames` later inside each valid run (head filled with the first label)."""
    out = gt_codes.copy()
    for start, stop in pm.valid_runs(gt_codes):
        window = gt_codes[start:stop]
        k = min(frames, len(window))
        out[start + k:stop] = window[:len(window) - k]
        out[start:start + k] = window[0]
    return out


def self_test(gt_codes: np.ndarray, shift: int = 25) -> list[str]:
    """Problems found (empty = pass): GT against itself must give F1 = 1, offsets 0, ECE 0; a GT
    shifted `shift` frames late must give median onset and end offsets of +shift."""
    as_p = lambda codes: np.where(codes == pm.X, np.nan, (codes == pm.D).astype(np.float64))  # noqa: E731
    problems = []
    same = score_copy(gt_codes, as_p(gt_codes))
    if same["alive_dead_f1"] != 1.0 or same["d_interval"]["mean_f1"] != 1.0:
        problems.append("GT vs itself: an F1 is not 1.0")
    if any(x != 0 for x in same["_onset"] + same["_end"]):
        problems.append("GT vs itself: a boundary offset is not 0")
    if same["ece"] != 0.0:
        problems.append("GT vs itself: ECE is not 0")
    late = score_copy(gt_codes, as_p(shift_d(gt_codes, shift)))
    for side in ("onset", "end"):
        if late[side]["median"] != float(shift):
            problems.append(f"GT {shift} frames late: median {side} offset is {late[side]['median']}, not +{shift}")
    return problems


# --- many copies -----------------------------------------------------------------------------------

def read_manifest(path: str | Path) -> list[dict]:
    """Rows {copy, gt, provider} from the manifest CSV; gt made absolute against the repo root."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or tuple(rows[0].keys()) != MANIFEST_COLUMNS:
        raise ValueError(f"{path}: expected columns {','.join(MANIFEST_COLUMNS)}")
    for row in rows:
        gt = Path(row["gt"])
        row["gt"] = gt if gt.is_absolute() else config.REPO_ROOT / gt
    return rows


def evaluate_manifest(manifest: str | Path, cache_dir=None, out_dir=None) -> dict:
    """Score every copy, write one JSON each, return the means and the per-provider offsets."""
    rows = read_manifest(manifest)
    per_copy, pooled = {}, {}
    for row in rows:
        report = score_copy(read_gt(row["gt"]), load_p_dead(row["copy"], cache_dir))
        write_report(report, row["copy"], out_dir)
        per_copy[row["copy"]] = report
        pool = pooled.setdefault(row["provider"], {"onset": [], "end": []})
        pool["onset"] += report["_onset"]
        pool["end"] += report["_end"]
    reports = list(per_copy.values())
    return {
        "n_copies": len(rows),
        "n_games": len({str(r["gt"]) for r in rows}),
        "alive_dead_f1": float(np.mean([r["alive_dead_f1"] for r in reports])),
        "d_interval_f1": float(np.mean([r["d_interval"]["mean_f1"] for r in reports])),
        "ece": float(np.mean([r["ece"] for r in reports])),
        "by_provider": {p: {"onset": summarise(v["onset"]), "end": summarise(v["end"])}
                        for p, v in sorted(pooled.items())},
    }


def experiments_row(summary: dict, change: str, date: str | None = None) -> str:
    """One experiments.csv line (its header: experiments.csv:1); dead-ball scores go in notes (decision (c))."""
    def fmt(x):
        return "NA" if x is None else f"{x:+.0f}"
    notes = [f"alive_dead_f1={summary['alive_dead_f1']:.3f}", f"dint_f1={summary['d_interval_f1']:.3f}",
             f"ece={summary['ece']:.3f}"]
    for provider, s in summary["by_provider"].items():
        notes.append(f"{provider}_onset_med={fmt(s['onset']['median'])}f")
        notes.append(f"{provider}_end_med={fmt(s['end']['median'])}f")
    fields = [date or _dt.date.today().isoformat(), "P3a", change, "", "",
              str(summary["n_games"]), str(summary["n_copies"]), "; ".join(notes)]
    buffer = __import__("io").StringIO()
    csv.writer(buffer, lineterminator="").writerow(fields)
    return buffer.getvalue()


# --- command line --------------------------------------------------------------------------------

def _print_report(name: str, report: dict) -> None:
    d = report["d_interval"]
    print(f"{name}: alive_dead_f1 {report['alive_dead_f1']:.3f} | D-interval F1 {d['mean_f1']:.3f} "
          f"({d['n_gt']} GT D runs, {d['n_pred']} predicted) | ECE {report['ece']:.3f}")
    for side in ("onset", "end"):
        s = report[side]
        if s["n"]:
            print(f"  {side} offset (frames, + = late): median {s['median']:+.0f}, p90 |error| "
                  f"{s['p90_abs']:.0f}, over {s['n']} paired runs")
        else:
            print(f"  {side} offset: no paired runs")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--game", help="copy name: reads $MTAH_CACHE/<copy>/p_dead.npy")
    mode.add_argument("--self-test", action="store_true", help="score --gt against itself and a shifted copy")
    mode.add_argument("--all", metavar="MANIFEST", help="CSV with columns copy,gt,provider")
    parser.add_argument("--gt", help="frame,label CSV (needed with --game and --self-test)")
    parser.add_argument("--change", default="", help="--all: text for the experiments.csv 'change' column")
    parser.add_argument("--out", help="report folder (default $MTAH_CACHE/reports/deadball)")
    args = parser.parse_args(argv)

    if args.all:
        summary = evaluate_manifest(args.all, out_dir=args.out)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print("\nexperiments.csv row (paste it yourself if this run is worth keeping):")
        print(experiments_row(summary, args.change))
        return 0
    if not args.gt:
        parser.error("--gt is required with --game and --self-test")
    gt = read_gt(args.gt)
    if args.self_test:
        problems = self_test(gt)
        print("SELF-TEST PASSED" if not problems else "SELF-TEST FAILED:\n  " + "\n  ".join(problems))
        return 1 if problems else 0
    report = score_copy(gt, load_p_dead(args.game))
    _print_report(args.game, report)
    print(f"report: {write_report(report, args.game, args.out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
