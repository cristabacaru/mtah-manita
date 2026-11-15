"""Score restart anchors against the true labels of P2's external games (P3b).

    python -m mtah.deadball.evaluate_restarts --suite [--levels target heavy] --change "what changed"

A true restart is a frame whose label is H or A right after a D, inside one graded run of the GT
(pm.valid_runs: never across half-time). For each game the anchors from restarts.detect are compared
with those frames:
- recall, precision, F1: one-to-one matching, closest pairs first, within each RESTARTS.EVAL_TOL_S;
- offset: anchor frame minus true frame, in frames (late = positive), on the loose matches;
- team: the anchor's team against the label of the true restart, on the loose matches;
- by template: per (source, type), the share of anchors within the tolerance of ANY true restart,
  against the share of graded frames that are (chance). Run with filter_still=False, so the still-ball
  candidates option B drops are measured too.

Limits. The GT has only H/A/D, so restart TYPES cannot be scored, only when and who. Ball dropout on
the suite is simulated, so a good score for "the ball reappears" also depends on the simulator.

Results are the mean over games per noise level (guide section 6). Prints aggregates only, and a
ready-to-paste experiments.csv row; never writes experiments.csv.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import sys

import numpy as np

from mtah.config import RESTARTS as R
from mtah.deadball import restarts
from mtah.deadball.evaluate import pm, read_gt, read_suite

LETTER = {pm.H: "H", pm.A: "A"}


def true_restarts(codes: np.ndarray) -> list[tuple[int, str]]:
    """[(frame, "H"|"A")]: every frame that is H or A right after a D, inside one graded run."""
    out = []
    for start, stop in pm.valid_runs(codes):
        run = codes[start:stop]
        for i in np.flatnonzero((run[:-1] == pm.D) & np.isin(run[1:], (pm.H, pm.A))):
            out.append((start + int(i) + 1, LETTER[int(run[i + 1])]))
    return out


def match(anchor_frames, true_frames, tol: int) -> list[tuple[int, int]]:
    """One-to-one (anchor index, true index) pairs within tol frames, closest pairs first."""
    pairs = sorted((abs(int(a) - int(t)), i, j) for i, a in enumerate(anchor_frames)
                   for j, t in enumerate(true_frames) if abs(int(a) - int(t)) <= tol)
    used_a, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_a and j not in used_t:
            used_a.add(i)
            used_t.add(j)
            out.append((i, j))
    return out


def _near_mask(frames, n: int, tol: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for f in frames:
        m[max(0, f - tol):f + tol + 1] = True
    return m


def score_game(anchors: list[dict], candidates: list[dict], codes: np.ndarray, fps: float) -> dict:
    """Counts for one game. `anchors` is the final output, `candidates` detect(filter_still=False)."""
    truth = true_restarts(codes)
    frames = [a["frame"] for a in anchors]
    true_frames = [t[0] for t in truth]
    tols = [int(round(s * fps)) for s in R.EVAL_TOL_S]
    report = {"n_true": len(truth), "n_anchors": len(anchors), "by_tol": {}}
    for seconds, tol in zip(R.EVAL_TOL_S, tols):
        tp = len(match(frames, true_frames, tol))
        report["by_tol"][seconds] = {
            "tp": tp,
            "recall": tp / len(truth) if truth else float("nan"),
            "precision": tp / len(anchors) if anchors else float("nan"),
            "f1": 2 * tp / (len(truth) + len(anchors)) if truth or anchors else float("nan"),
        }
    pairs = match(frames, true_frames, tols[-1])
    report["offsets"] = [frames[i] - true_frames[j] for i, j in pairs]
    named = [(anchors[i]["team"], truth[j][1]) for i, j in pairs if anchors[i]["team"] is not None]
    report["team_right"], report["team_total"] = sum(a == t for a, t in named), len(named)

    near = _near_mask(true_frames, len(codes), tols[0])
    graded = codes != pm.X
    report["chance_hits"], report["chance_total"] = int(near[graded].sum()), int(graded.sum())
    groups: dict = {}
    for a in candidates:
        g = groups.setdefault(f"{a['source']}/{a['type']}", [0, 0])
        g[0] += int(near[a["frame"]])
        g[1] += 1
    report["groups"] = groups
    bands: dict = {}
    for a in anchors:
        b = bands.setdefault(f"{a['confidence']:.3f}", [0, 0])
        b[0] += int(near[a["frame"]])
        b[1] += 1
    report["bands"] = bands
    return report


def _add(pool: dict, new: dict) -> None:
    for key, (hits, total) in new.items():
        p = pool.setdefault(key, [0, 0])
        p[0] += hits
        p[1] += total


def summarise_level(reports: list[dict]) -> dict:
    out = {"n_games": len(reports), "mean_true": float(np.mean([r["n_true"] for r in reports])),
           "mean_anchors": float(np.mean([r["n_anchors"] for r in reports])), "by_tol": {}}
    for seconds in R.EVAL_TOL_S:
        out["by_tol"][seconds] = {m: float(np.nanmean([r["by_tol"][seconds][m] for r in reports]))
                                  for m in ("recall", "precision", "f1")}
    offsets = [o for r in reports for o in r["offsets"]]
    out["offset_median"] = float(np.median(offsets)) if offsets else None
    out["offset_p90_abs"] = float(np.percentile(np.abs(offsets), 90)) if offsets else None
    right, total = sum(r["team_right"] for r in reports), sum(r["team_total"] for r in reports)
    out["team_accuracy"] = right / total if total else None
    out["chance"] = sum(r["chance_hits"] for r in reports) / max(1, sum(r["chance_total"] for r in reports))
    out["groups"], out["bands"] = {}, {}
    for r in reports:
        _add(out["groups"], r["groups"])
        _add(out["bands"], r["bands"])
    return out


def evaluate_rows(rows: list[dict], load_game=None) -> dict:
    """Mean per noise level over P2's suite rows ({copy, gt, level, source})."""
    if load_game is None:
        from mtah.data import load as load_game
    levels: dict = {}
    for row in rows:
        game = load_game(row["copy"])
        codes = read_gt(row["gt"])
        if len(codes) != int(game["meta"]["n_frames"]):
            raise ValueError(f"{row['copy']}: GT has {len(codes)} frames, the game {game['meta']['n_frames']}")
        report = score_game(restarts.detect(game), restarts.detect(game, filter_still=False), codes,
                            float(game["meta"]["fps"]))
        levels.setdefault(row["level"], []).append(report)
    return {"n_games": len({r["source"] for r in rows}),
            "levels": {name: summarise_level(reps) for name, reps in sorted(levels.items())}}


def experiments_row(summary: dict, change: str, date: str | None = None) -> str:
    """One experiments.csv line in the team's layout; the scores go in notes."""
    strict, loose = R.EVAL_TOL_S
    blocks = []
    for level, s in summary["levels"].items():
        blocks.append(f"{level} f1@{strict:g}s={s['by_tol'][strict]['f1']:.3f} f1@{loose:g}s={s['by_tol'][loose]['f1']:.3f} "
                      f"recall@{loose:g}s={s['by_tol'][loose]['recall']:.3f} precision@{loose:g}s={s['by_tol'][loose]['precision']:.3f} "
                      f"offset_med={s['offset_median']}f team_acc={s['team_accuracy']}")
    fields = [date or _dt.date.today().isoformat(), "P3b", change, "", "", str(summary["n_games"]),
              "/".join(summary["levels"]), "; ".join(blocks)]
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="").writerow(fields)
    return buffer.getvalue()


def _print(summary: dict) -> None:
    for level, s in summary["levels"].items():
        print(f"\n== {level}: {s['n_games']} games, per game {s['mean_true']:.0f} true restarts, "
              f"{s['mean_anchors']:.0f} anchors ==")
        for seconds, m in s["by_tol"].items():
            print(f"  within {seconds:g} s: recall {m['recall']:.3f}  precision {m['precision']:.3f}  f1 {m['f1']:.3f}")
        print(f"  offset (anchor - truth): median {s['offset_median']} frames, p90 of |offset| {s['offset_p90_abs']} frames")
        print(f"  restarting team right: {s['team_accuracy']}")
        strict = R.EVAL_TOL_S[0]
        print(f"  within {strict:g} s of a true restart (chance {s['chance']:.1%}), all candidates incl. dropped still-ball:")
        for key, (hits, total) in sorted(s["groups"].items()):
            print(f"    {key:<30} {total:>6}  {hits / total:6.1%}")
        print("  same, final anchors by confidence:")
        for key, (hits, total) in sorted(s["bands"].items()):
            print(f"    {key:<30} {total:>6}  {hits / total:6.1%}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", action="store_true", required=True, help="every game of P2's suite_manifest.json")
    parser.add_argument("--levels", nargs="+", help="only these noise levels (light target heavy)")
    parser.add_argument("--change", default="", help="text for the experiments.csv 'change' column")
    args = parser.parse_args(argv)
    summary = evaluate_rows(read_suite(levels=args.levels))
    _print(summary)
    print("\nexperiments.csv row (paste it yourself if this run is worth keeping):")
    print(experiments_row(summary, args.change))
    return 0


if __name__ == "__main__":
    sys.exit(main())
