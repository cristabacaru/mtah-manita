"""Score predictions on the external validation suite with the OFFICIAL scorer (guide section 6).

    python -m mtah.external.score_suite --results <dir> --owner Theo --change "what changed" [--bootstrap 200]

<dir> holds one CSV per suite game, named <game>__<level>.csv (e.g. dfl_J03WMX__target.csv), in
the submission format. Each is scored by evaluation/run_evaluation.py against that game's GT in
data/validation/<game>__<level>/. Prints a per-game table and the mean per level with a 95% CI from
resampling games, writes <dir>/suite_scores.json and appends one row to experiments.csv
(date, owner, change, spellF1_mean, macroF1_mean, games, noise_levels, notes). This is the number
everyone tunes on (never the public window).

    python -m mtah.external.score_suite --make-baseline <dir> [--window 50]

writes starter/baseline.py's prediction (unchanged code, run on each suite ATD) for every game:
the simulator calibration check. Real match: spellF1 0.243, macroF1 0.672.
"""

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter")]

from mtah import config  # noqa: E402
from mtah.external.build_suite import load_manifest, resolve  # noqa: E402

EXPERIMENTS = str(config.EXPERIMENTS_CSV)
COLUMNS = ["date", "owner", "change", "spellF1_mean", "macroF1_mean", "games", "noise_levels",
           "notes"]
KEEP = ("spell_f1_mean", "macro_f1", "poss", "n_gt_spells", "alive_dead_f1", "team_acc_gt_alive",
        "spell_bootstrap_low", "spell_bootstrap_high", "mean_latency")


def score_one(pred_csv, gt_csv, bootstrap):
    tmp = tempfile.mkdtemp(prefix="mtah_score_")
    try:
        res = os.path.join(tmp, "results")
        os.makedirs(res)
        shutil.copy(pred_csv, os.path.join(res, "pred.csv"))
        subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                        "--gt", gt_csv, "--results", res, "--out", tmp, "--no-detail",
                        "--bootstrap", str(bootstrap)], check=True, capture_output=True, text=True)
        with open(os.path.join(tmp, "scoreboard.csv")) as fh:
            row = next(csv.DictReader(fh))
        if row.get("status") != "scored":
            raise RuntimeError("{}: {}".format(pred_csv, row.get("reason")))
        return {k: (float(row[k]) if row.get(k) not in (None, "") else None) for k in KEEP}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ci(values, rng, draws=2000):
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    m = rng.choice(v, size=(draws, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def log_experiment(owner, change, rows, summary, manifest):
    """One row in the team's experiments.csv format; per-level numbers go in notes."""
    new = not os.path.exists(EXPERIMENTS) or os.path.getsize(EXPERIMENTS) == 0
    levels = sorted(summary)
    notes = "; ".join("{} spellF1 {:.4f} [{:.4f}, {:.4f}] macroF1 {:.4f}".format(
        lv, *summary[lv]["spell"], summary[lv]["macro"][0]) for lv in levels)
    if manifest.get("target_provisional", True) or not manifest.get("sim_params_fitted", False):
        notes += "; PROVISIONAL simulator"
    with open(EXPERIMENTS, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(COLUMNS)
        w.writerow([datetime.date.today().isoformat(), owner, change,
                    round(float(np.mean([r["spell_f1_mean"] for r in rows])), 4),
                    round(float(np.mean([r["macro_f1"] for r in rows])), 4),
                    len({r["fold"] for r in rows}), "/".join(levels), notes])


def score(results, owner, change, bootstrap, levels=None, log=True):
    manifest = load_manifest()
    games = [g for g in manifest["games"] if levels is None or g["level"] in levels]
    rows = []
    for g in games:
        pred = os.path.join(results, g["name"] + ".csv")
        if not os.path.exists(pred):
            continue
        r = score_one(pred, resolve(g["gt"]), bootstrap)
        r.update(name=g["name"], level=g["level"], fold=g["fold"])
        rows.append(r)
        print("{:<34} spellF1 {:.4f}  macroF1 {:.4f}  poss {:>5.0f} / {:<4.0f} aliveF1 {:.3f}"
              .format(g["name"], r["spell_f1_mean"], r["macro_f1"], r["poss"], r["n_gt_spells"],
                      r["alive_dead_f1"]), flush=True)
    if not rows:
        raise SystemExit("no predictions found in {}".format(results))
    rng = np.random.default_rng(config.SEED)
    summary = {}
    print()
    for level in sorted({r["level"] for r in rows}):
        sel = [r for r in rows if r["level"] == level]
        sp = ci([r["spell_f1_mean"] for r in sel], rng)
        mf = ci([r["macro_f1"] for r in sel], rng)
        summary[level] = {"n": len(sel), "spell": sp, "macro": mf}
        print("{:<8} n={:<2} spellF1 {:.4f} [{:.4f}, {:.4f}]   macroF1 {:.4f} [{:.4f}, {:.4f}]"
              .format(level, len(sel), *sp, *mf))
    if log:
        log_experiment(owner, change, rows, summary, manifest)
    with open(os.path.join(results, "suite_scores.json"), "w") as fh:
        json.dump({"rows": rows, "summary": summary}, fh, indent=1)
    return rows, summary


def make_baseline(out, window):
    import baseline
    os.makedirs(out, exist_ok=True)
    for g in load_manifest()["games"]:
        raw, meta = resolve(g["atd_raw"]), resolve(g["atd_meta"])
        team = baseline.team_of_each_track(meta)
        with open(os.path.join(resolve(g["folder"]), "game.json")) as fh:
            pitch = tuple(json.load(fh)["pitch"])
        codes = baseline.nearest_player_labels(raw, team, g["n"], pitch)
        labels = baseline.smooth(codes, window)
        # like the official baseline submission: label every frame (X only outside the halves)
        inside = np.zeros(g["n"], bool)
        for a, b in g["halves"]:
            inside[a:b] = True
        labels = np.where(inside, labels, "X")
        baseline.write_submission(os.path.join(out, g["name"] + ".csv"), labels)
        print("baseline", g["name"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--owner", default="")
    ap.add_argument("--change", default="")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--levels", nargs="+")
    ap.add_argument("--no-log", action="store_true", help="do not append to experiments.csv")
    ap.add_argument("--make-baseline")
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    if args.make_baseline:
        make_baseline(args.make_baseline, args.window)
        return
    results = os.path.abspath(args.results)
    score(results, args.owner, args.change or os.path.basename(results.rstrip("/")),
          args.bootstrap, args.levels, log=not args.no_log)


if __name__ == "__main__":
    main()
