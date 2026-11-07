# MTAH 2027: per-frame ball possession (H/A/D) for BAR–ATH, ranked on spellF1

**Before any task, read `docs/LABELLING_GUIDE.md`.** It defines the labels, the data sources, the
data schema, the GT format and the evaluation protocol, with the reason for every rule. The
README covers setup and the module contract.

## Commands
- Env: `source .venv/bin/activate` (Python 3.12; exact versions in `requirements-lock.txt`)
- Smoke test: `python starter/baseline.py --window 50 --out /tmp/mtah && python evaluation/validate_submission.py /tmp/mtah/baseline_smooth50.csv`
- Score one game: `python evaluation/run_evaluation.py --gt data/validation/<game>/<game>_gt.csv --results <dir> --out $MTAH_CACHE/reports --bootstrap 1000`
- Research checks: `python research/checks/c3_dead_ball.py` (others in `research/checks/README.md`)

## Rules that decide whether the submission is valid (details and reasons in the guide, §2)
- No human labels or human-made data about BAR–ATH, and no constants tied to its frames. Detect
  everything with code.
- Never tune or train on the public GT (`evaluation/gt/`). Develop on external games only.
- Output must be reproducible byte for byte: fixed seeds, deterministic settings.
- Never commit anything under `data/` or any frame-level array. Outputs go to `$MTAH_CACHE/<game>/`.

## Conventions
- New code goes in `mtah/`. Don't edit `evaluation/`, `starter/` or `games.py` (official). Wrap them.
- Every setting lives in `mtah/config.py`, in the owning module's section.
- Data follows the schema in guide §5: metres, `NaN` for missing, H attacks towards +x, per-half
  processing, per-frame team aggregates (never follow a player column over time).
- Report results as the mean over external games × noise levels and log a row in `experiments.csv`.

## Gotchas
- kloppy `sportec.load_open_tracking_data` drops dead frames unless `only_alive=False`.
- `X` on graded frames never helps: the scorer forward-fills it for spellF1 and counts it as a miss
  on macroF1.
- spellF1 noise band is about 0.04 on one game; don't trust single-game differences.
- Before calling work done, show the command you ran and its output, and go through the guide's §7
  checklist.
