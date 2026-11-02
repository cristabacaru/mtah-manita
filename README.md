# MTAH 2027: Ball possession (team Manita)

This is our entry for More Than A Hack 2027, Phase 1. We label each of the 199,319 frames of the
BAR–ATH tactical video as `H` / `A` / `D` / `X`. Entries are ranked on **spellF1**. The floor to
beat is `baseline_smooth50` at 0.2433.

**Deadline: Nov 17, 2026. We submit on Nov 16.**

---

## Ground rules (read these first)

1. **Never commit match data.** `data/*` is in `.gitignore` (only `data/README.md` is tracked).
   Before you push, run `git ls-files | grep '^data/'`. It must print only `data/README.md`. This
   repo becomes public at submission, and once data is in the git history, deleting the file does
   not remove it.
2. **No match data goes to a cloud LLM** (ChatGPT, Claude, Copilot chat, …). Asking an assistant to
   write code is fine. Pasting frame-level outputs, positions or labels is not. Share only
   aggregates: scores, counts and shares.
3. **Never tune or train on the public 5-minute GT.** It is not in this repo on purpose. We use it
   once, at freeze (Nov 14), only as a smoke test.
4. **No hand labels or human-made data about this game.** Labelled data from *other* games is
   allowed in any amount (Metrica samples, DFL/IDSSE).
5. **The output must be deterministic.** Fix the seeds, never hard-code frame numbers, and keep
   every setting in `mtah/config.py`.

---

## Setup (once per machine)

```bash
git clone https://github.com/cristabacaru/mtah-manita.git
cd mtah-manita

# 1. Data: unpack the enrolment bundle into data/ (see data/README.md).
#    You need data/tactical/ (required) and data/broadcast/ (optional).

# 2. Environment: Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt        # exact versions the team uses

# 3. Cache for parsed arrays, kept outside the repo
export MTAH_CACHE=~/.cache/mtah2027         # add this to your ~/.zshrc
```

**Check that it works:**
```bash
python starter/baseline.py --window 50 --out /tmp/mtah
python evaluation/validate_submission.py /tmp/mtah/baseline_smooth50.csv   # -> "Valid."
```
The baseline run takes about 2 s. If `validate_submission.py` prints "Valid.", your data and
environment are correct.

---

## Repository layout: where to build

```
mtah/                     OUR PIPELINE (all new work goes here)
├─ config.py              every setting and seed                                P1
├─ data.py                ATD -> per-half .npy arrays (metres, video frames)     P1
├─ evaluate.py            score(labels, gt) wrapper, experiments.csv logging     P1
├─ submit.py              writes <team>.csv (X outside graded halves) + sha256   P1
├─ external/              DFL/IDSSE + Metrica loaders, same schema as data.py    P2
├─ simulate.py            ball-dropout / noise simulator for external games      P2
├─ deadball.py            p_dead[t], restart anchors                             P3
├─ ball.py                ball-gap bridging (interpolation, RTS smoother)        P4
├─ features.py, model.py  team-possession features + LightGBM -> p_home[t]      P4
└─ decode.py              HSMM/Viterbi decoder, boundary snapping                P5
run.sh                    one command: raw data -> final CSV                     P1
experiments.csv           one row per experiment (who, what, external scores)
docs/TECHNICAL.md         technical doc, one section per owner

evaluation/               official scorer (copied from the challenge repo, do not edit)
starter/                  official loaders + baseline (do not edit, import from them)
games.py                  official game metadata and asset paths (do not edit)
research/checks/          exploratory scripts behind our verification report
tools/                    Metrica sample-game converters + GT viewer
data/                     match data and caches, NEVER committed
```

Files from the official repo (`evaluation/`, `starter/`, `games.py`) are MIT licensed. See
`LICENSE-metrica`. Leave them unchanged so our scores match the official scorer. If you need
different behaviour, wrap the function in `mtah/`.

### Module contract (agreed on day 1)

Every module works on **per-half arrays indexed by video frame**. External games must have the
exact same shape and schema as the target game, so every module runs on both unchanged.

| Producer | Output (in `data/cache/<game>/`) | Meaning |
|---|---|---|
| P3 `deadball.py` | `p_dead.npy`, shape `(n,)`, float in [0, 1] | probability the ball is dead |
| P3 `deadball.py` | `anchors.json`: list of `{frame, type, team, confidence}` | detected restarts (kickoff, throw-in, corner, goal kick, free kick) |
| P4 `model.py` | `p_home.npy`, shape `(n,)`, float in [0, 1] | P(home in possession \| ball alive) |
| P5 `decode.py` | `labels.npy`, shape `(n,)`, values `H`/`A`/`D` | final labels |

As long as these shapes hold, P3, P4 and P5 can work in parallel against stub inputs (e.g.
`p_dead` = ball-missing-for-3-s mask, `p_home` = nearest-team baseline).

---

## Day-to-day usage

### Branches and pull requests
```bash
git checkout main && git pull
git checkout -b <area>/<short-name>      # e.g. deadball/kickoff-detector
# ... work, commit often ...
git push -u origin <area>/<short-name>   # then open a pull request on GitHub
```
- Never push directly to `main`. Merge through a pull request that another teammate reviews.
- `main` must always produce a valid submission (`bash run.sh` then `validate_submission.py`).

### Scoring on external games (our development metric)
```bash
python evaluation/run_evaluation.py \
    --gt data/validation/<game>/<game>_gt.csv \
    --results <dir with your csv> --bootstrap 1000
```
Report the **mean spellF1 over all held-out external games and noise levels**, and add a row to
`experiments.csv`: date, owner, branch, change, spellF1, macroF1, notes. Keep a component only if
it adds more than ~0.02 with no bootstrap overlap.

### Research checks (aggregates only)
```bash
python research/checks/c3_dead_ball.py     # dead-ball signals from player speed
python research/checks/c5_restarts.py      # kick-off detector, restart templates
```
See `research/checks/README.md` for the full list.

### Before submitting
```bash
bash run.sh                                           # -> manita.csv
python evaluation/validate_submission.py manita.csv
shasum -a 256 manita.csv                              # run twice: hashes must match
git ls-files | grep '^data/'                          # only data/README.md
```

---

## Team

| Role | Owner |
|---|---|
| P1: Platform and submission lead | _tbd_ |
| P2: External data and simulator | _tbd_ |
| P3: Dead-ball module | Cristiana |
| P4: Team-possession model | _tbd_ |
| P5: Decoder, tuning and error analysis | _tbd_ |

## Timeline

| Dates | Milestone |
|---|---|
| Nov 1–2 | Skeleton, data layer, scorer wrapper, v0 quick win (golden fallback) |
| Nov 3–5 | v1 rules pipeline end to end, external validation suite live |
| Nov 6–10 | v2: LightGBM team and dead-ball models, HSMM, tag-driven costs |
| Nov 11–13 | Joint tuning on external games, ablations |
| Nov 14 | **Freeze**, one smoke test on the public window |
| Nov 15–16 | Determinism check, clean-machine reproduction, technical doc, **submit Nov 16** |
