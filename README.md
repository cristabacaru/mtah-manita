# MTAH 2027: Ball possession (team `<team_name>`)

> Replace `<team_name>` everywhere with the exact team name all members gave on the registration
> form. The scorer names our row on the board after the CSV file name.

This is our entry for More Than A Hack 2027, Phase 1. We label each of the 199,319 frames of the
BAR–ATH tactical video (25 fps) as `H` / `A` / `D` / `X`. Entries are ranked on **spellF1**. The floor to
beat is `baseline_smooth50` at 0.2433.

**Deadline: Nov 17, 2026. We submit on Nov 16.**

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

# 3. Cache folder, outside the repo: parsed tracking + every array our modules produce.
#    It is derived match data, so it must never sit in the repo. Safe to delete; it is rebuilt.
export MTAH_CACHE=~/.cache/mtah2027         # add this line to your ~/.zshrc
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

Items marked *(planned)* do not exist yet. Their owner creates them.

```
mtah/                     OUR PIPELINE (all new work goes here)
├─ config.py              every setting and seed                                P1 (planned)
├─ data.py                ATD -> per-half .npy arrays (metres, video frames)     P1 (planned)
├─ evaluate.py            score(labels, gt) wrapper, experiments.csv logging     P1 (planned)
├─ submit.py              writes <team_name>.csv (X outside graded halves) + sha256   P1 (planned)
├─ external/              DFL/IDSSE + Metrica loaders, same schema as data.py    P2 (planned)
├─ simulate.py            ball-dropout / noise simulator for external games      P2 (planned)
├─ deadball/              dead-ball module                                       P3
│  ├─ signals.py          ball-gap + motion-collapse signals (all frames)        P3a Cristiana (planned)
│  ├─ lines.py            ball distance past / to the touchline and goal line    P3a Cristiana (planned)
│  ├─ combine.py          all evidence -> final p_dead[t]                        P3a Cristiana (planned)
│  ├─ evaluate.py         D-interval F1 + boundary offset on external games      P3a Cristiana (planned)
│  ├─ tags.py             Smart Tagging -> tags.json (frames, type, H/A team)    P3b Alexia (planned)
│  └─ restarts.py         kick-off + restart detection, restart team -> anchors  P3b Alexia (planned)
├─ ball.py                ball-gap bridging (interpolation, RTS smoother)        P4 (planned)
├─ features.py, model.py  team-possession features + LightGBM -> p_home[t]      P4 (planned)
├─ decode.py              HSMM/Viterbi decoder, boundary snapping                P5
└─ optuna_tune.py         decoder tuning scaffold                                P5
run.sh                    one command: raw data -> final CSV              P1 (planned)
experiments.csv           one row per experiment (who, what, external scores)
docs/TECHNICAL.md         technical doc, one section per owner (skeleton)

evaluation/               official scorer (copied from the challenge repo, do not edit)
starter/                  official loaders + baseline (do not edit, import from them)
games.py                  official game metadata and asset paths (do not edit)
research/checks/          exploratory scripts, aggregates only
tools/                    Metrica sample-game converters + GT viewer   (to be added)
data/                     match data only (tactical/, broadcast/, validation/), NEVER committed
```

Files from the official repo (`evaluation/`, `starter/`, `games.py`) are MIT licensed. See
`LICENSE-metrica`. Leave them unchanged so our scores match the official scorer. If you need
different behaviour, wrap the function in `mtah/`.

### Module contract (proposal: confirm with the whole team, then remove this note)

Every module works on **per-half arrays indexed by video frame**. External games must have the
exact same shape and schema as the target game, so every module runs on both unchanged.

| Producer | Output (in `$MTAH_CACHE/<game>/`) | Meaning | Consumers |
|---|---|---|---|
| P3a `deadball/combine.py` | `p_dead.npy`, shape `(n,)`, float in [0, 1] | probability the ball is dead | P5 |
| P3b `deadball/restarts.py` | `anchors.json`: list of `{frame, type, team, confidence}` | detected restarts (kickoff, throw-in, corner, goal kick, free kick) | P5 |
| P3b `deadball/tags.py` | `tags.json`: list of `{start_frame, end_frame, type, team}` | Smart Tagging phases on video frames, team as `H`/`A` | P5 |
| P4 `model.py` | `p_home.npy`, shape `(n,)`, float in [0, 1] | P(home in possession \| ball alive) | P5 |
| P5 `decode.py` | `labels.npy`, shape `(n,)`, values `H`/`A`/`D` | final labels | `submit.py` |

`p_dead` uses tracking only, never anchors or tags, so it can be validated on external games (they
have no Smart Tagging). The decoder (P5) alone places D boundaries, using anchors and tags. P3b owns
the `anchors.json` and `tags.json` formats; `docs/deadball_SPEC.md` covers `p_dead` only.

As long as these shapes hold, P3, P4 and P5 can work in parallel against stub inputs (e.g.
`p_dead` = ball-missing-for-3-s mask, `p_home` = nearest-team baseline).

---

## Day-to-day usage

### Scoring on external games (our development metric)
```bash
python evaluation/run_evaluation.py \
    --gt data/validation/<game>/<game>_gt.csv \
    --results <dir with your csv files> \
    --out $MTAH_CACHE/reports --bootstrap 1000
```
- Always pass `--gt`. Its default (`public`) looks for the public window GT, which is not in
  this repo.
- `--results` scores every CSV in that folder.
- `--out` keeps `scoreboard.csv` and the reports out of the repo.
- The external GT files (`data/validation/game{1,2}/`) come from P2's converters in `tools/`.
Report the **mean spellF1 over all held-out external games and noise levels**, and add a row to
`experiments.csv`: date, owner, change, spellF1_mean, macroF1_mean, games, noise_levels, notes. Keep a component only if
it adds more than ~0.02 with no bootstrap overlap.

### Research checks (aggregates only)
```bash
python research/checks/c3_dead_ball.py     # dead-ball signals from player speed
python research/checks/c5_restarts.py      # kick-off detector, restart templates
```
See `research/checks/README.md` for the full list.

### Before submitting
```bash
bash run.sh                                           # -> <team_name>.csv
python evaluation/validate_submission.py <team_name>.csv
shasum -a 256 <team_name>.csv                         # run twice: hashes must match
git ls-files | grep '^data/'                          # only data/README.md
```
Submit through the official Google form, one response per team (the last one before the deadline
counts). You need:
- the team name;
- `<team_name>.csv`;
- the public URL of this repo (switch it to Public first);
- the technical document as `<team_name>.pdf` or `.md`.

You need a Google account to upload. There is no leaderboard and no feedback before the deadline.

---

## Team

| Role | Owner |
|---|---|
| P1: Platform and submission lead | Alexia |
| P2: External data and simulator | Theo |
| P3a: Dead ball (p_dead signal, combination, evaluation) | Cristiana |
| P3b: Restarts, Smart Tagging, anchors | Alexia |
| P4: Team-possession model | Bianca |
| P5: Decoder, tuning and error analysis | Andra |


