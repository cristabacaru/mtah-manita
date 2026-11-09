# Worklog (P2: external data and simulator)

> Copied from Theo's working repo on 2026-11-08. Paths below are the ones used then
> (`data/cache/...`, `data/validation/suite/...`); in this repo they come from `config.EXTERNAL`
> (see `mtah/external/README.md`).

A record of every step taken on this project: what was done, why, how to reproduce it, and what
we learned. Newest entries go at the bottom. This also feeds the technical document required for
the submission (see `SUBMISSION.md`).

> **Data rule we follow:** the competition match data (video, ATD, Smart Tagging, homography) must
> not be uploaded to a cloud LLM, including the coding assistant. Scripts that read it are run
> locally and only conclusions are discussed. Public data (the 5-minute public GT, the Metrica
> sample data) is not restricted.

---

## Step 1: Read the repository and write a project guide (2026-10-29)

**What:** read every document and script in the repository (`README`, `CHALLENGE`, `DATA`,
`EVALUATION`, `SUBMISSION`, `RULES`, `FAQ`, `OVERVIEW`, `possession_metrics.py`,
`run_evaluation.py`, `baseline.py`, the loaders and the three notebooks), and ran the scorer on
the four committed baselines.

**Output:** `C:\Users\User\Desktop\BarcaHack_Project_Guide.md`, which covers the task, the data,
every metric with worked examples, the rules, the baseline results and an ML glossary.

**Reproduce:**
```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

**Key facts:**
- The ranking metric is `spellF1`: possessions matched one-to-one by temporal IoU, F1 averaged
  over IoU 0.3–0.7. `macroF1` is the tie-break.
- The floor to beat is `baseline_smooth50`, spellF1 0.2433 on the full match.
- On the public 5 minutes the baseline labels 902 of 1,469 dead-ball frames as `H`. Dead-ball
  detection is its weakest part.
- Tuning on the public 5 minutes picks the wrong smoothing window, and it isn't allowed anyway.

---

## Step 2: Ground-truth viewer (2026-10-29)

**What:** `tools/view_gt.py` plays the tactical video with the GT label (and optionally a
submission's label) drawn on each frame, the tracked players projected through the homography,
and a colour-coded timeline of the window.

**Why:** to understand the labelling conventions (when `D` starts and ends, when possession
changes) by watching the 17 public possessions. Watching the video is allowed; turning it into
labels is not.

**Reproduce:**
```bash
pip install opencv-python
python3 tools/view_gt.py                                   # public 5 minutes
python3 tools/view_gt.py --pred evaluation/results/baseline_smooth50.csv
python3 tools/view_gt.py --list                            # GT segments with video timestamps
```
Keys: `space` play/pause, `a`/`d` ±1 frame, `s`/`w` ±1 s, `z`/`c` ±10 s, `n`/`p` next/previous GT
change, `e` next disagreement with `--pred`, `1`–`4` speed, `t` tracking on/off, `q` quit.

**Changes:** the white ball marker was removed at the user's request; the bottom line still says
whether the ball is tracked or missing.

**Notes:**
- The downloaded video is named `BAR-ATH_tactical-001.mp4`, while `games.py` expects
  `BAR-ATH_tactical.mp4`. The viewer falls back to any `*tactical*.mp4`. Rename the file so the
  other scripts find it too.
- Runs under WSL (WSLg). The Qt font warnings OpenCV prints are harmless.

---

## Step 3: Dead-ball conventions observed on the public 5 minutes (2026-10-29)

**What:** watched the 4 `D` segments of the public window with the viewer.

| # | stoppage | observation |
|---|---|---|
| 1 | offside | the offside pass was around frame 12700, but `D` starts at 12851 (~6 s later, at the whistle or flag). Until then the GT keeps the attacking team's possession. |
| 2 | foul | `D` starts at the whistle; players stop roughly 10 frames after `D` starts |
| 3 | corner | `D` from the frame the ball crosses the goal line (even while it is still bouncing) until the corner is kicked |
| 4 | goal kick | `D` from the frame the ball crosses the goal line until the keeper kicks it from the 6-yard box |

**Design decisions taken from this** (full version in section 12 of the guide):
- Two detectors start `D`: **geometric** (the ball crosses a line) and **behavioural** (both teams
  slow down together after a whistle). Offside doesn't need detecting as a tactical event; it's
  just a whistle.
- One detector ends `D`: a **restart from the spot the stoppage type requires** (ball still, then
  a burst of speed). Whatever happens with the ball during the stoppage is ignored.
- The stoppage type tells you which team gets the ball after the restart.
- **No typical stoppage durations**, because they are too variable (a keeper can waste 40 s).
  Only **minimum durations**, used to block impossible early restarts.
- The ~10-frame whistle lag must be **measured on validation matches**, not taken from the public
  window. The public observation is only a sanity check.

---

## Step 4: Download the Metrica public sample data (2026-10-29)

**What:** cloned the public Metrica Sports sample data: 3 anonymised matches with tracking and
human-logged events, from the same provider as the competition data.

**Why:** to build **validation matches**. A program derives `H`/`A`/`D` labels from the events,
the clean tracking is degraded to look like our ATD, and then any idea can be measured with the
official scorer instead of guessed. It can later double as training data, keeping one match held
out.

**Reproduce:**
```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data.git data/external/metrica-sample-data
```
It's about 234 MB. `data/*` is gitignored, so it can't be committed by accident.

**Licence:** the sample-data README has no formal licence. It asks users to "be responsible" and
to acknowledge the source if the data is used publicly. We acknowledge it in the technical
document. It is public, so the organisers can access it, as `RULES.md` requires.

**Contents:**

| | format | events | tracking frames | halves (frames) | ball present |
|---|---|---|---|---|---|
| Game 1 | CSV | 1,745 | 145,006 | 1–71,268 · 71,269–145,006 | 60.9% |
| Game 2 | CSV | 1,935 | 141,156 | 1–67,941 · 67,942–141,156 | 59.0% |
| Game 3 | FIFA/EPTS tracking + JSON events (same tracking format as the competition ATD) | 3,620 | — | — | — |

All are at 25 fps with 0–1 coordinates, and events are synchronised with tracking by frame number.
The event definitions are in `documentation/events-definitions.pdf`.

---

## Step 5: Inspect the events (2026-10-29)

**Reproduce:**
```bash
python3 tools/inspect_sample_data.py
```

**Event types (games 1 and 2):** `PASS`, `BALL LOST`, `RECOVERY`, `CHALLENGE`, `BALL OUT`,
`SET PIECE` (`KICK OFF`, `THROW IN`, `CORNER KICK`, `FREE KICK`, `PENALTY`, `…-RETAKEN`),
`SHOT`, `FAULT RECEIVED`, `CARD`. Game 3 adds `CARRY` (1,395 of them), which makes possession
inside a team's spell much easier to follow.

Useful subtypes for the dead-ball rules:
- `BALL LOST / OFFSIDE`, `BALL LOST / HAND BALL`, `BALL LOST / END HALF`
- `CHALLENGE / …-FAULT-WON / …-FAULT-LOST`, followed by `FAULT RECEIVED`, which is the foul
- `CHALLENGE / …-ADVANTAGE-…`, which is a foul where play continued, so no `D`
- `SHOT / …-OUT`, `SHOT / …-GOAL`, `BALL OUT / WOODWORK-GOAL`

**Worked example (game 2, frames 451–722):**

```
Away  BALL LOST  INTERCEPTION   451 → 504   Player15
Home  CHALLENGE  GROUND-WON     504
Home  RECOVERY   INTERCEPTION   504         Player3      → Home gains control
Home  BALL OUT                  504 → 534   end y = 1.01 → ball crosses the touchline at ~534
Away  SET PIECE  THROW IN       672         Player15     → restart, Away
Away  PASS                      672 → 722
```

Derived labels: `A` until 503, `H` from 504 to 533, `D` from 534 to 671, `A` from 672. The ball
track agrees: y = 0.98 at frame 525, missing from about 540 to 660, and back at y = 1.004 (on the
touchline) at 675.

**Important finding: the ball goes missing for different reasons in the two datasets.**
- In the sample data the ball is missing in long stretches: median gap 127 frames, p90 959
  frames. The example above shows why. **The ball is removed while it is out of play**, so
  "missing = dead" is almost true there.
- In the competition ATD the ball is missing mostly **during play** (occlusions, aerial balls), on
  41% of graded frames.
- Consequence for the noise simulation: removing the ball randomly is not enough. We also have to
  **put a plausible ball back during stoppages** (for example held at the restart spot or carried
  by a player), otherwise any method that learns "ball missing → `D`" will look far better on
  validation than on the real match.

---

## Step 6: Event → label converter and label checks (2026-10-29)

**What:** three new files in `tools/`:

| file | what it does |
|---|---|
| `sample_data.py` | loads games 1 and 2 (events + clean tracking for both teams) as arrays; shared by the other two |
| `events_to_labels.py` | the converter: a state machine over the events that writes `H`/`A`/`D`/`X` per frame in the competition format |
| `check_labels.py` | tests the derived labels against the clean tracking, which is independent of the events, and flags disagreements |

**Reproduce:**
```bash
python3 tools/events_to_labels.py     # writes data/validation/game{1,2}/...
python3 tools/check_labels.py         # writes data/validation/game{1,2}/game{N}_flags.csv
python3 evaluation/run_evaluation.py --gt data/validation/game2/game2_gt.csv --results <dir>
```

Per game, in `data/validation/game<N>/`: `game<N>_gt.csv` (the labels), `game<N>_gt_meta.json`,
`game<N>_stoppages.csv` (every stoppage with cause, restart type and restarting team),
`game<N>_notes.csv` (where the converter had to infer something) and `game<N>_flags.csv`
(the checks' output).

### The conversion rules

Built from the conventions in step 3 and the Metrica event definitions
(`documentation/events-definitions.pdf`):

- **Possession changes only when the other team takes control:** `PASS`, `SHOT`, `RECOVERY`, or
  `BALL LOST` (which asserts the team had the ball at its start). Challenges and loose balls
  change nothing.
- **The ball dies at:** the `BALL OUT` end frame; a `SHOT …-OUT/…-GOAL` end frame;
  `FAULT RECEIVED` (defined as "the referee immediately awards a free kick", so it marks the
  whistle); `BALL LOST OFFSIDE/FORCED` (end frame); `BALL LOST HAND BALL/END HALF` (start frame).
- **Ball-out stoppages are moved to the tracking crossing frame.** The logged `BALL OUT` end
  frame is a median 4–6 frames *after* the ball leaves the pitch. Our convention (step 3) is
  "`D` from the frame the ball is out", so those starts are snapped to where the clean tracking
  sees the ball leave the pitch (64/66 and 63/65 stoppages moved).
- **While dead, everything is ignored until a restart:** a `SET PIECE`, or a goal kick. Goal kicks
  are not `SET PIECE` events in this data: they are logged as `RECOVERY` (no subtype) +
  `PASS / GOAL KICK`, or `BALL LOST / GOAL KICK…`.
- **Period edges:** before the first kickoff, `X` (like the competition's first half). Before a
  later kickoff, `D` (like the competition's second half). The first frame of a later period is
  `X`, so the scorer scores the periods as separate runs.
- **Same-frame events keep the analyst's logging order** (`SET PIECE` before its `PASS`).

### Bugs found and fixed through the checks

1. **Same-frame ordering.** A throw-in's `PASS` was processed before its `SET PIECE`, which created
   hundreds of fake "pass while dead" restarts. Fixed by keeping the logging order.
2. **Late `BALL OUT` frames.** See the snapping rule above.
3. **A bare `RECOVERY` after a foul** was taken as a goal-kick restart. It now restarts play only
   after the ball went out.
4. **Keeper fetches.** In game 1's second half a bare `RECOVERY` is logged when the keeper
   *fetches* the ball, up to 30 s before the actual `GOAL KICK`. The goal kick is now the restart.

### Results after the fixes

| | game 1 | game 2 |
|---|---|---|
| possessions | 378 (mean 8.5 s, median 4.1 s) | 356 (mean 9.3 s, median 4.6 s) |
| possessions per minute of play | 3.9 | 3.8 (public competition window: 3.4) |
| `D` share of graded frames | 44.7% | 41.7% |
| stoppages | 104 | 101 |
| ball-out `D` start vs tracking crossing | median 0, 100% within 12 frames | median 0, 98% within 12 frames |
| restart at the right place | 63/69 | 63/67 |
| restarting team nearest the ball | 93/95 | 90/90 |
| nearest player on the possessing team (live, ball tracked) | 80.3% | 81.0% |
| stoppages the converter had to infer | 10 | 6 |
| flags to inspect | 61 | 42 |

The official scorer accepts the derived GT: scoring the GT against itself gives macroF1 = spellF1
= 1.0000.

**The official baseline on the clean tracking:**

| | macroF1 | spellF1 | possessions (truth) |
|---|---|---|---|
| game 1, raw | 0.8148 | 0.1228 | 2023 (378) |
| game 1, smooth ±50 | 0.8078 | 0.4312 | 482 (378) |
| game 2, raw | 0.8550 | 0.1017 | 2178 (356) |
| game 2, smooth ±50 | 0.8527 | 0.4176 | 464 (356) |

This is as expected. The labels and the tracking line up (macroF1 ~0.81–0.85), and the score is
much higher than the competition result (0.2433), because this tracking is clean and the ball
vanishes exactly when play stops. The noise simulation has to bring this down to the competition
level.

### What the remaining flags are

- **`inferred` (16):** a set piece with no stoppage logged before it. In most of these the ball is
  untracked throughout the inferred stretch (so it really was dead); the start of the stoppage is
  a guess (the last fault challenge, or the previous event's end).
- **`restart_position` (14):** mostly corners 3.5–8 m from the corner flag, where the first tracked
  ball frame is probably after the kick. Two "goal kicks" are 37–50 m from the goal line, so the
  log is probably wrong there.
- **`possession_agrees` (66), an open convention question.** Most of these are short "possessions"
  created by an **uncontrolled touch**. Examples from game 1:
  - frame 8581: a header (`RECOVERY` → `BALL LOST HEAD` straight after). The team "has the ball"
    for 2.8 s while it is in the air.
  - frame 1374: an interception that goes straight out (`RECOVERY INTERCEPTION` → `BALL OUT`). It
    gives 2 s of possession.

  The challenge rules say a touch without control, or an aerial duel won without control, is not a
  change of possession. The Metrica event log counts it as one. **Open decision:** should the
  converter require control (for example: a recovery followed immediately by losing the ball, or
  by the ball going out, doesn't change possession)? That would merge many short possessions and
  bring the rate (3.8–3.9 per minute) closer to the public window's 3.4 per minute.

---

## Next steps

1. **Decide the uncontrolled-touch rule** (above), apply it to the converter, rerun the checks.
2. Optionally, a 2D pitch viewer for the flagged moments (the sample data has no video).
3. **Noise simulation**: ball removed in bursts during play, a synthetic ball during stoppages,
   fragmented tracks, no identities, written in the competition ATD format.
4. **Calibration check**: the official baseline on the degraded matches should land near macroF1
   ~0.65 and spellF1 ~0.2.
5. Score with `python evaluation/run_evaluation.py --gt <validation_gt>.csv --results <dir>`.

---

## Step 7: P2 external data, convention decision, simulator, suite (2026-11-06)

**What:** new package `mtah/` (see `mtah/README.md`):
`schema.py` (the shared `Game` container, proposed M1 contract), `external/metrica.py`,
`external/dfl.py`, `external/atd_io.py`, `external/stats.py`, `simulate.py`,
`external/build_suite.py`, `external/score_suite.py`.

**DFL/IDSSE (Bassek et al. 2025, CC BY 4.0):** 7 matches downloaded from figshare (the
`ndownloader` links kloppy uses sit behind a bot check; `api.figshare.com/v2/file/download/<id>`
serves the same files). Parsed with a streaming lxml parser (~25 s and <1 GB per match). GT = the
provider flags: `BallStatus` 0 -> D, else `BallPossession` 1/2 -> H/A.

Flag checks (independent of the flags): the player nearest the ball (within 1.5 m) is on the
possessing team 87-92% of live frames; both half orientations agree with team centroids.
DFL flips to dead a median **+3 to +4 frames after** the ball leaves the pitch; our convention
(and the Metrica GT) starts D at the crossing, so DFL ball-out D starts are snapped back to the
crossing frame (`dfl.snap_out_starts`, ~70 per match, after which live-with-ball-outside = 0).

| | spells/min | mean s | median s | <1 s | <2 s | D share |
|---|---|---|---|---|---|---|
| DFL (7 matches) | 3.8-5.3 | 5.5-9.2 | 3.0-4.7 | 9-17% | 25-37% | 41-52% |
| Metrica `events` | 3.8-3.9 | 8.5-9.3 | 4.1-4.6 | 13.5% | 25-29% | 42-45% |
| Metrica `control` | 3.4-3.5 | 9.4-10.2 | 5.1-5.7 | 5-6.5% | 17-22% | 42-45% |

**Decision (uncontrolled touches): keep `events`.** DFL's spell-length distribution is closer to it
(KS 0.094 vs 0.133), as are the short-spell shares. `control` stays available
(`metrica{1,2}_control`) as a robustness variant.

**Round-trip checks:** a Game written as ATD by `atd_io` and read back by the unchanged
`starter/baseline.py` reproduces the clean Metrica baseline exactly (spellF1 0.4312 / 0.4176).

**Simulator first result (PROVISIONAL target):** baseline_smooth50 on Metrica at the `target`
level: spellF1 0.225 (real match 0.243) but macroF1 0.46 (real 0.672). With random-context gaps the
baseline cannot find D, so on the real ATD the missing ball must still correlate with stoppages.
The gap model is now conditioned on context cells (ball outside the pitch x players slow), with
hazard and gap length per cell, measured on the target by `stats.py --target` (aggregates only).

---

## Step 8: Simulator calibrated to the real target, suite rebuilt, team format (2026-11-07)

**Starting point:** `target_stats.json` (aggregates of the competition ATD) existed, but the suite
had been built with the PROVISIONAL target (`manifest.json: target_provisional: true`), at the
`target` level only. Against the real aggregates the simulator was far off: 6.1 ball gaps/min vs
1.4, gaps too short (median 18 vs 268 frames), ball seen outside the pitch 18% vs 4%, ball and
player jitter 10-20x too high, 32 tracklets/min vs 11.5.

**Main finding: on the real ATD the ball goes missing mostly while play is stopped.** Clean Metrica
(same provider, ball removed while dead) already looks almost like the target:

| | missing | gaps/min | gap median | missing at mean player speed <1 / 1-1.5 / 1.5-2 / 2-2.5 / >2.5 m/s |
|---|---|---|---|---|
| target ATD | 0.41 | 1.4 | 268 fr | 0.79 / 0.51 / 0.21 / 0.13 / 0.11 |
| clean Metrica game 1 | 0.39 | 1.6 | 208 fr | 0.80 / 0.56 / 0.20 / 0.10 / 0.03 |

So step 5's "missing mostly during play" is wrong for the target. A missing ball is a strong
dead-ball cue on the real match (important for P3), but weaker than in clean Metrica: ~11% of the
time while players run the ball is still missing. The simulator now reproduces that strength.

**Simulator changes (`mtah/simulate.py`):**
- Ball gaps are a two-state (seen/missing) process. Start hazard is log-linear in context cell
  (in/out x slow/fast), ball speed, crowd and edge (context at t-1, the bins stats.py measures);
  10% of gaps are short blips (target gap histogram below its empty 25-100-frame bins), the rest
  last at least 100 frames and end with a rate per mean-player-speed bin. No labels are used.
- Clean-ball runs outside the pitch longer than 1 s are treated as unseen before filling. DFL parks
  the ball 0.2-1.3 m outside the line through stoppages (27-31% of all frames); the target sees the
  ball outside on 4%.
- Ball and player positions are smoothed (Gaussian, normalised for NaN) or AR(1)-jittered until the
  second-difference spread matches the target (DFL players are too rough, the target is smooth).
- Tracklets: fitted lifetime scale and detection-gap share; least-recently-used slot reuse; target
  per-team slot counts; speed NaN prefix per tracklet to match the target's 8%.
- `mtah/external/calibrate.py` fits the free parameters, pooled over the 9 games, by iterative
  proportional fitting on the target hazard tables, a global rescale to the overall gap-start
  hazard, and the end rates to the missing-by-player-speed occupancy -> `data/cache/sim_params.json`
  (stamped with the target's hash).

**Calibration (mean over the 9 games at `target` vs the real ATD):**

| statistic | sim | target |
|---|---|---|
| ball missing | 0.357 | 0.407 |
| gaps per min | 1.57 | 1.42 |
| gap-start hazard | 0.00163 | 0.00158 |
| missing by player speed | 0.75 / 0.44 / 0.22 / 0.14 / 0.11 | 0.79 / 0.51 / 0.21 / 0.13 / 0.11 |
| ball seen outside the pitch | 0.024 | 0.042 |
| ball / player d2 std (m) | 0.0064 / 0.0017 | 0.0062 / 0.0016 |
| objects per frame | 20.6 | 20.5 |
| tracklets per min | 11.7 | 11.5 |
| speed NaN share | 0.080 | 0.080 |

Remaining differences, known and accepted:
- Overall missing is lower and gaps shorter (median 163 vs 268 frames). The match-up per
  player-speed bin is close, but the external games have more frames with fast players than
  BAR-ATH. A 3-s trailing speed for the end rate was tried and made it worse (median 140).
- The target has ~16% of frames with fewer than 20 objects and ~52% with 22-23 (bursts of many
  players lost at once). The simulator drops players independently, so its object counts are
  narrower (objects_hist L1 0.71). Not modelled yet.
- One-frame ball jumps are 2x the target (they come from the clean DFL data).

**Suite rebuilt:** 9 games x {light, target, heavy} = 27, `target_provisional: false`,
`sim_params_fitted: true`. Ball missing: light 0.25-0.34, target 0.31-0.41, heavy 0.40-0.49.

**Calibration check, official baseline (smooth 50) on the suite** (95% CI over games):

| level | spellF1 | macroF1 |
|---|---|---|
| light | 0.282 [0.263, 0.304] | 0.611 |
| target | 0.270 [0.258, 0.281] | 0.641 |
| heavy | 0.235 [0.221, 0.249] | 0.636 |
| **real match** | **0.243** | **0.672** |

Provisional suite before this step: 0.233 / 0.509. The real match falls between `target` and
`heavy`, so report both and treat `heavy` as the conservative number. (macroF1 is lowest at `light`
because the baseline writes D whenever the ball is missing.)

**Other fixes:**
- **DFL y axis was mirrored.** Sportec tracking runs y bottom-to-top (kloppy
  `SportecTrackingDataCoordinateSystem`), while ATD/Metrica runs top-to-bottom. `dfl.py` now uses
  `y = W/2 - Y`, and the caches were rebuilt. Distances, speeds and labels are unchanged; handedness is now right.
- DFL caches record competition, season, matchday, title and result (`meta.match`).
- `Game` metadata moved from `meta.json` to `game.json`, so the team format can use `meta.json` in the same
  folder (old caches still load).
- `experiments.csv` no longer stores `../../..` paths into the scratchpad.
- `stats.py` gained `hazard_by_player_speed` and `end_hazard_by_player_speed`. **To use them, rerun
  `python -m mtah.external.stats --target` locally**, then `calibrate` and `build_suite`. The
  calibrator then fits the end rates directly instead of through occupancy.

**Team format (agreed with P1):** `python -m mtah.external.export` writes `ball_xy.npy`,
`players_xy.npy`, `players_speed.npy`, `players_team.npy`, `meta.json` (+ `labels.npy`) to
`$MTAH_CACHE/<game>/` for the 11 clean games and the 27 suite games. H attacks +x in every half
(180-degree rotation, as kloppy `STATIC_HOME_AWAY`); speed comes from positions with k = round(0.2 fps).
Checked: the home centroid is left of the away centroid in 94-100% of frames of every half of every game.
Degradation is deterministic: two runs give a bit-identical copy of the suite game.

**Reproduce:**
```bash
python -m mtah.external.metrica && python -m mtah.external.dfl
python -m mtah.external.calibrate
python -m mtah.external.build_suite
python -m mtah.external.export
python -m mtah.external.score_suite --make-baseline <dir>
python -m mtah.external.score_suite --results <dir> --tag baseline_smooth50_calibrated
```
