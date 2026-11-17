# Technical document: team Manita

More Than A Hack 2027, Phase 1: one label (`H` / `A` / `D` / `X`) per frame of the BAR–ATH tactical
video (199,319 frames, 25 fps), ranked on spellF1.

| | |
|---|---|
| Possession file | `Manita.csv`, sha256 `3f152da1e1239e1cc977f564553b019ec8a87c845aa37d4b8a93994c0f623911` |
| Regenerate it | `bash run.sh --fresh` (§6) |
| Team | Alexia (platform and submission; restarts and tags), Theo (external data and simulator), Cristiana (dead ball), Bianca (team possession), Andra (decoder) |

## 1. Approach

**What we built.** A pipeline over the provided automatic tracking that answers two questions
separately for every frame, then decodes them jointly:

```
tracking ──► data.py ──┬─► p_dead[t]  "is the ball dead?"         (logistic regression)  ─┐
                       ├─► p_home[t]  "if alive, is it H's ball?" (LightGBM)             ├─► Viterbi decoder ─► labels ─► Manita.csv
                       └─► restart anchors (kick-off, throw-in, corner, goal kick)       ─┤
Smart Tagging ─────────► tags (set-piece intervals, as hints)                           ─┘
```

**Why this rather than something else.**
- **Learn on other games, never on this one.** The provided game has no labels we may use, and
  the public 5 minutes hold only about 17 possessions, too few to choose anything with. So every
  model and setting is learned on external matches with labels (§3). A simulator first degrades
  them to look like this match's tracking.
- **Split dead/alive from H/A.** The two are driven by different evidence. A dead ball shows up as
  missing or out-of-pitch ball positions and slow players; the team shows up as proximity and
  control. Separate models can be checked separately, and the dead-ball model was trained and
  scored on external games before the team model existed.
- **Tracking only, not video.** The provided tracking already solves detection and identity well
  enough. A video model would have cost the whole time budget and could not be trained without
  labelled video of other games.
- **A decoder instead of per-frame argmax.** spellF1 scores whole possessions, so one flickering
  frame splits a possession in two. A Viterbi decoder with switch costs and minimum durations
  produces runs, not frames.

**Validation.** The development metric is the official spellF1 (`evaluation/`, unedited), averaged
over 9 external matches × 3 simulated noise levels. Everything trained is evaluated
leave-one-match-out (the three noise levels of a match form one fold). A component is kept if it
improves the mean by more than about 0.02. Every run is a row in `experiments.csv`.

## 2. Assets: used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD + XML) | **yes, main input** | parsed to metres; slot order, home/away and attack direction read from the XML; H always attacks +x; speeds recomputed from positions (the column is labelled `kph` but holds m/s, and other providers disagree), speeds above 12 m/s dropped as slot identity jumps |
| Smart Tagging | **yes, hints only** | all 638 automatic tags on video frames; `SET PIECES` intervals make D → alive cheaper in the decoder. Not used in any model: external games have none, so their weight could not be learned |
| Tactical video | **watched only** | to understand the data and debug; no labels, corrections or constants taken from it |
| Public 5-minute GT | **pipeline checks only** | its 4 stoppages were watched to understand the conventions (offside, foul, corner, goal kick: when `D` starts and ends). No parameter was fitted on it and no choice used its score |
| Homography / camera movement | **tried, dropped** | a camera-aim signal was built in research; dropped because no external game has it, so its weight cannot be learned |
| Broadcast video and ATD | **tried, dropped** | alignment to the tactical timeline was explored (`research/checks/c7_broadcast.py`); not needed |
| A sharper ball filter, touch features and a turnover model | **tried, not submitted** | +0.08 on the external suite, but the effect does not transfer to this match (§5) |

## 3. Training data

No labelled data from the provided game was used, in any amount or for any purpose.

| Dataset | Source and licence | Who labelled it | How much | Used for |
|---|---|---|---|---|
| DFL / IDSSE open data, 7 matches (Bundesliga and 2. Bundesliga, 2022/23) | Bassek, Weber, Rein, Memmert (2025), *Scientific Data* 12, 195, CC BY 4.0 | the provider flags `BallStatus` / `BallPossession` shipped with the positions data; we labelled nothing | 668 graded minutes, 1.0 M frames | `p_dead`, `p_home`, restart confidence, decoder tuning |
| Metrica Sports sample games 1–2 | github.com/metrica-sports/sample-data | the human event log published by Metrica; converted to per-frame labels by our code (`tools/events_to_labels.py`, rules in §4) | 191 graded minutes, 0.29 M frames | same |

**Degraded copies.** Each match is degraded at three noise levels (27 copies) by `mtah/simulate.py`.
The simulator adds ball gaps driven by context, positional jitter and recycled tracker slots, and
is calibrated only on aggregate statistics of this match's tracking quality, never on labels. The
copies keep the original labels. On the real match the official baseline scores 0.243; on the
copies it scores 0.265 / 0.251 / 0.206 (`light` / `target` / `heavy`).

## 4. Possession rules implemented

The rules are fixed by the training labels and by a few explicit decisions.

**Loose balls, challenges, deflections.** Possession changes only when the other team takes
control. In the Metrica conversion that means a `PASS`, `SHOT`, `RECOVERY` or `BALL LOST` of the
other team; challenges, loose balls and deflections change nothing. The alternative, possession
changing on any touch (`control` variant), was dropped because the DFL possession-length
distribution matches `events` better (KS 0.094 against 0.133). The team model learns this
convention; features describe control (a player close to the ball, clearly closer than the
opponents, moving with it), not mere proximity.

**When the ball dies.**
- **Ball out:** from the frame the ball crosses the line. Provider flags come a few frames late
  (DFL a median 3–4, Metrica 4–6), so both are snapped back to the crossing in the tracking.
- **Whistles (fouls, offside):** from the whistle, i.e. as the labels mark it. The dead-ball model
  learns it from slowing players and the missing ball; on the public window, players stop about
  10 frames after the whistle.
- We never use typical stoppage durations, which vary too much; only minimum durations.

**Restarts.** `D` lasts until the restart is **taken**. An anchor's frame is the first frame the
ball moves faster than 3 m/s after a stoppage. Its type comes from the restart spot (corner arc,
touchline, goal area, centre mark), and the restarting team from the Laws for corners and goal
kicks (H always attacks +x), otherwise from the nearest player. The decoder makes D → alive
cheaper inside anchor windows, and snaps a D → alive boundary to a nearby high-confidence anchor.

**Runs.** Each half is decoded on its own. Separate switch costs for H↔A and alive↔dead, minimum
durations of 0.30 s alive and 0.97 s dead, and a short one-team flicker (< 0.6 s) between the
same team on both sides is merged.

## 5. The components and their results

**Data and submission** (Alexia; `mtah/data.py`, `run.sh`, `mtah/submit.py`). The same schema for
every game: metres from a corner, NaN for missing, halves inclusive. It checks itself, for example
rejecting a game whose positions sit more than 5 m off the pitch, which catches a centre-origin
loader. `submit.py` forces `X` outside the halves, refuses `X` inside them, and runs the official
validator.

**Dead ball, `p_dead`** (Cristiana; the logistic combiner was built by Alexia from her spec;
`mtah/deadball/`). Features: `log(1 + s)` of the ball gap's length and position, the distance to
the nearest line of the last and next sighting, how far the ball is outside the pitch, the
players' mean speed (absolute and relative to the last 10 s), and missing flags. NaN is filled with
the training median, then L2 logistic regression. The largest weights: player speed (−3.17), ball
outside the pitch (+1.77), gap length (+0.81). Leave-one-match-out, `target` level, against the rule
"dead when the ball is missing for 3 s or more": alive/dead F1 0.704 → **0.876**, dead-interval F1
0.246 → **0.434**, calibration error 0.353 → **0.034**.

**Team, `p_home`** (Bianca; `mtah/model.py`, `features.py`, `ball.py`). A Kalman filter with an RTS
smoother bridges ball gaps of up to 4 s. 44 features: proximity, control events (the last and next
controlling team, control share in the past and future 1 s and 3 s), and team shape for frames
without a ball. LightGBM with mirrored augmentation (180° rotation and H/A swap, which removes home
bias), then isotonic calibration on out-of-fold predictions. With the dead/alive split taken from
the labels, so the team model is judged on its own: spellF1 **0.627**, against 0.438 for "last
controlling team"; accuracy 0.894.

**Restarts and tags** (Alexia; `mtah/deadball/restarts.py`, `tags.py`). An anchor's confidence is
the measured share of real restarts for its evidence group on the external suite: 0.86 when the
type is recognised, the players were slow before and the ball was clearly played; 0.37 or 0.27
otherwise; groups at chance level are dropped. Within 1 s: precision 0.67, recall 0.33, median
timing error 0 frames, restarting team right 90 % of the time.

**Decoder** (Andra; `mtah/decode.py`). Emissions are `log p_dead` and `log((1 − p_dead) · p_home)` /
`log((1 − p_dead) · (1 − p_home))`, scaled. The 6 parameters were tuned with Optuna (TPE, seed 0,
60 trials) on the external suite. To check for overfitting it was tuned on half the matches and
scored on the other half: 0.385 → 0.511 and 0.354 → 0.507, with every copy improving.

**Result, official scorer, 27 external copies:**

| | light | target | heavy | mean |
|---|---|---|---|---|
| Official baseline (`baseline_smooth50`) | 0.265 | 0.251 | 0.206 | 0.241 |
| Our pipeline, default decoder | 0.378 | 0.368 | 0.348 | 0.365 |
| **Our pipeline, submitted** | **0.528** | **0.517** | **0.483** | **0.509** |

On the provided game, where no score can be computed, the output is plausible: 301 possessions of
10.9 s on average, with 48.5 % of graded frames `D`.

## 6. Limitations

- **The simulator copies typical noise, not kicks.** The ball's typical jitter matches (median
  second difference 9 mm here, 11 mm on the copies), but kick-like jumps are 14× rarer in this
  match's tracking (0.27 % of frames against 3.7 %), which smooths kicks. That is why we did not
  submit our best suite result (0.592: a sharper ball filter, touch features, a turnover model).
  On this match its touch detector fires about 8× less often than on the copies, and it predicts
  fewer possession changes, not more. With more time: match the tail of the ball's movement in
  the simulator, then re-test.
- **Small and partly self-chosen validation.** 9 matches, and 5 of the 7 DFL games are home games
  of the same club. The decoder was tuned on the same matches; the held-out check limits, but
  does not remove, the optimism.
- **Restarts:** most corners are missed (4 of 12 tagged ones found), and so are the two
  match-start kick-offs, whose formation lies before the graded half.
- **The dead-ball start after a whistle** is learned from slowing players, so it lags the whistle
  slightly (median +21 frames on DFL).

## 7. Reproduction

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-lock.txt
export MTAH_CACHE=~/.cache/mtah2027          # outside the repository
# 1. the provided data bundle in data/ (data/README.md)
# 2. the external suite, from public data only (about 30 min; mtah/external/README.md, "Rebuild everything")
bash run.sh --fresh                          # data -> p_dead -> p_home -> tags -> restarts -> labels -> Manita.csv
python evaluation/validate_submission.py Manita.csv
shasum -a 256 Manita.csv                     # 3f152da1e1239e1cc977f564553b019ec8a87c845aa37d4b8a93994c0f623911
```

- The trained team model is committed in `models/` (learned from external games only).
  `python -m mtah.model --train` regenerates it byte for byte. `p_dead` is fitted at run time
  (about 1.5 min).
- Seeds are fixed and LightGBM runs deterministically. Two full runs from raw data, and a clean
  checkout in another path containing spaces, gave byte-identical files at every stage.
- The repository contains no match data: only `data/README.md` is tracked under `data/`, and
  generated arrays live in `$MTAH_CACHE`, outside the repository.
- The official `evaluation/`, `starter/` and `games.py` are used unedited (MIT, `LICENSE-metrica`).
