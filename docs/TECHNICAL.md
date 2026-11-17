# Technical document: team Manita

More Than A Hack 2027, Phase 1. The task is to give every frame of the BAR–ATH tactical video
(199,319 frames at 25 fps) one label: `H` if the home team has the ball, `A` if the away team has
it, `D` if the ball is dead, and `X` outside the halves. Submissions are ranked on spellF1.

| | |
|---|---|
| Possession file | `Manita.csv`, sha256 `3f152da1e1239e1cc977f564553b019ec8a87c845aa37d4b8a93994c0f623911` |
| How to regenerate it | `bash run.sh --fresh` (see §7) |
| Team | Alexia (platform and submission; restarts and tags), Theo (external data and simulator), Cristiana (dead ball), Bianca (team possession), Andra (decoder) |

## 1. Approach

### What we built

Our pipeline works from the automatic tracking provided with the match. For every frame it asks
two separate questions: *is the ball dead?* and, if it is alive, *which team has it?* A decoder then
combines both answers into clean possessions.

```
tracking ──► data.py ──┬─► p_dead[t]  "is the ball dead?"         (logistic regression)  ─┐
                       ├─► p_home[t]  "if alive, is it H's ball?" (LightGBM)             ├─► Viterbi decoder ─► labels ─► Manita.csv
                       └─► restart anchors (kick-off, throw-in, corner, goal kick)       ─┤
Smart Tagging ─────────► tags (set-piece intervals, as hints)                           ─┘
```

### Why we built it this way

**We learn from other games, never from this one.** We are not allowed to use labels from the
provided match, and the public 5-minute window only contains about 17 possessions, which is far
too few to choose anything with. So every model and every setting was learned on external matches
that come with labels (§3). Before using them, we run them through a simulator that makes their
tracking look as noisy as the tracking of this match.

**We treat "dead or alive" and "home or away" as two different problems.** They depend on
different evidence. A dead ball shows up as the ball going missing or leaving the pitch, and
players slowing down. Which team has the ball shows up in who is close to it and who controls it.
Keeping the two models apart meant we could test each one on its own. In fact, the dead-ball
model was trained and scored on external games before the team model even existed.

**We use the tracking, not the video.** The provided tracking already handles detecting and
following players well enough. Building a video model would have taken our whole time budget, and
we would have had no labelled video of other games to train it on.

**We decode whole possessions instead of picking the best label frame by frame.** spellF1 scores
possessions, not frames, so a single frame that flickers to the wrong team splits one possession
into two. A Viterbi decoder, which charges a cost for every switch and enforces minimum
durations, produces stable runs instead of noisy frames.

### How we checked our choices

Our development metric is the official spellF1 scorer from `evaluation/`, used without any edits.
We average it over 9 external matches, each at 3 simulated noise levels. Every trained model is
evaluated leave-one-match-out: a match is left out of training and scored, and its three noise
levels always stay together. We only kept a change if it improved the average by more than about
0.02, and every run is logged as a row in `experiments.csv`.

## 2. Assets: what we used and what we dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD + XML) | **Yes, our main input** | We convert it to metres. Slot order, which team is home and the direction of attack come from the XML, and we rotate the data so that H always attacks towards +x. We recompute speeds from the positions, because the speed column is labelled `kph` but actually holds m/s, and other providers disagree. Speeds above 12 m/s are dropped, since they come from tracker slots jumping between players. |
| Smart Tagging | **Yes, but only as hints** | We use all 638 automatic tags on video frames. `SET PIECES` intervals make it cheaper for the decoder to switch from dead to alive. We don't feed them to any model: the external games have no tags, so we couldn't learn how much to trust them. |
| Tactical video | **Watched only** | To understand the data and debug the pipeline. We took no labels, corrections or constants from it. |
| Public 5-minute ground truth | **Pipeline checks only** | We watched its 4 stoppages (an offside, a foul, a corner and a goal kick) to understand when `D` starts and ends. We fitted no parameter on it and made no choice based on its score. |
| Homography / camera movement | **Tried, dropped** | We built a signal from where the camera is pointing, but no external game has this data, so we couldn't learn its weight. |
| Broadcast video and its ATD | **Tried, dropped** | We explored aligning it to the tactical timeline (`research/checks/c7_broadcast.py`), but didn't need it. |
| A sharper ball filter, touch features and a turnover model | **Tried, not submitted** | They added +0.08 on the external suite, but the gain doesn't carry over to this match (see §6). |

## 3. Training data

We used no labelled data from the provided match, in any amount or for any purpose.

| Dataset | Source and licence | Who labelled it | How much | What we used it for |
|---|---|---|---|---|
| DFL / IDSSE open data: 7 Bundesliga and 2. Bundesliga matches, season 2022/23 | Bassek, Weber, Rein, Memmert (2025), *Scientific Data* 12, 195, CC BY 4.0 | The provider: the `BallStatus` and `BallPossession` flags ship with the positions data. We labelled nothing ourselves. | 668 graded minutes, 1.0 M frames | `p_dead`, `p_home`, restart confidence, decoder tuning |
| Metrica Sports sample games 1 and 2 | github.com/metrica-sports/sample-data | Metrica's published human event log, which our code turns into per-frame labels (`tools/events_to_labels.py`, rules in §4) | 191 graded minutes, 0.29 M frames | The same |

**Degraded copies.** `mtah/simulate.py` makes three degraded copies of each match (light, target
and heavy noise), 27 copies in total. It adds ball gaps that depend on the game situation,
positional jitter, and tracker slots that get reused for different players. We calibrated it only
on overall statistics of this match's tracking quality, never on its labels. The copies keep their
original labels. As a sanity check (not a fitting target), the unmodified official baseline
scores 0.243 on the public 5-minute window of the real match and 0.265 / 0.251 / 0.206 on the
light / target / heavy copies.

## 4. The possession rules we implemented

Most of our rules come from the training labels; a few are explicit decisions of ours.

**Loose balls, challenges and deflections.** Possession only changes when the other team actually
takes control of the ball. In the Metrica conversion, that means the other team makes a `PASS`, a
`SHOT`, a `RECOVERY` or a `BALL LOST`. Challenges, loose balls and deflections change nothing. We
also tried the alternative, where possession changes on any touch, but dropped it: the possession
lengths in the DFL data match our rule better (a KS distance of 0.094, against 0.133). The team
model learns this convention from the data. Its features describe control rather than just being
near the ball: a player who is close to it, clearly closer than the opponents, and moving with it.

**When the ball becomes dead.**
- **Ball out of play:** from the frame where the ball crosses the line. The provider flags arrive
  a few frames late (a median of 3–4 frames for DFL and 4–6 for Metrica), so in both cases we move
  them back to the crossing we see in the tracking.
- **Whistles (fouls, offside):** from the whistle, as the external training labels mark it. The
  dead-ball model learns this from players slowing down and the ball going missing. Watching the
  public window showed the same thing (players stop about 10 frames after the whistle), but that
  observation was only a sanity check: it is not a parameter anywhere in our code, and the lag
  the model has is the one it learned on external games.
- We never rely on how long a stoppage usually lasts, because that varies too much. We only use
  minimum durations.

**Restarts.** The ball stays `D` until the restart is actually **taken**. We place an anchor on the
first frame after a stoppage where the ball moves faster than 3 m/s. We work out the type of
restart from where it happens (corner arc, touchline, goal area or centre mark). For corners and
goal kicks the Laws of the Game tell us which team restarts, because we know H always attacks
towards +x; for the others we take the team of the nearest player. Inside an anchor's window the
decoder finds it cheaper to switch from dead to alive, and it moves a nearby dead-to-alive
boundary onto the anchor when the anchor is highly confident.

**Runs.** Each half is decoded on its own. Switching between H and A costs something different
from switching between alive and dead. An alive run lasts at least 0.30 s and a dead run at least
0.97 s. When a very short run of one team (under 0.6 s) sits between two runs of the other team,
we merge it away.

## 5. The components and their results

**Data and submission** (Alexia; `mtah/data.py`, `run.sh`, `mtah/submit.py`). Every game, ours
and the external ones, goes into the same format: metres measured from a corner, NaN for missing
values, and halves that include both their first and last frame. The data module checks itself.
For example, it rejects a game whose positions sit more than 5 m off the pitch, which catches a
loader that measures from the centre instead of the corner. `submit.py` writes `X` outside the
halves, refuses `X` inside them, and runs the official validator before anything is submitted.

**Dead ball, `p_dead`** (Cristiana; the logistic regression was built by Alexia from her
specification; `mtah/deadball/`). The features are: how long the ball has been missing and where
in the gap we are (on a `log(1 + s)` scale), how close the ball was to a line when it was last
seen and when it is next seen, how far the ball is outside the pitch, how fast the players are
moving on average (both absolutely and compared to the last 10 s), and flags for missing data.
Missing values are filled with the training median, then an L2-regularised logistic regression
gives the probability. The strongest weights are player speed (−3.17), ball outside the pitch
(+1.77) and gap length (+0.81). We compared it with the simple rule "the ball is dead when it has
been missing for 3 s or more", leave-one-match-out at the `target` noise level. The alive/dead F1
went from 0.704 to **0.876**, the F1 on dead-ball intervals from 0.246 to **0.434**, and the
calibration error dropped from 0.353 to **0.034**.

**Team, `p_home`** (Bianca; `mtah/model.py`, `features.py`, `ball.py`). A Kalman filter with an
RTS smoother fills in ball gaps of up to 4 s. The model uses 44 features: proximity to the ball;
control events, such as which team last controlled the ball, which team controls it next, and how
much each team controlled it in the past and next 1 s and 3 s; and team shape, for frames where
the ball is missing. It is a LightGBM model trained with mirrored copies of the data (rotated by
180° with H and A swapped, which removes any bias towards the home team) and then calibrated with
isotonic regression on out-of-fold predictions. To judge the team model on its own, we took the
alive/dead split from the true labels: it then reaches a spellF1 of **0.627**, against 0.438 for
simply repeating the last team in control, and a frame accuracy of 0.894.

**Restarts and tags** (Alexia; `mtah/deadball/restarts.py`, `tags.py`). Each anchor's confidence
is how often anchors with the same evidence turned out to be real restarts on the external suite.
It is 0.86 when we recognise the type of restart, the players were slow beforehand and the ball
was clearly played, and 0.37 or 0.27 in the weaker cases. Groups that were no better than chance
are dropped. Within 1 s of a real restart, precision is 0.67 and recall 0.33, the median timing
error is 0 frames, and the restarting team is right 90 % of the time.

**Decoder** (Andra; `mtah/decode.py`). The decoder reads three scores per frame: `log p_dead` for
dead, and `log((1 − p_dead) · p_home)` and `log((1 − p_dead) · (1 − p_home))` for H and A, all
scaled. Its 6 parameters were tuned with Optuna (TPE sampler, seed 0, 60 trials) on the external
suite. To check that this didn't overfit, we also tuned it on half of the matches and scored it
on the other half: the held-out scores went from 0.385 to 0.511 and from 0.354 to 0.507, and every
single copy improved.

**Results with the official scorer on the 27 external copies:**

| | light | target | heavy | mean |
|---|---|---|---|---|
| Official baseline (`baseline_smooth50`) | 0.265 | 0.251 | 0.206 | 0.241 |
| Our pipeline, default decoder | 0.378 | 0.368 | 0.348 | 0.365 |
| **Our pipeline, as submitted** | **0.528** | **0.517** | **0.483** | **0.509** |

We can't compute a score on the provided match, but the output looks plausible: 301 possessions
lasting 10.9 s on average, and 48.5 % of the graded frames labelled `D`.

## 6. Limitations

- **The simulator copies everyday noise, but not kicks.** The ball's typical jitter matches well
  (a median second difference of 9 mm on this match against 11 mm on the copies). But sudden,
  kick-like jumps are 14 times rarer in this match's tracking (0.27 % of frames against 3.7 %),
  because its kicks are smoothed. That is why we did not submit our best suite result (0.592, with
  a sharper ball filter, touch features and a turnover model). On this match, its touch detector
  fires about 8 times less often than on the copies, and it predicts fewer possession changes
  rather than more. With more time, we would make the simulator reproduce these large ball
  movements too, and test again.
- **Our validation set is small and partly self-chosen.** It has 9 matches, and 5 of the 7 DFL
  games are home games of the same club. The decoder was also tuned on these same matches. The
  held-out check limits how optimistic our numbers are, but doesn't remove it entirely.
- **Restarts:** we miss most corners (we find 4 of the 12 tagged ones), and also both kick-offs
  that start a half, because the players line up before the graded half begins.
- **Dead ball after a whistle:** the model learns it from players slowing down, so it marks the
  ball dead a little after the whistle (a median of 21 frames late on DFL).

## 7. Reproduction

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-lock.txt
export MTAH_CACHE=~/.cache/mtah2027          # must be outside the repository
# 1. put the provided data bundle in data/ (see data/README.md)
# 2. build the external suite from public data only (about 30 min; mtah/external/README.md, "Rebuild everything")
bash run.sh --fresh                          # data -> p_dead -> p_home -> tags -> restarts -> labels -> Manita.csv
python evaluation/validate_submission.py Manita.csv
shasum -a 256 Manita.csv                     # 3f152da1e1239e1cc977f564553b019ec8a87c845aa37d4b8a93994c0f623911
```

- The trained team model is committed in `models/` and was learned from external games only.
  `python -m mtah.model --train` rebuilds it byte for byte. `p_dead` is fitted while the pipeline
  runs, which takes about 1.5 minutes.
- Seeds are fixed and LightGBM runs in deterministic mode. Two full runs from the raw data, plus a
  clean checkout in a different folder whose path contains spaces, produced byte-identical files
  at every stage.
- The repository contains no match data. Under `data/`, only `data/README.md` is tracked, and all
  generated arrays live in `$MTAH_CACHE`, outside the repository.
- We use the official `evaluation/`, `starter/` and `games.py` without any edits (MIT licence,
  `LICENSE-metrica`).
