# Labelling guide: shared labels, data format and evaluation

The reference for everyone on the team and for any Claude session in this repo. It fixes what must be
**the same across modules**: the labels, the rules, the data schema, the GT format and how we score.
How each module works is up to its owner (see the module contract in `README.md`).

**Status of each item:**
- **[fact]** checked in the official docs, the scorer code, our research checks or the external-data
  source. Research-check numbers come from running `research/checks/c*.py` (aggregates only).
- **[decision]** agreed by the team.
- **[proposal]** not agreed yet. The owner named next to it confirms or changes it.
- **[unverified]** stated by a team document, not yet checked against the data or source.

When code and this guide disagree, fix one of them in the same change.

---

## 1. What we are labelling

**[fact]** One label per frame of the BAR–ATH tactical video (199,319 frames, 25 fps; `games.py:43-44`):

| Label | Meaning |
|---|---|
| `H` | Barcelona (home) has possession, ball alive |
| `A` | Athletic Club (away) has possession, ball alive |
| `D` | ball dead (not in play), whoever restarts it |
| `X` | no answer. **Write it only on ungraded frames** (outside the halves). |

**[fact]** How the official GT is made (`CHALLENGE.md:62-100`):
- It is derived **mechanically from Metrica's corrected tracking**: `D` where that tracking flags the
  ball dead, otherwise the team the tracking names as owner. Nothing is smoothed afterwards.
- Possession stays with the last team through loose or contested balls.
- A touch without control, a deflection, or an aerial duel without control is **not** a change of
  possession.
- When exactly D starts after a whistle is not specified (`CHALLENGE.md:102-103`).

### How the score reacts

**[fact]** From `evaluation/possession_metrics.py` (the authority, `EVALUATION.md:153`):

- **spellF1 (ranking metric).** A *spell* is a maximal run of `H` or of `A`; `D` runs separate spells
  and are never spells. Spells are matched one-to-one by temporal IoU, F1 averaged over IoU 0.3–0.7
  (`:60`). Consequences, checked on synthetic arrays:
  - a **missed `D`** merges two spells; a **spurious `D`** splits one;
  - **one flicker** inside a spell: 2 false spells at IoU 0.3–0.5, 1 miss + 3 false at 0.6–0.7;
  - a **boundary error** only costs when it pushes IoU under a threshold, so it hurts **short spells**:
    with both edges of every D 1 s late, spellF1 is 0.62 on 3 s spells, 0.81 on 5 s, 1.0 on 10 s.
    **Shrinking a spell costs more than growing it** by the same amount. A whole-file 1 s shift on the
    real GT gives 0.6539 (`EVALUATION.md:69`).
- **`X` on graded frames.** Forward-filled with our previous label in the same half before spellF1
  (`:274-293`), and a miss on macroF1. So `X` is never better than writing the previous label.
- **macroF1**: per-frame F1 over H/A/D; first tie-break (`CHALLENGE.md:144`).
- **`alive_dead_f1`**: the alive/dead call alone, with **ALIVE as the positive class** (`:208-225`).
- **Transitions** at 12 and 25 frames (`:46`): bad at 12 but good at 25 means a systematic delay.
- **Noise band**: about **0.04** spellF1 on a full match (`:70`); `run_evaluation.py` applies it only
  with ≥ 100 GT spells (`run_evaluation.py:82`). Average over games and noise levels (§6).

---

## 2. Compliance: what makes a result valid

| Rule | Source / why |
|---|---|
| No human labels on BAR–ATH, in any amount or for any purpose (including validation). No hand-corrected tracking. | `RULES.md:56-59`. The game is the test set. |
| No human-made data about BAR–ATH (Opta, StatsBomb, corrected tracking). Such data about **other** games is fine. | `RULES.md:82-84`. |
| No constants tied to specific frames of BAR–ATH. Detect such things with code. | `RULES.md:70-71`. |
| The public 5-minute GT only checks the pipeline (frames aligned, file well-formed, labels meaning what you think). No tuning or training on it. | `RULES.md:77-80`. **[decision]** we run it once, at freeze. |
| Running the repo must reproduce the submitted CSV **byte for byte**. | `RULES.md:86-89`. Fixed seeds, deterministic settings. |
| Match data and frame-level derived arrays never go into git. | `RULES.md:107-111`; the repo becomes public. |

**Allowed:** watching the video to debug; unsupervised learning on BAR–ATH; algorithmic pseudo-labels;
any labelled data from other games (document it in the technical doc) (`RULES.md:61-71`).

---

## 3. Data sources

| Source | What it has | Labels | Watch out |
|---|---|---|---|
| **BAR–ATH** (`data/tactical/`) | ATD tracking (46 slots), Smart Tagging (638 automatic phases), homography | none (test set) | slots are not persons; ball missing on 43.1% of all frames, 40.7% of graded ones |
| **DFL / IDSSE**, 7 matches, Bundesliga 1 and 2, 2022/23, CC BY 4.0 (kloppy `_providers/sportec.py:211-213`) | tracking + per-frame `BallStatus` and `BallPossession` | **provider flags** (closest to our GT) | see §4.1 |
| **Metrica sample games 1–2** | tracking CSV + events | events, through P2's converter | event-derived labels approximate the convention |
| **SkillCorner open data** (optional) | broadcast tracking | **none usable for D**: kloppy marks the ball dead whenever no owner is set (`kloppy/.../skillcorner.py:101`) | never use its `ball_state` as GT |

**[fact] Where the BAR–ATH ball goes missing: mostly during stoppages, not during live play.**
- 99.9% of ball-missing graded frames sit in gaps of 3 s or more (150 gaps, 133 of them ≥ 3 s; c1).
- 81% of ball-missing frames are slow (team mean speed < 1.5 m/s); median speed 1.04 m/s inside gaps
  ≥ 3 s against 2.00 m/s with the ball visible (c3).
- Some loss is live play: missing but fast = 7.9% of graded frames (c3).
- `DATA.md:113-115` claims the opposite, without numbers. Slow is not the same as dead (no GT).

---

## 4. GT format for external games (owner: P2)

```
data/validation/<game>/<game>_gt.csv          frame,label   (H/A/D, X outside the halves)
data/validation/<game>/<game>_gt_meta.json    fps, halves, label counts (as in the official meta)
```
`frame` is that game's own frame index from 0, so `run_evaluation.py --gt <csv>` scores it unchanged.

### 4.1 DFL / IDSSE gotchas (kloppy 3.18.0, installed source)

| Gotcha | Why it matters |
|---|---|
| **[fact]** `only_alive` defaults to **True** (`_providers/sportec.py:191`) and drops every dead frame. Pass `only_alive=False`. | Otherwise no `D` and holes in the frame index. |
| **[fact]** Frames with **no ball record are skipped even with `only_alive=False`** (`sportec/deserializer.py:374`). | Check that the frame index has no holes. |
| **[fact]** Any `BallPossession` other than 1 becomes *away* (`deserializer.py:398-402`); a missing `BallStatus` becomes DEAD (`:405`). | Count raw values once per file. |
| **[unverified]** Possession is never blank, even while the ball is dead. | If true, `D` must come from `BallStatus`. |
| Frame rate and pitch size come from the metadata. | Never assume 25 fps or 105 × 68. |

### 4.2 Convention for uncontrolled touches

**[decision]** Label Metrica both ways (possession changes on any touch / only on control), then pick
the variant whose statistics are closer to DFL's flag labels. The public window is never the deciding
input.

### 4.3 GT acceptance checks (before anyone uses a game)

1. The GT scored against itself with `run_evaluation.py` gives spellF1 = 1.0.
2. Label counts add up to the frame count; `X` only outside the halves.
3. Print spells per minute, mean spell duration and D share. **[unverified]** Metrica 3.8–3.9 spells/min,
   42–45% D. Record DFL's values the first time.

### 4.4 Degraded copies (owner: P2)

3 noise levels per game, from fixed seeds, in the same format. They should reproduce BAR–ATH's
**aggregate** statistics: missing-ball rate by context (mostly stoppages, see §3), gap-length histogram
(150 gaps, 133 ≥ 3 s), median slot tracklet 35.5 s, 20.5 objects per frame (c1, c6).

---

## 5. One data schema for every game

Every loader (BAR–ATH in `mtah/data.py`, external games in `mtah/external/`) returns the **same
structure**, so every module runs unchanged on any game. All values are computed per game; no module
contains a number specific to one game.

| Key | Shape / type | Rule | Why |
|---|---|---|---|
| `ball_xy` | `(n, 2)` float, metres | `NaN` when missing, never 0 or −1 | gaps are a signal; a fake 0 is a ball at the corner flag |
| `players_xy` | `(n, P, 2)` float, metres | one column per slot or player ID, `NaN` when absent | |
| `players_speed` | `(n, P)` float, m/s | **[decision, P1]** computed from `players_xy` for every game: \|xy(t+k) − xy(t−k)\| / 2k frames, k = round(0.2 s × fps); `NaN` where the slot is empty, where either end is missing, and **above 12 m/s**. Never the provider's speed column. | one formula on every game (see below) |
| `players_team` | `(P,)` of `"H"`/`"A"` | from that game's metadata | |
| `meta["pitch"]` | `(L, W)` metres | from metadata (BAR–ATH: `games.py:62`, 105.3 × 68.0) | line positions are derived from it |
| `meta["fps"]` | int | from metadata | durations in seconds stay correct |
| `meta["halves"]` | `[(first, last), ...]` frames, **inclusive** | BAR–ATH: `games.half_video_blocks()`; external: kloppy periods | modules work per half, never across half-time. The scorer uses half-open runs (`possession_metrics.py:259-271`). |
| frame index | `i` = frame `i` of that recording | BAR–ATH: the video frame = submission row | no off-by-offset errors |

**[decision, P1, 2026-11-07; P2 applies it in `mtah/external/`]** **H always attacks towards x = L**,
in every half and every game.
- External games: `dataset.transform(to_orientation=Orientation.STATIC_HOME_AWAY)` in kloppy ("the home
  team plays from left to right in both periods", kloppy 3.18.0 `domain/models/common.py:386`).
  **[unverified]** which way y points after the transform depends on the coordinate system passed;
  P2 checks it against BAR–ATH (y = 0 on the same touchline).
- BAR–ATH: in each half where H attacks right to left, x → L − x and y → W − y (a 180° rotation, so
  left/right of the attacking team is kept). Direction read from the metadata, never a frame number:
  Barcelona (H) `attack_direction_first_half = left_to_right` (`BAR-ATH_tactical_atd.xml:85-86`),
  Athletic `right_to_left` (`:94-95`); c6 confirms the swap at half-time.

**[decision, P1, 2026-11-07] Speed is computed from positions, never taken from the provider.**
The sources disagree, so one formula keeps speed features comparable across games:
- BAR–ATH: the ATD speed column is m/s and matches position speed (median ratio 0.995, correlation
  0.93, median difference 0.11 m/s per player-frame; graded frames under 1.5 m/s: 49.0% column vs
  50.8% positions), but 9.8% of tracked cells have no speed value. Position speed covers 99.6% of
  tracked graded cells against 92.0% for the column (c8). The XML labels that column `kph`
  (`BAR-ATH_tactical_atd.xml:1131-1136`), but its values are m/s: they match position speed at ratio
  ≈ 1, not 3.6, and the maximum is 9.43.
- DFL / IDSSE: kloppy reads `S` with `float(data.get("S", 0))`, so a missing speed becomes **0**, i.e.
  "standing still" (kloppy 3.18.0 `sportec/deserializer.py:415`). **[unverified]** the unit of `S`.
- Metrica CSV: kloppy sets no speed at all (`metrica_csv.py:113-121`).

**[decision, P1, 2026-11-07] Speeds above 12 m/s are set to `NaN`** (`config.SPEED_MAX_MPS`).
Columns are tracker slots, so a slot handed to another person shows up as a jump: on BAR–ATH,
position speed reaches 99.5 m/s and 0.048% of graded player-frames with a speed exceed 12 m/s, while
the provider's own column never exceeds 9.43. One such spike lifts a 22-player frame mean by about
4.5 m/s, which hides exactly the motion collapse `p_dead` looks for. 12 m/s is a physical limit
(fastest human sprint ≈ 12.4 m/s), not a property of one game, so it applies to every game.
Positions are kept; only the speed is dropped.

**[decision, P1] Storage:** one folder per game, `$MTAH_CACHE/<game>/`, one `.npy` per key above
(`ball_xy.npy`, `players_xy.npy`, `players_speed.npy`, `players_team.npy`) plus `meta.json`
(`pitch`, `fps`, `halves`, `n_frames`). Read it back with `mtah.data.load(game)`.

**[fact]** BAR–ATH columns are tracker slots, not people (24 home + 22 away; median tracklet 35.5 s;
c6). Modules use **per-frame team aggregates only**: mean speed, nearest player of each team, counts
within a radius. Never follow a column over time.

**Module outputs** go to `$MTAH_CACHE/<game>/`, with the shapes in the `README.md` module contract.

---

## 6. Evaluation protocol (everyone)

1. **Development metric: mean spellF1 over held-out external games × 3 noise levels**, plus macroF1
   (and `alive_dead_f1` for the dead-ball module).
   ```bash
   python evaluation/run_evaluation.py --gt data/validation/<game>/<game>_gt.csv \
       --results <dir> --out $MTAH_CACHE/reports --bootstrap 1000
   ```
2. **Anything trained is evaluated leave-one-match-out.** 7 + 2 games is small, so leakage dominates.
3. **Keep a component only if it improves the external mean by more than ~0.02** with
   non-overlapping bootstrap intervals.
4. **Log every experiment** in `experiments.csv`: date, owner, change, spellF1_mean, macroF1_mean, games,
   noise_levels, notes.
5. **Target sanity statistics** (unsupervised, allowed) for every full BAR–ATH run: spell count, mean
   spell duration, D share. **[unverified]** expected about 330–420 spells of 9–12 s
   They are alarms, **never tuning targets**.

---

## 7. Pitfalls checklist (check before saying "done")

- [ ] No frame numbers or game-specific constants in module code (`grep` for 5- and 6-digit numbers).
- [ ] Nothing reads `evaluation/gt/` except the single freeze-day smoke test.
- [ ] kloppy DFL loads use `only_alive=False`, and the frame index has no holes.
- [ ] Missing values are `NaN` end to end, with no `fillna(0)` on positions.
- [ ] Each half is processed separately; nothing is smoothed across half-time.
- [ ] `X` appears only outside the halves in the submission.
- [ ] Scores are the external mean over games × noise levels, with bootstrap intervals, logged in
      `experiments.csv`.
- [ ] The run is deterministic: two runs give the same sha256.
- [ ] `git ls-files | grep '^data/'` prints only `data/README.md`.

---

**Sources:** `CHALLENGE.md`, `RULES.md`, `EVALUATION.md`, `DATA.md` and `evaluation/possession_metrics.py`
(official repo); `research/checks/` outputs;
kloppy 3.18.0 source; Bassek et al., Sci Data 12, 195 (2025), data DOI 10.6084/m9.figshare.28196177.
