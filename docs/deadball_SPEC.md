# p_dead spec (P3a, Cristiana)

**In one sentence:** for every frame, give the probability that the ball is dead (`D`), using tracking
only, so that the decoder (P5) can place the stoppages that separate possessions.

Scope: `p_dead` only; restarts, anchors and tags belong to P3b (`README.md:87-88,93-94`). Decisions agreed
2026-11-07. Sources are `file:line`; **[unverified]** marks claims not yet checked; **[choice]** marks a
team choice with no external source.

## 1. Decisions

| # | Decision | Reason |
|---|---|---|
| 1 | `p_dead` never snaps boundaries. The decoder alone places D edges, using anchors and tags as costs. | One owner per edge; `p_dead` stays measurable without anchors (`README.md:92-93`). |
| 2 | Combiner = logistic regression, clipped to [ε, 1−ε]. No isotonic step. | P5 multiplies log-probs by one `emission_scale` (`mtah/optuna_tune.py:38,45`), which rescales log-odds. An exact 0 or 1 becomes −∞ in the Viterbi sum (`mtah/decode.py:60`). |
| 3 | Motion = absolute speed **and** speed relative to a trailing window. | Speed has no natural break on BAR–ATH (bimodality 0.515 < 0.555, `c3_dead_ball.py` output); tempo differs between leagues. |
| 4 | Continuous features, no hard "≥ 3 s" or "past line by X m" rules. | The combiner learns the cut-offs; 7.9 % of graded frames are ball-missing but fast (c3 output). |
| 5 | No tags, anchors or BAR–ATH-only signals inside `p_dead`. | Their weight can't be learned: external games have no Smart Tagging (`README.md:92-93`) and no homography (the DFL figshare article 28196177 has only matchinformation, events and positions files; the Metrica sample-data repo has only tracking CSVs and events; file listings checked 2026-11-07). |
| 6 | Own metrics mirror spellF1 (§4). Windows are centred (offline), cut at half edges. | Same matching as the ranking metric; c3 uses centred windows (`c3_dead_ball.py:12`, `common.py:120-130`). |
| 7 | **`p_dead` is honest (calibrated).** The decoder does any leaning towards "alive". | A too-long D shrinks spells, which costs more than growing them (guide `:47-48`). One dial for it: `switch_dead` / min durations (`decode.py:23-32`, `optuna_tune.py:40-42`). Leaning in both places would correct twice. |
| 8 | Training games = a setting, default **every external game P2 delivers**: today 7 DFL matches + Metrica games 1–2 with the `events` convention (P2's report, 2026-11-07; not yet in this repo). Provider timing compared on them (§4). | The D-onset convention is unspecified (guide `:35`). Work starts before P2's games exist. |
| 9 | Camera motion is **not** in `p_dead` but is saved as `camera_step.npy` for P5 to use or ignore. | It only exists for BAR–ATH, so it can't be weighted on external games (decision 5). |

## 2. Inputs and outputs

**Input:** the per-game schema of guide §5 (`LABELLING_GUIDE.md:139-147`):
- `ball_xy (n,2)` in metres, NaN = missing;
- `players_xy (n,P,2)`, `players_speed (n,P)` in m/s, `players_team (P,)` in {H, A};
- `meta`: `pitch (L,W)`, `fps`, `halves [(first,last)]` (inclusive).

Rules on the input:
- Only per-frame team aggregates are used (guide `:212-214`).
- Every feature is computed inside one half.
- "H attacks towards +x" is now a P1 decision (guide §5). No `p_dead` feature depends on it: line features are unchanged under the flip (scratch check, max difference 7e-15).
- `players_speed` is computed by P1's loader from positions (±0.2 s centred difference), never the provider column (guide §5). On BAR–ATH, "slow" (< 1.5 m/s) is 50.8 % of graded frames with it, against 49.0 % with the column (c8 output); c3 figures quoted here use the column.
- Speed is computed over the whole recording before the half split (`mtah/data.py:259`), so the first and last 0.2 s of a half use positions just outside it; accepted (5 frames per edge).
- Read through `mtah.data.load(game)` (P1): `$MTAH_CACHE/<game>/{ball_xy,players_xy,players_speed,players_team}.npy` + `meta.json`.

**Outputs (P3a)**, in `$MTAH_CACHE/<game>/`:

| File | Shape / type | Content |
|---|---|---|
| `p_dead.npy` | `(n,)` float64 | in [ε, 1−ε] inside halves, NaN outside (decoder slices halves only, `decode.py:225-226`) |
| `deadball_features.npz` | `X (n,k)` float64, `names (k,)` str | the features of §3, for P5's error analysis |
| `camera_step.npy` (BAR–ATH only) | `(n,)` float64, metres | camera-aim movement per frame, NaN where the homography is invalid (`common.py:133-143`) |

**P3b formats.** Current proposals (`README.md:87-88`); Alexia owns them and `p_dead` reads neither. See §6.
- `anchors.json`: `[{"frame": int, "type": str, "team": "H"|"A", "confidence": float}]`, type ∈ kickoff, throw-in, corner, goal kick, free kick (exact strings and confidence range: Alexia)
- `tags.json` (**decided 2026-11-07**, written by `mtah/deadball/tags.py`, format in its docstring):
  `[{"start_frame", "end_frame" (exclusive), "code", "type" | null, "team": "H"|"A"|null, "possession_team": "H"|"A"|null, "half": int|null}]`, all 638 instances

## 3. Features (`mtah/deadball/signals.py`, `lines.py`; parameters in `mtah/config.py`, section `DEADBALL`)

All parameter defaults below are **[choice]**. Gap features use log(1 + s), not a cap: a 60 s cap would
flatten 19 % of BAR–ATH ball-missing graded frames (scratch check, 2026-11-07); the log keeps their order
without letting the longest gaps dominate the fit.

| Feature | Detects | Inputs | Parameters (default) |
|---|---|---|---|
| `gap_len_s` | log(1 + length in s) of the ball-missing run containing t (0 if visible) | `ball_xy`, `fps` | none |
| `t_since_seen_s`, `t_until_seen_s` | log(1 + s) since / until the ball was last / next seen (edges matter most) | `ball_xy`, `fps` | none |
| `last_line_dist_m`, `next_line_dist_m` | ball last / next seen near a line → went out or restart: in a gap, max(0, min(x, L−x, y, W−y)) of the last / next sighting (0 on or past a line); where visible, of the current ball | `ball_xy`, `pitch` | none |
| `ball_out_m` | visible ball past a line: min(`OUT_CAP_M`, hypot(dx, dy)), dx = max(0, −x, x−L), dy = max(0, −y, y−W); NaN while the ball is missing | `ball_xy`, `pitch` | `OUT_CAP_M` (5) |
| `speed_abs` | team motion collapse: per frame, mean `players_speed` over tracked players of both teams; then a centred mean over ± round(`SPEED_SMOOTH_HALF_S` × fps) frames | `players_speed`, `fps` | `SPEED_SMOOTH_HALF_S` (0.5) |
| `speed_rel` | `speed_abs` / max(`REL_FLOOR`, median of `speed_abs` over the trailing round(`REL_WIN_S` × fps) frames, t included) | `players_speed`, `fps` | `REL_WIN_S` (10), `REL_FLOOR` (0.1 m/s) |
| `*_missing` | 1 where the feature is NaN inside a half, else 0 (`speed_missing` covers `speed_abs` and `speed_rel`). `signals.py` leaves the feature NaN; the combiner fills it with the training median | all | none |

**Motion details** (decided 2026-11-07, all **[choice]**; one rule for every game):
- Windows in frames: round(seconds × fps) with Python `round` (25 fps: ± 12 frames for smoothing, i.e. 1.0 s, and 250 frames trailing; 30 fps: ± 15 and 300).
- Every window stops at the half edge and shrinks there, never NaN because of the edge. This matches the gap rule (an edge counts as a boundary) and c3's `rolling_mean` (`research/checks/common.py:120-131`).
- At the start of a half the trailing window holds whatever history the half has (at the first frame, only t itself, so `speed_rel` = 1 unless the floor applies).
- NaN frames inside a window are skipped; the result is NaN only if every frame of the window is NaN. The trailing window includes t, so `speed_rel` is defined exactly where `speed_abs` is.
- A frame's mean speed needs at least one tracked player with a speed; no minimum count.
- `players_speed` is read as float64 (it is stored as float32, `mtah/data.py:130`). No numpy all-NaN warnings: empty frames and empty windows are handled before any `nanmedian` call.

**Line details** (decided 2026-11-07, all **[choice]**; `mtah/deadball/lines.py`):
- (a) Where the ball is visible, `last_line_dist_m` and `next_line_dist_m` hold the current ball's distance: a real measurement, so no jump at gap edges; `gap_len_s` = 0 still tells visible frames apart.
- (b) A gap that opens or closes a half has no sighting on that side inside the half: that side is NaN, flagged by `last_line_missing` / `next_line_missing` (two indicators, the sides go missing independently). Unlike the gap times of step 2, an edge has a time but no ball position.
- (c) `ball_out_m` is NaN while the ball is missing, with `ball_out_missing` (equal to `gap_len_s` > 0; kept so every feature that can be NaN has its own indicator).
- (d) Outside a corner, `ball_out_m` is the straight-line distance to the pitch rectangle (3 m and 4 m past two lines → 5 m).
- Outside the pitch the line distance is 0 (decided 2026-11-07 after the BAR–ATH check): with an unsigned distance a ball last seen 3 m behind the goal line looked like one 3 m inside. With 0, "< 2 m" equals c1's `near_line` exactly: on BAR–ATH gaps ≥ 10 s, 35.9 % start and 33.3 % end near a line, as c1; the unsigned version gave 25.6 % and 16.7 %. How far out stays in `ball_out_m`.
- (e) "Line" = the two touchlines and two goal lines, never the halfway line, as c1's `near_line` (`research/checks/c1_ball_gaps.py:27-28`). Kick-offs belong to P3b.
- Pitch (L, W) comes from `meta["pitch"]`, origin at a corner, pitch from 0 to L and 0 to W (BAR–ATH: `mtah/data.py:255-258`; external games **[unverified]**, P2 checks, guide `:154-155`). Line distances are unchanged when the ball is mirrored or turned 180°. On BAR–ATH no visible ball is outside two lines at once (0.000 %), so (d) does not bite there.

Evidence on BAR–ATH (aggregates only):
- Long gaps start within 2 m of a line 36 % of the time, against 9.1 % for a visible ball at random (c1 output).
- Median speed is 1.04 m/s in gaps of 3 s or more, against 2.00 m/s with the ball visible (c3 output).

**Combiner** (`combine.py`):
- Standardised features → logistic regression with L2, fitted by `scipy.optimize.minimize(method="L-BFGS-B")`. scipy 1.18.1 is pinned (`requirements-lock.txt:103`); scikit-learn is not installed (checked in `.venv`).
- Parameters: `L2` (1.0), `EPS` (1e-3), `TRAIN_GAMES` (all in `data/validation/`).
- Rows: graded frames of degraded external games. Label = provider D flag (DFL `BallStatus`; Metrica GT `D`).
- Fitting is leave-one-match-out (guide `:228`).
- **v0 stub** until external games exist: `p_dead = clip(gap length ≥ 3 s)`, on the raw length (`STUB_GAP_S`), not on the log `gap_len_s` (`README.md:96-97`).

## 4. Evaluation (`mtah/deadball/evaluate.py`) and "done when"

Run per external game × 3 noise levels and per half. Hard call: `p_dead > 0.5`.
- **`alive_dead_f1`**: the official function (`possession_metrics.py:208-225`; ALIVE is the positive class).
- **D-interval F1**: D runs = `segments()` filtered to D (`:296-309`), then `spell_overlaps` + `match_spells` (`:472-514`), mean over `SPELL_IOUS` (`:60`). `spell_scores` itself can't be used: it keeps H/A only (`:464-469`, `:519-520`).
- **Boundary offset**: signed onset and end error, in frames, of D runs matched at IoU ≥ 0.3. Reported as median and p90, overall and **per provider** (DFL vs Metrica, decision 8). This needs our own pair-returning matcher, because `matched_gt_flags` returns flags only (`:492-508`).
- **Calibration**: expected calibration error, 10 equal-width bins weighted by frame count, over graded frames.
- **Ablation**: drop one feature group at a time (gap / line / motion).

**Evaluation details** (decided 2026-11-07, all **[choice]**; settings `HARD_CALL`, `ECE_BINS`, `OFFSET_IOU` in `config.DEADBALL`):
- Hard call: a frame is D when `p_dead > HARD_CALL` (strictly), otherwise alive (scored as H; `alive_dead_f1` counts H and A alike). Everything is computed inside the GT's valid runs (`valid_runs`, `possession_metrics.py:259-271`), whole game and per half.
- (a) Boundary offset = prediction minus GT, in frames (late = positive), on D runs paired by the official greedy order (`matched_gt_flags`, `:492-508`). A side where the GT run **and** the predicted run both sit on the half edge is left out: it is 0 by construction and would flatter the median. If only one of them sits on the edge, the error is real and kept. Reported as median and p90 of |error|.
- (b) No bootstrap yet. It comes as its own step once external games and the fitted combiner exist; what to resample (games, D runs or time blocks) is chosen then. Every score is a function of one GT + `p_dead` pair, so the bootstrap only has to call it on resampled pieces.
- (c) `experiments.csv` keeps its columns. `--all` prints a ready-to-paste row: owner P3a, `spellF1_mean` and `macroF1_mean` empty until P5's decoder exists, `games` = number of GT files, `noise_levels` = number of copies, scores in `notes` as `alive_dead_f1=…; dint_f1=…; ece=…; <provider>_onset_med=…f`. It never writes the file.
- (d) Copies are listed in a manifest CSV `copy,gt,provider` written once P2 gives the layout (asked 2026-11-08); nothing is guessed from folder names. `copy` names `$MTAH_CACHE/<copy>/p_dead.npy`; `gt` is that game's `_gt.csv`; `provider` groups the offsets (decision 8). **[unverified]** that the 3 noise copies share the original game's GT.

Report the mean over games × noise levels and log a row in `experiments.csv` (guide `:231-232`). Once P5's decoder exists, re-score through it with `run_evaluation.py`.

| Piece | Done when | Command (modules planned) |
|---|---|---|
| `signals.py`, `lines.py` | all synthetic tests (§5) pass | `python tests/deadball/run_all.py` (plain asserts; pytest not installed) |
| `combine.py` | two runs give the same sha256 of `p_dead.npy`; nothing outside [ε, 1−ε] inside halves | `python -m mtah.deadball.combine --game <g>` twice, then `shasum -a 256 $MTAH_CACHE/<g>/p_dead.npy` |
| `evaluate.py` | GT scored against itself: F1s = 1.0, offsets = 0, ECE = 0; GT shifted 25 frames late gives +25 | `python -m mtah.deadball.evaluate --self-test --gt <gt.csv>` |
| calibration | mean ECE < 0.05 **[choice]** | `python -m mtah.deadball.evaluate --all <manifest.csv>` |
| module | beats the v0 stub on mean `alive_dead_f1` and D-interval F1 by > 0.02, no bootstrap overlap (guide `:229-230`) | `python -m mtah.deadball.evaluate --all <manifest.csv>`; `--bootstrap` not built yet (decision (b)) |

## 5. Synthetic tests (no ground truth needed)

- **Gap**:
  - ball visible, then 100 NaN frames at 25 fps, then visible → `gap_len_s` = log(5) inside the gap and 0 outside;
  - `t_since_seen_s` rises from log(1.04) to log(5), strictly increasing;
  - a gap crossing a half edge is split in two.
- **Line**:
  - ball at (L+2, W/2) → `ball_out_m` = 2; at (L/2, W/2) → 0; at (−10, W/2) → `OUT_CAP_M`;
  - a gap after a ball last seen at y = 0.5 m → `last_line_dist_m` = 0.5 for the whole gap.
- **Motion**:
  - 3 m/s for 20 s, then 0.5 m/s → `speed_abs` crosses 1.75 within ±0.5 s of the change, and `speed_rel` < 0.3 just after it;
  - all players NaN → feature NaN and indicator = 1, never an exception.
- **Units and orientation**:
  - x mirrored → identical features;
  - a pitch of 100 × 60 moves the line features with it (no 105.3 or 68 in code);
  - fps = 30 keeps durations in seconds.
- **Combiner**: separable toy data → values in [ε, 1−ε], same sha256 twice.
- **Calibration**: 100,000 labels drawn as Bernoulli(p), p ~ Uniform(0,1), seed 0 → ECE < 0.02 (measured 0.003);
  p replaced by 0.8·p + 0.2 → ECE > 0.05 (expected 0.10). Not 0.9·p + 0.1: its expected ECE is exactly 0.05.
- **Metrics**: D shifted 25 frames late → median onset offset = +25; one extra D → D-interval precision < 1.
- **camera_step**: constant homography → 0 everywhere; invalid frames → NaN.

## 6. Open questions for Alexia (P3b)

1. **`anchors.json`, `frame`:** is it the first live frame (the restart is taken) or the moment the ball is placed? Are anchors only inside halves?
2. ~~**`tags.json`, `end_frame`:** keep it exclusive? Is `team` the raw tag or flipped on DEFENDING codes?~~
   **Answered 2026-11-07 (Alexia):** `end_frame` is exclusive, as the official loader. Both: `team` is the raw
   tag as H/A, `possession_team` is flipped for the four codes in `config.RESTARTS.TAG_DEFENDING_CODES`
   (a fixed rule from the code names; c4 agreement with the nearest-player team is 9–33 % for those codes).
3. **External games:** can the tracking-only parts of `restarts.py` (e.g. the kick-off detector, `c5_restarts.py:29-49`) run on P2's games? Otherwise P5 can't tune anchor weights anywhere.
4. **Dependency:** does `restarts.py` want `p_dead` as an input (e.g. look for restarts at the end of D runs)? If yes, `p_dead` runs first.
5. **`camera_step.npy`:** is it useful for restart detection too? Shall we agree its file name and units?
