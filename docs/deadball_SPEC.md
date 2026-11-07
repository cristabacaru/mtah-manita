# p_dead spec (P3a, Cristiana)

Scope: only `p_dead`. Other modules belong to their owners. Agreed 2026-11-07.
Sources are cited as `file:line`. Claims still unverified are listed in the session notes.

## 1. Decisions

| # | Decision | Reason |
|---|---|---|
| 1 | `p_dead` never snaps boundaries. The decoder (P5) alone places D edges, using anchors as costs. | One owner per edge. `p_dead` stays measurable without anchors (`README.md:94`). |
| 2 | Combiner = logistic regression, fitted leave-one-match-out on degraded external games, output clipped to [ε, 1−ε]. No isotonic step. | P5 applies one global `emission_scale` to log-probs (`mtah/optuna_tune.py:38,45`): it rescales log-odds, so linear log-odds fit. Isotonic gives exact 0/1 → −∞ in the Viterbi (`mtah/decode.py:60,137`). |
| 3 | Motion: absolute speed **and** speed relative to the trailing window, both as features. | BAR–ATH speed has no natural break (bimodality 0.515, c3 output); external games differ in tempo. |
| 4 | Continuous features, no hard "≥ 3 s" or "past line by X m" rules. | Lets the combiner learn cut-offs on external games; 7.9 % of graded frames are ball-missing but fast (c3 output). |
| 5 | `p_dead.npy` is tag-free. Tags and anchors go straight to the decoder (`decode.py:158`). | Their weight can't be learned: external games have no Smart Tagging (guide `:323-324`). |
| 6 | Own metrics mirror spellF1 (§4). Windows are centred (offline). | Same matching logic as the ranking metric; c3 already uses centred windows (`c3_dead_ball.py:12`). |

## 2. Inputs and outputs

**Input:** the per-game schema of guide §5: `ball_xy (n,2)` m, NaN = missing;
`players_xy (n,P,2)`, `players_speed (n,P)` m/s; `players_team (P,)` in {H, A}; `meta`: `pitch (L,W)`,
`fps`, `halves [(first,last)]` inclusive (`games.py:197-208`). H attacks towards +x. Only per-frame team
aggregates are used (guide `:260-262`). Every feature is computed inside one half; windows are cut at half edges.

**Output (P3a)**, in `$MTAH_CACHE/<game>/`:
- `p_dead.npy`: `(n,)` float64; values in [ε, 1−ε] inside halves, NaN outside halves.
- `deadball_features.npz`: the feature matrix `(n, k)` plus names, for P5's error analysis.

`p_dead` reads no anchors or tags, so it does not depend on P3b's formats.

## 3. Features (`mtah/deadball/signals.py`, `lines.py`; parameters in `mtah/config.py`, section `DEADBALL`)

| Feature | Detects | Inputs | Parameters (default) |
|---|---|---|---|
| `gap_len_s` | length of the ball-missing run containing t (0 if visible) | `ball_xy` | `GAP_CAP_S` (60) |
| `t_since_seen_s`, `t_until_seen_s` | position inside a gap (edges matter most) | `ball_xy` | `GAP_CAP_S` |
| `last_line_dist_m`, `next_line_dist_m` | ball last/next seen near a line → out of play or restart | `ball_xy`, `pitch` | none |
| `ball_out_m` | visible ball past a line: max(0, distance outside) | `ball_xy`, `pitch` | `OUT_CAP_M` (5) |
| `speed_abs` | team motion collapse: centred mean of mean player speed | `players_speed` | `SPEED_HALF_WIN_S` (0.5) |
| `speed_rel` | `speed_abs` / median over trailing window | `players_speed` | `REL_WIN_S` (10), `REL_FLOOR` (0.1 m/s) |
| `*_missing` | indicator where an input is NaN (feature then = training median) | all | none |

Combiner (`combine.py`): standardised features → logistic regression with L2, fitted by
`scipy.optimize.minimize(method="L-BFGS-B")` (scipy 1.18.1 pinned, `requirements-lock.txt:99`; scikit-learn
is not installed). Parameters: `L2` (1.0), `EPS` (1e-3), `SEED` (0). Training rows: graded frames of
degraded external games, label = provider dead flag. Until external games exist: **v0 stub**
`p_dead = clip(gap_len_s ≥ 3)` (`README.md:98-99`).

## 4. Evaluation (`mtah/deadball/evaluate.py`) and "done when"

Per external game × 3 noise levels, hard call `p_dead > 0.5`, per half:
- `alive_dead_f1` via `evaluation/possession_metrics.py:208` (ALIVE is the positive class).
- **D-interval F1**: one-to-one greedy IoU matching of D runs (same code path as `spell_scores`,
  `possession_metrics.py:472-542`, with D as the label), mean over IoU 0.3–0.7.
- **Boundary offset**: signed onset and end errors (frames) of D runs matched at IoU ≥ 0.3; median and p90.
- **Ablation**: drop one feature group at a time (gap / line / motion).
Report the mean over games × noise levels, log a row in `experiments.csv`. After P5 exists, re-score
through the decoder with `run_evaluation.py`.

| Piece | Done when | Command |
|---|---|---|
| `signals.py`, `lines.py` | all synthetic tests (§5) pass | `python tests/test_deadball.py` (plain asserts; pytest is not installed) |
| `combine.py` | two runs give the same sha256 of `p_dead.npy`; no value outside [ε, 1−ε] | `python -m mtah.deadball.combine --game <g>` twice, `shasum -a 256` |
| `evaluate.py` | GT scored against itself: all metrics 1.0, offsets 0 | `python -m mtah.deadball.evaluate --self-test <g>` |
| module | beats the v0 stub on mean `alive_dead_f1` and D-interval F1 by > 0.02, with no bootstrap overlap | `python -m mtah.deadball.evaluate --all` |

## 5. Synthetic tests (no ground truth needed)

- **Gap**: ball visible, 100 NaN frames, visible → `gap_len_s` = 4.0 s inside the gap and 0 outside;
  `t_since_seen_s` rises 0.04 → 4.0. A gap crossing a half edge is split in two.
- **Line**: ball at (L+2, W/2) → `ball_out_m` = 2; at (L/2, W/2) → 0; at (−10, …) → `OUT_CAP_M`.
- **Last/next seen**: gap starting after a ball at y = 0.5 m → `last_line_dist_m` = 0.5 for the whole gap.
- **Motion**: speeds 3 m/s for 20 s then 0.5 m/s → `speed_abs` falls within ±0.5 s of the change;
  `speed_rel` < 0.3 just after it; all-NaN players → NaN plus indicator = 1, never an error.
- **Orientation/units**: mirrored x (H/A swapped) gives identical `p_dead`; no output depends on 105.3 or 68.
- **Combiner**: perfectly separable toy data → p_dead within [ε, 1−ε], deterministic sha256.
- **Metrics**: D shifted 25 frames late → onset offset median = +25; one extra D → D-interval precision < 1.

## 6. What this part needs from others

- **External games with ground truth** (P2), to fit and score the combiner. Until then: the v0 stub.
- Nothing else. My settings go in the `DEADBALL` section of `mtah/config.py`.
