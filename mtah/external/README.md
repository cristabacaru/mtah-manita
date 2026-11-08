# mtah/external: external games, simulator, validation suite (P2)

Owner: Theo. Log of every step and decision: `research/WORKLOG_P2.md` (steps 4–8).

## What you get

| what | where | read it with |
|---|---|---|
| 11 clean external games in the shared schema | `$MTAH_CACHE/<game>/` | `mtah.data.load("<game>")` |
| 27 suite games (9 games × light/target/heavy) in the shared schema | `$MTAH_CACHE/<game>__<level>/` | `mtah.data.load("<game>__<level>")` |
| per-frame labels H/A/D/X of every external game | `$MTAH_CACHE/<game>/labels.npy` | `np.load(...)` |
| GT for the scorer (guide §4) | `data/validation/<game>/<game>_gt.csv` + `_gt_meta.json` | `run_evaluation.py --gt` |
| suite games as competition-format ATD | `data/validation/<game>__<level>/<game>__<level>_atd.{txt,xml}` | any ATD parser |
| suite list with leave-one-match-out folds | `data/validation/suite_manifest.json` | `build_suite.load_manifest()` |

Everything follows guide §5: metres from a corner, H attacks towards x = L in every half (180°
rotation, as kloppy `STATIC_HOME_AWAY`), y in the ATD direction, speed from `data.position_speed`
(NaN above 12 m/s), teams `"H"`/`"A"`, halves inclusive. All paths come from `config.EXTERNAL`.

## Score your module on the suite (the development metric, guide §6)

```bash
# one CSV per suite game, named <game>__<level>.csv, in the submission format, then:
python -m mtah.external.score_suite --results <dir> --owner <you> --change "what changed" --bootstrap 200
```
It prints per-game scores and the mean per level with a 95% CI over games, and appends a row to
`experiments.csv`. Folds: `fold` in the manifest (each source match is one fold, its 3 levels share
it): train on 8 folds, score the 9th.

**Calibration reference** (official baseline, smooth 50; `score_suite --make-baseline <dir>`):
real BAR–ATH spellF1 0.243 / macroF1 0.672. Suite (2026-11-07): `light` 0.282 / 0.611,
`target` 0.270 / 0.641, `heavy` 0.235 / 0.636. The real match sits between `target` and
`heavy`: report both, treat `heavy` as the conservative number.

## Practice matches

| game | competition | match | GT |
|---|---|---|---|
| dfl_J03WMX | Bundesliga 22/23, MD 34 | 1. FC Köln – FC Bayern München (1:2) | DFL flags |
| dfl_J03WN1 | Bundesliga 22/23, MD 34 | VfL Bochum – Bayer 04 Leverkusen (3:0) | DFL flags |
| dfl_J03WOH | 2. Bundesliga 22/23, MD 6 | Fortuna Düsseldorf – Jahn Regensburg (4:0) | DFL flags |
| dfl_J03WOY | 2. Bundesliga 22/23, MD 8 | Fortuna Düsseldorf – Hansa Rostock (3:1) | DFL flags |
| dfl_J03WPY | 2. Bundesliga 22/23, MD 12 | Fortuna Düsseldorf – 1. FC Nürnberg (0:1) | DFL flags |
| dfl_J03WQQ | 2. Bundesliga 22/23, MD 15 | Fortuna Düsseldorf – FC St. Pauli (1:0) | DFL flags |
| dfl_J03WR9 | 2. Bundesliga 22/23, MD 17 | Fortuna Düsseldorf – 1. FC Kaiserslautern (1:2) | DFL flags |
| metrica1_events | Metrica sample game 1 (anonymised) | – | event log, `events` convention |
| metrica2_events | Metrica sample game 2 (anonymised) | – | event log, `events` convention |

`metrica{1,2}_control` (uncontrolled touches merged, guide §4.2) are exported as a robustness
variant, not in the suite. Five of the seven DFL matches are Fortuna Düsseldorf home games:
look at per-game numbers too.

## Facts the other modules should know

- **On BAR–ATH the ball goes missing mostly while play is stopped**: 79% of frames are missing
  when the mean player speed is under 1 m/s, 11% when it is above 2.5 m/s; gaps are long (median
  268 frames). A strong dead-ball cue, but not perfect. The simulator reproduces it from context,
  never from labels (WORKLOG_P2 step 8).
- DFL keeps the ball parked just outside the line through stoppages; the simulator treats such
  runs (> 1 s outside) as unseen, as on BAR–ATH.
- DFL/Sportec y runs bottom to top; `dfl.py` converts it to the ATD direction (`y = W/2 - Y`).

## Rebuild everything (public data only)

```bash
pip install lxml numpy pandas scipy
git clone --depth 1 https://github.com/metrica-sports/sample-data.git data/external/metrica-sample-data
python -m mtah.external.dfl --download        # 7 DFL matches (~2.6 GB) -> data/external/dfl
python -m mtah.external.metrica               # -> $MTAH_CACHE/p2/games/metrica{1,2}_{events,control}
python -m mtah.external.dfl                   # -> $MTAH_CACHE/p2/games/dfl_<id>
python -m mtah.external.stats --target        # run LOCALLY by a human: BAR-ATH -> aggregates only
python -m mtah.external.calibrate             # fit the simulator -> $MTAH_CACHE/p2/sim_params.json (~6 min)
python -m mtah.external.build_suite           # 27 games -> data/validation/<game>__<level>/ (~10 min)
python -m mtah.external.export                # shared schema -> $MTAH_CACHE/<game>[__<level>]/
```

## Files

| file | does |
|---|---|
| `metrica.py`, `dfl.py` | load the clean games + GT into `mtah.schema.Game` (internal format: `game.npz` + `game.json`) |
| `stats.py` | aggregate tracking-quality statistics of any game (the only thing read from BAR–ATH) |
| `../simulate.py` | degrade a clean game to look like BAR–ATH's ATD (ball gaps, noise, tracklets) |
| `calibrate.py` | fit the simulator's parameters to the BAR–ATH aggregates |
| `build_suite.py` | degraded copies + ATD + GT + manifest |
| `export.py` | `Game` -> shared schema through `mtah.data.save` |
| `score_suite.py` | official scorer on every suite game, CI over games, `experiments.csv` row |
| `atd_io.py` | read / write the competition ATD format |
| `../../tools/` | Metrica event -> label converter, label checks, GT video viewer |

## Data sources and licences
- DFL/IDSSE open data: Bassek, Weber, Rein, Memmert (2025), *An integrated dataset of spatiotemporal
  and event data in elite soccer*, Scientific Data 12, 195. CC BY 4.0. GT = provider flags
  `BallStatus` / `BallPossession`.
- Metrica Sports sample data (github.com/metrica-sports/sample-data), games 1–2. GT derived from the
  human event log by `tools/events_to_labels.py`.
- No data from BAR–ATH is used for training or labels. The simulator uses only aggregate statistics
  of its tracking quality.
