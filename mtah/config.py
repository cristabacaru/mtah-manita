"""Every setting the pipeline uses, in one place, one section per owner.

    from mtah import config
    config.CACHE_DIR, config.game_cache("BAR-ATH"), config.DEADBALL.GAP_CAP_S, ...

Check what it resolves to on your machine:

    python -m mtah.config

Rules (docs/LABELLING_GUIDE.md):
- Modules work on any game, so they read fps, pitch and halves from that game's `meta`
  (guide section 5), never from the TARGET GAME block below. That block is only for the BAR-ATH
  loader (data.py) and the submission writer (submit.py).
- No frame numbers written in here or in any module: they come from games.py (official, unedited).
- Add your module's settings in your own section, with the default the spec agreed on.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# =================================================================================================
# P1 platform (Alexia): paths, target game, submission, seed
# =================================================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent

# games.py lives at the repo root; make it importable however this module is run.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import games  # noqa: E402

# Must match the registration form exactly: the scorer names our entry after the CSV file name.
TEAM_NAME = "Manita"

# -- target game: BAR-ATH. Only data.py and submit.py read these. -------------------------------
GAME = games.GAME
SOURCE = games.DEFAULT_SOURCE                       # tactical: the submission is indexed to it
FPS = games.GAMES[GAME]["fps"]
N_FRAMES = games.source_frames(SOURCE, GAME)        # rows in the submission
PITCH = tuple(games.GAMES[GAME]["pitch"])           # (length, width) in metres
# Graded video frames, inclusive: [(half, first, last), ...]. Outside them the submission is X.
HALF_BLOCKS = games.half_video_blocks(GAME)
N_GRADED = sum(last - first + 1 for _, first, last in HALF_BLOCKS)

# The tracking file numbers its frames from 1; video and submission count from 0
# (starter/load_atd.py): video_frame = atd_frame_id - ATD_FRAME_ID_START.
ATD_FRAME_ID_START = 1

# -- data layer (mtah/data.py, guide section 5) --------------------------------------------------
# players_speed = |xy(t+k) - xy(t-k)| / 2k frames, k = round(SPEED_HALF_WINDOW_S * fps).
# 0.2 s: research/checks/c8_speed.py shows it matches the ATD speed column (median ratio 0.995).
SPEED_HALF_WINDOW_S = 0.2
# Speeds above this are set to NaN: a slot handed to another person (slots are not people, guide 5)
# shows up as a jump, up to 99.5 m/s on BAR-ATH. 12 m/s is a physical limit (fastest human sprint
# about 12.4 m/s), not a property of one game; the ATD's own speed column never exceeds 9.43.
SPEED_MAX_MPS = 12.0
# data.save() rejects a game where more than OUTSIDE_MAX_SHARE of player positions lie more than
# OUTSIDE_MARGIN_M outside [0, L] x [0, W]: catches a loader that kept the origin at the centre.
# BAR-ATH: 0.0024 % beyond 5 m; a centre origin would put about 37 % there (guide section 5).
OUTSIDE_MARGIN_M = 5.0
OUTSIDE_MAX_SHARE = 0.01

# -- labels (guide section 1) ------------------------------------------------------------------
LABELS = ("H", "A", "D", "X")   # X only outside the halves: on a graded frame it is a miss

# -- input data (gated, never committed) ---------------------------------------------------------
DATA_DIR = Path(games.DATA_DIR)                     # ./data, or $MTAH_DATA_DIR
ASSETS = {name: Path(p) for name, p in games.asset_paths(GAME, source=SOURCE).items()}
# keys: video, atd_meta, atd_raw, homography, smart_tagging
VALIDATION_DIR = DATA_DIR / "validation"            # external GT: <game>/<game>_gt.csv (guide 4)

# -- cache: everything computed from match data (never committed) -------------------------------
CACHE_DIR = Path(os.environ.get("MTAH_CACHE") or Path.home() / ".cache" / "mtah2027").expanduser()
REPORTS_DIR = CACHE_DIR / "reports"                 # run_evaluation.py --out

# -- submission: repo root, as README "Before submitting" expects; ignored by .gitignore ----------
SUBMISSION_CSV = REPO_ROOT / f"{TEAM_NAME}.csv"


def game_cache(game: str = GAME) -> Path:
    """Folder for one game's arrays (p_dead.npy, p_home.npy, labels.npy, ...). Works for any game."""
    return CACHE_DIR / game


# -- experiment log (committed) ------------------------------------------------------------------
EXPERIMENTS_CSV = REPO_ROOT / "experiments.csv"

# -- reproducibility: two runs must give the same sha256 (guide section 2) -----------------------
# 0 to match what already exists: the official scorer's BOOTSTRAP_SEED, optuna_tune.py --seed,
# and docs/deadball_SPEC.md.
SEED = 0


def set_seeds(seed: int = SEED) -> None:
    """Seed python and numpy. LightGBM and Optuna take the seed as an argument: pass config.SEED."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


# =================================================================================================
# P2 external data and simulator (Theo)
# =================================================================================================
class EXTERNAL:
    """Paths and settings of mtah/external/ and mtah/simulate.py (guide sections 4 and 5).

    Raw downloads sit in data/external/ (gated, never committed). Everything computed goes under
    $MTAH_CACHE: the internal mtah.schema.Game caches in p2/games/, the target aggregates and fitted
    simulator parameters in p2/, and each game in the shared schema in $MTAH_CACHE/<game>/ (data.save).
    GT and the competition-format ATD of every validation game go to data/validation/<game>/ (guide 4).
    """
    RAW_DIR = DATA_DIR / "external"                     # metrica-sample-data/, dfl/
    METRICA_DIR = RAW_DIR / "metrica-sample-data" / "data"
    DFL_DIR = RAW_DIR / "dfl"
    P2_DIR = CACHE_DIR / "p2"
    GAMES_DIR = P2_DIR / "games"                        # clean external games, mtah.schema.Game
    TARGET_STATS = P2_DIR / "target_stats.json"         # BAR-ATH aggregates (stats.py --target)
    SIM_PARAMS = P2_DIR / "sim_params.json"             # fitted by calibrate.py
    SUITE_MANIFEST = VALIDATION_DIR / "suite_manifest.json"
    # Validation suite: 7 DFL + 2 Metrica games (uncontrolled-touch convention "events", WORKLOG_P2
    # step 7) x 3 noise levels, leave-one-match-out folds.
    DFL_MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
    SUITE_GAMES = tuple("dfl_" + m for m in DFL_MATCHES) + ("metrica1_events", "metrica2_events")
    CLEAN_GAMES = SUITE_GAMES + ("metrica1_control", "metrica2_control")
    LEVELS = ("light", "target", "heavy")
    SEED = 0


# =================================================================================================
# P3a dead-ball p_dead (Cristiana). Defaults from docs/deadball_SPEC.md section 3.
# =================================================================================================
class DEADBALL:
    STUB_GAP_S = 3.0          # v0 placeholder: dead where the ball is missing for >= this many s
    # Gap features are log(1 + s), with no cap (spec section 3).
    OUT_CAP_M = 5.0           # cap on ball_out_m
    SPEED_SMOOTH_HALF_S = 0.5  # centred half-window smoothing speed_abs (not config.SPEED_HALF_WINDOW_S)
    REL_WIN_S = 10.0          # trailing window for speed_rel
    REL_FLOOR = 0.1           # m/s, floor on the speed_rel denominator
    L2 = 1.0                  # logistic regression L2 penalty
    EPS = 1e-3                # p_dead clipped to [EPS, 1 - EPS]
    HARD_CALL = 0.5           # evaluate.py: a frame is called dead when p_dead > HARD_CALL (spec 4)
    ECE_BINS = 10             # evaluate.py: equal-width calibration bins over [0, 1] (spec 4)
    OFFSET_IOU = 0.3          # evaluate.py: D runs paired for the boundary offset at IoU >= this (spec 4)
    SEED = 0


# =================================================================================================
# P3b Smart Tagging and restart anchors (Alexia)
# =================================================================================================
class RESTARTS:
    # tags.py: Smart Tagging codes whose Team tag names the DEFENDING side, so possession_team is
    # the other team. A fixed rule from the code names (decision 2026-11-07), not fitted; c4 agrees:
    # these four match the nearest-player team on 9-33% of tags, BUILD UP on 91%.
    TAG_DEFENDING_CODES = (
        "DEFENDING IN ATTACKING THIRD",
        "DEFENDING IN MIDDLE THIRD",
        "DEFENDING IN DEFENSIVE THIRD",
        "DEFENSIVE TRANSITION",
    )

    # restarts.py (decisions 2026-11-07: R1 frame = moment the ball is played, R2 tracking only,
    # R3 unclassified restarts kept as "unknown", R4 independent of p_dead). Values are [choice],
    # from the Laws of the Game, c5_restarts.py, or the measurement quoted; none is fitted.
    GAP_MIN_S = 3.0             # a ball-missing run this long is a stoppage candidate (c1, c5: 133 such gaps)
    STILL_MIN_S = 1.0           # ball visible and still this long is a stoppage candidate (c5 template A)
    STILL_DRIFT_M = 0.5         # "still": moved less than this over STILL_MIN_S (c5)
    BALL_PLAYED_MPS = 3.0       # restart frame = first frame the ball is faster than this. BAR-ATH: after a
                                # gap >= 3 s the ball passes 3 m/s in 93 % of cases, median 0.2 s after reappearing
    SEARCH_S = 10.0             # look for that movement up to this long after the candidate starts
    SLOW_MPS = 1.5              # team mean speed below this before a restart counts as "play was stopped" (c3)
    SLOW_BEFORE_S = 2.0         # ... measured over this long before the candidate starts
    MERGE_S = 3.0               # candidates closer than this are one restart
    WINDOW_S = 1.0              # anchors.json window: frame +/- this, the decoder's tolerance
    # Confidence v2 (2026-11-15): the measured hit rate of each evidence group on P2's suite (share
    # of anchors within 1 s of a true D -> H/A change, target + heavy pooled, 9 games; chance 3.6 %):
    # (type recognised, slow_before, ball_played) -> confidence. Groups not listed hit 0-3 %, about
    # chance, and are dropped. Evidence: mtah/deadball/evaluate_restarts.py, experiments.csv 2026-11-15.
    # Still-ball candidates still need slow_before and a recognised type (option B, 2026-11-08): the
    # shares were measured with that rule in place.
    CONFIDENCE = {
        (True, True, True): 0.86,     # 391 / 453
        (False, True, True): 0.37,    # 103 / 278
        (True, False, True): 0.27,    # 21 / 77
    }
    # Restart zones, Laws of the Game (corner arc 1 m, goal area 5.5 m x 18.32 m) plus c5's margins
    # for tracking noise.
    CORNER_R_M = 3.0
    THROW_IN_M = 2.0            # distance to a touchline (or past it)
    GOAL_AREA_DEPTH_M = 6.5     # 5.5 m + 1 m
    GOAL_AREA_HALF_WIDTH_M = 18.32 / 2 + 1.0
    CENTRE_R_M = 3.0
    # Kick-off formation (c5 kickoff_episodes, "loose" variant: found 5 of 6 tagged kick-offs, no false hits)
    KICKOFF_TOL_M = 2.0         # a player may be this far into the other half
    KICKOFF_MAX_VIOLATORS = 2
    KICKOFF_MIN_PLAYERS = 16
    KICKOFF_HOLD_S = 2.0
    # evaluate_restarts.py: an anchor matches a true D -> H/A change within this many seconds. Strict
    # = the anchors' own WINDOW_S. Loose, because providers differ on when "dead" ends. [choice]
    EVAL_TOL_S = (1.0, 3.0)


# =================================================================================================
# P4 team-possession model (Bianca)
# =================================================================================================
class POSSESSION:
    """p_home[t] = P(H in possession | ball alive). Files: ball.py, features.py, model.py."""
    # -- ball.py: gap bridging (constant-velocity Kalman + RTS smoother, per half) ---------------
    BRIDGE_MAX_S = 4.0        # fill a ball gap only if it is at most this long and seen at both ends
    # p4-finetuning, screened LOMO oracle-D (2026-11-17): 0.5 m / 8 -> 0.6424, 0.1 / 20 -> 0.7002,
    # 0.03 / 40 -> 0.7018, 0.01 / 60 -> 0.6875, 0.03 / 80 -> 0.6913. The old values smoothed kicks away
    # (P2 measured the real ball jitter at ~0.006 m second difference).
    KF_ACC_STD = 40.0         # m/s^2, white-acceleration process noise [main 8.0]
    KF_MEAS_STD = 0.03        # m, ball position measurement noise [main 0.5]
    # -- features.py ------------------------------------------------------------------------------
    VEL_HALF_WINDOW_S = 0.2   # player velocity vectors: same centred difference as players_speed
    NEAR_RADII_M = (2.0, 5.0)  # players of each team within these distances of the ball
    CONTROL_DIST_M = 1.5      # control event: nearest player of one team within this distance ...
    CONTROL_MARGIN_M = 1.0    # ... and closer than the other team's nearest by at least this ...
    CONTROL_REL_SPEED = 3.0   # ... and moving with the ball (|v_ball - v_player| below, m/s)
    CONTEXT_S = (0.5, 1.0, 3.0)  # past / future window lengths; 0.5 s for short possessions (p4-finetuning)
    # touch events (p4-finetuning): the ball changes velocity sharply next to a player. Short
    # possessions (< 3 s, 42 % of spells) are decided by single touches the control rule misses.
    # re-screened after the sharper KF (LOMO oracle-D, base 0.7029): window 0.08 -> 0.7096, 0.04 -> 0.7067;
    # dist 1.5 -> 0.7053; dv 2 -> 0.6982, 4.5 -> 0.7044; window 0.08 + dist 1.5 -> 0.7130 (+ dv 4.5: 0.7135)
    TOUCH_HALF_WINDOW_S = 0.08  # [was 0.12] dv = |v_ball(t + k) - v_ball(t - k)|, k = round(this * fps)
    TOUCH_DV_MPS = 3.0          # a touch is a local peak of dv above this ...
    TOUCH_DIST_M = 1.5          # [was 2.0] ... with the nearest player (either team) within this distance
    # training weights (p4-finetuning): frames where spellF1 is decided
    TURNOVER_WINDOW_S = 1.0     # frames within this of a direct H <-> A change in the GT ...
    TURNOVER_WEIGHT = 3.0       # ... get this weight
    SHORT_SPELL_S = 3.0         # frames inside GT possessions shorter than this ...
    SHORT_SPELL_WEIGHT = 3.0    # ... get this weight (the larger of the two applies)
    # turnover model p_turn (p4-finetuning): P(direct H <-> A change within TURN_WINDOW_S of t), same
    # features and LOMO as p_home; the decoder makes H <-> A switches cheaper where it is high
    TURN_MODEL = True
    TURN_WINDOW_S = 0.5
    SHAPE_VEL_S = 1.0         # team centroid velocity: centred difference over +/- this
    # -- model.py ---------------------------------------------------------------------------------
    V0_CONF = 0.8             # v0 rule: p_home = V0_CONF when the last control was H, 1 - V0_CONF if A
    TRAIN_STRIDE = 5          # train on every k-th alive frame (neighbours are near duplicates)
    MIRROR = True             # augment: 180-degree rotation + swap H/A (removes any home bias)
    TTA = True                # predict (p(x) + 1 - p(mirror x)) / 2: symmetric by construction
    CALIBRATE = True          # isotonic map fitted on out-of-fold predictions
    EPS = 1e-3                # p_home clipped to [EPS, 1 - EPS]
    LGBM = dict(
        # p4-finetuning capacity screen (LOMO oracle-D, base 0.7130): 800 rounds 0.7179, stride 2 0.7143,
        # 63 leaves / 600 rounds 0.7195, 63 leaves / 1000 rounds 0.7198 (plateau)
        objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,   # [was 31]
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
        num_threads=4, deterministic=True, force_col_wise=True, seed=0, verbose=-1,
    )
    N_ROUNDS = 600            # [was 400]
    SEED = 0
    MODEL_DIR = CACHE_DIR / "p4"                  # reports (report.json, ablation.json)
    # Trained model + calibration, COMMITTED so the pipeline does not retrain on every machine.
    # Learned from external games only (no BAR-ATH data); `python -m mtah.model --train` regenerates
    # them byte for byte from the same suite. p_home_meta.json says which suite that was.
    WEIGHTS_DIR = REPO_ROOT / "models"


# =================================================================================================
# P5 decoder and tuning (Andra)
# =================================================================================================
class DECODER:
    # p4-finetuning branch: retuned 2026-11-17 for the variant-C p_home (touch features + turnover /
    # short-spell weights), same method as on main (Optuna TPE seed 0, 60 trials, 27 suite copies,
    # tag_bonus fixed). Suite 0.5187 (main's decoder values) -> 0.5243; held-out by match 0.522 /
    # 0.522 against main's 0.511 / 0.507. main's values in brackets.
    EMISSION_SCALE = 0.265      # [main 0.387]
    SWITCH_TEAM = 2.367         # [main 1.991]
    SWITCH_DEAD = 18.305        # [main 25.692]
    MIN_LIVE_S = 0.639          # [main 0.296]
    MIN_DEAD_S = 1.333          # [main 0.968]
    ANCHOR_BONUS = 9.190        # [main 8.534]
    TAG_BONUS = 4.0             # not tuned: external games have no Smart Tagging
    # p4-finetuning (2026-11-17): only HOME_SCALE and TURN_BONUS tuned (Optuna TPE seed 0, 40 trials,
    # other params fixed). Held out by match: 0.5804 -> 0.5816, 0.5734 -> 0.5792; all 27: 0.5773 -> 0.5829
    HOME_SCALE = 0.121          # p4-finetuning: separate weight of the H-vs-A term; None = EMISSION_SCALE
    TURN_BONUS = 3.093          # p4-finetuning: H <-> A switch cheaper by TURN_BONUS * p_turn[t]; 0 = off
    SNAP_MIN_CONFIDENCE = 0.6       # excludes low-confidence "unknown" restart anchors
    SHORT_LIVE_SPELL_S = 0.6        # merge only same-team-bracketed H/A flickers


# =================================================================================================
# checks
# =================================================================================================
def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# Derived match data in the repo would be published the moment the repo goes public.
if _inside(CACHE_DIR, REPO_ROOT):
    raise RuntimeError(
        f"MTAH_CACHE points inside the repository ({CACHE_DIR}).\n"
        "Cached arrays are derived match data and must never be committed. Point it outside, e.g.\n"
        "  export MTAH_CACHE=~/.cache/mtah2027"
    )


def missing_assets(names=("atd_meta", "atd_raw", "homography", "smart_tagging")) -> list[Path]:
    """Which of the named input files are not on disk (empty list = all present)."""
    return [ASSETS[n] for n in names if not ASSETS[n].is_file()]


def summary() -> str:
    lines = [
        f"team            {TEAM_NAME}",
        f"game / source   {GAME} / {SOURCE}",
        f"frames          {N_FRAMES} at {FPS} fps, {N_GRADED} graded",
        "halves (video)  " + ", ".join(f"H{h} {a}..{b}" for h, a, b in HALF_BLOCKS),
        f"pitch           {PITCH[0]} x {PITCH[1]} m",
        f"data dir        {DATA_DIR}",
        f"cache dir       {CACHE_DIR}"
        + ("" if os.environ.get("MTAH_CACHE") else "   (default: MTAH_CACHE not set)"),
        f"submission      {SUBMISSION_CSV}",
        f"seed            {SEED}",
    ]
    for name, path in ASSETS.items():
        lines.append(f"  {name:<14}{'ok     ' if path.is_file() else 'MISSING'} {path.name}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
