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
    pass


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
    # Confidence bands (option B, 2026-11-08): recognised type 0.75-1.0, "unknown" 0.25-0.5, each cue
    # (slow_before, ball_played) adds CONF_STEP. Still-ball candidates of unknown type are dropped.
    # Evidence: research/checks/c9_restarts_vs_tags.py.
    CONF_KNOWN = 0.75
    CONF_UNKNOWN = 0.25
    CONF_STEP = 0.125
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


# =================================================================================================
# P4 team-possession model (Bianca)
# =================================================================================================
class POSSESSION:
    pass


# =================================================================================================
# P5 decoder and tuning (Andra)
# =================================================================================================
class DECODER:
    pass


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
