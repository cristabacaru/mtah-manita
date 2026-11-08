"""Degrade a clean external Game so it looks like the competition's automatic tracking (ATD).

    g2 = degrade(game, level="target", seed=0)            # deterministic

Calibrated against aggregate statistics of the target ATD ($MTAH_CACHE/p2/target_stats.json, made by
`python -m mtah.external.stats --target`, aggregates only). The gap-process and tracklet parameters
are fitted to those aggregates by `python -m mtah.external.calibrate`, which writes
$MTAH_CACHE/p2/sim_params.json. Without target stats a PROVISIONAL target is used; without fitted
params the initial guesses are used. Both print a warning: suite numbers are then indicative only.

What it does, in order (all per half, all from one seeded RNG):
 1. Fill the ball where the clean data has none (Metrica drops it while dead). The ball is held
    where it was last seen, then moved to where it is next seen (the restart spot), then rests.
 2. Ball gaps: a two-state (seen / missing) process driven by context, never by labels.
      start hazard  h(t) = exp(A[cell] + B[ball speed] + C[crowd] + E[edge] + D[player speed])
                     (context at t-1, the same bins stats.py measures on the target)
      a gap is a short blip (length from the target's short-gap histogram) with prob p_blip,
      otherwise a long gap of at least `min_long` frames that ends at the first frame where
      u < r(t) = exp(R[player speed at t-1]).
    On the target the ball is missing 79% of the time when players are slow and 11% when they
    run, in long gaps (median ~11 s): the missing ball is a stoppage cue there too, and the
    simulator reproduces that strength (not more) through the player-speed end rates.
 3. Ball noise: the clean ball is smoothed or AR(1)-jittered until its second-difference spread
    matches the target; z kept on the target's share of frames (highest z).
 4. Players: positions smoothed / jittered to the target second-difference spread; each true
    track is cut into tracklets (fitted lifetime scale and detection-gap share so tracklets per
    minute and objects per frame match), packed into the target's per-team slots (least recently
    used first, like a tracker recycling ids), 1% get the wrong team; speed recomputed from the
    positions with a NaN prefix per tracklet to match the target's speed-NaN share.
Labels are copied unchanged: degradation changes the evidence, never the truth.
"""

import hashlib
import json
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from mtah.schema import Game, HOME, AWAY, runs, speed_from_xy  # noqa: E402
from mtah.external import stats as S  # noqa: E402

from mtah import config  # noqa: E402

TARGET_PATH = str(config.EXTERNAL.TARGET_STATS)
PARAMS_PATH = str(config.EXTERNAL.SIM_PARAMS)
FPS = 25

# Used only until target_stats.json exists. Rough guesses (marked ?) except where known.
PROVISIONAL = {
    "provisional": True,
    "ball_missing": 0.41,
    "gap_start_hazard": 0.0016,                                                        # ?
    "gap_hist": [0.01, 0.06, 0.01, 0.01, 0.01, 0.01, 0.0, 0.0, 0.26, 0.33, 0.29],      # ?
    "hazard_by_crowd": {"rate": [0.0013, 0.0014, 0.0024, 0.0041]},                     # ?
    "hazard_by_ball_speed": {"rate": [0.0042, 0.0013, 0.0003, 0.0001]},                # ?
    "hazard_by_edge": {"rate": [0.0058, 0.0032, 0.0019, 0.0009]},                      # ?
    "cells": {"names": list(S.CELLS), "hazard": [0.001, 0.0023, 0.009, 0.0043]},       # ?
    "missing_by_player_speed": {"rate": [0.79, 0.51, 0.21, 0.13, 0.11]},               # ?
    "ball_d2_std": 0.006, "ball_jump_gt3m": 0.0001, "ball_z_present": 0.08,           # ?
    "objects_mean": 20.5, "slots": 46, "slots_home": 23, "tracklets_per_min": 11.5,
    "tracklet_life_hist": [0.0, 0.002, 0.18, 0.26, 0.28, 0.28],                       # ?
    "player_d2_std": 0.0016, "player_speed_nan_share": 0.08,                          # ?
}

LEVELS = {  # multipliers relative to the target
    "light": {"gap": 0.7, "frag": 0.6, "noise": 0.7},
    "target": {"gap": 1.0, "frag": 1.0, "noise": 1.0},
    "heavy": {"gap": 1.4, "frag": 1.6, "noise": 1.5},
}
WRONG_TEAM = 0.01   # share of tracklets put in a slot of the other team (not measurable on target)
N_PS = len(S.PSPEED_BINS) - 1


def load_target():
    if os.path.exists(TARGET_PATH):
        with open(TARGET_PATH) as fh:
            return json.load(fh)
    warnings.warn("no {} -- using PROVISIONAL simulator targets".format(TARGET_PATH))
    return dict(PROVISIONAL)


def target_hash(target):
    return hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()[:12]


def _rng(name, level, seed):
    h = hashlib.sha256("{}|{}|{}".format(name, level, seed).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _sample_from_hist(rng, hist, bins, size, cap):
    """Integers drawn bin-by-probability, then log-uniform inside [lo, hi)."""
    p = np.asarray(hist, float)
    p = p / p.sum()
    k = rng.choice(len(p), size=size, p=p)
    lo = np.asarray(bins[:-1], float)[k]
    hi = np.minimum(np.asarray(bins[1:], float)[k], cap)
    x = np.exp(rng.uniform(np.log(lo), np.log(np.maximum(hi, lo + 1))))
    return np.maximum(1, np.floor(x)).astype(int)


def _rates(table, k):
    """Target rate table -> float array of length k, NaN where missing."""
    r = [x if x is not None else np.nan for x in (table or {}).get("rate", [])][:k]
    return np.array(r + [np.nan] * (k - len(r)), float)


# ------------------------------------------------------------------ parameters
def split_gap_hist(target):
    """(short-gap histogram, min_long, p_blip) from the target gap-length histogram: gaps before
    the longest run of empty bins are blips (detector dropouts), the rest are long gaps."""
    h = np.asarray(target.get("gap_hist") or PROVISIONAL["gap_hist"], float)
    best, cur, end = 0, 0, None
    for i, v in enumerate(h):
        cur = cur + 1 if v == 0 else 0
        if cur > best:
            best, end = cur, i
    if end is None:                                   # no empty bins: blips are < 25 frames
        cut = S.GAP_BINS.index(25)
        k_long = cut
    else:
        cut = end - best + 1
        k_long = end + 1
    short = h[:cut]
    p_blip = float(short.sum() / max(h.sum(), 1e-9))
    return (short / short.sum() if short.sum() > 0 else None), int(S.GAP_BINS[k_long]), p_blip


def initial_params(target):
    short, min_long, p_blip = split_gap_hist(target)
    cell = _rates(target.get("cells"), 4) if "cells" in target else np.full(4, np.nan)
    base = target.get("gap_start_hazard", 0.0016)
    cell = np.where(np.isfinite(cell) & (cell > 0), cell, base)
    return {"A": np.log(cell).tolist(), "B": [0.0] * 4, "C": [0.0] * 4, "E": [0.0] * 4,
            "D": [0.0] * N_PS, "R": [float(np.log(1 / 300))] * N_PS,
            "p_blip": p_blip, "min_long": min_long,
            "short_hist": None if short is None else short.tolist(),
            "life_scale": 1.0, "drop_share": 0.07, "fitted": False}


def load_params(target):
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as fh:
            p = json.load(fh)
        if p.get("target_hash") != target_hash(target):
            warnings.warn("sim_params.json was fitted to a different target -- rerun "
                          "python -m mtah.external.calibrate")
        return p
    warnings.warn("no {} -- using UNFITTED simulator parameters".format(PARAMS_PATH))
    return initial_params(target)


# ------------------------------------------------------------------ 1. fill the ball
PARKED_OUT = 25    # frames; a clean ball outside the pitch for longer is treated as not seen


def fill_ball(g):
    """Ball xy everywhere inside the halves: hold / travel / rest inside clean-data gaps.

    Clean-ball runs outside the pitch longer than PARKED_OUT frames count as gaps too: DFL keeps
    the ball parked 0.2-1.3 m outside the line through most stoppages (~27-31% of frames), while
    the target's ball is seen outside on ~4% of frames and Metrica's on ~3%. Filling makes the ball
    wait at the line and then move to the restart spot, as it does in Metrica. Brief exits (the
    ball actually going out) are kept."""
    ball = g.ball[:, :2].astype(np.float64).copy()
    L, W = g.pitch
    out = (ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W)
    st, en, v = runs(out)
    for s0, e0, o in zip(st, en, v):
        if o and e0 - s0 > PARKED_OUT:
            ball[s0:e0] = np.nan
    for a, b in g.halves:
        seg = ball[a:b]
        seen = np.isfinite(seg[:, 0])
        if not seen.any():
            continue
        st, en, v = runs(seen)
        for s, e, vis in zip(st, en, v):
            if vis:
                continue
            prev = seg[s - 1] if s > 0 else seg[e]
            nxt = seg[e] if e < len(seg) else prev
            L = e - s
            t = np.arange(L) / max(1, L)
            # hold until 40%, travel until 85%, rest at the restart spot
            w = np.clip((t - 0.4) / 0.45, 0, 1)[:, None]
            seg[s:e] = prev * (1 - w) + nxt * w
    return ball


# ------------------------------------------------------------------ 2. ball gap process
def gap_context(g, ball):
    """Per-frame context bins at t (same definitions as stats.aggregate_stats)."""
    L, W = g.pitch
    P = g.players
    d = np.hypot(P[..., 0] - ball[:, None, 0], P[..., 1] - ball[:, None, 1])
    crowd = np.sum(d < 3.0, axis=1)
    vel = np.zeros(g.n)
    for a, b in g.halves:
        vel[a + 1:b] = np.nan_to_num(np.hypot(np.diff(ball[a:b, 0]), np.diff(ball[a:b, 1])) * FPS)
    edge = np.nan_to_num(np.minimum.reduce([ball[:, 0], L - ball[:, 0], ball[:, 1],
                                            W - ball[:, 1]]), nan=50)
    pspeed = np.nan_to_num(S.mean_player_speed(g))
    return {"cell": S.context_cell(edge < 0, pspeed),
            "bs": np.digitize(vel, S.BSPEED_BINS[1:-1]),
            "crowd": np.digitize(crowd, S.CROWD_BINS[1:-1]),
            "edge": np.digitize(edge, S.EDGE_BINS[1:-1]),
            "ps": np.digitize(pspeed, S.PSPEED_BINS[1:-1])}


def _prev(x):
    return np.r_[x[:1], x[:-1]]


def hazards(ctx, p, mult=1.0):
    c = {k: _prev(v) for k, v in ctx.items()}
    lh = (np.asarray(p["A"])[c["cell"]] + np.asarray(p["B"])[c["bs"]] +
          np.asarray(p["C"])[c["crowd"]] + np.asarray(p["E"])[c["edge"]] +
          np.asarray(p["D"])[c["ps"]])
    lr = np.asarray(p["R"])[c["ps"]]
    return np.exp(lh) * mult, np.minimum(1.0, np.exp(lr) / mult)


def gap_draws(rng, n, p):
    """The random numbers of the gap process (drawn once, so a fit uses common random numbers)."""
    short = p.get("short_hist")
    blip = (_sample_from_hist(rng, short, S.GAP_BINS[:len(short) + 1], n, cap=10 ** 6)
            if short else np.ones(n, int))
    return {"s": rng.random(n), "e": rng.random(n), "k": rng.random(n), "blip": blip}


def run_gaps(halves, h, r, u, p):
    """missing mask of the two-state process."""
    n = len(h)
    end_ev = u["e"] < r
    idx = np.where(end_ev, np.arange(n), n)
    nxt = np.minimum.accumulate(idx[::-1])[::-1]          # first end event at or after t
    start_ev = u["s"] < h
    missing = np.zeros(n, bool)
    p_blip, min_long = p["p_blip"], int(p["min_long"])
    for a, b in halves:
        cand = np.flatnonzero(start_ev[a + 1:b]) + a + 1
        free = a + 1
        for t in cand:
            if t < free:
                continue
            if u["k"][t] < p_blip:
                e = t + int(u["blip"][t])
            else:
                m = t + min_long
                e = int(nxt[m]) if m < b else b
            e = min(e, b)
            missing[t:e] = True
            free = e + 1                                  # at least one seen frame between gaps
    return missing


def gap_measure(missing, ctx, halves, n):
    """Counts (num, den) per bin with the definitions of stats.aggregate_stats; poolable."""
    half = np.zeros(n, np.int8)
    for i, (a, b) in enumerate(halves, 1):
        half[a:b] = i
    graded = half > 0
    seen = ~missing
    prev_seen = np.r_[False, seen[:-1]]
    same_half = np.r_[False, half[1:] == half[:-1]] & graded
    at_risk = graded & prev_seen & same_half
    start = at_risk & ~seen
    in_gap = graded & ~prev_seen & same_half
    out = {"start": [float(start.sum()), float(at_risk.sum())],
           "miss": [float((~seen & graded).sum()), float(graded.sum())],
           "n_gaps": float((start).sum())}
    sizes = {"cell": 4, "bs": 4, "crowd": 4, "edge": 4, "ps": N_PS}
    for k, size in sizes.items():
        c = _prev(ctx[k])
        out["h_" + k] = [np.bincount(c[start], minlength=size).astype(float).tolist(),
                         np.bincount(c[at_risk], minlength=size).astype(float).tolist()]
    c = _prev(ctx["ps"])
    out["end_ps"] = [np.bincount(c[in_gap & seen], minlength=N_PS).astype(float).tolist(),
                     np.bincount(c[in_gap], minlength=N_PS).astype(float).tolist()]
    out["occ_ps"] = [np.bincount(ctx["ps"][graded & ~seen], minlength=N_PS).astype(float).tolist(),
                     np.bincount(ctx["ps"][graded], minlength=N_PS).astype(float).tolist()]
    gl = []
    for a, b in halves:
        st, en, v = runs(seen[a:b])
        gl += list((en - st)[~v & (st > 0)])
    out["gap_len"] = gl
    outside = ctx["edge"] == 0
    out["out_seen"] = [float((graded & seen & outside).sum()), float((graded & seen).sum())]
    return out


def ball_dropout(g, ball, p, rng, mult=1.0, ctx=None):
    ctx = ctx if ctx is not None else gap_context(g, ball)
    h, r = hazards(ctx, p, mult)
    return run_gaps(g.halves, h, r, gap_draws(rng, g.n, p), p)


# ------------------------------------------------------------------ 3/4. noise matching
def robust_d2(x, mask):
    """stats-style robust std of the second difference of x (n,) or (n, K) on rows in mask."""
    d2 = (x[2:] - 2 * x[1:-1] + x[:-2])[mask[1:-1]]
    return S._robust_std(np.asarray(d2).ravel()) or 0.0


def _gauss_nan(x, halves, sigma):
    """Gaussian smoothing along time that ignores NaN (normalised convolution), per half."""
    out = x.copy()
    for a, b in halves:
        seg = x[a:b]
        m = np.isfinite(seg)
        num = gaussian_filter1d(np.where(m, seg, 0.0), sigma, axis=0, mode="nearest")
        den = gaussian_filter1d(m.astype(float), sigma, axis=0, mode="nearest")
        with np.errstate(all="ignore"):
            out[a:b] = np.where(m, num / den, np.nan)
    return out


def _ar1(rng, shape, sigma_d2, rho=0.8):
    """AR(1) noise along axis 0 whose second difference has std ~ sigma_d2."""
    s = sigma_d2 / np.sqrt(6 - 8 * rho + 2 * rho ** 2)
    e = rng.normal(0, s * np.sqrt(1 - rho ** 2), shape)
    return lfilter([1.0], [1.0, -rho], e, axis=0)


def match_d2(xs, halves, mask, target_sd, rng):
    """Smooth (if too rough) or jitter (if too smooth) the coordinate arrays in xs so the robust
    second-difference spread of xs[0] matches target_sd. Returns new arrays and what was done."""
    cur = robust_d2(xs[0], mask)
    if target_sd is None or target_sd <= 0:
        return xs, {"d2_clean": cur}
    if cur > target_sd * 1.05:
        lo, hi = 0.0, 12.0
        for _ in range(14):
            mid = (lo + hi) / 2
            sub = _gauss_nan(xs[0], halves, mid) if mid > 0 else xs[0]
            if robust_d2(sub, mask) > target_sd:
                lo = mid
            else:
                hi = mid
        sig = (lo + hi) / 2
        return [_gauss_nan(x, halves, sig) for x in xs], {"d2_clean": cur, "smooth_sigma": sig}
    if cur < target_sd * 0.95:
        add = np.sqrt(target_sd ** 2 - cur ** 2)
        return [x + _ar1(rng, x.shape, add) for x in xs], {"d2_clean": cur, "jitter_d2": add}
    return xs, {"d2_clean": cur}


# ------------------------------------------------------------------ 4. players
def fragment_players(g, xy, target, p, rng, frag=1.0):
    """Cut true tracks (xy: (n, S0, 2)) into tracklets and pack them into the target's slots."""
    n = g.n
    n_slots = int(target.get("slots", 46))
    n_home = int(target.get("slots_home", n_slots // 2))
    slot_team = np.r_[np.full(n_home, HOME), np.full(n_slots - n_home, AWAY)].astype(np.int8)
    life_hist = target["tracklet_life_hist"]
    drop_share = min(0.6, p["drop_share"] * frag)
    scale = p["life_scale"] / frag

    tracklets = []    # (start, stop, player)
    for k in range(xy.shape[1]):
        vis = np.isfinite(xy[:, k, 0])
        for a, b in g.halves:
            st, en, v = runs(vis[a:b])
            for s, e in zip(st[v] + a, en[v] + a):
                t = s
                while t < e:
                    life = int(_sample_from_hist(rng, life_hist, S.LIFE_BINS, 1, cap=60000)[0])
                    life = max(1, int(life * scale))
                    stop = min(e, t + life)
                    tracklets.append((t, stop, k))
                    mean_gap = life * drop_share / max(1e-6, 1 - drop_share)
                    gap = int(rng.exponential(mean_gap)) if mean_gap > 0 else 0
                    t = stop + max(gap, 1)

    tracklets.sort()
    out = np.full((n, n_slots, 2), np.nan, np.float64)
    free_at = np.zeros(n_slots, np.int64)
    placed, dropped = [], 0
    for s, e, k in tracklets:
        team = g.team[k]
        if rng.random() < WRONG_TEAM:
            team = 1 - team
        cand = np.flatnonzero((slot_team == team) & (free_at <= s))
        if len(cand) == 0:
            dropped += 1
            continue
        j = cand[np.argmin(free_at[cand])]           # least recently used free slot
        free_at[j] = e + 1
        out[s:e, j] = xy[s:e, k]
        placed.append((s, e, j))
    speed = speed_from_xy(out, g.halves)
    # speed NaN at the start of each tracklet (the target's speed field is NaN on ~8% of frames)
    share = float(target.get("player_speed_nan_share", 0.0))
    if placed and share > 0:
        lives = np.array([e - s for s, e, _ in placed])
        goal = share * lives.sum()
        lo, hi = 0, int(lives.max())
        while lo < hi:                       # smallest K with sum(min(life, K)) >= goal
            mid = (lo + hi) // 2
            if np.minimum(lives, mid).sum() >= goal:
                hi = mid
            else:
                lo = mid + 1
        K = lo
        for s, e, j in placed:
            speed[s:min(e, s + K), j] = np.nan
    players = np.concatenate([out, speed[..., None]], axis=2).astype(np.float32)
    return players, slot_team, {"tracklets": len(tracklets), "dropped_no_slot": dropped,
                                "drop_share": round(float(drop_share), 3)}


# ------------------------------------------------------------------ driver
def degrade(g, level="target", seed=0, target=None, params=None):
    target = target or load_target()
    p = params or load_params(target)
    lv = LEVELS[level]
    rng = _rng(g.name, level, seed)
    graded = g.graded()

    # 1-2: ball presence (context from the clean data, labels never used)
    ball_xy = fill_ball(g)
    missing = ball_dropout(g, ball_xy, p, rng, lv["gap"])

    # 3: ball noise matched to the target, z on the target's share of frames (highest z)
    bxy, binfo = match_d2([ball_xy[:, 0], ball_xy[:, 1]], g.halves, graded & ~missing,
                          target.get("ball_d2_std", 0.006) * lv["noise"], rng)
    ball = np.full((g.n, 3), np.nan, np.float32)
    ball[:, 0], ball[:, 1] = bxy
    z = g.ball[:, 2].astype(np.float64)
    zshare = float(target.get("ball_z_present", 0.0))
    if np.isfinite(z).any() and zshare > 0:
        thr = np.nanquantile(z[graded], 1 - zshare)
        keep = np.isfinite(z) & (z >= thr)
        ball[keep, 2] = z[keep]
    ball[missing] = np.nan
    ball[~graded] = np.nan

    # 4: players
    pxy, pinfo = match_d2([g.players[..., 0].astype(np.float64),
                           g.players[..., 1].astype(np.float64)], g.halves, graded,
                          target.get("player_d2_std", 0.0016) * lv["noise"], rng)
    players, slot_team, info = fragment_players(g, np.stack(pxy, axis=2), target, p, rng,
                                                lv["frag"])

    meta = dict(g.meta)
    meta.update({"degraded": {"level": level, "seed": seed, **info,
                              "ball_noise": {k: round(v, 4) for k, v in binfo.items()},
                              "player_noise": {k: round(v, 4) for k, v in pinfo.items()},
                              "provisional_target": bool(target.get("provisional", False)),
                              "fitted_params": bool(p.get("fitted", False))},
                 "clean": False})
    out = Game(name="{}__{}".format(g.name, level), players=players, team=slot_team, ball=ball,
               halves=list(g.halves), pitch=g.pitch,
               home_attacks_right_first_half=g.home_attacks_right_first_half,
               labels=None if g.labels is None else g.labels.copy(), fps=g.fps, meta=meta)
    return out.validate()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--level", default="target")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    src = Game.load(args.game)
    d = degrade(src, args.level, args.seed)
    tgt = load_target()
    S.print_compare(S.compare(S.aggregate_stats(d), tgt))
    print(d.meta["degraded"])
