"""Aggregate tracking-quality statistics of a Game -- the simulator's calibration target.

The SAME function measures the competition ATD and the degraded external games, so the two are
comparable by construction. Output is aggregates only (rates, histograms, quantiles): no frame,
no position, no label ever leaves this module. That is what lets a teammate run it on the
competition data and share the JSON.

    python -m mtah.external.stats --target                 # competition ATD -> $MTAH_CACHE/p2/target_stats.json
    python -m mtah.external.stats --game $MTAH_CACHE/p2/games/dfl_J03WMX     # any cached Game
    python -m mtah.external.stats --compare a.json b.json  # ratio table
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from mtah import config  # noqa: E402
from mtah.schema import Game, runs  # noqa: E402

GAP_BINS = [1, 2, 3, 5, 8, 13, 25, 50, 100, 200, 500, 10 ** 9]        # frames, [lo, hi)
LIFE_BINS = [1, 5, 25, 125, 625, 3125, 10 ** 9]
CROWD_BINS = [0, 1, 2, 3, 99]          # players within 3 m of the ball
BSPEED_BINS = [0, 2, 8, 15, 999]       # ball speed m/s before the frame
EDGE_BINS = [-99, 0, 3, 10, 99]        # distance to nearest line, metres (<0 = ball outside)
PSPEED_BINS = [0, 1, 1.5, 2, 2.5, 99]  # mean player speed m/s
FPS = 25
# Context cells for the gap model: ball outside the pitch x players slow. Hazard AND gap length
# are measured per cell, so "long gaps while play is stopped" is carried over to the simulator
# without ever looking at a label.
SLOW = 1.5                             # m/s, mean player speed (from positions)
CELLS = ("in_fast", "in_slow", "out_fast", "out_slow")


def mean_player_speed(g):
    """(n,) mean speed of the visible players, from POSITIONS (not the provider speed field, whose
    units / smoothing may differ between sources)."""
    from mtah.schema import speed_from_xy
    sp = speed_from_xy(g.players[..., :2], g.halves, smooth=5)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        return np.nanmean(sp, axis=1)


def context_cell(outside, pspeed):
    return (2 * outside.astype(int) + (np.nan_to_num(pspeed, nan=0) < SLOW)).astype(int)


def _hist(x, bins):
    h = np.histogram(x, bins=bins)[0].astype(float)
    return [round(v, 4) for v in (h / max(1.0, h.sum()))]


def _rate_by(cond, x, bins):
    """P(cond | x in bin) and bin support share."""
    idx = np.digitize(x, bins[1:-1])
    out, sup = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        out.append(round(float(cond[m].mean()), 4) if m.sum() > 50 else None)
        sup.append(round(float(m.mean()), 4))
    return {"rate": out, "share": sup, "bins": bins}


def _ffill(x, halves):
    out = np.array(x, np.float64)
    for a, b in halves:
        seg = out[a:b]
        idx = np.where(np.isfinite(seg), np.arange(len(seg)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = seg[idx]
        filled[~np.isfinite(seg) & (idx == 0) & ~np.isfinite(seg[0])] = np.nan
        out[a:b] = filled
    return out


def _robust_std(x):
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    return round(float(1.4826 * np.median(np.abs(x - np.median(x)))), 4)


def aggregate_stats(g, frames=None):
    """dict of aggregate statistics over the graded frames of a Game."""
    L, W = g.pitch
    graded = g.graded() if frames is None else frames
    half = g.half_of()
    P = g.players
    bx, by = g.ball[:, 0], g.ball[:, 1]
    seen = np.isfinite(bx)
    pseen = np.isfinite(P[..., 0])
    s = {"n_graded": int(graded.sum()), "slots": int(P.shape[1])}

    # ---- ball presence and gaps
    s["ball_missing"] = round(float((~seen[graded]).mean()), 4)
    gaps = []
    vis_runs = []
    for a, b in g.halves:
        st, en, v = runs(seen[a:b])
        lens = en - st
        gaps += list(lens[~v])
        vis_runs += list(lens[v])
    gaps, vis_runs = np.array(gaps), np.array(vis_runs)
    s["gap_hist"] = _hist(gaps, GAP_BINS)
    s["gap_bins"] = GAP_BINS[:-1]
    s["gaps_per_min"] = round(len(gaps) / (graded.sum() / FPS / 60), 3)
    s["gap_q"] = {q: float(np.percentile(gaps, q)) for q in (50, 75, 90, 99)} if len(gaps) else {}
    s["visible_run_q"] = {q: float(np.percentile(vis_runs, q)) for q in (50, 90)} \
        if len(vis_runs) else {}

    # ---- context of gap starts (hazard: visible at t-1 -> missing at t)
    d = np.hypot(P[..., 0] - bx[:, None], P[..., 1] - by[:, None])
    crowd = np.sum(d < 3.0, axis=1).astype(float)
    vel = np.full(g.n, np.nan)
    for a, b in g.halves:
        vx = np.diff(bx[a:b]) * FPS
        vy = np.diff(by[a:b]) * FPS
        vel[a + 1:b] = np.hypot(vx, vy)
    edge = np.minimum.reduce([bx, L - bx, by, W - by])
    pspeed = mean_player_speed(g)
    prev_seen = np.r_[False, seen[:-1]]
    same_half = np.r_[False, half[1:] == half[:-1]]
    at_risk = graded & prev_seen & same_half
    start = at_risk & ~seen
    ctx = lambda x: np.r_[np.nan, x[:-1]][at_risk]  # noqa: E731 -- context at t-1
    s["gap_start_hazard"] = round(float(start[at_risk].mean()), 5)
    s["hazard_by_crowd"] = _rate_by(start[at_risk], ctx(crowd), CROWD_BINS)
    s["hazard_by_ball_speed"] = _rate_by(start[at_risk], np.nan_to_num(ctx(vel), nan=0),
                                         BSPEED_BINS)
    s["hazard_by_edge"] = _rate_by(start[at_risk], ctx(edge), EDGE_BINS)
    # per context cell: hazard, support, and the length histogram of gaps starting there
    fx, fy = _ffill(bx, g.halves), _ffill(by, g.halves)      # last seen position while missing
    outside_now = (fx < 0) | (fx > L) | (fy < 0) | (fy > W)
    cell = context_cell(outside_now, pspeed)
    cell_prev = np.r_[0, cell[:-1]]
    gap_cell, gap_len = [], []
    for a, b in g.halves:
        st, en, v = runs(seen[a:b])
        for s0, e0, vis in zip(st, en, v):
            if not vis and s0 > 0:
                gap_cell.append(cell_prev[a + s0])
                gap_len.append(e0 - s0)
    gap_cell, gap_len = np.array(gap_cell, int), np.array(gap_len)
    cells = {"names": list(CELLS), "hazard": [], "share": [], "gap_hist": [], "n_gaps": []}
    for c in range(len(CELLS)):
        m = cell_prev[at_risk] == c
        cells["hazard"].append(round(float(start[at_risk][m].mean()), 5) if m.sum() > 100
                               else None)
        cells["share"].append(round(float(m.mean()), 4))
        gl = gap_len[gap_cell == c]
        cells["n_gaps"].append(int(len(gl)))
        cells["gap_hist"].append(_hist(gl, GAP_BINS) if len(gl) >= 20 else None)
    s["cells"] = cells
    s["missing_by_cell"] = _rate_by(~seen[graded], cell[graded].astype(float),
                                    [-0.5, 0.5, 1.5, 2.5, 3.5])
    s["missing_by_player_speed"] = _rate_by(~seen[graded], np.nan_to_num(pspeed[graded]),
                                            PSPEED_BINS)
    # start / end hazards of the ball gap process by mean player speed at t-1 (added in step 8;
    # absent from older target files -- the calibrator then fits occupancy instead)
    ps_prev = np.nan_to_num(np.r_[np.nan, pspeed[:-1]])
    s["hazard_by_player_speed"] = _rate_by(start[at_risk], ps_prev[at_risk], PSPEED_BINS)
    in_gap = graded & ~prev_seen & same_half
    s["end_hazard_by_player_speed"] = _rate_by(seen[in_gap], ps_prev[in_gap], PSPEED_BINS)
    if g.labels is not None:   # only external games have labels; never computed on the target
        lab = g.labels[graded]
        s["missing_by_label"] = {k: round(float((~seen[graded])[lab == k].mean()), 4)
                                 for k in "HAD" if (lab == k).any()}

    # ---- ball geometry and noise
    sv = seen & graded
    s["ball_z_present"] = round(float(np.isfinite(g.ball[sv, 2]).mean()), 4) if sv.any() else 0
    out = (bx < 0) | (bx > L) | (by < 0) | (by > W)
    s["ball_outside"] = round(float(out[sv].mean()), 4) if sv.any() else 0
    jump = np.r_[np.nan, np.hypot(np.diff(bx), np.diff(by))]
    both = sv & prev_seen & same_half
    s["ball_jump_gt3m"] = round(float((jump[both] > 3).mean()), 5) if both.any() else 0
    d2 = bx[2:] - 2 * bx[1:-1] + bx[:-2]
    s["ball_d2_std"] = _robust_std(d2[graded[1:-1]])

    # ---- players: counts, tracklets, jitter, speed field
    cnt = pseen[graded].sum(1)
    s["objects_mean"] = round(float(cnt.mean()), 3)
    s["objects_hist"] = _hist(cnt, list(range(0, 32, 2)) + [99])
    s["slots_home"] = int((g.team == 0).sum())
    for t, name in ((0, "home"), (1, "away")):
        s["objects_mean_" + name] = round(float(pseen[graded][:, g.team == t].sum(1).mean()), 3)
    lives, used = [], 0
    for k in range(P.shape[1]):
        any_k = False
        for a, b in g.halves:
            st, en, v = runs(pseen[a:b, k])
            lives += list((en - st)[v])
            any_k = any_k or v.any()
        used += any_k
    lives = np.array(lives)
    s["slots_used"] = int(used)
    s["tracklets_per_min"] = round(len(lives) / (graded.sum() / FPS / 60), 3)
    s["tracklet_life_hist"] = _hist(lives, LIFE_BINS)
    s["tracklet_life_q"] = {q: float(np.percentile(lives, q)) for q in (10, 50, 90)} \
        if len(lives) else {}
    px = P[..., 0]
    pd2 = (px[2:] - 2 * px[1:-1] + px[:-2])[graded[1:-1]]
    s["player_d2_std"] = _robust_std(pd2.ravel())
    sp = P[..., 2][graded][pseen[graded]]
    sp = sp[np.isfinite(sp)]
    s["player_speed_q"] = {q: round(float(np.percentile(sp, q)), 3) for q in (10, 50, 90, 99)} \
        if len(sp) else {}
    s["player_speed_nan_share"] = round(float(np.isnan(P[..., 2][graded][pseen[graded]]).mean()), 4)
    return s


def compare(sim, target, keys=None):
    """Rows (key, sim, target, ratio or L1) for every comparable statistic."""
    rows = []
    for k in keys or sorted(target):
        a, b = sim.get(k), target.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            rows.append((k, a, b, round(a / b, 3) if b else None))
        elif isinstance(b, list) and isinstance(a, list) and len(a) == len(b):
            rows.append((k + " (L1)", "", "", round(float(np.abs(np.subtract(a, b)).sum()), 3)))
        elif isinstance(b, dict) and isinstance(a, dict) and "hazard" in b:
            for i, nm in enumerate(b["names"]):
                ha, hb = a["hazard"][i], b["hazard"][i]
                rows.append(("cell {} hazard".format(nm), ha, hb,
                             round(ha / hb, 3) if ha and hb else None))
                ga, gb = a["gap_hist"][i], b["gap_hist"][i]
                if ga and gb:
                    rows.append(("cell {} gap_hist (L1)".format(nm), "", "",
                                 round(float(np.abs(np.subtract(ga, gb)).sum()), 3)))
        elif isinstance(b, dict) and isinstance(a, dict) and "rate" in b:
            ra = [x if x is not None else np.nan for x in a["rate"]]
            rb = [x if x is not None else np.nan for x in b["rate"]]
            rows.append((k, [round(x, 4) for x in ra], [round(x, 4) for x in rb], ""))
    return rows


def print_compare(rows):
    for k, a, b, r in rows:
        print("{:<32} {:>28} {:>28}  {}".format(k, str(a)[:28], str(b)[:28], r))


def target_game():
    """The competition match as a Game, built from the raw ATD. Never printed, never saved."""
    import games
    from mtah.external import atd_io
    from games import GAME, GAMES, half_video_blocks
    paths = games.require(["atd_meta", "atd_raw"])
    meta, raw = paths
    team = atd_io.team_of_slots(meta)
    frame, players, ball = atd_io.read_txt(raw, len(team))
    n = GAMES[GAME]["n_video_frames"]
    L, W = GAMES[GAME]["pitch"]
    P = np.full((n, len(team), 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    idx = frame - 1
    P[idx, :, 0], P[idx, :, 1], P[idx, :, 2] = players[..., 0] * L, players[..., 1] * W, \
        players[..., 2]
    B[idx, 0], B[idx, 1], B[idx, 2] = ball[:, 0] * L, ball[:, 1] * W, ball[:, 2]
    halves = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]
    return Game(name=GAME, players=P, team=team, ball=B, halves=halves, pitch=(L, W),
                home_attacks_right_first_half=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--game")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = (json.load(open(p)) for p in args.compare)
        print_compare(compare(a, b))
        return
    if args.target:
        sys.path.insert(0, ROOT)
        g = target_game()
        out = args.out or str(config.EXTERNAL.TARGET_STATS)
    else:
        g = Game.load(args.game)
        out = args.out or os.path.join(args.game, "stats.json")
    s = aggregate_stats(g)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(s, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
