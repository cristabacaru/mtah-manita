"""Fit the simulator's free parameters to the target's aggregate statistics.

    python -m mtah.external.calibrate                  # fit on the 9 suite games, write params
    python -m mtah.external.calibrate --report-only    # just the sim-vs-target table

Reads $MTAH_CACHE/p2/target_stats.json (aggregates only), writes $MTAH_CACHE/p2/sim_params.json,
then
degrades every game at the `target` level and prints the mean of their stats next to the target.

What is fitted, pooled over all games (one parameter set for everything):
  * gap start hazard: log-linear in context cell, ball speed, crowd, edge (and player speed when
    the target has hazard_by_player_speed), by iterative proportional fitting to the target's
    hazard tables;
  * gap end rate per player-speed bin: to the target's end_hazard_by_player_speed when present,
    otherwise to its missing_by_player_speed occupancy;
  * tracklet lifetime scale and detection-gap share: to tracklets_per_min and objects_mean.
Labels are never read.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from mtah.schema import Game, runs  # noqa: E402
from mtah import simulate as SIM  # noqa: E402
from mtah.external import stats as S  # noqa: E402
from mtah.external.build_suite import CACHE, DEFAULT_GAMES  # noqa: E402

KEY_STATS = ["ball_missing", "gaps_per_min", "gap_start_hazard", "ball_outside", "ball_d2_std",
             "ball_jump_gt3m", "ball_z_present", "objects_mean", "tracklets_per_min",
             "slots_used", "player_d2_std", "player_speed_nan_share"]
KEY_TABLES = ["missing_by_player_speed", "missing_by_cell", "hazard_by_ball_speed",
              "hazard_by_crowd", "hazard_by_edge", "hazard_by_player_speed",
              "end_hazard_by_player_speed"]


def _pool(measures, key):
    num = np.sum([np.asarray(m[key][0]) for m in measures], axis=0)
    den = np.sum([np.asarray(m[key][1]) for m in measures], axis=0)
    with np.errstate(all="ignore"):
        return np.where(den > 200, num / den, np.nan)


def _step(cur, tgt, damp):
    ok = np.isfinite(cur) & np.isfinite(tgt) & (cur > 0) & (tgt > 0)
    return np.where(ok, damp * np.log(np.where(ok, tgt, 1) / np.where(ok, cur, 1)), 0.0)


def fit_gaps(prep, target, p, iters=15, damp=0.8, log=print):
    t = {"cell": SIM._rates(target.get("cells"), 4) if target.get("cells") else None,
         "bs": SIM._rates(target.get("hazard_by_ball_speed"), 4),
         "crowd": SIM._rates(target.get("hazard_by_crowd"), 4),
         "edge": SIM._rates(target.get("hazard_by_edge"), 4),
         "ps": SIM._rates(target.get("hazard_by_player_speed"), SIM.N_PS)
         if target.get("hazard_by_player_speed") else None}
    t_end = (SIM._rates(target["end_hazard_by_player_speed"], SIM.N_PS)
             if target.get("end_hazard_by_player_speed") else None)
    t_occ = SIM._rates(target.get("missing_by_player_speed"), SIM.N_PS)
    factors = [("A", "cell"), ("B", "bs"), ("C", "crowd"), ("E", "edge"), ("D", "ps")]

    def simulate():
        out = []
        for g, ctx, u in prep:
            h, r = SIM.hazards(ctx, p)
            out.append(SIM.gap_measure(SIM.run_gaps(g.halves, h, r, u, p), ctx, g.halves, g.n))
        return out

    for it in range(iters):
        for name, key in factors:
            if t[key] is None:
                continue
            cur = _pool(simulate(), "h_" + key)
            p[name] = (np.asarray(p[name]) + _step(cur, t[key], damp)).tolist()
        # global rescale to the overall gap-start hazard: the context mix of the external games
        # cannot match the target's exactly, so the margins alone over-count gaps
        t_all = target.get("gap_start_hazard")
        if t_all:
            ms = simulate()
            cur = sum(m["start"][0] for m in ms) / sum(m["start"][1] for m in ms)
            p["A"] = (np.asarray(p["A"]) + damp * np.log(t_all / cur)).tolist()
        ms = simulate()
        if t_end is not None:
            p["R"] = (np.asarray(p["R"]) + _step(_pool(ms, "end_ps"), t_end, damp)).tolist()
        else:
            occ = _pool(ms, "occ_ps")
            odds = lambda x: np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4))  # noqa
            ok = np.isfinite(occ) & np.isfinite(t_occ)
            p["R"] = (np.asarray(p["R"]) + np.where(
                ok, damp * (np.log(odds(np.where(ok, occ, .5))) -
                            np.log(odds(np.where(ok, t_occ, .5)))), 0)).tolist()
        ms = simulate()
        miss = sum(m["miss"][0] for m in ms) / sum(m["miss"][1] for m in ms)
        start = sum(m["start"][0] for m in ms) / sum(m["start"][1] for m in ms)
        gl = np.concatenate([m["gap_len"] for m in ms])
        log("  gaps it {:>2}: missing {:.3f} (tgt {:.3f})  start hazard {:.5f} (tgt {:.5f})  "
            "gap median {:.0f} (tgt {})  occ {}".format(
                it, miss, target["ball_missing"], start, target.get("gap_start_hazard", np.nan),
                np.median(gl), target.get("gap_q", {}).get("50"),
                np.round(_pool(ms, "occ_ps"), 3).tolist()))
    return p


def _player_measure(players, g):
    graded = g.graded()
    seen = np.isfinite(players[..., 0])
    n_tr = 0
    for k in range(players.shape[1]):
        for a, b in g.halves:
            _, _, v = runs(seen[a:b, k])
            n_tr += int(v.sum())
    return n_tr / (graded.sum() / S.FPS / 60), float(seen[graded].sum(1).mean())


def fit_players(prep_players, target, p, iters=5, log=print):
    for it in range(iters):
        tpm, obj = [], []
        for g, xy in prep_players:
            rng = SIM._rng(g.name, "calib-players", it)
            pl, _, _ = SIM.fragment_players(g, xy, target, p, rng)
            a, b = _player_measure(pl, g)
            tpm.append(a)
            obj.append(b)
        tpm, obj = float(np.mean(tpm)), float(np.mean(obj))
        log("  players it {}: tracklets/min {:.2f} (tgt {})  objects {:.2f} (tgt {})  "
            "life_scale {:.3f} drop {:.3f}".format(it, tpm, target["tracklets_per_min"], obj,
                                                   target["objects_mean"], p["life_scale"],
                                                   p["drop_share"]))
        p["life_scale"] = float(p["life_scale"] * tpm / target["tracklets_per_min"])
        keep = (1 - p["drop_share"]) * target["objects_mean"] / obj
        p["drop_share"] = float(np.clip(1 - keep, 0.0, 0.5))
    return p


def mean_stats(stats_list):
    """Mean of numeric stats and of rate tables over games."""
    out = {}
    for k in stats_list[0]:
        vals = [s.get(k) for s in stats_list]
        if all(isinstance(v, (int, float)) for v in vals):
            out[k] = round(float(np.mean(vals)), 5)
        elif all(isinstance(v, dict) and "rate" in v for v in vals):
            r = np.array([[x if x is not None else np.nan for x in v["rate"]] for v in vals])
            with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                __import__("warnings").simplefilter("ignore", RuntimeWarning)
                out[k] = {"rate": [None if not np.isfinite(x) else round(float(x), 4)
                                   for x in np.nanmean(r, 0)]}
        elif all(isinstance(v, list) for v in vals) and len({len(v) for v in vals}) == 1:
            out[k] = np.round(np.mean(np.array(vals, float), 0), 4).tolist()
    return out


def report(sim, target, log=print):
    log("\n{:<30} {:>12} {:>12} {:>8}".format("statistic", "sim (mean)", "target", "ratio"))
    for k in KEY_STATS:
        a, b = sim.get(k), target.get(k)
        if a is None or b is None:
            continue
        log("{:<30} {:>12.5g} {:>12.5g} {:>8.3f}".format(k, a, b, a / b if b else float("nan")))
    for k in ("gap_hist", "tracklet_life_hist", "objects_hist"):
        if k in sim and k in target:
            log("{:<30} L1 {:.3f}".format(k, float(np.abs(np.subtract(sim[k], target[k])).sum())))
    for k in KEY_TABLES:
        if k in sim and k in target:
            log("{:<30} sim {}".format(k, sim[k]["rate"]))
            log("{:<30} tgt {}".format("", target[k]["rate"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    target = SIM.load_target()
    games = [Game.load(os.path.join(CACHE, nm)) for nm in args.games]

    if not args.report_only:
        p = SIM.initial_params(target)
        print("fitting gap process on", len(games), "games")
        prep = []
        for g in games:
            ball = SIM.fill_ball(g)
            ctx = SIM.gap_context(g, ball)
            prep.append((g, ctx, SIM.gap_draws(SIM._rng(g.name, "calib-gaps", 0), g.n, p)))
        p = fit_gaps(prep, target, p, args.iters)
        print("fitting tracklets")
        prep_pl = []
        for g in games:
            rng = SIM._rng(g.name, "calib-noise", 0)
            xy, _ = SIM.match_d2([g.players[..., 0].astype(np.float64),
                                  g.players[..., 1].astype(np.float64)], g.halves, g.graded(),
                                 target.get("player_d2_std", 0.0016), rng)
            prep_pl.append((g, np.stack(xy, axis=2)))
        p = fit_players(prep_pl, target, p)
        p.update({"fitted": True, "target_hash": SIM.target_hash(target),
                  "fitted_on": list(args.games),
                  "provisional_target": bool(target.get("provisional", False))})
        os.makedirs(os.path.dirname(SIM.PARAMS_PATH), exist_ok=True)
        with open(SIM.PARAMS_PATH, "w") as fh:
            json.dump(p, fh, indent=1)
        print("wrote", SIM.PARAMS_PATH)

    params = SIM.load_params(target)
    sts = []
    for g in games:
        d = SIM.degrade(g, "target", 0, target, params)
        sts.append(S.aggregate_stats(d))
    report(mean_stats(sts), target)


if __name__ == "__main__":
    main()
