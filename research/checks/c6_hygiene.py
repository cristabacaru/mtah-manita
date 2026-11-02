"""Idea 6 -- data hygiene (feasibility). Aggregates only."""
import collections
import warnings

import numpy as np

import common as c

warnings.simplefilter("ignore")
d = c.load_atd()
n, team, ptype = d["n"], d["team"], d["ptype"]
x, y = d["x"], d["y"]
g = c.graded_mask(n)
tracked = np.isfinite(x)

print("slots: home {}  away {}  position_type {}".format(
    int((team == 0).sum()), int((team == 1).sum()), dict(collections.Counter(ptype))))
gk = [i for i, p in enumerate(ptype) if p == "Goalkeeper"]
print("goalkeeper-flagged slots belong to team(s): {}".format(sorted({int(team[i]) for i in gk})))
print("goalkeeper-flagged slots present on {:.0%} / {:.0%} of graded frames".format(
    *[tracked[g, i].mean() for i in gk]))
print("no referee slot / team in metadata: teams = 2, every slot carries a team id")

cnt = tracked.sum(1)
home_n = tracked[:, team == 0].sum(1)
away_n = tracked[:, team == 1].sum(1)
print("\nobjects per graded frame: mean {:.1f}  p5/p50/p95 {:.0f}/{:.0f}/{:.0f}".format(
    cnt[g].mean(), *np.percentile(cnt[g], [5, 50, 95])))
print("frames with > 22 objects {:.2%}   with > 11 on one team {:.2%}   (home {:.2%}, away {:.2%})".format(
    np.mean(cnt[g] > 22), np.mean((home_n[g] > 11) | (away_n[g] > 11)), np.mean(home_n[g] > 11),
    np.mean(away_n[g] > 11)))
print("mean objects per team on graded frames: home {:.1f}  away {:.1f}".format(home_n[g].mean(), away_n[g].mean()))

# duplicates: two same-team objects closer than 0.5 m
close = 0
sample = np.flatnonzero(g)[::25]
for f in sample:
    for t in (0, 1):
        k = tracked[f] & (team == t)
        px, py = x[f, k], y[f, k]
        if len(px) > 1:
            dd = np.hypot(px[:, None] - px[None], py[:, None] - py[None])
            close += np.any(dd[np.triu_indices(len(px), 1)] < 0.5)
print("frames (1/s sample) with two same-team objects < 0.5 m apart: {:.1%}".format(close / len(sample)))

# ends: team centroid x by half
print("\nteam centroid x by half (m), metadata says Barcelona attacks left_to_right in half 1:")
for h, m in enumerate(c.half_masks(n), 1):
    hx = np.nanmean(np.where(team[None] == 0, x, np.nan)[m], axis=1)
    ax = np.nanmean(np.where(team[None] == 1, x, np.nan)[m], axis=1)
    print("  half {}: home median {:.1f}  away median {:.1f}  home left of away on {:.0%} of frames".format(
        h, np.nanmedian(hx), np.nanmedian(ax), np.nanmean(hx < ax)))
    for i in gk:
        print("    GK slot (team {}) median x {:.1f}".format(team[i], np.nanmedian(x[m, i])))

# identity swaps / jumps
dx = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
gg = g[1:] & g[:-1]
both = np.isfinite(dx) & gg[:, None]
jumps = (dx > 5) & both
print("\nslot-frame steps (both frames tracked): {:,}".format(int(both.sum())))
print("steps > 5 m in one frame: {:,} ({:.3%}), i.e. {:.1f} per minute of play".format(
    int(jumps.sum()), jumps.sum() / both.sum(), jumps.sum() / (g.sum() / c.FPS / 60)))
print("steps > 1 m in one frame (25 m/s): {:.3%}".format(np.mean(dx[both] > 1)))
births = (~tracked[:-1] & tracked[1:]) & gg[:, None]
print("slot births (NaN -> tracked): {:.1f} per minute; median slot tracklet length {:.1f} s".format(
    births.sum() / (g.sum() / c.FPS / 60),
    np.median(np.concatenate([np.diff(np.stack(c.runs(tracked[:, s] & g)), axis=0).ravel()
                              for s in range(tracked.shape[1])])) / c.FPS))
sp = d["speed"]
step_speed = np.full_like(sp, np.nan)
step_speed[1:] = dx * c.FPS
ok = np.isfinite(sp) & np.isfinite(step_speed) & (step_speed < 15)
print("reported speed vs 1-frame displacement speed: corr {:.2f}, median |diff| {:.2f} m/s".format(
    np.corrcoef(sp[ok], step_speed[ok])[0, 1], np.median(np.abs(sp[ok] - step_speed[ok]))))

# referees / assistant referees hiding in team slots? Objects standing outside the touchlines.
outside = tracked & ((y < -1) | (y > c.W + 1) | (x < -1) | (x > c.L + 1))
for t, name in ((0, "home"), (1, "away")):
    k = outside[:, team == t][g]
    print("graded frames with a {} object > 1 m outside the lines: {:.1%}".format(name, np.mean(k.any(1))))
over = g & ((home_n > 11) | (away_n > 11))
print("of frames with > 11 on a team, share that also have an object > 1 m outside: {:.0%}".format(
    np.mean(outside[over].any(1))))
