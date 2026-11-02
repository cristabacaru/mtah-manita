"""Idea 1 -- structure of missing-ball gaps (feasibility). Aggregates only."""
import numpy as np

import common as c

d = c.load_atd()
bx, by = d["bx"], d["by"]
vis = np.isfinite(bx)
starts, ends, lengths = [], [], []
for half in c.half_masks(d["n"]):
    s, e = c.runs(half & ~vis)
    starts += list(s); ends += list(e); lengths += list(e - s)
starts, ends, lengths = map(np.array, (starts, ends, lengths))
graded = c.graded_mask(d["n"])
n_missing = int((graded & ~vis).sum())
print("graded frames {}  missing-ball {} ({:.1%})  gaps {}".format(
    graded.sum(), n_missing, n_missing / graded.sum(), len(lengths)))

print("\nthreshold   % of gaps shorter   % of missing frames in gaps shorter")
for sec in (0.2, 0.5, 1, 3, 10):
    k = lengths < sec * c.FPS
    print("  < {:>4} s        {:5.1%}              {:5.1%}".format(sec, k.mean(), lengths[k].sum() / lengths.sum()))
print("gap length p50 / p90 / p99 / max (s): {:.2f} / {:.2f} / {:.1f} / {:.0f}".format(
    *(np.percentile(lengths, [50, 90, 99, 100]) / c.FPS)))


def near_line(x, y, m=2.0):
    return (x < m) | (x > c.L - m) | (y < m) | (y > c.W - m)


def outside(x, y):
    return (x < 0) | (x > c.L) | (y < 0) | (y > c.W)


# ball position on the last visible frame before the gap / first visible frame after
ok = (starts > 0) & (ends < d["n"])
pre, post = starts[ok] - 1, ends[ok]
pre_ok, post_ok = vis[pre], vis[post]
base = graded & vis
print("\nbase rate: visible graded ball within 2 m of a line {:.1%}, outside the lines {:.1%}".format(
    near_line(bx[base], by[base]).mean(), outside(bx[base], by[base]).mean()))
print("\n{:<12} {:>6} {:>18} {:>18} {:>16}".format("gap length", "n", "start near line", "end near line", "start outside"))
for lo, hi in ((0, 1), (1, 3), (3, 10), (10, 1e9)):
    k = (lengths[ok] >= lo * c.FPS) & (lengths[ok] < hi * c.FPS) & pre_ok & post_ok
    a, b = pre[k], post[k]
    print("{:<12} {:>6} {:>18.1%} {:>18.1%} {:>16.1%}".format(
        "{}-{} s".format(lo, hi if hi < 1e9 else "inf"), k.sum(),
        near_line(bx[a], by[a]).mean(), near_line(bx[b], by[b]).mean(), outside(bx[a], by[a]).mean()))

# how far the ball "jumps" across short gaps (is linear interpolation sane?)
k = (lengths[ok] < c.FPS) & pre_ok & post_ok
jump = np.hypot(bx[post[k]] - bx[pre[k]], by[post[k]] - by[pre[k]])
speed = jump / (lengths[ok][k] / c.FPS + 1 / c.FPS)
print("\ngaps < 1 s: displacement across gap p50/p90 {:.1f} / {:.1f} m; implied speed p50/p90 {:.0f} / {:.0f} m/s; "
      "share implying > 40 m/s {:.1%}".format(*np.percentile(jump, [50, 90]), *np.percentile(speed, [50, 90]), (speed > 40).mean()))
