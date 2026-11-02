"""Idea 2 -- nearest-player baseline with X instead of D on ball-missing frames (feasibility).

Reports possession counts and median length per half only, never a score. The scorer
forward-fills X within each half, so the filled sequence is what produces possessions; that is
what is counted here, using the same smoothing as starter/baseline.py.
"""
import os
import sys

import numpy as np

import common as c

sys.path.insert(0, os.path.join(c.ROOT, "starter"))
from baseline import smooth  # noqa: E402

d = c.load_atd()
x, y, bx, by = d["x"], d["y"], d["bx"], d["by"]
vis = np.isfinite(bx)
dist = np.hypot(x - bx[:, None], y - by[:, None])
dist[~np.isfinite(dist)] = np.inf
codes = np.full(d["n"], -1, dtype=np.int8)
codes[vis] = d["team"][np.argmin(dist[vis], axis=1)]


def fill(labels, half):
    out = labels.copy()
    idx = np.flatnonzero(half)
    seg = out[idx]
    known = seg != "X"
    pos = np.where(known, np.arange(len(seg)), -1)
    pos = np.maximum.accumulate(pos)
    first = np.flatnonzero(known)[0]
    pos[pos < 0] = first
    out[idx] = seg[pos]
    return out


def possessions(labels, half):
    lab = labels[half]
    poss = np.isin(lab, ["H", "A"])
    change = np.concatenate(([True], lab[1:] != lab[:-1]))
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(lab))
    keep = poss[starts]
    return (ends - starts)[keep]


halves = c.half_masks(d["n"])
print("{:<26} {:>8} {:>8} {:>10} {:>10}  {}".format("variant", "poss H1", "poss H2", "median s", "mean s", "D share"))
for w in (25, 50, 125):
    base = smooth(codes, w)
    xvar = np.where(vis, base, "X")       # X where the ball itself is missing on that frame
    for name, lab in (("baseline smooth{}".format(w), base), ("X-variant smooth{}".format(w), xvar)):
        lens, nper = [], []
        filled = lab.copy()
        for h in halves:
            filled = fill(filled, h)
            p = possessions(filled, h)
            nper.append(len(p)); lens += list(p)
        g = c.graded_mask(d["n"])
        print("{:<26} {:>8} {:>8} {:>10.1f} {:>10.1f}  {:.1%}".format(
            name, nper[0], nper[1], np.median(lens) / c.FPS, np.mean(lens) / c.FPS, np.mean(filled[g] == "D")))
print("\nspell counts are a plausibility check only, not a score.")
