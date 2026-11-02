"""Idea 3 -- dead-ball signals that need no ball (agreement between independent automatic signals)."""
import numpy as np

import common as c

d = c.load_atd()
n = d["n"]
g = c.graded_mask(n)
with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
    __import__("warnings").simplefilter("ignore")
    speed = np.nanmean(d["speed"], axis=1)      # mean over tracked objects, m/s
sm = c.rolling_mean(speed, c.FPS // 2)          # +/- 0.5 s

# ---- distribution / bimodality ----------------------------------------------------------
v = sm[g & np.isfinite(sm)]
m, s = v.mean(), v.std()
skew = np.mean(((v - m) / s) ** 3)
kurt = np.mean(((v - m) / s) ** 4) - 3
N = len(v)
bc = (skew ** 2 + 1) / (kurt + 3 * (N - 1) ** 2 / ((N - 2) * (N - 3)))
hist, edges = np.histogram(v, bins=np.arange(0, 6.01, 0.1))
h = np.convolve(hist, np.ones(3) / 3, mode="same")
peaks = [round(edges[i] + 0.05, 2) for i in range(1, len(h) - 1) if h[i] > h[i - 1] and h[i] >= h[i + 1]
         and h[i] > 0.01 * h.max()]
print("mean player speed (+/-0.5 s) over graded frames: p10/p50/p90 {:.2f}/{:.2f}/{:.2f} m/s".format(
    *np.percentile(v, [10, 50, 90])))
print("bimodality coefficient {:.3f} (>0.555 suggests bimodal); histogram modes (0.1 m/s bins): {}".format(bc, peaks))
print("coarse histogram, share of graded frames per 0.5 m/s bin:")
hb, eb = np.histogram(v, bins=np.arange(0, 5.01, 0.5))
print("  " + "  ".join("{:.1f}-{:.1f}:{:4.1%}".format(eb[i], eb[i + 1], hb[i] / N) for i in range(len(hb))))
for thr in (1.0, 1.5, 2.0):
    print("share of graded frames with mean speed < {} m/s: {:.1%}".format(thr, np.mean(v < thr)))

# ---- signals ----------------------------------------------------------------------------
slow = g & (sm < 1.5)
step = c.camera_step()
still = g & (c.rolling_mean(step, c.FPS // 2) < 0.02)
vis = np.isfinite(d["bx"])
missing = g & ~vis
bx, by = d["bx"], d["by"]
out_now = g & vis & ((bx < 0) | (bx > c.L) | (by < 0) | (by > c.W))
setp = g & c.tagging_masks(n, lambda code: code == "SET PIECES")
signals = {"slow<1.5": slow, "cam still": still, "ball missing": missing, "ball outside lines": out_now,
           "ST set piece": setp}
print("\nshare of graded frames:")
for k, a in signals.items():
    print("  {:<20} {:5.1%}".format(k, a.sum() / g.sum()))

names = list(signals)
print("\nJaccard (row vs column), and P(column | row) in brackets")
print("{:<20}".format("") + "".join("{:>22}".format(k) for k in names))
for a in names:
    row = "{:<20}".format(a)
    for b in names:
        A, B = signals[a], signals[b]
        row += "{:>22}".format("{:.2f} ({:.0%})".format(c.jaccard(A, B), (A & B).sum() / max(A.sum(), 1)))
    print(row)

# simple vote: a frame is "dead" when >= 2 of slow / still / missing agree
vote = (slow.astype(int) + still.astype(int) + missing.astype(int)) >= 2
print("\n>=2 of {slow, cam still, ball missing}: %.1f%% of graded frames" % (100 * vote.sum() / g.sum()))
print("ball missing AND slow: %.1f%%   ball missing AND NOT slow: %.1f%%" % (
    100 * (missing & slow).sum() / g.sum(), 100 * (missing & ~slow).sum() / g.sum()))

# speed inside long gaps vs outside
gap_long = np.zeros(n, bool)
for half in c.half_masks(n):
    s0, e0 = c.runs(half & ~vis)
    for a, b in zip(s0, e0):
        if b - a >= 3 * c.FPS:
            gap_long[a:b] = True
print("median mean-speed: inside ball gaps >=3 s {:.2f} m/s | ball visible {:.2f} m/s".format(
    np.nanmedian(sm[gap_long & g]), np.nanmedian(sm[g & vis])))
# slow-onset lag: within gaps, how many seconds until speed drops below 1.5
s0, e0 = c.runs(gap_long & g)
first_slow = []
for a, b in zip(s0, e0):
    k = np.flatnonzero(sm[a:b] < 1.5)
    first_slow.append(k[0] / c.FPS if len(k) else np.nan)
fs = np.array(first_slow)
print("gaps >=3 s that ever go slow: {:.0%}; median time from gap start to slow {:.1f} s".format(
    np.mean(np.isfinite(fs)), np.nanmedian(fs)))
