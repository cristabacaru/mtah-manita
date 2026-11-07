"""Idea 8 -- player speed: is the ATD speed column m/s, and does speed from positions match it?

Behind the guide section 5 decision to compute players_speed from positions for every game
(external sources disagree: DFL fills missing speed with 0, Metrica CSV has none). Prints
aggregates only.
"""
import warnings

import numpy as np

import common as c

warnings.simplefilter("ignore")
d = c.load_atd()
x, y, s = d["x"], d["y"], d["speed"]          # metres (raw x pitch from games.py); speed as delivered
g = c.graded_mask(d["n"])
K = round(0.2 * c.FPS)                         # centred difference over +/-0.2 s, as data.py does


def position_speed(k):
    """|position(t+k) - position(t-k)| / (2k frames), m/s; NaN where either end is missing."""
    dx = np.full_like(x, np.nan)
    dy = np.full_like(y, np.nan)
    dx[k:-k] = x[2 * k:] - x[:-2 * k]
    dy[k:-k] = y[2 * k:] - y[:-2 * k]
    return np.hypot(dx, dy) * c.FPS / (2 * k)


tracked = np.isfinite(x)
print("tracked cells (position present) with no speed value: {:.1%}".format(np.mean(~np.isfinite(s[tracked]))))
v = s[g][np.isfinite(s[g])]
print("speed column, graded: p1/p50/p90/p99/max = {} m/s".format(np.round(np.percentile(v, [1, 50, 90, 99, 100]), 2)))

for k in (1, K):
    p = position_speed(k)
    m = np.isfinite(p) & np.isfinite(s) & g[:, None]
    a, b = p[m], s[m]
    print("\nposition speed +/-{} frames vs column ({} cells with both):".format(k, a.size))
    print("  position p50/p90/p99 = {} m/s".format(np.round(np.percentile(a, [50, 90, 99]), 2)))
    print("  corr {:.3f}   median ratio column/position (position > 0.5 m/s) {:.3f}".format(
        np.corrcoef(a, b)[0, 1], np.nanmedian(np.where(a > 0.5, b / a, np.nan))))
    print("  |column - position|: median {:.3f}  p90 {:.3f} m/s".format(
        np.median(np.abs(b - a)), np.percentile(np.abs(b - a), 90)))

p = position_speed(K)
mc, mp = np.nanmean(s, axis=1), np.nanmean(p, axis=1)
m = g & np.isfinite(mc) & np.isfinite(mp)
print("\nper-frame mean over players (+/-{} frames): corr {:.3f}, median ratio {:.3f}, median |diff| {:.3f} m/s".format(
    K, np.corrcoef(mc[m], mp[m])[0, 1], np.median(mc[m] / mp[m]), np.median(np.abs(mc[m] - mp[m]))))
print("graded frames with mean speed (+/-0.5 s) < 1.5 m/s: column {:.1%}  position {:.1%}".format(
    np.mean(c.rolling_mean(mc, c.FPS // 2)[g] < 1.5), np.mean(c.rolling_mean(mp, c.FPS // 2)[g] < 1.5)))
print("tracked graded cells with a speed: column {:.1%}  position {:.1%}".format(
    np.mean(np.isfinite(s[g][tracked[g]])), np.mean(np.isfinite(p[g][tracked[g]]))))
