"""Idea 4 -- Smart Tagging as a prior (agreement between independent automatic signals)."""
import collections
import os
import sys
import warnings

import numpy as np

import common as c

sys.path.insert(0, os.path.join(c.ROOT, "starter"))
from baseline import smooth  # noqa: E402

warnings.simplefilter("ignore")
d = c.load_atd()
n = d["n"]
g = c.graded_mask(n)
ev = c.load_tagging()

print("== counts by code x Team ==")
tab = collections.Counter((e["code"].strip().upper(), e["tags"].get("Team", "-")) for e in ev)
codes = sorted({k[0] for k in tab})
teams = sorted({k[1] for k in tab})
print("{:<30}".format("code") + "".join("{:>14}".format(t) for t in teams) + "{:>8}".format("total"))
for code in codes:
    print("{:<30}".format(code) + "".join("{:>14}".format(tab.get((code, t), 0)) for t in teams)
          + "{:>8}".format(sum(tab.get((code, t), 0) for t in teams)))
types = collections.Counter(e["tags"].get("Type") for e in ev if e["code"].strip().upper() == "SET PIECES")
print("SET PIECES by Type:", dict(types))
dur = collections.defaultdict(list)
for e in ev:
    dur[e["code"].strip().upper()].append(e["end_s"] - e["start_s"])
print("median duration (s) by code:", {k: round(float(np.median(v)), 1) for k, v in sorted(dur.items())})

# ---- does Team mean the team in possession? agreement with nearest-player majority -------
near = c.nearest_team(d)
home_name = "Barcelona"
print("\n== Team tag vs nearest-player majority inside the interval (ball-visible frames) ==")
print("{:<30} {:>5} {:>18}".format("code", "n", "tag == nearest"))
agree_by_code = {}
for code in codes:
    hits = tot = 0
    for e in ev:
        if e["code"].strip().upper() != code or e["tags"].get("Team") in (None, "N/A"):
            continue
        seg = near[e["start_frame"]:e["end_frame"]]
        seg = seg[seg >= 0]
        if len(seg) < 10:
            continue
        maj = c.HOME if np.mean(seg == c.HOME) >= 0.5 else c.AWAY
        tag = c.HOME if e["tags"]["Team"] == home_name else c.AWAY
        hits += maj == tag; tot += 1
    if tot:
        agree_by_code[code] = hits / tot
        print("{:<30} {:>5} {:>18.0%}".format(code, tot, hits / tot))

# ---- per-frame "phase team": possession team implied by the most recent team-tagged phase --
# Codes whose Team is the defending side (by the agreement above, < 50%) are flipped.
flip = {k for k, v in agree_by_code.items() if v < 0.5}
print("codes treated as 'Team = defending side':", sorted(flip))
phase = np.full(n, -1, dtype=np.int8)
for e in sorted(ev, key=lambda e: e["start_frame"]):
    t = e["tags"].get("Team")
    if t in (None, "N/A"):
        continue
    team = c.HOME if t == home_name else c.AWAY
    if e["code"].strip().upper() in flip:
        team = 1 - team
    phase[e["start_frame"]:e["end_frame"]] = team
cov = g & (phase >= 0)
print("graded frames covered by a team-tagged phase: {:.1%}".format(cov.sum() / g.sum()))
both = cov & (near >= 0)
print("frame agreement phase-team vs raw nearest-player (both defined): {:.1%}".format(
    np.mean(phase[both] == near[both])))


def changes(lab, valid):
    """Frames where the (H/A) label changes between consecutive defined frames, graded only."""
    idx = np.flatnonzero(valid)
    v = lab[idx]
    k = np.flatnonzero(v[1:] != v[:-1]) + 1
    return idx[k]


sm = smooth(near, 50)
near_lab = np.where(sm == "H", 0, np.where(sm == "A", 1, -1))
pc = changes(phase, cov)
nc = changes(near_lab, g & (near_lab >= 0))
pc = pc[g[pc]]


def hit_rate(a, b, tol):
    b = np.sort(b)
    j = np.searchsorted(b, a)
    lo = np.abs(a - b[np.clip(j - 1, 0, len(b) - 1)])
    hi = np.abs(b[np.clip(j, 0, len(b) - 1)] - a)
    return np.mean(np.minimum(lo, hi) <= tol)


rng = np.random.default_rng(0)
gi = np.flatnonzero(g)
chance = np.mean([hit_rate(rng.choice(gi, len(pc)), nc, c.FPS) for _ in range(20)])
print("\nphase-team changes: {}   nearest-player (smooth50) team changes: {}".format(len(pc), len(nc)))
print("phase-team changes with a nearest-player change within +/-1 s: {:.0%} (chance {:.0%})".format(
    hit_rate(pc, nc, c.FPS), chance))
print("  within +/-3 s: {:.0%} (chance {:.0%})".format(
    hit_rate(pc, nc, 3 * c.FPS), np.mean([hit_rate(rng.choice(gi, len(pc)), nc, 3 * c.FPS) for _ in range(20)])))

# ---- set-piece starts vs the end of slow phases -------------------------------------------
with np.errstate(all="ignore"):
    speed = c.rolling_mean(np.nanmean(d["speed"], axis=1), c.FPS // 2)
slow = g & (speed < 1.5)
s0, e0 = c.runs(slow)
long_slow_end = e0[(e0 - s0) >= 2 * c.FPS]
sp = [e for e in ev if e["code"].strip().upper() == "SET PIECES"]
print("\n== SET PIECES starts vs end of slow phases (>= 2 s under 1.5 m/s) ==")
print("{:<24} {:>4} {:>20} {:>12} {:>16}".format("type", "n", "start inside slow", "end <=3 s", "median |dt| s"))
for typ in sorted({e["tags"].get("Type") for e in sp}):
    st = np.array([e["start_frame"] for e in sp if e["tags"].get("Type") == typ])
    j = np.searchsorted(long_slow_end, st)
    dt = np.minimum(np.abs(st - long_slow_end[np.clip(j - 1, 0, len(long_slow_end) - 1)]),
                    np.abs(long_slow_end[np.clip(j, 0, len(long_slow_end) - 1)] - st)) / c.FPS
    print("{:<24} {:>4} {:>20.0%} {:>12.0%} {:>16.1f}".format(typ, len(st), np.mean(slow[st]), np.mean(dt <= 3),
                                                            np.median(dt)))
rand = rng.choice(gi, 2000)
j = np.searchsorted(long_slow_end, rand)
dt = np.minimum(np.abs(rand - long_slow_end[np.clip(j - 1, 0, len(long_slow_end) - 1)]),
                np.abs(long_slow_end[np.clip(j, 0, len(long_slow_end) - 1)] - rand)) / c.FPS
print("{:<24} {:>4} {:>20.0%} {:>12.0%} {:>16.1f}".format("random graded frame", len(rand), np.mean(slow[rand]),
                                                        np.mean(dt <= 3), np.median(dt)))
vis = np.isfinite(d["bx"])
print("\nsigned: seconds from set-piece start to the NEXT slow-phase end (restart = players speed up)")
for typ in sorted({e["tags"].get("Type") for e in sp}):
    st = np.array([e["start_frame"] for e in sp if e["tags"].get("Type") == typ])
    nxt = long_slow_end[np.clip(np.searchsorted(long_slow_end, st), 0, len(long_slow_end) - 1)]
    dt = (nxt - st) / c.FPS
    print("  {:<24} median {:5.1f} s   share within 0-10 s {:4.0%}".format(typ, np.median(dt), np.mean((dt >= 0) & (dt <= 10))))
nxt = long_slow_end[np.clip(np.searchsorted(long_slow_end, rand), 0, len(long_slow_end) - 1)]
dt = (nxt - rand) / c.FPS
print("  {:<24} median {:5.1f} s   share within 0-10 s {:4.0%}".format("random graded frame", np.median(dt), np.mean((dt >= 0) & (dt <= 10))))
print("set-piece starts on a ball-missing frame: {:.0%} (graded base rate {:.0%})".format(
    np.mean([not vis[e["start_frame"]] for e in sp if g[e["start_frame"]]]), np.mean(~vis[g])))
print("\nGOALS instances: {}   Kick Off (After Goal): {}   Kick Off (Start): {}".format(
    sum(e["code"].strip().upper() == "GOALS" for e in ev), types.get("Kick Off (After Goal)", 0),
    types.get("Kick Off (Start)", 0)))
by_half = collections.Counter((e["tags"].get("Half"), e["tags"].get("Type") or e["code"]) for e in ev
                              if e["code"].strip().upper() == "GOALS" or "Kick Off" in (e["tags"].get("Type") or ""))
print("by half:", dict(sorted(by_half.items())))
