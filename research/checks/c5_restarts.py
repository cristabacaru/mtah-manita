"""Idea 5 -- restart templates from the Laws of the Game (feasibility + agreement with Smart Tagging).

Parameters are fixed from the Laws and tracking noise, not fitted. Prints counts only.
"""
import collections
import warnings

import numpy as np

import common as c

warnings.simplefilter("ignore")
d = c.load_atd()
n, team = d["n"], d["team"]
x, y, bx, by = d["x"], d["y"], d["bx"], d["by"]
tracked = np.isfinite(x)
vis = np.isfinite(bx)
blocks = c.half_masks(n)
mid = c.L / 2
speed = c.rolling_mean(np.nanmean(d["speed"], axis=1), c.FPS // 2)
ev = c.load_tagging()
st_ko = [e["start_frame"] for e in ev if "Kick Off" in (e["tags"].get("Type") or "")]

# home defends the left half in half 1 and the right half in half 2 (verified in c6_hygiene)
home_left = np.zeros(n, bool)
home_left[:np.flatnonzero(blocks[1])[0] - 1000] = True


def kickoff_episodes(tol_m=1.0, max_violators=1, min_objects=16, ball_r=3.0, hold_s=2.0):
    own = np.where(home_left[:, None] == (team[None] == 0), x <= mid + tol_m, x >= mid - tol_m)
    violators = (tracked & ~own).sum(1)
    ball_ok = ~vis | (np.hypot(bx - mid, by - c.W / 2) <= ball_r)
    cond = (tracked.sum(1) >= min_objects) & (violators <= max_violators) & ball_ok & (speed < 1.5)
    s0, e0 = c.runs(cond)
    # merge interruptions under 1 s
    ms, me = [], []
    for a, b in zip(s0, e0):
        if ms and a - me[-1] < c.FPS:
            me[-1] = b
        else:
            ms.append(a); me.append(b)
    out = []
    for a, b in zip(ms, me):
        if b - a < hold_s * c.FPS:
            continue
        after = speed[b:b + 5 * c.FPS]
        if len(after) and np.nanmax(after) > 2.0:   # "then movement"
            out.append(b)
    return np.array(out)


def per_half(frames):
    res = []
    for h in blocks:
        lo, hi = np.flatnonzero(h)[[0, -1]]
        res.append(int(np.sum((frames >= lo - 10 * c.FPS) & (frames <= hi))))
    return res


def matched(frames, refs, tol_s=15):
    refs = np.array(refs)
    return int(sum(np.any(np.abs(refs - f) <= tol_s * c.FPS) for f in frames))


print("== kick-off detector ==  expected per half: 1 + goals in that half")
print("Smart Tagging kick-offs (Start + After Goal) per half: {}".format(per_half(np.array(st_ko))))
print("{:<44} {:>10} {:>14} {:>16}".format("variant", "per half", "match ST +/-15s", "ST recalled"))
for label, kw in (("default (tol 1 m, <=1 violator, hold 2 s)", {}),
                  ("strict  (tol 0.5 m, 0 violators)", {"tol_m": 0.5, "max_violators": 0}),
                  ("loose   (tol 2 m, <=2 violators)", {"tol_m": 2.0, "max_violators": 2}),
                  ("no hold (hold 0.5 s)", {"hold_s": 0.5})):
    k = kickoff_episodes(**kw)
    print("{:<44} {:>10} {:>14} {:>16}".format(label, str(per_half(k)), "{}/{}".format(matched(k, st_ko), len(k)),
                                               "{}/{}".format(matched(st_ko, k), len(st_ko))))

# ---- stationary-ball and gap-end restart templates ----------------------------------------
GOAL_AREA_W = 18.32 / 2 + 1.0
g = c.graded_mask(n)


def classify(px, py):
    cx = np.minimum(px, c.L - px)
    cy = np.minimum(py, c.W - py)
    if np.hypot(cx, cy) <= 3.0:
        return "corner"
    if cy <= 2.0:
        return "throw-in"
    if cx <= 6.5 and abs(py - c.W / 2) <= GOAL_AREA_W:
        return "goal kick"
    if np.hypot(px - mid, py - c.W / 2) <= 3.0:
        return "centre spot"
    return "other"


# A: ball visible and still (< 0.5 m drift over 1 s) for >= 1 s
disp = np.full(n, np.nan)
disp[c.FPS:] = np.hypot(bx[c.FPS:] - bx[:-c.FPS], by[c.FPS:] - by[:-c.FPS])
still = g & vis & (disp < 0.5)
s0, e0 = c.runs(still)
A = collections.Counter(classify(bx[a], by[a]) for a, b in zip(s0, e0) if b - a >= c.FPS)
# B: first visible ball position after a missing-ball gap >= 3 s
B = collections.Counter()
for h in blocks:
    s0, e0 = c.runs(h & ~vis)
    for a, b in zip(s0, e0):
        if b - a >= 3 * c.FPS and b < n and vis[b]:
            B[classify(bx[b], by[b])] += 1
# C: visible ball crossing a line (inside -> outside)
inside = (bx >= 0) & (bx <= c.L) & (by >= 0) & (by <= c.W)
cross = np.flatnonzero(g[1:] & vis[1:] & vis[:-1] & inside[:-1] & ~inside[1:]) + 1
C = collections.Counter("touchline" if (by[f] < 0 or by[f] > c.W) else "goal line" for f in cross)
# dedupe crossings within 3 s
print("\n== restart templates (counts per match) ==   typical: throw-ins 40-50, corners ~10, goal kicks 15-20")
print("{:<14} {:>18} {:>20} {:>14}".format("class", "A: still ball >=1s", "B: ball reappears", "Smart Tagging"))
st_map = {"corner": "Corner Kick", "throw-in": "Throw-in", "goal kick": "Goal Kick", "centre spot": "Kick Off"}
st_types = collections.Counter(e["tags"].get("Type") for e in ev if e["code"].strip().upper() == "SET PIECES")
for k in ("throw-in", "corner", "goal kick", "centre spot", "other"):
    stc = sum(v for t, v in st_types.items() if k in st_map and st_map[k] in (t or ""))
    print("{:<14} {:>18} {:>20} {:>14}".format(k, A.get(k, 0), B.get(k, 0), stc if k in st_map else "-"))
print("\nvisible ball crossing a line (inside -> outside, graded): {} total, {}".format(len(cross), dict(C)))
print("gaps >= 3 s: {}  of which ball reappears visible: {}".format(
    sum(1 for h in blocks for a, b in zip(*c.runs(h & ~vis)) if b - a >= 3 * c.FPS),
    sum(B.values())))
