"""Idea 7 -- broadcast feed as a second source, aligned without video (feasibility).

Alignment: normalised cross-correlation of team-centroid series (home x, away x, all-player y),
sampled at 5 fps, sliding 60 s broadcast windows across the whole tactical match. Aggregates only.
"""
import warnings

import numpy as np

import common as c
from load_homography import load as load_h

warnings.simplefilter("ignore")
STEP = 5                      # frames -> 5 fps
WIN = 60 * c.FPS // STEP      # window length, reset per configuration below
HOP = 30 * c.FPS // STEP
MIN_VALID = 0.8               # share of window samples with a valid broadcast homography
HP = 10 * c.FPS // STEP       # high-pass: subtract a centred 10 s moving average
SHARP_MARGIN = 0.15           # best minus best-outside-10s NCC to call a peak "clear"

t = c.load_atd("tactical")
b = c.load_atd("broadcast")
_, _, bvalid = load_h(source="broadcast")
g = c.graded_mask(t["n"])

# ---- presence and homography stretches ----------------------------------------------------
bvis = np.isfinite(b["bx"])
print("broadcast frames {}  ball present {:.1%}  | on valid-homography frames {:.1%}  | on invalid {:.1%}".format(
    b["n"], bvis.mean(), bvis[bvalid].mean(), bvis[~bvalid].mean()))
print("broadcast objects per frame on valid / invalid homography: {:.1f} / {:.1f}".format(
    np.isfinite(b["x"]).sum(1)[bvalid].mean(), np.isfinite(b["x"]).sum(1)[~bvalid].mean()))
s0, e0 = c.runs(~bvalid)
dur = (e0 - s0) / c.FPS
print("invalid-homography stretches: {}  duration p50/p90/max {:.1f}/{:.1f}/{:.0f} s  (total {:.0%} of frames)".format(
    len(dur), *np.percentile(dur, [50, 90, 100]), (~bvalid).mean()))


def channels(d, mask=None):
    hx = np.nanmean(np.where(d["team"][None] == 0, d["x"], np.nan), axis=1)
    ax = np.nanmean(np.where(d["team"][None] == 1, d["x"], np.nan), axis=1)
    ay = np.nanmean(d["y"], axis=1)
    cnt = np.isfinite(d["x"]).sum(1)
    ok = (cnt >= 8) & np.isfinite(hx) & np.isfinite(ax)
    if mask is not None:
        ok &= mask
    X = np.stack([hx, ax, ay], 1)[::STEP]
    return X, ok[::STEP]


def interp(X, ok):
    X = X.copy()
    idx = np.arange(len(X))
    for k in range(X.shape[1]):
        good = ok & np.isfinite(X[:, k])
        X[:, k] = np.interp(idx, idx[good], X[good, k])
    return X


T, Tok = channels(t)
T_raw = interp(T, Tok)
B, Bok = channels(b, bvalid)
Bi_raw = interp(B, Bok)


def highpass(X):
    return X - np.stack([c.rolling_mean(X[:, k], HP) for k in range(X.shape[1])], 1)


def ncc_all(series, w):
    """NCC of window w (m,) against every offset of series (N,)."""
    series, w = np.asarray(series, np.float64), np.asarray(w, np.float64)
    m = len(w)
    w = (w - w.mean())
    w = w / (np.linalg.norm(w) + 1e-9)
    nfft = 1 << int(np.ceil(np.log2(len(series) + m)))
    corr = np.fft.irfft(np.fft.rfft(series, nfft) * np.conj(np.fft.rfft(w, nfft)), nfft)[:len(series) - m + 1]
    cs = np.concatenate(([0], np.cumsum(series)))
    cs2 = np.concatenate(([0], np.cumsum(series ** 2)))
    s1 = cs[m:] - cs[:-m]
    s2 = cs2[m:] - cs2[:-m]
    var = s2 - s1 ** 2 / m
    flat = var < 1e-3 * m                     # flat stretches (e.g. interpolated halftime) blow up
    out = corr / np.sqrt(np.maximum(var, 1e-9))
    out[flat] = 0.0
    return out


def align(T, flip):
    TT = T.copy()
    if flip:
        TT[:, 0] = c.L - TT[:, 0]; TT[:, 1] = c.L - TT[:, 1]; TT[:, 2] = c.W - TT[:, 2]
    rows = []
    for start in range(0, len(Bi) - WIN, HOP):
        if Bok[start:start + WIN].mean() < MIN_VALID:
            continue
        score = np.mean([ncc_all(TT[:, k], Bi[start:start + WIN, k]) for k in range(3)], axis=0)
        best = int(np.argmax(score))
        excl = score.copy()
        excl[max(0, best - 10 * c.FPS // STEP):best + 10 * c.FPS // STEP] = -np.inf
        rows.append((start, best, score[best], score[best] - excl.max()))
    return np.array(rows)


def summarise(r, label):
    if len(r) == 0:
        print("{}: no windows".format(label))
        return None
    clear = r[:, 3] >= SHARP_MARGIN
    off = (r[:, 1] - r[:, 0]) * STEP / c.FPS          # tactical time - broadcast time, s
    print("{}: windows {}  best NCC p50 {:.2f}  margin p50 {:.2f}  clear {:.0%}".format(
        label, len(r), np.median(r[:, 2]), np.median(r[:, 3]), clear.mean()), end="")
    consec = clear[1:] & clear[:-1] & (np.diff(r[:, 0]) == HOP)
    if consec.sum():
        d_off = np.abs(np.diff(off))[consec]
        print("  | consecutive clear pairs {}  agree <=1 s {:.0%}  offset p10/p50/p90 {:.0f}/{:.0f}/{:.0f} s".format(
            consec.sum(), np.mean(d_off <= 1), *np.percentile(off[clear], [10, 50, 90])))
    else:
        print("  | no consecutive clear pairs")
    return r, off, clear


results = {}
for win_s in (60, 30):
    WIN, HOP = win_s * c.FPS // STEP, win_s * c.FPS // STEP // 2
    for hp in (False, True):
        T = highpass(T_raw) if hp else T_raw
        Bi = highpass(Bi_raw) if hp else Bi_raw
        for flip in (False, True):
            label = "win {} s, {:<9}, {:<8}".format(win_s, "high-pass" if hp else "raw", "mirrored" if flip else "as-is")
            results[(win_s, hp, flip)] = summarise(align(T, flip), label)

# pick the configuration with the most self-consistent windows (no labels involved)
def n_agree(res):
    if res is None:
        return -1
    r, off, clear = res
    consec = clear[1:] & clear[:-1] & (np.diff(r[:, 0]) == np.diff(r[:, 0]).min())
    return int(np.sum(np.abs(np.diff(off))[consec] <= 1))
best_cfg = max(results, key=lambda k: n_agree(results[k]))
print("\nmost self-consistent configuration: win {} s, high-pass {}, mirrored {}".format(*best_cfg))
WIN = best_cfg[0] * c.FPS // STEP
HOP = WIN // 2
keep = results[best_cfg]

# ---- per-half offset, refined to the frame, then fill rate ---------------------------------
r, off, clear = keep
consec = np.zeros(len(r), bool)
for i in range(len(r)):
    for j in (i - 1, i + 1):
        if 0 <= j < len(r) and clear[i] and clear[j] and abs(off[i] - off[j]) <= 1:
            consec[i] = True
print("\nconfirmed windows (clear + a neighbour agrees within 1 s): {}/{}".format(consec.sum(), len(r)))
tvis = np.isfinite(t["bx"])
tmiss = g & ~tvis
offsets = {}
for h, half in enumerate(c.half_masks(t["n"]), 1):
    lo, hi = np.flatnonzero(half)[[0, -1]]
    tf_win = r[:, 1] * STEP
    inh = consec & (tf_win >= lo) & (tf_win <= hi)
    vals, cnts = np.unique(np.round(off[inh]).astype(int), return_counts=True)
    if not len(vals):
        print("half {}: no confirmed windows".format(h)); continue
    coarse = int(vals[np.argmax(cnts)]) * c.FPS
    print("half {}: confirmed windows {}, share at the modal offset (+/-1 s) {:.0%}".format(
        h, inh.sum(), np.mean(np.abs(off[inh] - vals[np.argmax(cnts)]) <= 1)))
    best = None
    tf = np.arange(lo, hi + 1)
    for lag in range(coarse - 2 * c.FPS, coarse + 2 * c.FPS + 1):
        bf = tf - lag
        ok = (bf >= 0) & (bf < b["n"])
        bf2, tf2 = bf[ok], tf[ok]
        k = tvis[tf2] & bvis[bf2] & bvalid[bf2]
        if k.sum() < 1000:
            continue
        dist = np.hypot(t["bx"][tf2[k]] - b["bx"][bf2[k]], t["by"][tf2[k]] - b["by"][bf2[k]])
        med = np.median(dist)
        if best is None or med < best[1]:
            best = (lag, med, k.sum(), np.mean(dist < 2))
    offsets[h] = best[0]
    print("  refined: median tactical-vs-broadcast ball distance {:.2f} m on {:,} co-visible frames; "
          "within 2 m {:.0%};  (+/-1 s off the best lag: {:.1f} m)".format(best[1], best[2], best[3],
          np.nanmedian(np.hypot(*(np.array([t["bx"][tf2[k]], t["by"][tf2[k]]]) -
                                np.array([b["bx"][np.clip(bf2[k] + c.FPS, 0, b["n"] - 1)],
                                          b["by"][np.clip(bf2[k] + c.FPS, 0, b["n"] - 1)]]))))))
    bf = tf - best[0]
    ok = (bf >= 0) & (bf < b["n"])
    tf2, bf2 = tf[ok], bf[ok]
    miss = tmiss[tf2]
    fillable = miss & bvis[bf2] & bvalid[bf2]
    print("  tactical ball-missing frames in this half that the broadcast ball fills: {:.1%} "
          "(broadcast homography invalid on {:.0%} of them)".format(
              fillable.sum() / max(miss.sum(), 1), np.mean(~bvalid[bf2][miss])))
print("\nalignment is one constant offset per half; frame-level offsets are not printed on purpose")
