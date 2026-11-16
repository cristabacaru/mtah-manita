"""Ball-gap bridging for the team-possession model (P4).

    from mtah import ball
    b = ball.bridge(game)        # game = mtah.data.load(name)
    b["xy"], b["vel"], b["seen"], b["bridged"]

A constant-velocity Kalman filter + Rauch-Tung-Striebel smoother runs on each stretch of a half in
which no ball gap is longer than POSSESSION.BRIDGE_MAX_S. Inside such a stretch it

  * denoises the seen ball (the ATD ball jitters; distances to players are what the model uses);
  * fills short gaps, using the velocity on both sides (a pass between two sightings);
  * gives a ball velocity on every frame it covers.

Longer gaps, and gaps that touch a half edge (seen on one side only), stay NaN: extrapolating a ball
for seconds invents evidence. Those frames are covered by the team-shape features instead.

Outputs, all (n,) or (n, 2) float64 / bool, NaN / False outside the halves:
    xy       smoothed ball position, metres; NaN where not covered
    vel      smoothed ball velocity, m/s; NaN where not covered
    seen     the raw ball is there
    bridged  not seen, but filled by the smoother
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from mtah.config import POSSESSION
from mtah.deadball.util import ball_missing, check_game, halves, runs


def _rts(z: np.ndarray, dt: float, acc_std: float, meas_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Kalman filter + RTS smoother, constant-velocity model, x and y independent.

    z: (m, 2) positions with NaN where unobserved; z[0] and z[-1] are observed.
    Returns (pos, vel), each (m, 2). Both axes share one covariance (same model, same missing
    pattern), so the recursion keeps three scalars for it instead of two 2x2 matrices.
    """
    m = len(z)
    q = acc_std ** 2
    q00, q01, q11 = q * dt ** 4 / 4.0, q * dt ** 3 / 2.0, q * dt ** 2
    r = meas_std ** 2
    obs = np.isfinite(z[:, 0])
    zx, zy = z[:, 0].tolist(), z[:, 1].tolist()

    # filtered state (fp: position, fv: velocity per axis) and covariance; predicted covariance
    fpx, fvx, fpy, fvy = [0.0] * m, [0.0] * m, [0.0] * m, [0.0] * m
    f00, f01, f11 = [0.0] * m, [0.0] * m, [0.0] * m
    p00s, p01s, p11s = [0.0] * m, [0.0] * m, [0.0] * m

    px, vx, py, vy = zx[0], 0.0, zy[0], 0.0
    c00, c01, c11 = 1e4, 0.0, 100.0                     # wide prior; the t = 0 update sets px = z[0]
    for t in range(m):
        if t > 0:                                       # predict
            px, py = px + dt * vx, py + dt * vy
            c00, c01, c11 = (c00 + 2 * dt * c01 + dt * dt * c11 + q00,
                             c01 + dt * c11 + q01,
                             c11 + q11)
        p00s[t], p01s[t], p11s[t] = c00, c01, c11
        if obs[t]:                                      # update with the position
            s = c00 + r
            k0, k1 = c00 / s, c01 / s
            ex, ey = zx[t] - px, zy[t] - py
            px, vx, py, vy = px + k0 * ex, vx + k1 * ex, py + k0 * ey, vy + k1 * ey
            c00, c01, c11 = c00 - k0 * c00, c01 - k0 * c01, c11 - k1 * c01
        fpx[t], fvx[t], fpy[t], fvy[t] = px, vx, py, vy
        f00[t], f01[t], f11[t] = c00, c01, c11

    spx, svx, spy, svy = fpx[:], fvx[:], fpy[:], fvy[:]
    s00, s01, s11 = f00[-1], f01[-1], f11[-1]
    for t in range(m - 2, -1, -1):
        # C = P_f(t) F' P_pred(t+1)^-1, all 2x2 symmetric except C itself
        a00, a01, a11 = f00[t], f01[t], f11[t]
        b00, b01, b11 = p00s[t + 1], p01s[t + 1], p11s[t + 1]
        det = b00 * b11 - b01 * b01
        i00, i01, i11 = b11 / det, -b01 / det, b00 / det
        m00, m01 = a00 + dt * a01, a01                  # P_f F'  (row 0: [a00 + dt a01, a01])
        m10, m11 = a01 + dt * a11, a11                  #         (row 1: [a01 + dt a11, a11])
        g00, g01 = m00 * i00 + m01 * i01, m00 * i01 + m01 * i11
        g10, g11 = m10 * i00 + m11 * i01, m10 * i01 + m11 * i11
        dx_p, dx_v = spx[t + 1] - (fpx[t] + dt * fvx[t]), svx[t + 1] - fvx[t]
        dy_p, dy_v = spy[t + 1] - (fpy[t] + dt * fvy[t]), svy[t + 1] - fvy[t]
        spx[t], svx[t] = fpx[t] + g00 * dx_p + g01 * dx_v, fvx[t] + g10 * dx_p + g11 * dx_v
        spy[t], svy[t] = fpy[t] + g00 * dy_p + g01 * dy_v, fvy[t] + g10 * dy_p + g11 * dy_v
        # smoothed covariance, kept for the next step back: P_f + C (P_s(t+1) - P_pred(t+1)) C'
        e00, e01, e11 = s00 - b00, s01 - b01, s11 - b11
        h00, h01 = g00 * e00 + g01 * e01, g00 * e01 + g01 * e11
        h10, h11 = g10 * e00 + g11 * e01, g10 * e01 + g11 * e11
        s00 = a00 + h00 * g00 + h01 * g01
        s01 = a01 + h00 * g10 + h01 * g11
        s11 = a11 + h10 * g10 + h11 * g11
    return np.column_stack([spx, spy]), np.column_stack([svx, svy])


def bridge(game: Mapping) -> dict:
    """Smoothed ball position and velocity, gaps up to BRIDGE_MAX_S filled (module docstring)."""
    check_game(game, need=("ball_xy",))
    meta = game["meta"]
    n, fps = int(meta["n_frames"]), float(meta["fps"])
    max_gap = int(round(POSSESSION.BRIDGE_MAX_S * fps))
    z_all = np.asarray(game["ball_xy"], dtype=np.float64)
    missing = ball_missing(game)

    xy = np.full((n, 2), np.nan)
    vel = np.full((n, 2), np.nan)
    seen = np.zeros(n, dtype=bool)
    bridged = np.zeros(n, dtype=bool)
    for half in halves(game):
        miss = missing[half]
        seen[half] = ~miss
        if miss.all():
            continue
        # cut the half at every gap that is too long or touches an edge; smooth each piece
        size = len(miss)
        cut = np.zeros(size, dtype=bool)                # frames left NaN
        for start, stop in zip(*runs(miss)):
            if start == 0 or stop == size or stop - start > max_gap:
                cut[start:stop] = True
        for start, stop in zip(*runs(~cut)):
            z = z_all[half][start:stop]
            pos, v = _rts(z, 1.0 / fps, POSSESSION.KF_ACC_STD, POSSESSION.KF_MEAS_STD)
            xy[half.start + start:half.start + stop] = pos
            vel[half.start + start:half.start + stop] = v
        bridged[half] = miss & ~cut
    return {"xy": xy, "vel": vel, "seen": seen, "bridged": bridged}
