"""Per-frame features for p_home = P(H in possession | ball alive) (P4).

    from mtah import features
    X, names = features.compute(game)          # (n, F) float64, NaN outside the halves
    g2 = features.mirror(game)                 # 180-degree rotation + H/A swap (augmentation)

The GT keeps possession with the last team in control through loose balls and deflections, so the
features describe three things:

1. proximity now   (bridged ball):  nearest H / A player, counts within NEAR_RADII_M, ball speed;
2. control         a "control event" is a frame where one team's nearest player is close to the
                   ball, clearly closer than the other team's, and moves with it. From the events:
                   the last and the next controlling team, the time since / until them, and the
                   share of control frames per team over past and future windows (CONTEXT_S);
3. team shape      when the ball is missing during live play only shape is left: centroids,
                   spread, and how fast each centroid moves along x (H attacks towards +x).

Rules (guide section 5): per-frame team aggregates only, never one slot followed over time (a
player's velocity is the same +/- 0.2 s centred difference players_speed uses); every window stays
inside one half; NaN is skipped inside windows and stays NaN where nothing is known (LightGBM
handles NaN natively, nothing is filled with 0).
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from mtah import ball as ball_mod
from mtah import config
from mtah.config import POSSESSION
from mtah.deadball import signals
from mtah.deadball.util import check_game, halves

NEED = ("ball_xy", "players_xy")         # per-frame arrays; players_team is per slot


# ------------------------------------------------------------------------------------------------
# small helpers (1-D, applied inside one half)
# ------------------------------------------------------------------------------------------------
def _window_mean(x: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Mean of the finite x over [t + lo, t + hi] (inclusive), cut at both ends; NaN if none."""
    ok = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(len(x))
    a, b = np.clip(t + lo, 0, len(x)), np.clip(t + hi + 1, 0, len(x))
    b = np.maximum(a, b)
    n_ok = count[b] - count[a]
    out = np.full(len(x), np.nan)
    np.divide(total[b] - total[a], n_ok, out=out, where=n_ok > 0)
    return out


def _centred_diff(x: np.ndarray, k: int, fps: float) -> np.ndarray:
    """(x[t + k] - x[t - k]) / (2k / fps); NaN where an end is missing or outside the array."""
    out = np.full(len(x), np.nan)
    if len(x) > 2 * k:
        out[k:-k] = (x[2 * k:] - x[:-2 * k]) * fps / (2 * k)
    return out


def _last_event(ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For a signed event array (+1 / -1 / 0): (value of the last nonzero at or before t, frames since).

    0 and NaN before the first event.
    """
    t = np.arange(len(ev))
    idx = np.maximum.accumulate(np.where(ev != 0, t, -1))
    has = idx >= 0
    val = np.where(has, ev[np.maximum(idx, 0)], 0.0)
    since = np.where(has, t - idx, np.nan).astype(np.float64)
    return val.astype(np.float64), since


# ------------------------------------------------------------------------------------------------
# frame-level (no time) quantities
# ------------------------------------------------------------------------------------------------
def _player_velocity(players_xy: np.ndarray, fps: float) -> np.ndarray:
    """(n, P, 2) m/s, centred difference over +/- VEL_HALF_WINDOW_S, NaN above SPEED_MAX_MPS."""
    k = max(1, int(round(POSSESSION.VEL_HALF_WINDOW_S * fps)))
    v = np.full(players_xy.shape, np.nan)
    if len(players_xy) > 2 * k:
        v[k:-k] = (players_xy[2 * k:] - players_xy[:-2 * k]) * fps / (2 * k)
    v[np.hypot(v[..., 0], v[..., 1]) > config.SPEED_MAX_MPS] = np.nan
    return v


def _nearest(dist: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per frame, (distance, slot index) of the nearest player among the slots in `mask`.

    Distance NaN (index 0) when the ball or every such player is missing.
    """
    d = np.where(mask[None, :] & np.isfinite(dist), dist, np.inf)
    idx = np.argmin(d, axis=1)
    dmin = d[np.arange(len(d)), idx]
    return np.where(np.isfinite(dmin), dmin, np.nan), idx


def _team_shape(px: np.ndarray, py: np.ndarray, mask: np.ndarray) -> dict:
    """Centroid, spread and tracked count of one team per frame (NaN when nobody is tracked)."""
    x = np.where(mask[None, :], px, np.nan)
    y = np.where(mask[None, :], py, np.nan)
    cnt = np.isfinite(x).sum(axis=1).astype(np.float64)
    with np.errstate(all="ignore"):
        sx, sy = np.nansum(x, axis=1), np.nansum(y, axis=1)
        cx, cy = sx / cnt, sy / cnt
        sdx = np.sqrt(np.nansum((x - cx[:, None]) ** 2, axis=1) / cnt)
        sdy = np.sqrt(np.nansum((y - cy[:, None]) ** 2, axis=1) / cnt)
    none = cnt == 0
    for a in (cx, cy, sdx, sdy):
        a[none] = np.nan
    return {"cx": cx, "cy": cy, "sdx": sdx, "sdy": sdy, "n": cnt}


# ------------------------------------------------------------------------------------------------
# public
# ------------------------------------------------------------------------------------------------
def mirror(game: Mapping) -> dict:
    """The same match seen from the other side: 180-degree rotation, H and A swapped.

    H still attacks towards +x afterwards (the old A attacked -x; the rotation turns it to +x), so
    the schema holds. Labels, if any, are swapped by the caller (swap_labels).
    """
    length, width = game["meta"]["pitch"]
    flip = np.array([length, width], dtype=np.float64)
    out = dict(game)
    out["ball_xy"] = flip - np.asarray(game["ball_xy"], dtype=np.float64)
    out["players_xy"] = flip - np.asarray(game["players_xy"], dtype=np.float64)
    team = np.asarray(game["players_team"])
    out["players_team"] = np.where(team == "H", "A", "H")
    return out


def swap_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return np.where(labels == "H", "A", np.where(labels == "A", "H", labels))


def control_events(game: Mapping, b: dict | None = None) -> np.ndarray:
    """(n,) +1 where H controls the ball, -1 where A does, 0 otherwise (see module docstring)."""
    return _frame_quantities(game, b)["ctrl"]


def _frame_quantities(game: Mapping, b: dict | None = None) -> dict:
    check_game(game, need=NEED)
    meta = game["meta"]
    fps = float(meta["fps"])
    if b is None:
        b = ball_mod.bridge(game)
    pxy = np.asarray(game["players_xy"], dtype=np.float64)
    team = np.asarray(game["players_team"])
    is_h, is_a = team == "H", team == "A"
    bxy = b["xy"]

    dist = np.hypot(pxy[..., 0] - bxy[:, None, 0], pxy[..., 1] - bxy[:, None, 1])
    d_h, i_h = _nearest(dist, is_h)
    d_a, i_a = _nearest(dist, is_a)

    pvel = _player_velocity(pxy, fps)
    rows = np.arange(len(pxy))
    rel_h = np.hypot(*(b["vel"] - pvel[rows, i_h]).T)
    rel_a = np.hypot(*(b["vel"] - pvel[rows, i_a]).T)
    rel_h[np.isnan(d_h)] = np.nan
    rel_a[np.isnan(d_a)] = np.nan

    near = {}
    ball_ok = np.isfinite(bxy[:, 0])
    for r in POSSESSION.NEAR_RADII_M:
        for name, m in (("h", is_h), ("a", is_a)):
            c = ((dist < r) & m[None, :]).sum(axis=1).astype(np.float64)
            near[f"n_{name}_{r:g}m"] = np.where(ball_ok, c, np.nan)

    def ctrl_of(d_own, d_other, rel):
        close = d_own < POSSESSION.CONTROL_DIST_M
        clear = np.where(np.isnan(d_other), True, d_own + POSSESSION.CONTROL_MARGIN_M < d_other)
        with_ball = np.where(np.isnan(rel), True, rel < POSSESSION.CONTROL_REL_SPEED)
        return close & clear & with_ball

    ctrl = ctrl_of(d_h, d_a, rel_h).astype(np.float64) - ctrl_of(d_a, d_h, rel_a)
    ctrl[~ball_ok] = 0.0
    return {"d_h": d_h, "d_a": d_a, "rel_h": rel_h, "rel_a": rel_a, "ctrl": ctrl, **near,
            "shape_h": _team_shape(pxy[..., 0], pxy[..., 1], is_h),
            "shape_a": _team_shape(pxy[..., 0], pxy[..., 1], is_a)}


GROUPS = ("proximity", "control", "shape", "gap")


def group_of(name: str) -> str:
    """Feature group of a column (module docstring, 1-3; gap = ball-missing context), for ablations."""
    if name.startswith(("ctrl_", "ddiff_")):
        return "control"
    if name.startswith(("cx_", "cy_", "sdx_", "sdy_", "n_tracked_", "vcx_", "sd_area")):
        return "shape"
    if name in ("gap_len_s", "t_since_seen_s", "t_until_seen_s"):
        return "gap"
    return "proximity"


def compute(game: Mapping) -> tuple[np.ndarray, list[str]]:
    """Feature matrix (n, F) float64 and the column names. NaN outside the halves."""
    check_game(game, need=NEED)
    meta = game["meta"]
    n, fps = int(meta["n_frames"]), float(meta["fps"])
    length, width = meta["pitch"]
    b = ball_mod.bridge(game)
    q = _frame_quantities(game, b)

    cols: dict[str, np.ndarray] = {}
    # 1. proximity now
    cols["d_h"], cols["d_a"] = q["d_h"], q["d_a"]
    cols["d_diff"] = q["d_a"] - q["d_h"]                     # > 0: H is closer
    cols["rel_h"], cols["rel_a"] = q["rel_h"], q["rel_a"]
    for k, v in q.items():
        if k.startswith("n_"):
            cols[k] = v
    cols["ball_speed"] = np.hypot(b["vel"][:, 0], b["vel"][:, 1])
    cols["ball_vx"] = b["vel"][:, 0]                         # + = towards the A goal
    cols["ball_x"] = b["xy"][:, 0] / length
    cols["ball_y_edge"] = np.minimum(b["xy"][:, 1], width - b["xy"][:, 1])
    cols["ball_seen"] = b["seen"].astype(np.float64)
    cols["ball_bridged"] = b["bridged"].astype(np.float64)
    # 3. team shape (per frame; velocities per half below)
    for t, s in (("h", q["shape_h"]), ("a", q["shape_a"])):
        cols[f"cx_{t}"] = s["cx"] / length
        cols[f"cy_{t}"] = s["cy"] / width
        cols[f"sdx_{t}"], cols[f"sdy_{t}"], cols[f"n_tracked_{t}"] = s["sdx"], s["sdy"], s["n"]
    cols["cx_diff"] = cols["cx_h"] - cols["cx_a"]
    cols["sd_area_diff"] = q["shape_h"]["sdx"] * q["shape_h"]["sdy"] - q["shape_a"]["sdx"] * q["shape_a"]["sdy"]
    # ball-missing context, shared with p_dead (log(1 + s))
    cols["gap_len_s"] = signals.gap_len_s(game)
    cols["t_since_seen_s"] = signals.t_since_seen_s(game)
    cols["t_until_seen_s"] = signals.t_until_seen_s(game)

    # 2. control + time context, per half
    timed = ["ctrl_last", "ctrl_last_s", "ctrl_next", "ctrl_next_s", "vcx_h", "vcx_a"]
    for w in POSSESSION.CONTEXT_S:
        timed += [f"ctrl_past_{w:g}s", f"ctrl_fut_{w:g}s", f"ddiff_past_{w:g}s", f"ddiff_fut_{w:g}s"]
    for name in timed:
        cols[name] = np.full(n, np.nan)
    kv = max(1, int(round(POSSESSION.SHAPE_VEL_S * fps)))
    for half in halves(game):
        ctrl = q["ctrl"][half]
        last, since = _last_event(ctrl)
        nxt, until = _last_event(ctrl[::-1])
        cols["ctrl_last"][half], cols["ctrl_last_s"][half] = last, np.log1p(since / fps)
        cols["ctrl_next"][half], cols["ctrl_next_s"][half] = nxt[::-1], np.log1p(until[::-1] / fps)
        dd = cols["d_diff"][half]
        for w in POSSESSION.CONTEXT_S:
            k = max(1, int(round(w * fps)))
            cols[f"ctrl_past_{w:g}s"][half] = _window_mean(ctrl, -k, -1)
            cols[f"ctrl_fut_{w:g}s"][half] = _window_mean(ctrl, 1, k)
            cols[f"ddiff_past_{w:g}s"][half] = _window_mean(dd, -k, -1)
            cols[f"ddiff_fut_{w:g}s"][half] = _window_mean(dd, 1, k)
        cols["vcx_h"][half] = _centred_diff(q["shape_h"]["cx"][half], kv, fps)
        cols["vcx_a"][half] = _centred_diff(q["shape_a"]["cx"][half], kv, fps)

    names = list(cols)
    X = np.column_stack([cols[c] for c in names]).astype(np.float64)
    inside = np.zeros(n, dtype=bool)
    for half in halves(game):
        inside[half] = True
    X[~inside] = np.nan
    return X, names
