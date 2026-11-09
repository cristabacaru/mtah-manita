"""The one per-match container every module consumes (the M1 contract).

The competition match (parsed from the ATD by P1) and every external validation game (Metrica,
DFL, degraded copies of them) are turned into a `Game`. A module that works on one works on all.

Conventions
-----------
* Row i is video frame i, 25 fps. For external games the "video frame" is tracking frame - 1.
* Coordinates are metres with the origin at a pitch corner: x in [0, L] along the length,
  y in [0, W] across it. This is exactly ATD fraction * pitch size.
* players[t, s] = (x, y, speed m/s) for track slot s; NaN where the slot is not seen. A slot is a
  tracklet in degraded / competition data and a real player in clean external data.
* team[s] = 0 home, 1 away.
* ball[t] = (x, y, z); NaN where the ball is not seen (z is NaN on most frames in the ATD).
* halves = [(a, b), ...] half-open frame ranges of the graded periods.
* labels (optional) = H / A / D / X per frame; X outside the halves.
* home_attacks_right_first_half: home attacks towards x = L in the first half.

Saved as <dir>/game.npz + <dir>/game.json; `Game.load(dir)` reverses `save`.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np

FPS = 25
HOME, AWAY = 0, 1
LABELS = ("H", "A", "D", "X")


@dataclass
class Game:
    name: str
    players: np.ndarray            # (n, S, 3) float32
    team: np.ndarray               # (S,) int8
    ball: np.ndarray               # (n, 3) float32
    halves: list                   # [(a, b)] half-open
    pitch: tuple                   # (L, W) metres
    home_attacks_right_first_half: bool
    labels: np.ndarray = None      # (n,) <U1 or None
    fps: int = FPS
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- basic views
    @property
    def n(self):
        return len(self.ball)

    def half_of(self):
        """(n,) int8: 1, 2, ... for frames inside a half, 0 elsewhere."""
        out = np.zeros(self.n, np.int8)
        for i, (a, b) in enumerate(self.halves, start=1):
            out[a:b] = i
        return out

    def graded(self):
        return self.half_of() > 0

    def home_attacks_right(self):
        """(n,) bool per frame (directions swap every half)."""
        h = self.half_of()
        first = self.home_attacks_right_first_half
        return np.where(h % 2 == 1, first, not first) & (h > 0)

    # ---------------------------------------------------------------- checks
    def validate(self):
        n, S = self.n, self.players.shape[1]
        assert self.players.shape == (n, S, 3), self.players.shape
        assert self.team.shape == (S,), self.team.shape
        assert set(np.unique(self.team)) <= {HOME, AWAY}
        assert self.ball.shape == (n, 3), self.ball.shape
        prev = 0
        for a, b in self.halves:
            assert prev <= a < b <= n, (a, b, n)
            prev = b
        if self.labels is not None:
            assert self.labels.shape == (n,)
            assert set(np.unique(self.labels)) <= set(LABELS)
            assert np.all(self.labels[~self.graded()] == "X"), "labels outside halves must be X"
        L, W = self.pitch
        xy = self.players[..., :2]
        ok = np.isfinite(xy[..., 0])
        if ok.any():
            assert np.nanmin(xy[..., 0]) > -15 and np.nanmax(xy[..., 0]) < L + 15, "x not metres?"
            assert np.nanmin(xy[..., 1]) > -15 and np.nanmax(xy[..., 1]) < W + 15, "y not metres?"
        return self

    def orientation_check(self):
        """Fraction of each half's frames where the home centroid sits on the side it defends.

        Home defends x=0 when it attacks right. Should be well above 0.5 for every half.
        """
        out = []
        for i, (a, b) in enumerate(self.halves):
            px = self.players[a:b, :, 0]
            with np.errstate(all="ignore"):
                hx = np.nanmean(np.where(self.team == HOME, px, np.nan), axis=1)
                ax = np.nanmean(np.where(self.team == AWAY, px, np.nan), axis=1)
            right = self.home_attacks_right_first_half if i % 2 == 0 else \
                not self.home_attacks_right_first_half
            ok = np.isfinite(hx) & np.isfinite(ax)
            agree = (hx < ax) if right else (hx > ax)
            out.append(float(agree[ok].mean()) if ok.any() else float("nan"))
        return out

    # ---------------------------------------------------------------- io
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        arrays = {"players": self.players.astype(np.float32), "team": self.team.astype(np.int8),
                  "ball": self.ball.astype(np.float32)}
        if self.labels is not None:
            arrays["labels"] = self.labels.astype("<U1")
        np.savez_compressed(os.path.join(folder, "game.npz"), **arrays)
        meta = {"name": self.name, "halves": [list(map(int, h)) for h in self.halves],
                "pitch": list(map(float, self.pitch)), "fps": int(self.fps),
                "home_attacks_right_first_half": bool(self.home_attacks_right_first_half),
                "n": int(self.n), "slots": int(self.players.shape[1]), "meta": self.meta}
        # game.json, not meta.json: the team's shared format (mtah/external/export.py) writes its
        # own meta.json into the same $MTAH_CACHE/<game>/ folder
        with open(os.path.join(folder, "game.json"), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
        return folder

    @classmethod
    def load(cls, folder):
        path = os.path.join(folder, "game.json")
        if not os.path.exists(path):                    # caches written before 2026-11-07
            path = os.path.join(folder, "meta.json")
        with open(path) as fh:
            m = json.load(fh)
        if "home_attacks_right_first_half" not in m:
            raise ValueError("{} holds no Game metadata (team-format meta.json?)".format(folder))
        z = np.load(os.path.join(folder, "game.npz"))
        return cls(name=m["name"], players=z["players"], team=z["team"], ball=z["ball"],
                   halves=[tuple(h) for h in m["halves"]], pitch=tuple(m["pitch"]),
                   home_attacks_right_first_half=m["home_attacks_right_first_half"],
                   labels=z["labels"] if "labels" in z.files else None, fps=m["fps"],
                   meta=m.get("meta", {}))


# -------------------------------------------------------------------- shared helpers
def runs(lab):
    """(starts, stops, values) of constant runs of a 1-D array."""
    lab = np.asarray(lab)
    if len(lab) == 0:
        return np.array([], int), np.array([], int), lab[:0]
    edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, len(lab)]
    return starts, stops, lab[starts]


def speed_from_xy(xy, halves, fps=FPS, smooth=5):
    """Speed (m/s) of each track from positions, per half, centred difference of a smoothed
    track. xy (n, S, 2) with NaN gaps; NaN stays NaN."""
    out = np.full(xy.shape[:2], np.nan, np.float32)
    for a, b in halves:
        seg = xy[a:b].astype(np.float64)
        if smooth > 1 and len(seg) > smooth:
            # moving average along time that propagates NaN (keeps gaps honest)
            cs = np.cumsum(np.nan_to_num(seg), axis=0)
            cn = np.cumsum(np.isfinite(seg[..., :1]), axis=0)
            pad = smooth // 2
            s = np.full_like(seg, np.nan)
            num = cs[smooth - 1:] - np.r_[np.zeros((1,) + seg.shape[1:]), cs[:-smooth]]
            cnt = cn[smooth - 1:] - np.r_[np.zeros((1,) + cn.shape[1:]), cn[:-smooth]]
            with np.errstate(all="ignore"):
                s[pad:pad + len(num)] = np.where(cnt == smooth, num / smooth, np.nan)
            seg = s
        v = np.full_like(seg, np.nan)
        v[1:-1] = (seg[2:] - seg[:-2]) * (fps / 2.0)
        out[a:b] = np.hypot(v[..., 0], v[..., 1])
    return out


def write_gt_csv(path, labels, fps=FPS, halves=None):
    """<name>.csv (frame,label) + <name>_meta.json: n_video_frames, n_graded_frames, fps, inclusive
    halves and label counts (guide section 4); the official scorer reads n_video_frames."""
    labels = np.asarray(labels)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    meta = {"n_video_frames": int(len(labels)), "n_graded_frames": int((labels != "X").sum()),
            "fps": int(fps), "label_counts": {k: int((labels == k).sum()) for k in LABELS}}
    if halves is not None:
        meta["halves"] = [[int(a), int(b) - 1] for a, b in halves]     # inclusive, like guide 5
    with open(path[:-len(".csv")] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    return path


def possession_summary(labels, fps=FPS):
    """Spell statistics using the scorer's definition (D splits spells, X ignored)."""
    labels = np.asarray(labels)
    graded = labels != "X"
    s, e, v = runs(labels)
    spell = np.isin(v, ["H", "A"])
    dur = (e - s)[spell] / fps
    dead = (e - s)[v == "D"] / fps
    minutes = graded.sum() / fps / 60.0
    return {
        "graded_min": round(minutes, 1),
        "spells": int(spell.sum()),
        "spells_per_min": round(spell.sum() / minutes, 2) if minutes else 0,
        "mean_s": round(float(dur.mean()), 2) if len(dur) else 0,
        "median_s": round(float(np.median(dur)), 2) if len(dur) else 0,
        "share_lt1s": round(float((dur < 1).mean()), 3) if len(dur) else 0,
        "share_lt2s": round(float((dur < 2).mean()), 3) if len(dur) else 0,
        "D_share": round(float((labels[graded] == "D").mean()), 3),
        "H_share": round(float((labels[graded] == "H").mean()), 3),
        "stoppages": int((v == "D").sum()),
        "dead_median_s": round(float(np.median(dead)), 2) if len(dead) else 0,
    }
