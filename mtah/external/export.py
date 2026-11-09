"""Write external games in the shared schema (guide section 5) with mtah.data.save.

    python -m mtah.external.export                      # 11 clean games + every suite game
    python -m mtah.external.export --games dfl_J03WMX   # one clean game

Each game lands in $MTAH_CACHE/<game>/ (clean) or $MTAH_CACHE/<game>__<level>/ (suite) and is read
back with mtah.data.load(game): ball_xy, players_xy, players_speed, players_team, meta. External
games also get labels.npy (n,) <U1 H/A/D/X there (training targets for leave-one-match-out); the
GT CSV for the scorer is in data/validation/<game>/<game>_gt.csv (guide section 4).

Conventions, all from guide section 5:
  * speed: mtah.data.position_speed on positions BEFORE the flip (NaN at gaps, above 12 m/s);
  * orientation: H attacks towards x = L in every half (180-degree rotation, mtah.data.rotate_halves,
    the same as kloppy's STATIC_HOME_AWAY);
  * y axis before rotation: ATD/Metrica, origin top-left, y top-to-bottom (DFL converted in dfl.py);
  * players_team "H"/"A"; halves inclusive [first, last].
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from mtah import config, data  # noqa: E402
from mtah.schema import Game, write_gt_csv  # noqa: E402
from mtah.external.build_suite import load_manifest, resolve  # noqa: E402

CACHE = str(config.EXTERNAL.GAMES_DIR)


def to_schema(g):
    """(arrays, meta) for mtah.data.save from an mtah.schema.Game."""
    pxy = g.players[..., :2].astype(np.float64)
    speed = data.position_speed(pxy, g.fps)                       # before the flip
    halves_incl = [(int(a), int(b) - 1) for a, b in g.halves]
    half_idx = data.frame_half(g.n, halves_incl)
    first_right = g.home_attacks_right_first_half
    home_right = np.where(half_idx % 2 == 0, first_right, not first_right)
    flip = ~home_right
    arrays = {"ball_xy": data.rotate_halves(g.ball[:, :2], flip, g.pitch),
              "players_xy": data.rotate_halves(pxy, flip, g.pitch),
              "players_speed": speed,
              "players_team": np.where(g.team == 0, "H", "A").astype("<U1")}
    meta = {"source": g.meta.get("source"), "n_frames": int(g.n), "fps": int(g.fps),
            "pitch": [float(v) for v in g.pitch], "halves": [list(h) for h in halves_incl],
            "home_name": g.meta.get("home", "Home"), "away_name": g.meta.get("away", "Away"),
            "orientation": "H attacks towards x = L in every half",
            "flipped_halves": sorted({int(i) for i in np.unique(half_idx[flip])}),
            "speed": "from positions, half window {} s, NaN above {} m/s".format(
                config.SPEED_HALF_WINDOW_S, config.SPEED_MAX_MPS),
            "external": True, "has_labels": g.labels is not None,
            "match": g.meta.get("match"), "licence": g.meta.get("licence"),
            "labels_source": g.meta.get("labels"), "degraded": g.meta.get("degraded")}
    return arrays, meta


def write(g, name=None):
    name = name or g.name
    arrays, meta = to_schema(g)
    folder = data.save(name, arrays, meta)
    if g.labels is not None:
        np.save(folder / "labels.npy", g.labels.astype("<U1"))
    return folder


def check(folder):
    """Share of frames with the H centroid at smaller x than A, per half (H attacks +x)."""
    d = data.load(os.path.basename(str(folder)))
    team, pxy = d["players_team"], np.asarray(d["players_xy"])
    out = []
    for a, b in d["meta"]["halves"]:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            hx = np.nanmean(pxy[a:b + 1, team == "H", 0], axis=1)
            ax = np.nanmean(pxy[a:b + 1, team == "A", 0], axis=1)
        ok = np.isfinite(hx) & np.isfinite(ax)
        out.append(round(float((hx[ok] < ax[ok]).mean()), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+")
    ap.add_argument("--no-suite", action="store_true")
    args = ap.parse_args()
    suite = {m["name"]: resolve(m["folder"]) for m in load_manifest()["games"]} \
        if os.path.exists(config.EXTERNAL.SUITE_MANIFEST) else {}
    todo = []
    for nm in args.games or config.EXTERNAL.CLEAN_GAMES:
        todo.append((nm, suite[nm], False) if nm in suite else (nm, os.path.join(CACHE, nm), True))
    if not args.games and not args.no_suite:
        todo += [(nm, folder, False) for nm, folder in suite.items()]
    for nm, src, clean in todo:
        g = Game.load(src)
        folder = write(g, nm)
        if clean and g.labels is not None:          # suite games already have their GT
            write_gt_csv(os.path.join(str(config.VALIDATION_DIR), nm, nm + "_gt.csv"), g.labels,
                         g.fps, g.halves)
        print("{:<34} -> {}  H left of A per half {}".format(nm, folder, check(folder)),
              flush=True)


if __name__ == "__main__":
    main()
