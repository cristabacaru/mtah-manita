"""Build the external validation suite: every clean game x every noise level, written as real
competition-format ATD files + GT, plus a manifest.

    python -m mtah.external.build_suite                         # 9 games x 3 levels, seed 0
    python -m mtah.external.build_suite --games metrica1_events --levels target

Output, per game and level, in data/validation/<game>__<level>/ (guide section 4):
    <game>__<level>_gt.csv + _gt_meta.json    frame,label (H/A/D, X outside the halves)
    <game>__<level>_atd.txt / _atd.xml        same format as the BAR-ATH ATD (any ATD parser runs)
    game.npz + game.json                      the same data as an mtah.schema.Game
    stats.json                                aggregate stats (compare with the target's)
and data/validation/suite_manifest.json listing all of them with their fold. Paths in the
manifest are relative to data/validation/. Then `python -m mtah.external.export` writes every suite
game into $MTAH_CACHE/<game>__<level>/ in the shared schema (mtah.data.load).

Folds: leave-one-match-out. Each source match is its own fold; the levels of one match share it.
"""

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from mtah import config  # noqa: E402
from mtah.schema import Game, possession_summary  # noqa: E402
from mtah.simulate import degrade, load_target, load_params, target_hash, LEVELS  # noqa: E402
from mtah.external import atd_io, stats  # noqa: E402

CACHE = str(config.EXTERNAL.GAMES_DIR)
VALIDATION = str(config.VALIDATION_DIR)
MANIFEST = str(config.EXTERNAL.SUITE_MANIFEST)
DEFAULT_GAMES = list(config.EXTERNAL.SUITE_GAMES)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fold_of(name):
    return "_".join(name.split("_")[:2]) if name.startswith("dfl") else name.split("_")[0]


def build_one(name, level, seed, target, params=None, write_atd=True):
    src = Game.load(os.path.join(CACHE, name))
    g = degrade(src, level, seed, target, params) if level != "clean" else src
    tag = "{}__{}".format(name, level) if seed == 0 else "{}__{}__s{}".format(name, level, seed)
    folder = os.path.join(VALIDATION, tag)
    g.name = tag
    g.save(folder)
    paths = atd_io.write(g, folder, tag) if write_atd else {}
    st = stats.aggregate_stats(g)
    with open(os.path.join(folder, "stats.json"), "w") as fh:
        json.dump(st, fh, indent=1)
    rel = lambda p: os.path.relpath(p, VALIDATION) if p else None  # noqa: E731
    return {"name": tag, "source": name, "level": level, "seed": seed, "fold": fold_of(name),
            "folder": tag, "atd_raw": rel(paths.get("atd_raw")),
            "atd_meta": rel(paths.get("atd_meta")), "gt": rel(paths.get("gt")),
            "n": g.n, "halves": [list(map(int, h)) for h in g.halves],
            "gt_summary": possession_summary(g.labels),
            "ball_missing": st["ball_missing"], "objects_mean": st["objects_mean"],
            "sha256_gt": sha256(paths["gt"]) if paths else None,
            "sha256_atd": sha256(paths["atd_raw"]) if paths else None}


def load_manifest():
    with open(MANIFEST) as fh:
        return json.load(fh)


def resolve(rel):
    """Manifest path (relative to data/validation/) -> absolute path."""
    return os.path.join(VALIDATION, rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS) + ["clean"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[config.EXTERNAL.SEED])
    args = ap.parse_args()
    target = load_target()
    params = load_params(target)
    os.makedirs(VALIDATION, exist_ok=True)
    manifest = load_manifest() if os.path.exists(MANIFEST) else {"games": []}
    have = {m["name"]: i for i, m in enumerate(manifest["games"])}
    for name in args.games:
        for level in args.levels:
            for seed in args.seeds:
                entry = build_one(name, level, seed, target, params)
                if entry["name"] in have:
                    manifest["games"][have[entry["name"]]] = entry
                else:
                    have[entry["name"]] = len(manifest["games"])
                    manifest["games"].append(entry)
                print("{:<34} missing {:.3f} objects {:5.2f} spells {}".format(
                    entry["name"], entry["ball_missing"], entry["objects_mean"],
                    entry["gt_summary"]["spells"]), flush=True)
    manifest["target_provisional"] = bool(target.get("provisional", False))
    manifest["sim_params_fitted"] = bool(params.get("fitted", False))
    manifest["target_hash"] = target_hash(target)
    manifest["paths_relative_to"] = "data/validation (config.VALIDATION_DIR)"
    manifest["games"].sort(key=lambda m: m["name"])
    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=1)
    print("manifest:", MANIFEST)


if __name__ == "__main__":
    main()
