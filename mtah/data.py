"""Tracking data in the shared schema (docs/LABELLING_GUIDE.md section 5), cached per game.

    python -m mtah.data            # build the BAR-ATH cache (about 15 s) and print a summary
    python -m mtah.data --force    # rebuild even if the cache is there

    from mtah import data
    d = data.load()                # BAR-ATH; data.load("<external game>") for P2's games
    d["ball_xy"], d["players_xy"], d["players_speed"], d["players_team"], d["meta"]

What is in the cache ($MTAH_CACHE/<game>/), for every game:

    ball_xy.npy        (n, 2)     float32  metres, NaN when the ball is missing
    players_xy.npy     (n, P, 2)  float32  metres, NaN when the slot is empty
    players_speed.npy  (n, P)     float32  m/s, from positions (position_speed), NaN at gaps
                                           and above config.SPEED_MAX_MPS (12 m/s)
    players_team.npy   (P,)       <U1      "H" / "A"
    meta.json          pitch [L, W], fps, halves [[first, last], ...] inclusive, n_frames, ...

Conventions (guide section 5): row i = frame i of that recording (for BAR-ATH, video frame i =
submission row i); H attacks towards x = L in every half; speed is computed from positions for
every game, never taken from the provider.

P2: build external games with your loader, then call save(game, ...) with the same arrays; it
checks the schema before writing, and position_speed() gives the same speed as here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from mtah import config

SCHEMA_VERSION = 1
ARRAY_KEYS = ("ball_xy", "players_xy", "players_speed", "players_team")


# =================================================================================================
# shared helpers (any game)
# =================================================================================================
def position_speed(players_xy: np.ndarray, fps: float,
                   half_window_s: float = config.SPEED_HALF_WINDOW_S,
                   max_speed: float = config.SPEED_MAX_MPS) -> np.ndarray:
    """Speed in m/s from positions: |xy(t+k) - xy(t-k)| / (2k / fps), k = round(half_window_s * fps).

    NaN where the slot is empty at t, where either end is missing, where an end falls outside the
    recording, or where the speed is above max_speed (a slot that jumped to another person, not a
    real run). Use it on positions BEFORE any
    per-half flip (a 180 degree flip mid-recording would look like a jump); it preserves distances,
    so flipping afterwards does not change the result.
    """
    xy = np.asarray(players_xy, dtype=np.float64)
    k = max(1, int(round(half_window_s * fps)))
    out = np.full(xy.shape[:-1], np.nan)
    if xy.shape[0] > 2 * k:
        step = xy[2 * k:] - xy[:-2 * k]
        out[k:-k] = np.hypot(step[..., 0], step[..., 1]) * fps / (2 * k)
    out[np.isnan(xy[..., 0])] = np.nan        # nobody in the slot at t: no speed, even if t +/- k exist
    out[out > max_speed] = np.nan             # identity jump, not a run (config.SPEED_MAX_MPS)
    return out


def rotate_halves(xy: np.ndarray, flip: np.ndarray, pitch) -> np.ndarray:
    """x -> L - x and y -> W - y on the frames where flip is True (180 degree rotation). NaN stays NaN."""
    length, width = pitch
    out = np.array(xy, dtype=np.float64, copy=True)
    out[flip, ..., 0] = length - out[flip, ..., 0]
    out[flip, ..., 1] = width - out[flip, ..., 1]
    return out


def frame_half(n_frames: int, halves) -> np.ndarray:
    """Per frame, the index (0, 1, ...) of the half whose orientation it is drawn in.

    A frame belongs to the latest half whose first frame it has reached; frames before the first
    half belong to half 0. Ungraded frames (before kick-off, half-time) are never scored, so this
    only decides which way they are drawn.
    """
    idx = np.zeros(n_frames, dtype=np.int8)
    for i, (first, _last) in enumerate(halves):
        idx[first:] = i
    return idx


def validate(arrays: dict, meta: dict) -> None:
    """Raise ValueError if arrays/meta do not follow the guide section 5 schema."""
    missing = [k for k in ARRAY_KEYS if k not in arrays]
    if missing:
        raise ValueError(f"missing arrays: {missing}")
    for k in ("pitch", "fps", "halves", "n_frames"):
        if k not in meta:
            raise ValueError(f"meta has no {k!r}")
    n = int(meta["n_frames"])
    ball, pxy, psp, team = (np.asarray(arrays[k]) for k in ARRAY_KEYS)
    p = team.shape[0]
    checks = [
        (ball.shape == (n, 2), f"ball_xy shape {ball.shape}, expected ({n}, 2)"),
        (pxy.shape == (n, p, 2), f"players_xy shape {pxy.shape}, expected ({n}, {p}, 2)"),
        (psp.shape == (n, p), f"players_speed shape {psp.shape}, expected ({n}, {p})"),
        (team.ndim == 1 and set(team.tolist()) <= {"H", "A"}, "players_team must be 1-D 'H'/'A'"),
        (len(meta["pitch"]) == 2 and min(meta["pitch"]) > 0, "pitch must be [L, W] > 0"),
        (all(0 <= a <= b < n for a, b in meta["halves"]), "halves must be inclusive [first, last] in range"),
        (np.array_equal(np.isnan(ball[:, 0]), np.isnan(ball[:, 1])), "ball_xy: x and y must be NaN together"),
        (np.all(np.isnan(psp[np.isnan(pxy[..., 0])])), "players_speed must be NaN where the slot is empty"),
        (np.all(psp[np.isfinite(psp)] >= 0), "players_speed must be >= 0"),
        (np.all(psp[np.isfinite(psp)] <= config.SPEED_MAX_MPS),
         f"players_speed must be <= {config.SPEED_MAX_MPS} m/s (use position_speed)"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise ValueError("schema check failed:\n  " + "\n  ".join(bad))


def save(game: str, arrays: dict, meta: dict) -> Path:
    """Validate, then write the arrays and meta.json to $MTAH_CACHE/<game>/. meta.json is written last,
    so a folder with a meta.json is complete."""
    validate(arrays, meta)
    folder = config.game_cache(game)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "meta.json").unlink(missing_ok=True)
    np.save(folder / "ball_xy.npy", np.asarray(arrays["ball_xy"], dtype=np.float32))
    np.save(folder / "players_xy.npy", np.asarray(arrays["players_xy"], dtype=np.float32))
    np.save(folder / "players_speed.npy", np.asarray(arrays["players_speed"], dtype=np.float32))
    np.save(folder / "players_team.npy", np.asarray(arrays["players_team"], dtype="<U1"))
    meta = dict(meta, game=game, schema_version=SCHEMA_VERSION)
    with open(folder / "meta.json", "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return folder


def load(game: str = config.GAME, mmap: bool = True) -> dict:
    """{ball_xy, players_xy, players_speed, players_team, meta} for one game from the cache.

    mmap=True opens the big arrays read-only without reading them into memory up front.
    """
    folder = config.game_cache(game)
    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        hint = "python -m mtah.data" if game == config.GAME else "P2's loader for that game"
        raise FileNotFoundError(f"no cache for {game!r} in {folder}. Build it first: {hint}")
    meta = json.loads(meta_path.read_text())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{game}: cache schema {meta.get('schema_version')}, code expects "
                           f"{SCHEMA_VERSION}. Rebuild it (python -m mtah.data --force for BAR-ATH).")
    mode = "r" if mmap else None
    out = {k: np.load(folder / f"{k}.npy", mmap_mode=mode if k != "players_team" else None)
           for k in ARRAY_KEYS}
    out["meta"] = meta
    return out


# =================================================================================================
# BAR-ATH (FIFA EPTS: <name>_atd.xml + <name>_atd.txt)
# =================================================================================================
def read_atd_meta(xml_path: Path) -> dict:
    """Slot order, slot -> H/A, team names, frame rate and first-half attack direction, all from the XML.

    Slot order comes from the DataFormatSpecification (PlayerChannelRef player<k>_x, in file order),
    home/away from Score idLocalTeam / idVisitingTeam, direction from attack_direction_first_half.
    """
    root = ET.parse(xml_path).getroot()
    teams = {t.get("id"): t for t in root.iter("Team")}
    score = next(root.iter("Score"))
    home_id, away_id = score.get("idLocalTeam"), score.get("idVisitingTeam")
    side_of_team = {home_id: "H", away_id: "A"}

    player_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    spec = next(root.iter("DataFormatSpecification"))
    # One SplitRegister (x, y, speed) per slot, in line order; its x channel names the player.
    slot_players = [channel_player[ref.get("playerChannelId")]
                    for ref in spec.iter("PlayerChannelRef")
                    if ref.get("playerChannelId").endswith("_x")]
    slot_team = np.array([side_of_team[player_team[pid]] for pid in slot_players], dtype="<U1")

    def first_half_direction(team_id):
        for pp in teams[team_id].iter("ProviderParameter"):
            if pp.findtext("Name") == "attack_direction_first_half":
                return pp.findtext("Value")
        raise ValueError(f"no attack_direction_first_half for team {team_id}")

    home_dir, away_dir = first_half_direction(home_id), first_half_direction(away_id)
    if {home_dir, away_dir} != {"left_to_right", "right_to_left"}:
        raise ValueError(f"unexpected attack directions: home {home_dir}, away {away_dir}")

    return {
        "slot_team": slot_team,
        "home_name": teams[home_id].findtext("Name"),
        "away_name": teams[away_id].findtext("Name"),
        "home_attack_first_half": home_dir,
        "frame_rate": float(root.findtext(".//FrameRate")),
        "frame_id_range": (int(spec.get("startFrame")), int(spec.get("endFrame"))),
        "ball_channels": [ref.get("channelId") for ref in spec.iter("BallChannelRef")],
    }


def parse_atd_raw(txt_path: Path, n_slots: int):
    """(frame_ids (n,), players (n, P, 3) [x, y, speed], ball (n, 3)), normalised units as delivered.

    Every line is "<frame>:<x,y,s>;...;<x,y,s>:<x,y,s>", so after turning ':' and ';' into ',' the
    file is a plain CSV with 1 + 3P + 3 columns, which pandas' C parser reads in seconds.
    """
    import pandas as pd

    text = Path(txt_path).read_text().replace(":", ",").replace(";", ",")
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64, engine="c").to_numpy()
    expected = 1 + 3 * n_slots + 3
    if table.shape[1] != expected:
        raise ValueError(f"{txt_path.name}: {table.shape[1]} columns per line, expected {expected}")
    frame_ids = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * n_slots].reshape(len(table), n_slots, 3)
    ball = table[:, 1 + 3 * n_slots:]
    return frame_ids, players, ball


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_target(force: bool = False) -> Path:
    """Parse the BAR-ATH ATD into the shared schema and save it. Returns the cache folder."""
    folder = config.game_cache(config.GAME)
    if (folder / "meta.json").is_file() and not force:
        return folder
    missing = config.missing_assets(("atd_meta", "atd_raw"))
    if missing:
        raise FileNotFoundError("match data not found: " + ", ".join(map(str, missing))
                                + " (see data/README.md)")

    xml_meta = read_atd_meta(config.ASSETS["atd_meta"])
    if xml_meta["ball_channels"][:2] != ["x", "y"]:
        raise ValueError(f"ball channels {xml_meta['ball_channels']}: expected x, y first")
    if xml_meta["frame_rate"] != config.FPS:
        raise ValueError(f"XML frame rate {xml_meta['frame_rate']} != games.py fps {config.FPS}")
    slot_team = xml_meta["slot_team"]

    frame_ids, players, ball = parse_atd_raw(config.ASSETS["atd_raw"], len(slot_team))
    n = config.N_FRAMES
    video = frame_ids - config.ATD_FRAME_ID_START
    if not np.array_equal(video, np.arange(n)):
        raise ValueError(f"ATD is not one record per video frame 0..{n - 1}")

    length, width = config.PITCH
    scale = np.array([length, width])
    players_xy = players[..., :2] * scale            # normalised -> metres (guide 5: games.py pitch)
    ball_xy = ball[:, :2] * scale
    speed = position_speed(players_xy, config.FPS)    # before the flip, see position_speed

    halves = [(first, last) for _, first, last in config.HALF_BLOCKS]
    half_idx = frame_half(n, halves)
    first_half_ltr = xml_meta["home_attack_first_half"] == "left_to_right"
    # Teams change ends at half-time: H attacks left to right in halves 0, 2, ... iff it did in half 0.
    home_ltr = np.where(half_idx % 2 == 0, first_half_ltr, not first_half_ltr)
    flip = ~home_ltr
    players_xy = rotate_halves(players_xy, flip, config.PITCH)
    ball_xy = rotate_halves(ball_xy, flip, config.PITCH)

    meta = {
        "source": config.SOURCE,
        "n_frames": n,
        "fps": config.FPS,
        "pitch": [length, width],
        "halves": [[int(a), int(b)] for a, b in halves],
        "home_name": xml_meta["home_name"],
        "away_name": xml_meta["away_name"],
        "orientation": "H attacks towards x = L in every half",
        "flipped_halves": sorted({int(i) for i in np.unique(half_idx[flip])}),
        "speed": f"from positions, half window {config.SPEED_HALF_WINDOW_S} s, "
                 f"NaN above {config.SPEED_MAX_MPS} m/s",
        "inputs_sha256": {config.ASSETS[k].name: _sha256(config.ASSETS[k]) for k in ("atd_meta", "atd_raw")},
    }
    arrays = {"ball_xy": ball_xy, "players_xy": players_xy, "players_speed": speed,
              "players_team": slot_team}
    return save(config.GAME, arrays, meta)


# =================================================================================================
# summary (aggregates only: counts and shares, no frame-level values)
# =================================================================================================
def summary(d: dict) -> str:
    meta = d["meta"]
    team = d["players_team"]
    ball = np.asarray(d["ball_xy"])
    pxy = np.asarray(d["players_xy"])
    sp = np.asarray(d["players_speed"])
    n = meta["n_frames"]
    graded = np.zeros(n, bool)
    for a, b in meta["halves"]:
        graded[a:b + 1] = True
    vis = np.isfinite(ball[:, 0])
    tracked = np.isfinite(pxy[..., 0])
    lines = [
        f"game            {meta['game']}  ({meta['home_name']} = H, {meta['away_name']} = A)",
        f"frames          {n} at {meta['fps']} fps, {int(graded.sum())} graded in {len(meta['halves'])} halves",
        f"slots           {len(team)}: {int((team == 'H').sum())} H, {int((team == 'A').sum())} A",
        f"ball visible    {vis.mean():.1%} of all frames, {vis[graded].mean():.1%} of graded",
        f"objects/frame   median {np.median(tracked[graded].sum(1)):.0f} on graded frames",
    ]
    s = sp[graded][np.isfinite(sp[graded])]
    lines.append("speed (m/s)     p50 {:.2f}  p99 {:.2f}  max {:.1f}   (tracked graded cells with a speed: {:.1%})".format(
        np.median(s), np.percentile(s, 99), s.max(), np.isfinite(sp[graded][tracked[graded]]).mean()))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN frames -> NaN mean, expected
        hx = np.nanmean(np.where(team == "H", pxy[..., 0], np.nan), axis=1)
        ax = np.nanmean(np.where(team == "A", pxy[..., 0], np.nan), axis=1)
    for i, (a, b) in enumerate(meta["halves"]):
        ok = np.isfinite(hx[a:b + 1]) & np.isfinite(ax[a:b + 1])
        lines.append(f"half {i + 1}          H mean x < A mean x on {np.mean(hx[a:b + 1][ok] < ax[a:b + 1][ok]):.0%} "
                     f"of frames (H defends x = 0 in every half, so this should be alike in all halves)")
    lines.append(f"cache           {config.game_cache(meta['game'])}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args(argv)
    build_target(force=args.force)
    print(summary(load(config.GAME)))


if __name__ == "__main__":
    main()
