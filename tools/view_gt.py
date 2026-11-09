"""Play the tactical video with the ground truth -- and optionally a submission -- drawn on top.

The public ground truth is 7500 frames of letters. This puts each letter on the frame it labels,
next to the tracked players and ball projected through the homography, so you can watch what the
ground truth calls dead, when it hands possession over, and what your own submission got wrong.

    pip install opencv-python
    python tools/view_gt.py                                   # the public 5 minutes
    python tools/view_gt.py --pred evaluation/results/baseline_smooth50.csv
    python tools/view_gt.py --start 113000 --end 115000 --gt none   # any stretch, no labels
    python tools/view_gt.py --list                            # GT segments as video timestamps

Keys (click in the video window first):

    space   play / pause                 a / d   -1 / +1 frame
    s / w   -1 s / +1 s                  z / c   -10 s / +10 s
    p / n   previous / next GT change    e       next frame where --pred disagrees with GT
    1 2 3 4 speed 0.25x 0.5x 1x 2x       t       tracking overlay on / off
    q, Esc  quit                         click the timeline bar to jump

Everything runs locally. Nothing here prints match data beyond frame numbers and labels.

Seeking in an mp4 through OpenCV is not guaranteed frame-exact on every build. Check it once: the
first-half kickoff is the first frame of games.half_video_blocks(), and the ball should be
struck on that frame. Playing
forward reads frames sequentially and is always exact.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, asset_paths, half_of_video_frame, half_video_blocks, require  # noqa: E402
import possession_metrics as pm  # noqa: E402
from baseline import team_of_each_track, HOME  # noqa: E402
from load_homography import load as load_homography, project  # noqa: E402

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is not installed:\n    pip install opencv-python")

# Params
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
WINDOW = "possession viewer"
DISPLAY_WIDTH = 1280
FPS = GAMES[GAME]["fps"]
SPEEDS = {ord("1"): 0.25, ord("2"): 0.5, ord("3"): 1.0, ord("4"): 2.0}

# BGR. Home blue, away red, dead grey, no answer black -- the same everywhere on screen.
COLOURS = {"H": (220, 120, 30), "A": (40, 40, 220), "D": (150, 150, 150), "X": (20, 20, 20)}
NAMES = {"H": "HOME (Barcelona)", "A": "AWAY (Athletic)", "D": "DEAD BALL", "X": "no label"}
BAR_HEIGHT = 16

FIELD_SEP, GROUP_SEP = ":", ";"


def read_labels(path, n_frames):
    """Upper-cased label array, or all X when there is no file to read."""
    if path is None:
        return np.full(n_frames, "X", dtype=object)
    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, n_frames)
    if reason:
        sys.exit("{}: {}".format(path, reason))
    return np.array([str(v).strip().upper() for v in table["label"].values], dtype=object)


def video_file():
    """The tactical video. Falls back to any BAR-ATH_tactical*.mp4, since downloads get renamed."""
    expected = asset_paths()["video"]
    if os.path.isfile(expected):
        return expected
    found = sorted(glob.glob(os.path.join(os.path.dirname(expected), "*tactical*.mp4")))
    if not found:
        sys.exit("no tactical video in {}".format(os.path.dirname(expected)))
    print("using {} (games.py expects {})".format(
        os.path.basename(found[0]), os.path.basename(expected)))
    return found[0]


def read_tracking(start, end):
    """(players, ball) for video frames start..end inclusive, normalised 0-1, NaN where missing.

    players is (n, slots, 2), ball is (n, 2). Streams the raw ATD and stops once past `end`, so a
    first-half window does not read the whole file.
    """
    raw_path = require(["atd_raw"])[0]
    n = end - start + 1
    players, ball = None, np.full((n, 2), np.nan)
    with open(raw_path) as handle:
        for line in handle:
            frame = int(line[:line.index(FIELD_SEP)]) - 1  # frame_id counts from 1
            if frame < start:
                continue
            if frame > end:
                break
            parts = line.rstrip().split(FIELD_SEP)
            groups = parts[1].split(GROUP_SEP)
            if players is None:
                players = np.full((n, len(groups), 2), np.nan)
            row = frame - start
            for slot, group in enumerate(groups):
                values = group.split(",")
                players[row, slot] = float(values[0]), float(values[1])
            values = parts[2].split(",")
            ball[row] = float(values[0]), float(values[1])
    return players, ball


def label_strip(labels, start, end, width):
    """A width x BAR_HEIGHT image of labels[start..end], one sampled frame per pixel column."""
    columns = np.linspace(start, end, width).round().astype(int)
    strip = np.zeros((BAR_HEIGHT, width, 3), dtype=np.uint8)
    for x, frame in enumerate(columns):
        strip[:, x] = COLOURS[labels[frame]]
    return strip


def timestamp(frame):
    """Video position as m:ss.ff -- what a video player shows when you seek to it."""
    seconds, sub = divmod(frame, FPS)
    return "{}:{:02d}.{:02d}".format(seconds // 60, seconds % 60, sub)


def print_segments(gt, start, end):
    """GT segments inside the window, as frame ranges and video timestamps."""
    print("{:>7} {:>7}  {:>9} {:>9}  {:>6}  label".format(
        "first", "last", "from", "to", "dur s"))
    breaks = np.flatnonzero(gt[start + 1:end + 1] != gt[start:end]) + start + 1
    bounds = [start] + breaks.tolist() + [end + 1]
    for first, stop in zip(bounds[:-1], bounds[1:]):
        print("{:>7} {:>7}  {:>9} {:>9}  {:>6.1f}  {} {}".format(
            first, stop - 1, timestamp(first), timestamp(stop - 1),
            (stop - first) / float(FPS), gt[first], NAMES[gt[first]]))


def draw_banner(image, y, title, label, scale):
    """One coloured label box with its caption."""
    cv2.rectangle(image, (10, y), (int(560 * scale), y + int(48 * scale)), COLOURS[label], -1)
    cv2.putText(image, "{}: {}  {}".format(title, label, NAMES[label]),
                (20, y + int(34 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale,
                (255, 255, 255), 2, cv2.LINE_AA)


def render(image, frame, ctx):
    """The frame with tracking, labels and timeline drawn on, resized for display."""
    image = image.copy()
    row = frame - ctx["start"]

    if ctx["show_tracking"] and ctx["players"] is not None and ctx["valid"][frame]:
        matrix = ctx["homography"][frame]
        xs, ys = project(matrix, ctx["players"][row, :, 0], ctx["players"][row, :, 1])
        for slot, (x, y) in enumerate(zip(xs, ys)):
            if np.isfinite(x) and np.isfinite(y):
                colour = COLOURS["H"] if ctx["teams"][slot] == HOME else COLOURS["A"]
                cv2.circle(image, (int(x), int(y)), 11, colour, 3, cv2.LINE_AA)

    scale = image.shape[1] / 1920.0
    draw_banner(image, 10, "GT", ctx["gt"][frame], scale)
    if ctx["pred"] is not None:
        draw_banner(image, 10 + int(56 * scale), "PRED", ctx["pred"][frame], scale)

    ball_seen = ctx["players"] is not None and np.isfinite(ctx["ball"][row, 0])
    half = half_of_video_frame(GAME, frame)
    info = "frame {}   video {}   half {}   ball {}   speed {}x{}".format(
        frame, timestamp(frame), half or "-", "tracked" if ball_seen else "MISSING",
        ctx["speed"], "" if ctx["playing"] else "   [paused]")
    cv2.putText(image, info, (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale,
                (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(image, info, (20, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale,
                (0, 0, 0), 2, cv2.LINE_AA)

    height = int(round(image.shape[0] * DISPLAY_WIDTH / float(image.shape[1])))
    shown = cv2.resize(image, (DISPLAY_WIDTH, height), interpolation=cv2.INTER_AREA)

    bars = [ctx["gt_strip"]] + ([ctx["pred_strip"]] if ctx["pred"] is not None else [])
    timeline = np.vstack(bars).copy()
    cursor = int(round((frame - ctx["start"]) * (DISPLAY_WIDTH - 1)
                       / float(max(1, ctx["end"] - ctx["start"]))))
    cv2.line(timeline, (cursor, 0), (cursor, timeline.shape[0] - 1), (0, 255, 255), 2)
    return np.vstack([shown, timeline]), height


def next_index(candidates, frame, forward):
    """The first candidate after (or before) frame, or frame itself if there is none."""
    if forward:
        later = candidates[candidates > frame]
        return int(later[0]) if len(later) else frame
    earlier = candidates[candidates < frame]
    return int(earlier[-1]) if len(earlier) else frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gt", default=PUBLIC_GT,
                        help="ground-truth CSV, or 'none' (default: the public 5 minutes)")
    parser.add_argument("--pred", help="a submission CSV to show beside the ground truth")
    parser.add_argument("--start", type=int, help="first video frame (default: first GT frame)")
    parser.add_argument("--end", type=int, help="last video frame (default: last GT frame)")
    parser.add_argument("--video", help="video path, if not the one games.py expects")
    parser.add_argument("--no-tracking", action="store_true", help="skip the ATD overlay")
    parser.add_argument("--list", action="store_true", help="print GT segments and exit")
    args = parser.parse_args()

    n_frames = GAMES[GAME]["n_video_frames"]
    gt = read_labels(None if args.gt == "none" else args.gt, n_frames)
    pred = read_labels(args.pred, n_frames) if args.pred else None

    covered = np.flatnonzero(gt != "X")
    start = args.start if args.start is not None else (int(covered[0]) if len(covered) else half_video_blocks(GAME)[0][1])
    end = args.end if args.end is not None else (int(covered[-1]) if len(covered) else start + 7499)
    start, end = max(0, start), min(n_frames - 1, end)

    if args.list:
        print_segments(gt, start, end)
        sys.exit(0)

    ctx = {"gt": gt, "pred": pred, "start": start, "end": end, "players": None,
           "show_tracking": not args.no_tracking, "speed": 1.0, "playing": False,
           "gt_strip": label_strip(gt, start, end, DISPLAY_WIDTH),
           "pred_strip": label_strip(pred, start, end, DISPLAY_WIDTH) if pred is not None
           else None}
    if not args.no_tracking:
        print("reading tracking for frames {}..{} ...".format(start, end))
        ctx["players"], ctx["ball"] = read_tracking(start, end)
        ctx["teams"] = team_of_each_track(require(["atd_meta"])[0])
        ctx["homography"], _, ctx["valid"] = load_homography()

    changes = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    changes = changes[(changes > start) & (changes <= end)]
    disagreements = (np.flatnonzero((pred != gt) & (gt != "X")) if pred is not None
                     else np.array([], dtype=int))

    capture = cv2.VideoCapture(args.video or video_file())
    if not capture.isOpened():
        sys.exit("OpenCV could not open the video")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    jump = {"to": None}
    layout = {"video_height": 0}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and y >= layout["video_height"]:
            jump["to"] = start + int(round(x * (end - start) / float(DISPLAY_WIDTH - 1)))

    cv2.setMouseCallback(WINDOW, on_mouse)

    frame, decoded, image = start, None, None
    while True:
        if decoded is None or frame != decoded + 1:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)  # seek; sequential reads skip this
        if frame != decoded:
            ok, image = capture.read()
            if not ok:
                sys.exit("could not decode video frame {}".format(frame))
            decoded = frame

        shown, layout["video_height"] = render(image, frame, ctx)
        cv2.imshow(WINDOW, shown)
        delay = max(1, int(1000.0 / (FPS * ctx["speed"]))) if ctx["playing"] else 30
        key = cv2.waitKey(delay) & 0xFF

        target = frame
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            ctx["playing"] = not ctx["playing"]
        elif key in SPEEDS:
            ctx["speed"] = SPEEDS[key]
        elif key == ord("t"):
            ctx["show_tracking"] = not ctx["show_tracking"]
        elif key in (ord("a"), ord("d"), ord("s"), ord("w"), ord("z"), ord("c")):
            step = {"a": -1, "d": 1, "s": -FPS, "w": FPS, "z": -10 * FPS, "c": 10 * FPS}
            target = frame + step[chr(key)]
            ctx["playing"] = False
        elif key in (ord("n"), ord("p")):
            target = next_index(changes, frame, key == ord("n"))
            ctx["playing"] = False
        elif key == ord("e"):
            target = next_index(disagreements, frame, True)
            ctx["playing"] = False
        elif ctx["playing"]:
            target = frame + 1

        if jump["to"] is not None:
            target, jump["to"] = jump["to"], None
        frame = min(end, max(start, target))
        if frame == end:
            ctx["playing"] = False

    capture.release()
    cv2.destroyAllWindows()
