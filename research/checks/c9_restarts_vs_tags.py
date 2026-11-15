"""Idea 9 -- restart anchors (mtah/deadball/restarts.py) against Smart Tagging set pieces.

Behind the option B decision (2026-11-08): which anchor templates are worth keeping. Smart Tagging is
automatic and incomplete (59 set pieces; a match has about 110 restarts, most free kicks are not
tagged), so it is a sanity check, not an answer key: a template that lands near tagged set pieces no
more often than a random graded frame carries no evidence. Prints aggregates only.

    python research/checks/c9_restarts_vs_tags.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import load as load_tags  # noqa: E402
from mtah import config, data  # noqa: E402
from mtah.deadball import restarts  # noqa: E402

MARGIN_S = 5
game = data.load()
n, fps = game["meta"]["n_frames"], game["meta"]["fps"]
margin = MARGIN_S * fps
set_pieces = [e for e in load_tags() if e["code"].strip().upper() == "SET PIECES"]

tagged = np.zeros(n, bool)
for e in set_pieces:
    tagged[max(0, e["start_frame"] - margin):e["end_frame"] + margin] = True
graded = np.zeros(n, bool)
for a, b in game["meta"]["halves"]:
    graded[a:b + 1] = True
chance = tagged[graded].mean()


def table(anchors, title):
    print(f"\n== {title}: {len(anchors)} anchors ==")
    print(f"{'source':<18}{'type':<11}{'n':>4} {'near a tagged set piece':>24}   (chance {chance:.0%})")
    for src in ("gap_end", "still_ball", "kickoff_formation"):
        for kind in restarts.TYPES:
            sel = [a["frame"] for a in anchors if a["source"] == src and a["type"] == kind]
            if sel:
                print(f"{src:<18}{kind:<11}{len(sel):>4} {tagged[sel].mean():>23.0%}")


print(f"set pieces tagged: {len(set_pieces)}; graded frames within +/-{MARGIN_S} s of one: {chance:.1%}")
table(restarts.detect(game, drop_weak=False), "all candidates (drop_weak=False)")
final = restarts.detect(game)
table(final, "final anchors (v2)")

near = lambda e: [a for a in final if e["start_frame"] - margin <= a["frame"] < e["end_frame"] + margin]
print(f"\ntagged set pieces with a final anchor nearby: {sum(bool(near(e)) for e in set_pieces)} / {len(set_pieces)}")
want = {"Throw-in": "throw_in", "Corner Kick": "corner", "Goal Kick": "goal_kick",
        "Kick Off (Start)": "kickoff", "Kick Off (After Goal)": "kickoff"}
for tag_type, kind in want.items():
    sel = [e for e in set_pieces if e["tags"].get("Type") == tag_type]
    found = sum(any(a["type"] == kind for a in near(e)) for e in sel)
    print(f"  {tag_type:<22} {found:>2} / {len(sel):<2} have an anchor of the same type nearby")
pairs = [(e, a) for e in set_pieces if e["tags"].get("Team") not in (None, "N/A")
         for a in near(e)[:1] if a["type"] == want.get(e["tags"].get("Type"))]
ok = sum(a["team"] == ("H" if e["tags"]["Team"] == game["meta"]["home_name"] else "A") for e, a in pairs)
print(f"restarting team agrees with the tag: {ok} / {len(pairs)} (tags with a team and a same-type anchor)")
