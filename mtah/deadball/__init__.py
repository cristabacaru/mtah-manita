"""Dead-ball module. Spec: docs/deadball_SPEC.md.

P3a (Cristiana), p_dead from tracking only:
    util.py      shared helpers: input check, runs of True, per-half loop, file writing
    signals.py   ball-gap and motion features                          (planned)
    lines.py     ball past / near the lines                            (planned)
    combine.py   features -> p_dead.npy; today the v0 placeholder
    evaluate.py  alive_dead_f1, D-interval F1, boundary offset, ECE     (planned)

Tests (synthetic, no match data) live next to the team's in tests/deadball/:
    python tests/deadball/run_all.py

P3b (Alexia): tags.py, restarts.py (planned).

Every function takes one game in the shared schema (guide section 5), as returned by
mtah.data.load(game), and works on any game: fps, pitch and halves come from game["meta"].
"""
