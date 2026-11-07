# Verification checks

Exploratory checks on the BAR–ATH data. Each one prints aggregates only (counts, shares,
percentiles): no frame-level values, labels or positions. None of them reads the ground truth.

    export MTAH_CACHE=/some/scratch/dir      # parsed-ATD cache, kept outside the repo
    .venv/bin/python research/checks/c1_ball_gaps.py

| script | idea |
|---|---|
| `common.py` | raw EPTS parser (same format handling as `starter/baseline.py`), shared helpers |
| `c1_ball_gaps.py` | 1. structure of missing-ball gaps |
| `c2_x_variant.py` | 2. nearest-player baseline with X on ball-missing frames: possession counts only |
| `c3_dead_ball.py` | 3. dead-ball signals with no ball (player speed, camera, gaps, Smart Tagging) |
| `c4_smart_tagging.py` | 4. Smart Tagging as a prior |
| `c5_restarts.py` | 5. kick-off detector and restart templates |
| `c6_hygiene.py` | 6. team slots, ends, duplicates, identity jumps |
| `c7_broadcast.py` | 7. broadcast-to-tactical alignment without video, and broadcast ball fill rate |

The first run parses both ATD text files into `$MTAH_CACHE` (defaults to `~/.cache/mtah2027`).
Never point the cache inside `data/` or the repo.
