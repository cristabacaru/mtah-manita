#!/usr/bin/env bash
# One command: raw match data -> Manita.csv (README "Before submitting").
#
#   bash run.sh            run every stage in order; stop at the first one that fails or is not written
#   bash run.sh --check    show each stage, its owner, whether it is written, and its outputs; run nothing
#   bash run.sh --from p_home   start at that stage (earlier outputs must already be in the cache)
#   bash run.sh --fresh    also rebuild the tracking cache (otherwise data is built only if missing)
#
# Stage contract (README module contract): each stage is `python -m <module> [--game <game>]`, reads
# from and writes to $MTAH_CACHE/<game>/, and is deterministic. A stage counts as "written" when its
# file exists and has an `if __name__ == "__main__":` block. After each stage, run.sh checks that
# every output it promises exists and was (re)written by that run.
#
# Logs go to $MTAH_CACHE/logs/, outside the repo. Plain bash 3.2, so it runs on macOS too.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# -- arguments ------------------------------------------------------------------------------------
CHECK=0; FRESH=0; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    --fresh) FRESH=1 ;;
    --from)  FROM="${2:?--from needs a stage name}"; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

# -- environment ----------------------------------------------------------------------------------
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  echo "no .venv found: create it first (README 'Setup')" >&2; exit 1
fi
"$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' \
  || { echo "the .venv is not Python 3.12 ($("$PY" --version)); see README 'Setup'" >&2; exit 1; }

export MTAH_CACHE="${MTAH_CACHE:-$HOME/.cache/mtah2027}"
export PYTHONHASHSEED=0                    # determinism: same input -> same sha256 (guide section 2)

GAME="$("$PY" -c 'from mtah import config; print(config.GAME)')"
CACHE="$("$PY" -c 'from mtah import config; print(config.game_cache())')"
SUBMISSION="$("$PY" -c 'from mtah import config; print(config.SUBMISSION_CSV)')"

if command -v sha256sum >/dev/null 2>&1; then SHA="sha256sum"; else SHA="shasum -a 256"; fi

# -- stages ---------------------------------------------------------------------------------------
# name | file | module | arguments | outputs (in the cache unless absolute) | owner
# Order follows the data flow: decode reads p_dead, p_home, tags and anchors; submit reads labels.
STAGES="
data|mtah/data.py|mtah.data||ball_xy.npy players_xy.npy players_speed.npy players_team.npy meta.json|P1 Alexia
p_dead|mtah/deadball/combine.py|mtah.deadball.combine|--game $GAME|p_dead.npy|P3a Cristiana
p_home|mtah/model.py|mtah.model|--game $GAME|p_home.npy p_turn.npy|P4 Bianca
tags|mtah/deadball/tags.py|mtah.deadball.tags|--game $GAME|tags.json|P3b Alexia
restarts|mtah/deadball/restarts.py|mtah.deadball.restarts|--game $GAME|anchors.json|P3b Alexia
labels|mtah/decode.py|mtah.decode|--game $GAME|labels.npy|P5 Andra
submit|mtah/submit.py|mtah.submit||$SUBMISSION|P1 Alexia
"

is_written() {  # file exists and can be run with python -m
  [ -f "$1" ] && grep -q '__name__ == "__main__"' "$1"
}

output_path() {
  case "$1" in /*) echo "$1" ;; *) echo "$CACHE/$1" ;; esac
}

if [ -n "$FROM" ] && ! echo "$STAGES" | grep -q "^$FROM|"; then
  echo "unknown stage '$FROM'; stages: $(echo "$STAGES" | grep '|' | cut -d'|' -f1 | tr '\n' ' ')" >&2
  exit 2
fi

# -- --check: status only -------------------------------------------------------------------------
if [ "$CHECK" = 1 ]; then
  printf "%-9s %-15s %-12s %s\n" "stage" "owner" "status" "outputs in $CACHE"
  echo "$STAGES" | grep '|' | while IFS='|' read -r name file module args outputs owner; do
    if is_written "$file"; then status="written"
    elif [ -f "$file" ]; then status="no CLI yet"
    else status="not written"; fi
    present=""
    for out in $outputs; do
      if [ -f "$(output_path "$out")" ]; then present="$present ok:$(basename "$out")"
      else present="$present missing:$(basename "$out")"; fi
    done
    printf "%-9s %-15s %-12s%s\n" "$name" "$owner" "$status" "$present"
  done
  exit 0
fi

# -- run ------------------------------------------------------------------------------------------
missing_data="$("$PY" -c 'from mtah import config; print(" ".join(map(str, config.missing_assets(("atd_meta", "atd_raw")))))')"
if [ -n "$missing_data" ]; then
  echo "match data not found: $missing_data (see data/README.md)" >&2; exit 1
fi

mkdir -p "$MTAH_CACHE/logs"
LOG="$MTAH_CACHE/logs/run_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee "$LOG") 2>&1
echo "run.sh  game $GAME  cache $CACHE  commit $(git rev-parse --short HEAD 2>/dev/null || echo '?')$(git diff --quiet 2>/dev/null || echo ' (uncommitted changes)')"

STAMP="$(mktemp)"
trap 'rm -f "$STAMP"' EXIT
started=0
[ -z "$FROM" ] && started=1

echo "$STAGES" | grep '|' | while IFS='|' read -r name file module args outputs owner; do
  [ "$name" = "$FROM" ] && started=1
  [ "$started" = 1 ] || { echo "-- $name: skipped (--from $FROM)"; continue; }

  if ! is_written "$file"; then
    echo
    echo "STOP at stage '$name' ($owner): $file is not written yet (or has no __main__ block)."
    echo "Stages after it did not run. 'bash run.sh --check' shows what is left."
    exit 3
  fi

  stage_args="$args"
  if [ "$name" = "data" ] && [ "$FRESH" = 1 ]; then stage_args="--force"; fi

  echo
  echo "== $name ($owner): python -m $module $stage_args"
  touch "$STAMP"
  sleep 1                                   # so "written by this run" survives 1-second mtimes
  t0=$(date +%s)
  # shellcheck disable=SC2086  # stage_args is a word list on purpose
  "$PY" -m "$module" $stage_args
  echo "-- $name done in $(( $(date +%s) - t0 )) s"

  for out in $outputs; do
    path="$(output_path "$out")"
    if [ ! -f "$path" ]; then
      echo "STOP: stage '$name' did not write $path" >&2; exit 4
    fi
    # data is reused from the cache unless --fresh, so its outputs may be older than this run
    if [ "$name" != "data" ] || [ "$FRESH" = 1 ]; then
      if [ ! "$path" -nt "$STAMP" ]; then
        echo "STOP: $path was not rewritten by stage '$name' (stale output)" >&2; exit 4
      fi
    fi
  done
done

echo
echo "Done: $SUBMISSION"
$SHA "$SUBMISSION"
echo "log: $LOG"
