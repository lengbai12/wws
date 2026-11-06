#!/usr/bin/env bash
# Collect the rebuildable per-session datasets - but only after a run that verifiably
# succeeded.
#
# This used to treat "the training process is gone" as "the training finished": an OOM or
# a crash took the same path as a clean finish, straight into `rm -rf`. It never looked at
# whether anything had succeeded, and it had no way to tell a partial run from a complete
# one.
#
# Now it reads the run record written by run_ops_training.sh and acts only when that run
# is committed `ok` with verified artifact hashes. Anything else - running, failed, no
# record at all - means it does nothing and says why. And deletion requires --delete:
# without it this is a dry run that only prints the list.
#
# Usage: collect_after_training.sh [--delete] [--run <run_id>]
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

REPORT=state/ops-collect.log
MANIFEST=state/deleted-derived-latest.txt
DATASETS=replays/parsed/datasets
PY=./.venv/bin/python
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$REPORT"; }

DELETE=0
RUN_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --delete) DELETE=1 ;;
    --run) RUN_ID="${2:?--run needs an id}"; shift ;;
    *) echo "usage: collect_after_training.sh [--delete] [--run <run_id>]" >&2; exit 2 ;;
  esac
  shift
done

# The run whose success this cleanup depends on: the given one, or the newest train run.
if [ -z "$RUN_ID" ]; then
  RUN_ID=$(ls -t state/runs/train-*.json 2>/dev/null | head -1 | xargs -r -n1 basename | sed 's/\.json$//')
fi
if [ -z "$RUN_ID" ]; then
  say "no train run record found; nothing to collect (a run that left no record did not succeed)"
  exit 0
fi

STATUS=$("$PY" tools/replay/run_record.py status "$RUN_ID" || true)
say "run $RUN_ID status=$STATUS"
if [ "$STATUS" != "ok" ]; then
  say "refusing to collect: run $RUN_ID is not a committed success (status=$STATUS)"
  exit 0
fi

if pgrep -x WorldOfWarships >/dev/null 2>&1; then
  say "refusing to collect: the game is running"
  exit 0
fi
if "$PY" tools/replay/run_record.py status "$RUN_ID" >/dev/null 2>&1; then
  :
else
  say "refusing to collect: run $RUN_ID no longer verifies"
  exit 0
fi

BEFORE=$(df --output=avail -BG /home/lengbai | tail -1 | tr -dc '0-9')
{
  echo "collect attempt at $(date '+%F %T')  run=$RUN_ID  delete=$DELETE"
  echo "rule: only derived, rebuildable data; the merged corpus is load-bearing"
  echo
} > "$MANIFEST"

collect() {
  local target="$1" reason="$2"
  [ -e "$target" ] || return 0
  local size
  size=$(du -sh "$target" 2>/dev/null | cut -f1)
  if [ "$DELETE" = "1" ]; then
    echo "$(printf '%-64s' "$target") $(printf '%8s' "$size")   $reason" >> "$MANIFEST"
    rm -rf -- "$target"
    say "  deleted  $target $size"
  else
    echo "WOULD DELETE $(printf '%-58s' "$target") $(printf '%8s' "$size")   $reason" >> "$MANIFEST"
    say "  would delete  $target $size"
  fi
}

for f in "$DATASETS"/minimap-*.npz "$DATASETS"/heatmap-*-g176-s1.0-c10.npz; do
  case "$f" in *merged*) continue ;; esac
  collect "$f" "per-session dataset (rebuildable from recordings/)"
done
for f in replays/incoming/*.dup-of-our-copy*; do
  collect "$f" "byte-identical duplicate of a replay already in the pool"
done

AFTER=$(df --output=avail -BG /home/lengbai | tail -1 | tr -dc '0-9')
say "free before ${BEFORE} GB, after ${AFTER} GB (a dry run frees nothing; df is the only honest measure)"
[ "$DELETE" = "1" ] || say "dry run: nothing was deleted. Pass --delete to act."
