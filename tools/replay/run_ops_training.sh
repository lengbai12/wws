#!/usr/bin/env bash
# Train the Operations minimap model and validate the result.
#
# Failure must not be able to look like success. This script used to run both steps with
# `set -uo pipefail` only - no `-e`, no exit-code checks - so a failed training still ran
# the validation, a failed validation still printed "=== done", and the script returned 0.
# It had no record of what it produced either, so a stale model could be validated as if
# this run had written it.
#
# Now: the whole body runs inside the shared heavy lock; every step is checked; the run is
# recorded under state/runs/; success is claimed only by `run_record commit`, which
# refuses unless the artifacts exist and were rewritten by this run; and `=== done ` is
# printed only after that commit. The collector keys on that line, so nothing downstream
# can act on a run that did not succeed.
#
# Usage: run_ops_training.sh [EPOCHS]
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

# Take the shared lock by re-executing this same script once, and only once.
if [ -z "${HEAVY_LOCK_HELD:-}" ]; then
  export HEAVY_LOCK_HELD=1
  exec tools/replay/heavy_lock.sh run train -- "$0" "$@"
fi

EPOCHS=${1:-30}
H=${H:-replays/parsed/datasets/heatmap-merged-14s-g176.npz}
I=${I:-replays/parsed/datasets/minimap-merged-14s-g176.npz}
M=${M:-replays/parsed/artifacts/minimap-heatmap-ops-14s.pt}
J=${J:-replays/parsed/artifacts/minimap-heatmap-ops-14s.json}
STATE=${STATE:-replays/parsed/artifacts/minimap-heatmap-ops-14s.state_dict.pt}

PY=./.venv/bin/python
echo "=== train $(date '+%H:%M:%S') epochs=$EPOCHS (unbuffered; the best epoch is written as it improves) ==="
run_id=$("$PY" tools/replay/run_record.py start train --artifact "$M" --artifact "$J")
echo "run $run_id"

if ! "$PY" -u tools/replay/peak_rss.py "$PY" -u tools/replay/train_minimap_heatmap.py \
     --heatmaps "$H" --images "$I" --model-out "$M" --manifest-out "$J" --epochs "$EPOCHS"; then
  "$PY" tools/replay/run_record.py fail "$run_id" --reason "training exited non-zero" || true
  echo "=== TRAINING FAILED $(date '+%H:%M:%S') ===" >&2
  exit 1
fi

echo "=== validate $(date '+%H:%M:%S') ==="
if ! "$PY" -u tools/replay/train_minimap_heatmap.py --validate --model "$M" --manifest "$J"; then
  "$PY" tools/replay/run_record.py fail "$run_id" --reason "validate exited non-zero" || true
  echo "=== VALIDATE FAILED $(date '+%H:%M:%S') ===" >&2
  exit 1
fi

if ! "$PY" tools/replay/run_record.py commit "$run_id" \
     --artifact "$M" --artifact "$J" --artifact "$STATE"; then
  echo "=== COMMIT FAILED $(date '+%H:%M:%S') ===" >&2
  exit 1
fi
echo "=== done $(date '+%H:%M:%S') run $run_id ==="
