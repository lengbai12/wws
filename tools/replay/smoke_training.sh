#!/usr/bin/env bash
# One epoch plus a validate, to prove the whole path end to end without committing an
# hour to it. This is the check that caught two real bugs on the day the device support
# was added: the sweep path passed no device while the model was already on CUDA, and
# the measurement tail built a split-sized tensor and OOM-killed the run.
#
# Heavier than it looks: it trains a full epoch and runs the measurement tail, so do not
# run it beside a training on this machine.
#
# Usage: smoke_training.sh [EPOCHS] [EXTRA_ARGS...]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

EPOCHS=${1:-1}
shift || true
H=replays/parsed/datasets/heatmap-merged-14s-g176.npz
I=replays/parsed/datasets/minimap-merged-14s-g176.npz
MODEL=${MODEL:-/tmp/smoke-training.pt}
MANIFEST=${MANIFEST:-/tmp/smoke-training.json}

echo "=== smoke train $(date '+%H:%M:%S') epochs=$EPOCHS ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
./.venv/bin/python -u tools/replay/peak_rss.py ./.venv/bin/python -u \
  tools/replay/train_minimap_heatmap.py \
  --heatmaps "$H" --images "$I" --model-out "$MODEL" --manifest-out "$MANIFEST" \
  --epochs "$EPOCHS" --eval-every 1 "$@"
echo "=== smoke validate $(date '+%H:%M:%S') ==="
./.venv/bin/python -u tools/replay/train_minimap_heatmap.py \
  --validate --model "$MODEL" --manifest "$MANIFEST"
echo "=== smoke done $(date '+%H:%M:%S') ==="
