#!/usr/bin/env bash
# One command for the minimap own-ship pipeline:
#
#   position stream ─► alignment ─► minimap projection ─► dataset ─► model ─► check
#
# Every stage is measured rather than assumed: the projection is fitted against
# replay truth, the dataset's labels are replay truth projected through it, and
# the exported model is re-checked against its own manifest at the end.
#
# Usage:
#   run_minimap_pipeline.sh RECORDING_DIR REPLAY_FILE [options]
#
# Options:
#   --events PATH        event stream (default replays/parsed/events.jsonl)
#   --position PATH      own-position stream (default replays/parsed/position-<id>.jsonl)
#   --step-s S           dataset sampling step, default 1.0
#   --epochs N           training epochs, default 40
#   --block-s S          split block length, default 12.0
#   --python PATH        interpreter holding numpy and torch, default .venv/bin/python
#   --force              rebuild streams and the projection, not the dataset
#
# Example:
#   tools/replay/run_minimap_pipeline.sh recordings/236066_random_Kurama_XI-fullrun01 \
#       replays/236066_random_Kurama_XI.wowsreplay
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 RECORDING_DIR REPLAY_FILE [options]" >&2
  exit 2
fi

RECORDING="$1"
REPLAY="$2"
shift 2

TOOLS=$(cd "$(dirname "$0")" && pwd)
WORKSPACE=$(cd "$TOOLS/../.." && pwd)
PYTHON="$WORKSPACE/.venv/bin/python"
EVENTS="$WORKSPACE/replays/parsed/events.jsonl"
POSITION=""
POSITION_SET=0
STEP_S=1.0
EPOCHS=40
BLOCK_S=12.0
FORCE=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --events) EVENTS="$2"; shift 2 ;;
    --position) POSITION="$2"; POSITION_SET=1; shift 2 ;;
    --step-s) STEP_S="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --block-s) BLOCK_S="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

SESSION=$(basename "$REPLAY")
ID="${SESSION%.wowsreplay}"
RECORDING_NAME=$(basename "$RECORDING")
PARSED="$WORKSPACE/replays/parsed"
[ "$POSITION_SET" -eq 0 ] && POSITION="$PARSED/position-$ID.jsonl"

HUD_CAL="$PARSED/calibration/hud-2560x1600.json"
PROJECTION="$PARSED/calibration/minimap-2560x1600.json"
DATASET="$PARSED/datasets/minimap-$ID.npz"
MODEL="$PARSED/artifacts/minimap-localizer-$ID.pt"
MANIFEST="$PARSED/artifacts/minimap-localizer-$ID.json"
ANCHORS="$PARSED/alignment/$RECORDING_NAME.anchors.csv"
ALIGNMENT="$PARSED/alignment/$RECORDING_NAME.json"

if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON; create it with:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install numpy torch lxml packaging pycryptodome" >&2
  exit 1
fi
"$PYTHON" -c "import numpy, torch" 2>/dev/null || {
  echo "$PYTHON lacks numpy or torch; install them before running this pipeline" >&2
  exit 1
}
for path in "$RECORDING/session.json" "$RECORDING/capture.mkv" "$HUD_CAL" "$REPLAY"; do
  [ -e "$path" ] || { echo "missing input: $path" >&2; exit 1; }
done

echo "=== 1/6 own-position stream"
if [ -s "$POSITION" ] && [ "$FORCE" -eq 0 ]; then
  echo "reusing $POSITION ($(grep -c '"event": "position"' "$POSITION") samples)"
else
  "$PYTHON" "$TOOLS/export_events.py" "$REPLAY" --kinds position --out "$POSITION"
fi

echo "=== 2/6 alignment"
if [ -s "$ALIGNMENT" ] && [ "$FORCE" -eq 0 ]; then
  echo "reusing $ALIGNMENT"
else
  mkdir -p "$PARSED/alignment"
  "$PYTHON" "$TOOLS/anchors_from_pylog.py" \
    --pylog "$WORKSPACE/profile/python.log" --recording "$RECORDING" \
    --events "$EVENTS" --session "$SESSION" \
    --out "$ANCHORS" --report "${ANCHORS%.csv}.report.json"
  "$PYTHON" "$TOOLS/align_timeline.py" "$ANCHORS" --out "$ALIGNMENT"
fi

echo "=== 3/6 minimap projection (fitted against replay truth)"
if [ -s "$PROJECTION" ] && [ "$FORCE" -eq 0 ]; then
  echo "reusing $PROJECTION"
else
  "$PYTHON" "$TOOLS/calibrate_minimap.py" fit \
    --video "$RECORDING/capture.mkv" --hud-calibration "$HUD_CAL" \
    --position "$POSITION" --alignment "$ALIGNMENT" \
    --events "$EVENTS" --session "$SESSION" --frames 40 --settle-s 30 \
    --out "$PROJECTION" --samples-out "$PARSED/calibration/minimap-samples.jsonl"
fi

echo "=== 4/6 dataset"
mkdir -p "$PARSED/datasets"
"$PYTHON" "$TOOLS/build_minimap_dataset.py" \
  --video "$RECORDING/capture.mkv" --hud-calibration "$HUD_CAL" \
  --minimap-calibration "$PROJECTION" --position "$POSITION" --alignment "$ALIGNMENT" \
  --events "$EVENTS" --session "$SESSION" --step-s "$STEP_S" --block-s "$BLOCK_S" \
  --out "$DATASET" --labels-out "${DATASET%.npz}.labels.jsonl" \
  --summary "${DATASET%.npz}.summary.json"

echo "=== 5/6 train"
mkdir -p "$PARSED/artifacts"
"$PYTHON" "$TOOLS/train_minimap.py" --dataset "$DATASET" \
  --model-out "$MODEL" --manifest-out "$MANIFEST" --epochs "$EPOCHS"

echo "=== 5b/6 inventory: every marker class, terrain, ring-gated check"
INVENTORY="$PARSED/inventory/$RECORDING_NAME.inventory.jsonl"
mkdir -p "$PARSED/inventory"
./.venv/bin/python --version >/dev/null 2>&1 || true
"$PYTHON" "$TOOLS/export_events.py" "$REPLAY" \
  --kinds position,track,spotted,zone,visibility,death --out "$PARSED/minimap-$ID.jsonl" >/dev/null
"$PYTHON" "$TOOLS/build_minimap_inventory.py" \
  --export "$PARSED/minimap-$ID.jsonl" --alignment "$ALIGNMENT" \
  --hud-calibration "$HUD_CAL" --minimap-calibration "$PROJECTION" \
  --zones "$PARSED/calibration/minimap-zones-41_Conquest.json" \
  --dataset "$DATASET" --out "$INVENTORY" \
  --summary "$PARSED/inventory/$RECORDING_NAME.inventory.summary.json" \
  --terrain-out "$PARSED/inventory/$RECORDING_NAME.terrain.png" \
  --overlay-out "$PARSED/inventory/$RECORDING_NAME.overlay-verify.png" --check

echo "=== 5c/6 multi-class heatmap dataset and model"
HEATMAP="$PARSED/datasets/heatmap-$ID.npz"
"$PYTHON" "$TOOLS/build_heatmap_dataset.py" --inventory "$INVENTORY" --dataset "$DATASET" \
  --out "$HEATMAP" --summary "${HEATMAP%.npz}.summary.json" --grid 176 --sigma 1.0
"$PYTHON" "$TOOLS/train_minimap_heatmap.py" --heatmaps "$HEATMAP" --images "$DATASET" \
  --model-out "$PARSED/artifacts/minimap-heatmap-unet-$ID.pt" \
  --manifest-out "$PARSED/artifacts/minimap-heatmap-unet-$ID.json" --epochs "$EPOCHS"

echo "=== 6/6 check the artifacts against their own manifests"
"$PYTHON" "$TOOLS/train_minimap.py" --validate --model "$MODEL" --manifest "$MANIFEST"
"$PYTHON" "$TOOLS/train_minimap.py" --preview --model "$MODEL" --manifest "$MANIFEST" \
  --preview-out "${DATASET%.npz}.preview-test.png" --preview-count 8
"$PYTHON" "$TOOLS/train_minimap_heatmap.py" --validate \
  --model "$PARSED/artifacts/minimap-heatmap-unet-$ID.pt" \
  --manifest "$PARSED/artifacts/minimap-heatmap-unet-$ID.json"
"$PYTHON" "$TOOLS/train_minimap_heatmap.py" --preview \
  --model "$PARSED/artifacts/minimap-heatmap-unet-$ID.pt" \
  --manifest "$PARSED/artifacts/minimap-heatmap-unet-$ID.json" \
  --preview-out "$PARSED/artifacts/minimap-heatmap-unet-$ID.preview-test.png" --preview-count 4

echo
echo "artifacts:"
echo "  position   $POSITION"
echo "  projection $PROJECTION"
echo "  dataset    $DATASET"
echo "  model      $MODEL"
echo "  manifest   $MANIFEST"
echo "  preview    ${DATASET%.npz}.preview-test.png"
