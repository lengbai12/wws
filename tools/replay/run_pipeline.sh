#!/usr/bin/env bash
# Run the label pipeline for one recording: anchors, alignment, labels, check.
#
#   replay playback log ──► anchors ──► alignment ──► labels ──► pixel check
#
# The HUD calibration is NOT produced here: choosing a HUD rectangle is a
# one-time human decision that must be verified against a frame with a known
# value, so this script only consumes it. Everything downstream of it is
# automatic and reproducible.
#
# Usage:
#   run_pipeline.sh RECORDING_DIR REPLAY_FILE \
#       --calibration PATH [--pylog PATH] [--step-s S] [--force]
#
# Example:
#   tools/replay/run_pipeline.sh recordings/236066_random_Kurama_XI-fullrun01 \
#       replays/236066_random_Kurama_XI.wowsreplay \
#       --calibration replays/parsed/calibration/hud-2560x1600.json
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 RECORDING_DIR REPLAY_FILE [--calibration PATH] [--pylog PATH] [--step-s S] [--force]" >&2
  exit 2
fi

RECORDING="$1"
REPLAY="$2"
shift 2

PYTHON="${WW_PYTHON:-/usr/bin/python3}"
VERIFY_LIMIT=40
VERIFY_TOLERANCE=0.03
STEP_S=0.5
CALIBRATION=""
PYLOG=""
FORCE=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --calibration) CALIBRATION="$2"; shift 2 ;;
    --pylog) PYLOG="$2"; shift 2 ;;
    --step-s) STEP_S="$2"; shift 2 ;;
    --verify-limit) VERIFY_LIMIT="$2"; shift 2 ;;
    --tolerance) VERIFY_TOLERANCE="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

TOOLS=$(cd "$(dirname "$0")" && pwd)
WORKSPACE=$(cd "$TOOLS/../.." && pwd)
[ -z "$PYLOG" ] && PYLOG="$WORKSPACE/profile/python.log"

SESSION=$(basename "$REPLAY")
RECORDING_NAME=$(basename "$RECORDING")
PARSED="$WORKSPACE/replays/parsed"
ALIGN_DIR="$PARSED/alignment"
LABEL_DIR="$PARSED/labels"

for path in "$RECORDING/session.json" "$RECORDING/frames.jsonl" "$RECORDING/capture.mkv" \
            "$PARSED/events.jsonl" "$REPLAY" "$PYLOG"; do
  if [ ! -e "$path" ]; then
    echo "missing input: $path" >&2
    exit 1
  fi
done
if [ -z "$CALIBRATION" ]; then
  echo "no --calibration given. Choose a HUD rectangle once, with a known value:" >&2
  echo "  $PYTHON $TOOLS/calibrate_hud.py detect $RECORDING/capture.mkv --at <s> \\" >&2
  echo "      --search x,y,w,h --element own_hp_bar --pick 0 --pad-y 5 --out <calibration.json>" >&2
  exit 2
fi
if [ ! -f "$CALIBRATION" ]; then
  echo "missing calibration: $CALIBRATION" >&2
  exit 1
fi
mkdir -p "$ALIGN_DIR" "$LABEL_DIR"

ANCHORS="$ALIGN_DIR/$RECORDING_NAME.anchors.csv"
ANCHORS_REPORT="$ALIGN_DIR/$RECORDING_NAME.anchors.report.json"
ALIGNMENT="$ALIGN_DIR/$RECORDING_NAME.json"
ALIGNED_EVENTS="$ALIGN_DIR/$RECORDING_NAME.events.aligned.jsonl"
LABELS="$LABEL_DIR/${SESSION%.wowsreplay}.own_hp.jsonl"
LABEL_SUMMARY="$LABEL_DIR/${SESSION%.wowsreplay}.own_hp.summary.json"
VERIFY="$LABEL_DIR/${SESSION%.wowsreplay}.own_hp.verify.json"

echo "=== 1/4 anchors from the client log: $RECORDING_NAME"
if [ -s "$ANCHORS" ] && [ "$FORCE" -eq 0 ]; then
  echo "reusing $ANCHORS (pass --force to rebuild)"
else
  "$PYTHON" "$TOOLS/anchors_from_pylog.py" \
    --pylog "$PYLOG" --recording "$RECORDING" --events "$PARSED/events.jsonl" \
    --session "$SESSION" --out "$ANCHORS" --report "$ANCHORS_REPORT"
fi
echo "anchors: $(($(wc -l < "$ANCHORS") - 1)) rows"

echo "=== 2/4 fit replay_clock to video_time"
"$PYTHON" "$TOOLS/align_timeline.py" \
  "$ANCHORS" --events "$PARSED/events.jsonl" \
  --aligned-out "$ALIGNED_EVENTS" --out "$ALIGNMENT"
"$PYTHON" - "$ALIGNMENT" <<'PY'
import json, sys
row = json.load(open(sys.argv[1]))
print("  rate %.9f  offset_s %.6f  anchors %d  rmse_s %.6f  max_abs_residual_s %.6f  r_squared %.9f"
      % (row["rate"], row["offset_s"], row["anchors"], row["rmse_s"],
         row["max_abs_residual_s"], row["r_squared"]))
PY

echo "=== 3/4 derive labels"
"$PYTHON" "$TOOLS/derive_labels.py" \
  --events "$PARSED/events.jsonl" --session "$SESSION" \
  --alignment "$ALIGNMENT" --calibration "$CALIBRATION" \
  --recording "$RECORDING" --step-s "$STEP_S" \
  --probe-video "$RECORDING/capture.mkv" \
  --out "$LABELS" --summary "$LABEL_SUMMARY"

echo "=== 4/4 check the labels against the pixels"
"$PYTHON" "$TOOLS/verify_hp_labels.py" \
  --labels "$LABELS" --video "$RECORDING/capture.mkv" \
  --calibration "$CALIBRATION" --limit "$VERIFY_LIMIT" \
  --tolerance "$VERIFY_TOLERANCE" --json-out "$VERIFY"

echo
echo "artifacts:"
echo "  anchors    $ANCHORS"
echo "  alignment  $ALIGNMENT"
echo "  labels     $LABELS"
echo "  check      $VERIFY"
