#!/usr/bin/env bash
# Restart a CN (国服) training-room battle with frame-verified steps.
#
# Chain: results --回到港口--> port --加入战斗--> lobby --已准备好--> ready
#        --加入战斗--> prebattle --开始战斗--> battle/loading
#
# Default is a DRY RUN: it captures one frame, prints the current screen and
# every pending step's click point (content / window / root) with the pixel
# criterion that must hold afterwards, and sends no input at all.  Only
# --execute walks the pointer with xdotool and clicks; each click is followed
# by a fresh capture and the run stops (exit 1) with the frame saved when the
# expected screen does not appear.
#
# Usage:
#   tools/capture/restart_training_room.sh                 # dry run
#   tools/capture/restart_training_room.sh --execute       # actually click
#   tools/capture/restart_training_room.sh --out state/xyz --step-timeout 15
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
DRIVER="$ROOT/tools/capture/training_room_ui.py"
DISPLAY_ID="${DISPLAY_ID:-:2}"
EXECUTE=0
OUT=""
STEP_TIMEOUT=12
MAX_STEPS=5
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1 ;;
    --out) OUT="${2:?--out needs a directory}"; shift ;;
    --out=*) OUT="${1#*=}" ;;
    --display) DISPLAY_ID="${2:?--display needs a value}"; shift ;;
    --display=*) DISPLAY_ID="${1#*=}" ;;
    --step-timeout) STEP_TIMEOUT="${2:?--step-timeout needs seconds}"; shift ;;
    --step-timeout=*) STEP_TIMEOUT="${1#*=}" ;;
    --max-steps) MAX_STEPS="${2:?--max-steps needs a count}"; shift ;;
    --max-steps=*) MAX_STEPS="${1#*=}" ;;
    -h|--help)
      awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -x "$PY" ]; then
  echo "missing interpreter: $PY" >&2
  exit 2
fi
if [ ! -f "$DRIVER" ]; then
  echo "missing driver: $DRIVER" >&2
  exit 2
fi

if [ -z "$OUT" ]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  if [ "$EXECUTE" -eq 1 ]; then
    OUT="$ROOT/state/training-room-restart-${STAMP}/run"
  else
    OUT="$ROOT/state/training-room-restart-${STAMP}/dry"
  fi
fi

MODE="dry-run"
if [ "$EXECUTE" -eq 1 ]; then
  MODE="execute"
fi

echo "restart_training_room: mode=$MODE display=$DISPLAY_ID out=$OUT"
if [ "$EXECUTE" -eq 1 ]; then
  echo "  every step is one bounded click, verified by a fresh capture; a missing"
  echo "  state change stops the run and saves the frame."
else
  echo "  no input will be sent (xdotool is not called for clicks in this mode)."
fi

ARGS=(--out "$OUT" --display "$DISPLAY_ID" --step-timeout "$STEP_TIMEOUT"
      --max-steps "$MAX_STEPS")
if [ "$EXECUTE" -eq 1 ]; then
  ARGS+=(--execute)
else
  ARGS+=(--text)
fi

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/xyz}"
export PYTHONPATH="$ROOT/tools/capture:$ROOT/tools/replay${PYTHONPATH:+:$PYTHONPATH}"

"$PY" "$DRIVER" "${ARGS[@]}" "${EXTRA[@]+"${EXTRA[@]}"}"
STATUS=$?
echo "restart_training_room: exit $STATUS"
exit "$STATUS"
