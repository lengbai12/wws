#!/usr/bin/env bash
# Attach the validated recorder to an already-running gamescope instance.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "usage: $0 OUTPUT_DIR GAME_SCOPE_PID SECONDS" >&2
  exit 2
fi
OUT="$1"
PID="$2"
SECONDS_TO_RECORD="$3"
case "$OUT" in
  /tmp|/tmp/*) echo "refusing to put a recording under /tmp" >&2; exit 2 ;;
esac
if [ -e "$OUT/capture.mkv" ] || [ -e "$OUT/frames.jsonl" ] || [ -e "$OUT/session.json" ]; then
  echo "recording output already exists: $OUT" >&2
  exit 1
fi
if ! kill -0 "$PID" 2>/dev/null; then
  echo "gamescope PID is not alive: $PID" >&2
  exit 1
fi
mkdir -p "$OUT"
exec /usr/bin/python3 tools/capture/gamescope_record.py \
  --out "$OUT" \
  --seconds "$SECONDS_TO_RECORD" \
  --bitrate-mbps 2 \
  --rc-mode cbr \
  --verify \
  --gamescope-pid "$PID"
