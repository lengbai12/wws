#!/usr/bin/env bash
# Cheap end-to-end check of capture_host.py that never launches the game.
#
# A headless gamescope with a trivial GL client provides a real PipeWire video
# source, so the command socket, the pipeline, the arrival-time sidecar, the
# stop path and the verification all run without touching Steam or the client.
# Artefacts stay under state/smoke so a mechanism check can never be mistaken
# for corpus material.
#
# Usage: bash tools/capture/smoke_capture_host.sh [SECONDS]
set -euo pipefail

WORK=$(cd "$(dirname "$0")/../.." && pwd)
STATE="$WORK/state/smoke"
SECONDS_TO_RECORD="${1:-20}"
OUT="state/smoke/recording-host-smoke"
PY=/usr/bin/python3
CTL=("$PY" "$WORK/tools/capture/capture_host.py" --socket "$STATE/control.sock")
mkdir -p "$STATE"

if [ -e "$STATE/control.sock" ]; then
  echo "a smoke host is already listening at $STATE/control.sock" >&2
  exit 1
fi
if [ -e "$WORK/$OUT/capture.mkv" ]; then
  echo "smoke output already exists at $OUT" >&2
  exit 1
fi

node_present() {
  # grep -q exits on the first match, so the dump goes to a file first: with
  # pipefail a SIGPIPE on pw-dump would otherwise fail the pipeline.
  pw-dump > "$STATE/pw-dump.json" 2>/dev/null || return 1
  grep -q '"node.name": "gamescope"' "$STATE/pw-dump.json"
}

# Reap smoke compositors from an earlier run. Discovery fails closed on an
# ambiguous source, so a leftover instance must not stay behind. The match is on
# this script's own gamescope command line, and this shell's process group is
# excluded so the awk cannot select the caller.
SELF_PGID=$(ps -o pgid= -p "$$" | tr -d ' ')
for pid in $(ps -eo pid=,pgid=,args= | awk -v self="$SELF_PGID" '
    $2 != self && /gamescope -W 1920 -H 1080/ && /--backend headless/ && /glxgears/ {print $1}'); do
  echo "== reaping smoke compositor $pid"
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 2

if ! node_present; then
  echo "== starting a headless gamescope with glxgears"
  setsid nohup gamescope -W 1920 -H 1080 -w 1920 -h 1080 --backend headless \
    -- glxgears > "$STATE/gamescope.log" 2>&1 < /dev/null &
  for _ in $(seq 1 30); do
    node_present && break
    sleep 1
  done
  node_present || { echo "gamescope video source did not appear" >&2; exit 1; }
fi

GAMESCOPE_PID=$(ps -eo pid=,args= | awk '
    /gamescope -W 1920 -H 1080/ && /--backend headless/ && /glxgears/ {print $1; exit}')
[ -n "$GAMESCOPE_PID" ] || { echo "could not resolve the smoke compositor pid" >&2; exit 1; }
echo "== smoke compositor pid $GAMESCOPE_PID"

echo "== starting the capture host"
setsid nohup "${CTL[@]}" host --state-dir "$STATE/host" --verify fast \
  > "$STATE/host.log" 2>&1 < /dev/null &
for _ in $(seq 1 20); do
  [ -e "$STATE/control.sock" ] && break
  sleep 0.5
done
[ -e "$STATE/control.sock" ] || { echo "host socket did not appear" >&2; exit 1; }

echo "== preflight"
"${CTL[@]}" preflight

echo "== record start ($SECONDS_TO_RECORD s)"
"${CTL[@]}" record start "$OUT" --no-wait-game --no-until-exit \
  --gamescope-pid "$GAMESCOPE_PID" \
  --seconds "$SECONDS_TO_RECORD"

for _ in $(seq 1 "$SECONDS_TO_RECORD"); do
  sleep 1
  "${CTL[@]}" status | tr -d '\n' | sed -n 's/.*"frames": \([0-9]*\).*/frames so far: \1/p'
done

echo "== record stop"
"${CTL[@]}" record stop

echo "== quit"
"${CTL[@]}" quit || true

echo "== container frame count"
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=nb_read_frames -of default=nk=1:nw=1 \
  "$WORK/$OUT/capture.mkv"
echo "== sidecar lines"
wc -l < "$WORK/$OUT/frames.jsonl"
echo "smoke artefacts in $OUT"
