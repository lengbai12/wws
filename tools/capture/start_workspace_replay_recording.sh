#!/usr/bin/env bash
# Start one complete WoWS replay and its workspace-resident gamescope recording.
# The output directory is intentionally never /tmp and is never overwritten.
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 OUTPUT_DIR REPLAY_FILE [SECONDS]" >&2
  exit 2
fi

OUT="$1"
REPLAY="$2"
SECONDS_TO_RECORD="$3"
# 0 or "until-exit" records until the game process disappears, which is what
# happens when a replay finishes and the client returns to the harbour.
RECORD_UNTIL_EXIT=0
if [ -z "$SECONDS_TO_RECORD" ]; then SECONDS_TO_RECORD=1500; fi
case "$SECONDS_TO_RECORD" in
  0|until-exit) RECORD_UNTIL_EXIT=1; SECONDS_TO_RECORD=3600 ;;
esac
SCRIPT_DIR=$(dirname "$0")

# A recording is a heavy job, so it takes the same lock every other heavy job takes: a
# batch cannot start beside a merge or a training. Re-entrant on purpose - auto_learn
# already holds the lock for the whole batch, and it exports HEAVY_LOCK_HELD so this
# per-session call does not ask for it twice (which would refuse itself).
if [ -z "${HEAVY_LOCK_HELD:-}" ]; then
  export HEAVY_LOCK_HELD=1
  exec "$SCRIPT_DIR/../replay/heavy_lock.sh" run recording -- "$0" "$@"
fi

case "$OUT" in
  /tmp|/tmp/*) echo "refusing to put a recording under /tmp" >&2; exit 2 ;;
esac
if [ -e "$OUT/capture.mkv" ] || [ -e "$OUT/frames.jsonl" ] || [ -e "$OUT/session.json" ]; then
  echo "recording output already exists: $OUT" >&2
  exit 1
fi
if [ ! -f "$REPLAY" ]; then
  echo "replay does not exist: $REPLAY" >&2
  exit 1
fi

REPLAY_ABS=$(readlink -f "$REPLAY")
# A stale instance of THIS replay is stopped by killing the game process itself and
# nothing else. The old version collected process GROUPS whose command line mentioned the
# replay and killed the whole group; that reaches the Proton prefix's steam.exe and
# wineserver, which is the documented cause of the next client dying with
# "Could not initialize platform manager( PlatformType::STEAM )". It also matched its own
# command line, because the driver's argv contains the replay path as well, so the wait
# below always burned its full timeout.
game_pids() {
  # comm proves it is the game; the command line proves it is THIS replay.
  ps -eo pid=,comm=,args= | awk -v replay="$REPLAY_ABS" '
    $2 == "WorldOfWarships" && index($0, replay) { print $1 }'
}

for pid in $(game_pids); do
  echo "stopping stale game for this replay: pid $pid" >&2
  kill -TERM "$pid" 2>/dev/null || true
done
[ -n "$(game_pids)" ] && sleep 3
for pid in $(game_pids); do
  kill -KILL "$pid" 2>/dev/null || true
done
# Wait for it to be really gone before launching, by the same predicate: a dying client
# still holds the prefix, and launching into that fails.
for _ in $(seq 1 30); do
  [ -z "$(game_pids)" ] && break
  sleep 1
done
# There is deliberately NO `wineserver -k` here any more. It used to run unconditionally,
# on the theory that a force-killed session leaves the prefix's Steam platform state
# stale ("Could not initialize platform manager( PlatformType::STEAM )!"). Two things are
# wrong with that: killing the prefix's wineserver takes down every Proton process sharing
# it - including the Steam client's steam.exe - which is what CAUSES that failure for the
# next launch; and the project's own rule is that Wine and the prefix are never touched by
# tooling. Stopping the game process itself (above) is the whole cleanup.
sleep 3

mkdir -p "$OUT"
printf '%s\n' "$REPLAY_ABS" > "$OUT/replay_path.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "$OUT/started_at.txt"

setsid nohup "$SCRIPT_DIR/launch_wows_replay.sh" "$REPLAY_ABS" \
  > "$OUT/replay-launch.log" 2>&1 < /dev/null &
GAME_PID=$!
printf '%s\n' "$GAME_PID" > "$OUT/gamescope.pid"

# setsid's returned PID may be a wrapper. Resolve the actual gamescope process
# containing this replay path; the recorder identity guard needs this PID.
ACTUAL_GAMESCOPE_PID=""
for _ in $(seq 1 30); do
  ACTUAL_GAMESCOPE_PID=$(ps -eo pid=,args= | awk -v replay="$REPLAY_ABS" '
    $0 ~ /gamescope -W/ && index($0, replay) { print $1; exit }
  ')
  [ -n "$ACTUAL_GAMESCOPE_PID" ] && break
  sleep 1
done
if [ -z "$ACTUAL_GAMESCOPE_PID" ]; then
  echo "could not resolve actual gamescope PID for replay" >&2
  exit 1
fi
printf '%s\n' "$ACTUAL_GAMESCOPE_PID" > "$OUT/actual-gamescope.pid"

# A gamescope node exists during Proton's explorer/xalia bootstrap.  Do not
# record that stage: wait for the actual matching WoWS executable first.
WOWS_PID=""
for _ in $(seq 1 180); do
  # The gamescope command line itself contains the replay and executable as
  # arguments. Match the Wine process name (historically comm=WorldOfWarships)
  # so the bootstrap compositor cannot satisfy this gate.
  WOWS_PID=$(ps -eo pid=,comm=,args= | awk -v replay="$REPLAY_ABS" '$2 == "WorldOfWarships" && index($0,replay) { print $1; exit }')
  [ -n "$WOWS_PID" ] && break
  if ! kill -0 "$ACTUAL_GAMESCOPE_PID" 2>/dev/null; then
    echo "gamescope exited before WorldOfWarships64 became ready" >&2
    exit 1
  fi
  sleep 1
done
if [ -z "$WOWS_PID" ]; then
  echo "matching WorldOfWarships64 process did not become ready" >&2
  exit 1
fi
printf '%s\n' "$WOWS_PID" > "$OUT/wows.pid"

READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$ACTUAL_GAMESCOPE_PID" 2>/dev/null; then
    echo "replay process exited before gamescope became capturable" >&2
    exit 1
  fi
  if pw-dump 2>/dev/null | /usr/bin/python3 -c '
import json, sys
pid = str(sys.argv[1])
try:
    dump = json.load(sys.stdin)
except Exception:
    raise SystemExit(1)
clients = {}
for item in dump:
    if item.get("type") == "PipeWire:Interface:Client":
        props = (item.get("info") or {}).get("props") or {}
        clients[item.get("id")] = props
for item in dump:
    if item.get("type") != "PipeWire:Interface:Node":
        continue
    props = (item.get("info") or {}).get("props") or {}
    if props.get("media.class") != "Video/Source":
        continue
    client = clients.get(props.get("client.id"), {})
    if (props.get("node.name") == "gamescope"
            and str(client.get("application.process.id")) == pid
            and str(client.get("pipewire.sec.pid")) == pid):
        raise SystemExit(0)
raise SystemExit(1)
' "$ACTUAL_GAMESCOPE_PID"; then
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "gamescope Video/Source node did not become ready" >&2
  exit 1
fi

# WW_VERIFY=fast checks duration and arrival times only. A long recording costs
# minutes of decode for a full frame count, so batch runs use fast and state in
# session.json that the frame count was not checked.
if [ "${WW_VERIFY:-full}" = "fast" ]; then VERIFY_ARGS=(--verify-fast); else VERIFY_ARGS=(--verify); fi
setsid nohup /usr/bin/python3 "$SCRIPT_DIR/gamescope_record.py" \
  --out "$OUT" \
  --seconds "$SECONDS_TO_RECORD" \
  --bitrate-mbps 2 \
  --rc-mode cbr \
  "${VERIFY_ARGS[@]}" \
  --gamescope-pid "$ACTUAL_GAMESCOPE_PID" \
  > "$OUT/recorder.log" 2>&1 < /dev/null &
RECORDER_PID=$!
printf '%s\n' "$RECORDER_PID" > "$OUT/recorder.pid"

# Stop the recorder when the replay finishes, or when the game exits, whichever
# comes first. A finished replay does NOT close the client: it returns to the
# results and harbour screens and keeps running, so the client log is the real
# signal. The recorder is stopped through its own signal handler so the
# container is finalised and the sidecar stays readable.
if [ "$RECORD_UNTIL_EXIT" -eq 1 ]; then
  PYLOG="${WW_PYLOG:-$SCRIPT_DIR/../../profile/python.log}"
  PYLOG_START=0
  [ -f "$PYLOG" ] && PYLOG_START=$(stat -c %s "$PYLOG")
  setsid nohup bash -c '
    pid="$1"; rec="$2"; out="$3"; log="$4"; from="$5"
    reason=""
    while :; do
      if ! kill -0 "$pid" 2>/dev/null; then reason="game-exited"; break; fi
      if [ -f "$log" ]; then
        if tail -c +"$((from + 1))" "$log" | grep -q "BattleReplay\] onReplayFinished"; then
          reason="replay-finished"; break
        fi
      fi
      sleep 2
    done
    printf "stop reason: %s\n" "$reason" > "$out/stop_reason.txt"
    sleep 6
    kill -TERM "$rec" 2>/dev/null || true
    # The client does NOT exit when a replay ends: it writes onReplayFinished and
    # returns to the harbour. Close it HERE rather than after the recorder has
    # finalised, because finalising runs --verify, which decodes every frame and can
    # take minutes on a long recording, and the client would sit in the harbour for
    # all of it.
    if [ -f "$out/wows.pid" ]; then
      kill -TERM "$(cat "$out/wows.pid")" 2>/dev/null || true
    fi
    printf "%s\n" "$(date --iso-8601=seconds)" > "$out/stopped_at.txt"
  ' _ "$WOWS_PID" "$RECORDER_PID" "$OUT" "$PYLOG" "$PYLOG_START" \
      > "$OUT/watchdog.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$OUT/watchdog.pid"
  printf '%s\n' "$PYLOG" > "$OUT/pylog_path.txt"
fi
# The minimap size is in-game UI state, changed with + and -, and it decides the
# rectangle every label is cropped with. It has already differed silently between
# recordings once: 10-10 recordings have a panel 46% of the size the calibration
# assumes. It is therefore reset on every replay instead of relied on. A press at
# the maximum is ignored by the game, so repeating the burst across the start-up
# window is safe.
if [ "${WW_FORCE_MINIMAP_MAX:-1}" = "1" ]; then
  setsid nohup "$SCRIPT_DIR/force_minimap_max.sh" --follow \
    > "$OUT/minimap-max.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$OUT/minimap-max.pid"
fi
printf '%s\n' "launcher_pid=$GAME_PID gamescope_pid=$ACTUAL_GAMESCOPE_PID recorder_pid=$RECORDER_PID" > "$OUT/processes.txt"
echo "started gamescope pid=$ACTUAL_GAMESCOPE_PID recorder pid=$RECORDER_PID output=$OUT"
