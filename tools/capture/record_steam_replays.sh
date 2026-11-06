#!/usr/bin/env bash
# Record several replays back to back through the resident Steam session.
#
# This is the Steam-route counterpart of record_replays.sh. It assumes what is
# already running, never starts a game two at a time, and stops the game client
# after capture stops, before the full container verification.
#
# Prerequisites (start once):
#   /usr/bin/python3 tools/capture/capture_host.py host --verify full &
#   bash tools/capture/start_steam_session.sh
#
# Usage:
#   bash tools/capture/record_steam_replays.sh recordings replays/<a>.15.9.wowsreplay.pending ...
#
# The session directory keeps the version tag of the replay file, so a 15.9.0
# recording can never be merged with a 15.8.0 one by name.
set -euo pipefail

WORK=$(cd "$(dirname "$0")/../.." && pwd)
HOST="$WORK/tools/capture/capture_host.py"
CTL="$WORK/tools/capture/steam_session_ctl.sh"
PY=/usr/bin/python3

if [ "$#" -lt 2 ]; then
  echo "usage: $0 OUTPUT_ROOT REPLAY [REPLAY ...]" >&2
  exit 2
fi
OUT_ROOT="$1"
shift
case "$OUT_ROOT" in
  /tmp|/tmp/*) echo "refusing to put recordings under /tmp" >&2; exit 2 ;;
esac
mkdir -p "$OUT_ROOT"

game_pids() { ps -eo pid=,stat=,comm= | awk '$3 == "WorldOfWarships" && $2 !~ /^Z/ {print $1}'; }

cleanup_game() {
  bash "$CTL" kill-game || return 1
  for _ in $(seq 1 30); do
    [ -z "$(game_pids)" ] && return 0
    sleep 1
  done
  echo "game client did not stop within 30 s" >&2
  return 1
}
active_game=0
cleanup_on_exit() {
  if [ "$active_game" -eq 1 ]; then cleanup_game || true; fi
}
trap cleanup_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$PY" "$HOST" preflight | grep -q '"target_resolved": true' || {
  echo "no capture source: is the Steam session running?" >&2; exit 1; }
"$PY" "$HOST" status | grep -q '"host_pid"' || {
  echo "no capture host is listening" >&2; exit 1; }

if [ -n "$(game_pids)" ]; then
  echo "a game client is already running; stop it first (steam_session_ctl.sh kill-game)" >&2
  exit 1
fi

failed=0
recorded=0
for replay in "$@"; do
  [ -f "$replay" ] || { echo "skip: no such replay: $replay" >&2; failed=$((failed + 1)); continue; }
  name=$(basename "$replay")
  name=${name%.wowsreplay.pending}
  name=${name%.wowsreplay}
  out="$OUT_ROOT/$name"
  if [ -e "$out/session.json" ]; then
    echo "skip: recording already exists: $out" >&2
    continue
  fi
  echo "=== recording $name"
  # --wait-battle: never capture a client that sits on the login screen.
  "$PY" "$HOST" record start "$out" --replay "$(readlink -f "$replay")" \
    --until-exit --wait-battle --accepted \
    || { echo "start refused for $name" >&2; failed=$((failed + 1)); continue; }
  # The minimap size is an in-game SETTING that resets to the client's default on every
  # launch, and the whole label chain assumes the maximum panel: a recording made at the
  # default size is cropped from the wrong rectangle, and nothing downstream reports it
  # (that is how the 2026-10-10 sessions shipped with most of the crop showing the 3D
  # world). The direct-launch route already presses `+` across the start-up window; this
  # route did not, which makes a Steam-route recording silently unusable. A press at the
  # maximum is ignored by the game, so the repeated burst is safe.
  active_game=1
  if ! bash "$CTL" launch "$replay"; then
    echo "launch failed for $name" >&2
    "$PY" "$HOST" record stop || true
    failed=$((failed + 1))
    cleanup_game || exit 1
    active_game=0
    continue
  fi
  # --follow exits without a live client, so launch must precede it.
  mkdir -p "$out"
  if [ "${WW_FORCE_MINIMAP_MAX:-1}" = "1" ]; then
    setsid nohup "$WORK/tools/capture/force_minimap_max.sh" --follow \
      > "$out/minimap-max.log" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$out/minimap-max.pid"
  fi
  # The host stops and finalises on its own: replay finished, or client gone, or
  # the internal cap. Wait for that instead of guessing a duration.
  for _ in $(seq 1 1440); do
    [ -f "$out/session.json" ] && break
    # Capture has reached EOS; verification no longer needs the harbour client.
    if [ "$active_game" -eq 1 ]; then
      status=$("$PY" "$HOST" status) || exit 1
      if printf '%s' "$status" | "$PY" -c 'import json,sys; from pathlib import Path; s=json.load(sys.stdin); r=s.get("recording") or {}; sys.exit(0 if s.get("phase")=="finalising" and r.get("state")=="stopped" and Path(r.get("out", "")).resolve()==Path(sys.argv[1]).resolve() else 1)' "$out"; then
        cleanup_game || exit 1
        active_game=0
      fi
    fi
    sleep 10
  done
  if [ -f "$out/session.json" ]; then
    echo "=== finished $name: $(grep -o '"sidecar_frames": [0-9]*' "$out/session.json" | head -1)"
    recorded=$((recorded + 1))
  else
    echo "no session.json for $out: the recording did not finalise" >&2
    failed=$((failed + 1))
  fi
  # One game at a time: the client stays in the harbour after a replay, so it is
  # stopped here, and only the game process is signalled.
  cleanup_game || exit 1
  active_game=0
done

echo "recorded $recorded, failed $failed"
[ "$failed" -eq 0 ]
