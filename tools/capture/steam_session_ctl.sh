#!/usr/bin/env bash
# Control a resident Steam session: launch a replay, check it, stop the client.
#
# Launching goes through the Steam client that start_steam_session.sh started
# inside gamescope, so the game renders on the compositor that capture_host.py
# captures. The client is version-matched by clientrunner, which always selects
# the newest build: 15.9.0 replays play, older builds do not.
#
# Usage:
#   steam_session_ctl.sh status
#   steam_session_ctl.sh launch <replay.wowsreplay.pending>
#   steam_session_ctl.sh kill-game
#   steam_session_ctl.sh stop-session
#
# Stopping never touches Wine: only the game process and, on request, the Steam
# client and gamescope are signalled. The prefix's wineserver is left alone, so
# the next launch does not fail with "Could not initialize platform manager".
set -euo pipefail

WORK=$(cd "$(dirname "$0")/../.." && pwd)
STATE_DIR="${WW_STEAM_STATE:-$WORK/state/steam-session}"
GAME_COMM=WorldOfWarships

state_field() {
  /usr/bin/python3 - "$STATE_DIR/session.json" "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
value = payload.get(sys.argv[2])
print("" if value is None else value)
PY
}

game_pids() {
  ps -eo pid=,stat=,comm= | awk -v comm="$GAME_COMM" '$3 == comm && $2 !~ /^Z/ {print $1}'
}

gamescope_pid() {
  if [ -f "$STATE_DIR/gamescope.pid" ]; then cat "$STATE_DIR/gamescope.pid"; fi
}

case "${1:-}" in
  status)
    GC=$(gamescope_pid)
    echo "state dir:      $STATE_DIR"
    echo "gamescope pid:  ${GC:-unknown}$([ -n "$GC" ] && kill -0 "$GC" 2>/dev/null && echo ' (alive)' || true)"
    echo "steam pid:      $(state_field steam_pid)"
    echo "steam display:  $(state_field steam_display)"
    echo "appid:          $(state_field steam_appid)"
    echo "game pids:      $(game_pids | tr '\n' ' ')"
    ;;

  launch)
    [ "$#" -eq 2 ] || { echo "usage: $0 launch <replay>" >&2; exit 2; }
    REPLAY=$(readlink -f "$2")
    [ -f "$REPLAY" ] || { echo "replay does not exist: $REPLAY" >&2; exit 1; }
    APPID=$(state_field steam_appid)
    APPID=${APPID:-552990}
    BEFORE=$(game_pids | tr '\n' ' ')
    # Talk to the Steam client of this session by using the environment it
    # created: the request reaches that client, which spawns the game as its
    # child, inside the same gamescope session.
    DISPLAY_VALUE=$(state_field DISPLAY)
    WAYLAND_VALUE=$(state_field WAYLAND_DISPLAY)
    printf '%s\n' "$REPLAY" > "$STATE_DIR/last_replay.txt"
    printf '%s\n' "$(date --iso-8601=seconds)" > "$STATE_DIR/last_launch_at.txt"
    env -u STEAM_COMPAT_DATA_PATH -u WINEPREFIX \
      ${DISPLAY_VALUE:+DISPLAY="$DISPLAY_VALUE"} \
      ${WAYLAND_VALUE:+WAYLAND_DISPLAY="$WAYLAND_VALUE"} \
      setsid nohup steam -applaunch "$APPID" "$REPLAY" \
      > "$STATE_DIR/launch.log" 2>&1 < /dev/null &
    echo "requested appid $APPID with $REPLAY (game pids before: ${BEFORE:-none})"
    for _ in $(seq 1 180); do
      NEW=$(game_pids | tr '\n' ' ')
      if [ "$NEW" != "$BEFORE" ] && [ -n "$NEW" ]; then
        printf '%s\n' "$NEW" | tr -d ' ' > "$STATE_DIR/game.pid"
        echo "game client pid(s): $NEW"
        exit 0
      fi
      sleep 1
    done
    echo "no game process appeared within 180 s; see $STATE_DIR/launch.log" >&2
    exit 1
    ;;

  kill-game)
    FOUND=0
    for pid in $(game_pids); do
      echo "stopping game client pid $pid"
      kill -TERM "$pid" 2>/dev/null || true
      FOUND=1
    done
    [ "$FOUND" -eq 1 ] || echo "no game client is running"
    ;;

  stop-session)
    for pid in $(game_pids); do kill -TERM "$pid" 2>/dev/null || true; done
    sleep 3
    STEAM_PID=$(state_field steam_pid)
    GC=$(gamescope_pid)
    # The Steam client and gamescope are asked to stop; the Wine prefix's
    # wineserver is deliberately not touched.
    [ -n "${STEAM_PID:-}" ] && kill -TERM "$STEAM_PID" 2>/dev/null || true
    [ -n "${GC:-}" ] && kill -TERM "$GC" 2>/dev/null || true
    echo "requested stop of the game, the Steam client and gamescope"
    ;;

  *)
    echo "usage: $0 {status|launch <replay>|kill-game|stop-session}" >&2
    exit 2
    ;;
esac
