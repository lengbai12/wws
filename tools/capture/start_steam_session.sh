#!/usr/bin/env bash
# A resident gamescope session that hosts the Steam client.
#
# A Steam-launched replay renders on whatever display the Steam client gives it.
# The client here is started inside gamescope, so every game it launches is a
# descendant of that compositor and therefore lands on the gamescope PipeWire
# node that capture_host.py captures. The client stays alive when a replay ends,
# so one session serves every replay and the capture host reattaches per replay
# instead of the game being wrapped again for each one.
#
# The session writes its state to state/steam-session/ so the capture host and
# the launcher can find the compositor process and the display it created.
#
# Usage:
#   bash tools/capture/start_steam_session.sh [--headless] [--steam-appid N]
#
# Nothing here touches the Wine prefix: the client, Proton and the prefix are
# started by Steam itself, exactly as they are for a normal launch.
set -euo pipefail

WORK=$(cd "$(dirname "$0")/../.." && pwd)
STATE_DIR="${WW_STEAM_STATE:-$WORK/state/steam-session}"
APPID=552990
BACKEND=()
STOP_EXISTING=0

# The client that is already running decides where a game renders. A client
# started outside this session keeps its own display, so the game would never
# reach our compositor and the capture source would stay empty. comm is checked
# as well as the command line: a pattern-only match selects the command that is
# doing the looking.
existing_clients() {
  ps -eo pid=,comm=,args= | awk '$2 == "steam" && /ubuntu12_32\/steam/ {print $1}'
}
game_running() {
  ps -eo pid=,comm= | awk '$2 == "WorldOfWarships" {print $1}'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --headless) BACKEND=(--backend headless); shift ;;
    --steam-appid) APPID="$2"; shift 2 ;;
    --stop-existing-client) STOP_EXISTING=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The geometry is a contract with every existing calibration (HUD rects, the
# minimap ROI, the metres-per-pixel scale), so it must not be changed on its own.
# Measured: on the 2560x1600 desktop the node is 2560x1600 and the 1920x1080
# render is scaled into a 2560x1440 content area with 80 px bars, which is
# exactly the layout the direct-launch route produces. Headless has no desktop
# output to fill, so there the node is the -W/-H size instead.
GAMESCOPE_ARGS=(-W 1920 -H 1080 -w 1920 -h 1080 -f "${BACKEND[@]}")

if [ -n "$(game_running)" ]; then
  echo "a game client is running; stop it before starting a session" >&2
  exit 1
fi
EXISTING=$(existing_clients | tr '\n' ' ')
if [ -n "$EXISTING" ]; then
  if [ "$STOP_EXISTING" -ne 1 ]; then
    echo "a Steam client is already running outside this session: $EXISTING" >&2
    echo "it would launch the game on its own display, not on this session's," >&2
    echo "so nothing would be capturable. Quit it, or pass --stop-existing-client" >&2
    exit 1
  fi
  echo "== stopping the Steam client outside this session: $EXISTING"
  for pid in $EXISTING; do kill -TERM "$pid" 2>/dev/null || true; done
  for _ in $(seq 1 30); do
    [ -z "$(existing_clients)" ] && break
    sleep 1
  done
  if [ -n "$(existing_clients)" ]; then
    echo "the running Steam client did not exit; not starting a second one" >&2
    exit 1
  fi
fi

mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"

if [ -f "$STATE_DIR/gamescope.pid" ]; then
  OLD=$(cat "$STATE_DIR/gamescope.pid")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "a steam session is already running (gamescope pid $OLD)" >&2
    exit 1
  fi
fi
# Matching is on the process name as well as the command line: a pattern-only
# match selects the very command that is looking for it.
if ps -eo pid=,comm=,args= | awk '$2 ~ /gamescope/ && /-W 1920/ && /-- steam/ {found=1} END {exit !found}'; then
  echo "a gamescope/steam session is already running" >&2
  exit 1
fi

printf '%s\n' "$(date --iso-8601=seconds)" > "$STATE_DIR/started_at.txt"
echo "== starting gamescope ${GAMESCOPE_ARGS[*]} -- steam"
setsid nohup gamescope "${GAMESCOPE_ARGS[@]}" -- steam \
  > "$STATE_DIR/steam.log" 2>&1 < /dev/null &

# setsid may return a wrapper pid, so the compositor is resolved by its command
# line, the same way the direct-launch supervisor resolves it.
GAMESCOPE_PID=""
for _ in $(seq 1 30); do
  GAMESCOPE_PID=$(ps -eo pid=,comm=,args= | awk '
      $2 ~ /gamescope/ && /-W 1920/ && /-- steam/ {print $1; exit}')
  [ -n "$GAMESCOPE_PID" ] && break
  sleep 1
done
if [ -z "$GAMESCOPE_PID" ]; then
  echo "gamescope did not start; see $STATE_DIR/steam.log" >&2
  exit 1
fi
printf '%s\n' "$GAMESCOPE_PID" > "$STATE_DIR/gamescope.pid"

# The video source must exist before a recording can attach. Process identity is
# checked here as well as in the capture host, so a stale node is never used.
READY=0
for _ in $(seq 1 60); do
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
        clients[item.get("id")] = (item.get("info") or {}).get("props") or {}
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
' "$GAMESCOPE_PID"; then
    READY=1
    break
  fi
  if ! kill -0 "$GAMESCOPE_PID" 2>/dev/null; then
    echo "gamescope exited while waiting for its video source" >&2
    exit 1
  fi
  sleep 1
done
if [ "$READY" -ne 1 ]; then
  echo "the gamescope video source did not become ready" >&2
  exit 1
fi

# Wait for the Steam client itself, then record the environment it created: a
# later applaunch talks to this client, and the game inherits these values.
STEAM_PID=""
for _ in $(seq 1 120); do
  STEAM_PID=$(ps -eo pid=,comm=,args= | awk -v gc="$GAMESCOPE_PID" '
      $2 == "steam" && $1 != gc {print $1; exit}')
  [ -n "$STEAM_PID" ] && break
  kill -0 "$GAMESCOPE_PID" 2>/dev/null || break
  sleep 1
done
if [ -z "$STEAM_PID" ]; then
  echo "warning: no steam client process was found yet" >&2
fi

/usr/bin/python3 - "$STATE_DIR" "$GAMESCOPE_PID" "$STEAM_PID" "$APPID" \
  "${GAMESCOPE_ARGS[@]}" <<'PY'
import json, os, sys
from pathlib import Path

state_dir, gamescope_pid, steam_pid, appid = (Path(sys.argv[1]), sys.argv[2],
                                              sys.argv[3], sys.argv[4])
args = sys.argv[5:]
display = {}
if steam_pid:
    try:
        environ = Path(f"/proc/{steam_pid}/environ").read_bytes()
    except OSError:
        environ = b""
    for entry in environ.split(b"\0"):
        if b"=" not in entry:
            continue
        key, _, value = entry.partition(b"=")
        key = key.decode("utf-8", "replace")
        if key in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_RUNTIME_DIR", "XAUTHORITY"):
            display[key] = value.decode("utf-8", "replace")
payload = {
    "gamescope_pid": int(gamescope_pid),
    "gamescope_args": args,
    "steam_pid": int(steam_pid) if steam_pid else None,
    "steam_appid": int(appid),
    "steam_display": display,
    "started_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
    "steam_log": str(state_dir / "steam.log"),
}
(state_dir / "session.json").write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"gamescope pid {gamescope_pid}, steam pid {steam_pid or 'unknown'}, "
      f"display {display.get('WAYLAND_DISPLAY') or display.get('DISPLAY') or 'unset'}")
PY

echo "steam session state in $STATE_DIR"
echo "next: bash tools/capture/steam_session_ctl.sh launch <replay.15.9.wowsreplay.pending>"
