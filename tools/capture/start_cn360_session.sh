#!/usr/bin/env bash
# A resident gamescope session that hosts the 360 (CN) Wargaming Game Center.
#
# Why: the perception/aiming work needs a *captured* live battle.  The Steam
# route runs the client inside gamescope so every game it launches is a
# descendant of that compositor and therefore lands on the gamescope PipeWire
# node that tools/capture/capture_host.py records.  The CN client is launched by
# the Game Center (WGC), so the same trick is applied one level up: gamescope
# hosts wine+wgc.exe, and the game it starts is a child of that compositor.
#
# The geometry is a contract with every existing calibration (HUD rects, minimap
# ROI, metres-per-pixel): -W/-H 1920x1080 scaled into a 2560x1440 content area
# with 80 px bars on a 2560x1600 desktop.  Do not change it on its own.
#
# Login: the launcher's embedded browser hangs on this network (see AGENTS.md
# 2026-10-16), so the shim that makes the login hosts fail fast is loaded here
# as well - it is limited to the launcher's own processes by FASTFAIL_PROCS, so
# the external browser that the WGC opens is not affected.
#
# Usage:
#   bash tools/capture/start_cn360_session.sh [--check] [--headless]
#                                             [--stop-existing]
#   .../cn360_session_ctl.sh  -> see tools/capture/cn360_session_ctl.sh
set -uo pipefail

WORK=$(cd "$(dirname "$0")/../.." && pwd)
STATE_DIR="${WW_CN360_STATE:-$WORK/state/cn360-session}"
PREFIX="${WINEPREFIX:-/home/lengbai/.wine}"
WGC="$PREFIX/drive_c/Program Files/360 Wargaming/GameCenter/wgc.exe"
SHIM="$WORK/state/cn360-tools/fastfail.so"
SHIM_SRC="$WORK/state/cn360-tools/fastfail.c"
FASTFAIL_LOG="${FASTFAIL_LOG:-$WORK/state/cn360-session-fastfail.log}"
LOGIN_HOSTS=(profile.wg.360.cn openapi.360.cn)
BACKEND=()
STOP_EXISTING=0
CHECK=0

# comm is checked as well as the command line: a pattern-only match selects the
# very command that is doing the looking.
wgc_running() {
  ps -eo pid=,comm= | awk '$2 == "wgc.exe" {print $1}'
}
game_running() {
  ps -eo pid=,comm= | awk '$2 ~ /^WorldOfWarships/ {print $1}'
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --headless) BACKEND=(--backend headless); shift ;;
    --stop-existing) STOP_EXISTING=1; shift ;;
    --check) CHECK=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# The game renders at its own preferred resolution; the *X screen* is -W/-H.
# If the client asks for 2560x1440 while the screen is 1920x1080, pointer
# input cannot reach the bottom ~360 px (measured: y clamps at 1079), so
# the launcher's bottom buttons become unclickable. Override the game
# size without changing the captured output size or the calibration.
GAME_W="${WW_CN360_GAME_W:-1920}"
GAME_H="${WW_CN360_GAME_H:-1080}"
# The window size follows the output size: a window larger than the X screen
# clamps pointer input before the bottom row of launcher buttons.
GAMESCOPE_ARGS=(-W "$GAME_W" -H "$GAME_H" -w "$GAME_W" -h "$GAME_H" -f "${BACKEND[@]}")

resolve_ips() {
  local ips=""
  for h in "${LOGIN_HOSTS[@]}"; do
    while read -r ip; do
      [ -n "$ip" ] || continue
      case ",$ips," in *",$ip,"*) continue ;; esac
      ips="${ips:+$ips,}$ip"
    done < <(getent ahostsv4 "$h" 2>/dev/null | awk '{print $1}' | sort -u)
  done
  printf '%s' "$ips"
}
IPS="$(resolve_ips)"

if [ "$CHECK" -eq 1 ]; then
  echo "=== check (no launch)"
  echo "workdir      : $WORK"
  echo "prefix       : $PREFIX"
  echo "state dir    : $STATE_DIR"
  echo "wgc.exe      : $([ -f "$WGC" ] && echo present || echo MISSING)"
  echo "shim .so     : $([ -f "$SHIM" ] && echo present || echo MISSING)"
  echo "gamescope    : $(command -v gamescope || echo MISSING)"
  echo "pw-dump      : $(command -v pw-dump || echo MISSING)"
  echo "login hosts  : ${LOGIN_HOSTS[*]}"
  echo "resolved IPs : ${IPS:-<none>}"
  echo "gamescope args: ${GAMESCOPE_ARGS[*]}"
  echo "wgc outside  : $(wgc_running | tr '\n' ' ')"
  echo "game running : $(game_running | tr '\n' ' ')"
  exit 0
fi

[ -f "$WGC" ] || { echo "MISSING launcher: $WGC" >&2; exit 1; }
command -v gamescope >/dev/null || { echo "MISSING gamescope" >&2; exit 1; }
command -v pw-dump >/dev/null || { echo "MISSING pw-dump" >&2; exit 1; }

if [ ! -f "$SHIM" ]; then
  echo "== building the login fast-fail shim"
  gcc -shared -fPIC -O2 -o "$SHIM" "$SHIM_SRC" -ldl -lpthread || exit 1
fi

if [ -n "$(game_running)" ]; then
  echo "a game client is already running; stop it before starting a session" >&2
  exit 1
fi

EXISTING=$(wgc_running | tr '\n' ' ')
if [ -n "$EXISTING" ]; then
  if [ "$STOP_EXISTING" -ne 1 ]; then
    echo "a Game Center is already running outside this session: $EXISTING" >&2
    echo "it would launch the game on its own display, not on this session's," >&2
    echo "so nothing would be capturable. Quit it, or pass --stop-existing." >&2
    exit 1
  fi
  echo "== stopping the Game Center outside this session: $EXISTING"
  for pid in $EXISTING; do kill -TERM "$pid" 2>/dev/null || true; done
  sleep 3
  for c in CrUtilityMain CrRendererMain WargamingErrorM; do pkill -TERM -x "$c" 2>/dev/null || true; done
  sleep 3
  if [ -n "$(wgc_running)" ]; then
    echo "the running Game Center did not exit; not starting a second one" >&2
    exit 1
  fi
fi

mkdir -p "$STATE_DIR"
chmod 700 "$STATE_DIR"

if [ -f "$STATE_DIR/gamescope.pid" ]; then
  OLD=$(cat "$STATE_DIR/gamescope.pid")
  if kill -0 "$OLD" 2>/dev/null; then
    echo "a cn360 session is already running (gamescope pid $OLD)" >&2
    exit 1
  fi
fi

printf '%s\n' "$(date --iso-8601=seconds)" > "$STATE_DIR/started_at.txt"
echo "== starting gamescope ${GAMESCOPE_ARGS[*]} -- wine wgc.exe"
setsid nohup gamescope "${GAMESCOPE_ARGS[@]}" -- \
  env WINEPREFIX="$PREFIX" WINEDEBUG=-all \
      LD_PRELOAD="$SHIM" \
      FASTFAIL_LOG="$FASTFAIL_LOG" \
      FASTFAIL_PROCS="wgc.exe,CrUtilityMain,CrRendererMain,wineserver" \
      FASTFAIL_IPS="$IPS" \
      wine "$WGC" \
  > "$STATE_DIR/session.log" 2>&1 < /dev/null &

GAMESCOPE_PID=""
for _ in $(seq 1 30); do
  GAMESCOPE_PID=$(ps -eo pid=,comm=,args= | awk -v w="-W $GAME_W" '
      $2 ~ /gamescope/ && index($0, w) && /wgc\.exe/ {print $1; exit}')
  [ -n "$GAMESCOPE_PID" ] && break
  sleep 1
done
if [ -z "$GAMESCOPE_PID" ]; then
  echo "gamescope did not start; see $STATE_DIR/session.log" >&2
  exit 1
fi
printf '%s\n' "$GAMESCOPE_PID" > "$STATE_DIR/gamescope.pid"

echo "== waiting for the gamescope PipeWire video source"
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
  sleep 1
done

/usr/bin/python3 - "$STATE_DIR" "$GAMESCOPE_PID" "$WGC" "$IPS" "$READY" <<'PY'
import json, os, sys
state, pid, wgc, ips, ready = sys.argv[1:6]
data = {
    "session": "cn360",
    "gamescope_pid": int(pid),
    "launcher": wgc,
    "login_hosts_blocked": [i for i in ips.split(",") if i],
    "pipewire_source_ready": ready == "1",
}
with open(os.path.join(state, "session.json"), "w") as fh:
    json.dump(data, fh, indent=2)
print(json.dumps(data, indent=2))
PY

if [ "$READY" -ne 1 ]; then
  echo "warning: no gamescope PipeWire video source found yet" >&2
fi
echo "cn360 session state in $STATE_DIR"
echo "next: log in inside the session (see AGENTS.md), then start the game from the launcher"
