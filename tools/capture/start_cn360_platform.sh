#!/usr/bin/env bash
# start_cn360_platform.sh - start the 360 (CN) Wargaming Game Center in Wine.
#
# Why this script exists (measured 2026-10-16)
# -------------------------------------------
# The Game Center's embedded browser (CEF / Chromium 92.0.4515 under Wine) hangs
# forever when it loads the 360 SSO login page: the request neither succeeds nor
# fails, so the launcher's error path never runs and the login panel just spins
# on "等待服务器响应......" (see AGENTS.md, 2026-10-16 21:30 section).
#
# When the login hosts are made to fail *fast* (connection refused instead of a
# silent black hole), the launcher immediately reports
#     [Auth] Notify failure: code = 6003; error = [Error.Authentication.WebLoginUrlFailure]
# and the panel shows "⚠ 授权错误。请稍后再试。（6003）" together with the fallback
# button "在浏览器上登录" (the execLoginInExternalBrowser action).  Clicking it
# opens the real login page in the system default browser (Firefox on this
# machine) and the launcher starts polling
#     POST https://wgn.wggames.cn/id/api/v3/game/ticket/consume/
# every 5 seconds until the browser-side login produces a ticket.
#
# So the default mode of this script starts the launcher with a small LD_PRELOAD
# shim (state/cn360-tools/fastfail.so) that refuses connections to the login
# hosts *only inside the launcher's own processes* (FASTFAIL_PROCS), which is
# exactly what is needed to reach the fallback button.
#
# Modes
#   (default)  launch with the fast-fail shim  -> login panel shows the fallback
#   --plain    launch without the shim         -> login panel hangs (the bug)
#   --check    resolve hosts, verify the shim, print the plan, do not launch
#   --stop     stop the launcher (kills only the launcher's own processes)
#
# Stopping never touches wineserver, the prefix, or any other Wine process.
set -uo pipefail

WORKDIR="${WORKDIR:-/home/lengbai/Downloads/wws}"
PREFIX="${WINEPREFIX:-/home/lengbai/.wine}"
DISPLAY_X="${DISPLAY:-:1}"
WGC="$PREFIX/drive_c/Program Files/360 Wargaming/GameCenter/wgc.exe"
SHIM="$WORKDIR/state/cn360-tools/fastfail.so"
SHIM_SRC="$WORKDIR/state/cn360-tools/fastfail.c"
FASTFAIL_LOG="${FASTFAIL_LOG:-$WORKDIR/state/cn360-platform-fastfail.log}"
SERVICE="${SERVICE:-wws-cn360-platform}"

# Hosts that the launcher's embedded login frame needs, and that this network
# black-holes on TCP 80/443 while answering ICMP.
LOGIN_HOSTS=(profile.wg.360.cn openapi.360.cn)

mode=shim
for a in "$@"; do
  case "$a" in
    --plain) mode=plain ;;
    --check) mode=check ;;
    --stop)  mode=stop ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

stop_launcher() {
  for c in wgc.exe CrUtilityMain CrRendererMain WargamingErrorM; do
    pkill -TERM -x "$c" 2>/dev/null && echo "  TERM $c"
  done
  sleep 3
  local left
  left=$(ps -eo comm= | grep -cE '^(wgc\.exe|CrUtilityMain|CrRendererMain|WargamingErrorM)$')
  echo "  remaining launcher processes: $left"
}

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

if [ "$mode" = stop ]; then
  echo "=== stopping launcher (only its own processes)"
  stop_launcher
  exit 0
fi

if [ "$mode" = check ]; then
  echo "=== check (no launch)"
  echo "workdir     : $WORKDIR"
  echo "prefix      : $PREFIX"
  echo "wgc.exe     : $([ -f "$WGC" ] && echo present || echo MISSING)"
  echo "shim .so    : $([ -f "$SHIM" ] && echo present || echo MISSING)  ($SHIM)"
  echo "shim .c     : $([ -f "$SHIM_SRC" ] && echo present || echo MISSING)"
  echo "login hosts : ${LOGIN_HOSTS[*]}"
  echo "resolved IPs: ${IPS:-<none>}"
  echo "launcher up : $(ps -eo comm= | grep -cE '^wgc\.exe$')"
  exit 0
fi

if [ ! -f "$WGC" ]; then echo "MISSING launcher: $WGC" >&2; exit 1; fi

if [ ! -f "$SHIM" ] && [ "$mode" = shim ]; then
  echo "=== building the shim"
  gcc -shared -fPIC -O2 -o "$SHIM" "$SHIM_SRC" -ldl -lpthread || exit 1
fi

if [ "$mode" = plain ]; then
  echo "=== starting launcher WITHOUT the shim (the login panel will hang)"
  env DISPLAY="$DISPLAY_X" WINEPREFIX="$PREFIX" WINEDEBUG=-all \
    wine "$WGC"
  exit $?
fi

if [ -z "$IPS" ]; then
  echo "WARNING: could not resolve ${LOGIN_HOSTS[*]}; starting without blocking" >&2
fi

echo "=== starting launcher WITH the fast-fail shim"
echo "    blocking: ${IPS:-<none>}"
echo "    log     : $FASTFAIL_LOG"
env DISPLAY="$DISPLAY_X" WINEPREFIX="$PREFIX" WINEDEBUG=-all \
  LD_PRELOAD="$SHIM" \
  FASTFAIL_LOG="$FASTFAIL_LOG" \
  FASTFAIL_PROCS="wgc.exe,CrUtilityMain,CrRendererMain,wineserver" \
  FASTFAIL_IPS="$IPS" \
  wine "$WGC"
