#!/usr/bin/env bash
# Press the minimap enlarge key so every recording uses the LARGEST minimap.
#
# The minimap size is in-game UI state, not a constant: the + and - keys change
# it, and it survives from one recording to the next only by luck. It has already
# differed silently once: recordings from 10-10 have a panel 46% of the size the
# HUD calibration assumes, while later ones have the full size. Every label is
# cropped with that rectangle, so the size has to be made deterministic instead of
# relied on.
#
# The keys go to the game's own X display. The game is a Proton client on the
# gamescope Xwayland server, and that server's DISPLAY is readable from the game
# process environment, which is how the press reaches the right window.
#
# Presses beyond the maximum are ignored by the game, so pressing more than enough
# is safe: too few would leave the panel smaller than the calibration.
#
# Usage:
#   force_minimap_max.sh                  # one burst (12 presses)
#   force_minimap_max.sh PRESSES [INTERVAL_S]
#   force_minimap_max.sh --follow [WINDOW_S] [PRESSES] [INTERVAL_S]
#
# --follow is for a recording: the moment the battle HUD appears is not fixed, and
# a press before it is ignored, so the burst is repeated across the start-up window
# and stops as soon as the client is gone.
set -euo pipefail

GAME_COMM=WorldOfWarships
MODE=burst
if [ "${1:-}" = "--follow" ]; then
  MODE=follow
  shift
fi
WINDOW=${WINDOW_S:-300}
PRESSES=${1:-12}
INTERVAL=${2:-1.0}

command -v xdotool >/dev/null || { echo "xdotool is not installed" >&2; exit 3; }

game_pid() {
  # comm is matched exactly: a pattern match would find this script's own command.
  pgrep -x "$GAME_COMM" | head -1 || true
}

send_burst() {
  local count=$1 step=${2:-$INTERVAL} pid display_value window
  pid=$(game_pid)
  [ -z "$pid" ] && return 2
  display_value=$(tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
    | awk -F= '$1 == "DISPLAY" {print $2}')
  if [ -z "$display_value" ]; then
    echo "the game has no DISPLAY: this build renders through Wayland only, so a" >&2
    echo "key cannot be delivered with xdotool" >&2
    return 4
  fi
  export DISPLAY="$display_value"
  window=$(xdotool search --onlyvisible --name "." 2>/dev/null | head -1 || true)
  if [ -n "$window" ]; then
    # Best effort: the gamescope server usually holds one fullscreen client, but
    # focus is not guaranteed while the client is still loading.
    xdotool windowactivate --sync "$window" 2>/dev/null || true
  fi
  echo "game pid $pid, DISPLAY $display_value, window ${window:-unknown}"
  for _ in $(seq 1 "$count"); do
    # KP_Add is the numeric keypad plus: it needs no modifier, so it is the
    # reliable form of the key this game binds to "enlarge the minimap".
    xdotool key --clearmodifiers KP_Add 2>/dev/null || true
    xdotool key --clearmodifiers plus 2>/dev/null || true
    sleep "$step"
  done
  echo "sent $count enlarge presses"
}

if [ "$MODE" = burst ]; then
  send_burst "$PRESSES"
  exit $?
fi

# --follow: repeat across the start-up window. A fresh client starts with a SMALL
# minimap (measured: the panel's top border sits at frame row 887 instead of the
# calibrated 795, and pressing + moves it to 795), so the size has to be re-set on
# every replay, and it has to happen as soon as the battle HUD exists because the
# recorder is already running by then. A tight cadence keeps the frames that still
# show the small panel down to about a second.
# The client may not exist yet when this starts. The Steam route launches through
# the client, and although steam_session_ctl.sh 'launch' waits for the process, a
# race here is silent and total: the burst never happens, the panel stays at the
# client's default size, the recorder still reports success, and the whole
# recording is cropped from the wrong rectangle. That is how the 2026-10-14
# 236925 recording failed while its minimap-max.log stayed empty. So wait for the
# client instead of exiting.
WAIT_DEADLINE=$(( $(date +%s) + WINDOW ))
while [ -z "$(game_pid)" ]; do
  if [ "$(date +%s)" -ge "$WAIT_DEADLINE" ]; then
    echo "no $GAME_COMM process appeared within ${WINDOW}s: nothing to send keys to" >&2
    exit 2
  fi
  sleep 1
done
PID=$(game_pid)
DEADLINE=$(( $(date +%s) + WINDOW ))
echo "following the client for up to ${WINDOW}s, bursting every ${INTERVAL}s"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if [ -z "$(game_pid)" ]; then
    echo "the client is gone; stopping"
    exit 0
  fi
  send_burst 3 0.2 || true
  sleep "$INTERVAL"
done
echo "window elapsed; stopping"
