#!/usr/bin/env bash
# Keep the moving-target loop alive across battles.
#
# The training room is already configured (one movable, unarmed bot), so a
# battle only needs "已准备好" then "加入战斗". Coordinates are for this
# 1920x1080 session; the loop verifies with the HUD before every click.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

OUT=${1:-state/dynamic-night}
HOURS=${2:-8}
DISPLAY_ID=${DISPLAY_ID:-:3}
STOP=state/stop-dynamic.txt
LOG="$OUT/autoplay.log"
mkdir -p "$OUT"; : > "$LOG"
export PYTHONPATH=tools/capture:tools/replay

log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

in_battle() {
  .venv/bin/python -B - "$@" <<'PY' 2>/dev/null
import subprocess, sys, numpy as np
from PIL import Image
import fire_hud
subprocess.run(['/usr/bin/python3','tools/capture/capture_host.py','snapshot','/tmp/autoplay.png'],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
im=np.asarray(Image.open('/tmp/autoplay.png').convert('RGB'))
r=fire_hud.FireHudReader().detect(im)
raise SystemExit(0 if (r.get('battle') or {}).get('active') is True else 1)
PY
}

click() { DISPLAY=$DISPLAY_ID xdotool mousemove "$1" "$2"; sleep 0.5; DISPLAY=$DISPLAY_ID xdotool click 1; }

ready_and_join() {
  # The room persists; the only two states seen here are the battle result page
  # ("回到港口" bottom centre) and the port with the room panel on the right.
  log "starting a battle: 回到港口 -> 已准备好 -> 加入战斗"
  click 960 995; sleep 8.0
  click 719 615; sleep 2.0
  click 960 52;  sleep 6.0
}

END=$(( $(date +%s) + HOURS*3600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  if [ -f "$STOP" ]; then log "stop file present; exiting"; break; fi
  if in_battle; then break; fi
  ready_and_join
  for _ in $(seq 1 24); do
    sleep 5
    if in_battle; then log "battle is up"; break 2; fi
    if [ -f "$STOP" ]; then break 2; fi
  done
  log "still not in battle after ~2 min; retrying"
  sleep 20
done

while [ "$(date +%s)" -lt "$END" ]; do
  if [ -f "$STOP" ]; then log "stop file present; exiting"; break; fi
  if ! in_battle; then
    log "battle ended; waiting for the next one"
    ready_and_join
    for _ in $(seq 1 24); do
      sleep 5
      if in_battle; then log "battle is up"; break; fi
      if [ -f "$STOP" ]; then break 2; fi
    done
    continue
  fi
  log "running the shooter loop"
  .venv/bin/python -u tools/capture/dynamic_session.py --out "$OUT" \
      --seconds 1500 --max-attempts 60 --interval 1.5 >>"$LOG" 2>&1
  log "shooter loop returned; battle state will be re-checked"
  sleep 5
done
log "autoplay finished"
