#!/usr/bin/env bash
# The operator's firing procedure, start to first shot.
#
#   1. point the camera at the nearest contact the minimap model reports
#      (one 2 s read; a 360-degree frame-by-frame sweep cannot see a 3-degree
#      wide nameplate inside the 15 s budget);
#   2. confirmation sweep: a ~90-degree arc, in case the bearing was stale;
#   3. coarse aim: nameplate vs triangular reticle in yaw, target range vs
#      impact range within 2 km in elevation;
#   4. image-confirmed scope; an enemy-free scope closes and hands back to 3;
#   5. lead from the target's fitted motion, then one bounded trigger.
# The whole run from the first search frame to the shot is capped at 15 s; a
# camera cut or a vertical sight that keeps flipping is a failure, never a pass.
# Default dry-run: --execute moves the mouse, --fire requests the trigger.
# --ensure-minimap-max presses six keys with a 1 s settle (~7 s). Measured
# 2026-10-23: that startup cost pushed the first frame past a nameplate that
# was only briefly drawn, so the run refused with 'no enemy nameplate in
# view' right after a detection had found one. Set WW_MINIMAP_MAX=1 when the
# minimap is not already at its maximum size.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:?usage: aim_live.sh <out-dir> [extra args]}"; shift || true
exec .venv/bin/python -B tools/capture/aim_servo.py \
    --out "$OUT" --seconds 118 --max-steps 40 \
    --scan --scan-steps ${WW_SCAN_STEPS:-14} --scan-step 400 --scan-confirm-steps 3 --scan-rows 1 --scan-tilt 3 --scan-min-deg 0 \
    --aim-budget-s ${WW_AIM_BUDGET_S:-30} --search-budget-s ${WW_SEARCH_BUDGET_S:-25} \
    --scope --scope-settle 0.8 --scope-empty-grace 5 --scope-probe 3 --range-closure --scope-range-gap-km 2 \
    --settle 0.3 --beyond-max-steps 6 --seed-gain -1.5 -1.4 --reacquire-px 0 --recenter-margin 140 \
    "$@"
