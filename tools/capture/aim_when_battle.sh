#!/usr/bin/env bash
# Wait for a live battle with a visible enemy, then run one bounded aim run.
#
#   tools/capture/aim_when_battle.sh --training-room --max-range-km <range> [--hours 4]
# Supply the actual main battery range (>4 km); preparation caps selection at 15 km.
#
# A battle only counts when the HUD reports it and the nameplate detector sees
# a trackable enemy; a menu, a results page or an empty sea is not a battle.
# The aim run itself refuses to act while you are touching the keyboard, so it
# will step aside instead of fighting you for input.
set -euo pipefail
cd "$(dirname "$0")/../.."

HOURS=4
INTERVAL=20
PREFIX="aim-auto"
ONCE=0
TRAINING_ROOM=0
MAX_RANGE_KM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hours) HOURS="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --out-prefix) PREFIX="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        --training-room) TRAINING_ROOM=1; shift ;;
        --max-range-km) MAX_RANGE_KM="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ $ONCE -ne 1 ]]; then
    if [[ $TRAINING_ROOM -ne 1 ]]; then
        echo "--training-room attestation is required for automatic preparation" >&2
        exit 2
    fi
    .venv/bin/python -B -c 'import math,sys; value=float(sys.argv[1]); sys.exit(0 if math.isfinite(value) and value>4 else 2)' "$MAX_RANGE_KM" || {
        echo "--max-range-km must be finite and greater than 4" >&2; exit 2;
    }
fi

check_once() {
    # The predicate's exit status is data, not an error: under `set -e` a bare
    # non-zero exit here would abort the watcher after one look.
    local out
    if out=$(.venv/bin/python -B - <<'PY'
import sys
sys.path.insert(0, 'tools/capture'); sys.path.insert(0, 'tools/replay')
import numpy as np
from locked_capture import LockedCapture, require_fresh
from prepare_locked_aim import FireHudReader, view_state, current_centre, detect_nameplates

capture = None
try:
    capture = LockedCapture()
    pixels, meta = capture.acquire()
    frame = np.asarray(pixels)
    hud = FireHudReader().detect(frame)
    view = view_state(hud, frame)
    centre, _ = current_centre(hud, frame, view)
    cands = detect_nameplates(frame, scope_observed=view == 'scope')
    require_fresh(meta)
    ready = hud.get('battle', {}).get('active') is True and centre is not None and bool(cands)
    print(f"battle={hud.get('battle', {}).get('active')} view={view} "
          f"reticle={centre is not None} enemies={len(cands)}")
except Exception as error:
    print(f'no fresh pinned battle frame: {error}')
    ready = False
finally:
    if capture is not None:
        capture.close()
raise SystemExit(0 if ready else 3)
PY
); then
        echo "$out"
        return 0
    fi
    echo "$out"
    return 3
}

DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
STAMP=$(date +%Y%m%dT%H%M%S)
echo "watching for a battle until $(date -d "@$DEADLINE" +%H:%M:%S), every ${INTERVAL}s"

if [[ $ONCE -eq 1 ]]; then
    if check_once; then echo "=== a battle with a visible enemy is on screen"; exit 0; fi
    echo "=== not a battle with a visible enemy"; exit 3
fi

while [[ $(date +%s) -lt $DEADLINE ]]; do
    if check_once; then
        echo "=== battle with a visible enemy at $(date +%H:%M:%S); starting one aim run"
        OUT="state/${PREFIX}-${STAMP}"
        if [[ -e "$OUT" ]]; then OUT="${OUT}-$(date +%s)"; fi
        tools/capture/aim_live.sh "$OUT" --training-room --max-range-km "$MAX_RANGE_KM" --execute || { echo "aim run refused (see $OUT/report.json)"; exit 1; }
        echo "=== one bounded preparation done; no firing; stopping the watcher"
        exit 0
    fi
    sleep "$INTERVAL"
done
echo "no battle with a visible enemy within ${HOURS}h"
