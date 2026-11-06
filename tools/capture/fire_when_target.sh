#!/usr/bin/env bash
# Fire once, automatically, as soon as the detectors have an enemy under the reticle.
#
#   tools/capture/fire_when_target.sh [--minutes 12] [--interval 5] [--prefix name]
#
# Each attempt is a fresh, isolated run directory.  The consumer itself refuses
# to fire without a confirmed battle HUD, a readable reticle, a loaded gun and a
# trackable enemy inside the tolerance, so a refusal costs one frame read and no
# input.  The loop stops at the first shot: later evidence (health-bar fill,
# saved frames) decides whether it hit.
set -uo pipefail
cd "$(dirname "$0")/../.."

MINUTES=12
INTERVAL=5
PREFIX="fire-auto"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --minutes) MINUTES="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

echo "loop start: up to ${MINUTES} min, every ${INTERVAL}s"
DEADLINE=$(( $(date +%s) + MINUTES * 60 ))
i=0
while [[ $(date +%s) -lt $DEADLINE ]]; do
    i=$((i+1))
    OUT="state/${PREFIX}-$(date +%Y%m%dT%H%M%S)-$i"
    LINE=$(.venv/bin/python -B tools/capture/fire_at_target.py --execute --out "$OUT" >"/tmp/wws-fire.log" 2>&1; echo $?)
    RC=$LINE
    tail -1 /tmp/wws-fire.log || true
    if [[ $RC -eq 0 ]]; then
        echo "=== fired: $OUT"
        exit 0
    fi
    sleep "$INTERVAL"
done
echo "no shot within ${MINUTES} min"
exit 3
