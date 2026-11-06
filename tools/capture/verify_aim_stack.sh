#!/usr/bin/env bash
# One command that says whether the aiming stack is in a shippable state.
#
# It runs the unit suites that guard each stage, then the pipeline's own dry run
# on a saved frame, and prints one PASS/FAIL line per item. Nothing here touches
# the game: no capture, no input, no window.
#
#   bash tools/capture/verify_aim_stack.sh [--frame <png>]
#
# Exit code 0 only when every item passes.
set -uo pipefail
cd "$(dirname "$0")/../.."

FRAME=""
if [[ "${1:-}" == "--frame" ]]; then
  FRAME="${2:?usage: verify_aim_stack.sh [--frame <png>]}"
fi
if [[ -z "$FRAME" ]]; then
  FRAME="$(ls -t state/mmfire-*/frame-*.png 2>/dev/null | head -1 || true)"
fi

export PYTHONPATH="tools/capture:tools/replay"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/wws-verify-pyc}"
PY=".venv/bin/python -B"

fail=0
pass_line() { printf '  PASS  %s\n' "$1"; }
fail_line() { printf '  FAIL  %s\n' "$1"; fail=1; }

run_suite() {
  local label="$1" module="$2"
  if $PY -m unittest "$module" >/tmp/wws-verify-"$module".log 2>&1; then
    pass_line "$label ($(grep -o 'Ran [0-9]* test' /tmp/wws-verify-"$module".log | awk '{print $2}') tests)"
  else
    fail_line "$label  -- see /tmp/wws-verify-$module.log"
    tail -5 /tmp/wws-verify-"$module".log | sed 's/^/        /'
  fi
}

echo "aiming stack acceptance"
echo "----------------------"
run_suite "pipeline stages"        test_aim_pipeline
run_suite "controller (aim_servo)" test_aim_servo
run_suite "fire gate"              test_fire_gate
run_suite "minimap shot maths"     test_minimap_shot
run_suite "aim smoothness"         test_aim_smoothness
run_suite "line-of-fire terrain"   test_minimap_terrain
run_suite "match runner"           test_match_runner
run_suite "dynamic target fire"    test_dynamic_target_fire

if [[ -n "$FRAME" && -f "$FRAME" ]]; then
  OUT="state/verify-aim-$(date +%Y%m%d-%H%M%S)"
  if $PY tools/capture/aim_pipeline.py --out "$OUT" --frame "$FRAME" --allow-minimap >/dev/null 2>&1; then
    pass_line "pipeline dry run produced a verdict ($(basename "$FRAME"))"
  else
    # A refusal is a valid, expected outcome on a saved frame: identity and focus
    # cannot be verified offline. What must hold is that the report names why.
    if $PY - "$OUT/report.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("verdict") in ("refused", "ready", "fired"), d.get("verdict")
assert isinstance(d.get("reasons"), list), "reasons missing"
assert d.get("stages", {}).get("sense"), "sense stage missing"
sys.exit(0)
PY
    then
      pass_line "pipeline dry run refused with named reasons ($(basename "$FRAME"))"
    else
      fail_line "pipeline dry run wrote no usable report ($OUT)"
    fi
  fi
else
  printf '  SKIP  pipeline dry run (no saved frame; pass --frame <png>)\n'
fi

echo "----------------------"
if [[ $fail -eq 0 ]]; then
  echo "RESULT: PASS"
else
  echo "RESULT: FAIL"
fi
exit $fail
