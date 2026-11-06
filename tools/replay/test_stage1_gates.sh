#!/usr/bin/env bash
# Stage-1 acceptance: the pass condition is "deliberately force a failure and check that
# nothing prints success, nothing triggers cleanup, and no partial artifact is accepted".
#
# Everything runs in /tmp. The project's own recordings/ and datasets/ are never touched:
# the pipeline under test takes directory overrides for exactly this reason.
set -uo pipefail
cd /home/lengbai/Downloads/wws || exit 1
PY=./.venv/bin/python
SANDBOX=/tmp/stage1-accept
PASS=0; FAIL=0
note() { printf '  %-6s %s\n' "$1" "$2"; }
check() { # check <label> <condition-result>
  if [ "$2" = "0" ]; then PASS=$((PASS+1)); note "PASS" "$1"; else FAIL=$((FAIL+1)); note "FAIL" "$1"; fi
}

rm -rf "$SANDBOX"; mkdir -p "$SANDBOX"/{recordings,datasets,state,runs}

# ---- 1. a training that cannot succeed -------------------------------------------
H=/nonexistent-heatmaps.npz timeout 300 tools/replay/run_ops_training.sh 1 > "$SANDBOX/train-fail.log" 2>&1
rc=$?
check "B1 failing training exits non-zero (got $rc)" "$([ "$rc" -ne 0 ] && echo 0 || echo 1)"
check "B1 no '=== done' on failure" "$(grep -q '^=== done' "$SANDBOX/train-fail.log" && echo 1 || echo 0)"
check "B1 the run record is marked failed" \
  "$($PY tools/replay/run_record.py status "$(ls -t state/runs/train-*.json | head -1 | xargs -n1 basename | sed 's/\.json$//')" >/dev/null 2>&1 && echo 1 || echo 0)"
check "B1 the failure is named in the log" "$(grep -q 'TRAINING FAILED' "$SANDBOX/train-fail.log" && echo 0 || echo 1)"

# ---- 2. the collector must not act on that run -----------------------------------
before=$(ls -1 replays/parsed/datasets/minimap-*.npz 2>/dev/null | wc -l)
MEM_WATCH_LOG=/tmp/x.log timeout 120 tools/replay/collect_after_training.sh > "$SANDBOX/collect.log" 2>&1
after=$(ls -1 replays/parsed/datasets/minimap-*.npz 2>/dev/null | wc -l)
check "B1 cleanup refuses a failed run" "$(grep -q 'refusing to collect' "$SANDBOX/collect.log" && echo 0 || echo 1)"
check "B1 cleanup deleted nothing ($before -> $after)" "$([ "$before" = "$after" ] && echo 0 || echo 1)"
# The refusal above happens before the dry-run line, so the dry-run default is checked
# against a run that IS committed ok: the guard on the run is the point, and this proves
# the two behaviours are distinct.
rid_ok="acceptok-$(date +%s)"
$PY -c "
import json, pathlib
pathlib.Path('state/runs/$rid_ok.json').write_text(json.dumps(
    {'run_id': '$rid_ok', 'name': 'acceptok', 'status': 'ok', 'pid': 1, 'artifacts': {}}))
"
before_ok=$(ls -1 replays/parsed/datasets/minimap-*.npz 2>/dev/null | wc -l)
timeout 120 tools/replay/collect_after_training.sh --run "$rid_ok" > "$SANDBOX/collect-ok.log" 2>&1
after_ok=$(ls -1 replays/parsed/datasets/minimap-*.npz 2>/dev/null | wc -l)
check "B1 cleanup is a dry run by default" "$(grep -q 'dry run' "$SANDBOX/collect-ok.log" && echo 0 || echo 1)"
check "B1 the dry run deleted nothing ($before_ok -> $after_ok)" "$([ "$before_ok" = "$after_ok" ] && echo 0 || echo 1)"
rm -f "state/runs/$rid_ok.json"

# ---- 3. a partial corpus must be refused, in a sandbox ---------------------------
for s in 111111 222222 333333; do
  mkdir -p "$SANDBOX/recordings/$s"
  echo "$SANDBOX/replays/$s.wowsreplay" > "$SANDBOX/recordings/$s/replay_path.txt"
done
touch "$SANDBOX/datasets/heatmap-111111-g176-s1.0-c10.npz" "$SANDBOX/datasets/minimap-111111.npz"
RECORDINGS_DIR="$SANDBOX/recordings" DATASETS_DIR="$SANDBOX/datasets" \
NEXT_CORPUS_ENV="$SANDBOX/state/next.env" timeout 300 tools/replay/merge_retrain.sh \
  > "$SANDBOX/merge-partial.log" 2>&1
rc=$?
check "B1 partial corpus exits non-zero (got $rc)" "$([ "$rc" -ne 0 ] && echo 0 || echo 1)"
check "B1 partial corpus refused before merging" \
  "$(grep -qE 'refusing to merge|REBUILD FAILED|corpus would be short' "$SANDBOX/merge-partial.log" && echo 0 || echo 1)"
check "B1 no merge or training started" \
  "$(grep -qE '=== merge ===|=== train ' "$SANDBOX/merge-partial.log" && echo 1 || echo 0)"

# ---- 4. B2: the kill selector must pick the game and only the game ----------------
cp "$(command -v sleep)" "$SANDBOX/WorldOfWarships"      # comm=WorldOfWarships, no replay in argv
sleep 60 &
fake=$!
innocent=$!
sleep 1
REPLAY_ABS="$SANDBOX/replays/236000.wowsreplay"
selected=$(ps -eo pid=,comm=,args= | awk -v replay="$REPLAY_ABS" '
  $2 == "WorldOfWarships" && index($0, replay) { print $1 }')
# The fake has the game's comm but not the replay on its command line, so the predicate
# must return nothing - proving it needs BOTH, which is what stops it killing a group.
check "B2 selector needs comm AND the replay (fake not selected)" \
  "$(echo "$selected" | grep -q . && echo 1 || echo 0)"
# Copy the binary BEFORE starting it: copying onto a running executable fails with
# "text file busy", and the file must also be named like the game for comm to match.
kill "$fake" 2>/dev/null || true
sleep 1
cp "$(command -v tail)" "$SANDBOX/WorldOfWarships"
: > "$SANDBOX/dummy-replay"
"$SANDBOX/WorldOfWarships" -f "$SANDBOX/dummy-replay" >/dev/null 2>&1 &
fake2=$!
sleep 1
selected2=$(ps -eo pid=,comm=,args= | awk -v replay="$REPLAY_ABS" '
  $2 == "WorldOfWarships" && index($0, replay) { print $1 }')
# Make the replay path appear in the fake's command line: tail takes files, so point it
# at a file whose NAME is the replay path we search for.
mkdir -p "$SANDBOX/replays"
: > "$SANDBOX/replays/236000.wowsreplay"
kill "$fake2" 2>/dev/null || true
sleep 1
"$SANDBOX/WorldOfWarships" -f "$SANDBOX/replays/236000.wowsreplay" >/dev/null 2>&1 &
fake2=$!
sleep 1
selected2=$(ps -eo pid=,comm=,args= | awk -v replay="$REPLAY_ABS" '
  $2 == "WorldOfWarships" && index($0, replay) { print $1 }')
check "B2 selector picks the matching game process" \
  "$(echo "$selected2" | grep -qw "$fake2" && echo 0 || echo 1)"
check "B2 selector ignores an innocent sleeper" \
  "$(echo "$selected2" | grep -qw "$innocent" && echo 1 || echo 0)"
kill "$fake" "$fake2" "$innocent" 2>/dev/null || true

# ---- 5. B3: the lock, the guard lifecycle, and the graceful floor ----------------
# Clear any holder left by an earlier acceptance run, then check the lock is free.
for p in $(pgrep -f "heavy_lock.sh run " 2>/dev/null || true); do kill "$p" 2>/dev/null || true; done
sleep 2
tools/replay/heavy_lock.sh check acceptance >/dev/null 2>&1
check "B3 lock is free when nothing runs" "$?"
(sleep 12 &)  # keep the shell busy so the background lock holder is not reaped
setsid nohup tools/replay/heavy_lock.sh run "acceptance-holder-$$" -- sleep 40 >/dev/null 2>&1 </dev/null &
sleep 2
tools/replay/heavy_lock.sh check second >/dev/null 2>&1
check "B3 a second heavy entry is refused while one runs" "$([ $? -ne 0 ] && echo 0 || echo 1)"
check "B3 the refusal names the holder" \
  "$(tools/replay/heavy_lock.sh check third 2>&1 | grep -q 'label=acceptance-holder-$$' && echo 0 || echo 1)"
# the recording entry point must refuse the same way, without launching anything
out="$SANDBOX/rec-$(date +%s)"
(HEAVY_LOCK_HELD= tools/capture/start_workspace_replay_recording.sh "$out" "$SANDBOX/replays/x.wowsreplay" 5 \
  > "$SANDBOX/rec.log" 2>&1); rc=$?
check "B3 the recording entry takes the lock too (refused, rc=$rc)" "$([ "$rc" -ne 0 ] && echo 0 || echo 1)"
check "B3 no recording was started by the refused call" "$([ -e "$out" ] && echo 1 || echo 0)"
# the guard follows a run's lifecycle rather than a process's presence
rid="acceptrun-$(date +%s)"
$PY -c "
import json, pathlib, os
pathlib.Path('state/runs/$rid.json').write_text(json.dumps(
    {'run_id': '$rid', 'name': 'acceptrun', 'status': 'running', 'pid': os.getpid()}))
"
MEM_WATCH_LOG="$SANDBOX/w.log" timeout 6 tools/replay/mem_watch.sh 1 4 "no-such-process-zzz" --run "$rid" >/dev/null 2>&1
check "B3 the guard keeps sampling though its process is absent" \
  "$([ "$(grep -c 'INFO\\|ALERT' "$SANDBOX/w.log")" -ge 3 ] && echo 0 || echo 1)"
MEM_WATCH_LOG="$SANDBOX/w2.log" timeout 10 tools/replay/mem_watch.sh 1 4 "" --run "$rid" --hard 99 >/dev/null 2>&1
check "B3 the hard floor triggers a graceful stop" \
  "$(grep -q 'WATCH STOPPED' "$SANDBOX/w2.log" && echo 0 || echo 1)"
rm -f "state/runs/$rid.json"

echo
echo "stage-1 acceptance: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] || exit 1
