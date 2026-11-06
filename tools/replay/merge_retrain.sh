#!/usr/bin/env bash
# Post-recording pipeline: rebuild what is missing -> merge -> train -> validate.
#
# The failure modes this closes, all of which were possible before:
#   * the merge's exit code was never checked - only "the output file is not empty", and
#     a half-written ZIP is not empty;
#   * a session whose dataset could not be rebuilt was skipped silently, so the corpus
#     could quietly shrink and the training would happily fit the shrunken one;
#   * nothing recorded what the merge produced.
#
# Structure, and why: the merge phase runs as its own locked, recorded run; the lock is
# released when that phase returns, and only then does the driver start the training,
# which takes the lock for itself. Nesting them would deadlock, and running them
# together would break the one-heavy-job rule this machine needs.
#
# Usage: merge_retrain.sh [SESSIONS_TO_CLEAR...]
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

LOG=replays/parsed/merge-retrain.log
QUARANTINE=replays/parsed/minimap-rect-quarantine.txt
# Overridable so the refusal paths can be exercised in a sandbox instead of by planting
# fake sessions in the project's own directories.
RECORDINGS=${RECORDINGS_DIR:-recordings}
DATASETS=${DATASETS_DIR:-replays/parsed/datasets}
DATASETS_ROOT=${DATASETS}
NEXT=${NEXT_CORPUS_ENV:-state/next-corpus.env}
PY=./.venv/bin/python
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

# ---- driver ----------------------------------------------------------------------
if [ "${1:-}" != "--merge-phase" ]; then
  say "=== post-recording pipeline ==="
  AVAILABLE=$(free -g | awk 'NR==2 {print $7}')
  say "available ${AVAILABLE} GB"
  [ "${AVAILABLE:-0}" -lt 10 ] && { say "refusing: less than 10 GB available"; exit 1; }
  tools/replay/heavy_lock.sh run merge -- "$0" --merge-phase "$@"
  # The lock is free again here (heavy_lock.sh released it when the phase returned).
  [ -f "$NEXT" ] || { say "the merge phase left no corpus description; not training"; exit 1; }
  # shellcheck disable=SC1090
  set -a; . "$NEXT"; set +a
  say "=== merge verified; training on $H ==="
  # The training phase's OWN output goes into this log, including the shared lock's
  # refusal, which it prints to stderr. Without this redirect the driver's stdout is
  # wherever the caller put it -- a nohup with `>/dev/null` is the normal unattended
  # launch -- and a training that never starts (or dies on its first line) leaves a log
  # that ends at "training on ..." and says nothing about why. That is exactly how the
  # 2026-10-14 01:19 run looked: merge fine, then silence, and no train record at all.
  # EPOCHS is defaulted in its OWN statement, not as a prefix assignment: a prefix
  # assignment is placed in the CHILD's environment, so `EPOCHS=${EPOCHS:-30} exec ...
  # "$EPOCHS"` expands the argument from the parent's environment, where EPOCHS is unset
  # and `set -u` turns that into "EPOCHS: unbound variable" -- the driver aborted right
  # where it hands off. That is what the 2026-10-14 01:19 "merge fine then silence" run
  # really was, and it was still there at 12:56. H/I/M/J/STATE are already exported by
  # the `set -a` above.
  : "${EPOCHS:=30}"
  exec tools/replay/run_ops_training.sh "$EPOCHS" >> "$LOG" 2>&1
fi

# ---- merge phase (the caller already holds the heavy lock) -----------------------
shift
for session in "$@"; do
  if [ -f "$QUARANTINE" ] && grep -qx "$session" "$QUARANTINE"; then
    grep -vx "$session" "$QUARANTINE" > "$QUARANTINE.tmp" && mv "$QUARANTINE.tmp" "$QUARANTINE"
    say "cleared $session from the quarantine list (freshly re-recorded)"
  fi
done

sessions=()
for path in recordings/*/replay_path.txt; do
  [ -e "$path" ] || continue
  name=$(basename "$(dirname "$path")")
  case "$name" in _*) continue ;; esac
  sessions+=("$name")
done
say "${#sessions[@]} recorded session(s)"

failed_rebuilds=0
for name in "${sessions[@]}"; do
  if [ -e "$DATASETS/heatmap-${name}-g176-s1.0-c10.npz" ] && [ -e "$DATASETS/minimap-${name}.npz" ]; then
    continue
  fi
  grep -qx "$name" "$QUARANTINE" 2>/dev/null && continue
  say "  rebuilding $name"
  if ! "$PY" -u tools/replay/auto_learn.py --only "$name" --no-record \
        --skip-merge --skip-train >> "$LOG" 2>&1; then
    say "  REBUILD FAILED for $name"
    failed_rebuilds=$((failed_rebuilds + 1))
  fi
done
if [ "$failed_rebuilds" -gt 0 ]; then
  say "refusing to merge: $failed_rebuilds rebuild(s) failed, so the corpus would be short"
  exit 1
fi

PAIRS=(); names=()
for name in "${sessions[@]}"; do
  heat="$DATASETS/heatmap-${name}-g176-s1.0-c10.npz"
  image="$DATASETS/minimap-${name}.npz"
  if [ ! -e "$heat" ] || [ ! -e "$image" ]; then
    say "refusing to merge: no dataset pair for $name (a corpus that silently drops a session is worse than none)"
    exit 1
  fi
  if grep -qx "$name" "$QUARANTINE" 2>/dev/null; then
    say "  skipping $name (quarantined framing)"
    continue
  fi
  PAIRS+=(--pair "$heat:$image"); names+=("$name")
done
COUNT=${#names[@]}
say "merging $COUNT session(s)"
[ "$COUNT" -lt 5 ] && { say "refusing: a session-level split needs at least 5 sessions"; exit 1; }

OUT_H="$DATASETS/heatmap-merged-${COUNT}s-g176.npz"
OUT_I="$DATASETS/minimap-merged-${COUNT}s-g176.npz"
OUT_S="$DATASETS/heatmap-merged-${COUNT}s.summary.json"
if [ -e "$OUT_H" ] || [ -e "$OUT_I" ]; then
  say "refusing: $OUT_H or its image twin already exists"
  exit 1
fi

say "=== merge ==="
merge_run=$("$PY" tools/replay/run_record.py start merge --artifact "$OUT_H" --artifact "$OUT_I")
say "merge run $merge_run"
if ! "$PY" -u tools/replay/merge_heatmap_datasets.py "${PAIRS[@]}" \
      --summary replays/parsed/summary.jsonl \
      --out-heatmaps "$OUT_H" --out-images "$OUT_I" --out-summary "$OUT_S" \
      --holdout 2 --validation 2 >> "$LOG" 2>&1; then
  "$PY" tools/replay/run_record.py fail "$merge_run" --reason "merge exited non-zero" || true
  say "merge FAILED (non-zero exit); nothing else was started"
  exit 1
fi

# Independent check that nothing was dropped: the merge writes its own session list, so
# counting it does not trust the same code path that built the pairs.
MERGED=$("$PY" - "$OUT_H" <<'PYEOF'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as archive:
    print(len({str(value) for value in archive["session_ids"]}))
PYEOF
)
if [ "$MERGED" != "$COUNT" ]; then
  "$PY" tools/replay/run_record.py fail "$merge_run" \
      --reason "merged corpus holds $MERGED sessions, expected $COUNT" || true
  say "merge FAILED: output holds $MERGED sessions, $COUNT were given"
  exit 1
fi
if ! "$PY" tools/replay/run_record.py commit "$merge_run" \
      --artifact "$OUT_H" --artifact "$OUT_I" --artifact "$OUT_S"; then
  say "merge COMMIT failed (artifacts missing or unchanged); nothing else was started"
  exit 1
fi
say "merged $COUNT session(s): $(du -h "$OUT_H" | cut -f1) + $(du -h "$OUT_I" | cut -f1)"

{
  echo "COUNT=$COUNT"
  echo "H=$OUT_H"
  echo "I=$OUT_I"
  echo "M=replays/parsed/artifacts/minimap-heatmap-ops-${COUNT}s.pt"
  echo "J=replays/parsed/artifacts/minimap-heatmap-ops-${COUNT}s.json"
  echo "STATE=replays/parsed/artifacts/minimap-heatmap-ops-${COUNT}s.state_dict.pt"
} > "$NEXT"
say "wrote $NEXT for the training phase"
