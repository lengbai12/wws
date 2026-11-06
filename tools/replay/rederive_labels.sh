#!/usr/bin/env bash
# Re-derive the per-session label chain for the whole queue.
#
# Why stage by stage, and never `--force "*"`:
#   * `*` also forces the RECORD stage (auto_learn.py tests `args.force not in
#     ("record", "*")`), which would launch the game in the middle of a labelling run;
#   * `*` also forces `align`, which re-derives the anchors from the client's
#     profile/python.log -- a log that rotates. A session whose `[BattleReplay] play`
#     line has aged out then fails, and that is exactly how 236066 was lost. The
#     exports and alignments on disk are still valid; only the two stages that READ
#     them changed (the vehicle-classification order and the validity mask).
#
# The two stages are forced because auto_learn skips a stage whenever its output file
# exists, so a fix to a derivation does not reach an existing corpus by itself.
#
# Usage: rederive_labels.sh [--dry-run]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

LOG=replays/parsed/rederive-labels.log
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

AVAILABLE=$(free -g | awk 'NR==2 {print $7}')
say "=== re-derive start; available ${AVAILABLE} GB ==="
if [ "${AVAILABLE:-0}" -lt 10 ]; then
  say "refusing: less than 10 GB available"
  exit 1
fi
if pgrep -x WorldOfWarships >/dev/null 2>&1; then
  say "refusing: the game is running"
  exit 1
fi

for stage in inventory heatmap; do
  if [ "$DRY" = 1 ]; then
    say "dry run: would force $stage"
    continue
  fi
  say "=== force $stage ==="
  if ! ./.venv/bin/python -u tools/replay/auto_learn.py --force "$stage" \
        --skip-train --skip-merge >> "$LOG" 2>&1; then
    say "=== $stage FAILED (see the log above) ==="
    exit 1
  fi
  say "=== $stage done ==="
done
say "=== re-derive done; available $(free -g | awk 'NR==2 {print $7}') GB ==="
