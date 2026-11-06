#!/usr/bin/env bash
# The one lock every heavy job takes, so a second one cannot start beside it.
#
# Why this exists: the memory rule on this machine is "one heavy job at a time" (30 GB,
# no swap; two overlapping attempts were OOM-killed within half a minute), but each
# entry point did its own ad-hoc check - start_ops_run.sh looked for a training, merge_
# retrain.sh looked for nothing, and a direct run_ops_training.sh bypassed both. A lock
# is the only way that holds across entry points.
#
# The game is NOT covered by the lock (it is started by Steam, not by us), so that check
# stays separate and is done here too, for every caller.
#
# Usage:
#   heavy_lock.sh run <label> -- <command...>   # hold the lock while the command runs
#   heavy_lock.sh check <label>                 # exit 0 if free, 1 if held or the game runs
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

LOCK=state/heavy.lock
mkdir -p state

check() {
  local label="$1"
  if pgrep -x WorldOfWarships >/dev/null 2>&1; then
    echo "heavy_lock($label): refused, the game is running" >&2
    return 1
  fi
  if [ -e "$LOCK" ] && ! flock -n "$LOCK" true 2>/dev/null; then
    echo "heavy_lock($label): refused, another heavy job holds the lock:" >&2
    sed 's/^/    /' "$LOCK" >&2
    return 1
  fi
  return 0
}

write_owner() {
  local label="$1"
  {
    printf 'label=%s\n' "$label"
    printf 'pid=%s\n' "$$"
    printf 'since=%s\n' "$(date '+%F %T')"
    printf 'cmd=%s\n' "$*"
  } > "$LOCK.owner"
}

case "${1:-}" in
  check)
    check "${2:-unnamed}"
    exit $?
    ;;
  run)
    label="${2:?usage: heavy_lock.sh run <label> -- <command...>}"
    shift 2
    [ "${1:-}" = "--" ] && shift
    check "$label" || exit 3
    # fd 9 holds the lock for as long as this shell lives: a caller that dies still
    # releases it, and there is no quoting to get wrong. The owner file is only so a
    # human can see who held it last.
    exec 9>"$LOCK" || exit 4
    flock -n 9 || { echo "heavy_lock($label): lost the race for the lock" >&2; exit 3; }
    write_owner "$label" "$*"
    cat "$LOCK.owner" > "$LOCK"
    "$@"
    exit $?
    ;;
  *)
    sed -n '2,16p' "$0"
    exit 2
    ;;
esac
