#!/usr/bin/env bash
# Watch available memory while a heavy job runs, and stop it before the kernel does.
#
# Why this exists: the machine has 30 GB and no swap, and one heavy job was lost because
# nothing was watching. Checking memory only before a launch is not enough - the peak
# arrives minutes later.
#
# What was wrong with the first version, and is fixed here:
#   * it exited as soon as the guarded process was not found on one sample. A guard
#     started before its job (or one that runs beside a train -> validate handover) then
#     watched nothing at all. It is now bound to a RUN: it keeps sampling while the run
#     record says "running", regardless of transient process gaps, and stops when the run
#     reaches a terminal state.
#   * it only ever wrote "ALERT" lines, while the documentation promised a graceful stop
#     below 2 GB. Below the hard floor it now records why and sends SIGTERM to the run's
#     process - graceful on purpose: the training writes its best epoch as it improves,
#     so a stop costs at most a few epochs, while an OOM costs the whole run.
#
# Two measured profiles on this machine:
#   training a corpus model : RSS about 14 GB, of which about 6 GB is anonymous and the
#                             rest is reclaimable mmap of the corpus
#   streaming corpus merge  : peak RSS 0.24 to 0.27 GB
#
# and one anti-pattern that cost a run: materialising a whole corpus member with
# ``np.load(...)[member]`` allocates the entire array as anonymous memory (21 GB was
# measured) and gets OOM-killed. Always go through the mmap twin ``<npz stem>.<member>.npy``.
#
# Usage:
#   mem_watch.sh [INTERVAL_S] [MIN_FREE_GB] [PATTERN] [--run <run_id>] [--hard 2]
#
# Written to state/mem-watch.log, one line per sample, plus ALERT and STOP lines.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

INTERVAL=${1:-60}
MIN_FREE_GB=${2:-4}
GUARD=${3:-}
# The first three are positional; whatever is left is options. Shift one at a time, so a
# call with fewer positionals still keeps its options.
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift 
RUN_ID=""
HARD_GB=2
while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN_ID="${2:?--run needs an id}"; shift ;;
    --hard) HARD_GB="${2:?--hard needs a number}"; shift ;;
    *) ;;
  esac
  shift
done

LOG=${MEM_WATCH_LOG:-state/mem-watch.log}
PY=./.venv/bin/python
: > "$LOG"
echo "watch started $(date '+%F %T')  interval=${INTERVAL}s  alert below ${MIN_FREE_GB} GB  hard floor ${HARD_GB} GB  guard='${GUARD:-none}' run='${RUN_ID:-none}'" >> "$LOG"

run_status() {
  # The exit code is NOT the status: run_record.py status returns 1 for anything that is
  # not ok (by design), so `|| echo gone` would append a second line to a perfectly live
  # run's status and end the watch immediately. Read the output instead.
  [ -z "$RUN_ID" ] && return 0
  "$PY" tools/replay/run_record.py status "$RUN_ID" 2>/dev/null | head -1
}

stopped=0
while :; do
  # Columns by position: the label on this machine is localised, so matching "Mem:" finds
  # nothing and every reading comes out empty.
  read -r total used free shared buffcache available < <(free -g | awk 'NR==2 {print $2, $3, $4, $5, $6, $7}')

  guard_rss=""
  if [ -n "$GUARD" ]; then
    # The pattern matches this very script too (it is its own third argument) and any
    # launcher around the job, so exclude ourselves and take the largest RSS among what
    # is left: that is the one whose memory can get the machine killed.
    pid=$(for p in $(pgrep -f "$GUARD" 2>/dev/null || true); do
            [ "$p" = "$$" ] && continue
            [ -r "/proc/$p/status" ] || continue
            printf '%s %s\n' "$(awk '/^VmRSS:/ {print $2}' "/proc/$p/status" 2>/dev/null || echo 0)" "$p"
          done | sort -rn | head -1 | awk '{print $2}')
    if [ -n "$pid" ]; then
      guard_rss=$(awk '/^VmRSS:/ {r=$2} /^RssAnon:/ {a=$2}
                       END {printf " guard pid=%s rss=%.1fG anon=%.1fG", pid, r/1048576, a/1048576}' \
        pid="$pid" "/proc/$pid/status" 2>/dev/null)
    else
      guard_rss=" guard='$GUARD' not running"
    fi
  fi

  line="$(date '+%F %T') available=${available}G total=${total}G used=${used}G cache=${buffcache}G${guard_rss}"
  if [ "${available:-0}" -lt "$HARD_GB" ]; then
    echo "STOP $line  (below the hard floor of ${HARD_GB} GB)" >> "$LOG"
    stopped=1
    break
  elif [ "${available:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "ALERT $line" >> "$LOG"
  else
    echo "INFO $line" >> "$LOG"
  fi

  # The run is the lifecycle, not the process: a train -> validate handover, or a guard
  # started a moment before its job, must not end the watch.
  if [ -n "$RUN_ID" ]; then
    status=$(run_status)
    case "$status" in
      running) ;;
      # No record at all means the run is over (or was never recorded); either way there
      # is nothing left to watch, and spinning on it would be the very failure this guard
      # is supposed to prevent.
      "" | gone) echo "$(date '+%F %T') no run record for $RUN_ID; watch ends" >> "$LOG"; break ;;
      *) echo "$(date '+%F %T') the run is $status; watch ends" >> "$LOG"; break ;;
    esac
  elif [ -n "$GUARD" ] && ! pgrep -f "$GUARD" >/dev/null 2>&1; then
    echo "$(date '+%F %T') guard='$GUARD' not running and no run to follow; watch ends" >> "$LOG"
    break
  fi
  sleep "$INTERVAL"
done

if [ "$stopped" = "1" ]; then
  # Graceful on purpose: the training writes its best epoch as it improves, so at most a
  # few epochs are lost, while an OOM takes the whole run.
  if [ -n "$RUN_ID" ]; then
    pid=$("$PY" - "$RUN_ID" <<'PYEOF'
import json, sys
from pathlib import Path
try:
    print(json.loads(Path("state/runs/%s.json" % sys.argv[1]).read_text())["pid"])
except Exception:
    print("")
PYEOF
)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "$(date '+%F %T') stopping run $RUN_ID (pid $pid) gracefully: memory below ${HARD_GB} GB" >> "$LOG"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  fi
  echo "$(date '+%F %T') WATCH STOPPED: memory hit the hard floor" >> "$LOG"
fi
