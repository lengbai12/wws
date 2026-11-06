#!/usr/bin/env bash
# Start an Operations training run plus its two helpers, all detached from the shell.
#
#   the run         tools/replay/run_ops_training.sh
#   memory guard    tools/replay/mem_watch.sh    (samples available memory while it runs)
#   collector       tools/replay/collect_after_training.sh (waits, then frees ~4.9 GB)
#
# One heavy job at a time: this refuses to start if a training is already running.
# Usage: start_ops_run.sh [EPOCHS]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

if pgrep -f "train_minimap_heatmap[.]py" >/dev/null 2>&1; then
  echo "refusing: a training is already running (pid $(pgrep -f 'train_minimap_heatmap[.]py' | tr '\n' ' '))"
  exit 1
fi
if pgrep -x WorldOfWarships >/dev/null 2>&1; then
  echo "refusing: the game is running; the game and training never coexist on 30 GB"
  exit 1
fi

EPOCHS=${1:-30}
AVAILABLE=$(free -g | awk 'NR==2 {print $7}')
echo "available ${AVAILABLE} GB"
if [ "${AVAILABLE:-0}" -lt 12 ]; then
  echo "refusing: available memory is below the 12 GB the run needs"
  exit 1
fi

setsid nohup tools/replay/run_ops_training.sh "$EPOCHS" \
  > replays/parsed/ops-train.log 2>&1 </dev/null &
setsid nohup tools/replay/mem_watch.sh 30 4 "train_minimap_heatmap[.]py" \
  > /dev/null 2>&1 </dev/null &
setsid nohup tools/replay/collect_after_training.sh \
  > /dev/null 2>&1 </dev/null &
sleep 3
# Match on the script name only: pgrep -f also matches this launcher, whose own
# command line contains every pattern.
report() { pgrep -f "tools/replay/$1" 2>/dev/null | grep -v "^$$\$" | tr '\n' ' '; }
echo "started: run=$(report run_ops_training) guard=$(report mem_watch) "\
"collector=$(report collect_after_training)"
