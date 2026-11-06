#!/bin/bash
# Record and process the newly downloaded replays, ONE SESSION AT A TIME.
#
# One invocation of auto_learn.py per session keeps `sessions ready` below three, so
# the merge stage never runs — the merge is currently the memory bottleneck (5
# sessions are 14.7 GB uncompressed and the machine has no swap), so running them
# separately is what makes an unattended night safe.
#
# The recording itself is delegated to auto_learn's record stage, which calls the
# project's own capture script (tools/capture/start_workspace_replay_recording.sh,
# which wraps the verified launch_wows_replay.sh and its stop-the-game-only cleanup).
# Nothing here touches Wine directly, and no session that was already learned is
# re-recorded.
cd /home/lengbai/Downloads/wws || exit 1
LOG=replays/parsed/auto-learn-night5.log
SESSIONS="236814 236815 236832 236834 236835 236841"
for s in $SESSIONS; do
    echo "=== $s start $(date '+%m-%d %H:%M:%S')" >> "$LOG"
    ./.venv/bin/python tools/replay/auto_learn.py --only "$s" >> "$LOG" 2>&1
    echo "=== $s done  $(date '+%m-%d %H:%M:%S')" >> "$LOG"
done
echo "=== ALL DONE $(date '+%m-%d %H:%M:%S')" >> "$LOG"
