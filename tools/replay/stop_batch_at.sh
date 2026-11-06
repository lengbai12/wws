#!/usr/bin/env bash
# Stop the recording batch at a wall-clock deadline, cleanly and resumably.
#
# Order matters, and each step is here for a measured reason:
#
#   1. The driver is signalled FIRST, so it cannot launch another session while the
#      machine is being shut down.
#   2. The GAME is signalled next, alone. The recorder's own inline watchdog sees
#      the client disappear and finalises the container, so the partial recording
#      keeps a readable sidecar instead of a torn one.
#   3. The recorder is allowed to finish. Its full verify decodes every frame of a
#      20-minute AV1 file, which takes minutes.
#   4. A recording left without session.json is moved aside. The recorder refuses
#      to write into a directory that already holds a container, so leaving it in
#      place would silently block that replay from ever being recorded again.
#   5. The reason is written to a marker file, so the health check reports "stopped
#      on purpose, resumable" instead of "the driver vanished".
#
# Steam is deliberately left running: it is not part of the recording, and stopping
# it here would only add a way for this script to fail.
#
# Usage:
#   stop_batch_at.sh 19:10 ["reason"]
set -uo pipefail

DEADLINE=${1:?usage: stop_batch_at.sh HH:MM [reason]}
REASON=${2:-"stop requested before the machine is shut down"}
cd "$(dirname "$0")/../.." || exit 1
WORKSPACE=$PWD
LOG=$WORKSPACE/state/stop-batch.log
MARKER=$WORKSPACE/replays/parsed/batch-stop.json
: > "$LOG"
echo "stop scheduled for $DEADLINE (now $(date '+%H:%M:%S'))" >> "$LOG"

TARGET=$(date -d "today $DEADLINE" +%s 2>/dev/null) || {
  echo "cannot parse the deadline $DEADLINE" >> "$LOG"; exit 2; }
while [ "$(date +%s)" -lt "$TARGET" ]; do sleep 20; done
echo "deadline reached at $(date '+%H:%M:%S')" >> "$LOG"

DRIVER=$(pgrep -f "auto_learn[.]py" | head -1 || true)
if [ -n "$DRIVER" ]; then
  kill -TERM "$DRIVER" && echo "signalled the driver $DRIVER" >> "$LOG"
else
  echo "no driver is running" >> "$LOG"
fi
sleep 3

GAME=$(pgrep -x WorldOfWarships | head -1 || true)
if [ -n "$GAME" ]; then
  kill -TERM "$GAME" && echo "signalled the game $GAME" >> "$LOG"
else
  echo "no game client is running" >> "$LOG"
fi

for _ in $(seq 1 90); do
  pgrep -f "gamescope_record[.]py" >/dev/null || break
  sleep 5
done
echo "recorder finalise finished at $(date '+%H:%M:%S')" >> "$LOG"

/usr/bin/python3 - "$WORKSPACE" "$DEADLINE" "$REASON" <<'PY' >> "$LOG" 2>&1
import json, shutil, sys, time
from pathlib import Path
workspace, deadline, reason = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
recordings = workspace / "recordings"
unfinalised = []
for directory in sorted(recordings.glob("*")):
    if not directory.is_dir() or directory.name.startswith("_"):
        continue
    if (directory / "capture.mkv").exists() and not (directory / "session.json").exists():
        target = recordings / ("_interrupted-" + directory.name)
        if target.exists():
            shutil.rmtree(target)
        directory.rename(target)
        unfinalised.append(directory.name)
        print(f"moved the unfinished recording {directory.name} to {target.name}")
replays = sorted((workspace / "replays").glob("*.wowsreplay"))
recorded = 0
for replay in replays:
    for directory in recordings.glob("*"):
        marker = directory / "replay_path.txt"
        if not marker.exists():
            continue
        if str(Path(marker.read_text().strip()).resolve()) == str(replay.resolve()) \
                and (directory / "session.json").exists():
            recorded += 1
            break
marker = workspace / "replays/parsed/batch-stop.json"
marker.write_text(json.dumps({
    "deadline": deadline,
    "stopped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "reason": reason,
    "unfinalised_recordings": unfinalised,
    "recorded": recorded,
    "pending": len(replays) - recorded,
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"recorded {recorded}/{len(replays)}; wrote {marker}")
PY

echo "DONE $(date '+%H:%M:%S')" >> "$LOG"
