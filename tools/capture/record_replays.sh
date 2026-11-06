#!/usr/bin/env bash
# Record several replays back to back, one workspace directory per replay.
#
# Each replay gets its own directory, the recording stops when that replay's
# game process exits, and the next replay starts only after the recorder has
# finalised its container. An existing directory is never overwritten, so a
# failed attempt stays available for diagnosis.
#
# Usage:
#   record_replays.sh OUTPUT_ROOT REPLAY [REPLAY ...]
#
# Example:
#   record_replays.sh recordings replays/236066_random_Kurama_XI.wowsreplay \
#       replays/236188_random_Harugumo_X.wowsreplay
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 OUTPUT_ROOT REPLAY [REPLAY ...]" >&2
  exit 2
fi

OUT_ROOT="$1"
shift
case "$OUT_ROOT" in
  /tmp|/tmp/*) echo "refusing to put recordings under /tmp" >&2; exit 2 ;;
esac
SCRIPT_DIR=$(dirname "$0")
mkdir -p "$OUT_ROOT"

# The direct-Proton route needs a RUNNING, LOGGED-IN Steam client. WoWS
# initialises Steamworks through ~/.steam/steam.pipe, and without it the game
# exits within a second without writing its log: 4 frames land in the container
# and nothing else reports a reason. The check is here because that failure is
# silent and costs a replay slot; comm is checked, never a command-line pattern.
if ! ps -eo comm= | grep -qx "steam"; then
  echo "no Steam client is running: the direct route needs one for Steamworks," >&2
  echo "otherwise the game exits within a second and the recording is empty." >&2
  echo "Start it with: setsid nohup /usr/bin/steam >/dev/null 2>&1 &" >&2
  echo "and confirm the login in ~/.local/share/Steam/logs/connection_log.txt." >&2
  exit 1
fi

# A full frame count makes ffprobe decode the whole file, which costs about 90 s
# on a 20-minute recording. It is the default because it is the corpus standard:
# session.json then states that the container frame count equals the sidecar, so
# the frame-to-time mapping is known sound instead of assumed. Set WW_VERIFY=fast
# to trade that evidence for speed.
export WW_VERIFY="${WW_VERIFY:-full}"
echo "verify mode: $WW_VERIFY (set WW_VERIFY=fast to skip the frame-count check)"

failed=0
recorded=0
for replay in "$@"; do
  if [ ! -f "$replay" ]; then
    echo "skip: replay does not exist: $replay" >&2
    failed=$((failed + 1))
    continue
  fi
  name=$(basename "$replay" .wowsreplay)
  out="$OUT_ROOT/$name"
  if [ -e "$out/capture.mkv" ] || [ -e "$out/frames.jsonl" ] || [ -e "$out/session.json" ]; then
    echo "skip: recording already exists: $out" >&2
    continue
  fi

  echo "=== recording $replay -> $out"
  if ! bash "$SCRIPT_DIR/start_workspace_replay_recording.sh" "$out" "$replay" until-exit; then
    echo "start failed for $replay" >&2
    failed=$((failed + 1))
    continue
  fi

  recorder=$(cat "$out/recorder.pid" 2>/dev/null || true)
  if [ -z "$recorder" ]; then
    echo "no recorder pid written for $out" >&2
    failed=$((failed + 1))
    continue
  fi
  # The watchdog stops the recorder when the game exits; wait for that here so
  # the next launch cannot collide with a prefix still shutting down.
  while kill -0 "$recorder" 2>/dev/null; do sleep 5; done
  if [ -f "$out/session.json" ]; then
    echo "=== finished $name: $(grep -o '"frames": [0-9]*' "$out/session.json" | head -1)"
    recorded=$((recorded + 1))
  else
    echo "no session.json for $out: the recording did not finalise" >&2
    failed=$((failed + 1))
  fi
done

echo "recorded $recorded, failed $failed"
[ "$failed" -eq 0 ]
