#!/usr/bin/env bash
# Find which config a client exit writes, by diffing the raw candidate files.
#
# Only two files on this machine change with the game: the prefix's system.reg and
# user.reg. Hashes say that much but not which key moved, so this copies those
# files whole: one copy while a client is running with the minimap already
# enlarged, and one copy after that client is gone. diff then names the key.
#
# The first version of this script took its "running" snapshot without checking
# that a client WAS running, and the client had already exited, so the comparison
# was empty. The presence check is therefore the first thing this version does.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
LOG=state/cfg-exit.log
BEFORE=state/cfg-raw-running
AFTER=state/cfg-raw-after-exit
: > "$LOG"

CANDIDATES=(
  "$HOME/.local/share/Steam/steamapps/compatdata/552990/pfx/user.reg"
  "$HOME/.local/share/Steam/steamapps/compatdata/552990/pfx/system.reg"
  "$HOME/.local/share/Steam/steamapps/compatdata/552990/pfx/userdef.reg"
  "$HOME/.local/share/Steam/steamapps/common/World of Warships/preferences.xml"
  "$HOME/Downloads/wws/preferences.xml"
)

copy_all() {
  local target=$1
  mkdir -p "$target"
  for path in "${CANDIDATES[@]}"; do
    [ -f "$path" ] || continue
    cp -p "$path" "$target/$(basename "$path").$(echo "$path" | md5sum | cut -c1-6)"
  done
}

echo "waiting for a client to be running $(date '+%H:%M:%S')" >> "$LOG"
for _ in $(seq 1 720); do
  pgrep -x WorldOfWarships >/dev/null && break
  sleep 5
done
if ! pgrep -x WorldOfWarships >/dev/null; then
  echo "no client appeared within 60 minutes" >> "$LOG"; exit 1
fi
echo "client is running; snapshot 1 of 2 $(date '+%H:%M:%S')" >> "$LOG"
copy_all "$BEFORE"
/usr/bin/python3 /tmp/snap_cfg.py state/cfg-run.txt >> "$LOG" 2>&1

echo "waiting for that client to exit $(date '+%H:%M:%S')" >> "$LOG"
for _ in $(seq 1 900); do
  pgrep -x WorldOfWarships >/dev/null || break
  sleep 5
done
if pgrep -x WorldOfWarships >/dev/null; then
  echo "the client is still running after 75 minutes" >> "$LOG"; exit 1
fi
sleep 30
echo "client gone; snapshot 2 of 2 $(date '+%H:%M:%S')" >> "$LOG"
copy_all "$AFTER"
/usr/bin/python3 /tmp/snap_cfg.py state/cfg-after-exit.txt >> "$LOG" 2>&1

{
  echo "# Raw config diff across one client exit (running -> gone, +30 s)."
  echo "# A key that moves here is written when the client shuts down, which is where"
  echo "# the minimap size would live if it is not kept in memory only."
  echo
  for path in "${CANDIDATES[@]}"; do
    [ -f "$path" ] || continue
    name=$(basename "$path").$(echo "$path" | md5sum | cut -c1-6)
    if [ -f "$BEFORE/$name" ] && [ -f "$AFTER/$name" ]; then
      echo "=== $path"
      diff -u "$BEFORE/$name" "$AFTER/$name" | grep -E '^[+-][^+-]' | head -40 \
        || echo "    (unchanged)"
    fi
  done
} > state/cfg-exit-diff.txt 2>&1
echo "DONE $(date '+%H:%M:%S')" >> "$LOG"
