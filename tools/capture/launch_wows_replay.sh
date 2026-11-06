#!/usr/bin/env bash
# Offline playback of a 15.8.0 replay with its MATCHING on-disk WoWS client.
#
# HOW THIS WORKS (read docs/replay-playback-and-capture.md before changing it):
#   A .wowsreplay only plays in a client of the same build. Steam's own launcher
#   (WorldOfWarships.exe -> clientrunner) always starts the NEWEST installed
#   build, so `steam -applaunch 552990 <replay>` feeds a 15.8.0 replay to the
#   15.9.0 client and it dies with EXCEPTION_ACCESS_VIOLATION.
#   The fix is NOT to downgrade anything: the 15.8.0 client is still on disk at
#   bin/13187581/, so call it directly under gamescope + Proton.
#
# VERIFIED: this launched bin/13187581/bin64/WorldOfWarships64.exe with the
# Kurama XI replay and rendered a real battle HUD (ship labels, 190:134 score,
# 15:43 timer, 91 632 own HP, F/R/T/Y/U consumable bar, minimap).
#
# Steam is deliberately NOT used here. The direct-Proton path is OFFLINE-only:
# a live battle needs Steam's login pipeline (LoginApp rejects the bypass).
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 REPLAY_FILE" >&2
  exit 2
fi

STEAM="$HOME/.local/share/Steam"
export STEAM_COMPAT_DATA_PATH="$STEAM/steamapps/compatdata/552990"
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$STEAM"
export STEAM_COMPAT_INSTALL_PATH="$STEAM/steamapps/common/World of Warships"
export SteamAppId=552990
export SteamGameId=552990

exec gamescope -W 1920 -H 1080 -w 1920 -h 1080 -f -- \
  "$STEAM/steamapps/common/Proton - Experimental/proton" waitforexitandrun \
  "$STEAM/steamapps/common/World of Warships/bin/13187581/bin64/WorldOfWarships64.exe" \
  "$1"
