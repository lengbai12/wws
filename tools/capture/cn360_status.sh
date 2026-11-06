#!/usr/bin/env bash
# cn360_status.sh - report the state of the 360 (CN) World of Warships install
# and the entry points we will use for online play and offline replay.
#
# 2026-10-16: the CN client is installed by the 360 Wargaming Game Center (WGC)
# into C:\Games\World_of_Warships_CN360 inside the Wine prefix, with the build
# number taken from WGC (currently 15.9.0.0.13357822).  Note that this is NOT
# the same build as the Steam 15.9 client (13357625), so replays are not
# interchangeable between the two clients.
#
# Usage: tools/capture/cn360_status.sh [--workspace DIR]
set -uo pipefail

GAMEDIR="${GAMEDIR:-/home/lengbai/.wine/drive_c/Games/World_of_Warships_CN360}"
WGC="${WGC:-/home/lengbai/.wine/drive_c/Program Files/360 Wargaming/GameCenter/wgc.exe}"

echo "=== game dir: $GAMEDIR"
if [ ! -d "$GAMEDIR" ]; then
  echo "MISSING: game dir"
else
  echo "--- game_info.xml state:"
  if [ -f "$GAMEDIR/game_info.xml" ]; then
    sed -n 's/.*<installed>\(.*\)<\/installed>.*/installed=\1/p' "$GAMEDIR/game_info.xml"
    sed -n 's/.*<version name="\([a-z]*\)" installed="\([^"]*\)" available="\([^"]*\)".*/part=\1 installed=\2 available=\3/p' "$GAMEDIR/game_info.xml"
  else
    echo "game_info.xml MISSING"
  fi

  echo "--- staged package:"
  find "$GAMEDIR/updates" -maxdepth 2 -name '*.dspkg' -printf '  %p size=%s blocks=%b\n' 2>/dev/null
  find "$GAMEDIR/updates" -maxdepth 1 -name '*.resume' -printf '  resume=%f\n' 2>/dev/null

  echo "--- installed tree (once unpacked):"
  for p in WorldOfWarships.exe bin; do
    if [ -e "$GAMEDIR/$p" ]; then echo "  present: $p"; else echo "  absent:  $p"; fi
  done
  if [ -d "$GAMEDIR/bin" ]; then
    for d in "$GAMEDIR"/bin/*; do
      [ -d "$d" ] || continue
      echo "  build dir: $(basename "$d")"
      [ -f "$d/bin64/WorldOfWarships64.exe" ] && echo "    bin64/WorldOfWarships64.exe present ($(stat -c %s "$d/bin64/WorldOfWarships64.exe") bytes)"
    done
  fi
fi

echo "=== launcher: $WGC"
[ -f "$WGC" ] && echo "  present ($(stat -c %s "$WGC") bytes)" || echo "  MISSING"

echo "=== processes"
ps -eo pid=,etimes=,comm=,args= 2>/dev/null | awk '
  $3 ~ /^wgc\.exe$/ || $3 ~ /^CrUtilityMain$/ || $3 ~ /^CrRendererMain$/ {printf "  launcher: pid=%s up=%ss comm=%s\n",$1,$2,$3}
  $3 ~ /^WorldOfWarships$/ || $3 ~ /^WorldOfWarships64/ {printf "  game:     pid=%s up=%ss comm=%s\n",$1,$2,$3}
'
echo "=== entry points"
cat <<'EOF'
  online play  : WGC (wws-cn360-platform) -> 战舰世界 tile -> start button
                 launcher cmd:
                   env DISPLAY=:1 WINEPREFIX=/home/lengbai/.wine WINEDEBUG=-all \
                     wine "$WGC"
  offline replay: expected to be the same route as the Steam 15.8 client:
                   bin/<build>/bin64/WorldOfWarships64.exe <abs path to .wowsreplay>
                 NOT yet verified for the CN client - confirm after the install
                 finishes (WGC may have to be running so the client can get its
                 session).
  build number : CN 15.9.0.0.13357822  vs  Steam 15.9 13357625  (NOT interchangeable)
EOF
