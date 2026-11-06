#!/bin/bash
cd /home/lengbai/Downloads/wws || exit 1
LOG=replays/parsed/auto-learn-record24.log
KEEP="236066_random_Kurama_XI 236188_random_Harugumo_X 236269_coop_Valparaiso_IX 236291_pve_ops_Schlieffen_X 236597_random_M_Immelmann_X 236616_pve_ops_Bremen_X 236680_random_Archerfish_X 236773_pve_ops_LeTerrible_VIII"
for f in replays/*.wowsreplay; do
    s=$(basename "$f" .wowsreplay)
    case " $KEEP " in *" $s "*) continue;; esac
    if ps -eo rss=,args= --sort=-rss | awk '$1>1500000 && $0 ~ /python/ {found=1} END{exit !found}'; then
        echo "=== $s skipped: heavy python running $(date '+%H:%M:%S')" >> "$LOG"; continue
    fi
    avail=$(free -g | awk 'NR==2{print $7}')
    echo "=== $s start $(date '+%H:%M:%S') avail ${avail}G" >> "$LOG"
    if [ "${avail:-0}" -lt 8 ]; then echo "    skipped: only ${avail}G free" >> "$LOG"; continue; fi
    ./.venv/bin/python tools/replay/auto_learn.py --only "$s" >> "$LOG" 2>&1
    echo "=== $s done $(date '+%H:%M:%S') avail $(free -g | awk 'NR==2{print $7}')G" >> "$LOG"
done
echo "=== ALL DONE $(date '+%H:%M:%S')" >> "$LOG"
