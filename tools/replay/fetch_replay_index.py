#!/usr/bin/env python3
"""Walk the community index's Operations pool into a local JSONL.

The site's own `version` column is a hint, never a fact: it has mislabelled map names,
and it fills in defaults for rows its parser cannot read (all 15.9 rows, and 17 of 42
15.8 Operations rows). So this records what the index claims and nothing more; the
file's own metadata decides what a replay actually is after it is downloaded.

Use curl, not urllib: the site answers 403 to urllib's default user agent.

    /usr/bin/python3 tools/replay/fetch_replay_index.py --game-mode pve \
        --out replays/parsed/index-pve.jsonl

Selection criteria are printed and written next to the output so a later batch can say
exactly which pool the queue came from.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

API = "https://replayswows.com/api/search/index"


def fetch(game_mode, page, version=None, order="uploaded_at", by="desc"):
    url = "%s?gameMode=%s&order=%s&by=%s&page=%d" % (API, game_mode, order, by, page)
    if version:
        url += "&version=%s" % version
    result = subprocess.run(
        ["curl", "-s", "--max-time", "60", url,
         "-H", "Accept: application/json", "-H", "User-Agent: curl/8"],
        capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("curl failed on page %d: %s" % (page, result.stderr[:120]))
    try:
        return json.loads(result.stdout)
    except ValueError:
        raise SystemExit("page %d did not return JSON: %s" % (page, result.stdout[:200]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game-mode", default="pve", help="pve is Operations/co-op")
    ap.add_argument("--version", default=None,
                    help="the index's own version filter; it lies, so only use it to narrow")
    ap.add_argument("--order", default="uploaded_at")
    ap.add_argument("--by", default="desc")
    ap.add_argument("--max-pages", type=int, default=40)
    ap.add_argument("--out", default="replays/parsed/index-pve.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    page = 1
    while page <= args.max_pages:
        payload = fetch(args.game_mode, page, args.version, args.order, args.by)
        batch = payload.get("replays") or []
        if not batch:
            break
        rows.extend(batch)
        pages = payload.get("pages") or {}
        total = pages.get("total") or pages.get("last") or "?"
        print("  page %2d  %3d rows (running %d)  pages=%s"
              % (page, len(batch), len(rows), json.dumps(pages)))
        if pages.get("last") and page >= int(pages["last"]):
            break
        page += 1
        time.sleep(0.4)                     # be polite; this is somebody's site

    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    versions = {}
    for row in rows:
        key = str(row.get("version"))
        versions[key] = versions.get(key, 0) + 1
    print("\n%d rows -> %s" % (len(rows), out))
    print("index version column: %s" % json.dumps(versions, sort_keys=True))
    print("id range: %s .. %s"
          % (min(r.get("id", 0) for r in rows), max(r.get("id", 0) for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
