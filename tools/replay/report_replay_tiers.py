#!/usr/bin/env python3
"""Level of every Operations replay, so a queue can be filtered by tier.

The owner plays high-tier rooms, and Operations maps differ between the low and high
brackets, so low-tier replays must not enter the corpus. Two signals are reported side
by side, because they answer slightly different questions and a disagreement is worth
seeing:

  ship level   the recording player's own ship, resolved through the gamedata's
               ships.json (id -> level). This is what the owner is actually playing.
  scenario     the level band written into the operation's own name, e.g.
               WW2_OPERATION_3_67LVL (6-7) or LOW_LVL_OPERATION_2_LVL_56 (5-6). The
               newer PCVO operations only say MEDIUM_LVL / HIGH_LVL.

    /usr/bin/python3 tools/replay/report_replay_tiers.py [--max-level 7]

Prints the table, then the list to exclude. Exit code is 0 either way: this reports, the
caller decides.
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_replay_meta as meta_reader

SHIPS = "state/toalba/gamedata/data/ships.json"
SCORES = "replays/parsed/scored-replays.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="*", default=["replays/*.wowsreplay",
                                                "replays/incoming/*"])
    ap.add_argument("--max-level", type=int, default=7,
                    help="levels at or below this are excluded (the owner's rule)")
    ap.add_argument("--ships", default=SHIPS)
    ap.add_argument("--scores", default=SCORES)
    ap.add_argument("--json-out", default="replays/parsed/replay-tiers.json")
    args = ap.parse_args()

    ships = {int(k): v for k, v in json.loads(Path(args.ships).read_text()).items()}
    scores = {}
    for line in Path(args.scores).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            scores[row["id"]] = row

    rows = []
    for pattern in args.globs:
        for path in sorted(glob.glob(pattern)):
            try:
                meta = meta_reader.read_meta(path)
            except Exception:
                continue
            if meta.get("matchGroup") != "pve":
                continue
            found = re.search(r"(\d{6})", Path(path).name)
            rid = found.group(1) if found else "?"
            score = scores.get(rid, {})
            ship_id = score.get("own_ship_id")
            ship = ships.get(ship_id) if ship_id else None
            rows.append({
                "id": rid,
                "path": path,
                "map": (meta.get("mapName") or "?").replace("spaces/", ""),
                "scenario": meta.get("scenario") or "",
                "ship_id": ship_id,
                "ship": ship["name"] if ship else None,
                "level": ship["level"] if ship else None,
            })

    rows.sort(key=lambda r: (r["level"] if r["level"] is not None else 99, r["map"], r["id"]))
    print("%-8s %-6s %-22s %-34s %s" % ("id", "level", "map", "own ship", "scenario"))
    for row in rows:
        level = row["level"] if row["level"] is not None else "?"
        print("%-8s %-6s %-22s %-34s %s"
              % (row["id"], level, row["map"], (row["ship"] or "?")[:34],
                 row["scenario"][:44]))

    exclude = [r for r in rows if r["level"] is not None and r["level"] <= args.max_level]
    unknown = [r for r in rows if r["level"] is None]
    print("\n%d replay(s); exclude (level <= %d): %d"
          % (len(rows), args.max_level, len(exclude)))
    for row in exclude:
        print("   %-8s level %-3s %-20s %s"
              % (row["id"], row["level"], row["map"], row["scenario"][:40]))
    if unknown:
        print("level unknown (no own ship in the score file): %d" % len(unknown))
        for row in unknown:
            print("   %-8s %-20s %s" % (row["id"], row["map"], row["scenario"][:40]))

    # A ship outside the scenario's stated band would mean one of the two signals is
    # being read wrong, so it is called out rather than averaged away.
    for row in rows:
        band = re.search(r"_(\d{2,4})LVL", row["scenario"] or "")
        if band and row["level"]:
            digits = band.group(1)
            bounds = [int(digits[:2]), int(digits[2:])] if len(digits) == 4 else [int(digits[0])]
            if not any(abs(row["level"] - b) <= 1 for b in bounds):
                print("   NOTE %s: ship level %s vs scenario band %s"
                      % (row["id"], row["level"], digits))

    Path(args.json_out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
