#!/usr/bin/env python3
"""Order the next recording batch by coverage gap, not by damage.

The owner's rule: a map can have two or three spawn patterns and our samples may cover
only some of them, so a model trained on this corpus is only as good as the coverage it
has. This works out which recordings would close the gaps.

Signature per recording, computed the same way for the corpus and for the candidates so
the two are comparable:

  * take the enemy vehicles from the roster (everyone whose team_id differs from the
    recording player's) - no relation parsing needed, team_id says it;
  * take their positions in the opening window;
  * normalise by the battle's own extent, which removes the need for a map-size table
    and makes two recordings of one map directly comparable;
  * histogram that into an 8x8 grid and normalise again.

Recordings of one map that share an opening land together; a different spawn pattern
moves the mass elsewhere. The corpus is clustered first, so the printed queue says which
existing pattern each candidate would fill and which pattern would be NEW.

Position files are produced on demand with export_events.py --kinds position (about
twenty seconds per replay); the corpus ones already exist from the recording runs.

    ./.venv/bin/python tools/replay/classify_next_batch.py --candidates 'replays/incoming/*'
"""

import argparse
import glob
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

from cluster_spawn_patterns import cosine_distance
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_replay_meta as meta_reader

GAMEDATA = "state/toalba/gamedata/data/scripts_entity/entity_defs"
try:
    from wows_replay_parser import parse_replay
except ImportError:
    raise SystemExit("run this with state/toalba/venv/bin/python: the replay parser "
                     "lives in its own environment")

CELLS = 8           # 8x8 cells of CELL_M metres, anchored on the own ship's start
CELL_M = 4000.0     # a fixed cell size: normalising by each battle's own extent made
                    # every pair orthogonal (distance 1.000 for everything), because a
                    # different route gives a different bounding box. 8x8 cells of 4 km
                    # spans 32 km, which covers a whole Operations map: at 1 km cells the
                    # opening enemies sat outside the window and every histogram was zero.
OPENING_S = 60.0
NEEDED = 3          # recordings per pattern before a pattern counts as covered


def position_file(replay_id):
    return Path("replays/parsed") / ("position-%s.jsonl" % replay_id)


def ensure_positions(path, replay_id):
    out = position_file(replay_id)
    if out.exists() and out.stat().st_size > 0:
        return out
    print("  exporting positions for %s ..." % replay_id)
    result = subprocess.run(
        ["./.venv/bin/python", "tools/replay/export_events.py", str(path),
         "--kinds", "position", "--out", str(out)],
        capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        print("    failed: %s" % (result.stderr or result.stdout)[-200:])
        return None
    return out


def signature(path, replay_id):
    """Coarse-grid histogram of the opening enemy positions, or None."""
    # Two sources, because they cover different ships and they agree on the id space:
    # our own export (export_events.py --kinds position) carries the RECORDING player's
    # position stream only - the own ship sends a different packet - while the parser's
    # Position events carry every other vehicle, in exactly the roster's vehicle_id
    # space (verified: 56 of 56 match). So the anchor comes from ours and the enemies
    # from the parser.
    out = ensure_positions(path, replay_id)
    if out is None:
        return None
    roster, own_positions, session = [], [], {}
    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.get("event")
        if kind == "session":
            session = row
        elif kind == "roster":
            roster.append(row)
        elif kind == "position":
            own_positions.append(row)
    if not roster or not own_positions:
        return None

    player = session.get("recording_player")
    own = next((r for r in roster if r.get("name") == player), None)
    if own is None:
        return None
    enemies = {r["vehicle_id"] for r in roster
               if r.get("team_id") != own.get("team_id")}
    if not enemies:
        return None

    # Anchor on the own ship's first position: it is the one fixed thing on a map, so
    # "the enemies appear over there" keeps its meaning across recordings and patterns,
    # and a fixed cell size keeps the cells the same width on every map.
    start = own_positions[0]
    replay = parse_replay(replay_path=str(path), gamedata_path=GAMEDATA)

    cells = np.zeros(CELLS * CELLS)
    for event in replay.events:
        if type(event).__name__ != "PositionEvent":
            continue
        if event.entity_id not in enemies or event.timestamp > OPENING_S:
            continue
        column = int((event.x - start["x"]) / CELL_M)
        index = int((event.z - start["z"]) / CELL_M)
        if not (0 <= column < CELLS and 0 <= index < CELLS):
            continue
        cells[index * CELLS + column] += 1.0
    total = cells.sum()
    return {"map": session.get("map"), "scenario": session.get("scenario"),
            "enemies": len(enemies), "in_window": int(total),
            "histogram": cells / total if total else cells}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patterns", default="replays/parsed/spawn-patterns-hightier.json",
                    help="the corpus coverage table from cluster_spawn_patterns.py")
    ap.add_argument("--candidates", nargs="*", default=["replays/incoming/*"])
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--exclude", default="",
                    help="ids already recorded or recording, comma separated")
    ap.add_argument("--queue-out", default="replays/parsed/next-batch-queue.txt")
    args = ap.parse_args()

    excluded = {name.strip() for name in args.exclude.split(",") if name.strip()}
    table = json.loads(Path(args.patterns).read_text())
    corpus = []
    for map_name, entry in table.items():
        for pattern in entry["patterns"]:
            for recording in pattern["recordings"]:
                corpus.append((map_name, recording))
    print("corpus: %d recording(s) over %d map(s)" % (len(corpus), len(table)))

    corpus_signatures = {}
    for map_name, recording in corpus:
        rid = recording.split("_")[0]
        hits = sorted(Path("replays").glob("%s*.wowsreplay" % rid))
        if not hits:
            print("  corpus %s: replay not found" % recording)
            continue
        result = signature(hits[0], rid)
        if result:
            corpus_signatures[recording] = result["histogram"]
    print("signed %d corpus recording(s)" % len(corpus_signatures))

    # Existing patterns, as the minimap-based table already grouped them.
    patterns = {}
    for map_name, entry in table.items():
        for index, pattern in enumerate(entry["patterns"], 1):
            members = [m for m in pattern["recordings"] if m in corpus_signatures]
            patterns[(map_name, index)] = members
    gaps = {key: members for key, members in patterns.items() if len(members) < NEEDED}
    print("under-covered: %d pattern(s) (need %d each)" % (len(gaps), NEEDED))
    for (map_name, index), members in sorted(gaps.items()):
        print("   %-22s pattern %d: %d  %s" % (map_name, index, len(members),
                                               ", ".join(members)))

    candidates = []
    ids = []
    for pattern in args.candidates:
        ids.extend(sorted(glob.glob(pattern)))
    for path in ids:
        found = re.search(r"(\d{6})", Path(path).name)
        if not found:
            continue
        rid = found.group(1)
        if rid in {c[1].split("_")[0] for c in corpus}:
            continue
        if rid in excluded:
            continue
        try:
            meta = meta_reader.read_meta(path)
        except Exception:
            continue
        if meta.get("matchGroup") != "pve":
            continue
        result = signature(path, rid)
        if not result:
            print("  %s: no signature" % rid)
            continue
        if result["in_window"] == 0:
            print("  %s: no enemy positions inside the window" % rid)
        map_name = result["map"]
        best = None
        for (pattern_map, index), members in patterns.items():
            if pattern_map != map_name or not members:
                continue
            distance = min(cosine_distance(result["histogram"], corpus_signatures[m])
                           for m in members)
            if best is None or distance < best[1]:
                best = (index, distance)
        # A map the corpus has never seen is a new map, not an already-covered one.
        known_map = any(key[0] == map_name for key in patterns)
        fills = ((map_name, best[0]) in gaps) if (best and known_map) else not known_map
        candidates.append({"id": rid, "map": map_name, "scenario": result["scenario"],
                           "pattern": best[0] if best else None,
                           "distance": best[1] if best else None,
                           "fills_gap": fills,
                           "new_pattern": bool(best and best[1] > args.threshold)})

    candidates.sort(key=lambda c: (not c["fills_gap"], c["new_pattern"],
                                   c["distance"] if c["distance"] is not None else 9.9,
                                   c["map"], c["id"]))
    print("\n=== queue (%d candidate(s), gap-filling first) ===" % len(candidates))
    lines = []
    for index, c in enumerate(candidates, 1):
        tag = ("NEW map" if c["pattern"] is None else
               "NEW pattern" if c["new_pattern"] else
               ("fills %s pattern %s" % (c["map"], c["pattern"]) if c["fills_gap"]
                else "already covered (%s pattern %s)" % (c["map"], c["pattern"])))
        line = "%2d. %-8s %-20s %-34s %-6s %s" % (
            index, c["id"], c["map"], (c["scenario"] or "")[:34],
            "%.3f" % c["distance"] if c["distance"] is not None else "  n/a", tag)
        print("  " + line)
        lines.append("%s %s" % (c["id"], c["map"]))

    Path(args.queue_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nwrote %s" % args.queue_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
