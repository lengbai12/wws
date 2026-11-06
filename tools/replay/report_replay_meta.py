"""Report the metadata each replay file carries about itself.

The community index and any map-name shortcut are only discovery hints; the file's
own metadata is the ground truth for which battle it is. This reads only the
metadata blocks (never the packet stream) so it stays cheap over a whole pool.

Usage:
    report_replay_meta.py [GLOB ...]
"""

import argparse
import glob
import json
import struct
from collections import Counter

SIGNATURE = b"\x12\x32\x34\x11"

FIELDS = (
    "clientVersionFromExe",
    "mapName",
    "matchGroup",
    "gameMode",
    "scenario",
    "battleDuration",
    "playerName",
    "playerVehicle",
    "dateTime",
)


def read_meta(path):
    """Parse the header blocks of a replay, skipping the compressed packet stream."""
    with open(path, "rb") as handle:
        head = handle.read(8)
        if head[:4] != SIGNATURE:
            raise ValueError("not a replay (signature %s)" % head[:4].hex(" "))
        blocks_count = struct.unpack_from("<i", head, 4)[0]
        offset = 8
        meta = {}
        for index in range(blocks_count):
            with open(path, "rb") as handle:
                handle.seek(offset)
                size = struct.unpack("<i", handle.read(4))[0]
                offset += 4
                blob = handle.read(size)
                offset += size
            if index or not blob:
                continue
            try:
                meta = json.loads(blob)
            except (ValueError, UnicodeDecodeError):
                meta = {}
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="*", default=["replays/*.wowsreplay",
                                                "replays/incoming/*",
                                                "replays/steam159/*"])
    ap.add_argument("--tsv", action="store_true",
                    help="machine-readable rows instead of the report")
    args = ap.parse_args()

    paths = []
    for pattern in args.globs:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit("no replays matched")

    rows = []
    for path in paths:
        try:
            meta = read_meta(path)
        except (ValueError, OSError) as error:
            print("%-52s SKIP  %s" % (path, error))
            continue
        build = ",".join((meta.get("clientVersionFromExe") or "").split(",")[:3])
        rows.append({
            "path": path,
            "build": build,
            "map": (meta.get("mapName") or "").replace("spaces/", ""),
            "match_group": meta.get("matchGroup"),
            "game_mode": meta.get("gameMode"),
            "scenario": meta.get("scenario"),
            "duration": meta.get("battleDuration"),
            "own": meta.get("playerName"),
            "ship": meta.get("playerVehicle"),
        })

    if args.tsv:
        for row in rows:
            print("\t".join(str(row[key]) for key in (
                "path", "build", "map", "match_group", "game_mode", "scenario",
                "duration", "own", "ship")))
        return

    print("%d replay files\n" % len(rows))
    for row in rows:
        print("%-52s %-16s %-20s %-10s %-14s %s"
              % (row["path"].split("/")[-1], row["build"], row["map"],
                 row["match_group"], row["game_mode"], row["scenario"] or ""))

    print("\n=== builds ===")
    for build, count in Counter(r["build"] for r in rows).most_common():
        print("  %-16s %d" % (build or "(none)", count))
    print("\n=== (matchGroup, gameMode, scenario) ===")
    for key, count in Counter((r["match_group"], r["game_mode"], r["scenario"])
                              for r in rows).most_common():
        print("  %-40s %d" % (str(key), count))
    print("\n=== maps ===")
    for key, count in Counter(r["map"] for r in rows).most_common():
        print("  %-24s %d" % (key, count))


if __name__ == "__main__":
    main()
