"""Summarise self-scored replays per Operations map.

Reads the jsonl written by the scoring pass (one row per replay, scored with the
toalba parser: own damage + ribbons, i.e. quality without the community index)
and prints a per-map ladder so a recording batch can be picked by map.

Usage:
    report_replay_scores.py [--scores replays/parsed/scored-replays.jsonl]
                            [--recordings recordings] [--min-damage 200000]
"""

import argparse
import json
import pathlib
import re


def recorded_ids(recordings):
    ids = set()
    for path in pathlib.Path(recordings).glob("*/replay_path.txt"):
        name = path.read_text(errors="replace").strip()
        found = re.search(r"(\d{6})", pathlib.Path(name).name)
        if found:
            ids.add(found.group(1))
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="replays/parsed/scored-replays.jsonl")
    ap.add_argument("--recordings", default="recordings")
    ap.add_argument("--min-damage", type=float, default=200000.0)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in pathlib.Path(args.scores).read_text().splitlines()
        if line.strip()
    ]
    seen = recorded_ids(args.recordings)

    ops = [r for r in rows if r.get("is_operations_map")]
    print(
        "scored %d replays; %d on Operations maps; %d of those already recorded"
        % (len(rows), len(ops), sum(1 for r in ops if r["id"] in seen))
    )

    by_map = {}
    for row in ops:
        by_map.setdefault(row["map"], []).append(row)

    for map_name in sorted(by_map, key=lambda m: -len(by_map[m])):
        group = sorted(by_map[map_name], key=lambda r: -r["own_damage"])
        good = [r for r in group if r["own_damage"] >= args.min_damage]
        todo = [r for r in good if r["id"] not in seen]
        print(
            "\n%-20s  %d replay(s), %d >= %d dmg, %d of those unrecorded"
            % (map_name, len(group), len(good), int(args.min_damage), len(todo))
        )
        for row in group:
            mark = "rec" if row["id"] in seen else "   "
            keep = "*" if row["own_damage"] >= args.min_damage else " "
            print(
                "  %s%s %s  dmg=%-8d ribbons=%-5d %-4s build=%s"
                % (
                    mark,
                    keep,
                    row["id"],
                    int(row["own_damage"]),
                    row["ribbons"],
                    "%ds" % int(row["duration_s"]),
                    row["build"],
                )
            )


if __name__ == "__main__":
    main()
