#!/usr/bin/env python3
"""Score candidate replays with our own parser, without the community index.

The web index fills in defaults for the rows it cannot parse (for 15.8 Operations
that is 17 of 42 rows reading "0 damage / DRAW", and every 15.9 row), so quality has
to come from the file itself. This walks the replay's own events and reports who the
recording player was, how hard they hit, and how many ribbons they earned.

It runs on the parser installed in this repository, NOT on the project's own venv:

    state/toalba/venv/bin/python tools/replay/score_replays.py 'replays/incoming/*' \
        --out replays/parsed/scored-replays.jsonl

That environment and its per-build gamedata live under state/toalba/ on purpose: the
first copies were built in /tmp, and a reboot wiped them, so every operational artefact
now lives inside the repository.

One replay at a time, one process, about a gigabyte of peak RSS: this machine has
30 GB and no swap, and parallel workers have OOM-killed runs here twice.
"""

import argparse
import collections
import glob
import json
import sys
import time
from pathlib import Path

GAMEDATA = "state/toalba/gamedata/data/scripts_entity/entity_defs"


def score(replay, replay_id):
    own = next((p for p in replay.players if p.relation == 0), None)
    damage = 0.0
    if own is not None:
        # The client's own result table is the authoritative damage figure. The file
        # from the first scan used a per-event sum whose exact filter cannot be
        # reconstructed, and it read about 2 % lower (407915 against 416027 on
        # 236587); no selection threshold is close enough for that to matter, but the
        # numbers do shift slightly when the pool is rescanned.
        #
        # Do NOT sum Damage events by entity: their entity_id/target_id is the ship
        # that TOOK the hit, so summing those gives damage received, not damage dealt.
        try:
            results = replay.battle_results()
            damage = float(results.players[results.own_db_id].stats.get("damage") or 0.0)
        except Exception as error:                          # older files may lack it
            print("    no battle results for %s: %s" % (replay_id, str(error)[:50]))
    mapping = replay.map_name.replace("spaces/", "")
    return {
        "id": replay_id,
        # The full four-part version, as the first scan recorded it (15,8,0,13187581):
        # the third part alone cannot tell 15.8 from 15.9's parser cache.
        "build": str(replay.game_version),
        "map": mapping,
        "is_operations_map": mapping[:1].lower() == "s" or mapping[:3] in ("33_", "13_", "37_"),
        "duration_s": round(replay.duration, 1),
        "players": len(replay.players),
        "relations": dict(collections.Counter(p.relation for p in replay.players)),
        "own_name": own.name if own else None,
        "own_team": own.team_id if own else None,
        "own_ship_id": own.ship_id if own else None,
        "own_max_health": own.max_health if own else None,
        "own_damage": damage,
        "ribbons": len(list(replay.recording_player_ribbons())),
        "events": len(replay.events),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replays", nargs="+", help="replay paths or globs")
    ap.add_argument("--gamedata", default=GAMEDATA)
    ap.add_argument("--out", default="replays/parsed/scored-replays.jsonl")
    ap.add_argument("--limit", type=int, help="stop after this many replays")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing --out file instead of refusing")
    args = ap.parse_args()

    try:
        from wows_replay_parser import parse_replay
    except ImportError:
        raise SystemExit("run this with state/toalba/venv/bin/python: the parser is "
                         "not installed in the project venv")

    paths = []
    for pattern in args.replays:
        paths.extend(sorted(glob.glob(pattern)))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit("no replays matched")

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit("%s exists; pass --force to rescan over it (the old numbers "
                         "shift by about 2 %%, see score())" % out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for index, path in enumerate(paths, 1):
            started = time.time()
            try:
                replay = parse_replay(replay_path=path, gamedata_path=args.gamedata)
            except Exception as error:                      # a broken file is data too
                print("%-46s SKIP  %s" % (Path(path).name, str(error)[:60]))
                continue
            row = score(replay, Path(path).name.split(".")[0])
            row["seconds"] = round(time.time() - started, 1)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print("%-46s %-20s own_dmg=%-9d ribbons=%-5d %4.1fs"
                  % (row["id"], row["map"], int(row["own_damage"]), row["ribbons"],
                     row["seconds"]))
    print("\nwrote %s (%d replays)" % (out, len(paths)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
