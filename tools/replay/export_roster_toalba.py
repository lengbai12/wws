#!/usr/bin/env python3
"""Emit the roster rows the label chain needs, read with the toalba parser.

Why this exists: the packet-level export (`export_events.py`) resolves the roster from
`replays_unpack`'s players info, and on 15.9 that comes back empty - every field None,
`link_route: unresolved`. The roster is what separates teammates from enemies
(`build_minimap_inventory.py`: teammates are drawn from the position stream plus their
team, enemies only while `Avatar.spottedEntities` names them), so an unresolved roster
makes the truth untrustworthy and the corpus unusable.

The toalba parser gives the same information, and its `relation` field (0 self / 1 ally /
2 enemy) was cross-validated against our own parser on a 12v12 PvP replay: one self,
eleven allies, twelve enemies. This writes roster rows in EXACTLY the schema
`export_events.py` writes, plus `relation`, so a consumer can take either source.

Run it with the parser's own environment, not the project venv:

    state/toalba/venv/bin/python tools/replay/export_roster_toalba.py <replay> --out <jsonl>

`export_events.py` calls this automatically when its own roster comes back unresolved,
so the label chain needs no other change.
"""

import argparse
import json
import sys
from pathlib import Path

GAMEDATA = "state/toalba/gamedata/data/scripts_entity/entity_defs"

try:
    from wows_replay_parser import parse_replay
except ImportError:
    raise SystemExit("run this with state/toalba/venv/bin/python: the parser lives in "
                     "its own environment")


def frags_by_account(replay):
    """Kills per account, keyed by account id, or an empty map when unavailable."""
    try:
        results = replay.battle_results()
    except Exception:
        return {}
    out = {}
    for entry in results.players.values():
        stats = getattr(entry, "stats", None) or {}
        account = stats.get("account_db_id")
        if account is not None:
            out[int(account)] = stats.get("frags")
    return out


def position_entities(replay):
    """Vehicle ids seen in the position stream: the space our roster's ids live in."""
    return {event.entity_id for event in replay.events
            if type(event).__name__ == "PositionEvent"}


def resolve_vehicle(player, positions):
    """The vehicle id our own export would use, and how it was found.

    The parser's `entity_id` is a ship-type id, not a vehicle: on a measured session
    every player's `avatar_id + 1` was the vehicle, one after another (371580 -> 371581,
    371582 -> 371583, ...). That is the same "avatar+1 pairing" our packet-level export
    documents as a last resort. So: prefer the pairing when it lands on a vehicle we
    actually saw in the position stream, then the raw entity id, then nothing - and say
    which route was taken, so an unresolved row is visible instead of silently wrong.
    """
    avatar = getattr(player, "avatar_id", None)
    if avatar is not None and (avatar + 1) in positions:
        return avatar + 1, "toalba.avatar+1"
    entity = getattr(player, "entity_id", None)
    if entity is not None and entity in positions:
        return entity, "toalba.entity_id"
    return None, "unresolved"


def roster_rows(replay):
    kills = frags_by_account(replay)
    positions = position_entities(replay)
    # team_id from the parser is wrong for bots (a measured session had every bot on
    # team 0 while our own export, and its relation, put them on the enemy team), so the
    # team is derived from `relation`, which is the field that was cross-validated: one
    # self, allies on the self player's team, enemies on the other one.
    own_team = next((int(p.team_id) for p in replay.players
                     if getattr(p, "relation", None) == 0), None)
    rows = []
    for player in replay.players:
        account = getattr(player, "account_id", None)
        vehicle, route = resolve_vehicle(player, positions)
        relation = getattr(player, "relation", None)
        if relation == 0:
            team = own_team
        elif relation == 1:
            team = own_team
        elif relation == 2:
            team = (1 - own_team) if own_team in (0, 1) else None
        else:
            team = None
        rows.append({
            "event": "roster",
            "avatar_id": getattr(player, "avatar_id", None),
            "vehicle_id": vehicle,
            "link_route": route,
            "account_id": account,
            "name": getattr(player, "name", None),
            "ship_id": vehicle,
            "team_id": team,
            "is_bot": getattr(player, "is_bot", None),
            "is_alive": getattr(player, "is_alive", None),
            "max_health": getattr(player, "max_health", None),
            "frags": kills.get(int(account)) if account is not None else None,
            # Not in the packet-level schema, and the reason this exporter exists: it
            # answers "friend or enemy" without needing a team table, and unlike team_id
            # it is right for bots.
            "relation": relation,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("replay")
    ap.add_argument("--gamedata", default=GAMEDATA)
    ap.add_argument("--out", help="write here; default is stdout")
    args = ap.parse_args()

    replay = parse_replay(replay_path=args.replay, gamedata_path=args.gamedata)
    rows = roster_rows(replay)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    named = sum(1 for row in rows if row["name"])
    resolved = sum(1 for row in rows if row["vehicle_id"] is not None)
    print("%s: %d roster row(s), %d named, %d with a vehicle"
          % (Path(args.replay).name, len(rows), named, resolved), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
