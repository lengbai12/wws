#!/usr/bin/env python3
"""Export a time-resolved event stream from World of Warships replays.

The replay container is decoded by :mod:`reader`. Entity, property and method
semantics come from the pinned upstream schema fetched by :mod:`fetch_schema`;
this module only turns those into flat JSONL records.

Events emitted (one JSON object per line):

``session``      one per replay: build, map, mode, packet/clock statistics
``roster``       one per replay: entity id to player and ship identity
``death``        ship destruction, with victim, killer and death cause
``damage``       damage instance from one vehicle to another
``consumable``   consumable activation, with type id and remaining work time
``health``       health property change for a vehicle
``alive``        isAlive property change for a vehicle

Requires the interpreter dependencies of the upstream parser: ``lxml``,
``packaging``, ``pycryptodome``.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import reader  # noqa: E402
from fetch_schema import DEFAULT_BUILD, DEFAULT_CACHE, DEFAULT_COMMIT, fetch  # noqa: E402

# Upstream subscribes through class-level maps that outlive one replay.
_SUBSCRIPTION_RESET = ("_methods_subscriptions", "_properties_subscriptions",
                       "_nested_properties_subscription")



POSITION_PACKET = 0x2C
POSITION_PAYLOAD = 32


def own_position_events(packets, own_id):
    """Own-ship position samples, decoded from 0x2C packets.

    No exported property carries the own ship's absolute position: the 45-byte
    0x0A position packets cover every OTHER vehicle and omit it, and
    ``Avatar.vehiclePosition`` is never sent in this replay at all. The own ship
    sends a 32-byte 0x2C packet instead, laid out as an entity id followed by
    seven little-endian float32 values.

    The float positions were confirmed against data whose value was already
    known: the same offsets decode the 0x0A payload of ship 1448660 as
    (y 0.0, x 540.002, y 0.0, z 400.002), matching the track file that records
    that ship at x 540, y 0, z 400. The seventh float is the yaw.
    """
    events = []
    for packet in packets:
        if packet.type != POSITION_PACKET or len(packet.payload) != POSITION_PAYLOAD:
            continue
        entity = int.from_bytes(packet.payload[:4], "little", signed=True)
        if entity != own_id:
            continue
        y, x, _y2, z, yaw = struct.unpack_from("<5f", packet.payload, 4)
        events.append({
            "event": "position", "clock": round(packet.clock, 3), "vehicle": own_id,
            "x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "yaw": round(yaw, 4),
        })
    return events



TRACK_PACKET = 0x0A
TRACK_PAYLOAD = 45
TRACK_YAW_OFFSET = 32


def vehicle_track_events(packets):
    """Every vehicle's position and heading, decoded from 45-byte 0x0A packets.

    Vehicles other than the recording player's own ship send this packet; the own
    ship sends 0x2C instead, so the two streams together cover the whole battle.
    The layout is an entity id, four float32 values whose second and fourth are
    x and z, then a float32 yaw at offset 32. Both offsets were confirmed against
    data whose value was already known: this packet decodes ship 1448660 as
    (x 540.0, z 400.0), and the yaw at offset 32 as -3.142, matching that ship's
    recorded track row exactly.
    """
    events = []
    for packet in packets:
        if packet.type != TRACK_PACKET or len(packet.payload) != TRACK_PAYLOAD:
            continue
        vehicle = int.from_bytes(packet.payload[:4], "little", signed=True)
        _y, x, _y2, z = struct.unpack_from("<4f", packet.payload, 4)
        yaw = struct.unpack_from("<f", packet.payload, TRACK_YAW_OFFSET)[0]
        events.append({"event": "track", "clock": round(packet.clock, 3),
                       "vehicle": vehicle, "x": round(x, 3), "z": round(z, 3),
                       "yaw": round(yaw, 4)})
    return events



def _pair(value):
    """Read (x, z) from a two-element sequence."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    return None, None


def reset_subscriptions(entity_module) -> None:
    """Clear class-level subscription maps so replays in one process stay independent."""
    for name in _SUBSCRIPTION_RESET:
        setattr(entity_module.Entity, name, {})


def subscribe(entity_module, table: str, key: str, callback) -> None:
    """Append to a subscription bucket.

    Upstream ``subscribe_method_call`` / ``subscribe_property_change`` guard on the
    bare method name while storing composite ``Entity_Method`` keys, so a second
    subscriber rebuilds the bucket and silently drops the first one. Appending
    directly keeps every subscriber, including the controller's own.
    """
    bucket = getattr(entity_module.Entity, table)
    bucket.setdefault(key, []).append(callback)


def _load_parser(cache_root: Path):
    sys.path.insert(0, str(cache_root))
    try:
        import replay_unpack.core.entity as entity_module
        from replay_unpack.clients.wows.helper import get_definitions
        from replay_unpack.clients.wows.player import ReplayPlayer
    except ImportError as exc:
        raise SystemExit(
            f"cannot import the replay parser from {cache_root}: {exc}\n"
            "install its dependencies: pip install lxml packaging pycryptodome"
        ) from exc
    return entity_module, get_definitions, ReplayPlayer


def _load_death_types(cache_root: Path, version: str) -> dict:
    name = version.replace(".", "_")
    module = cache_root / "replay_unpack/clients/wows/versions" / name / "constants.py"
    if not module.is_file():
        return {}
    import importlib.util

    spec = importlib.util.spec_from_file_location(f"wows_constants_{name}", module)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    raw = getattr(loaded, "DEATH_TYPES", {}) or {}
    return {int(k): v.get("name") for k, v in raw.items()}


def _load_consumable_types(cache_root: Path) -> dict[int, str]:
    table = cache_root / "enrich/consumable_type_ids.json"
    if not table.is_file():
        return {}
    data = json.loads(table.read_text())
    return {int(k): v for k, v in data.items() if k.lstrip("-").isdigit()}


def _decode_consumable(blob: bytes) -> tuple[int | None, str]:
    """The usage payload is ``[flags, type_id]``; keep the raw form for auditing."""
    if len(blob) == 2:
        return blob[1], blob.hex()
    return None, blob.hex()


class Collector:
    def __init__(self, entity_module, death_types: dict, consumable_types: dict):
        self.entity_module = entity_module
        self.death_types = death_types
        self.consumable_types = consumable_types
        self.clock = 0.0
        self.events: list[dict] = []
        self.avatar_ship: dict[int, int] = {}
        self.vehicle_ids: set[int] = set()
        self.zone_state: dict[int, dict] = {}
        self.squadron_info: dict[int, dict] = {}
        self.method_name = ""
        # Which kinds were asked for, so optional subscriptions are only added
        # when their rows will be kept.
        self.kinds: set[str] = set()
        self.visibility_time: dict[int, float] = {}

    # -- subscribers ----------------------------------------------------
    def on_vehicle_death(self, avatar, killed, fragger, death_type):
        self.events.append({
            "event": "death", "clock": round(self.clock, 3),
            "victim": killed, "killer": fragger, "death_type": death_type,
            "death_type_name": self.death_types.get(death_type),
        })

    def on_damages(self, vehicle, damages):
        """``receiveDamagesOnShip`` is called on the victim; each entry names an attacker.

        The upstream controller reads this the other way round, which cannot hold:
        several ships would then receive less damage than they lose in health. The
        victim reading keeps health_loss <= damage_received for every ship.
        """
        for entry in damages:
            self.events.append({
                "event": "damage", "clock": round(self.clock, 3),
                "victim": vehicle.id, "attacker": entry.get("vehicleID"),
                "damage": entry.get("damage"),
            })

    def on_consumable(self, vehicle, **kwargs):
        blob = kwargs.get("consumableUsageParams") or b""
        type_id, raw = _decode_consumable(blob)
        self.events.append({
            "event": "consumable", "clock": round(self.clock, 3),
            "vehicle": vehicle.id, "consumable_type": type_id,
            # Consumable type 0 is crashCrew, so test for None rather than falsiness.
            "consumable_name": self.consumable_types.get(type_id) if type_id is not None else None,
            "work_time_left": kwargs.get("workTimeLeft"), "raw": raw,
        })

    def on_avatar_ship(self, avatar, value):
        self.avatar_ship[avatar.id] = value
        self.vehicle_ids.add(value)

    def absorb_entities(self, controller) -> None:
        """The parser's entity table is authoritative for what exists in the stream."""
        for entity in controller.entities.values():
            if entity.get_name() == "Vehicle":
                self.vehicle_ids.add(entity.id)

    def on_health(self, vehicle, value):
        self.events.append({"event": "health", "clock": round(self.clock, 3),
                            "vehicle": vehicle.id, "value": value})

    def on_spotted(self, avatar, entities):
        """``Avatar.spottedEntities`` is the visibility signal, OWN_CLIENT only.

        The parser decodes the payload into an ordered list of entity ids (the
        wire form is one count byte followed by that many 4-byte little-endian
        ids). An EMPTY list is a real answer: nothing is spotted at that instant,
        which is the only way to tell "not spotted" from "no information".
        """
        values = [int(item) for item in entities] if entities is not None else []
        self.events.append({"event": "spotted", "clock": round(self.clock, 3),
                            "entities": values, "count": len(values)})

    def emit_zone(self, zone, value, field):
        """Capture-point and other interactive-zone state.

        A zone carries no world position in the replay, so only its identity and
        state are exported here; the map fixes where it is drawn. The field name
        is bound by the caller because the parser invokes a property callback
        with exactly (entity, value).
        """
        snapshot = self.zone_state.setdefault(zone.id, {})
        snapshot[field] = value
        self.events.append({"event": "zone", "clock": round(self.clock, 3),
                            "zone": zone.id, "name": snapshot.get("name"),
                            "team_id": snapshot.get("teamId"),
                            "type": snapshot.get("type"),
                            "radius": snapshot.get("radius"),
                            "owner_id": snapshot.get("ownerId"),
                            "visual_state": snapshot.get("visualState")})

    def on_visibility_flags(self, vehicle, value):
        """``Vehicle.visibilityFlags`` is a per-vehicle bitmask, ALL_CLIENTS.

        ``Avatar.spottedEntities`` is the recording player's OWN visibility, while
        the minimap draws what the whole TEAM has detected, so the two disagree
        and the flags are the better source for what appears on the map. Measured
        evidence: sampling the pixel under each unspotted enemy's projected
        position found a drawn enemy marker in 35 % of cases, which the player's
        own visibility cannot explain.
        """
        self.events.append({"event": "visibility", "clock": round(self.clock, 3),
                            "vehicle": vehicle.id, "flags": int(value),
                            "visibility_time": self.visibility_time.get(vehicle.id)})

    def on_visibility_time(self, vehicle, value):
        self.visibility_time[vehicle.id] = value

    def on_squadron_minimap(self, avatar, *args, **kwargs):
        """Squadron markers, from the two methods the minimap itself uses.

        Planes are NOT entities in the replay: a carrier battle's entity table holds
        only ships, zones and smoke. The squadron reaches the client as method calls
        instead, and the pair ``receive_addMinimapSquadron`` /
        ``receive_updateMinimapSquadron`` carries the position and team the minimap
        draws, which is exactly the label that is wanted.
        """
        values = list(args)
        name = self.method_name
        if name == "receive_updateMinimapSquadron" and len(values) >= 2:
            squadron, position = values[0], values[1]
            x, z = _pair(position)
            self.squadron_info.setdefault(int(squadron), {})
            self.events.append({"event": "squadron", "clock": round(self.clock, 3),
                                "squadron": int(squadron),
                                "team": self.squadron_info[int(squadron)].get("team"),
                                "params_id": self.squadron_info[int(squadron)].get("params_id"),
                                "x": x, "z": z,
                                "planes": self.squadron_info[int(squadron)].get("planes"),
                                "max_health": self.squadron_info[int(squadron)].get("max_health"),
                                "present": True})
            return
        if name == "receive_addMinimapSquadron" and len(values) >= 4:
            squadron, team, params_id, position = values[0], values[1], values[2], values[3]
            x, z = _pair(position)
            info = self.squadron_info.setdefault(int(squadron), {})
            info.update({"team": team, "params_id": params_id})
            self.events.append({"event": "squadron", "clock": round(self.clock, 3),
                                "squadron": int(squadron), "team": team,
                                "params_id": params_id, "x": x, "z": z,
                                "planes": info.get("planes"),
                                "max_health": info.get("max_health"), "present": True})
            return
        if name == "receive_removeMinimapSquadron" and values:
            squadron = int(values[0])
            self.events.append({"event": "squadron", "clock": round(self.clock, 3),
                                "squadron": squadron,
                                "team": self.squadron_info.get(squadron, {}).get("team"),
                                "params_id": self.squadron_info.get(squadron, {}).get("params_id"),
                                "x": None, "z": None,
                                "planes": self.squadron_info.get(squadron, {}).get("planes"),
                                "max_health": self.squadron_info.get(squadron, {}).get("max_health"),
                                "present": False})
            return
        if name == "receive_addSquadron":
            params_id = kwargs.get("paramsID")
            for squadron, info in self.squadron_info.items():
                if str(info.get("params_id")) == str(params_id):
                    info["planes"] = kwargs.get("totalNumPlanes")
                    info["max_health"] = kwargs.get("maxHealth")
            return

    def on_squadron_planes(self, avatar, *args, **kwargs):
        if self.method_name == "receive_planeDeath" and len(args) >= 2:
            self.events.append({"event": "plane_death", "clock": round(self.clock, 3),
                                "squadron": int(args[0]),
                                "plane_ids": [int(v) for v in args[1]],
                                "team": args[2] if len(args) > 2 else None,
                                "owner": args[3] if len(args) > 3 else None})

    def on_artillery(self, avatar, *args):
        """Salvoes, each carrying the muzzle position and the aim point.

        ``tarPos`` is where the shot was aimed, so the aim direction at the moment of
        firing IS recoverable even though the camera that draws the aim line is not.
        """
        for salvo in (args[0] if args and isinstance(args[0], list) else []):
            shots = salvo.get("shots") or []
            if not shots:
                continue
            shot = shots[0]
            position = shot.get("pos") or (None, None, None)
            aim = shot.get("tarPos") or (None, None, None)
            self.events.append({
                "event": "shot", "clock": round(self.clock, 3),
                "owner": int(salvo.get("ownerID") or 0),
                "x": position[0], "y": position[1], "z": position[2],
                "aim_x": aim[0], "aim_y": aim[1], "aim_z": aim[2],
                "pitch": shot.get("pitch"), "speed": shot.get("speed")})

    def on_waypoints(self, avatar, *args):
        if self.method_name == "receive_resetWaypoints":
            points = args[0] if args and isinstance(args[0], (list, tuple)) else []
            self.events.append({"event": "waypoints", "clock": round(self.clock, 3),
                                "count": len(points)})

    def on_alive(self, vehicle, value):
        self.events.append({"event": "alive", "clock": round(self.clock, 3),
                            "vehicle": vehicle.id, "value": value})


    def register(self):
        entity = self.entity_module
        methods = [
            ("Avatar", "receiveVehicleDeath", self.on_vehicle_death),
            ("Vehicle", "receiveDamagesOnShip", self.on_damages),
            ("Vehicle", "onConsumableUsed", self.on_consumable),
        ]
        # A method callback is invoked as func(entity, *args), so the method's own
        # name has to be bound by a closure: several squadron handlers share one
        # function and must tell each other apart.
        def named(method_name, callback):
            def invoke(entity, *args, **kwargs):
                self.method_name = method_name
                return callback(entity, *args, **kwargs)
            return invoke

        # The optional methods are added BEFORE the subscription loop: appending to
        # `methods` after it subscribes nothing, which is how the squadron export
        # first produced zero events.
        if "squadron" in self.kinds:
            for method in ("receive_addMinimapSquadron", "receive_updateMinimapSquadron",
                           "receive_removeMinimapSquadron", "receive_addSquadron"):
                methods.append(("Avatar", method, self.on_squadron_minimap))
            methods.append(("Avatar", "receive_planeDeath", self.on_squadron_planes))
        if "shot" in self.kinds:
            methods.append(("Avatar", "receiveArtilleryShots", self.on_artillery))
        if "waypoints" in self.kinds:
            methods.append(("Avatar", "receive_resetWaypoints", self.on_waypoints))
        for cls, method, callback in methods:
            subscribe(entity, "_methods_subscriptions", f"{cls}_{method}",
                      named(method, callback))
        properties = [
            ("Avatar", "ownShipId", self.on_avatar_ship),
            ("Vehicle", "health", self.on_health),
            ("Vehicle", "isAlive", self.on_alive),
        ]
        if "spotted" in self.kinds:
            properties.append(("Avatar", "spottedEntities", self.on_spotted))
        if "visibility" in self.kinds:
            properties.append(("Vehicle", "visibilityFlags", self.on_visibility_flags))
            properties.append(("Vehicle", "visibilityTime", self.on_visibility_time))
        if "zone" in self.kinds:
            for field in ("teamId", "type", "radius", "ownerId", "visualState", "name"):
                properties.append(("InteractiveZone", field,
                                   lambda zone, value, field=field:
                                   self.emit_zone(zone, value, field)))
        for cls, prop, callback in properties:
            subscribe(entity, "_properties_subscriptions", f"{cls}_{prop}", callback)

    def track_clock(self, player):
        original = player._process_packet

        def wrapper(time, packet):
            self.clock = time
            return original(time, packet)

        player._process_packet = wrapper


def _vehicle_entities(packets) -> set[int]:
    """Vehicle entity ids seen in 45-byte Position packets.

    The recording player's own ship sends PlayerPosition instead, so this set is
    only a cross-check; the authoritative set comes from the parser's entities.
    """
    return {int.from_bytes(p.payload[:4], "little", signed=True)
            for p in packets if p.type == 0x0A and len(p.payload) == 45}


def own_entity_from_packets(packets, known_entities=None):
    """The own ship's entity id, read from its own 0x2C position packets.

    The 0x2C payload begins with the sending entity's id, so the own ship names itself
    even when the roster cannot be resolved -- which is exactly the 15.9 case, where the
    roster row for the recording player has no vehicle at all.

    That stream carries MORE than the ship: measured on 236387 it holds both 371584 and
    371585, with 7909 and 7907 packets, so "the first packet" is not a usable rule. Only
    371585 is a vehicle the parser's entity table knows, and only 371585 has kill credit
    (4) and visibility flags (49) while 371584 has none of either -- i.e. the companion
    id is a phantom. So the candidate the caller recognises as a real entity wins, the
    most frequent candidate is the fallback, and an absent entity table means the caller
    gets the frequency answer rather than an exception.
    """
    counts = collections.Counter()
    for packet in packets:
        if packet.type == POSITION_PACKET and len(packet.payload) == POSITION_PAYLOAD:
            counts[int.from_bytes(packet.payload[:4], "little", signed=True)] += 1
    if not counts:
        return None
    if known_entities:
        for entity, _packets in counts.most_common():
            if entity in known_entities:
                return entity
    return counts.most_common(1)[0][0]


def cache_for_build(cache_root: Path, version: list[str]) -> Path:
    """The cached parser copy that carries this replay's version directory.

    The cache keeps one whole copy of the library per build, and each copy ships only
    the version directory it was fetched for (`<hash>_15_8_0` holds `versions/15_8_0`,
    `<hash>_15_9_0` holds `versions/15_9_0`). Loading the parser from the wrong copy is
    exactly what produced "version 15_9_0 is not supported currently": the 15.9
    controller is present, in the other directory. So the copy is chosen by the
    replay's own build, and the fallback keeps the old behaviour when no copy matches.
    """
    if len(version) < 3:
        return cache_root
    wanted = "_".join(version[:3])
    # The caller may hand us the cache ROOT or the build directory `fetch()` already
    # selected (it returns `cache/<commit>_<version>`), and both are legitimately "where
    # the copies live". Searching only the directory we were handed is what kept a 15.9
    # replay loading the 15.8 copy: the sibling `<hash>_15_9_0` sits one level up, so the
    # version directory existed on disk and was never looked at.
    for base in (cache_root, cache_root.parent):
        for candidate in sorted(base.glob("*_%s" % wanted)):
            if (candidate / "replay_unpack/clients/wows/versions" / wanted).is_dir():
                return candidate
    return cache_root


def toalba_roster(path: Path) -> list[dict]:
    """Roster rows from the toalba parser, run in its own environment.

    The two parsers live in different virtual environments on purpose (the toalba one
    was installed for the 15.9 work and must not leak into the project venv), so this is
    a subprocess rather than an import. An empty list means "no fallback available",
    which leaves the packet-level result untouched.
    """
    import subprocess
    python = Path("state/toalba/venv/bin/python")
    script = Path(__file__).resolve().parent / "export_roster_toalba.py"
    if not python.exists() or not script.exists():
        return []
    result = subprocess.run([str(python), str(script), str(path)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("  toalba roster unavailable: %s" % result.stderr.strip()[-200:])
        return []
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def roster_problem(entries, own_vehicle):
    """Say why the packet-level roster cannot carry the truth, or "" when it can.

    ``build_minimap_inventory.py`` separates teammates from enemies by team, so a
    roster with no team anywhere makes the labels untrustworthy, and one that cannot
    place the recording player leaves the own-position export without an id. The old
    test was "non-empty AND no name anywhere", which missed an empty table as well as
    every partial case.
    """
    if not entries:
        return "no roster rows"
    rows = list(entries.values())
    if not any(row.get("name") for row in rows):
        return "no name on any roster row"
    if not any(row.get("teamId") is not None for row in rows):
        return "no team on any roster row"
    if own_vehicle is None:
        return "no vehicle for the recording player"
    return ""


def export(path: Path, cache_root: Path, kinds: set[str]) -> list[dict]:
    replay = reader.read(path)
    packets = reader.walk(replay.stream)
    meta = replay.meta
    version_list = (meta.get("clientVersionFromExe") or "").split(",")
    build = ".".join(version_list[:3])
    if not build or build.startswith("."):
        raise SystemExit(f"{path.name}: replay does not report a client build")

    # Chosen before the parser is imported, because the import binds the whole library
    # to one cache copy for the rest of the process.
    cache_root = cache_for_build(cache_root, [part.strip() for part in version_list])
    entity_module, _, ReplayPlayer = _load_parser(cache_root)

    death_types = _load_death_types(cache_root, build)
    consumable_types = _load_consumable_types(cache_root)

    collector = Collector(entity_module, death_types, consumable_types)
    collector.kinds = set(kinds)
    # Reset before the controller subscribes, then append ours so neither is lost.
    reset_subscriptions(entity_module)
    player = ReplayPlayer(version_list)
    collector.register()
    collector.track_clock(player)
    player.play(replay.stream)

    collector.absorb_entities(player._battle_controller)
    observed_vehicles = collector.vehicle_ids
    roster_raw = player._battle_controller._players.get_info()

    def link_vehicle(entry):
        """Resolve a roster entry to a vehicle entity.

        The roster ``shipId`` field already holds the vehicle entity id, which covers
        players and bots alike. ``Avatar.ownShipId`` is preferred when it arrives.
        The avatar/vehicle pairing is only a last resort, and only for ids actually
        observed in the stream so an unrelated avatar cannot invent a vehicle.
        """
        avatar_id = entry.get("avatarId")
        if avatar_id is not None and collector.avatar_ship.get(avatar_id) is not None:
            return collector.avatar_ship[avatar_id], "Avatar.ownShipId"
        ship_id = entry.get("shipId")
        if isinstance(ship_id, int) and ship_id in observed_vehicles:
            return ship_id, "roster.shipId"
        if avatar_id is not None and avatar_id + 1 in observed_vehicles:
            return avatar_id + 1, "avatar+1 pairing"
        return None, "unresolved"

    links = {id(e): link_vehicle(e) for e in roster_raw.values()}

    own_vehicle_id = None
    for entry in roster_raw.values():
        if entry.get("name") != meta.get("playerName"):
            continue
        vehicle, _route = links.get(id(entry), (None, "unresolved"))
        own_vehicle_id = vehicle
        break
    routes = collections.Counter(route for _, route in links.values())
    resolved = sum(1 for vehicle, _ in links.values() if vehicle is not None)
    link_note = (f"{resolved}/{len(roster_raw)} roster entries resolved via "
                 + ", ".join(f"{route}={n}" for route, n in routes.most_common()
                             if route != "unresolved"))

    session = {
        "event": "session", "replay": path.name,
        "build": build, "map": (meta.get("mapName") or "").replace("spaces/", ""),
        "match_group": meta.get("matchGroup"), "game_mode": meta.get("gameMode"),
        "scenario": meta.get("scenario"),
        "recording_player": meta.get("playerName"),
        "recording_vehicle": meta.get("playerVehicle"),
        "players": len(roster_raw), "vehicles": len(meta.get("vehicles") or []),
        "packets": len(packets), "decoded_bytes": len(replay.stream),
        "clock_span_s": round(max((p.clock for p in packets), default=0.0), 1),
        "observed_vehicle_entities": len(observed_vehicles),
        "position_only_vehicle_entities": len(_vehicle_entities(packets)),
        "roster_link": link_note,
    }

    roster = []
    for entry in roster_raw.values():
        avatar_id = entry.get("avatarId")
        vehicle_id, route = links[id(entry)]
        roster.append({
            "event": "roster", "avatar_id": avatar_id,
            "vehicle_id": vehicle_id, "link_route": route,
            "account_id": entry.get("accountDBID"), "name": entry.get("name"),
            "ship_id": entry.get("shipId"), "team_id": entry.get("teamId"),
            "is_bot": entry.get("isBot"), "is_alive": entry.get("isAlive"),
            "max_health": entry.get("maxHealth"), "frags": entry.get("fragsCount"),
        })

    # The packet-level roster comes back unusable on 15.9, and that is what makes the
    # truth untrustworthy there: build_minimap_inventory.py separates teammates from
    # enemies by team. The toalba parser resolves the same information - and its
    # `relation` (0 self / 1 ally / 2 enemy) is the field that survives for bots, where
    # team_id does not - so it replaces the roster whenever the packet one cannot carry
    # the truth. Certified against this very output on 15.8: 15/15 names, 0 team, 0
    # health and 0 is_bot mismatches -- and on 15.8 the packet path resolves 39/39, so
    # the fallback does not trigger there.
    problem = roster_problem(roster_raw, own_vehicle_id)
    if problem:
        fallback = toalba_roster(path)
        if fallback:
            roster = fallback
            # Our own row must be re-resolved from the NEW roster. The id used to come
            # from the packet table alone, so a fallback left the own-position export
            # without a vehicle while reporting that the roster had been replaced.
            own_vehicle_id = next(
                (row["vehicle_id"] for row in fallback
                 if row.get("name") == meta.get("playerName")
                 and row.get("vehicle_id") is not None),
                next((row["vehicle_id"] for row in fallback
                      if row.get("relation") == 0 and row.get("vehicle_id") is not None),
                     own_vehicle_id))
            session["roster_link"] = (
                "%d/%d roster entries from toalba (teams derived from relation); the "
                "packet path was unusable (%s)"
                % (len(fallback), len(roster_raw), problem))
            print("  %s: packet roster unusable (%s); took %d rows from toalba"
                  % (path.name, problem, len(fallback)))
    # The own ship is not in the 0x0A position stream (it sends 0x2C), so on a build
    # whose roster does not place the recording player there is no other way in. The id
    # is read from the packet itself; a disagreement with the roster is REPORTED rather
    # than silently resolved, because the two routes are independent evidence and the
    # own position stream -- and therefore the `own` channel -- depends on this id.
    packet_own = own_entity_from_packets(packets, observed_vehicles)
    if own_vehicle_id is not None and packet_own is not None and own_vehicle_id != packet_own:
        session["own_vehicle_conflict"] = (
            "the roster says %s, the 0x2C packet says %s" % (own_vehicle_id, packet_own))
        session["own_vehicle_source"] = "roster (the 0x2C packet disagrees)"
    elif own_vehicle_id is None and packet_own is not None:
        own_vehicle_id = packet_own
        session["own_vehicle_source"] = "0x2C packet (the roster had no vehicle for us)"
    elif own_vehicle_id is not None:
        session["own_vehicle_source"] = "roster"
    else:
        session["own_vehicle_source"] = "none"
    session["own_vehicle_id"] = own_vehicle_id

    by_vehicle = {entry["vehicle_id"]: entry for entry in roster if entry["vehicle_id"]}
    for event in collector.events:
        if event["event"] in ("death", "damage"):
            victim = by_vehicle.get(event.get("victim"))
            killer = by_vehicle.get(event.get("killer") or event.get("attacker"))
            event["victim_name"] = (victim or {}).get("name")
            event["killer_name"] = (killer or {}).get("name")
        elif event["event"] in ("consumable", "health", "alive", "position", "track",
                                "visibility"):
            owner = by_vehicle.get(event.get("vehicle"))
            event["owner_name"] = (owner or {}).get("name")

    if "position" in kinds and own_vehicle_id is not None:
        collector.events.extend(own_position_events(packets, own_vehicle_id))
    if "track" in kinds:
        collector.events.extend(vehicle_track_events(packets))

    kept = [e for e in collector.events if e["event"] in kinds]
    kept.sort(key=lambda e: e["clock"])
    return [session, *roster, *kept]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replays", nargs="+", type=Path,
                        help="replay files, or directories to scan")
    parser.add_argument("--out", type=Path, default=Path("events.jsonl"))
    parser.add_argument("--build", default=DEFAULT_BUILD)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--kinds", default="death,damage,consumable,health,alive",
                        help="comma separated event kinds to keep")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true",
                        help="show upstream parser errors (unimplemented packet variants)")
    args = parser.parse_args()

    # The upstream parser logs unimplemented packet variants at ERROR level.
    logging.getLogger().setLevel(logging.INFO if args.verbose else logging.CRITICAL)

    cache_root = fetch(args.build, args.commit, args.cache, args.refresh_cache)

    targets = []
    for item in args.replays:
        if item.is_dir():
            targets.extend(sorted(item.glob("*.wowsreplay")))
        else:
            targets.append(item)
    if not targets:
        raise SystemExit("no replay files matched")

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.out.open("w", encoding="utf-8") as stream:
        for path in targets:
            records = export(path, cache_root, kinds)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            per_replay: dict[str, int] = {}
            for record in records:
                per_replay[record["event"]] = per_replay.get(record["event"], 0) + 1
            for key, value in per_replay.items():
                counts[key] = counts.get(key, 0) + value
            session = records[0]
            print(f"{path.name:<44} {session['match_group']:<12} "
                  f"{session['clock_span_s']:>7}s  " +
                  " ".join(f"{k}={v}" for k, v in sorted(per_replay.items())))

    print(f"\nwrote {args.out}")
    print("totals: " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
