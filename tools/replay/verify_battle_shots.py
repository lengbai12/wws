#!/usr/bin/env python3
"""Verify main-battery fire, hits, and damage from a WoWS replay.

The report uses independent parser surfaces:

* OwnShip packet 0x20 proves recording-vehicle ownership; no avatar arithmetic.
* receiveArtilleryShots plus an exact same-clock shootOnClient weapon_type=0
  callback proves main shells; competing callbacks remain unclassified.
* the own_db_id post-battle result row provides whole-battle main hits/damage;
  differing header/database IDs require the arena row to bind both to OwnShip.
  DamageEvent.entity_id is a receiver, so it is never summed as damage dealt.

CN build 13357822 has been inspected with the installed parser and existing
entity definitions without changing the gamedata checkout. Each report hashes
the replay and entity schema. A schema hash records provenance, not proof that
other client builds have identical field semantics.

Capture correlation requires explicit increasing replay_clock_s/capture_wall_s
anchors; one anchor assumes 1x playback, multiple anchors interpolate linearly.
--run-start-wall/--run-end-wall restrict actual emitted-shell evidence to a run;
--target-accepted-wall evaluates the five-second requirement. Result hit/damage
counters stay battle-wide because no per-shot attacker/hit mapping is assumed.
Saved evidence JSON exposes reported HP values and checks image hashes, but does
not turn supplied HP numbers or a fired flag into independent combat evidence.

Run with the parser environment (one replay at a time):

    state/toalba/venv/bin/python tools/replay/verify_battle_shots.py \
      replays/236587.wowsreplay \
      --gamedata state/toalba/gamedata/data/scripts_entity/entity_defs

A replay that has no result packet is reported as ``partial`` with explicit
missing prerequisites; it is never reported as a successful hit/damage check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GAMEDATA = "state/toalba/gamedata/data/scripts_entity/entity_defs"
MAIN_WEAPON_FIELDS = {
    "shots": ("shots_main_ap", "shots_main_cs", "shots_main_he"),
    "hits": ("hits_main_ap", "hits_main_cs", "hits_main_he"),
    "damage": ("damage_main_ap", "damage_main_cs", "damage_main_he"),
}
SECONDARY_WEAPON_FIELDS = {
    "shots": ("shots_atba_ap", "shots_atba_cs", "shots_atba_he"),
    "hits": ("hits_atba_ap", "hits_atba_cs", "hits_atba_he"),
    "damage": ("damage_atba_ap", "damage_atba_cs", "damage_atba_he"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gamedata_provenance(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    root = Path(path)
    if not root.is_dir():
        return {"path": str(root), "status": "missing"}
    digest = hashlib.sha256()
    count = 0
    for item in sorted(root.rglob("*")):
        if item.is_file() and item.suffix in (".def", ".xml"):
            digest.update(item.relative_to(root).as_posix().encode("utf-8") + b"\0")
            digest.update(bytes.fromhex(_sha256(item)))
            count += 1
    return {"path": str(root), "sha256": digest.hexdigest(), "schema_files": count, "hash_format": "sorted relative UTF-8 path + NUL + SHA256 bytes per .def/.xml"}


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _sum_fields(stats: dict[str, Any], fields: tuple[str, ...]) -> int | float | None:
    values = [stats.get(name) for name in fields if stats.get(name) is not None]
    if not values or any(not _finite_number(value) or value < 0 for value in values):
        return None
    return sum(float(value) for value in values)


def _by_ammo(stats: dict[str, Any], prefix: str) -> dict[str, int | float | None]:
    return {ammo: stats.get(f"{prefix}_main_{ammo.lower()}") for ammo in ("AP", "CS", "HE")}


def _player_for_recording(replay: Any) -> Any | None:
    return next((player for player in getattr(replay, "players", ()) if getattr(player, "relation", None) == 0), None)



def _result_row(replay: Any, player: Any, gamedata_path: str | None = None) -> tuple[Any | None, str | None, dict[str, Any]]:
    identity: dict[str, Any] = {"header_account_id": getattr(player, "account_id", None)}
    try:
        results = replay.battle_results()
    except Exception as exc:
        return None, f"battle_results decode failed: {type(exc).__name__}: {exc}", identity
    if results is None:
        return None, "replay has no post-battle results packet", identity
    own_db_id = getattr(results, "own_db_id", None)
    identity["result_own_db_id"] = own_db_id
    account_id = getattr(player, "account_id", None)
    if not account_id:
        return None, "recording-player header account ID is missing", identity
    if own_db_id != account_id:
        # CN's replay header id and database accountDBID are distinct namespaces.
        # Reuse the installed parser's wire decoder to bind both IDs to OwnShip.
        try:
            from wows_replay_parser.roster import _decode_arena_players, _extract_arena_blobs, _find_arena_state_packet, _load_key_maps
            packet = _find_arena_state_packet(getattr(replay, "packets", []))
            blobs = _extract_arena_blobs(packet) if packet is not None else None
            if blobs is None:
                return None, "different header/database IDs require onArenaStateReceived identity evidence", identity
            player_keys, _ = _load_key_maps(Path(gamedata_path) if gamedata_path is not None else None)
            arena_players = _decode_arena_players(blobs[1], player_keys)
            own_vehicle = getattr(getattr(replay, "tracker", None), "own_vehicle_id", None)
            matches = [row for row in arena_players if row.get("id") == account_id and row.get("shipId") == own_vehicle]
        except Exception as exc:
            return None, f"arena account database identity decode failed: {type(exc).__name__}: {exc}", identity
        if len(matches) != 1:
            return None, "unique arena header-id/OwnShip identity binding is unavailable", identity
        arena = matches[0]
        identity.update({"arena_id": arena.get("id"), "arena_account_db_id": arena.get("accountDBID"), "arena_ship_id": arena.get("shipId"), "evidence": "onArenaStateReceived playersStates: id == recording header id; shipId == OwnShip; accountDBID == BattleResults accountDBID"})
        if arena.get("accountDBID") != own_db_id:
            return None, "BattleResults own_db_id disagrees with OwnShip-bound arena accountDBID", identity
    else:
        identity["evidence"] = "recording-player account_id equals BattleResults accountDBID"
    replay_arena = getattr(replay, "meta", {}).get("arenaUniqueId")
    result_arena = getattr(results, "arena_unique_id", None)
    identity.update({"replay_arena_unique_id": replay_arena, "result_arena_unique_id": result_arena})
    if replay_arena is not None and result_arena is not None and replay_arena != result_arena:
        return None, "BattleResults arenaUniqueID disagrees with replay arenaUniqueId", identity
    players = getattr(results, "players", {})
    if own_db_id not in players:
        return None, "post-battle results lack the recording player's own_db_id row", identity
    return players[own_db_id], None, identity


def _capture_correlation(path: str | None, first_s: float | None, last_s: float | None, shell_times: list[float] | None = None) -> dict[str, Any]:
    """Correlate replay event time only when an explicit clock anchor exists.

    A recording session's ``wall_start`` alone does not identify replay t=0;
    treating it as such would produce a fabricated correlation.  Callers may
    provide ``anchors`` with ``replay_clock_s`` and ``capture_wall_s`` (or a
    single pair with those names) in a JSON file.
    """
    out: dict[str, Any] = {"status": "unknown", "reason": "no explicit replay/capture clock anchor"}
    if not path:
        return out
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unknown", "reason": f"capture clock unreadable: {type(exc).__name__}: {exc}"}
    anchors = data.get("anchors") if isinstance(data, dict) else None
    if anchors is None and isinstance(data, dict) and "replay_clock_s" in data and "capture_wall_s" in data:
        anchors = [data]
    if not isinstance(anchors, list):
        return {"status": "unknown", "reason": "capture file has no explicit anchors"}
    clean = [a for a in anchors if isinstance(a, dict) and _finite_number(a.get("replay_clock_s")) and _finite_number(a.get("capture_wall_s"))]
    if not clean:
        return {"status": "unknown", "reason": "capture file has no usable replay_clock_s/capture_wall_s anchors"}
    if len(clean) != len(anchors):
        return {"status": "unknown", "reason": "capture file contains invalid anchors"}
    if any(float(left["replay_clock_s"]) >= float(right["replay_clock_s"]) or float(left["capture_wall_s"]) >= float(right["capture_wall_s"]) for left, right in zip(clean, clean[1:])):
        return {"status": "unknown", "reason": "capture anchors are duplicate or regressing"}

    def map_time(clock: float | None) -> float | None:
        if clock is None:
            return None
        if len(clean) == 1:
            a = clean[0]
            return float(a["capture_wall_s"]) + (clock - float(a["replay_clock_s"]))
        # Piecewise linear mapping; extrapolation is allowed only at the ends,
        # because the caller supplied the clock model explicitly.
        if clock <= float(clean[0]["replay_clock_s"]):
            left, right = clean[0], clean[1]
        elif clock >= float(clean[-1]["replay_clock_s"]):
            left, right = clean[-2], clean[-1]
        else:
            left = clean[0]
            right = clean[1]
            for candidate_left, candidate_right in zip(clean, clean[1:]):
                if float(candidate_left["replay_clock_s"]) <= clock <= float(candidate_right["replay_clock_s"]):
                    left, right = candidate_left, candidate_right
                    break
        x0, x1 = float(left["replay_clock_s"]), float(right["replay_clock_s"])
        y0, y1 = float(left["capture_wall_s"]), float(right["capture_wall_s"])
        if x1 == x0:
            return None
        return y0 + (clock - x0) * (y1 - y0) / (x1 - x0)

    return {
        "status": "correlated",
        "anchor_count": len(clean),
        "first_shell_capture_wall_s": map_time(first_s),
        "last_shell_capture_wall_s": map_time(last_s),
        "shell_capture_wall_s": [map_time(clock) for clock in shell_times or []],
        "clock_source": {"path": path, "sha256": _sha256(Path(path))},
    }


def verify_parsed_replay(
    replay: Any,
    *,
    source_path: str | Path | None = None,
    gamedata_path: str | None = None,
    parser_name: str = "wows_replay_parser",
    capture_clock_path: str | None = None,
) -> dict[str, Any]:
    """Build a provenance-rich report from one already parsed replay."""
    player = _player_for_recording(replay)
    missing: list[str] = []
    source: dict[str, Any] = {"parser": parser_name, "gamedata": gamedata_path}
    if source_path is not None:
        path = Path(source_path)
        source.update({"path": str(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    source.update({"game_version": str(getattr(replay, "game_version", "unknown")), "duration_s": getattr(replay, "duration", None)})
    source["gamedata_provenance"] = _gamedata_provenance(gamedata_path)
    if player is None:
        missing.append("recording player (relation == 0)")
        return {"status": "unknown", "source": source, "missing_prerequisites": missing}
    try:
        from wows_replay_parser.events.models import DamageReceivedStatEvent, GunFireEvent, SecondaryFireEvent, ShotCreatedEvent
    except ImportError as exc:
        return {"status": "unknown", "source": source, "missing_prerequisites": [f"parser event models unavailable: {exc}"]}
    events = list(getattr(replay, "events", ()))
    tracker_vehicle_id = getattr(getattr(replay, "tracker", None), "own_vehicle_id", None)
    candidate_ids = {int(tracker_vehicle_id)} if isinstance(tracker_vehicle_id, int) and tracker_vehicle_id > 0 else set()
    if not candidate_ids:
        missing.append("tracker own_vehicle_id (cannot prove projectile owner identity)")
        return {"status": "unknown", "source": source, "missing_prerequisites": missing}
    event_ids = candidate_ids
    main_salvos = [e for e in events if isinstance(e, GunFireEvent) and e.entity_id in event_ids and getattr(e, "weapon_type", None) == 0]
    secondary_salvos = [e for e in events if isinstance(e, SecondaryFireEvent) and e.entity_id in event_ids]
    all_owner_shells = [e for e in events if isinstance(e, ShotCreatedEvent) and e.owner_id in event_ids]
    main_times = {float(e.timestamp) for e in main_salvos}
    secondary_times = {float(e.timestamp) for e in secondary_salvos}
    other_times = {float(e.timestamp) for e in events if isinstance(e, GunFireEvent) and e.entity_id in event_ids and getattr(e, "weapon_type", None) != 0}
    shells = [e for e in all_owner_shells if float(e.timestamp) in main_times and float(e.timestamp) not in secondary_times | other_times]
    secondary_shells = [e for e in all_owner_shells if float(e.timestamp) in secondary_times and float(e.timestamp) not in main_times | other_times]
    classified_ids = {id(e) for e in shells + secondary_shells}
    unclassified_shells = [e for e in all_owner_shells if id(e) not in classified_ids]
    if unclassified_shells:
        missing.append("unambiguous main-vs-secondary callback correlation for some owner-created shells")
    first_shell = min((float(e.timestamp) for e in shells), default=None)
    last_shell = max((float(e.timestamp) for e in shells), default=None)

    result_row, result_error, result_identity = _result_row(replay, player, gamedata_path)
    if result_row is None:
        missing.append(result_error or "own post-battle result row")
        stats: dict[str, Any] = {}
    else:
        stats = dict(getattr(result_row, "stats", {}) or {})

    main_hits = _sum_fields(stats, MAIN_WEAPON_FIELDS["hits"])
    main_damage = _sum_fields(stats, MAIN_WEAPON_FIELDS["damage"])
    result_shots = _sum_fields(stats, MAIN_WEAPON_FIELDS["shots"])
    secondary_hits = _sum_fields(stats, SECONDARY_WEAPON_FIELDS["hits"])
    secondary_damage = _sum_fields(stats, SECONDARY_WEAPON_FIELDS["damage"])
    event_stats: dict[str, dict[str, int | float]] = defaultdict(lambda: {"count_field": 0, "damage": 0.0})
    for event in events:
        if isinstance(event, DamageReceivedStatEvent) and event.entity_id in event_ids and event.stat_type == "ENEMY":
            if event.damage_param in ("MAIN_AP", "MAIN_CS", "MAIN_HE"):
                event_stats[event.damage_param]["count_field"] = max(event_stats[event.damage_param]["count_field"], int(event.cumulative_count))
                event_stats[event.damage_param]["damage"] = max(event_stats[event.damage_param]["damage"], float(event.cumulative_total))

    if not shells:
        missing.append("recording player's ShotCreatedEvent shells")
    if main_hits is None:
        missing.append("own result main-battery hit fields")
    if main_damage is None:
        missing.append("own result main-battery damage fields")
    if result_shots is not None and not unclassified_shells and result_shots != len(shells):
        missing.append("main shell events disagree with BattleResults shots_main_*; stream completeness or field semantics unresolved")

    report = {
        "status": "verified" if not missing else "partial",
        "source": source,
        "ownership": {
            "player_name": getattr(player, "name", None),
            "relation": getattr(player, "relation", None),
            "avatar_id": getattr(player, "avatar_id", None),
            "vehicle_event_ids": sorted(event_ids),
            "battle_results_identity": result_identity,
            "evidence": "tracker.own_vehicle_id decoded from OwnShip packet 0x20; relation == 0 identifies recording player",
        },
        "main_battery": {
            "salvos_fired": len(main_salvos),
            "shells_fired": len(shells) if not unclassified_shells else None,
            "confirmed_shells_fired": len(shells),
            "first_shell_replay_s": first_shell,
            "last_shell_replay_s": last_shell,
            "shell_events": [{"replay_s": float(e.timestamp), "shot_id": getattr(e, "shot_id", None), "owner_id": e.owner_id, "params_id": getattr(e, "params_id", None)} for e in shells],
            "hits": int(main_hits) if main_hits is not None else None,
            "damage": float(main_damage) if main_damage is not None else None,
            "result_shots_crosscheck": int(result_shots) if result_shots is not None else None,
            "by_ammo": {"shots": _by_ammo(stats, "shots"), "hits": _by_ammo(stats, "hits"), "damage": _by_ammo(stats, "damage")},
            "evidence": [
                "receiveArtilleryShots shell owner == OwnShip vehicle; exact timestamp matches weapon_type=0 GunFireEvent, with no secondary/non-main callback at that timestamp",
                "GunFireEvent.entity_id == OwnShip vehicle and weapon_type == 0 (main-battery salvo callback count)",
                "BattleResults own player row: hits_main_* and damage_main_*",
            ],
        },
        "secondary_battery": {
            "salvos_fired": len(secondary_salvos),
            "shells_fired": len(secondary_shells) if not unclassified_shells else None,
            "hits": int(secondary_hits) if secondary_hits is not None else None,
            "damage": float(secondary_damage) if secondary_damage is not None else None,
            "evidence": "SecondaryFireEvent and shots_atba_*/hits_atba_*/damage_atba_* are kept separate",
        },
        "event_crosscheck": {
            "main_damage_stat_by_ammo": dict(event_stats),
            "stat_count_semantics": "raw cumulative_count; not interpreted as main-shell hits (CN sample disagrees with hits_main_*)",
            "owner_created_shells": len(all_owner_shells),
            "unclassified_owner_shells": [{"replay_s": float(e.timestamp), "shot_id": getattr(e, "shot_id", None), "params_id": getattr(e, "params_id", None)} for e in unclassified_shells],
            "damage_events_excluded": "DamageEvent attacker_id/entity_id does not identify recording-player ownership in this replay stream",
        },
        "capture_correlation": _capture_correlation(capture_clock_path, first_shell, last_shell, [float(e.timestamp) for e in shells]),
        "missing_prerequisites": missing,
    }
    return report


def verify_saved_evidence(path: str | Path) -> dict[str, Any]:
    """Read reported HUD measurements and check frame hashes without inventing shot evidence."""
    evidence_path = Path(path)
    source: dict[str, Any] = {"path": str(evidence_path), "kind": "saved_training_evidence"}
    try:
        source["sha256"] = _sha256(evidence_path)
        data = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "unknown", "source": source, "missing_prerequisites": [f"evidence JSON unreadable: {type(exc).__name__}: {exc}"]}
    if not isinstance(data, dict):
        return {"status": "unknown", "source": source, "missing_prerequisites": ["evidence JSON root is not an object"]}
    hashes = data.get("hashes") if isinstance(data.get("hashes"), dict) else {}
    shot = data.get("shot") if isinstance(data.get("shot"), dict) else {}
    hp = data.get("hp") if isinstance(data.get("hp"), dict) else {}
    frames = []
    for key in ("before_frame", "after_frame"):
        name = hp.get(key)
        expected = hashes.get(name) if isinstance(name, str) else None
        frame = Path(name) if isinstance(name, str) else None
        if frame is not None and not frame.is_absolute() and not frame.is_file():
            frame = evidence_path.parent / frame
        try:
            actual = _sha256(frame) if frame is not None else None
        except OSError:
            actual = None
        frames.append({"field": key, "path": str(frame) if frame is not None else None, "expected_sha256": expected, "actual_sha256": actual, "hash_matches": actual is not None and actual == expected})
    missing = [
        "matching replay parsed with build-specific gamedata for independent shell/hit/attacker evidence",
        "shot event source (ShotCreatedEvent/GunFireEvent); evidence fired flag is only a claim",
        "independent hit event or own BattleResults hits_main_* fields",
        "independent image HP decoding; supplied before/after numbers remain reported measurements",
    ]
    if not all(frame["hash_matches"] for frame in frames):
        missing.append("readable before/after image files matching supplied SHA256 hashes")
    reported_delta = None
    if _finite_number(hp.get("before")) and _finite_number(hp.get("after")):
        reported_delta = float(hp["before"]) - float(hp["after"])
    return {
        "status": "partial",
        "source": source,
        "battle": data.get("battle"),
        "shot_claim": {"fired": shot.get("fired"), "reason": shot.get("fire_reason", shot.get("reason")), "report": shot.get("report")},
        "reported_target_hp": {"before": hp.get("before"), "after": hp.get("after"), "damage_delta": reported_delta, "frames": frames, "independently_decoded": False},
        "main_battery": {"salvos_fired": None, "shells_fired": None, "hits": None, "damage": None, "evidence": "saved JSON claims and image provenance only"},
        "missing_prerequisites": missing,
    }


def correlate_run(report: dict[str, Any], start_wall: float, end_wall: float, target_accepted_wall: float | None = None) -> dict[str, Any]:
    """Select actual main shell events within a capture-wall interval."""
    if not _finite_number(start_wall) or not _finite_number(end_wall) or start_wall >= end_wall:
        return {"status": "unknown", "reason": "run wall-clock bounds must be finite and increasing"}
    if target_accepted_wall is not None and (not _finite_number(target_accepted_wall) or not start_wall <= target_accepted_wall <= end_wall):
        return {"status": "unknown", "reason": "target acceptance must fall within run bounds"}
    correlation = report.get("capture_correlation", {})
    if correlation.get("status") != "correlated":
        return {"status": "unknown", "reason": "run requires explicit valid replay/capture clock anchors"}
    main = report.get("main_battery", {})
    events = main.get("shell_events", [])
    times = correlation.get("shell_capture_wall_s", [])
    if len(events) != len(times):
        return {"status": "unknown", "reason": "shell event clock mapping is incomplete"}
    selected = [{**event, "capture_wall_s": wall} for event, wall in zip(events, times) if _finite_number(wall) and start_wall <= wall <= end_wall]
    out = {"status": "verified" if main.get("shells_fired") is not None else "partial", "start_wall_s": start_wall, "end_wall_s": end_wall, "confirmed_main_shells_fired": len(selected), "shell_events": selected, "main_hits": None, "main_damage": None, "missing_prerequisites": ["per-shot hit/damage attribution; BattleResults counts cover the whole battle"]}
    if target_accepted_wall is not None:
        after = [event["capture_wall_s"] - target_accepted_wall for event in selected if event["capture_wall_s"] >= target_accepted_wall]
        delay = min(after, default=None)
        complete_window = report.get("status") == "verified" and main.get("shells_fired") is not None and end_wall >= target_accepted_wall + 5
        within_five = True if delay is not None and delay <= 5.0 else (False if complete_window else None)
        out["target_acceptance"] = {"capture_wall_s": target_accepted_wall, "first_actual_main_shell_delay_s": delay, "actual_fire_within_5s": within_five}
    return out


def verify_replay(path: str, *, gamedata_path: str, capture_clock_path: str | None = None) -> dict[str, Any]:
    try:
        from wows_replay_parser import parse_replay
    except ImportError as exc:
        return {"status": "unknown", "source": {"path": path}, "missing_prerequisites": [f"run with parser venv: {exc}"]}
    try:
        replay = parse_replay(replay_path=path, gamedata_path=gamedata_path)
    except Exception as exc:
        return {"status": "unknown", "source": {"path": path}, "missing_prerequisites": [f"replay parse failed: {type(exc).__name__}: {exc}"]}
    return verify_parsed_replay(replay, source_path=path, gamedata_path=gamedata_path, capture_clock_path=capture_clock_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="one replay or saved evidence JSON")
    parser.add_argument("--saved-evidence", action="store_true", help="interpret source as saved training evidence JSON")
    parser.add_argument("--gamedata", default=DEFAULT_GAMEDATA)
    parser.add_argument("--capture-clock", help="JSON with explicit replay/capture clock anchors")
    parser.add_argument("--out", help="write the JSON report to this path")
    parser.add_argument("--run-start-wall", type=float, help="capture-wall seconds at run start; requires --run-end-wall and explicit clock anchors")
    parser.add_argument("--run-end-wall", type=float, help="capture-wall seconds at run end")
    parser.add_argument("--target-accepted-wall", type=float, help="capture-wall seconds of target acceptance, for actual fire within five seconds")
    args = parser.parse_args(argv)
    if (args.run_start_wall is None) != (args.run_end_wall is None) or (args.target_accepted_wall is not None and args.run_start_wall is None):
        parser.error("run correlation requires both --run-start-wall and --run-end-wall")
    report = (verify_saved_evidence(args.source) if args.saved_evidence else verify_replay(args.source, gamedata_path=args.gamedata, capture_clock_path=args.capture_clock))
    if args.run_start_wall is not None:
        report["run_evidence"] = correlate_run(report, args.run_start_wall, args.run_end_wall, args.target_accepted_wall)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if args.out:
        Path(args.out).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report.get("status") == "verified" and ("run_evidence" not in report or report["run_evidence"]["status"] == "verified") else 2


if __name__ == "__main__":
    sys.exit(main())
