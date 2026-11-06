#!/usr/bin/env python3
"""Field contracts for exported replay records, and observation staleness.

Validation is fail-closed. Every declared field is required, an unknown field is
an error, and a malformed or truncated final line is an error. Add a field to
``export_events.py`` only together with its declaration here, otherwise the
exporter and this contract disagree silently.

Game-specific meaning is deliberately absent: this module only fixes shapes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Field specification: one or more primitive names joined by "|". A trailing
# "none" admits JSON null. Booleans are never accepted as numbers.
_PRIMITIVES = {
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "num": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "str": lambda v: isinstance(v, str),
    "bool": lambda v: isinstance(v, bool),
    # A ordered list of entity ids, as the spotted-entities property delivers.
    # An empty list is a real answer: nothing is spotted at that instant.
    "list_int": lambda v: isinstance(v, list)
                          and all(isinstance(item, int) and not isinstance(item, bool)
                                  for item in v),
}

SCHEMAS: dict[str, dict[str, str]] = {
    "session": {
        "event": "str", "replay": "str", "build": "str", "map": "str",
        "match_group": "str|none", "game_mode": "num|none", "scenario": "str|none",
        "recording_player": "str|none", "recording_vehicle": "str|none",
        "players": "int", "vehicles": "int", "packets": "int", "decoded_bytes": "int",
        "clock_span_s": "num", "observed_vehicle_entities": "int",
        "position_only_vehicle_entities": "int", "roster_link": "str",
    },
    "roster": {
        "event": "str", "avatar_id": "int|none", "vehicle_id": "int|none",
        "link_route": "str", "account_id": "num|none", "name": "str|none",
        "ship_id": "num|none", "team_id": "num|none", "is_bot": "bool|none",
        "is_alive": "bool|none", "max_health": "num|none", "frags": "num|none",
    },
    "death": {
        "event": "str", "clock": "num", "victim": "int", "killer": "int|none",
        "death_type": "num|none", "death_type_name": "str|none",
        "victim_name": "str|none", "killer_name": "str|none",
    },
    "damage": {
        "event": "str", "clock": "num", "victim": "int", "attacker": "int",
        "damage": "num|none", "victim_name": "str|none", "killer_name": "str|none",
    },
    "consumable": {
        "event": "str", "clock": "num", "vehicle": "int",
        "consumable_type": "int|none", "consumable_name": "str|none",
        "work_time_left": "num|none", "raw": "str", "owner_name": "str|none",
    },
    "health": {
        "event": "str", "clock": "num", "vehicle": "int", "value": "num",
        "owner_name": "str|none",
    },
    "alive": {
        "event": "str", "clock": "num", "vehicle": "int", "value": "num",
        "owner_name": "str|none",
    },
    "position": {
        "event": "str", "clock": "num", "vehicle": "int", "x": "num", "y": "num",
        "z": "num", "yaw": "num|none", "owner_name": "str|none",
    },
    "track": {
        "event": "str", "clock": "num", "vehicle": "int", "x": "num", "z": "num",
        "yaw": "num|none", "owner_name": "str|none",
    },
    "visibility": {
        "event": "str", "clock": "num", "vehicle": "int", "flags": "num",
        "visibility_time": "num|none",
    },
    "spotted": {
        "event": "str", "clock": "num", "entities": "list_int", "count": "int",
    },
    "squadron": {
        "event": "str", "clock": "num", "squadron": "int", "team": "num|none",
        "params_id": "num|none", "x": "num|none", "z": "num|none",
        "planes": "num|none", "max_health": "num|none", "present": "bool",
    },
    "plane_death": {
        "event": "str", "clock": "num", "squadron": "int", "plane_ids": "list_int",
        "team": "num|none", "owner": "num|none",
    },
    "shot": {
        "event": "str", "clock": "num", "owner": "int", "x": "num", "y": "num",
        "z": "num", "aim_x": "num|none", "aim_y": "num|none", "aim_z": "num|none",
        "pitch": "num|none", "speed": "num|none",
    },
    "waypoints": {
        "event": "str", "clock": "num", "count": "int",
    },
    "zone": {
        "event": "str", "clock": "num", "zone": "int", "name": "str|none",
        "team_id": "num|none", "type": "num|none", "radius": "num|none",
        "owner_id": "num|none", "visual_state": "num|none",
    },
}

# Event kinds that carry a replay clock and are therefore ordered in time.
CLOCKED_KINDS = tuple(k for k, f in SCHEMAS.items() if "clock" in f and k != "session")


def _spec_problems(name: str, spec: str) -> str | None:
    parts = spec.split("|")
    if parts[-1] == "none":
        parts = parts[:-1]
    if not parts or any(part not in _PRIMITIVES for part in parts):
        return f"bad field specification for {name}: {spec}"
    return None


def validate_record(record: dict) -> list[str]:
    """Shape problems for one record. Empty list means the record conforms."""
    if not isinstance(record, dict):
        return [f"record is {type(record).__name__}, not an object"]
    kind = record.get("event")
    if not isinstance(kind, str):
        return ["record has no string 'event' field"]
    schema = SCHEMAS.get(kind)
    if schema is None:
        return [f"unknown event kind {kind!r}"]

    problems = [f"unknown field {name!r}" for name in record if name not in schema]
    for name, spec in schema.items():
        if name not in record:
            problems.append(f"missing field {name!r}")
            continue
        value = record[name]
        allowed = spec.split("|")
        if value is None and "none" in allowed:
            continue
        primitives = [p for p in allowed if p != "none"]
        if not any(_PRIMITIVES[p](value) for p in primitives):
            problems.append(
                f"field {name!r} is {type(value).__name__}, expected {'|'.join(primitives)}")
    return problems


def parse_records(path: str | Path) -> tuple[list[tuple[int, dict]], list[str]]:
    """Parse one JSONL file. Returns (line number, record) pairs plus problems.

    A line that is not a JSON object is reported with its line number. This
    catches a truncated final line, which is otherwise easy to overlook.
    """
    records: list[tuple[int, dict]] = []
    problems: list[str] = []
    with Path(path).open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                problems.append(f"line {lineno}: blank line")
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                tail = " (truncated final line?)" if "Unterminated" in str(exc) else ""
                problems.append(f"line {lineno}: invalid JSON: {exc}{tail}")
                continue
            if not isinstance(value, dict):
                problems.append(f"line {lineno}: not a JSON object")
                continue
            records.append((lineno, value))
    return records, problems


def validate_records(records: list[tuple[int, dict]]) -> list[str]:
    problems: list[str] = []
    for lineno, record in records:
        problems.extend(f"line {lineno}: {text}" for text in validate_record(record))
    return problems


def fold_duplicates(records: list[tuple[int, dict]]) -> tuple[list[tuple[int, dict]], dict[str, int]]:
    """Drop exact duplicate records, keeping the first occurrence.

    The packet stream repeats some packets verbatim, so exported rows can repeat
    as well. Counts of events must be taken from the folded set; the returned
    per-kind counts record what was folded.
    """
    unique: list[tuple[int, dict]] = []
    folded: dict[str, int] = {}
    seen: set[str] = set()
    for lineno, record in records:
        key = json.dumps(record, sort_keys=True)
        if key in seen:
            folded[record.get("event", "?")] = folded.get(record.get("event", "?"), 0) + 1
            continue
        seen.add(key)
        unique.append((lineno, record))
    return unique, folded


def self_check() -> list[str]:
    """Problems in this module's own declarations, for use by the tests."""
    problems = []
    for kind, schema in SCHEMAS.items():
        if "event" not in schema:
            problems.append(f"{kind}: schema has no 'event' field")
        for name, spec in schema.items():
            note = _spec_problems(name, spec)
            if note:
                problems.append(f"{kind}: {note}")
    return problems


# --- observation staleness -------------------------------------------------
#
# An observation is usable only while it is fresh. ``recv_clock`` is the
# observation time and ``age_bound_s`` is the known upstream delay, so the
# usable deadline depends on how late the data can arrive.
#
# The default timeout matches the measured packet cadence of the sample set
# (clock gap mode 0.10 s), so it is a derived value, not a guess.

OBSERVATION_TIMEOUT_S = 0.1


@dataclass(frozen=True)
class ObservationStamp:
    seq: int
    epoch: int
    recv_clock: float
    age_bound_s: float = 0.0
    timeout_s: float = OBSERVATION_TIMEOUT_S

    def has_age_bound(self) -> bool:
        return 0.0 <= self.age_bound_s < self.timeout_s

    @property
    def deadline_clock(self) -> float:
        return self.recv_clock + self.timeout_s - self.age_bound_s

    def age_s(self, now_clock: float) -> float:
        return now_clock - self.recv_clock + self.age_bound_s

    def fresh(self, now_clock: float) -> bool:
        return self.has_age_bound() and self.recv_clock <= now_clock < self.deadline_clock


def label_window(clock: float, ttl_s: float) -> tuple[float, float]:
    """Half-open validity window of a label observed at ``clock``."""
    if ttl_s < 0:
        raise ValueError("ttl_s must not be negative")
    return clock, clock + ttl_s
