#!/usr/bin/env python3
"""Session-level splits with a holdout that fitting code cannot reach.

Splits are assigned per session, never per event or per frame, because events
from one battle are highly correlated. Assignment is deterministic from a seed,
so the same input always yields the same split.

Fitting code must call :func:`assert_fittable` for every session it reads. The
guard is the mechanism that keeps the holdout out of fitting, so a holdout
result cannot be quoted as an independent measurement when it is not.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SPLITS = ("fit", "validation", "holdout")


def _order(sessions: list[dict], seed: str) -> list[dict]:
    return sorted(sessions, key=lambda s: hashlib.sha256(
        f"{seed}:{s['replay']}".encode()).hexdigest())


def _allocate(groups: dict[str, list[dict]], quota: int) -> list[dict]:
    """Round-robin across groups so each split keeps every group it can."""
    picked: list[dict] = []
    cursor = dict.fromkeys(groups, 0)
    keys = sorted(groups)
    while len(picked) < quota:
        progressed = False
        for key in keys:
            if len(picked) >= quota:
                break
            if cursor[key] < len(groups[key]):
                picked.append(groups[key][cursor[key]])
                cursor[key] += 1
                progressed = True
        if not progressed:
            break
    return picked


def make_splits(sessions: list[dict], holdout: int = 2, validation: int = 2,
                seed: str = "wws-layer1", stratify: str = "match_group") -> dict:
    """Assign every session to exactly one split.

    ``sessions`` holds at least ``{"replay": str}``; ``stratify`` names a field
    whose values are represented in each split when the group sizes allow it.
    """
    if not sessions:
        raise ValueError("sessions must not be empty")
    names = [s["replay"] for s in sessions]
    if len(set(names)) != len(names):
        raise ValueError("duplicate session in input")
    total = len(sessions)
    if holdout + validation >= total:
        raise ValueError("holdout and validation must leave at least one fit session")

    ordered = _order(sessions, seed)
    groups: dict[str, list[dict]] = {}
    for session in ordered:
        groups.setdefault(str(session.get(stratify, "unknown")), []).append(session)
    for values in groups.values():
        values.sort(key=lambda s: hashlib.sha256(f"{seed}:{s['replay']}".encode()).hexdigest())

    holdout_rows = _allocate(groups, holdout)
    remaining = {key: [s for s in values if s not in holdout_rows]
                 for key, values in groups.items()}
    validation_rows = _allocate(remaining, validation)
    assigned = {s["replay"]: "holdout" for s in holdout_rows}
    assigned.update({s["replay"]: "validation" for s in validation_rows})
    for session in sessions:
        assigned.setdefault(session["replay"], "fit")

    return {
        "seed": seed, "stratify": stratify, "sessions": assigned,
        "counts": {split: sum(1 for v in assigned.values() if v == split) for split in SPLITS},
        "groups": {key: sorted(s["replay"] for s in values) for key, values in groups.items()},
    }


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assert_fittable(splits: dict, replay: str) -> None:
    """Raise unless ``replay`` may be read by fitting code."""
    assigned = splits["sessions"].get(replay)
    if assigned is None:
        raise KeyError(f"session {replay!r} is absent from the split file")
    if assigned == "holdout":
        raise AssertionError(
            f"session {replay!r} is in holdout and must not be read while fitting")


def check_disjoint(splits: dict) -> list[str]:
    """Structural problems in a split file."""
    problems = []
    assigned = splits.get("sessions")
    if not isinstance(assigned, dict):
        return ["split file has no 'sessions' object"]
    unknown = {v for v in assigned.values() if v not in SPLITS}
    if unknown:
        problems.append(f"unknown split names: {sorted(unknown)}")
    counts = {split: sum(1 for v in assigned.values() if v == split) for split in SPLITS}
    for split, count in counts.items():
        if count == 0:
            problems.append(f"split {split!r} is empty")
    if counts["fit"] == 0:
        problems.append("no session can be fitted")
    declared = splits.get("counts")
    if isinstance(declared, dict) and declared != counts:
        problems.append(f"declared counts {declared} disagree with the assignment {counts}")
    return problems


def sessions_from(path: str | Path) -> list[dict]:
    """Read the session rows of an exported event file as split input."""
    sessions = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") == "session":
                sessions.append(record)
    if not sessions:
        raise ValueError(f"{path}: no session rows")
    return sessions


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("events", type=Path, help="exported events JSONL to take sessions from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--validation", type=int, default=2)
    parser.add_argument("--seed", default="wws-layer1")
    parser.add_argument("--stratify", default="match_group")
    args = parser.parse_args()

    splits = make_splits(sessions_from(args.events), args.holdout, args.validation,
                         args.seed, args.stratify)
    problems = check_disjoint(splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    for replay, split in sorted(splits["sessions"].items(), key=lambda kv: kv[1]):
        print(f"{split:<11} {replay}")
    print("counts " + " ".join(f"{k}={v}" for k, v in splits["counts"].items()))
    if problems:
        raise SystemExit("split problems: " + "; ".join(problems))


if __name__ == "__main__":
    main()
