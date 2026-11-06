#!/usr/bin/env python3
"""Check exported replay records against their contracts, order, and provenance.

Runs every layer-1 check over one or more exported JSONL files and writes a JSON
report. Exits non-zero when an error is found, so it can gate a pipeline.

Errors are violations of a declared rule. Notes are facts worth reading that do
not violate a rule, such as a record whose value is legitimately absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import contracts
import splits as split_tools
import staleness

# A session's declared span may be shorter than the last observed clock by less
# than this tolerance, because the span comes from packet clocks rounded to 0.1 s.
SPAN_TOLERANCE_S = 1.0


def _blocks(records: list[tuple[int, dict]]) -> tuple[dict[str, list[tuple[int, dict]]], list[str]]:
    """Group records by replay. Rows before the first session row are an error."""
    blocks: dict[str, list[tuple[int, dict]]] = {}
    problems: list[str] = []
    current: str | None = None
    for lineno, record in records:
        if record.get("event") == "session":
            current = record.get("replay")
            blocks.setdefault(current, [])
            continue
        if current is None:
            problems.append(f"line {lineno}: {record.get('event')} record before any session row")
            continue
        blocks[current].append((lineno, record))
    return blocks, problems


def _check_contracts(records: list[tuple[int, dict]], source: str) -> list[str]:
    return [f"{source}: {text}" for text in contracts.validate_records(records)]


def _check_order(blocks: dict[str, list[tuple[int, dict]]], source: str) -> tuple[list[str], list[str]]:
    """Clock monotonicity within each replay, plus its last observed clock."""
    errors: list[str] = []
    notes: list[str] = []
    for replay, rows in blocks.items():
        previous_clock: float | None = None
        for lineno, record in rows:
            if record.get("event") not in contracts.CLOCKED_KINDS:
                continue
            clock = float(record["clock"])
            if previous_clock is not None and clock < previous_clock:
                errors.append(
                    f"{source}: {replay}: line {lineno} clock {clock} precedes {previous_clock}")
            previous_clock = clock
        clocked = [float(r["clock"]) for _, r in rows if r.get("event") in contracts.CLOCKED_KINDS]
        if clocked:
            notes.append(f"{source}: {replay}: last clock {max(clocked):.1f}s")
    return errors, notes


def _check_duplicates(blocks: dict[str, list[tuple[int, dict]]],
                      source: str) -> tuple[list[str], dict[str, int]]:
    """Per-replay counts of exact duplicate rows.

    Duplicates are a data-quality fact, not a defect: the packet stream repeats
    some packets verbatim, which was measured directly. They are reported so
    count-based results can be taken from the folded set instead.
    """
    notes: list[str] = []
    counts: dict[str, int] = {}
    for replay, rows in blocks.items():
        folded: dict[str, int] = {}
        seen: set[str] = set()
        for _, record in rows:
            key = json.dumps(record, sort_keys=True)
            if key in seen:
                folded[record.get("event", "?")] = folded.get(record.get("event", "?"), 0) + 1
                continue
            seen.add(key)
        counts[replay] = sum(folded.values())
        if folded:
            detail = ", ".join(f"{k}={v}" for k, v in sorted(folded.items()))
            notes.append(f"{source}: {replay}: {counts[replay]} duplicate rows ({detail})")
    return notes, counts


def _check_provenance(records: list[tuple[int, dict]], source: str) -> tuple[list[str], list[str]]:
    """Every row must be attributable to a session that precedes it."""
    errors: list[str] = []
    notes: list[str] = []
    sessions = [r for _, r in records if r.get("event") == "session"]
    builds = {s.get("build") for s in sessions}
    if len(builds) > 1:
        errors.append(f"{source}: mixed client builds in one file: {sorted(builds)}")
    names = [s.get("replay") for s in sessions]
    if len(set(names)) != len(names):
        errors.append(f"{source}: a session row repeats: {sorted(names)}")
    notes.append(f"{source}: {len(sessions)} sessions, builds {sorted(builds)}")

    routes: dict[str, int] = {}
    unresolved: list[str] = []
    for _, record in records:
        if record.get("event") != "roster":
            continue
        route = str(record.get("link_route"))
        routes[route] = routes.get(route, 0) + 1
        if record.get("vehicle_id") is None:
            unresolved.append(str(record.get("name")))
    notes.append(f"{source}: link routes {sorted(routes.items())}")
    if unresolved:
        notes.append(f"{source}: {len(unresolved)} roster rows without a vehicle: {unresolved}")

    unnamed = [r for _, r in records if r.get("event") == "consumable"
               and r.get("consumable_name") is None]
    if unnamed:
        notes.append(f"{source}: {len(unnamed)} consumable rows with no decoded name: "
                     + "; ".join(f"line {n} raw={r.get('raw')}" for n, r in
                                 [(n, r) for n, r in records
                                  if r.get("event") == "consumable"
                                  and r.get("consumable_name") is None]))
    unlabelled = [r for _, r in records if r.get("event") == "death"
                  and r.get("victim_name") is None]
    if unlabelled:
        notes.append(f"{source}: {len(unlabelled)} death rows without a victim name")
    return errors, notes


def _check_span(blocks: dict[str, list[tuple[int, dict]]],
                sessions: dict[str, dict], source: str) -> list[str]:
    errors = []
    for replay, rows in blocks.items():
        declared = sessions[replay].get("clock_span_s")
        clocked = [float(r["clock"]) for _, r in rows if r.get("event") in contracts.CLOCKED_KINDS]
        if not clocked or declared is None:
            continue
        latest = max(clocked)
        if latest > float(declared) + SPAN_TOLERANCE_S:
            errors.append(
                f"{source}: {replay}: event clock {latest:.1f}s exceeds declared span "
                f"{float(declared):.1f}s by more than {SPAN_TOLERANCE_S}s")
    return errors


def _check_ttl(blocks: dict[str, list[tuple[int, dict]]], source: str) -> tuple[list[str], list[str]]:
    """Derive a lifetime per kind and prove the overlap check can fail.

    Two rules are enforced. First, the safe lifetime (half the minimum spacing)
    must produce no overlap. Second, a deliberately oversized lifetime MUST
    produce an overlap: a check that cannot fail proves nothing.
    """
    errors: list[str] = []
    notes: list[str] = []
    for replay, rows in blocks.items():
        events = [record for _, record in rows]
        safe: dict[str, float] = {}
        for kind in staleness.SUBJECT_FIELDS:
            if not any(event.get("event") == kind for event in events):
                continue
            report = staleness.recommend_ttl(events, kind)
            notes.append(
                f"{source}: {replay}: {kind} gaps={report['gaps']} "
                f"min={report['min_gap_s']} typical={report['quantile_gap_s']} "
                f"safe_ttl={report['safe_ttl_s']} simultaneous={report['simultaneous']}")
            if report["min_gap_s"] == 0:
                notes.append(
                    f"{source}: {replay}: {kind} has same-subject events at one clock, "
                    "so no positive lifetime is safe for it")
            if report["safe_ttl_s"]:
                safe[kind] = report["safe_ttl_s"]
        if not safe:
            continue
        violations = staleness.coverage_violations(events, safe)
        if violations:
            errors.append(
                f"{source}: {replay}: safe lifetime overlaps a later event, "
                f"e.g. {violations[0]}")
        oversized = {kind: max(ttl * 4.0, 1.0) for kind, ttl in safe.items()}
        if not staleness.coverage_violations(events, oversized):
            errors.append(
                f"{source}: {replay}: the overlap check detects nothing at an oversized "
                f"lifetime {oversized}, so it cannot detect a real overlap")
    return errors, notes


def _totals(records: list[tuple[int, dict]], source: str) -> tuple[list[str], dict[str, dict]]:
    """Raw and folded per-kind counts. Use the folded counts for any metric."""
    raw: dict[str, int] = {}
    for _, record in records:
        raw[record.get("event", "?")] = raw.get(record.get("event", "?"), 0) + 1
    _, folded = contracts.fold_duplicates(records)
    unique = {kind: count - folded.get(kind, 0) for kind, count in raw.items()}
    notes = [f"{source}: totals raw={dict(sorted(raw.items()))} "
             f"folded={dict(sorted(unique.items()))}"]
    return notes, {"raw": dict(sorted(raw.items())), "folded": dict(sorted(unique.items()))}


def duplicate_packet_budget(directory: Path, replays: list[str]) -> dict[str, int]:
    """Extra duplicate ``EntityProperty`` and ``EntityMethod`` packets per replay.

    A duplicate row must be explained by a duplicate packet. The count is a
    ceiling, so a row count above it proves the exporter produced rows that the
    stream does not contain.
    """
    import reader

    budget: dict[str, int] = {}
    for replay in replays:
        path = directory / replay
        if not path.is_file():
            continue
        packets = reader.walk(reader.read(path).stream)
        identities: dict[tuple, int] = {}
        for packet in packets:
            if packet.type in (0x07, 0x08):
                key = (packet.type, round(packet.clock, 3), packet.payload)
                identities[key] = identities.get(key, 0) + 1
        budget[replay] = sum(n - 1 for n in identities.values() if n > 1)
    return budget


def run(paths: list[Path], splits_path: Path | None,
        packet_budget: dict[str, int] | None = None) -> dict:
    errors: list[str] = []
    notes: list[str] = []
    totals: dict[str, dict] = {}
    sessions_by_source: dict[str, dict[str, dict]] = {}
    for path in paths:
        source = path.name
        records, parse_problems = contracts.parse_records(path)
        errors.extend(f"{source}: {text}" for text in parse_problems)
        errors.extend(_check_contracts(records, source))
        blocks, block_problems = _blocks(records)
        errors.extend(f"{source}: {text}" for text in block_problems)
        order_errors, order_notes = _check_order(blocks, source)
        errors.extend(order_errors)
        notes.extend(order_notes)
        duplicate_notes, duplicate_counts = _check_duplicates(blocks, source)
        notes.extend(duplicate_notes)
        if packet_budget is not None:
            for replay, rows in duplicate_counts.items():
                limit = packet_budget.get(replay)
                if limit is not None and rows > limit:
                    errors.append(
                        f"{source}: {replay}: {rows} duplicate rows exceed the "
                        f"{limit} duplicate packets available in the stream")
        provenance_errors, provenance_notes = _check_provenance(records, source)
        errors.extend(provenance_errors)
        notes.extend(provenance_notes)
        sessions = {r["replay"]: r for _, r in records if r.get("event") == "session"}
        sessions_by_source[source] = sessions
        errors.extend(_check_span(blocks, sessions, source))
        # Lifetime evidence is taken from the folded rows, because a repeated row
        # would otherwise look like two same-subject events at one clock.
        folded_blocks = {replay: contracts.fold_duplicates(rows)[0]
                         for replay, rows in blocks.items()}
        ttl_errors, ttl_notes = _check_ttl(folded_blocks, source)
        errors.extend(ttl_errors)
        notes.extend(ttl_notes)
        totals_notes, totals[source] = _totals(records, source)
        notes.extend(totals_notes)

    if splits_path is not None:
        declared = split_tools.load_splits(splits_path)
        errors.extend(f"{splits_path.name}: {text}" for text in split_tools.check_disjoint(declared))
        known = {name for sessions in sessions_by_source.values() for name in sessions}
        unknown = sorted(set(declared["sessions"]) - known)
        if unknown:
            errors.append(f"{splits_path.name}: sessions absent from the records: {unknown}")
        missing = sorted(known - set(declared["sessions"]))
        if missing:
            errors.append(f"{splits_path.name}: sessions without a split: {missing}")

    return {"errors": errors, "notes": notes, "totals": totals,
            "checked": [str(path) for path in paths]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path, help="exported JSONL files")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--replays", type=Path,
                        help="replay directory, to bound duplicate rows by duplicate packets")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    budget = None
    if args.replays:
        names = sorted({record["replay"] for path in args.sources
                        for _, record in contracts.parse_records(path)[0]
                        if record.get("event") == "session"})
        budget = duplicate_packet_budget(args.replays, names)

    report = run(args.sources, args.splits, budget)
    for note in report["notes"]:
        print(f"note  {note}")
    for error in report["errors"]:
        print(f"ERROR {error}")
    print(f"\n{len(report['errors'])} errors, {len(report['notes'])} notes")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
        print(f"wrote {args.out}")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
