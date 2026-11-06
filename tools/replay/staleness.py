#!/usr/bin/env python3
"""Derive per-kind label lifetimes from observed event spacing.

A label stays valid for a bounded time. If the bound is too large, an older
observation is still inside its validity window when a newer one for the same
subject arrives, and two live observations describe the same moment.

A subject is a composite key, because two different things can legitimately
happen to one ship at the same instant: a salvo from two attackers, or two
different consumables. The hard ceiling for a lifetime is therefore the MINIMUM
observed spacing, not a quantile of it. The quantile is reported as guidance
only.

The method is game-independent: it uses event times and the declared subject
fields.
"""
from __future__ import annotations

from collections import defaultdict

from contracts import label_window

# Subject fields per kind. A tuple, because one entity can carry several
# independent streams of the same kind.
SUBJECT_FIELDS = {
    "death": ("victim",),
    "damage": ("victim", "attacker"),
    "consumable": ("vehicle", "consumable_type"),
    "health": ("vehicle",),
    "alive": ("vehicle",),
}


def subject(event: dict, kind: str) -> tuple | None:
    fields = SUBJECT_FIELDS.get(kind)
    if fields is None:
        raise KeyError(f"no subject fields declared for kind {kind!r}")
    if any(field not in event for field in fields):
        return None
    return tuple(event[field] for field in fields)


def _by_subject(events: list[dict], kind: str) -> dict[tuple, list[float]]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for event in events:
        if event.get("event") != kind:
            continue
        key = subject(event, kind)
        if key is None:
            continue
        grouped[key].append(float(event["clock"]))
    for times in grouped.values():
        times.sort()
    return grouped


def gaps(events: list[dict], kind: str) -> list[float]:
    """Spacings between consecutive same-subject events of one kind, in seconds."""
    return [b - a for times in _by_subject(events, kind).values()
            for a, b in zip(times, times[1:])]


def simultaneous_pairs(events: list[dict], kind: str) -> list[dict]:
    """Same-subject pairs at an identical clock.

    A pair at the same clock with the same subject is either a genuine
    simultaneous event that the subject key cannot separate, or a row that
    should be folded. It forces a zero lifetime, so it is reported for review.
    """
    found = []
    for key, times in _by_subject(events, kind).items():
        for a, b in zip(times, times[1:]):
            if b == a:
                found.append({"kind": kind, "subject": key, "clock": a})
    return found


def quantile(sorted_values: list[float], fraction: float) -> float | None:
    """Nearest-rank quantile of an already sorted non-empty list."""
    if not sorted_values:
        return None
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    index = min(len(sorted_values) - 1, max(0, round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def recommend_ttl(events: list[dict], kind: str, fraction: float = 0.05) -> dict:
    """Lifetime evidence for one kind.

    ``safe_ttl_s`` is derived from the minimum spacing and cannot overlap a later
    same-subject event. ``typical_ttl_s`` is derived from the low quantile and is
    guidance only: it can overlap whenever a spacing below the quantile occurs.
    """
    values = sorted(gaps(events, kind))
    minimum = values[0] if values else None
    low = quantile(values, fraction)
    return {
        "kind": kind,
        "subject_fields": SUBJECT_FIELDS[kind],
        "gaps": len(values),
        "min_gap_s": minimum,
        "safe_ttl_s": minimum / 2 if minimum is not None else None,
        "quantile": fraction,
        "quantile_gap_s": low,
        "typical_ttl_s": low / 2 if low is not None else None,
        "max_gap_s": values[-1] if values else None,
        "simultaneous": len(simultaneous_pairs(events, kind)),
    }


def coverage_violations(events: list[dict], ttl_by_kind: dict[str, float]) -> list[dict]:
    """Events that a later same-subject event overlaps.

    A violation means the lifetime is too large: when the later event occurs, the
    earlier label would still be inside its validity window.
    """
    violations = []
    for kind, ttl in ttl_by_kind.items():
        if kind not in SUBJECT_FIELDS:
            raise KeyError(f"no subject fields declared for kind {kind!r}")
        if ttl < 0:
            raise ValueError(f"lifetime for {kind!r} must not be negative")
        grouped = _by_subject(events, kind)
        for key, times in grouped.items():
            for earlier, later in zip(times, times[1:]):
                _, end = label_window(earlier, ttl)
                if later < end:
                    violations.append({
                        "kind": kind, "subject": key, "ttl_s": ttl,
                        "earlier_clock": earlier, "later_clock": later,
                        "valid_until": end, "overlap_s": end - later,
                    })
    return violations
