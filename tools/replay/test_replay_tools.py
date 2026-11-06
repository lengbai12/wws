#!/usr/bin/env python3
"""Tests for the layer-1 replay tooling.

Run: /tmp/wowsenv/bin/python tools/replay/test_replay_tools.py

The tests cover the positive path and, more importantly, the negative path: a
validator that cannot reject malformed input is not a validator.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import bounds
import check
import contracts
import splits as split_tools
import staleness

PARSED = Path(__file__).resolve().parents[2] / "replays/parsed"


def consumable(clock: float, vehicle: int = 1, kind_id: int = 0) -> dict:
    return {"event": "consumable", "clock": clock, "vehicle": vehicle,
            "consumable_type": kind_id, "consumable_name": "crashCrew",
            "work_time_left": 5.0, "raw": "0100", "owner_name": "p"}


def session_row(replay: str = "r", build: str = "15.8.0", span: float = 100.0) -> dict:
    return {"event": "session", "replay": replay, "build": build, "map": "map",
            "match_group": "pvp", "game_mode": 7, "scenario": "domination_3point",
            "recording_player": "p", "recording_vehicle": "ship", "players": 24,
            "vehicles": 24, "packets": 100, "decoded_bytes": 1000,
            "clock_span_s": span, "observed_vehicle_entities": 24,
            "position_only_vehicle_entities": 23, "roster_link": "24/24"}  # noqa: RUF001


class Contracts(unittest.TestCase):
    def test_declarations_are_well_formed(self) -> None:
        self.assertEqual([], contracts.self_check())

    def test_accepts_real_records(self) -> None:
        records, problems = contracts.parse_records(PARSED / "events.jsonl")
        self.assertEqual([], problems)
        self.assertEqual([], contracts.validate_records(records))
        self.assertGreater(len(records), 49_000)

    def test_rejects_unknown_field(self) -> None:
        record = consumable(1.0)
        record["surprise"] = 1
        self.assertIn("unknown field 'surprise'", contracts.validate_record(record))

    def test_rejects_missing_field(self) -> None:
        record = consumable(1.0)
        del record["raw"]
        self.assertIn("missing field 'raw'", contracts.validate_record(record))

    def test_rejects_wrong_type(self) -> None:
        record = consumable(1.0)
        record["clock"] = "1.0"
        self.assertTrue(any("expected" in p and "clock" in p
                            for p in contracts.validate_record(record)))

    def test_rejects_bool_as_number(self) -> None:
        record = consumable(1.0)
        record["vehicle"] = True
        self.assertTrue(any("vehicle" in p for p in contracts.validate_record(record)))

    def test_rejects_unknown_kind(self) -> None:
        self.assertIn("unknown event kind 'torpedo'",
                      contracts.validate_record({"event": "torpedo"}))

    def test_rejects_truncated_final_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.jsonl"
            path.write_text(json.dumps(consumable(1.0)) + "\n" + '{"event": "consumable", "cloc',
                            encoding="utf-8")
            records, problems = contracts.parse_records(path)
            self.assertEqual(1, len(records))
            self.assertEqual(1, len(problems))
            self.assertIn("truncated final line", problems[0])

    def test_rejects_blank_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.jsonl"
            path.write_text(json.dumps(consumable(1.0)) + "\n\n", encoding="utf-8")
            _, problems = contracts.parse_records(path)
            self.assertEqual(1, len(problems))


class Staleness(unittest.TestCase):
    def test_stamp_expires(self) -> None:
        stamp = contracts.ObservationStamp(seq=1, epoch=0, recv_clock=10.0, age_bound_s=0.0,
                                          timeout_s=0.1)
        self.assertTrue(stamp.fresh(10.0))
        self.assertTrue(stamp.fresh(10.05))
        self.assertFalse(stamp.fresh(10.1))
        self.assertFalse(stamp.fresh(9.9))

    def test_stamp_without_age_bound_is_never_fresh(self) -> None:
        stamp = contracts.ObservationStamp(seq=1, epoch=0, recv_clock=10.0,
                                          age_bound_s=0.2, timeout_s=0.1)
        self.assertFalse(stamp.has_age_bound())
        self.assertFalse(stamp.fresh(10.0))

    def test_safe_ttl_does_not_overlap(self) -> None:
        events = [consumable(t) for t in (0.0, 10.0, 20.0)]
        report = staleness.recommend_ttl(events, "consumable")
        self.assertEqual(10.0, report["min_gap_s"])
        self.assertEqual(5.0, report["safe_ttl_s"])
        self.assertEqual([], staleness.coverage_violations(
            events, {"consumable": report["safe_ttl_s"]}))

    def test_typical_ttl_is_guidance_only(self) -> None:
        # One short spacing among many long ones: the low quantile stops being
        # the minimum, and a ceiling taken from it overlaps the short spacing.
        times = [0.0, 0.2] + [10.2 + 10.0 * i for i in range(19)]
        events = [consumable(t) for t in times]
        report = staleness.recommend_ttl(events, "consumable")
        self.assertAlmostEqual(0.2, report["min_gap_s"], places=9)
        self.assertAlmostEqual(10.0, report["quantile_gap_s"], places=9)
        self.assertAlmostEqual(0.1, report["safe_ttl_s"], places=9)
        self.assertEqual([], staleness.coverage_violations(
            events, {"consumable": report["safe_ttl_s"]}))
        self.assertTrue(staleness.coverage_violations(
            events, {"consumable": report["quantile_gap_s"]}))

    def test_oversized_ttl_is_detected(self) -> None:
        events = [consumable(t) for t in (0.0, 10.0, 20.0)]
        violations = staleness.coverage_violations(events, {"consumable": 15.0})
        self.assertEqual(2, len(violations))
        self.assertEqual(0.0, violations[0]["earlier_clock"])

    def test_gaps_are_per_subject(self) -> None:
        # Vehicle 2 appears once, so it contributes no gap.
        events = [consumable(0.0, vehicle=1), consumable(5.0, vehicle=2),
                  consumable(3.0, vehicle=1)]
        self.assertEqual([3.0], staleness.gaps(events, "consumable"))

    def test_subject_is_composite(self) -> None:
        # Two different consumables on one ship at one instant are legitimate.
        events = [consumable(4.0, vehicle=1, kind_id=0), consumable(4.0, vehicle=1, kind_id=8)]
        report = staleness.recommend_ttl(events, "consumable")
        self.assertEqual(0, report["simultaneous"])
        self.assertEqual([], staleness.gaps(events, "consumable"))
        # The same consumable twice at one instant is a pair for review.
        repeated = [consumable(4.0, vehicle=1), consumable(4.0, vehicle=1)]
        self.assertEqual(1, staleness.recommend_ttl(repeated, "consumable")["simultaneous"])
        self.assertEqual(1, len(staleness.coverage_violations(repeated, {"consumable": 1.0})))

    def test_unknown_kind_is_refused(self) -> None:
        with self.assertRaises(KeyError):
            staleness.coverage_violations([], {"torpedo": 1.0})


class Bounds(unittest.TestCase):
    def test_matches_closed_form_at_zero_errors(self) -> None:
        # For k = 0 the Clopper-Pearson upper bound is 1 - alpha**(1/n).
        for n in (5, 10, 40):
            self.assertAlmostEqual(1.0 - 0.025 ** (1.0 / n),
                                   bounds.clopper_pearson(0, n)[1], places=9)

    def test_matches_closed_form_at_no_errors_one_sided(self) -> None:
        self.assertAlmostEqual(1.0 - 0.05 ** 0.1, bounds.upper_bound(0, 10), places=9)

    def test_matches_closed_form_at_all_successes(self) -> None:
        for n in (5, 10, 40):
            self.assertAlmostEqual(0.025 ** (1.0 / n),
                                   bounds.clopper_pearson(n, n)[0], places=9)

    def test_beta_quantile_inverts_the_cdf(self) -> None:
        for a, b, p in ((1.0, 9.0, 0.975), (3.0, 7.0, 0.5), (20.0, 20.0, 0.9)):
            x = bounds.beta_quantile(p, a, b)
            self.assertAlmostEqual(p, bounds.regularized_incomplete_beta(a, b, x), places=9)

    def test_known_table_value(self) -> None:
        # 95% interval for 3 successes out of 20, a standard Clopper-Pearson case.
        lower, upper = bounds.clopper_pearson(3, 20)
        self.assertAlmostEqual(0.0321, lower, places=3)
        self.assertAlmostEqual(0.3789, upper, places=3)

    def test_bounds_are_ordered_and_finite(self) -> None:
        for k in range(0, 11):
            lower, upper = bounds.clopper_pearson(k, 10)
            self.assertTrue(0.0 <= lower <= upper <= 1.0)
            self.assertTrue(math.isfinite(lower) and math.isfinite(upper))

    def test_upper_bound_is_monotone_in_errors(self) -> None:
        values = [bounds.upper_bound(k, 20) for k in range(21)]
        self.assertEqual(values, sorted(values))

    def test_bad_input_is_refused(self) -> None:
        for args in ((0, 0), (11, 10), (-1, 10)):
            with self.assertRaises(ValueError):
                bounds.clopper_pearson(*args)
        with self.assertRaises(ValueError):
            bounds.clopper_pearson(1, 10, 1.0)
        with self.assertRaises(ValueError):
            bounds.clopper_pearson(1, 10, 0.95, sides=3)

    def test_rejection_takes_the_largest_bounded_prefix(self) -> None:
        # Clean head, errored tail: the largest prefix that keeps the bound wins.
        rows = [{"score": 1.0 - i / 100, "error": i >= 25} for i in range(30)]
        threshold = 0.15
        report = bounds.calibrate_rejection(rows, max_error_fraction=threshold)
        expected = max(
            n for n in range(1, 31)
            if bounds.upper_bound(sum(1 for row in rows[:n] if row["error"]), n) <= threshold)
        self.assertEqual(expected, report["accepted"])
        self.assertLess(report["accepted"], 30)
        self.assertLessEqual(report["bound"], threshold)
        self.assertEqual(report["accepted"] + 1, report["rejected_at"])
        self.assertEqual(rows[report["accepted"]]["score"], report["rejected_score"])

    def test_rejection_accepts_exactly_what_the_bound_allows(self) -> None:
        # With no observed errors the one-sided bound is 1 - alpha**(1/n), so the
        # accepted prefix is the largest n that bounds the error rate.
        threshold = 0.2
        rows = [{"score": 1.0 - i / 100, "error": False} for i in range(20)]
        report = bounds.calibrate_rejection(rows, max_error_fraction=threshold)
        expected = max(n for n in range(1, 21) if bounds.upper_bound(0, n) <= threshold)
        self.assertEqual(expected, report["accepted"])
        self.assertLess(report["bound"], threshold)
        self.assertIsNone(report["rejected_at"])

    def test_rejection_refuses_every_prefix_that_cannot_be_bounded(self) -> None:
        # One error in two rows cannot be bounded below 10% at any prefix.
        rows = [{"score": 1.0, "error": True}, {"score": 0.5, "error": False}]
        report = bounds.calibrate_rejection(rows, max_error_fraction=0.1)
        self.assertEqual(0, report["accepted"])
        self.assertEqual(1, report["rejected_at"])
        self.assertEqual(1.0, report["rejected_score"])

    def test_rejection_needs_enough_rows_before_it_accepts(self) -> None:
        # An exact bound is wide at small counts, so a short clean list is refused.
        few = [{"score": 1.0 - i / 10, "error": False} for i in range(5)]
        self.assertEqual(0, bounds.calibrate_rejection(few, 0.1)["accepted"])
        many = [{"score": 1.0 - i / 40, "error": False} for i in range(40)]
        self.assertEqual(40, bounds.calibrate_rejection(many, 0.1)["accepted"])


class Splits(unittest.TestCase):
    def sessions(self) -> list[dict]:
        return [{"replay": f"r{i}", "match_group": ("pvp", "pve", "cooperative")[i % 3]}
                for i in range(8)]

    def test_assignment_is_complete_and_deterministic(self) -> None:
        first = split_tools.make_splits(self.sessions())
        second = split_tools.make_splits(self.sessions())
        self.assertEqual(first, second)
        self.assertEqual(8, len(first["sessions"]))
        self.assertEqual([], split_tools.check_disjoint(first))

    def test_assignment_does_not_depend_on_input_order(self) -> None:
        forward = split_tools.make_splits(self.sessions())
        backward = split_tools.make_splits(list(reversed(self.sessions())))
        self.assertEqual(forward["sessions"], backward["sessions"])

    def test_every_group_keeps_a_fit_session(self) -> None:
        splits = split_tools.make_splits(self.sessions())
        for group in set(s["match_group"] for s in self.sessions()):
            members = [s["replay"] for s in self.sessions() if s["match_group"] == group]
            self.assertTrue(any(splits["sessions"][m] != "holdout" for m in members))

    def test_holdout_is_refused_for_fitting(self) -> None:
        splits = split_tools.make_splits(self.sessions())
        held = [r for r, v in splits["sessions"].items() if v == "holdout"]
        self.assertTrue(held)
        with self.assertRaises(AssertionError):
            split_tools.assert_fittable(splits, held[0])
        fit = [r for r, v in splits["sessions"].items() if v == "fit"][0]
        self.assertIsNone(split_tools.assert_fittable(splits, fit))

    def test_unknown_session_is_refused(self) -> None:
        splits = split_tools.make_splits(self.sessions())
        with self.assertRaises(KeyError):
            split_tools.assert_fittable(splits, "r99")

    def test_impossible_quota_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            split_tools.make_splits(self.sessions(), holdout=7, validation=1)
        with self.assertRaises(ValueError):
            split_tools.make_splits([{"replay": "a"}, {"replay": "a"}])

    def test_declared_count_mismatch_is_detected(self) -> None:
        splits = split_tools.make_splits(self.sessions())
        splits["counts"]["holdout"] += 1
        self.assertTrue(any("disagree" in p for p in split_tools.check_disjoint(splits)))

    def test_real_sessions_split(self) -> None:
        sessions = split_tools.sessions_from(PARSED / "events.jsonl")
        splits = split_tools.make_splits(sessions)
        # The invariant, not the corpus size: this used to assert a frozen 8, so every
        # session added upstream turned a correct split into a red test. What must hold
        # is that the split covers every session in the file, stays disjoint, and that
        # the file still has enough sessions for a session-level split at all.
        self.assertEqual(len(sessions), len(splits["sessions"]))
        self.assertGreaterEqual(len(sessions), 5)
        self.assertEqual([], split_tools.check_disjoint(splits))


class Checker(unittest.TestCase):
    def test_real_files_pass(self) -> None:
        report = check.run([PARSED / "events.jsonl", PARSED / "consumables.jsonl"], None)
        self.assertEqual([], report["errors"])

    def test_real_counts_are_reported_raw_and_folded(self) -> None:
        report = check.run([PARSED / "events.jsonl"], None)
        totals = report["totals"]["events.jsonl"]
        # The invariants, not the corpus size: this used to assert a frozen 886, so each
        # recording session turned a correct report into a red test. What must hold is
        # the SHAPE of the report -- every family is counted in both views, and folding
        # only ever removes duplicates from the families that repeat packets.
        for kind in ("session", "roster", "death", "damage", "health", "consumable"):
            self.assertGreater(totals["raw"][kind], 0)
            self.assertLessEqual(totals["folded"][kind], totals["raw"][kind])
        # Positions and health repeat per packet, so they fold down.
        self.assertGreater(totals["raw"]["damage"], totals["folded"]["damage"])
        self.assertGreater(totals["raw"]["health"], totals["folded"]["health"])
        # Events fold to one row each, so raw and folded agree.
        for kind in ("session", "roster", "death", "consumable"):
            self.assertEqual(totals["raw"][kind], totals["folded"][kind])

    def test_detects_clock_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regress.jsonl"
            rows = [session_row(span=100.0), consumable(5.0), consumable(1.0)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = check.run([path], None)
            self.assertTrue(any("precedes" in e for e in report["errors"]))

    def test_duplicate_row_is_a_note_not_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dup.jsonl"
            rows = [session_row(span=100.0), consumable(5.0), consumable(5.0)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = check.run([path], None)
            self.assertEqual([], report["errors"])
            self.assertTrue(any("1 duplicate rows" in n for n in report["notes"]))
            self.assertEqual(2, report["totals"]["dup.jsonl"]["raw"]["consumable"])
            self.assertEqual(1, report["totals"]["dup.jsonl"]["folded"]["consumable"])

    def test_duplicate_rows_above_the_packet_budget_are_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dup.jsonl"
            rows = [session_row(span=100.0), consumable(5.0), consumable(5.0)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = check.run([path], None, packet_budget={"r": 0})
            self.assertTrue(any("exceed the" in e for e in report["errors"]))

    def test_surfaces_a_contract_error_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extra.jsonl"
            bad = consumable(5.0)
            bad["extra"] = 1
            path.write_text("\n".join(json.dumps(r) for r in [session_row(), bad]) + "\n",
                            encoding="utf-8")
            report = check.run([path], None)
            self.assertTrue(any("unknown field 'extra'" in e for e in report["errors"]))

    def test_detects_row_before_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orphan.jsonl"
            path.write_text(json.dumps(consumable(1.0)) + "\n", encoding="utf-8")
            report = check.run([path], None)
            self.assertTrue(any("before any session" in e for e in report["errors"]))

    def test_detects_clock_beyond_declared_span(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "span.jsonl"
            rows = [session_row(span=1.0), consumable(50.0)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = check.run([path], None)
            self.assertTrue(any("exceeds declared span" in e for e in report["errors"]))

    def test_detects_mixed_builds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "builds.jsonl"
            rows = [session_row("a", "15.8.0", 10.0), session_row("b", "15.9.0", 10.0)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            report = check.run([path], None)
            self.assertTrue(any("mixed client builds" in e for e in report["errors"]))

    def test_detects_split_referring_to_absent_session(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            splits_path = Path(directory) / "splits.json"
            splits_path.write_text(json.dumps({
                "sessions": {"ghost": "holdout", "other": "fit", "third": "validation"},
                "counts": {"fit": 1, "validation": 1, "holdout": 1},
            }), encoding="utf-8")
            report = check.run([PARSED / "consumables.jsonl"], splits_path)
            self.assertTrue(any("absent from the records" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
