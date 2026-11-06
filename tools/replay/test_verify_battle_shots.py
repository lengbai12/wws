#!/usr/bin/env python3
"""Focused tests for independent replay shell/hit/damage evidence."""
from __future__ import annotations

import hashlib
import json
import tempfile
from unittest.mock import patch
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

# The verifier imports event classes lazily so it can be imported by tooling
# outside the parser venv. These test doubles preserve the parser's class-based
# event contract without parsing a replay or loading gamedata.


class GunFireEvent:
    def __init__(self, timestamp, entity_id, weapon_type=0):
        self.timestamp, self.entity_id, self.weapon_type = timestamp, entity_id, weapon_type


class SecondaryFireEvent:
    def __init__(self, timestamp, entity_id):
        self.timestamp, self.entity_id = timestamp, entity_id


class ShotCreatedEvent:
    def __init__(self, timestamp, owner_id):
        self.timestamp, self.owner_id = timestamp, owner_id


class DamageReceivedStatEvent:
    def __init__(self, entity_id, damage_param, stat_type, cumulative_count, cumulative_total):
        self.entity_id = entity_id
        self.damage_param = damage_param
        self.stat_type = stat_type
        self.cumulative_count = cumulative_count
        self.cumulative_total = cumulative_total


_fake_models = types.ModuleType("wows_replay_parser.events.models")
_fake_models.DamageReceivedStatEvent = DamageReceivedStatEvent
_fake_models.GunFireEvent = GunFireEvent
_fake_models.SecondaryFireEvent = SecondaryFireEvent
_fake_models.ShotCreatedEvent = ShotCreatedEvent

import verify_battle_shots as verifier  # noqa: E402


class VerifierTests(unittest.TestCase):
    def setUp(self):
        models = patch.dict(sys.modules, {"wows_replay_parser.events.models": _fake_models})
        models.start()
        self.addCleanup(models.stop)

    def replay(self, *, result=True):
        own = SimpleNamespace(relation=0, name="own", avatar_id=100, account_id=7)
        stats = {
            "shots_main_ap": 2, "shots_main_he": 1, "shots_main_cs": None,
            "hits_main_ap": 1, "hits_main_he": 2, "hits_main_cs": None,
            "damage_main_ap": 100, "damage_main_he": 250, "damage_main_cs": None,
            "shots_atba_ap": 4, "shots_atba_he": None, "shots_atba_cs": None,
            "hits_atba_ap": 1, "hits_atba_he": None, "hits_atba_cs": None,
            "damage_atba_ap": 12, "damage_atba_he": None, "damage_atba_cs": None,
        }
        events = [
            GunFireEvent(10.0, 101), GunFireEvent(11.0, 101),
            SecondaryFireEvent(12.0, 101),
            ShotCreatedEvent(10.0, 101), ShotCreatedEvent(10.0, 101),
            ShotCreatedEvent(11.0, 101),
            ShotCreatedEvent(12.0, 101),
            # Foreign shell must not enter the own-shell count.
            ShotCreatedEvent(11.0, 201),
            DamageReceivedStatEvent(101, "MAIN_HE", "ENEMY", 2, 250.0),
        ]
        results = SimpleNamespace(own_db_id=7, players={7: SimpleNamespace(stats=stats)}) if result else None
        tracker = SimpleNamespace(own_vehicle_id=101)
        return SimpleNamespace(
            players=[own, SimpleNamespace(relation=1, name="ally", avatar_id=200)],
            events=events, tracker=tracker, game_version="15,8,0,13187581", duration=20.0,
            battle_results=lambda: results,
        )

    def test_main_counts_are_owned_and_secondary_is_separate(self):
        report = verifier.verify_parsed_replay(self.replay(), gamedata_path="fixture-gamedata")
        self.assertEqual("verified", report["status"])
        self.assertEqual(2, report["main_battery"]["salvos_fired"])
        self.assertEqual(3, report["main_battery"]["shells_fired"])
        self.assertEqual(3, report["main_battery"]["hits"])
        self.assertEqual(350.0, report["main_battery"]["damage"])
        self.assertEqual(1, report["secondary_battery"]["salvos_fired"])
        self.assertEqual(1, report["secondary_battery"]["hits"])
        self.assertEqual("unknown", report["capture_correlation"]["status"])

    def test_resultless_replay_is_partial_with_missing_prerequisite(self):
        report = verifier.verify_parsed_replay(self.replay(result=False))
        self.assertEqual("partial", report["status"])
        self.assertIsNone(report["main_battery"]["hits"])
        self.assertIsNone(report["main_battery"]["damage"])
        self.assertTrue(any("post-battle results" in item for item in report["missing_prerequisites"]))

    def test_capture_clock_uses_explicit_anchor(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            anchor = Path(directory) / "clock.json"
            anchor.write_text(json.dumps({"replay_clock_s": 10.0, "capture_wall_s": 1000.0}), encoding="utf-8")
            report = verifier.verify_parsed_replay(self.replay(), capture_clock_path=str(anchor))
        self.assertEqual("correlated", report["capture_correlation"]["status"])
        self.assertEqual(1000.0, report["capture_correlation"]["first_shell_capture_wall_s"])
        self.assertEqual(1001.0, report["capture_correlation"]["last_shell_capture_wall_s"])

    def test_saved_evidence_does_not_claim_shell_or_hit_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.png"
            frame.write_bytes(b"image bytes only; no decoded HP")
            evidence = Path(directory) / "evidence.json"
            evidence.write_text(json.dumps({"shot": {"fired": True}, "hp": {"before": 39100, "after": 38760, "before_frame": str(frame), "after_frame": str(frame)}, "hashes": {str(frame): hashlib.sha256(frame.read_bytes()).hexdigest()}}))
            report = verifier.verify_saved_evidence(evidence)
        self.assertEqual("partial", report["status"])
        self.assertEqual(340.0, report["reported_target_hp"]["damage_delta"])
        self.assertFalse(report["reported_target_hp"]["independently_decoded"])
        self.assertTrue(all(frame["hash_matches"] for frame in report["reported_target_hp"]["frames"]))
        self.assertIsNone(report["main_battery"]["shells_fired"])
        self.assertIsNone(report["main_battery"]["hits"])

    def test_no_ownship_identity_never_uses_avatar_arithmetic(self):
        replay = self.replay()
        replay.tracker.own_vehicle_id = None
        report = verifier.verify_parsed_replay(replay)
        self.assertEqual("unknown", report["status"])
        self.assertTrue(any("own_vehicle_id" in item for item in report["missing_prerequisites"]))

    def test_ambiguous_projectile_keeps_total_unknown(self):
        replay = self.replay()
        replay.events.append(SecondaryFireEvent(10.0, 101))
        report = verifier.verify_parsed_replay(replay)
        self.assertEqual("partial", report["status"])
        self.assertIsNone(report["main_battery"]["shells_fired"])
        self.assertEqual(1, report["main_battery"]["confirmed_shells_fired"])
        self.assertEqual(2, len(report["event_crosscheck"]["unclassified_owner_shells"]))

    def test_result_owner_mismatch_is_not_counted(self):
        replay = self.replay()
        replay.players[0].account_id = 99
        report = verifier.verify_parsed_replay(replay)
        self.assertEqual("partial", report["status"])
        self.assertIsNone(report["main_battery"]["damage"])

    def test_result_shell_count_mismatch_is_partial(self):
        replay = self.replay()
        replay.battle_results().players[7].stats["shots_main_ap"] = 20
        report = verifier.verify_parsed_replay(replay)
        self.assertEqual("partial", report["status"])
        self.assertTrue(any("disagree" in item for item in report["missing_prerequisites"]))

    def test_capture_rejects_duplicate_or_regressing_clocks(self):
        for pair in ((10, 1001), (9, 1001), (11, 999), (11, 1000)):
            with self.subTest(pair=pair), tempfile.TemporaryDirectory() as directory:
                anchor = Path(directory) / "clock.json"
                anchor.write_text(json.dumps({"anchors": [{"replay_clock_s": 10, "capture_wall_s": 1000}, {"replay_clock_s": pair[0], "capture_wall_s": pair[1]}]}))
                report = verifier.verify_parsed_replay(self.replay(), capture_clock_path=str(anchor))
            self.assertEqual("unknown", report["capture_correlation"]["status"])

    def test_source_hash_binds_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.wowsreplay"
            source.write_bytes(b"isolated replay source fixture")
            report = verifier.verify_parsed_replay(self.replay(), source_path=source)
        self.assertEqual(hashlib.sha256(b"isolated replay source fixture").hexdigest(), report["source"]["sha256"])

    def test_missing_saved_evidence_is_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            report = verifier.verify_saved_evidence(Path(directory) / "missing.json")
        self.assertEqual("unknown", report["status"])

    def test_run_window_uses_actual_shell_clock_and_leaves_damage_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            anchor = Path(directory) / "clock.json"
            anchor.write_text(json.dumps({"replay_clock_s": 10.0, "capture_wall_s": 1000.0}))
            report = verifier.verify_parsed_replay(self.replay(), capture_clock_path=str(anchor))
        run = verifier.correlate_run(report, 999.0, 1006.0, 999.5)
        self.assertEqual(3, run["confirmed_main_shells_fired"])
        self.assertEqual(0.5, run["target_acceptance"]["first_actual_main_shell_delay_s"])
        self.assertTrue(run["target_acceptance"]["actual_fire_within_5s"])
        self.assertIsNone(run["main_hits"])
        self.assertIsNone(run["main_damage"])

    def test_arena_record_binds_distinct_header_and_database_ids(self):
        replay = self.replay()
        replay.players[0].account_id = 537696811
        arena_row = {"id": 537696811, "accountDBID": 7, "shipId": 101}
        roster = types.ModuleType("wows_replay_parser.roster")
        roster._find_arena_state_packet = lambda packets: object()
        roster._extract_arena_blobs = lambda packet: [b"", b"fixture"]
        roster._load_key_maps = lambda path: ({}, {})
        roster._decode_arena_players = lambda blob, keys: [arena_row]
        with patch.dict(sys.modules, {"wows_replay_parser.roster": roster}):
            report = verifier.verify_parsed_replay(replay)
        self.assertEqual("verified", report["status"])
        self.assertEqual(350.0, report["main_battery"]["damage"])
        self.assertEqual(7, report["ownership"]["battle_results_identity"]["arena_account_db_id"])
        for field, value in (("accountDBID", 99), ("shipId", 201), ("id", 99)):
            with self.subTest(field=field):
                old = arena_row[field]
                arena_row[field] = value
                with patch.dict(sys.modules, {"wows_replay_parser.roster": roster}):
                    rejected = verifier.verify_parsed_replay(replay)
                arena_row[field] = old
                self.assertEqual("partial", rejected["status"])
                self.assertIsNone(rejected["main_battery"]["damage"])

    def test_non_main_weapon_callback_does_not_prove_main_shells(self):
        replay = self.replay()
        replay.events[0].weapon_type = 2
        report = verifier.verify_parsed_replay(replay)
        self.assertEqual("partial", report["status"])
        self.assertIsNone(report["main_battery"]["shells_fired"])
        self.assertEqual(1, report["main_battery"]["confirmed_shells_fired"])

    def test_run_without_capture_clock_is_unknown(self):
        run = verifier.correlate_run(verifier.verify_parsed_replay(self.replay()), 1000.0, 1010.0)
        self.assertEqual("unknown", run["status"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
