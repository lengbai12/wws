#!/usr/bin/env python3
"""Tests for the nine fire gates and the caption-first range reader."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import fire_gate as G  # noqa: E402
import aim_servo as A  # noqa: E402


class AllClearTests(unittest.TestCase):
    def test_every_gate_passing_allows_the_shot(self):
        decision = G.fire_decision(G.clear_evidence())
        self.assertTrue(decision['allow'], decision['blockers'])
        self.assertEqual(decision['blockers'], [])
        self.assertEqual(set(decision['status']), set(G.GATES))
        self.assertEqual(decision['status']['scope'], 'not_requested')

    def test_body_visible_false_never_blocks(self):
        decision = G.fire_decision(G.clear_evidence(body_visible=False))
        self.assertTrue(decision['allow'], decision['blockers'])
        self.assertEqual(decision['blockers'], [])
        codes = [warning['code'] for warning in decision['warnings']]
        self.assertIn(G.BODY_NOT_VISIBLE, codes)

    def test_missing_scope_evidence_without_a_scope_request_does_not_block(self):
        decision = G.fire_decision(G.clear_evidence(scope_requested=False,
                                                    scope_evidenced=None,
                                                    scope_target_visible=None))
        self.assertTrue(decision['allow'], decision['blockers'])
        self.assertEqual(decision['status']['scope'], 'not_requested')


class HardGateTests(unittest.TestCase):
    """One test per hard condition: breaking only that condition blocks it."""

    def assert_blocked_by(self, blocker, **overrides):
        decision = G.fire_decision(G.clear_evidence(**overrides))
        self.assertFalse(decision['allow'])
        self.assertEqual(decision['blockers'], [blocker], decision['gates'])

    def test_battle_not_active(self):
        self.assert_blocked_by(G.BATTLE_NOT_ACTIVE, battle_active=False)
        self.assert_blocked_by(G.BATTLE_NOT_ACTIVE, battle_active=None)

    def test_own_ship_not_alive(self):
        self.assert_blocked_by(G.OWN_SHIP_NOT_ALIVE, own_alive=False)
        self.assert_blocked_by(G.OWN_SHIP_NOT_ALIVE, own_alive=None)

    def test_weapon_not_a_gun(self):
        self.assert_blocked_by(G.WEAPON_NOT_GUN, weapon_kind='torpedo')

    def test_reload_not_ready(self):
        self.assert_blocked_by(G.RELOAD_NOT_READY, reload_ready=False)
        self.assert_blocked_by(G.RELOAD_NOT_READY, reload_ready=None)

    def test_enemy_plate_missing(self):
        self.assert_blocked_by(G.ENEMY_PLATE_MISSING, enemy_plate=None)

    def test_enemy_plate_not_from_this_frame(self):
        self.assert_blocked_by(G.ENEMY_PLATE_NOT_CURRENT, frame_fresh=False)

    def test_enemy_plate_depleted(self):
        self.assert_blocked_by(G.ENEMY_PLATE_DEPLETED,
                               enemy_plate={"bbox": [10, 20, 120, 17], "hp": 0,
                                            "red_px": 900})

    def test_enemy_plate_hp_unreadable_is_a_warning(self):
        # Operator rule (2026-10-23): "a roughly right solution may fire". The
        # enemy's exact HP only exists to avoid shooting a wreck, so an
        # unreadable HP must not veto a shot -- it is reported as a warning. The
        # wreck case below still blocks.
        decision = G.fire_decision(G.clear_evidence(
            enemy_plate={"bbox": [10, 20, 120, 17], "hp": None, "red_px": 900}))
        self.assertTrue(decision["allow"], decision)
        codes = [w.get("code") for w in decision["warnings"]]
        self.assertIn(G.ENEMY_PLATE_HP_UNKNOWN, codes)

    def test_enemy_plate_without_red_pixels(self):
        self.assert_blocked_by(G.ENEMY_PLATE_NO_RED_PIXELS,
                               enemy_plate={"bbox": [10, 20, 120, 17], "hp": 20000,
                                            "red_px": 0})

    def test_aim_error_outside_tolerance(self):
        self.assert_blocked_by(G.AIM_ERROR_OUT_OF_TOLERANCE, aim_error_px=120.0,
                               fire_tolerance_px=60.0)

    def test_aim_error_unknown(self):
        self.assert_blocked_by(G.AIM_ERROR_UNKNOWN, aim_error_px=None)

    def test_fire_tolerance_missing(self):
        self.assert_blocked_by(G.FIRE_TOLERANCE_MISSING, fire_tolerance_px=None)

    def test_identity_not_verified(self):
        self.assert_blocked_by(G.IDENTITY_NOT_VERIFIED, identity_ok=False)

    def test_focus_not_confirmed(self):
        self.assert_blocked_by(G.FOCUS_NOT_CONFIRMED, focus_ok=False)

    def test_scope_not_evidenced(self):
        self.assert_blocked_by(G.SCOPE_NOT_EVIDENCED, scope_requested=True,
                               scope_evidenced=False, scope_target_visible=True)

    def test_scope_target_not_visible(self):
        self.assert_blocked_by(G.SCOPE_TARGET_NOT_VISIBLE, scope_requested=True,
                               scope_evidenced=True, scope_target_visible=False)

    def test_scope_requested_and_fully_evidenced_passes(self):
        decision = G.fire_decision(G.clear_evidence(scope_requested=True,
                                                   scope_evidenced=True,
                                                   scope_target_visible=True))
        self.assertTrue(decision['allow'], decision['blockers'])
        self.assertEqual(decision['status']['scope'], 'pass')


class WarningTests(unittest.TestCase):
    def test_range_gap_beyond_tolerance_is_only_a_warning(self):
        decision = G.fire_decision(G.clear_evidence(range_gap_km=1.2,
                                                   range_tolerance_km=0.5))
        self.assertTrue(decision['allow'], decision['blockers'])
        self.assertEqual(decision['blockers'], [])
        self.assertEqual(len(decision['warnings']), 1)
        warning = decision['warnings'][0]
        self.assertEqual(warning['code'], G.RANGE_GAP_OUTSIDE_TOLERANCE)
        self.assertEqual(warning['expected_short_or_long_km'], 1.2)
        self.assertEqual(warning['miss_direction'], 'short')
        self.assertEqual(warning['range_gap_km'], 1.2)

    def test_range_gap_inside_tolerance_warns_nothing(self):
        decision = G.fire_decision(G.clear_evidence(range_gap_km=0.3,
                                                   range_tolerance_km=0.5))
        codes = [warning['code'] for warning in decision['warnings']]
        self.assertNotIn(G.RANGE_GAP_OUTSIDE_TOLERANCE, codes)

    def test_a_long_gap_is_tagged_long(self):
        decision = G.fire_decision(G.clear_evidence(range_gap_km=-0.9,
                                                   range_tolerance_km=0.5))
        self.assertEqual(decision['warnings'][0]['miss_direction'], 'long')

    def test_unknown_range_value_warns_nothing(self):
        decision = G.fire_decision(G.clear_evidence(range_gap_km=None))
        self.assertTrue(decision['allow'])
        self.assertEqual(decision['warnings'], [])


class RangeReaderTests(unittest.TestCase):
    def frame(self):
        return np.zeros((200, 300, 3), np.uint8)

    def test_a_readable_caption_never_calls_the_minimap(self):
        calls = {'caption': 0, 'minimap': 0}

        def caption(frame, bbox):
            calls['caption'] += 1
            return 7.7

        def minimap(frame):
            calls['minimap'] += 1
            return 99.0

        reader = G.RangeReader(caption, minimap)
        reading = reader.read(self.frame(), [100, 100, 120, 17])
        self.assertEqual(reading['target_km'], 7.7)
        self.assertEqual(reading['source'], 'nameplate_caption')
        self.assertEqual(calls, {'caption': 1, 'minimap': 0})

    def test_the_minimap_is_the_fallback_when_the_caption_is_unreadable(self):
        calls = {'caption': 0, 'minimap': 0}

        def caption(frame, bbox):
            calls['caption'] += 1
            return None

        def minimap(frame):
            calls['minimap'] += 1
            return 13.1

        reader = G.RangeReader(caption, minimap)
        reading = reader.read(self.frame(), [100, 100, 120, 17])
        self.assertEqual(reading['target_km'], 13.1)
        self.assertEqual(reading['source'], 'minimap')
        self.assertEqual(calls, {'caption': 1, 'minimap': 1})

    def test_one_frame_is_read_once(self):
        calls = {'caption': 0, 'minimap': 0}

        def caption(frame, bbox):
            calls['caption'] += 1
            return 8.2

        reader = G.RangeReader(caption, lambda frame: None)
        frame = self.frame()
        first = reader.read(frame, [100, 100, 120, 17])
        second = reader.read(frame, [100, 100, 120, 17])
        self.assertFalse(first['cached'])
        self.assertTrue(second['cached'])
        self.assertEqual(calls['caption'], 1)
        self.assertEqual(second['target_km'], 8.2)

    def test_a_new_frame_is_read_again(self):
        calls = {'caption': 0}

        def caption(frame, bbox):
            calls['caption'] += 1
            return 8.2

        reader = G.RangeReader(caption, lambda frame: None)
        reader.read(self.frame(), [100, 100, 120, 17])
        reader.read(self.frame(), [100, 100, 120, 17])
        self.assertEqual(calls['caption'], 2)

    def test_no_plate_never_reads_a_caption(self):
        calls = {'caption': 0, 'minimap': 0}

        def caption(frame, bbox):
            calls['caption'] += 1
            return 7.0

        def minimap(frame):
            calls['minimap'] += 1
            return 12.0

        reading = G.RangeReader(caption, minimap).read(self.frame(), None)
        self.assertEqual(calls, {'caption': 0, 'minimap': 1})
        self.assertEqual(reading['source'], 'minimap')


class ServoWiringTests(unittest.TestCase):
    """The servo shares these gates instead of re-testing them."""

    def test_scope_blocks_fire_answers_from_the_gate(self):
        self.assertFalse(A.scope_blocks_fire(False, False))
        self.assertFalse(A.scope_blocks_fire(False, True))
        self.assertTrue(A.scope_blocks_fire(True, False))
        self.assertTrue(A.scope_blocks_fire(True, None))
        self.assertFalse(A.scope_blocks_fire(True, True))

    def test_step_range_evidence_prefers_the_caption(self):
        frame = np.zeros((1600, 2560, 3), np.uint8)
        with patch.object(A, 'hud_distance_km', return_value=9.6) as caption, \
                patch.object(A, 'minimap_target_km', return_value=9.9) as minimap, \
                patch.object(A, 'hud_aim_km', return_value=9.4):
            row = A.step_range_evidence(frame, [100, 100, 120, 17], reticle=[1280, 805])
        self.assertEqual(caption.call_count, 1)
        self.assertEqual(minimap.call_count, 0)
        self.assertEqual(row['target_km'], 9.6)
        self.assertEqual(row['range_source'], 'nameplate_caption')
        self.assertAlmostEqual(row['range_gap_km'], 0.2, places=2)

    def test_step_range_evidence_falls_back_to_the_minimap(self):
        frame = np.zeros((1600, 2560, 3), np.uint8)
        with patch.object(A, 'hud_distance_km', return_value=None) as caption, \
                patch.object(A, 'minimap_target_km', return_value=9.9) as minimap, \
                patch.object(A, 'hud_aim_km', return_value=9.4):
            row = A.step_range_evidence(frame, [100, 100, 120, 17], reticle=[1280, 805])
        self.assertEqual(caption.call_count, 1)
        self.assertEqual(minimap.call_count, 1)
        self.assertEqual(row['target_km'], 9.9)
        self.assertEqual(row['range_source'], 'minimap')


class TerrainGateTests(unittest.TestCase):
    """Land on the line of fire is the only occlusion the minimap can prove."""

    def evidence(self, **over):
        base = dict(battle_active=True, own_alive=True, weapon_kind="gun",
                    reload_ready=True, enemy_plate={"bbox": [10, 20, 120, 17],
                                                    "hp": 20000, "red_px": 900},
                    frame_fresh=True, aim_error_px=20.0, fire_tolerance_px=140.0,
                    identity_ok=True, focus_ok=True)
        base.update(over)
        return G.Evidence(**base)

    def test_land_on_the_line_blocks_the_shot(self):
        decision = G.fire_decision(self.evidence(
            terrain_clear=False, terrain_detail={"land_samples": 7}))
        self.assertFalse(decision["allow"])
        self.assertIn(G.TERRAIN_BLOCKS_SHOT, decision["blockers"])
        self.assertEqual(decision["gates"]["terrain_clear"]["status"], "fail")

    def test_a_clear_line_passes(self):
        decision = G.fire_decision(self.evidence(terrain_clear=True))
        self.assertTrue(decision["allow"], decision["blockers"])
        self.assertEqual(decision["gates"]["terrain_clear"]["status"], "pass")

    def test_no_terrain_evidence_does_not_veto(self):
        decision = G.fire_decision(self.evidence(terrain_clear=None))
        self.assertTrue(decision["allow"], decision["blockers"])
        self.assertEqual(decision["gates"]["terrain_clear"]["status"], "unknown")

    def test_an_unrendered_hull_alone_does_not_veto(self):
        decision = G.fire_decision(self.evidence(
            terrain_clear=True, body_visible=False))
        self.assertTrue(decision["allow"], decision["blockers"])

    def test_an_unrendered_hull_with_land_on_the_line_blocks(self):
        decision = G.fire_decision(self.evidence(
            terrain_clear=False, body_visible=False))
        self.assertFalse(decision["allow"])
        self.assertIn(G.TERRAIN_BLOCKS_SHOT, decision["blockers"])


if __name__ == '__main__':
    unittest.main()
