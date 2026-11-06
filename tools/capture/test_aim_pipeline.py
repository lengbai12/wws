#!/usr/bin/env python3
"""The pipeline must refuse for the right reason and never invent evidence."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import aim_pipeline as P  # noqa: E402


def sense(**over):
    base = dict(battle=True, weapon='gun', reticle=[1280.5, 805.0],
                reticle_source='native_band', aim_km=13.0, aim_km_source='anchored',
                flight_s=7.8, plates=[], plate_count=0,
                minimap=dict(km=13.1, relative_bearing_deg=-30.0))
    base.update(over)
    return base


class TargetStageTests(unittest.TestCase):
    def test_nameplate_wins_over_the_minimap(self):
        target = P.stage_target(sense(plate_count=1, plates=[dict(
            bbox=[500, 300, 120, 17], trackable=True, distance_px=40.0,
            body_visible=True)]), allow_minimap=True)
        self.assertEqual(target['basis'], 'nameplate')

    def test_minimap_is_only_used_when_it_is_allowed(self):
        refused = P.stage_target(sense(), allow_minimap=False)
        self.assertEqual(refused['basis'], 'none')
        self.assertEqual(refused['reason'], 'no_target_evidence')
        allowed = P.stage_target(sense(), allow_minimap=True)
        self.assertEqual(allowed['basis'], 'minimap_contact')

    def test_no_evidence_at_all_stays_none(self):
        target = P.stage_target(sense(minimap=None), allow_minimap=True)
        self.assertEqual(target['basis'], 'none')


class AimStageTests(unittest.TestCase):
    def test_plate_error_is_measured_against_the_reticle(self):
        row = P.stage_aim(sense(), dict(basis='nameplate', plate=dict(
            bbox=[100, 700, 120, 17], trackable=True, distance_px=90.0,
            body_visible=True)))
        # centre of [100,700,120,17] is (160, 708.5)
        self.assertEqual(row['error_px'], [-1120.5, -96.5])
        self.assertEqual(row['action']['kind'], 'plate')

    def test_minimap_action_carries_yaw_and_elevation(self):
        row = P.stage_aim(sense(aim_km=14.1), dict(
            basis='minimap_contact', plate=None,
            contact=dict(km=13.1, relative_bearing_deg=-30.0)))
        self.assertAlmostEqual(row['action']['yaw_units'], -750.0, places=1)
        self.assertAlmostEqual(row['error_km'], 1.0, places=2)
        self.assertIn('elevation_units', row['action'])

    def test_missing_elevation_sensor_is_reported_not_guessed(self):
        row = P.stage_aim(sense(aim_km=None, flight_s=None), dict(
            basis='minimap_contact', plate=None,
            contact=dict(km=13.1, relative_bearing_deg=-30.0)))
        self.assertIn('no_elevation_sensor', row['notes'])
        self.assertNotIn('elevation_units', row['action'])

    def test_flight_time_is_used_when_the_km_reader_fails(self):
        row = P.stage_aim(sense(aim_km=None, flight_s=4.5), dict(
            basis='minimap_contact', plate=None,
            contact=dict(km=13.1, relative_bearing_deg=-30.0)))
        self.assertIn('elevation_on_flight_time', row['notes'])
        self.assertAlmostEqual(row['action']['target_flight_s'], 7.86, places=2)


class GateStageTests(unittest.TestCase):
    def test_a_plate_solution_inside_tolerance_is_allowed(self):
        decision = P.stage_gate(sense(), dict(
            basis='nameplate', plate=dict(bbox=[500, 300, 120, 17], body_visible=True),
            contact=None), dict(error_px=[30.0, 20.0]), 140.0)
        # The weapon reads gun, but the pipeline deliberately leaves reload and
        # the process identity unset, so those still block.
        self.assertNotIn('weapon_not_gun', decision['blockers'])
        self.assertIn('reload_not_ready', decision['blockers'])
        self.assertIsNone(decision['gates']['aim_error']['blocker'])
        self.assertTrue(decision['gates']['enemy_plate_current']['ok'])

    def test_minimap_basis_satisfies_the_plate_gate_only_when_fresh(self):
        fresh = P.stage_gate(sense(), dict(basis='minimap_contact', plate=None,
                                           contact=dict(km=13.1, relative_bearing_deg=-30.0)),
                             dict(error_px=None), 140.0)
        self.assertTrue(fresh['gates']['enemy_plate_current']['ok'])
        self.assertEqual(fresh['gates']['enemy_plate_current']['detail']['basis'],
                         'minimap_contact')
        stale = P.stage_gate(sense(), dict(basis='minimap_contact', plate=None, contact=None),
                             dict(error_px=None), 140.0)
        self.assertFalse(stale['gates']['enemy_plate_current']['ok'])

    def test_no_battle_refuses_without_looking_at_targets(self):
        decision = P.stage_gate(sense(battle=None), dict(basis='none', plate=None,
                                                         contact=None),
                                dict(error_px=None), 140.0)
        self.assertFalse(decision['allow'])


class DelegationTests(unittest.TestCase):
    """--execute must reach aim_servo, not vanish (it was deleted once)."""

    def _args(self, **over):
        class A:
            servo_seconds = 30.0
            fire_tolerance = 140.0
            max_steps = 20
            scan_steps = 12
            oneshot_budget_s = 25.0
            scan_budget_s = 60.0
            wait_reload_s = 0.0
            fire = True
        for k, v in over.items():
            setattr(A, k, v)
        return A()

    def test_the_delegation_function_exists(self):
        self.assertTrue(hasattr(P, '_run_servo'))

    def test_oneshot_is_chosen_when_a_plate_is_in_view(self):
        import tempfile, json as _json, subprocess as _sp
        from pathlib import Path as _P
        out = _P(tempfile.mkdtemp())
        seen = {}

        class Fake:
            returncode = 0
            stdout = ''
            stderr = ''
        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            (out / 'servo').mkdir(exist_ok=True)
            (out / 'servo/report.json').write_text(_json.dumps(dict(fired=True, fire_reason='ok')))
            return Fake()
        original = _sp.run
        _sp.run = fake_run
        try:
            verdict = P._run_servo(out, self._args(), plate_visible=True)
        finally:
            _sp.run = original
        self.assertIn('--oneshot', seen['cmd'])
        self.assertNotIn('--scan', seen['cmd'])
        self.assertTrue(verdict['fired'])
        self.assertEqual(verdict['mode'], 'oneshot')

    def test_scan_is_chosen_when_nothing_is_in_view(self):
        import tempfile, json as _json, subprocess as _sp
        from pathlib import Path as _P
        out = _P(tempfile.mkdtemp())
        seen = {}

        class Fake:
            returncode = 0
            stdout = ''
            stderr = ''
        def fake_run(cmd, **kw):
            seen['cmd'] = cmd
            (out / 'servo').mkdir(exist_ok=True)
            (out / 'servo/report.json').write_text(_json.dumps(dict(fired=False, error='no enemy nameplate in view')))
            return Fake()
        original = _sp.run
        _sp.run = fake_run
        try:
            verdict = P._run_servo(out, self._args(), plate_visible=False)
        finally:
            _sp.run = original
        self.assertIn('--scan', seen['cmd'])
        self.assertFalse(verdict['fired'])
        self.assertEqual(verdict['mode'], 'scan')


if __name__ == '__main__':
    unittest.main()
