#!/usr/bin/env python3
"""Behavioral navigation contracts; no input injection or inference."""
import math
import unittest

from build_navigation_reference import PATTERN, build_reference
from navigation_intent import NavigationIntent
from verify_navigation_intent import CONTEXT


def reference():
    r = build_reference()
    r['waypoints'] = [dict(x=100.0, y=100.0, s_px=0.0),
                      dict(x=150.0, y=100.0, s_px=50.0),
                      dict(x=200.0, y=100.0, s_px=100.0)]
    r['length_px'] = 100.0
    return r


def snapshot(x=100, y=100, seq=0, epoch=0, captured=None):
    captured = 1_000_000_000 + seq * 1_000_000_000 if captured is None else captured
    return dict(epoch=epoch, seq=seq, fresh=True, captured_ns=captured,
                valid_until_ns=captured + 750_000_000,
                own=dict(pixel_x=x, pixel_y=y))


def decide(consumer, row, **kwargs):
    context = dict(CONTEXT, now_ns=row['captured_ns'])
    context.update(kwargs)
    return consumer.decide(row, **context)


class NavigationTests(unittest.TestCase):
    def test_actual_projection_and_provenance(self):
        r = build_reference()
        p = r['waypoints'][0]
        self.assertAlmostEqual(p['x'], .44760868152757827 * -271.1 + .005457798047379158 * -283.5 + 2193.826939343883 - 1836)
        self.assertAlmostEqual(p['y'], .001614817285486262 * -271.1 - .451233159932871 * -283.5 + 1156.492463197363 - 796)
        self.assertEqual(r['ship'], 'PGSB210-Schlieffen')
        self.assertEqual(r['pattern'], PATTERN)
        self.assertEqual(r['source_coordinate']['units'], 'replay_world_units')
        for p in r['waypoints']:
            self.assertTrue(0 <= p['x'] < 720 and 0 <= p['y'] < 723)

    def test_progress_transition_and_completion(self):
        c = NavigationIntent(reference())
        last = 0
        for seq, x in enumerate((100, 120, 140, 160, 180, 200)):
            result = decide(c, snapshot(x=x, seq=seq))
            self.assertGreaterEqual(result['progress']['s_px'], last)
            self.assertLessEqual(result['progress']['s_px'] - last, 30)
            last = result['progress']['s_px']
            if x < 200:
                self.assertEqual(result['action'], 'autopilot_waypoint')
                self.assertLessEqual(result['valid_until_ns'], snapshot(x=x, seq=seq)['valid_until_ns'])
                self.assertIsNone(result['heading']['yaw_rad'])
                self.assertFalse(result['input_authorized'])
        self.assertEqual(result['reason'], 'reference_complete')
        self.assertIsNone(result['waypoint'])

    def test_clock_does_not_advance_route(self):
        c = NavigationIntent(reference())
        a = decide(c, snapshot())
        b = decide(c, snapshot(seq=1, captured=10**15))
        self.assertEqual(a['progress'], b['progress'])
        self.assertEqual(a['waypoint'], b['waypoint'])

    def test_revoke_causes_independent(self):
        for kwargs, expected in ((dict(focused=False), 'focus_lost'),
                                 (dict(source_available=False), 'source_lost'),
                                 (dict(mechanism='unknown'), 'unknown_navigation_mechanism'),
                                 (dict(pattern='unknown'), 'unknown_map_ship_or_pattern'),
                                 (dict(map_id=None), 'unknown_map_ship_or_pattern'),
                                 (dict(ship=None), 'unknown_map_ship_or_pattern'),
                                 (dict(now_ns=2_000_000_000), 'snapshot_expired_or_future')):
            with self.subTest(expected=expected, kwargs=kwargs):
                result = decide(NavigationIntent(reference()), snapshot(), **kwargs)
                self.assertEqual(result['reason'], expected)
                self.assertIsNone(result['waypoint'])
                self.assertFalse(result['movement_authorized'])

    def test_missing_stale_and_invalid(self):
        for mutate, expected in ((dict(own=None), 'own_missing'),
                                 (dict(fresh=False), 'snapshot_not_fresh'),
                                 (dict(valid_until_ns=None), 'invalid_snapshot'),
                                 (dict(own=dict(pixel_x=math.nan, pixel_y=2)), 'invalid_own'),
                                 (dict(own=dict(pixel_x=True, pixel_y=2)), 'invalid_own'),
                                 (dict(own=dict(pixel_x=720, pixel_y=2)), 'invalid_own')):
            row = snapshot()
            row.update(mutate)
            self.assertEqual(decide(NavigationIntent(reference()), row)['reason'], expected)

    def test_backtracking_and_deviation(self):
        c = NavigationIntent(reference())
        decide(c, snapshot(x=120))
        result = decide(c, snapshot(x=100, seq=1))
        self.assertEqual(result['reason'], 'backtracking')
        self.assertEqual(result['progress']['s_px'], 20)
        self.assertEqual(decide(NavigationIntent(reference()), snapshot(y=200))['reason'], 'off_reference_corridor')

    def test_forward_bound_does_not_teleport(self):
        c = NavigationIntent(reference())
        result = decide(c, snapshot(x=200))
        self.assertEqual(result['reason'], 'off_reference_corridor')
        self.assertEqual(result['progress']['s_px'], 0)

    def test_epoch_reset_and_ordering(self):
        c = NavigationIntent(reference())
        decide(c, snapshot(x=120))
        changed = snapshot(x=100, seq=0, epoch=1, captured=2_000_000_000)
        self.assertEqual(decide(c, changed)['reason'], 'source_changed')
        self.assertEqual(c.progress, 0)
        self.assertEqual(decide(c, snapshot(seq=1, epoch=1, captured=3_000_000_000))['action'], 'autopilot_waypoint')
        self.assertEqual(decide(c, snapshot(seq=2, epoch=0, captured=4_000_000_000))['reason'], 'old_source_epoch')

    def test_clock_regressions_and_duplicate(self):
        c = NavigationIntent(reference())
        decide(c, snapshot())
        self.assertEqual(decide(c, snapshot())['reason'], 'out_of_order')
        result = decide(c, snapshot(seq=1, captured=999_999_999))
        self.assertEqual(result['reason'], 'decision_clock_regressed')
        row = snapshot(seq=2, captured=900_000_000)
        self.assertEqual(decide(c, row, now_ns=1_100_000_000)['reason'], 'out_of_order')

    def test_focus_and_source_revoke_clear_progress_before_recovery(self):
        c = NavigationIntent(reference())
        self.assertEqual(decide(c, snapshot(x=120))['action'], 'autopilot_waypoint')
        self.assertEqual(decide(c, snapshot(seq=1, x=120), focused=False)['reason'], 'focus_lost')
        self.assertEqual(c.progress, 0.0)
        recovered = decide(c, snapshot(seq=2, x=120))
        self.assertEqual(recovered['action'], 'autopilot_waypoint')
        self.assertEqual(recovered['progress']['s_px'], 20.0)
        self.assertEqual(decide(c, snapshot(seq=3, x=140), source_available=False)['reason'], 'source_lost')
        self.assertEqual(c.progress, 0.0)
        recovered = decide(c, snapshot(seq=4, x=120))
        self.assertEqual(recovered['action'], 'autopilot_waypoint')
        self.assertEqual(recovered['progress']['s_px'], 20.0)

    def test_invalid_own_does_not_authorize_or_project(self):
        c = NavigationIntent(reference())
        invalid = snapshot()
        invalid['own'] = None
        result = decide(c, invalid)
        self.assertEqual(result['reason'], 'own_missing')
        self.assertFalse(result['movement_authorized'])
        self.assertIsNone(result['waypoint'])
        recovered = decide(c, snapshot(seq=1, x=120))
        self.assertEqual(recovered['action'], 'autopilot_waypoint')
    def test_invalid_own_consumes_stamp(self):
        c = NavigationIntent(reference())
        invalid = snapshot()
        invalid['own'] = None
        self.assertEqual(decide(c, invalid)['reason'], 'own_missing')
        duplicate = snapshot(seq=0)
        self.assertEqual(decide(c, duplicate)['reason'], 'out_of_order')


    def test_stale_new_epoch_retires_previous_source_and_progress(self):
        c = NavigationIntent(reference())
        decide(c, snapshot(x=120))
        stale = snapshot(epoch=1, seq=0, captured=2_000_000_000)
        stale.update(fresh=False, own=None, captured_ns=None, valid_until_ns=None)
        result = c.decide(stale, now_ns=2_000_000_000, **CONTEXT)
        self.assertEqual(result['reason'], 'snapshot_not_fresh')
        self.assertEqual(result['progress']['s_px'], 0)
        self.assertEqual(decide(c, snapshot(epoch=0, seq=2, captured=3_000_000_000))['reason'],
                         'old_source_epoch')
        recovered = decide(c, snapshot(epoch=1, seq=1, captured=4_000_000_000))
        self.assertEqual(recovered['progress']['s_px'], 0)
        self.assertEqual(recovered['action'], 'autopilot_waypoint')

    def test_recovery_requires_capture_after_focus_or_source_loss(self):
        for loss in ('focused', 'source_available'):
            with self.subTest(loss=loss):
                c = NavigationIntent(reference())
                decide(c, snapshot(x=120))
                row = snapshot(seq=1, x=120, captured=1_100_000_000)
                decide(c, row, now_ns=1_200_000_000, **{loss: False})
                result = decide(c, row, now_ns=1_300_000_000)
                self.assertEqual(result['reason'], 'snapshot_before_recovery')
                self.assertFalse(result['movement_authorized'])
                self.assertIsNone(result['waypoint'])
                recovered = decide(c, snapshot(seq=2, x=120, captured=1_400_000_000))
                self.assertEqual(recovered['action'], 'autopilot_waypoint')

    def test_unknown_reference_rejected(self):
        for field, value in (('map', 'unknown'), ('ship', 'unknown'), ('pattern', 'unknown'),
                             ('projection_sha256', ''), ('length_px', math.inf)):
            r = reference()
            r[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                NavigationIntent(r)
        r = reference()
        r['waypoints'][1]['x'] = math.nan
        with self.assertRaises(ValueError):
            NavigationIntent(r)

    def test_reference_pattern_change_revokes(self):
        c = NavigationIntent(reference())
        decide(c, snapshot(x=120))
        result = decide(c, snapshot(seq=1), pattern='other_reference')
        self.assertEqual(result['reason'], 'unknown_map_ship_or_pattern')
        self.assertIsNone(result['waypoint'])


if __name__ == '__main__':
    unittest.main()
