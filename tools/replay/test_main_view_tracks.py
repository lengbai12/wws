"""Ship and ordinary HUD-marker continuity, ambiguity and source-reset behavior."""
import copy
import unittest

from main_view_tracks import MainViewTracker


def view(xs, shift=(0., 0., 1.)):
    detections = [dict(nameplate_bbox=[x, 400, 80, 8], team='enemy') for x in xs]
    return dict(detections=detections, camera_shift=None if shift is None else
                dict(zip(('dx', 'dy', 'score'), shift)))



def marker_view(markers, shift=(0., 0., 1.)):
    return dict(detections=[], markers=markers,
                camera_shift=None if shift is None else
                dict(zip(('dx', 'dy', 'score'), shift)))


def aircraft(x, aircraft_type='fighter', team='enemy', **extra):
    result = dict(**{'class': 'aircraft'}, bbox=[x, 500, 20, 20],
                  aircraft_type=aircraft_type, team=team,
                  marker_family='controllable', evidence='aircraft')
    result.update(extra)
    return result


def torpedo(x, marker_type='normal', **extra):
    result = dict(**{'class': 'torpedo'}, bbox=[x, 500, 16, 16],
                  marker_type=marker_type, evidence='native_enemy_torpedo_warning_sprite')
    result.update(extra)
    return result

class MainViewTrackTests(unittest.TestCase):
    def test_continuity_uses_camera_compensation_and_observation_dt(self):
        c = MainViewTracker()
        a = c.update(view([100]), 1_000_000_000)['detections'][0]
        b = c.update(view([120], (10., 0., 1.)), 1_500_000_000)['detections'][0]
        self.assertEqual(a['track_id'], b['track_id'])
        self.assertEqual(b['track_state'], 'tracked')
        self.assertEqual(b['screen_velocity_px_s'], [20., 0.])
        self.assertEqual(b['track_hits'], 2)

    def test_overlapping_contenders_cannot_inherit_identity(self):
        c = MainViewTracker()
        old = c.update(view([100]), 1_000_000_000)['detections'][0]['track_id']
        out = c.update(view([98, 102]), 1_100_000_000)['detections']
        self.assertTrue(all(d['track_state'] == 'ambiguous' for d in out))
        self.assertNotIn(old, [d['track_id'] for d in out])
        self.assertTrue(all(d['screen_velocity_px_s'] is None for d in out))

    def test_missing_frame_and_reset_do_not_reuse_identity(self):
        c = MainViewTracker()
        old = c.update(view([100]), 1_000_000_000)['detections'][0]['track_id']
        c.update(view([]), 1_100_000_000)
        new = c.update(view([100]), 1_200_000_000)['detections'][0]['track_id']
        self.assertNotEqual(old, new)
        c.reset()
        latest = c.update(view([100]), 1_300_000_000)['detections'][0]
        self.assertNotEqual(new, latest['track_id'])
        self.assertEqual(latest['track_hits'], 1)

    def test_missing_camera_keeps_identity_but_never_claims_motion(self):
        """Without a usable camera shift a nearby match may still carry identity.

        The displacement then mixes camera motion with target motion, so the
        track keeps its id, claims no screen velocity, and says why.
        """
        c = MainViewTracker()
        old = c.update(view([100]), 1_000_000_000)['detections'][0]['track_id']
        new = c.update(view([100], None), 1_100_000_000)['detections'][0]
        self.assertEqual(old, new['track_id'])
        self.assertIsNone(new['screen_velocity_px_s'])
        self.assertIn('uncompensated', new['velocity_semantics'])

    def test_a_far_detection_under_the_fallback_starts_a_new_track(self):
        c = MainViewTracker()
        old = c.update(view([100]), 1_000_000_000)['detections'][0]['track_id']
        new = c.update(view([400], None), 1_100_000_000)['detections'][0]
        self.assertNotEqual(old, new['track_id'])

    def test_a_long_gap_never_bridges_identity(self):
        for stamp, shift in ((2_600_000_000, None), (2_100_000_000, (0., 0., 1.))):
            with self.subTest(stamp=stamp):
                c = MainViewTracker()
                old = c.update(view([100]), 1_000_000_000)['detections'][0]['track_id']
                new = c.update(view([100], shift), stamp)['detections'][0]
                self.assertNotEqual(old, new['track_id'])
                self.assertIsNone(new['screen_velocity_px_s'])

    def test_lock_binds_only_current_unique_healthbar(self):
        c = MainViewTracker()
        current = c.update(view([100]), 1_000_000_000)
        lock = dict(confirmed=True, target_association='unique_healthbar', detection_index=0)
        out = c.bind_lock(copy.deepcopy(lock), current)
        self.assertEqual(out['target_track_id'], current['detections'][0]['track_id'])
        self.assertNotIn('target_id', out)  # no calibrated firing target
        lock['confirmed'] = False
        self.assertIsNone(c.bind_lock(lock, current)['target_track_id'])
    def test_lock_drops_ambiguous_track_identity(self):
        c = MainViewTracker()
        c.update(view([100]), 1_000_000_000)
        current = c.update(view([98, 102]), 1_100_000_000)
        lock = dict(confirmed=True, target_association='unique_healthbar', detection_index=0)
        out = c.bind_lock(lock, current)
        self.assertEqual('ambiguous', out['track_state'])
        self.assertIsNone(out['target_track_id'])
        self.assertFalse(out['confirmed'])
        self.assertEqual(out['state'], 'ambiguous')
        self.assertEqual(out['target_association'], 'ambiguous_track')
        self.assertEqual(out['reason'], 'ambiguous_track_association')

    def test_ui_target_tracks_healthbar_motion_without_fabricating_hull(self):
        tracker = MainViewTracker()
        def frame(x, clock):
            current = view([x], (10., 0., 1.))
            current['detections'][0]['class'] = 'ship'
            tracker.update(current, clock)
            lock = tracker.bind_lock(dict(confirmed=True,
                target_association='unique_healthbar', detection_index=0), current)
            return tracker.ui_target(lock, current), current, lock
        first, _, _ = frame(100, 1_000_000_000)
        target, current, lock = frame(120, 1_500_000_000)
        self.assertEqual(first['id'], target['id'])
        self.assertEqual('tracked', target['state'])
        self.assertEqual([20., 0.], target['screen_velocity_px_s'])
        self.assertIsNone(target['anchor_px'])
        self.assertIsNone(target['velocity_px_s'])
        self.assertIsNone(target['world_velocity'])
        lock['confirmed'] = False
        self.assertIsNone(tracker.ui_target(lock, current))
        lock['confirmed'] = True
        current['detections'][0]['track_state'] = 'ambiguous'
        self.assertIsNone(tracker.ui_target(lock, current))

    def test_ui_target_rejects_wrong_side_or_track_identity(self):
        tracker = MainViewTracker()
        for team in ('enemy', 'ally'):
            current = view([100])
            current['detections'][0].update({'class': 'ship', 'team': team})
            tracker.update(current, 1_000_000_000)
            lock = tracker.bind_lock(dict(confirmed=True,
                target_association='unique_healthbar', detection_index=0), current)
            if team == 'enemy':
                lock['target_track_id'] += 1
            self.assertIsNone(tracker.ui_target(lock, current))


    def test_ordinary_marker_continuity_requires_isolated_same_type_team(self):
        c = MainViewTracker()
        first = c.update(marker_view([aircraft(100)]), 1_000_000_000)['markers'][0]
        current = c.update(marker_view([aircraft(125)], shift=(5., 0., 1.)), 1_500_000_000)['markers'][0]
        self.assertEqual(first['track_id'], current['track_id'])
        self.assertEqual(current['track_state'], 'tracked')
        self.assertEqual(current['screen_velocity_px_s'], [40., 0.])
        self.assertIn('not_physical_or_world_velocity', current['velocity_semantics'])

    def test_marker_swap_cross_class_or_type_cannot_inherit_identity(self):
        c = MainViewTracker()
        first = c.update(marker_view([aircraft(100)]), 1_000_000_000)['markers'][0]
        swapped = c.update(marker_view([torpedo(100)], shift=(0., 0., 1.)), 1_100_000_000)['markers'][0]
        self.assertNotEqual(first['track_id'], swapped['track_id'])
        self.assertEqual(swapped['track_state'], 'tentative')
        self.assertIsNone(swapped['screen_velocity_px_s'])

    def test_ambiguous_marker_candidates_do_not_inherit_identity(self):
        c = MainViewTracker()
        first = c.update(marker_view([aircraft(100)]), 1_000_000_000)['markers'][0]
        current = c.update(marker_view([aircraft(95), aircraft(105)], shift=(0., 0., 1.)), 1_100_000_000)['markers']
        self.assertTrue(all(item['track_state'] == 'ambiguous' for item in current))
        self.assertNotIn(first['track_id'], [item['track_id'] for item in current])

    def test_separated_salvo_markers_keep_two_way_nearest_ids(self):
        for make in (torpedo, aircraft):
            with self.subTest(kind=make.__name__):
                tracker = MainViewTracker()
                first = tracker.update(marker_view([make(100), make(140)]),
                                       1_000_000_000)['markers']
                current = tracker.update(marker_view([make(142), make(103)]),
                                         1_200_000_000)['markers']
                self.assertEqual([first[1]['track_id'], first[0]['track_id']],
                                 [m['track_id'] for m in current])
                self.assertEqual(['tracked', 'tracked'], [m['track_state'] for m in current])
                self.assertEqual([[10., 0.], [15., 0.]],
                                 [m['screen_velocity_px_s'] for m in current])

    def test_marker_crossing_or_close_competitor_stays_ambiguous(self):
        for positions in ([119, 121], [101, 106]):
            with self.subTest(positions=positions):
                tracker = MainViewTracker()
                first = tracker.update(marker_view([torpedo(100), torpedo(140)]),
                                       1_000_000_000)['markers']
                current = tracker.update(marker_view([torpedo(x) for x in positions]),
                                         1_200_000_000)['markers']
                self.assertTrue(all(m['track_state'] == 'ambiguous' for m in current))
                self.assertTrue(all(m['screen_velocity_px_s'] is None for m in current))
                self.assertTrue(set(m['track_id'] for m in first).isdisjoint(
                    m['track_id'] for m in current))

    def test_alerts_remain_hud_only_without_track_or_velocity(self):
        c = MainViewTracker()
        alert = aircraft(900, aircraft_type='unknown', team='unknown', marker_family='central_dangers',
                         observation_semantics='screen_hud_danger_indicator_only')
        out = c.update(marker_view([alert]), 1_000_000_000)['markers'][0]
        self.assertIsNone(out['track_id'])
        self.assertEqual(out['track_state'], 'observed')
        self.assertIsNone(out['screen_velocity_px_s'])
        self.assertIn('entity_motion_unknown', out['velocity_semantics'])


if __name__ == '__main__':
    unittest.main()
