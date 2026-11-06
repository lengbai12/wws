"""Behavior regressions for conservative observation-only enemy continuity."""
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from enemy_tracks import EnemyTracker


def detection(x, y=50, species='unknown'):
    return dict(u=x / 100, v=y / 100, species=species)


def tracker(**kwargs):
    return EnemyTracker(width=101, height=101, **kwargs)


class EnemyTrackBehavior(unittest.TestCase):
    def test_stable_moving_id_and_actual_dt_velocity(self):
        subject = tracker()
        first = subject.update([detection(20, 40, 'Cruiser')], 10)[0]
        second = subject.update([detection(24, 42, 'Cruiser')], 10.5)[0]
        third = subject.update([detection(36, 48, 'Cruiser')], 12)[0]
        self.assertEqual(first['track_id'], second['track_id'])
        self.assertEqual(first['track_id'], third['track_id'])
        self.assertEqual('tentative', first['state'])
        self.assertFalse(first['usable'])
        self.assertIsNone(first['velocity_px_s'])
        self.assertEqual('confirmed', third['state'])
        self.assertTrue(third['observed'])
        self.assertTrue(third['usable'])
        self.assertEqual(3, third['hits'])
        self.assertAlmostEqual(8, third['velocity_px_s']['x'])
        self.assertAlmostEqual(4, third['velocity_px_s']['y'])
        self.assertEqual(0, third['age_s'])
        self.assertEqual(12, third['last_seen_s'])
        self.assertEqual((.36, .48), (third['u'], third['v']))
        json.dumps(third, allow_nan=False)

    def test_crossing_abstains_without_duplicate_identities(self):
        subject = tracker()
        initial = subject.update([detection(20), detection(60)], 0)
        confirmed = subject.update([detection(28), detection(52)], 1)
        self.assertTrue(all(row['usable'] for row in confirmed))
        crossing = subject.update([detection(36), detection(44)], 2)
        self.assertEqual([row['track_id'] for row in initial],
                         [row['track_id'] for row in crossing])
        self.assertEqual(2, len(crossing))
        for row, previous in zip(crossing, confirmed):
            self.assertEqual('ambiguous', row['state'])
            self.assertFalse(row['observed'])
            self.assertFalse(row['usable'])
            self.assertEqual(previous['u'], row['u'])
            self.assertEqual(previous['last_seen_s'], row['last_seen_s'])
            self.assertEqual(previous['hits'], row['hits'])
            self.assertEqual(1, row['age_s'])

    def test_entire_connected_component_is_ambiguous(self):
        subject = tracker(max_speed_px_s=0, position_slack_px=10,
                          association_margin_px=0)
        initial = subject.update([detection(20), detection(38)], 0)
        rows = subject.update([detection(11), detection(29), detection(47)], 1)
        self.assertEqual([row['track_id'] for row in initial],
                         [row['track_id'] for row in rows])
        self.assertTrue(all(row['state'] == 'ambiguous' for row in rows))
        self.assertTrue(all(not row['observed'] and not row['usable'] for row in rows))

    def test_one_track_multiple_candidates_and_multiple_tracks_one_candidate(self):
        for initial, following in (([detection(20)], [detection(19), detection(21)]),
                                   ([detection(19), detection(21)], [detection(20)])):
            with self.subTest(initial=initial):
                subject = tracker()
                before = subject.update(initial, 0)
                after = subject.update(following, 1)
                self.assertEqual(len(before), len(after))
                self.assertTrue(all(row['state'] == 'ambiguous' for row in after))

    def test_input_order_cannot_resolve_crossing(self):
        outputs = []
        for reverse in (False, True):
            subject = tracker()
            subject.update([detection(20), detection(60)], 0)
            subject.update([detection(28), detection(52)], 1)
            following = [detection(36), detection(44)]
            outputs.append(subject.update(following[::-1] if reverse else following, 2))
        self.assertEqual(outputs[0], outputs[1])

    def test_missing_lost_reacquired_with_new_confirmation_streak(self):
        subject = tracker()
        first = subject.update([detection(20)], 0)[0]
        subject.update([detection(28)], 1)
        lost = subject.update([], 1.25)[0]
        self.assertEqual('lost', lost['state'])
        self.assertFalse(lost['observed'])
        self.assertFalse(lost['usable'])
        self.assertEqual(.28, lost['u'])
        self.assertEqual(.25, lost['age_s'])
        reacquired = subject.update([detection(40)], 2.5)[0]
        self.assertEqual(first['track_id'], reacquired['track_id'])
        self.assertEqual('tentative', reacquired['state'])
        self.assertTrue(reacquired['observed'])
        self.assertFalse(reacquired['usable'])
        self.assertEqual(0, reacquired['age_s'])
        self.assertAlmostEqual(8, reacquired['velocity_px_s']['x'])
        confirmed = subject.update([detection(44)], 3)[0]
        self.assertEqual('confirmed', confirmed['state'])
        self.assertTrue(confirmed['usable'])
        self.assertEqual(4, confirmed['hits'])

    def test_ambiguity_also_restarts_confirmation(self):
        subject = tracker()
        subject.update([detection(20)], 0)
        subject.update([detection(20)], .5)
        subject.update([detection(19), detection(21)], 1)
        row = subject.update([detection(20)], 1.5)[0]
        self.assertEqual('tentative', row['state'])
        self.assertTrue(row['observed'])
        self.assertFalse(row['usable'])
        self.assertTrue(subject.update([detection(20)], 2)[0]['usable'])

    def test_expiry_boundary_and_freshness(self):
        subject = tracker(confirm_hits=1)
        first = subject.update([detection(20)], 0)[0]
        self.assertEqual(first, subject.expire(0)[0])
        boundary = subject.expire(2.5)[0]
        self.assertEqual(first['track_id'], boundary['track_id'])
        self.assertEqual(2.5, boundary['age_s'])
        self.assertEqual('lost', boundary['state'])
        self.assertFalse(boundary['observed'])
        self.assertFalse(boundary['usable'])
        self.assertEqual(.2, boundary['u'])
        self.assertEqual(boundary, subject.expire(2.5)[0])
        self.assertEqual([], subject.expire(math.nextafter(2.5, math.inf)))

    def test_update_at_expiry_boundary_reacquires_but_after_allocates(self):
        for clock, same in ((2.5, True), (math.nextafter(2.5, math.inf), False)):
            with self.subTest(clock=clock):
                subject = tracker()
                first = subject.update([detection(20)], 0)[0]
                following = subject.update([detection(20)], clock)[0]
                self.assertEqual(same, first['track_id'] == following['track_id'])

    def test_species_conflict_is_ambiguous_and_creates_no_duplicate(self):
        subject = tracker()
        subject.update([detection(20, species='Cruiser')], 0)
        confirmed = subject.update([detection(20, species='Cruiser')], .5)[0]
        rows = subject.update([detection(20, species='Destroyer')], 1)
        self.assertEqual(1, len(rows))
        row = rows[0]
        self.assertEqual(confirmed['track_id'], row['track_id'])
        self.assertEqual('Cruiser', row['species'])
        self.assertEqual('ambiguous', row['state'])
        self.assertFalse(row['observed'])
        self.assertFalse(row['usable'])
        self.assertEqual(.5, row['last_seen_s'])
        self.assertEqual(2, row['hits'])

    def test_species_cannot_disambiguate_spatially_overlapping_candidates(self):
        subject = tracker()
        subject.update([detection(20, species='Cruiser'),
                        detection(30, species='Destroyer')], 0)
        rows = subject.update([detection(24, species='Cruiser'),
                               detection(26, species='Destroyer')], .5)
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row['state'] == 'ambiguous' for row in rows))

    def test_unknown_species_defaults_and_preserves_known_evidence(self):
        subject = tracker()
        row = subject.update([dict(u=.2, v=.5)], 0)[0]
        self.assertEqual('unknown', row['species'])
        known = subject.update([detection(20, species='Cruiser')], .5)[0]
        unknown = subject.update([detection(20)], 1)[0]
        self.assertEqual(row['track_id'], unknown['track_id'])
        self.assertEqual('Cruiser', known['species'])
        self.assertEqual('Cruiser', unknown['species'])
        self.assertTrue(unknown['usable'])

    def test_displacement_gate_uses_crop_pixel_scale(self):
        subject = EnemyTracker(width=720, height=723, max_speed_px_s=8,
                               position_slack_px=0, association_margin_px=0)
        first = subject.update([dict(u=100 / 719, v=100 / 722)], 0)[0]
        next_row = subject.update([dict(u=104 / 719, v=102 / 722)], 1)[0]
        self.assertEqual(first['track_id'], next_row['track_id'])
        self.assertAlmostEqual(4, next_row['velocity_px_s']['x'])
        self.assertAlmostEqual(2, next_row['velocity_px_s']['y'])
        rows = subject.update([dict(u=120 / 719, v=102 / 722)], 2)
        self.assertEqual(2, len(rows))
        self.assertEqual('lost', rows[0]['state'])
        self.assertEqual('tentative', rows[1]['state'])

    def test_prediction_uses_velocity_and_never_becomes_an_observation(self):
        subject = tracker(position_slack_px=0, association_margin_px=0)
        subject.update([detection(10)], 0)
        confirmed = subject.update([detection(18)], 1)[0]
        rows = subject.update([detection(10)], 2)
        self.assertEqual(2, len(rows))
        self.assertEqual('lost', rows[0]['state'])
        self.assertEqual(confirmed['u'], rows[0]['u'])
        self.assertEqual(confirmed['last_seen_s'], rows[0]['last_seen_s'])
        self.assertFalse(rows[0]['observed'])
        self.assertFalse(rows[0]['usable'])

    def test_margin_blocks_near_competitor_and_cannot_authorize_association(self):
        subject = tracker(max_speed_px_s=0, position_slack_px=6,
                          association_margin_px=3)
        initial = subject.update([detection(20)], 0)[0]
        rows = subject.update([detection(25), detection(28)], 1)
        self.assertEqual(1, len(rows))
        self.assertEqual('ambiguous', rows[0]['state'])
        self.assertEqual(initial['track_id'], rows[0]['track_id'])
        subject.reset()
        subject.update([detection(20)], 0)
        rows = subject.update([detection(28)], 1)
        self.assertEqual(1, len(rows))
        self.assertEqual('ambiguous', rows[0]['state'])
        self.assertFalse(rows[0]['observed'])

    def test_reset_and_expiry_never_reuse_allocated_ids(self):
        subject = tracker()
        first = subject.update([detection(20)], 100)[0]
        subject.reset()
        second = subject.update([detection(20)], 0)[0]
        self.assertGreater(second['track_id'], first['track_id'])
        self.assertEqual(1, second['hits'])
        self.assertIsNone(second['velocity_px_s'])
        self.assertEqual([], subject.expire(3))
        third = subject.update([detection(20)], 4)[0]
        self.assertGreater(third['track_id'], second['track_id'])

    def test_output_is_detached_and_metadata_cannot_change_identity(self):
        subject = tracker()
        item = dict(u=.2, v=.5, species='Cruiser', vehicle_id=123,
                    score=math.nan, icon_candidate='Destroyer')
        first = subject.update([item], 0)[0]
        item['u'] = .9
        second = subject.update([detection(20, species='Cruiser')], 1)[0]
        second['velocity_px_s']['x'] = 999
        second['u'] = .9
        snapshot = subject.expire(1)[0]
        self.assertEqual(first['track_id'], snapshot['track_id'])
        self.assertEqual(.2, snapshot['u'])
        self.assertEqual(0, snapshot['velocity_px_s']['x'])
        self.assertNotIn('vehicle_id', snapshot)


class EnemyTrackValidation(unittest.TestCase):
    def test_clock_errors_are_explicit_and_leave_state_unchanged(self):
        subject = tracker(confirm_hits=1)
        initial = subject.update([detection(20)], 1)
        for clock in (0, 1, True, None, '2', math.nan, math.inf, -math.inf):
            with self.subTest(clock=clock):
                with self.assertRaises(ValueError):
                    subject.update([detection(21)], clock)
                self.assertEqual(initial, subject.expire(1))
        for clock in (0, True, None, '2', math.nan, math.inf, -math.inf):
            with self.subTest(expire_clock=clock):
                with self.assertRaises(ValueError):
                    subject.expire(clock)
                self.assertEqual(initial, subject.expire(1))

    def test_expire_advances_sequence_clock(self):
        subject = tracker()
        subject.expire(1)
        self.assertEqual([], subject.expire(1))
        with self.assertRaises(ValueError):
            subject.update([], 1)
        subject.update([], 2)
        with self.assertRaises(ValueError):
            subject.expire(1.5)

    def test_malformed_detections_are_rejected_atomically(self):
        subject = tracker()
        initial = subject.update([detection(20)], 0)
        malformed = (None, {}, 'abc', iter([]), [None], [[]],
                     [dict(u=.2)], [dict(v=.5)],
                     [dict(u=True, v=.5)], [dict(u='.2', v=.5)],
                     [dict(u=.2, v=None)], [dict(u=-.01, v=.5)],
                     [dict(u=.2, v=1.01)], [dict(u=math.nan, v=.5)],
                     [dict(u=.2, v=math.inf)], [dict(u=-math.inf, v=.5)],
                     [dict(u=.2, v=.5, species=None)],
                     [dict(u=.2, v=.5, species='')],
                     [detection(21), dict(u=math.nan, v=.5)])
        for value in malformed:
            with self.subTest(detections=value):
                with self.assertRaises(ValueError):
                    subject.update(value, 1)
                self.assertEqual(initial, subject.expire(0))
        self.assertEqual(initial[0]['track_id'],
                         subject.update([detection(20)], 1)[0]['track_id'])

    def test_normalized_edges_and_tuple_sequence_are_allowed(self):
        subject = tracker()
        rows = subject.update((dict(u=0, v=0), dict(u=1, v=1)), 0)
        self.assertEqual([(0., 0.), (1., 1.)], [(row['u'], row['v']) for row in rows])

    def test_elapsed_clock_overflow_is_rejected(self):
        subject = tracker()
        initial = subject.update([detection(20)], -1e308)
        with self.assertRaises(ValueError):
            subject.update([], 1e308)
        with self.assertRaises(ValueError):
            subject.expire(1e308)
        self.assertEqual(initial, subject.expire(-1e308))

    def test_velocity_overflow_rejects_update_without_mutation(self):
        subject = tracker()
        initial = subject.update([detection(20)], 0)
        with self.assertRaises(ValueError):
            subject.update([detection(20.001)], math.nextafter(0., 1.))
        self.assertEqual(initial, subject.expire(0))
        self.assertEqual(initial[0]['track_id'],
                         subject.update([detection(21)], 1)[0]['track_id'])

    def test_zero_gap_keeps_only_same_clock_observations(self):
        subject = tracker(max_gap_s=0)
        initial = subject.update([detection(20)], 0)
        self.assertEqual(initial, subject.expire(0))
        self.assertEqual([], subject.expire(math.nextafter(0., 1.)))

    def test_configuration_rejects_overflowing_geometry(self):
        for options in (dict(width=10 ** 400), dict(height=10 ** 400),
                        dict(max_speed_px_s=1e308, max_gap_s=10),
                        dict(position_slack_px=1e308, association_margin_px=1e308)):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    EnemyTracker(**options)

    def test_configuration_rejects_nonfinite_and_malformed_values(self):
        for name in ('width', 'height'):
            for value in (True, 0, 1, 2.5, '720', math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        EnemyTracker(**{name: value})
        for value in (True, 0, -1, 2.5, '2'):
            with self.subTest(confirm_hits=value):
                with self.assertRaises(ValueError):
                    tracker(confirm_hits=value)
        for name in ('max_speed_px_s', 'position_slack_px',
                     'association_margin_px', 'max_gap_s'):
            for value in (True, None, '1', -1, math.nan, math.inf, -math.inf):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        tracker(**{name: value})


if __name__ == '__main__':
    unittest.main()
