"""Bounded observation plans must not turn UI evidence into input authority."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scope_observation import ScopeObservationPlanner


class ScopeObservationContract(unittest.TestCase):
    def observation(self, seq, captured, mode='scope', track=7, state='tracked'):
        return dict(epoch=0, seq=seq, captured_ns=captured,
            valid_until_ns=captured+750_000_000, fresh=True,
            hud=dict(battle=dict(active=True), weapon=dict(kind='gun', status='observed'),
                     aiming_view=dict(status='observed' if mode == 'scope' else 'unknown',
                                      mode='scope_ruler' if mode == 'scope' else 'unknown')),
            camera=dict(status='observed' if mode == 'external' else 'unknown', mode=mode),
            view=dict(detections=[dict(team='enemy', evidence='nameplate_healthbar',
                nameplate_bbox=[1100, 600, 120, 12], track_id=track, track_state=state)]),
            lock=dict(confirmed=False))

    def decide(self, planner, seq, captured, **kwargs):
        return planner.decide(self.observation(seq, captured, **kwargs),
                              now_ns=captured+10_000_000, focused=True)

    def test_unknown_view_cannot_request_shift(self):
        result = self.decide(ScopeObservationPlanner(), 0, 1_000_000_000, mode='unknown')
        self.assertEqual('view_unknown', result['state'])
        self.assertIsNone(result['suggested_key'])
        self.assertFalse(result['input_authorized'])

    def test_unacknowledged_transition_times_out_without_repeating_shift(self):
        planner = ScopeObservationPlanner()
        first = self.decide(planner, 0, 1_000_000_000, mode='external')
        self.assertEqual('request_scope', first['action'])
        waiting = self.decide(planner, 1, 1_250_000_000, mode='external')
        self.assertEqual('awaiting_scope', waiting['state'])
        self.assertIsNone(waiting['suggested_key'])
        failed = self.decide(planner, 2, 3_000_000_000, mode='external')
        self.assertEqual('transition_failed', failed['state'])
        self.assertIsNone(failed['suggested_key'])
        self.assertFalse(failed['input_authorized'])

    def test_scope_requires_reacquisition_and_never_authorizes_fire(self):
        planner = ScopeObservationPlanner()
        self.decide(planner, 0, 1_000_000_000, mode='external')
        first = self.decide(planner, 1, 1_250_000_000)
        self.assertEqual('reacquiring', first['state'])
        stable = self.decide(planner, 2, 1_500_000_000)
        self.assertEqual('confirm_hull', stable['action'])
        self.assertFalse(stable['fire_authorized'])
        self.assertFalse(stable['input_authorized'])
        self.assertEqual(1_660_000_000, stable['valid_until_ns'])

    def test_previous_lease_boundary_breaks_same_id_continuity(self):
        planner = ScopeObservationPlanner()
        self.decide(planner, 0, 1_000_000_000)
        # Decision is exactly at the previous frame's deadline.
        later = self.decide(planner, 1, 1_740_000_000)
        self.assertEqual('reacquiring', later['state'])

    def test_changed_track_or_missing_sequence_requires_reacquisition(self):
        for seq, track in ((1, 8), (2, 7)):
            with self.subTest(seq=seq, track=track):
                planner = ScopeObservationPlanner()
                self.decide(planner, 0, 1_000_000_000)
                result = self.decide(planner, seq, 1_250_000_000, track=track)
                self.assertEqual('reacquiring', result['state'])

    def test_ambiguity_cannot_select_healthbar_without_unique_lock(self):
        observation = self.observation(0, 1_000_000_000)
        observation['view']['detections'].append(dict(observation['view']['detections'][0], track_id=8))
        planner = ScopeObservationPlanner()
        result = planner.decide(observation, now_ns=1_010_000_000, focused=True)
        self.assertEqual('healthbar_ambiguous', result['reason'])
        observation.update(seq=1, captured_ns=1_100_000_000)
        observation['lock'] = dict(confirmed=True, target_association='unique_healthbar', detection_index=1)
        selected = planner.decide(observation, now_ns=1_110_000_000, focused=True)
        self.assertEqual(1, selected['detection_index'])
        self.assertEqual('reacquiring', selected['state'])

    def test_expiry_focus_loss_and_disconnect_revoke_plan(self):
        for controls in (dict(now_ns=1_750_000_000, focused=True),
                         dict(now_ns=1_010_000_000, focused=False),
                         dict(now_ns=1_010_000_000, focused=True, source_available=False)):
            with self.subTest(controls=controls):
                result = ScopeObservationPlanner().decide(self.observation(0, 1_000_000_000), **controls)
                self.assertEqual('revoke', result['action'])
                self.assertTrue(result['release_buttons'])
                self.assertIsNone(result['healthbar_pixel'])
                self.assertIsNone(result['suggested_key'])


if __name__ == '__main__':
    unittest.main()
