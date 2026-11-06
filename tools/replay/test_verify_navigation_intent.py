"""Saved navigation decisions must respect their recorded decision clock."""
import json
from pathlib import Path
import tempfile
import unittest

from build_navigation_reference import build_reference
from verify_navigation_intent import verify


class SavedDecisionClockTests(unittest.TestCase):
    def row(self):
        origin = build_reference()['waypoints'][0]
        return dict(session='236374', sample=0, replay_clock_s=30,
            observation=dict(epoch=0, seq=0, fresh=True, captured_ns=1_000_000_000,
                valid_until_ns=1_750_000_000,
                own=dict(pixel_x=origin['x'], pixel_y=origin['y'])))

    def run_row(self, row):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'observations.jsonl'
            source.write_text(json.dumps(row) + '\n')
            verify(source, root / 'result')
            return json.loads((root / 'result/intents.jsonl').read_text())

    def test_recorded_expiry_boundary_revokes_despite_cached_fresh(self):
        row = self.row()
        row['decision_ns'] = row['observation']['valid_until_ns']
        result = self.run_row(row)
        self.assertEqual('snapshot_expired_or_future', result['intent']['reason'])
        self.assertFalse(result['intent']['movement_authorized'])
        self.assertIsNone(result['intent']['waypoint'])

    def test_invalid_recorded_decision_cannot_fall_back_to_acquisition(self):
        for value in (None, True, -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'decision_ns'):
                row = self.row()
                row['decision_ns'] = value
                self.run_row(row)

    def test_legacy_simulation_uses_acquisition_and_never_replay_clock(self):
        row = self.row()
        row['replay_clock_s'] = 1_000_000
        result = self.run_row(row)
        self.assertEqual('reference_waypoint', result['intent']['reason'])
        self.assertEqual('captured_ns_simulation', result['decision_clock_source'])
        row['observation']['captured_ns'] = None
        with self.assertRaisesRegex(ValueError, 'metrics-only'):
            self.run_row(row)


if __name__ == '__main__':
    unittest.main()
