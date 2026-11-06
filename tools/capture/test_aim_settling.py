"""Screen stability must not survive quality failures or transient motion."""
import unittest
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import probe_aim_settling as probe

from analyze_aim_settling import stable_tail


def pair(start, end, dx=0., sufficient=True):
    return dict(from_ns=int(start*1e9), to_ns=int(end*1e9),
                motion=dict(sufficient=sufficient, screen_dx_px=dx, screen_dy_px=0.))


class SettlingTests(unittest.TestCase):
    def test_transient_motion_resets_stable_tail(self):
        pairs = [pair(0, .2), pair(.2, .4, 3), pair(.4, .6), pair(.6, .8), pair(.8, 1.)]
        result = stable_tail(pairs)
        self.assertEqual(result['start_ns'], 400_000_000)
        self.assertEqual(result['intervals'], 3)

    def test_late_motion_invalidates_apparent_early_stability(self):
        self.assertIsNone(stable_tail([pair(0, .2), pair(.2, .4), pair(.4, .6), pair(.6, .8, 2)]))

    def test_ambiguous_match_cannot_establish_stationary_scene(self):
        self.assertIsNone(stable_tail([pair(0, .2), pair(.2, .4), pair(.4, .6, sufficient=False)]))

    def test_short_or_single_interval_is_insufficient(self):
        self.assertIsNone(stable_tail([pair(0, 1)]))
        self.assertIsNone(stable_tail([pair(0, .1), pair(.1, .2), pair(.2, .3)]))

    def test_slow_accumulating_drift_is_not_stability(self):
        self.assertIsNone(stable_tail([pair(0, .2, .2), pair(.2, .4, .2), pair(.4, .6, .2)]))

    def test_unsampled_gap_cannot_count_toward_stability(self):
        self.assertIsNone(stable_tail([pair(0, .2), pair(.2, .4), pair(.8, 1.)]))

    def test_long_capture_interval_cannot_prove_stable_tail(self):
        self.assertIsNone(stable_tail([pair(0, .3), pair(.3, .6), pair(.6, .9)]))

    def test_preview_never_touches_live_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'preview'
            with patch.object(probe, 'preflight', side_effect=AssertionError('live access')), \
                    patch.object(probe.Probe, 'x', side_effect=AssertionError('input')), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(['--dry-run', '--out', str(out)]), 0)
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['inputs'], [])
            self.assertEqual(report['samples'], [])
            self.assertFalse(report['calibration_valid'])

    def test_invalid_live_bounds_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'invalid'
            for flags in (['--seconds', '31'], ['--dx', '21'], ['--dx', '0'],
                          ['--seconds', 'nan'], ['--execute', '--dry-run']):
                with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    probe.main(['--out', str(out), *flags])
                self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
