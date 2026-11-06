import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_main_view_observation import continuity_report


class MainViewContinuityContract(unittest.TestCase):
    def row(self, seq, *, track_id=1, decision_delay=100_000_000, epoch=0):
        captured = seq*250_000_000
        return dict(video_time_s=394+seq*.25, requested_video_time_s=394+seq*.25,
            captured_ns=captured, decision_ns=captured+decision_delay,
            observation=dict(fresh=True, valid_until_ns=captured+750_000_000,
                epoch=epoch, seq=seq, view=dict(detections=[dict(track_id=track_id,
                    track_state='tracked' if seq else 'tentative')])))

    def test_same_id_with_fresh_dense_decisions(self):
        result = continuity_report([self.row(0), self.row(1), self.row(2)])
        self.assertEqual(3, result['longest_dense_same_id_run_frames'])
        self.assertTrue(result['entire_sequence_dense_and_fresh'])

    def test_same_id_does_not_hide_previous_lease_gap(self):
        result = continuity_report([self.row(0), self.row(1, decision_delay=500_000_000)])
        self.assertEqual(2, result['longest_same_id_run_frames'])
        self.assertEqual(1, result['longest_dense_same_id_run_frames'])
        self.assertEqual(0, result['freshness_gaps'][0]['uncovered_ns'])
        self.assertFalse(result['entire_sequence_dense_and_fresh'])

    def test_fast_video_pts_do_not_hide_slow_acquisition(self):
        rows = [self.row(0), self.row(1)]
        rows[1]['captured_ns'] = 600_000_000
        rows[1]['decision_ns'] = 700_000_000
        result = continuity_report(rows)
        self.assertEqual(.25, result['cadence'][0]['source_dt_s'])
        self.assertEqual(.6, result['cadence'][0]['acquisition_dt_s'])
        self.assertFalse(result['entire_sequence_dense_and_fresh'])

    def test_source_change_and_unrelated_ids_cannot_extend_run(self):
        for second in (self.row(1, track_id=2), self.row(1, epoch=1)):
            result = continuity_report([self.row(0), second])
            self.assertEqual(1, result['longest_same_id_run_frames'])
            self.assertEqual(1, result['longest_dense_same_id_run_frames'])

    def test_expired_view_does_not_extend_run(self):
        rows = [self.row(0), self.row(1), self.row(2)]
        rows[1]['observation'].update(fresh=False, view=None, valid_until_ns=None)
        result = continuity_report(rows)
        self.assertEqual(1, result['longest_same_id_run_frames'])
        self.assertFalse(result['entire_sequence_dense_and_fresh'])

    def test_cached_fresh_flag_cannot_hide_expired_decision(self):
        for offset in (0, -1):
            rows = [self.row(0), self.row(1)]
            rows[1]['observation']['valid_until_ns'] = rows[1]['decision_ns']+offset
            result = continuity_report(rows)
            self.assertEqual(1, result['fresh_frames'])
            self.assertFalse(result['entire_sequence_dense_and_fresh'])

    def test_decision_before_capture_is_not_fresh(self):
        rows = [self.row(0), self.row(1, decision_delay=-1)]
        result = continuity_report(rows)
        self.assertEqual(1, result['fresh_frames'])
        self.assertFalse(result['entire_sequence_dense_and_fresh'])


if __name__ == '__main__':
    unittest.main()
