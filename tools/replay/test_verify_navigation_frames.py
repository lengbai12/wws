from fractions import Fraction
from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_navigation_frames import NativeVideoSequence, continuity_report
from verify_navigation_frames import recording_identity
from build_navigation_reference import build_reference


class RecordingIdentityContract(unittest.TestCase):
    def recording(self, directory, *, session='236374', map_id='s10_USS_CL',
                  ship='PGSB210-Schlieffen'):
        root = Path(directory)
        replay = root / (session + '.wowsreplay')
        meta = json.dumps(dict(mapName='spaces/' + map_id, playerVehicle=ship)).encode()
        replay.write_bytes(b'\x12\x32\x34\x11' + struct.pack('<ii', 1, len(meta)) + meta)
        video = root / 'capture.mkv'
        video.write_bytes(b'identity fixture; never decoded')
        (root / 'replay_path.txt').write_text(str(replay.resolve()) + '\n')
        return video

    def test_matching_recording_has_replay_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.recording(directory)
            identity = recording_identity(video, build_reference())
            self.assertEqual(identity['session'], '236374')
            self.assertEqual(identity['map'], 's10_USS_CL')
            self.assertEqual(identity['ship'], 'PGSB210-Schlieffen')
            self.assertEqual(len(identity['replay_sha256']), 64)

    def test_other_session_map_or_ship_cannot_use_reference_context(self):
        for change in (dict(session='236387'), dict(map_id='s06_Atoll'),
                       dict(ship='PASC510-Des_Moines')):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                video = self.recording(directory, **change)
                with self.assertRaisesRegex(ValueError, 'recording identity'):
                    recording_identity(video, build_reference())

    def test_unbound_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            video = self.recording(directory)
            (Path(directory) / 'replay_path.txt').unlink()
            with self.assertRaisesRegex(ValueError, 'replay binding'):
                recording_identity(video, build_reference())



class ForwardDecodeContract(unittest.TestCase):
    def video(self, directory, width=2560, height=1600):
        path = Path(directory) / 'frames.mkv'
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('ffv1', rate=4)
            stream.width, stream.height = width, height
            stream.pix_fmt = 'yuv420p'
            stream.codec_context.thread_count = 2
            for i in range(4):
                frame = av.VideoFrame.from_ndarray(
                    np.full((height, width, 3), 40 + i * 40, np.uint8), format='rgb24')
                frame.pts, frame.time_base = i, Fraction(1, 4)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    def test_forward_selection_keeps_actual_time_and_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = NativeVideoSequence(self.video(directory))
            try:
                first, pts, _ = source.read_at(.1)
                self.assertEqual(.25, pts)
                second, pts, skipped = source.read_at(.7)
                self.assertEqual(.75, pts)
                self.assertEqual(1, skipped)
                self.assertEqual((1600, 2560, 3), second.shape)
                self.assertGreater(float(second.mean()), float(first.mean()) + 60)
            finally:
                source.close()

    def test_end_before_target_fails_instead_of_repeating_last_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            source = NativeVideoSequence(self.video(directory))
            try:
                source.read_at(.5)
                with self.assertRaises(EOFError):
                    source.read_at(1.)
            finally:
                source.close()

    def test_wrong_native_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = NativeVideoSequence(self.video(directory, 64, 48))
            try:
                with self.assertRaisesRegex(ValueError, 'native'):
                    source.read_at(0)
            finally:
                source.close()


class ContinuityEvidenceContract(unittest.TestCase):
    def row(self, seq, *, usable=True, track_id=1, decision_ns=None,
            epoch=0, observation_seq=None):
        captured = seq * 250_000_000
        observation_seq = seq if observation_seq is None else observation_seq
        return dict(seq=seq, video_time_s=500 + seq * .25, captured_ns=captured,
            decision_ns=captured + 100_000_000 if decision_ns is None else decision_ns,
            observation=dict(epoch=epoch, seq=observation_seq, fresh=True,
                valid_until_ns=captured + 750_000_000,
                enemies=[dict(track_id=track_id, state='confirmed' if usable else 'tentative',
                              observed=True, usable=usable)]))

    def test_fresh_frames_and_unrelated_usable_ids_do_not_prove_continuity(self):
        for rows in ([self.row(0, usable=False), self.row(1, usable=False)],
                     [self.row(0), self.row(1, track_id=2)]):
            report = continuity_report(rows)
            self.assertEqual(2, report['fresh_frames'])
            self.assertFalse(report['dense_track_continuity_demonstrated'])

    def test_same_id_requires_dense_decisions_and_unexpired_previous_lease(self):
        report = continuity_report([self.row(0), self.row(1)])
        self.assertTrue(report['dense_track_continuity_demonstrated'])
        self.assertEqual(2, report['longest_dense_usable_id_run_frames'])
        report = continuity_report([self.row(0), self.row(1, decision_ns=750_000_000)])
        self.assertFalse(report['dense_track_continuity_demonstrated'])
        self.assertEqual(1, len(report['cadence_failures']))
        self.assertEqual(0, report['freshness_gaps'][0]['uncovered_ns'])

    def test_enemy_run_rejects_epoch_or_sequence_reuse(self):
        for mutation in ('epoch', 'sequence'):
            with self.subTest(mutation=mutation):
                rows = [self.row(0), self.row(1)]
                if mutation == 'epoch':
                    rows[1]['observation']['epoch'] = 1
                else:
                    rows[1]['observation']['seq'] = 0
                report = continuity_report(rows)
                self.assertEqual(1, report['longest_dense_usable_id_run_frames'])
                self.assertFalse(report['dense_track_continuity_demonstrated'])
                self.assertEqual(1, len(report['identity_failures']))
                self.assertFalse(report['entire_sequence_dense_and_fresh'])

    def test_enemy_run_rejects_capture_clock_reversal(self):
        rows = [self.row(0), self.row(1)]
        rows[1]['captured_ns'] = rows[0]['captured_ns'] - 1
        report = continuity_report(rows)
        self.assertEqual(1, report['longest_dense_usable_id_run_frames'])
        self.assertFalse(report['dense_track_continuity_demonstrated'])
        self.assertEqual(1, len(report['cadence_failures']))

    def test_cached_fresh_flag_cannot_hide_invalid_decision_clock(self):
        for invalid in ('at_expiry', 'before_capture'):
            with self.subTest(invalid=invalid):
                rows = [self.row(0), self.row(1)]
                rows[1]['decision_ns'] = (rows[1]['observation']['valid_until_ns']
                    if invalid == 'at_expiry' else rows[1]['captured_ns']-1)
                report = continuity_report(rows)
                self.assertEqual(1, report['fresh_frames'])
                self.assertEqual(1, report['usable_frames'])
                self.assertEqual(1, report['confirmed_frames'])
                self.assertEqual(1, report['longest_dense_usable_id_run_frames'])
                self.assertFalse(report['entire_sequence_dense_and_fresh'])

    def navigation_rows(self):
        rows = [self.row(seq) for seq in range(3)]
        for row in rows:
            row['observation'].update(epoch=0, seq=row['seq'], own=dict(pixel_x=20, pixel_y=30))
            row['intent'] = dict(action='autopilot_waypoint', movement_authorized=True,
                waypoint=dict(x=30, y=40), valid_until_ns=row['decision_ns']+400_000_000,
                reference=dict(map='s10_USS_CL', ship='PGSB210-Schlieffen',
                    pattern='reference_236374', reference_sha256='a'*64,
                    projection_sha256='c'*64),
                progress=dict(s_px=float(row['seq']), fraction=row['seq']/100,
                    position_projected=True), input_authorized=False)
        return rows

    def test_waypoint_continuity_does_not_require_enemy_detections(self):
        rows = self.navigation_rows()
        for row in rows:
            row['observation']['enemies'] = []
        report = continuity_report(rows)
        self.assertFalse(report['dense_track_continuity_demonstrated'])
        self.assertEqual(3, report['navigation']['longest_dense_waypoint_run_frames'])
        self.assertTrue(report['navigation']['entire_sequence_dense_with_valid_waypoints'])
        self.assertFalse(report['navigation']['input_execution_verified'])

    def test_expired_intent_breaks_continuity_even_with_fresh_observation(self):
        rows = self.navigation_rows()
        rows[0]['intent']['valid_until_ns'] = rows[1]['decision_ns']
        report = continuity_report(rows)
        self.assertEqual([], report['freshness_gaps'])
        self.assertEqual(0, report['navigation']['intent_expiry_gaps'][0]['uncovered_ns'])
        self.assertEqual(2, report['navigation']['longest_dense_waypoint_run_frames'])
        self.assertFalse(report['navigation']['entire_sequence_dense_with_valid_waypoints'])

    def test_revoke_missing_own_and_invalid_lease_cannot_count_as_waypoints(self):
        for mutation in ('revoke', 'missing_own', 'expired', 'exceeds_observation'):
            with self.subTest(mutation=mutation):
                rows = self.navigation_rows()
                middle = rows[1]
                if mutation == 'revoke':
                    middle['intent'].update(action='revoke', movement_authorized=False, waypoint=None)
                elif mutation == 'missing_own':
                    middle['observation']['own'] = None
                elif mutation == 'expired':
                    middle['intent']['valid_until_ns'] = middle['decision_ns']
                else:
                    middle['intent']['valid_until_ns'] = middle['observation']['valid_until_ns']+1
                report = continuity_report(rows)['navigation']
                self.assertEqual(2, report['valid_waypoint_frames'])
                self.assertEqual(1, report['longest_dense_waypoint_run_frames'])
                self.assertFalse(report['dense_waypoint_continuity_demonstrated'])

    def test_source_or_reference_change_cannot_join_waypoint_runs(self):
        for mutation in ('epoch', 'reference', 'sequence'):
            with self.subTest(mutation=mutation):
                rows = self.navigation_rows()[:2]
                if mutation == 'epoch':
                    rows[1]['observation']['epoch'] = 1
                elif mutation == 'reference':
                    rows[1]['intent']['reference']['reference_sha256'] = 'b'*64
                else:
                    rows[1]['observation']['seq'] = 0
                report = continuity_report(rows)['navigation']
                self.assertEqual(1, report['longest_dense_waypoint_run_frames'])
                self.assertFalse(report['entire_sequence_dense_with_valid_waypoints'])

    def test_missing_reference_or_unprojected_progress_cannot_prove_navigation(self):
        for mutation in ('missing_reference', 'empty_reference', 'projection_hash',
                         'missing_progress', 'unprojected', 'nonfinite'):
            with self.subTest(mutation=mutation):
                rows = self.navigation_rows()[:2]
                for row in rows:
                    intent = row['intent']
                    if mutation == 'missing_reference':
                        intent.pop('reference')
                    elif mutation == 'empty_reference':
                        intent['reference'] = {}
                    elif mutation == 'projection_hash':
                        intent['reference'].pop('projection_sha256')
                    elif mutation == 'missing_progress':
                        intent.pop('progress')
                    elif mutation == 'unprojected':
                        intent['progress']['position_projected'] = False
                    else:
                        intent['progress']['s_px'] = float('nan')
                report = continuity_report(rows)['navigation']
                self.assertEqual(0, report['valid_waypoint_frames'])
                self.assertFalse(report['dense_waypoint_continuity_demonstrated'])

    def test_progress_reversal_breaks_same_reference_waypoint_run(self):
        for field in ('s_px', 'fraction'):
            with self.subTest(field=field):
                rows = self.navigation_rows()[1:]
                rows[1]['intent']['progress'][field] = 0.0
                report = continuity_report(rows)['navigation']
                self.assertEqual(2, report['valid_waypoint_frames'])
                self.assertEqual(1, report['longest_dense_waypoint_run_frames'])
                self.assertFalse(report['entire_sequence_dense_with_valid_waypoints'])

    def test_malformed_geometry_and_hashes_cannot_count_as_navigation(self):
        for mutation in ('empty_waypoint', 'nan_waypoint', 'outside_waypoint',
                         'empty_own', 'nan_own', 'outside_own', 'short_hash', 'nonhex_hash'):
            with self.subTest(mutation=mutation):
                rows = self.navigation_rows()[:2]
                for row in rows:
                    intent, own = row['intent'], row['observation']['own']
                    if mutation == 'empty_waypoint':
                        intent['waypoint'] = {}
                    elif mutation == 'nan_waypoint':
                        intent['waypoint']['x'] = float('nan')
                    elif mutation == 'outside_waypoint':
                        intent['waypoint']['y'] = 723
                    elif mutation == 'empty_own':
                        row['observation']['own'] = {}
                    elif mutation == 'nan_own':
                        own['pixel_x'] = float('nan')
                    elif mutation == 'outside_own':
                        own['pixel_x'] = -1
                    elif mutation == 'short_hash':
                        intent['reference']['reference_sha256'] = 'a'
                    else:
                        intent['reference']['projection_sha256'] = 'z'*64
                report = continuity_report(rows)['navigation']
                self.assertEqual(0, report['valid_waypoint_frames'])
                self.assertFalse(report['dense_waypoint_continuity_demonstrated'])


if __name__ == '__main__':
    unittest.main()
