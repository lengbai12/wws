"""Offline regressions for saved-probe analysis; never sends game input."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import analyze_aim_probe as analysis


class SavedProbeAnalysisTests(unittest.TestCase):
    def test_capture_png_alpha_preserves_color_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.full((1600, 2560, 4), (30, 60, 90, 255), np.uint8)
            image[:80, :, 3] = 0
            path = root / 'capture.png'
            self.assertTrue(cv2.imwrite(str(path), image))
            snapshots = [dict(name=name, path=str(path), status='completed',
                              node=9, process_id=77, bytes=path.stat().st_size,
                              request_elapsed_s=i * 2, completed_elapsed_s=i * 2 + 1)
                         for i, name in enumerate(('baseline', 'before', 'after', 'rollback'))]
            report = dict(status='completed', execute=True, identity_verified=True,
                          identity=dict(pid=12, display=':2', start_ticks='123'),
                          source=dict(id=9, process_id=77, serial=99, name='gamescope'),
                          focus_checks=[dict(pid=12, window='42')], snapshots=snapshots,
                          rollback_input_sent=True, unreleased_own_keys=[],
                          limits=dict(no_mouse_buttons=True, seconds=10), elapsed_s=8,
                          inputs=[dict(relative=[80, 0], elapsed_s=3.5),
                                  dict(relative=[-80, 0], elapsed_s=5.5)])
            report_path = root / 'report.json'
            report_path.write_text(json.dumps(report))
            alpha = analysis.analyze_report(report_path)
            self.assertTrue(alpha['source_valid'], alpha['rejections'])
            self.assertTrue(all(row['valid'] for row in alpha['images'].values()))
            self.assertFalse(alpha['local_response_valid'])
            self.assertTrue(cv2.imwrite(str(path), image[:, :, :3]))
            for snapshot in report['snapshots']:
                snapshot['bytes'] = path.stat().st_size
            report_path.write_text(json.dumps(report))
            color = analysis.analyze_report(report_path)
            self.assertTrue(color['source_valid'], color['rejections'])
            self.assertEqual(alpha['pairs'], color['pairs'])

    def test_stationary_hud_cannot_dominate_changed_scene_flow(self):
        before = np.zeros((1600, 2560, 3), np.uint8)
        rng = np.random.default_rng(231)
        for x, y in rng.integers([420, 920], [2000, 1190], size=(160, 2)):
            cv2.rectangle(before, (int(x), int(y)), (int(x+7), int(y+7)), (180, 180, 180), -1)
        after = cv2.warpAffine(before, np.float32([[1, 0, 12], [0, 1, -6]]), (2560, 1600))
        # Dense bright static content in the excluded reticle/nameplate belt,
        # plus stationary HUD-like content outside that belt.
        for x, y in rng.integers([420, 650], [2000, 810], size=(600, 2)):
            for image in (before, after):
                cv2.rectangle(image, (int(x), int(y)), (int(x+7), int(y+7)), (255, 255, 255), -1)
        for x, y in rng.integers([420, 340], [2000, 440], size=(600, 2)):
            for image in (before, after):
                cv2.rectangle(image, (int(x), int(y)), (int(x+7), int(y+7)), (255, 255, 255), -1)
        flow = analysis.scene_motion(before, after)
        self.assertTrue(flow['sufficient'])
        self.assertAlmostEqual(flow['median_scene_dx_px'], 12, delta=1)
        self.assertAlmostEqual(flow['median_scene_dy_px'], -6, delta=1)
        self.assertFalse(flow['world_calibration_valid'])
        self.assertTrue(all(not 650 <= track['from_px'][1] < 830 for track in flow['track_evidence']))

    def test_fixed_scene_homography_measures_static_background(self):
        before = np.zeros((1600, 2560, 3), np.uint8)
        rng = np.random.default_rng(22)
        for x, y in rng.integers([420, 340], [2100, 1190], size=(900, 2)):
            cv2.rectangle(before, (int(x), int(y)), (int(x + 9), int(y + 9)),
                          (150, 170, 190), -1)
        after = cv2.warpAffine(before, np.float32([[1, 0, 12], [0, 1, -6]]),
                               (2560, 1600))
        # Bright stationary HUD-like content must not establish the transform.
        cv2.rectangle(before, (20, 20), (350, 300), (255, 255, 255), -1)
        cv2.rectangle(after, (20, 20), (350, 300), (255, 255, 255), -1)
        flow = analysis.fixed_scene_motion(before, after)
        self.assertTrue(flow['valid'])
        self.assertTrue(flow['sufficient'], flow)
        self.assertAlmostEqual(flow['median_screen_dx_px'], 12, delta=1)
        self.assertAlmostEqual(flow['median_screen_dy_px'], -6, delta=1)
        self.assertFalse(flow['world_calibration_valid'])

    def test_fixed_scene_rejects_wrong_native_images(self):
        image = np.zeros((1600, 2560, 3), np.uint8)
        result = analysis.fixed_scene_motion(image.astype(float), image)
        self.assertFalse(result['valid'])
        self.assertIn('native uint8', result['reason'])

    def test_stationary_and_wrong_native_images_are_explicit(self):
        image = np.zeros((1600, 2560, 3), np.uint8)
        stationary = analysis.scene_motion(image, image)
        self.assertTrue(stationary['valid'])
        self.assertFalse(stationary['sufficient'])
        self.assertEqual(stationary['mean_abs_rgb'], 0)
        self.assertFalse(analysis.scene_motion(image.astype(float), image.astype(float))['valid'])

    def test_reusing_report_does_not_establish_repeatability(self):
        result = {'report': {'sha256': 'same'}, 'identity': {'pid': 1},
                  'capture_source': {'id': 2}, 'images': {}, 'checks': {},
                  'pairs': {}, 'local_response_valid': False}
        aggregate = analysis.aggregate([result, result])
        self.assertEqual(aggregate['status'], 'invalid')
        self.assertIsNone(aggregate['local_scene_px_per_input_unit'])
        self.assertTrue(any('distinct source reports' in reason for reason in aggregate['missing_conditions']))

    def test_invalid_source_cannot_supply_even_a_diagnostic_sample(self):
        result = {'report': {'path': 'rejected', 'sha256': 'unique'},
                  'identity': {'pid': 1}, 'capture_source': {'id': 2},
                  'images': {}, 'source_valid': False, 'input_relative': [8, 0],
                  'local_response_valid': False, 'rejections': ['capture source mismatch'],
                  'checks': {'lower_scene': {'reversible_scene_change_valid': True}},
                  'pairs': {'before->after': {
                      'scene': {'fixed_scene': {'sufficient': True,
                                               'median_screen_dx_px': 12,
                                               'median_screen_dy_px': 0}},
                      'lower_scene': {'sufficient': True, 'median_scene_dx_px': 12,
                                      'median_scene_dy_px': 0}}}}
        self.assertEqual(analysis.aggregate([result])['accepted_samples'], 0)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'analysis.json'
            with patch.object(analysis, 'analyze_report', return_value=result), patch('builtins.print'):
                self.assertEqual(analysis.main(['--report', 'rejected', '--out', str(out)]), 1)
            evidence = json.loads(out.read_text())
        self.assertEqual(evidence['fixed_scene_response']['accepted_samples'], 0)
        self.assertEqual(evidence['fixed_scene_response']['status'], 'invalid')


class ReticleLocalDiagnosticTests(unittest.TestCase):
    @staticmethod
    def textured_frame():
        rng = np.random.default_rng(83)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (1600, 2560), dtype=np.uint8),
                                (0, 0), 1)
        return np.repeat(gray[:, :, None], 3, axis=2)

    @staticmethod
    def moved(image, dx, dy):
        return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]), (2560, 1600))

    def test_translation_reverse_and_zero_net_with_stationary_hud(self):
        before = self.textured_frame()
        after = self.moved(before, 24, -12)
        # Screen-fixed HUD must not supply a zero-motion vote.
        rng = np.random.default_rng(18)
        belt = rng.integers(0, 256, (180, 560, 3), dtype=np.uint8)
        for image in (before, after):
            image[650:830, 1000:1560] = belt
        forward = analysis.reticle_local_motion(before, after)
        inverse = analysis.reticle_local_motion(after, before)
        zero = analysis.reticle_local_motion(before, before.copy())
        for value in (forward, inverse, zero):
            self.assertTrue(value['sufficient'], value)
            self.assertFalse(value['calibration_valid'])
            self.assertEqual(value['anchor_xy_px'], [1280.5, 805.0])
            self.assertEqual(value['roi_xyxy'], [1000, 560, 1560, 1050])
            self.assertEqual(value['ecc']['motion_model'], 'translation_only')
        self.assertAlmostEqual(forward['screen_dx_px'], 24, delta=.5)
        self.assertAlmostEqual(forward['screen_dy_px'], -12, delta=.5)
        self.assertAlmostEqual(inverse['screen_dx_px'], -24, delta=.5)
        self.assertAlmostEqual(inverse['screen_dy_px'], 12, delta=.5)
        pairs = {'baseline->before': {'reticle_local': zero},
                 'before->after': {'reticle_local': forward},
                 'after->rollback': {'reticle_local': inverse},
                 'before->rollback': {'reticle_local': zero}}
        checks = analysis.reticle_local_consistency(pairs, True)
        self.assertTrue(checks['accepted'], checks)
        self.assertLess(checks['forward_inverse_error_px'], 1)
        self.assertLess(checks['zero_net_error_px'], .1)
        # The same forward displacement cannot stand in for an inverse.
        pairs['after->rollback']['reticle_local'] = forward
        checks = analysis.reticle_local_consistency(pairs, True)
        self.assertFalse(checks['accepted'])
        self.assertIn('forward_inverse_disagreement', checks['failed_conditions'])
        pairs['after->rollback']['reticle_local'] = inverse
        pairs['before->rollback']['reticle_local'] = forward
        checks = analysis.reticle_local_consistency(pairs, True)
        self.assertFalse(checks['accepted'])
        self.assertIn('zero_net_not_restored', checks['failed_conditions'])
        pairs['before->rollback']['reticle_local'] = zero
        self.assertFalse(analysis.reticle_local_consistency(pairs, False)['accepted'])
        pairs['before->after']['reticle_local'] = zero
        pairs['after->rollback']['reticle_local'] = zero
        checks = analysis.reticle_local_consistency(pairs, True)
        self.assertIn('no_resolved_input_response', checks['failed_conditions'])

    def test_blank_and_wrong_native_images_rejected(self):
        image = np.zeros((1600, 2560, 3), np.uint8)
        result = analysis.reticle_local_motion(image, image)
        self.assertTrue(result['valid'])
        self.assertFalse(result['sufficient'])
        self.assertIn('texture', result['reason'])
        self.assertFalse(analysis.reticle_local_motion(image.astype(float), image)['valid'])

    def test_periodic_two_dimensional_texture_is_ambiguous(self):
        yy, xx = np.indices((1600, 2560))
        gray = np.uint8(128+50*np.sin(xx*2*np.pi/16)+50*np.sin(yy*2*np.pi/16))
        before = np.repeat(gray[:, :, None], 3, axis=2)
        after = self.moved(before, 24, -8)
        value = analysis.reticle_local_motion(before, after)
        self.assertFalse(value['sufficient'], value)
        self.assertFalse(value['calibration_valid'])

    def test_different_textured_scenes_cannot_establish_translation(self):
        before = self.textured_frame()
        after = np.flip(before, axis=1).copy()
        value = analysis.reticle_local_motion(before, after)
        self.assertFalse(value['sufficient'], value)
        self.assertFalse(value['calibration_valid'])

    def test_full_rank_diagnostics_never_become_fire_calibration(self):
        results = []
        for index, command in enumerate(([20, 0], [-20, 0], [0, 20], [0, -20])*2):
            dx, dy = command
            results.append({'report': {'path': str(index), 'sha256': str(index)},
                            'identity': {'pid': 1}, 'capture_source': {'id': 2},
                            'images': {name: {'file': {'sha256': f'{index}-{name}'}}
                                       for name in ('baseline', 'before', 'after', 'rollback')},
                            'source_valid': True, 'input_relative': command,
                            'local_response_valid': True, 'rejections': [],
                            'reticle_local_checks': {'accepted': True},
                            'checks': {'lower_scene': {'reversible_scene_change_valid': True}},
                            'pairs': {'before->after': {
                                'reticle_local': {'screen_dx_px': dx, 'screen_dy_px': dy},
                                'lower_scene': {'sufficient': True, 'median_scene_dx_px': dx,
                                                'median_scene_dy_px': dy}}}})
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'analysis.json'
            argv = [item for index in range(8) for item in ('--report', str(index))]
            with patch.object(analysis, 'analyze_report', side_effect=results), patch('builtins.print'):
                self.assertEqual(analysis.main(argv+['--out', str(out)]), 0)
            evidence = json.loads(out.read_text())
        self.assertEqual(evidence['reticle_local_response']['accepted_samples'], 8)
        self.assertEqual(evidence['local_response']['input_rank'], 2)
        self.assertEqual(evidence['local_response']['status'], 'diagnostic_only')
        self.assertIsNone(evidence['local_response']['local_scene_px_per_input_unit'])
        self.assertTrue(any('camera mode and zoom' in reason
                            for reason in evidence['local_response']['missing_conditions']))
        self.assertFalse(evidence['calibration_valid'])
        self.assertFalse(evidence['world_calibration_valid'])
        self.assertFalse(evidence['reticle_local_response']['calibration_valid'])
        self.assertIsNone(evidence['reticle_local_response']['screen_px_per_mouse_unit'])
        self.assertEqual(evidence['mouse_response']['status'], 'invalid')
        self.assertIsNone(evidence['mouse_response']['screen_px_per_mouse_unit'])
        self.assertTrue(any('settling' in reason for reason in evidence['mouse_response']['missing_conditions']))


if __name__ == '__main__':
    unittest.main()
