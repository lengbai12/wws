"""Pixel identity behavior; controller/firing authority remains separate."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_zoom_identity import CrossZoomIdentity
from main_view_perception import detect_nameplates

ROOT = Path(__file__).resolve().parents[2]


def scene(labels):
    frame = np.zeros((800, 1200, 3), np.uint8)
    rows = []
    for x, y, text in labels:
        row = dict(nameplate_bbox=[x, y, 120, 16], trackable=True)
        rows.append(row)
        if text:
            cv2.putText(frame, text, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX,
                        .55, (245,245,245), 1, cv2.LINE_AA)
    return frame, rows


class HeaderIdentityContracts(unittest.TestCase):
    def test_offscreen_search_margin_preserves_visible_name_but_not_clipped_text(self):
        from cross_zoom_identity import header_signature
        full, rows = scene([(1030, 400, 'EnemyBravo')])
        # The 300px search extends past the right edge, but the text is complete.
        signature, evidence = header_signature(full, rows[0])
        self.assertIsNotNone(signature, evidence)
        clipped = full[:, :1110]
        signature, evidence = header_signature(clipped, rows[0])
        self.assertIsNone(signature)

    def test_touching_native_letters_preserve_identity_but_erased_name_does_not(self):
        producer = self.producer()
        expected = None
        for label in ('before','after'):
            frame = np.asarray(Image.open(Path(__file__).parent/'main_view_fixtures'/
                f'haus-joined-{label}-20261019.png').convert('RGB'))
            row = dict(nameplate_bbox=[1253,671,123,17])
            current, evidence = producer.identify(frame,[row],selected=row)
            self.assertEqual(evidence['status'],'confirmed')
            expected = expected or current[0]['target_id']
            self.assertEqual(current[0]['target_id'],expected)
        erased = frame.copy()
        erased[647:668,1290:1340] = 90
        _, evidence = self.producer().identify(erased,[row],selected=row)
        self.assertEqual(evidence['status'],'unconfirmed')

    def test_real_moving_lock_circle_is_registered_before_name_comparison(self):
        producer = self.producer()
        expected = None
        for label, box in (('before',[1334,667,124,17]),('after',[1265,663,123,17])):
            frame = np.asarray(Image.open(Path(__file__).parent/'main_view_fixtures'/
                f'haus-registered-{label}-20261019.png').convert('RGB'))
            row = dict(nameplate_bbox=box)
            current, evidence = producer.identify(frame,[row],selected=row)
            self.assertEqual(evidence['status'],'confirmed')
            expected = expected or current[0]['target_id']
            self.assertEqual(current[0]['target_id'],expected)

    def test_real_short_name_keeps_identity_when_lock_ring_covers_colons(self):
        frames = []
        for label in ('unlocked', 'locked'):
            crop = np.asarray(Image.open(Path(__file__).parent/'main_view_fixtures'/
                f'short-haus-{label}-20261019.png').convert('RGB'))
            frame = np.zeros((1600,2560,3),np.uint8)
            frame[500:645,700:1030] = crop
            frames.append((frame,dict(nameplate_bbox=[750,601,123,17])))
        producer = self.producer()
        before, evidence = producer.identify(frames[0][0], [frames[0][1]], selected=frames[0][1])
        self.assertEqual(evidence['status'], 'confirmed')
        current, evidence = producer.identify(frames[1][0], [frames[1][1]])
        self.assertEqual(evidence['status'], 'confirmed')
        self.assertEqual(current[0]['target_id'], before[0]['target_id'])
        frame, row = frames[1]
        frame[574:598,780:840] = frame[574:598,700:760]
        _, rejected = self.producer().identify(frame,[row],selected=row)
        self.assertEqual(rejected['status'], 'unconfirmed')

    def test_real_name_box_jitter_of_six_pixels_keeps_the_same_identity(self):
        # Measured live: the health-bar box grew 124 -> 138 px between frames, so
        # the bar-anchored name crop shifted by about seven pixels. The unchanged
        # appearance thresholds must still recognise the same name; a two-pixel
        # registration can never reach it.
        crop = np.asarray(Image.open(Path(__file__).parent/'main_view_fixtures'/
                                    'locked-name-haus-20261019.png').convert('RGB'))
        frame = np.zeros((1600,2560,3),np.uint8)
        frame[500:590,700:1013] = crop
        row = dict(nameplate_bbox=[795,540,123,17])
        producer = self.producer()
        matched, evidence = producer.identify(frame,[row],selected=row)
        self.assertEqual(evidence['status'],'confirmed')
        for shift in (6, -6):
            with self.subTest(shift=shift):
                jittered = dict(nameplate_bbox=[795+shift,540,123,17])
                current, evidence = producer.identify(frame,[jittered])
                self.assertEqual(evidence['status'],'confirmed')
                self.assertEqual(current[0]['target_id'],matched[0]['target_id'])


        crop = np.asarray(Image.open(Path(__file__).parent/'main_view_fixtures'/
                                    'locked-name-haus-20261019.png').convert('RGB'))
        frame = np.zeros((1600,2560,3),np.uint8)
        frame[500:590,700:1013] = crop
        row = dict(nameplate_bbox=[795,540,123,17])
        producer = self.producer()
        matched, evidence = producer.identify(frame,[row],selected=row)
        self.assertEqual(evidence['status'],'confirmed')
        shifted = np.zeros_like(frame)
        shifted[530:620,1000:1313] = crop
        current, evidence = producer.identify(shifted,[dict(nameplate_bbox=[1095,570,123,17])])
        self.assertEqual(evidence['status'],'confirmed')
        self.assertEqual(current[0]['target_id'],matched[0]['target_id'])
        frame[515:537,820:885] = 0
        _, rejected = self.producer().identify(frame,[row],selected=row)
        self.assertEqual(rejected['status'],'unconfirmed')

    def test_real_popov_ring_registration_preserves_name_not_circle(self):
        crop = np.asarray(Image.open(Path(__file__).parent / 'main_view_fixtures' /
                          'locked-name-popov-20261019.png').convert('RGB'))
        frame = np.zeros((1600,2560,3),np.uint8)
        frame[560:690,1445:1775] = crop
        row = dict(nameplate_bbox=[1531,646,128,17])
        producer = self.producer()
        matched, evidence = producer.identify(frame,[row],selected=row)
        self.assertEqual(evidence['status'],'confirmed')
        shifted = np.zeros_like(frame)
        shifted[590:720,1045:1375] = crop
        targets, next_evidence = producer.identify(
            shifted,[dict(nameplate_bbox=[1131,676,128,17])])
        self.assertEqual(next_evidence['status'],'confirmed')
        self.assertEqual(targets[0]['target_id'],matched[0]['target_id'])
        # Remove the text band: the remaining ring/class must never identify a ship.
        frame[624:642,1570:1620] = frame[624:642,1470:1520]
        _, rejected = self.producer().identify(frame,[row],selected=row)
        self.assertEqual(rejected['status'],'unconfirmed')

    def test_real_bright_fragmented_name_requires_original_visible_strokes(self):
        from cross_zoom_identity import header_signature
        crop = np.asarray(Image.open(Path(__file__).parent / 'main_view_fixtures' /
                          'locked-name-bright-fragments-20261019.png').convert('RGB'))
        frame = np.zeros((1600,2560,3),np.uint8)
        frame[560:690,1450:1780] = crop
        row = dict(nameplate_bbox=[1559,646,124,17])
        signature, evidence = header_signature(frame,row)
        self.assertIsNotNone(signature)
        self.assertGreaterEqual(evidence['stroke_pixels'],75)
        producer = self.producer()
        matched, initial = producer.identify(frame,[row],selected=row)
        self.assertEqual(initial['status'],'confirmed')
        shifted = np.zeros_like(frame)
        shifted[590:720,1050:1380] = crop
        target, subsequent = producer.identify(shifted,[dict(nameplate_bbox=[1159,676,124,17])])
        self.assertEqual(subsequent['status'],'confirmed')
        self.assertEqual(target[0]['target_id'],matched[0]['target_id'])
        frame[624:641,1565:1628] = frame[624:641,1480:1543]
        signature, _ = header_signature(frame,row)
        self.assertIsNone(signature)

    def test_real_lock_toggle_preserves_identity_in_each_direction(self):
        frames = []
        for label in ('locked_before', 'unlocked', 'locked_restored'):
            crop = np.asarray(Image.open(Path(__file__).parent / 'main_view_fixtures' /
                              f'lock-state-{label}-20261019.png').convert('RGB'))
            frame = np.zeros((1600, 2560, 3), np.uint8)
            frame[400:530, 700:924] = crop
            height = 17 if label == 'locked_restored' else 16
            frames.append((frame, dict(nameplate_bbox=[750, 510, 124, height])))
        for anchor in range(3):
            producer = self.producer()
            frame, row = frames[anchor]
            first, established = producer.identify(frame, [row], selected=row)
            self.assertEqual(established['status'], 'confirmed')
            for frame, row in frames:
                matched, evidence = producer.identify(frame, [row])
                self.assertEqual(evidence['status'], 'confirmed')
                self.assertEqual(matched[0]['target_id'], first[0]['target_id'])

    def test_similar_names_and_single_changed_letter_do_not_inherit_identity(self):
        original, rows = scene([(200, 300, 'Tributs_Alpha')])
        producer = self.producer()
        producer.identify(original, rows, selected=rows[0])
        for text in ('Tributs_Alphb', 'Tributs_Bravo', 'Other_Player'):
            with self.subTest(text=text):
                frame, current = scene([(200, 300, text)])
                matched, evidence = producer.identify(frame, current)
                self.assertEqual(evidence['status'], 'unconfirmed')
                self.assertNotIn('target_id', matched[0])

    def producer(self):
        producer = CrossZoomIdentity()
        producer.observe_source({'pid': 11, 'window': 'one'}, battle_active=True)
        return producer

    def test_native_text_crossing_distant_positions_preserves_original_identity(self):
        producer = self.producer()
        wide, rows = scene([(200, 300, 'XKnight_Alpha'), (650, 400, 'Other_Player')])
        anchored, evidence = producer.identify(wide, rows, selected=rows[0], frame_source='wide.png')
        identity = anchored[0]['target_id']
        scoped, current = scene([(700, 500, 'XKnight_Alpha'), (200, 300, 'Other_Player')])
        matched, evidence = producer.identify(scoped, current, frame_source='scope.png')
        self.assertEqual(matched[0]['target_id'], identity)
        self.assertNotIn('target_id', matched[1])
        self.assertEqual(evidence['status'], 'confirmed')
        self.assertGreaterEqual(evidence['score'], evidence['minimum_score'])
        self.assertGreaterEqual(evidence['margin'], evidence['minimum_margin'])
        self.assertEqual(evidence['anchor']['frame_source'], 'wide.png')
        self.assertFalse(evidence['replay_truth_used'])
        # Evidence crosses the JSON worker boundary, including native OpenCV boxes.
        import json
        decoded = json.loads(json.dumps({'rows': matched, 'evidence': evidence}))
        self.assertEqual(decoded['evidence']['target_id'], identity)

    def test_anchor_is_explicit_range_selection_not_candidate_order(self):
        producer = self.producer()
        frame, rows = scene([(200, 300, 'Near_Reticle'), (650, 400, 'Range_Selected')])
        matched, evidence = producer.identify(frame, rows, selected=rows[1])
        self.assertNotIn('target_id', matched[0])
        self.assertEqual(evidence['matched_index'], 1)
        self.assertIn('target_id', matched[1])

    def test_single_blank_or_low_support_cannot_become_an_identity(self):
        for label in ('', 'I', 'ab'):
            with self.subTest(label=label):
                producer = self.producer()
                frame, rows = scene([(200, 300, label)])
                rows[0]['target_id'] = 'caller-cannot-prove-identity'
                matched, evidence = producer.identify(frame, rows, selected=rows[0])
                self.assertEqual(evidence['status'], 'unconfirmed')
                self.assertNotIn('target_id', matched[0])
                self.assertIsNone(producer.anchor)

    def test_unrelated_single_candidate_cannot_inherit_anchor(self):
        producer = self.producer()
        frame, rows = scene([(200, 300, 'XKnight_Alpha')])
        producer.identify(frame, rows, selected=rows[0])

        original = producer.target_id
        for label in ('Completely_Else', '', 'X'):
            current, rows = scene([(200, 300, label)])
            matched, evidence = producer.identify(current, rows, selected=rows[0])
            self.assertEqual(evidence['status'], 'unconfirmed')
            self.assertNotIn('target_id', matched[0])
            self.assertEqual(producer.target_id, original)
    def test_unreadable_competitor_cannot_hide_duplicate_name(self):
        producer = self.producer()
        frame, rows = scene([(200, 300, 'XKnight_Alpha'), (650, 400, '')])
        matched, evidence = producer.identify(frame, rows, selected=rows[0])
        self.assertEqual(evidence['reason'], 'insufficient_competing_name_text')
        self.assertTrue(all('target_id' not in row for row in matched))

    def test_identical_labels_refuse_both_initial_anchor_and_scoped_match(self):
        frame, rows = scene([(200, 300, 'XKnight_Alpha'), (650, 400, 'XKnight_Alpha')])
        producer = self.producer()
        matched, evidence = producer.identify(frame, rows, selected=rows[0])
        self.assertEqual(evidence['reason'], 'appearance_ambiguous')
        self.assertIsNone(producer.anchor)
        single, first = scene([(200, 300, 'XKnight_Alpha')])
        producer.identify(single, first, selected=first[0])
        matched, evidence = producer.identify(frame, rows)
        self.assertEqual(evidence['reason'], 'appearance_ambiguous')
        self.assertTrue(all('target_id' not in row for row in matched))

    def test_battle_loss_and_source_change_clear_anchor(self):
        for source, active in [({'pid': 11, 'window': 'one'}, False),
                               ({'pid': 12, 'window': 'one'}, True), (None, True)]:
            with self.subTest(source=source, active=active):
                producer = self.producer()
                frame, rows = scene([(200, 300, 'XKnight_Alpha')])
                producer.identify(frame, rows, selected=rows[0])
                self.assertTrue(producer.observe_source(source, battle_active=active))
                self.assertIsNone(producer.target_id)
                self.assertIsNone(producer.anchor)

    def test_real_saved_header_pair_ignores_health_and_range(self):
        # These are genuine saved frames around the recorded scope request.
        # Current ruler detection does not confirm its resulting camera state;
        # this regression asserts real appearance only, never a zoom/fire claim.
        paths = [ROOT/'state/static-target-scope-live-20261017-j8'/name
                 for name in ('before.png', 'after.png')]
        if not all(path.is_file() for path in paths):
            self.skipTest('local genuine saved pair unavailable')
        producer = self.producer()
        identity = None
        for path in paths:
            frame = np.asarray(Image.open(path).convert('RGB'))
            rows = detect_nameplates(frame, scope_observed=True)
            self.assertEqual(len(rows), 1)
            matched, evidence = producer.identify(frame, rows, selected=rows[0], frame_source=str(path))
            self.assertEqual(evidence['status'], 'confirmed')
            identity = identity or matched[0]['target_id']
            self.assertEqual(matched[0]['target_id'], identity)

    def test_real_lock_ring_occlusion_keeps_text_identity_after_translation(self):
        crop = np.asarray(Image.open(Path(__file__).parent / 'main_view_fixtures' /
                                    'locked-name-tributs-20261019.png').convert('RGB'))
        frame = np.zeros((1600,2560,3),np.uint8)
        frame[400:400+crop.shape[0],700:700+crop.shape[1]] = crop
        row = dict(nameplate_bbox=[750,510,123,17])
        producer = self.producer()
        matched, evidence = producer.identify(frame,[row],selected=row)
        self.assertEqual(evidence['status'],'confirmed')
        self.assertEqual(evidence['anchor']['occlusion']['kind'],'confirmed_ui_lock_ring')
        shifted = np.zeros_like(frame)
        shifted[430:430+crop.shape[0],1100:1100+crop.shape[1]] = crop
        target, evidence = producer.identify(shifted,[dict(nameplate_bbox=[1150,540,123,17])])
        self.assertEqual(target[0]['target_id'],matched[0]['target_id'])
        # Removing all name strokes must not let the remaining circle identify a ship.
        frame[484:508,770:925] = 0
        _, rejected = self.producer().identify(frame,[row],selected=row)
        self.assertEqual(rejected['status'],'unconfirmed')


if __name__ == '__main__':
    unittest.main()
