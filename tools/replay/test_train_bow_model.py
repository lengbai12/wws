"""Small synthetic contract regressions; these fixtures are not training labels."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

import train_bow_model as T


class TrainerTests(unittest.TestCase):
    def batch(self, positive=True, known=True, size=16):
        mask = torch.zeros(1, 1, size, size)
        mask[:, :, 3:10, 3:10] = float(positive)
        bow = torch.from_numpy(T.heatmap([5, 4], size=size, sigma=1))[None, None]
        stern = torch.from_numpy(T.heatmap([5, 9], size=size, sigma=1))[None, None]
        if not positive:
            bow.zero_()
            stern.zero_()
        return (torch.zeros(1, 3, size, size), mask, bow, stern,
                torch.tensor([3 if known and positive else -1]), torch.tensor([positive]),
                torch.tensor([[[5., 4.], [5., 9.]]]))

    def outputs(self, size=16):
        return tuple(torch.zeros(shape, requires_grad=True) for shape in
                     ((1, 1, size, size), (1, 1, size, size),
                      (1, 1, size, size), (1, T.ORIENTATION_BINS)))

    def test_spatial_targets_normalized_at_boundaries(self):
        for point in ([0, 0], [255, 255], [120.25, 99.75]):
            target = T.heatmap(point)
            self.assertAlmostEqual(float(target.sum()), 1., places=6)
            self.assertTrue(np.isfinite(target).all())

    def test_keypoint_loss_rejects_flat_collapse_and_moves_peak(self):
        target = torch.from_numpy(T.heatmap([5, 4], size=16, sigma=1))[None, None]
        flat = torch.zeros_like(target, requires_grad=True)
        flat_loss = T.spatial_cross_entropy(flat, target).mean()
        flat_loss.backward()
        self.assertLess(float(flat.grad[0, 0, 4, 5]), 0.)
        self.assertGreater(float(flat.grad[0, 0, 15, 15]), 0.)
        # A uniform negative logit offset cannot pretend to improve keypoints.
        collapsed = torch.full_like(target, -30.)
        self.assertAlmostEqual(float(T.spatial_cross_entropy(collapsed, target)), float(flat_loss), places=5)
        good = target.clamp_min(1e-30).log()
        wrong = torch.roll(good, shifts=7, dims=-1)
        self.assertLess(float(T.spatial_cross_entropy(good, target)), float(flat_loss))
        self.assertGreater(float(T.spatial_cross_entropy(wrong, target)), float(flat_loss))

    def test_negative_and_unknown_labels_do_not_invent_supervision(self):
        outputs = self.outputs()
        terms = T.loss_terms(outputs, self.batch(positive=False))
        sum(terms.values()).backward()
        self.assertGreater(float(outputs[0].grad.abs().sum()), 0.)
        for logits in outputs[1:]:
            self.assertEqual(float(logits.grad.abs().sum()), 0.)
        outputs = self.outputs()
        terms = T.loss_terms(outputs, self.batch(known=False))
        sum(terms.values()).backward()
        self.assertGreater(float(outputs[1].grad.abs().sum()), 0.)
        self.assertGreater(float(outputs[2].grad.abs().sum()), 0.)
        self.assertEqual(float(outputs[3].grad.abs().sum()), 0.)

    def test_argmax_decode_and_metrics_use_pixel_distance_and_iou(self):
        batch = self.batch()
        pm = torch.where(batch[1] > 0, 10., -10.)
        pb = torch.full_like(pm, -10.)
        ps = torch.full_like(pm, -10.)
        pb[0, 0, 4, 8] = 10.  # exactly 3px from true bow
        ps[0, 0, 9, 5] = 10.
        po = torch.zeros(1, T.ORIENTATION_BINS)
        po[0, 3] = 10.
        outputs = (pm, pb, ps, po)
        self.assertEqual(T.decode_keypoints(pb).tolist(), [[8., 4.]])
        metrics = T.Metrics()
        metrics.update(outputs, batch, T.loss_terms(outputs, batch))
        got = metrics.result()
        self.assertEqual(got["bow_error_px"], 3.)
        self.assertEqual(got["stern_error_px"], 0.)
        self.assertEqual(got["mean_keypoint_error_px"], 1.5)
        self.assertEqual(got["mask_iou"], 1.)
        self.assertEqual(got["orientation_accuracy"], 1.)
        self.assertEqual(got["orientation_counts"][3], 1)
        self.assertEqual(got["orientation_correct"][3], 1)
        # All background is not a good mask, despite majority background pixels.
        metrics = T.Metrics()
        outputs = (torch.full_like(pm, -10.), pb, ps, po)
        metrics.update(outputs, batch, T.loss_terms(outputs, batch))
        self.assertEqual(metrics.result()["positive_mask_iou"], 0.)

    def test_sparse_hull_cannot_reduce_loss_by_predicting_only_sea(self):
        target = torch.zeros(1, 1, 256, 256)
        target[:, :, 30:35, 40:80] = 1
        collapsed = torch.full_like(target, -6.)
        correct = torch.where(target > .5, 6., -6.)
        self.assertGreater(float(T.balanced_mask_loss(collapsed, target)), 2.)
        self.assertLess(float(T.balanced_mask_loss(correct, target)), .01)

    def test_empty_mask_supervises_background_without_nan(self):
        logits = torch.zeros(1, 1, 16, 16, requires_grad=True)
        loss = T.balanced_mask_loss(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(bool((logits.grad > 0).all()))

    def test_augmentation_moves_endpoints_and_mask_together(self):
        image = np.zeros((256, 256, 3), np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        mask[80:121, 50:201] = 255
        image[mask > 0] = 255
        rgb, transformed, points = T.augment_sample(
            image, mask, [[55, 100], [195, 100]], np.random.default_rng(4))
        self.assertFalse(np.array_equal(mask, transformed))
        for x, y in points:
            self.assertGreater(transformed[round(y), round(x)], 0)
            self.assertGreater(rgb[round(y), round(x)].min(), 200)
        self.assertLess(points[0][0], points[1][0])

    def test_augmentation_preserves_negative_without_inventing_endpoints(self):
        image = np.zeros((256, 256, 3), np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        _, transformed, points = T.augment_sample(image, mask, [], np.random.default_rng(4))
        self.assertEqual(points, [])
        self.assertFalse(transformed.any())

    def test_unknown_orientation_not_counted_as_correct_or_as_bin_zero(self):
        batch = self.batch(known=False)
        outputs = self.outputs()
        metrics = T.Metrics()
        metrics.update(outputs, batch, T.loss_terms(outputs, batch))
        got = metrics.result()
        self.assertIsNone(got["orientation_accuracy"])
        self.assertEqual(got["orientation_counts"], [0] * 16)

    def test_three_way_session_split_deterministic_and_disjoint(self):
        rows = [{"session": f"session-{i // 3}", "image": str(i)} for i in range(30)]
        a = T.split_sessions(rows, 7)
        b = T.split_sessions(list(reversed(rows)), 7)
        session_sets = {name: {r["session"] for r in part} for name, part in a.items()}
        self.assertEqual(session_sets, {name: {r["session"] for r in part} for name, part in b.items()})
        self.assertTrue(all(session_sets.values()))
        self.assertFalse(session_sets["train"] & session_sets["validation"])
        self.assertFalse(session_sets["train"] & session_sets["test"])
        self.assertFalse(session_sets["test"] & session_sets["validation"])
        self.assertEqual(sum(map(len, a.values())), len(rows))
        for count in (1, 2):
            with self.assertRaisesRegex(ValueError, "three sessions"):
                T.split_sessions([{"session": str(i)} for i in range(count)], 7)

    def inventory_row(self, root, name, value, session, source_sha=None):
        image = np.full((256, 256, 3), value, np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        mask[10:20, 10:20] = 255
        image_name, mask_name = f"{name}.png", f"{name}.mask.png"
        cv2.imwrite(str(root / image_name), image)
        cv2.imwrite(str(root / mask_name), mask)
        return {"image": image_name, "mask": mask_name, "source_image": image_name,
                "image_sha256": T.pixel_sha256(image),
                "source_sha256": source_sha or T.pixel_sha256(image), "session": session,
                "annotation_source": "manual", "reviewer": "fixture-reviewer",
                "label_status": "positive", "bow_point": [14, 11], "stern_point": [14, 18],
                "orientation_bin": "unknown"}

    def test_pixel_leakage_detected_even_when_png_bytes_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.inventory_row(root, "a", 42, "one")
            b = self.inventory_row(root, "b", 42, "two")
            # Different compression encodings do not conceal identical pixels.
            image = cv2.imread(str(root / b["image"]))
            cv2.imwrite(str(root / b["image"]), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            with self.assertRaisesRegex(ValueError, "decoded pixel leakage"):
                T.prepare_inventory({"train": [a], "validation": [b], "test": []}, root)

    def test_distinct_crops_from_same_source_cannot_cross_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = self.inventory_row(root, "a", 42, "one")
            b = self.inventory_row(root, "b", 43, "two", source_sha=a["source_sha256"])
            with self.assertRaisesRegex(ValueError, "source image leakage"):
                T.prepare_inventory({"train": [a], "validation": [b], "test": []}, root)

    def test_pixel_inventory_and_changed_crop_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.inventory_row(root, "a", 42, "one")
            inventory = T.prepare_inventory({"train": [row]}, root)
            self.assertEqual(inventory["files"]["a.png"]["sha256"], T.sha256(root / "a.png"))
            row.update(label_status="positive", bow_point=[14, 11], stern_point=[14, 18], orientation_bin="unknown")
            batch = T.load_batch([row], root)
            self.assertEqual(tuple(batch[0].shape), (1, 3, 256, 256))
            self.assertAlmostEqual(float(batch[0][0, 0, 0, 0]), 42 / 255., places=6)
            cv2.imwrite(str(root / "a.png"), np.zeros((256, 256, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "changed during training"):
                T.load_batch([row], root)

    def test_preprocessing_converts_bgr_to_rgb_without_resizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.inventory_row(root, "color", 0, "one")
            image = np.zeros((256, 256, 3), np.uint8)
            image[:] = [10, 20, 30]
            cv2.imwrite(str(root / row["image"]), image)
            row["image_sha256"] = row["source_sha256"] = T.pixel_sha256(image)
            T.prepare_inventory({"train": [row]}, root)
            rgb = T.load_batch([row], root)[0][0, :, 0, 0]
            torch.testing.assert_close(rgb, torch.tensor([30., 20., 10.]) / 255.)

    def test_evaluation_is_bounded_by_batch_not_holdout_size(self):
        rows = list(range(7))
        seen = []

        def load(part, root):
            seen.append(len(part))
            base = self.batch()
            return tuple(t.expand(len(part), *t.shape[1:]).clone() for t in base)

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                n, _, h, w = x.shape
                return (torch.zeros(n, 1, h, w), torch.zeros(n, 1, h, w),
                        torch.zeros(n, 1, h, w), torch.zeros(n, 16))

        with patch.object(T, "load_batch", side_effect=load):
            result = T.evaluate(FakeModel(), rows, Path("unused"), 3, torch.device("cpu"))
        self.assertEqual(seen, [3, 3, 1])
        self.assertEqual(result["samples"], 7)

    def test_coverage_never_grants_fire_authorization(self):
        rows = [{"session": "s", "label_status": "positive", "orientation_bin": i,
                 "annotation_source": "assistant_visual"} for i in range(16)]
        self.assertEqual(T.split_summary(rows)["missing_orientation_bins"], [])
        self.assertFalse(T.metadata()["accepted"])
        self.assertFalse(T.metadata()["fire_control_authorized"])
        rows[0]["orientation_bin"] = "unknown"
        self.assertEqual(T.split_summary(rows)["missing_orientation_bins"], [0])
        self.assertEqual(T.split_summary(rows)["unknown_orientation_positives"], 1)
        self.assertEqual(T.split_summary(rows)["annotation_source_counts"], {"assistant_visual": 16})
        self.assertFalse(T.metadata(orientation_supervised=False)["orientation_supervised"])
        self.assertTrue(T.metadata()["orientation_supervised"])

    def test_failed_atomic_checkpoint_preserves_previous_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            T.atomic_save(path, {"epoch": 1}, checkpoint=True)
            previous = path.read_bytes()
            with patch.object(T.torch, "save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    T.atomic_save(path, {"epoch": 2}, checkpoint=True)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["best.pt"])

    def test_json_manifest_is_atomic_and_rejects_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            T.atomic_save(path, {"accepted": False})
            with self.assertRaises(ValueError):
                T.atomic_save(path, {"error_px": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"accepted": False})


if __name__ == "__main__":
    unittest.main()
