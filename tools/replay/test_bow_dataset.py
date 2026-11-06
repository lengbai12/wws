import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import bow_dataset as B
from annotate_bow import AnnotationUI, export_candidates, save_sample


class BowDatasetTests(unittest.TestCase):
    def make_root(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        y, x = np.mgrid[:256, :256]
        image = np.stack((x, y, (x + y) % 256), axis=-1).astype(np.uint8)
        mask = np.zeros((256, 256), np.uint8)
        cv2.rectangle(mask, (80, 80), (170, 170), 255, -1)
        cv2.imwrite(str(root / "a.png"), image)
        cv2.imwrite(str(root / "source.png"), image)
        cv2.imwrite(str(root / "a.mask.png"), mask)
        return root

    def row(self, root, session="real-session-1", **overrides):
        image = cv2.imread(str(root / "a.png"))
        source = cv2.imread(str(root / "source.png"))
        row = dict(image="a.png", mask="a.mask.png", bow_point=[120, 90],
                   stern_point=[120, 160], orientation_bin=3, session=session,
                   label_status="positive", reviewed=True, annotation_source="manual",
                   reviewer="human-a", source_image="source.png", source_roi=[0, 0, 256, 256],
                   image_sha256=B.pixel_sha256(image), source_sha256=B.pixel_sha256(source),
                   crop_transform=B.CROP_CONVENTION)
        row.update(overrides)
        return row

    def annotations(self, root, rows):
        path = root / "annotations.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def negative(self, root, **overrides):
        cv2.imwrite(str(root / "zero.png"), np.zeros((256, 256), np.uint8))
        return self.row(root, mask="zero.png", label_status="negative", bow_point=None,
                        stern_point=None, orientation_bin="unknown", rejection_reason="occluded", **overrides)

    def test_valid_reviewed_row(self):
        root = self.make_root()
        got = B.validate_row(self.row(root), root)
        self.assertEqual(got["orientation_bin"], 3)
        self.assertEqual(got["bow_point"], [120., 90.])

    def test_validator_accepts_one_real_session(self):
        root = self.make_root()
        rows = B.load_annotations(self.annotations(root, [self.row(root)]))
        self.assertEqual([row["session"] for row in rows], ["real-session-1"])

    def test_requires_existing_contract_fields(self):
        root = self.make_root()
        for field in B.REQUIRED:
            with self.subTest(field=field):
                row = self.row(root)
                del row[field]
                with self.assertRaisesRegex(ValueError, "missing fields"):
                    B.validate_row(row, root)

    def test_rejects_unreviewed_or_nonmanual_sources(self):
        root = self.make_root()
        for overrides in ({"reviewed": False}, {"reviewed": 1},
                          {"annotation_source": "dark_blob"}, {"reviewer": " "},
                          {"session": ""}, {"session": 3}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                B.validate_row(self.row(root, **overrides), root)

    def test_all_sixteen_camera_bins_and_unknown(self):
        root = self.make_root()
        for orientation in [*range(16), "unknown"]:
            with self.subTest(orientation=orientation):
                self.assertEqual(B.validate_row(self.row(root, orientation_bin=orientation), root)["orientation_bin"], orientation)
        row = self.row(root)
        del row["orientation_bin"]
        self.assertEqual(B.validate_row(row, root)["orientation_bin"], "unknown")
        self.assertIn("camera-relative", B.ORIENTATION_CONVENTION)
        self.assertIn("4=starboard", B.ORIENTATION_CONVENTION)
        self.assertIn("12=port", B.ORIENTATION_CONVENTION)

    def test_rejects_invalid_orientation_types_and_range(self):
        root = self.make_root()
        for value in [True, False, None, -1, 16, 1.0, "0", float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "orientation_bin"):
                B.validate_row(self.row(root, orientation_bin=value), root)

    def test_points_require_finite_nonbool_numeric_coordinates(self):
        root = self.make_root()
        for field in ("bow_point", "stern_point"):
            for value in ([256, 90], [-1, 90], [True, 90], ["90", 90],
                          [float("nan"), 90], [90, float("inf")], [90, -float("inf")],
                          None, [90], [90, 90, 90]):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    B.validate_row(self.row(root, **{field: value}), root)

    def test_points_must_differ_and_have_mask_support(self):
        root = self.make_root()
        with self.assertRaisesRegex(ValueError, "must differ"):
            B.validate_row(self.row(root, stern_point=[120, 90]), root)
        for field in ("bow_point", "stern_point"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unsupported"):
                B.validate_row(self.row(root, **{field: [2, 2]}), root)
        B.validate_row(self.row(root, bow_point=[80 - B.POINT_MASK_TOLERANCE, 90]), root)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            B.validate_row(self.row(root, bow_point=[80 - B.POINT_MASK_TOLERANCE - .1, 90]), root)

    def test_masks_must_be_binary_uint8_single_channel_and_have_support(self):
        root = self.make_root()
        variants = [np.full((256, 256), 1, np.uint8),
                    np.zeros((256, 256), np.uint16),
                    np.zeros((256, 256, 3), np.uint8),
                    np.zeros((255, 256), np.uint8),
                    np.zeros((256, 256), np.uint8)]
        for mask in variants:
            cv2.imwrite(str(root / "bad.png"), mask)
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaises(ValueError):
                B.validate_row(self.row(root, mask="bad.png"), root)

    def test_negative_rejection_contract(self):
        root = self.make_root()
        row = self.negative(root)
        self.assertIsNone(B.validate_row(row, root)["bow_point"])
        for reason in B.REJECTION_REASONS:
            B.validate_row(dict(row, rejection_reason=reason), root)
        for overrides in ({"mask": "a.mask.png"}, {"bow_point": [90, 90]},
                          {"stern_point": [90, 90]}, {"orientation_bin": 0},
                          {"rejection_reason": "guess"}, {"rejection_reason": None}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                B.validate_row(dict(row, **overrides), root)
        with self.assertRaisesRegex(ValueError, "rejection_reason"):
            B.validate_row(self.row(root, rejection_reason="occluded"), root)

    def test_checks_existing_paths_dimensions_and_actual_pixels(self):
        root = self.make_root()
        for field in ("image", "mask", "source_image"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not exist"):
                B.validate_row(self.row(root, **{field: "missing.png"}), root)
        for field in ("image_sha256", "source_sha256"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                B.validate_row(self.row(root, **{field: "0" * 64}), root)
        cv2.imwrite(str(root / "small.png"), np.zeros((32, 32, 3), np.uint8))
        with self.assertRaisesRegex(ValueError, "256x256"):
            B.validate_row(self.row(root, image="small.png"), root)
        # A self-consistent crop hash alone must not hide a wrong source frame.
        source = cv2.imread(str(root / "source.png"))
        source[0, 0] = [17, 29, 31]
        cv2.imwrite(str(root / "source.png"), source)
        with self.assertRaisesRegex(ValueError, "pixels do not match"):
            B.validate_row(self.row(root), root)

    def test_source_roi_finite_integer_bounds_and_transform(self):
        root = self.make_root()
        for roi in ([True, 0, 256, 256], [0, 0, float("nan"), 256],
                    [0, 0, 256.0, 256], [0, 0, 0, 256], [-1, 0, 256, 256],
                    [0, 0, 257, 256], [0, 0, 256], None):
            with self.subTest(roi=roi), self.assertRaisesRegex(ValueError, "source_roi"):
                B.validate_row(self.row(root, source_roi=roi), root)
        with self.assertRaisesRegex(ValueError, "crop_transform"):
            B.validate_row(self.row(root, crop_transform="stretch"), root)

    def test_square_padding_matches_inference_geometry(self):
        root = self.make_root()
        source = np.full((100, 200, 3), 127, dtype=np.uint8)
        cv2.imwrite(str(root / "source.png"), source)
        crop = B.source_crop(source, [0, 0, 200, 100])
        self.assertTrue(np.all(crop[:128] == 127))
        self.assertFalse(np.any(crop[128:]))
        cv2.imwrite(str(root / "a.png"), crop)
        row = self.negative(root, source_roi=[0, 0, 200, 100])
        B.validate_row(row, root)

    def test_duplicate_pixel_identity_is_rejected(self):
        root = self.make_root()
        path = self.annotations(root, [self.row(root), self.row(root)])
        with self.assertRaisesRegex(ValueError, ":2: duplicate crop pixel"):
            B.load_annotations(path)

    def test_same_frame_cannot_acquire_invented_sessions(self):
        root = self.make_root()
        first = self.row(root)
        source = cv2.imread(str(root / "source.png"))
        second_crop = B.source_crop(source, [0, 0, 200, 256])
        cv2.imwrite(str(root / "second.png"), second_crop)
        second = self.negative(root, session="invented-session", source_roi=[0, 0, 200, 256],
                               image="second.png", image_sha256=B.pixel_sha256(second_crop))
        with self.assertRaisesRegex(ValueError, "multiple sessions"):
            B.load_annotations(self.annotations(root, [first, second]))

    def test_json_errors_and_empty_input_have_context(self):
        root = self.make_root()
        path = root / "annotations.jsonl"
        for content in ("{bad}\n", "[]\n", "null\n"):
            path.write_text(content)
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "annotations.jsonl:1:"):
                B.load_annotations(path)
        path.write_text(" \n")
        with self.assertRaisesRegex(ValueError, "no annotations"):
            B.load_annotations(path)

    def test_summary_reports_only_real_known_camera_coverage(self):
        root = self.make_root()
        rows = [B.validate_row(self.row(root, orientation_bin=0), root),
                B.validate_row(self.row(root, orientation_bin="unknown"), root),
                B.validate_row(self.negative(root), root)]
        summary = B.annotation_summary(rows)
        self.assertEqual(summary["orientation_bins"], [0])
        self.assertEqual(summary["orientation_bin_counts"], [1] + [0] * 15)
        self.assertEqual(summary["unknown_orientation_positives"], 1)
        self.assertEqual(summary["negative_samples"], 1)
        self.assertFalse(summary["all_16_bins_observed"])

    def save(self, root, **overrides):
        source_path = root / "source.png"
        values = dict(annotations=root / "annotations.jsonl", sample_id="real-frame-1",
                      source_path=source_path, source=cv2.imread(str(source_path)),
                      roi=[0, 0, 256, 256], session="actual-session", reviewer="human-a",
                      polygon=[[80, 80], [170, 80], [170, 170], [80, 170]],
                      bow=[120, 90], stern=[120, 160], orientation="unknown", reviewed=True)
        values.update(overrides)
        return save_sample(**values)

    def test_assistant_visual_save_preserves_truthful_source_and_unknown_view(self):
        root = self.make_root()
        row = self.save(root, annotation_source="assistant_visual", reviewer="assistant:model")
        loaded = B.load_annotations(root / "annotations.jsonl")
        self.assertEqual(loaded[0], row)
        self.assertEqual(row["annotation_source"], "assistant_visual")
        self.assertEqual(row["reviewer"], "assistant:model")
        self.assertEqual(row["orientation_bin"], "unknown")
        self.assertEqual(B.annotation_summary(loaded)["annotation_source_counts"], {"assistant_visual": 1})

    def test_assistant_visual_requires_explicit_review_and_reviewer(self):
        root = self.make_root()
        for overrides in ({"reviewed": False}, {"reviewer": " "}, {"annotation_source": "automatic"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.save(root, **({"annotation_source": "assistant_visual"} | overrides))
        self.assertFalse((root / "annotations.jsonl").exists())

    def test_manual_save_is_auditable_and_refuses_overwrite(self):
        root = self.make_root()
        row = self.save(root)
        loaded = B.load_annotations(root / "annotations.jsonl")
        self.assertEqual(loaded[0], row)
        self.assertEqual(row["source_original_path"], str((root / "source.png").resolve()))
        manifest = (root / "annotations.jsonl").read_bytes()
        image = (root / row["image"]).read_bytes()
        with self.assertRaises(ValueError):
            self.save(root)
        self.assertEqual((root / "annotations.jsonl").read_bytes(), manifest)
        self.assertEqual((root / row["image"]).read_bytes(), image)

    def test_manual_save_rejects_missing_review_or_fake_polygon(self):
        root = self.make_root()
        for overrides in ({"reviewed": False}, {"polygon": []},
                          {"polygon": [[True, 90], [90, 100], [100, 100]]},
                          {"bow": [float("nan"), 90]}, {"sample_id": "../escape"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.save(root, **overrides)
        self.assertFalse((root / "annotations.jsonl").exists())
        self.assertFalse((root / "real-frame-1.png").exists())
        self.assertFalse((root / "annotations.jsonl.lock").exists())

    def test_save_negative_never_invents_point_or_view_labels(self):
        root = self.make_root()
        row = self.save(root, polygon=[], bow=None, stern=None, orientation="unknown", negative_reason="invisible")
        self.assertEqual(row["label_status"], "negative")
        self.assertIsNone(row["bow_point"])
        self.assertFalse(np.any(cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)))

    def test_save_existing_asset_is_not_overwritten(self):
        root = self.make_root()
        target = root / "real-frame-1.png"
        target.write_bytes(b"existing asset")
        with self.assertRaises(FileExistsError):
            self.save(root)
        self.assertEqual(target.read_bytes(), b"existing asset")
        self.assertFalse((root / "annotations.jsonl").exists())

    def test_ui_requires_explicit_view_and_revokes_review_on_change(self):
        ui = AnnotationUI(np.zeros((256, 256, 3), dtype=np.uint8))
        ui.polygon = [[80, 80], [170, 80], [170, 170]]
        ui.bow, ui.stern = [90, 90], [150, 150]
        ui.key(ord("r"))
        self.assertFalse(ui.reviewed)
        # Unknown must be an explicit human choice, never an inferred view.
        ui.key(ord("u"))
        ui.key(ord("r"))
        self.assertTrue(ui.reviewed)
        self.assertEqual(ui.orientation, "unknown")
        ui.mouse(cv2.EVENT_LBUTTONDOWN, 300, 300, None, None)
        self.assertFalse(ui.reviewed)
        ui.key(ord("n"))
        self.assertIsNone(ui.bow)
        self.assertIsNone(ui.stern)
        self.assertFalse(ui.polygon)
        self.assertEqual(ui.orientation, "unknown")
        ui.key(ord("r"))
        self.assertTrue(ui.reviewed)

    def test_native_bgra_source_preserves_rgb_without_alpha_compositing(self):
        root = self.make_root()
        original = cv2.imread(str(root/'source.png'))
        bgra = np.dstack((original, np.zeros(original.shape[:2],np.uint8)))
        cv2.imwrite(str(root/'native.png'), bgra)
        self.assertTrue(np.array_equal(B.read_source(root/'native.png'), original))

    def native_source(self, root):
        image = np.zeros((1600, 2560, 3), np.uint8)
        image[:256, :256] = cv2.imread(str(root / "source.png"))
        cv2.imwrite(str(root / "source.png"), image)

    def test_export_queue_is_auditable_but_has_no_fake_labels(self):
        root = self.make_root()
        self.native_source(root)
        output = root / "pending"
        plate = {"nameplate_bbox": [90, 20, 30, 8]}
        with patch("main_view_perception.detect_nameplates", return_value=[plate]) as detector:
            result = export_candidates([root / "source.png"], "actual-session", output,
                                       scope_observed=True)
        self.assertEqual(result["labels_created"], 0)
        self.assertEqual(result["known_orientation_coverage"], [])
        self.assertEqual(result["candidates"], 1)
        self.assertTrue(detector.call_args.kwargs["scope_observed"])
        row = json.loads((output / "queue.jsonl").read_text())
        self.assertFalse(row["reviewed"])
        self.assertEqual(row["review_status"], "pending")
        for field in ("mask", "bow_point", "stern_point", "orientation_bin", "label_status"):
            self.assertNotIn(field, row)
        source = cv2.imread(str(output / row["source_image"]))
        crop = cv2.imread(str(output / row["image"]))
        self.assertEqual(B.pixel_sha256(source), row["source_sha256"])
        self.assertEqual(B.pixel_sha256(crop), row["image_sha256"])
        self.assertTrue(np.array_equal(crop, B.source_crop(source, row["source_roi"])))
        self.assertIn("--roi", row["annotation_command"])
        with self.assertRaisesRegex(ValueError, "missing fields"):
            B.load_annotations(output / "queue.jsonl")
        with self.assertRaises(FileExistsError):
            export_candidates([root / "source.png"], "actual-session", output,
                              scope_observed=True)

    def test_export_requires_scope_attestation_and_rejects_duplicate_frames(self):
        root = self.make_root()
        self.native_source(root)
        cv2.imwrite(str(root / "a.png"), cv2.imread(str(root / "source.png")))
        with self.assertRaisesRegex(ValueError, "scope observation"):
            export_candidates([root / "source.png"], "actual-session", root / "not-created",
                              scope_observed=False)
        self.assertFalse((root / "not-created").exists())
        with patch("main_view_perception.detect_nameplates", return_value=[]):
            with self.assertRaisesRegex(ValueError, "duplicate source pixels"):
                export_candidates([root / "source.png", root / "a.png"], "actual-session",
                                  root / "pending", scope_observed=True)
        self.assertFalse((root / "pending").exists())

    def test_export_rejects_non_native_pixels_without_leaving_a_queue(self):
        root = self.make_root()
        for width, height in ((1924, 1058), (2560, 1440)):
            cv2.imwrite(str(root / "wrong.png"), np.zeros((height, width, 3), np.uint8))
            with self.subTest(size=(width, height)):
                with self.assertRaisesRegex(ValueError, f"got {width}x{height}"):
                    export_candidates([root / "wrong.png"], "actual-session", root / "pending",
                                      scope_observed=True)
                self.assertFalse((root / "pending").exists())


if __name__ == "__main__":
    unittest.main()
