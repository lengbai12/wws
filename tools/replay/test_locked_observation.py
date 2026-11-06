import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locked_observation as O
from fire_hud import FireHudReader
from main_view_perception import detect_nameplates

ROOT = Path(__file__).resolve().parents[2]


def hud():
    return dict(aiming_view=dict(status="observed", mode="scope_ruler", zoom_confirmed=None,
                               raw=dict(sides=[dict(accepted=True, spacing_px=20., span_px=600.)]*2,
                                        ticks_xy=[[500+i*20, 800] for i in range(60)])),
                flight_time=dict(status="observed", seconds=.1,
                                 distance=dict(status="observed", km=10.,
                                               raw=dict(numbers=[dict(status="observed", value=10.,
                                                                      raw=dict(candidate_text="10.00km"))]))))


def target():
    return dict(target_id="enemy-a", nameplate_bbox=[1230, 600, 100, 10],
                aim_px=[2500, 999], distance_interval_km=[9.995, 10.005])


def evidence(frame, stamp):
    return dict(source=dict(node_id=1), identity=dict(pid=123, start_ticks=1),
                window="window-a", raw_pixel_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                captured_monotonic_ns=stamp, capture_age_semantics=O.CAPTURE_SEMANTICS)


class LockedObservationTests(unittest.TestCase):
    def setUp(self):
        self.observer = O.LockedObservation()
        self.lock_patch = patch.object(O, "confirm_target", return_value=dict(confirmed=True, detection_index=0))
        self.lock = self.lock_patch.start()
        self.addCleanup(self.lock_patch.stop)
        self.shift_patch = patch.object(O, "estimate_camera_shift", return_value=(0., 0., .95))
        self.shift = self.shift_patch.start()
        self.addCleanup(self.shift_patch.stop)

    def observe(self, i, *, data=None, selected=None, candidates=None, changes=None, age_ns=10_000_000):
        frame = np.full((1600, 2560, 3), 50, dtype=np.uint8)
        frame[0, 0] = (i % 255, 20, 30)
        stamp = 10_000_000_000 + i*250_000_000
        meta = evidence(frame, stamp)
        meta.update(changes or {})
        selected = target() if selected is None else selected
        candidates = [selected] if candidates is None else candidates
        with patch.object(O.time, "monotonic_ns", return_value=stamp + age_ns):
            result = self.observer.observe(frame, hud() if data is None else data,
                                           candidates, selected, [1280, 800], meta)
        json.dumps(result, allow_nan=False)
        self.assertFalse(result["fire_authorized"])
        return result

    def warm(self, **kwargs):
        return [self.observe(i, **kwargs) for i in range(6)][-1]

    def test_unknown_numeric_zoom_uses_observed_current_ruler_layout(self):
        out = self.warm()
        self.assertEqual("aligned", out["status"])
        self.assertIsNone(out["zoom"])
        self.assertTrue(out["control"]["range_alignment"]["contained"])
        self.assertEqual(1280., out["target_anchor_x"])
        self.assertEqual(0., out["control"]["horizontal_error_px"])
        self.assertEqual([0, 0], out["move"])

    def test_static_target_screen_and_moving_background_has_lead(self):
        self.shift.return_value = (2., 0., .95)
        out = self.warm()
        self.assertAlmostEqual(-8., out["solution"]["horizontal_velocity_px_s"])
        self.assertLess(out["solution"]["horizontal_lead_px"], 0)
        self.assertIn("local", out["solution"]["motion_semantics"])

    def test_same_identity_ring_checks_every_candidate(self):
        selected = target()
        another = dict(selected, target_id="enemy-b", nameplate_bbox=[1500, 600, 100, 10])
        self.lock.return_value = dict(confirmed=True, detection_index=1)
        out = self.observe(0, selected=selected, candidates=[selected, another])
        self.assertEqual("rejected", out["status"])
        self.assertFalse(out["locked"])
        self.assertEqual(2, len(self.lock.call_args.args[1]))

    def test_repeated_target_id_is_ambiguous_even_with_ring(self):
        selected = target()
        out = self.observe(0, selected=selected, candidates=[selected, dict(selected)])
        self.assertEqual("rejected", out["status"])

    def test_nested_ambiguous_numbers_never_supply_top_level_values(self):
        for key in ("seconds", "km"):
            data = hud()
            if key == "seconds":
                data["flight_time"].pop("seconds")
                data["flight_time"]["raw"] = dict(numbers=[dict(status="observed", value=5)])
            else:
                data["flight_time"]["distance"].pop("km")
            out = self.observe(0, data=data)
            self.assertEqual("rejected", out["status"])
            self.assertEqual("current_aim_range_or_flight_time_unconfirmed", out["reason"])

    def test_actual_ocr_precision_is_retained(self):
        out = self.observe(0)
        self.assertEqual([9.995, 10.005], out["aim_range_interval_km"])
        data = hud()
        data["flight_time"]["distance"]["raw"]["numbers"][0]["raw"]["candidate_text"] = "10.0km"
        out = self.observe(1, data=data)
        self.assertEqual([9.95, 10.05], out["aim_range_interval_km"])
        self.assertEqual(out['status'], 'waiting')
        self.assertNotEqual(out['reason'], 'range_quantization_infeasible')

    def test_missing_or_changed_scope_geometry_clears_history(self):
        first = self.warm()
        changed = hud()
        for side in changed["aiming_view"]["raw"]["sides"]:
            side["spacing_px"] = 25
        out = self.observe(6, data=changed)
        self.assertEqual("waiting", out["status"])
        self.assertEqual(1, out["solution"]["samples"])
        self.assertNotEqual(first["view_id"], out["view_id"])
        changed["aiming_view"]["raw"].pop("sides")
        self.assertEqual("rejected", self.observe(7, data=changed)["status"])

    def test_source_identity_and_camera_epoch_change_reset_history(self):
        for change in (dict(source=dict(node_id=2)), dict(window="window-b"),
                       dict(identity=dict(pid=999)), dict(camera_epoch=2)):
            self.observer.reset()
            first = self.warm()
            out = self.observe(6, changes=change)
            self.assertEqual(1, out["solution"]["samples"])
            self.assertNotEqual(first["view_id"], out["view_id"])

    def test_repeated_timestamp_pixel_hash_and_expiry_clear_every_model(self):
        self.warm()
        out = self.observe(5)
        self.assertEqual("rejected", out["status"])
        self.assertEqual(0, len(self.observer.motion._range_points))
        self.assertIsNone(self.observer.control.gain_x_px_per_unit)
        fresh = self.observe(6)
        self.assertEqual(1, fresh["solution"]["samples"])
        stale = self.observe(7, age_ns=750_000_001)
        self.assertEqual("rejected", stale["status"])
        self.assertIsNone(self.observer._previous_frame)
        future = self.observe(8, age_ns=-1)
        self.assertEqual("rejected", future["status"])

    def test_missing_source_or_wrong_pixel_digest_rejects(self):
        for change in (dict(source=None), dict(raw_pixel_sha256="wrong"),
                       dict(capture_age_semantics="drawable_update")):
            self.assertEqual("rejected", self.observe(0, changes=change)["status"])

    def test_explicit_reset_changes_epoch_identity(self):
        first = self.observe(0)
        self.observer.reset()
        second = self.observe(1)
        self.assertNotEqual(first["view_id"], second["view_id"])

    def test_same_pixels_with_later_timestamp_reset_history(self):
        self.warm()
        # Native image index wraps at 255 but acquisition time moves forward.
        out = self.observe(260)
        self.assertEqual("rejected", out["status"])
        self.assertEqual("capture_repeated_or_out_of_order", out["reason"])
        self.assertIsNone(self.observer._previous_frame)

    def test_range_selected_distance_evidence_handoff(self):
        selected = target()
        selected.pop("distance_interval_km")
        selected["distance_evidence"] = dict(status="observed", interval_km=[9.995, 10.005])
        out = self.observe(0, selected=selected)
        self.assertEqual([9.995, 10.005], out["target_range_interval_km"])

    def test_lock_loss_clears_models_and_source_frame(self):
        self.warm()
        self.lock.return_value = dict(confirmed=False, detection_index=None)
        out = self.observe(6)
        self.assertEqual("rejected", out["status"])
        self.assertFalse(out["locked"])
        self.assertEqual([], self.observer.motion._range_points)
        self.assertIsNone(self.observer._previous_frame)

    def test_wide_projection_reports_uncertainty_without_fake_alignment(self):
        selected = dict(target(), distance_interval_km=[9.5, 10.5])
        out = self.warm(selected=selected)
        self.assertEqual("rejected", out["status"])
        self.assertEqual("range_quantization_infeasible", out["reason"])
        self.assertFalse(out["fire_authorized"])
        self.assertEqual([0, 0], out["move"])

    def test_real_frame_consumer_uses_real_hud_and_lock(self):
        self.lock_patch.stop()
        self.shift_patch.stop()
        frame = np.asarray(Image.open(ROOT / "state/aimprobe-live-scope/key-action.png").convert("RGB"))
        candidates = [dict(c, target_id="enemy-"+str(i)) for i, c in enumerate(detect_nameplates(frame))]
        self.assertTrue(candidates)
        selected = dict(candidates[0], distance_interval_km=[9.95, 10.05])
        data = FireHudReader().detect(frame)
        stamp = 10_000_000_000
        with patch.object(O.time, "monotonic_ns", return_value=stamp+1):
            out = self.observer.observe(frame, data, candidates, selected, [1280, 800], evidence(frame, stamp))
        json.dumps(out, allow_nan=False)
        self.assertTrue(out["locked"])
        self.assertFalse(out["fire_authorized"])
        self.assertEqual([0, 0], out["move"])


if __name__ == "__main__":
    unittest.main()
