from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locked_target_motion import LockedTargetMotion, SCHEMA


class LockedTargetMotionTests(unittest.TestCase):
    def setUp(self):
        self.motion = LockedTargetMotion()

    def observe(self, t, *, range_km=10.0, x=100.0, dx=0.0, conf=0.95,
                target="enemy", view="scope", now_age=0.05, flight=5.0,
                aim=10.0, locked=True, delay=0.0):
        return self.motion.observe(
            target_id=target,
            view_id=view,
            captured_ns=int((t + 1.0) * 1e9),
            now_ns=int((t + 1.0 + now_age) * 1e9),
            range_interval_km=[range_km - 0.05, range_km + 0.05],
            aim_range_km=aim,
            flight_s=flight,
            locked=locked,
            image_shift=(dx, 0.0, conf),
            target_x=x,
            input_delay_s=delay,
        )

    def train(self, *, radial_rate=0.0, screen_rate=0.0, bg_rate=0.0, **kwargs):
        rows = []
        for i in range(5):
            t = i * 0.5
            rows.append(self.observe(
                t,
                range_km=10.0 + radial_rate * t,
                x=100.0 + (screen_rate + bg_rate) * t,
                dx=bg_rate * 0.5,
                **kwargs,
            ))
        return rows

    def test_ready_requires_both_axes_and_never_authorizes_fire(self):
        rows = self.train(radial_rate=0.1, screen_rate=2.0)
        out = rows[-1]
        self.assertEqual(SCHEMA, out["schema"])
        self.assertEqual("ready", out["status"])
        self.assertFalse(out["fire_authorized"])
        self.assertAlmostEqual(0.1, out["range_rate_km_s"], places=6)
        self.assertAlmostEqual(2.0, out["horizontal_velocity_px_s"], places=6)

    def test_range_prediction_available_while_horizontal_waits(self):
        for i in range(4):
            out = self.observe(i * 0.5, range_km=10.0 + i * 0.05, x=100.0,
                               conf=0.1 if i else 0.95)
        self.assertEqual("waiting", out["status"])
        self.assertIsNotNone(out["predicted_range_km"])
        self.assertIsNone(out["horizontal_velocity_px_s"])

    def test_quantization_is_preserved_in_uncertainty(self):
        out = self.train(radial_rate=0.0)[-1]
        self.assertGreaterEqual(out["range_rate_uncertainty_km_s"], 0.05)
        interval = out["predicted_range_interval_km"]
        self.assertLess(interval[0], out["predicted_range_km"])
        self.assertGreater(interval[1], out["predicted_range_km"])

    def test_explicit_two_decimal_intervals_keep_measured_precision(self):
        for i in range(5):
            out = self.motion.observe(
                target_id="enemy", view_id="scope", captured_ns=(i + 1)*500_000_000,
                now_ns=(i + 1)*500_000_000, range_interval_km=[9.995, 10.005],
                aim_range_km=10, flight_s=.1, locked=True,
                image_shift=dict(dx=0, dy=0, score=.95), target_x=100)
        self.assertEqual("ready", out["status"])
        low, high = out["predicted_range_interval_km"]
        self.assertLess(high-low, .1)
        self.assertGreaterEqual(high-low, .01)
        json.dumps(out, allow_nan=False)

    def test_static_screen_anchor_with_background_motion_has_nonzero_lead(self):
        for i in range(5):
            out = self.observe(i * .5, x=100.0, dx=1.5)
        self.assertEqual("ready", out["status"])
        self.assertAlmostEqual(-3.0, out["horizontal_velocity_px_s"], places=6)
        self.assertAlmostEqual(-15.15, out["horizontal_lead_px"], places=6)
    def test_missing_background_shift_clears_horizontal_history(self):
        self.train(screen_rate=2.0)
        out = self.motion.observe(
            target_id="enemy", view_id="scope", captured_ns=3_500_000_000,
            now_ns=3_550_000_000, range_interval_km=[9.95, 10.05],
            aim_range_km=10.0, flight_s=5.0, locked=True,
            image_shift=None, target_x=105.0,
        )
        self.assertEqual("waiting", out["status"])
        self.assertIsNone(out["horizontal_velocity_px_s"])
        self.assertIn("horizontal_shift_missing", out["reason"])

    def test_camera_translation_is_subtracted_from_target_motion(self):
        out = self.train(screen_rate=2.0, bg_rate=8.0)[-1]
        self.assertAlmostEqual(2.0, out["horizontal_velocity_px_s"], places=6)

    def test_approaching_and_receding_ranges_have_signed_rates(self):
        approaching = self.train(radial_rate=-0.2)[-1]
        receding_estimator = LockedTargetMotion()
        self.motion = receding_estimator
        receding = self.train(radial_rate=0.2)[-1]
        self.assertLess(approaching["range_rate_km_s"], 0.0)
        self.assertGreater(receding["range_rate_km_s"], 0.0)

    def test_flight_and_input_delay_both_extend_horizon(self):
        out = self.train(screen_rate=2.0, delay=0.4)[-1]
        self.assertAlmostEqual(5.45, out["horizon_s"], places=6)
        self.assertAlmostEqual(out["horizontal_velocity_px_s"] * 5.45,
                               out["horizontal_lead_px"], places=6)

    def test_duplicate_and_old_epoch_timestamps_do_not_accumulate(self):
        self.observe(1.0)
        duplicate = self.observe(1.0)
        self.assertEqual("rejected", duplicate["status"])
        self.assertEqual("duplicate_timestamp", duplicate["reason"])
        self.assertEqual(0, duplicate["samples"])
        self.observe(1.0)
        old = self.motion.observe(
            target_id="enemy", view_id="scope", captured_ns=500_000_000,
            now_ns=2_100_000_000, range_interval_km=[9.95, 10.05],
            aim_range_km=10, flight_s=5, locked=True,
            image_shift=(0, 0, 1), target_x=100,
        )
        self.assertEqual("rejected", old["status"])
        self.assertEqual("timestamp_out_of_order", old["reason"])

    def test_target_view_lock_and_expiry_reset_history(self):
        self.observe(0.0)
        switched = self.observe(0.5, target="other")
        self.assertEqual("target_or_view_changed", switched["reason"])
        lost = self.observe(1.0, locked=False)
        self.assertEqual("lock_lost", lost["reason"])
        stale = self.motion.observe(
            target_id="enemy", view_id="scope", captured_ns=2_000_000_000,
            now_ns=2_800_000_000, range_interval_km=[9.95, 10.05],
            aim_range_km=10, flight_s=5, locked=True,
        )
        self.assertEqual("observation_expired", stale["reason"])


    def test_fractional_timestamps_and_missing_identities_are_rejected(self):
        base = dict(
            target_id="enemy", view_id=("source", "scope"),
            captured_ns=1_000_000_000, now_ns=1_000_000_000,
            range_interval_km=[9.95, 10.05], aim_range_km=10,
            flight_s=5, locked=True,
        )
        for update in (
            {"captured_ns": 1.5},
            {"now_ns": 1.5},
            {"captured_ns": 0},
            {"target_id": ""},
            {"target_id": None},
            {"view_id": "  "},
            {"view_id": ()},
        ):
            observation = dict(base)
            observation.update(update)
            out = LockedTargetMotion().observe(**observation)
            self.assertEqual("rejected", out["status"], update)
            self.assertIn(out["reason"], {"invalid_time", "invalid_identity"}, update)
    def test_invalid_values_and_large_uncertainty_wait(self):
        bad = self.motion.observe(
            target_id="enemy", view_id="scope", captured_ns=float("nan"),
            now_ns=0, range_interval_km=[9, 10], aim_range_km=10,
            flight_s=5, locked=True,
        )
        self.assertEqual("rejected", bad["status"])
        self.assertIsNone(bad["range_rate_km_s"])
        wide = LockedTargetMotion()
        for i in range(5):
            out = wide.observe(
                target_id="e", view_id="v", captured_ns=(i + 1) * 500_000_000,
                now_ns=(i + 1) * 500_000_000, range_interval_km=[5.0, 15.0],
                aim_range_km=10, flight_s=5, locked=True,
                image_shift=(0, 0, 1), target_x=i * 2.0,
            )
        self.assertEqual("waiting", out["status"])
        self.assertEqual("range_rate_uncertainty_high", out["reason"])

    def test_cli_reads_jsonl_and_writes_hashes_and_solutions(self):
        with tempfile.TemporaryDirectory() as td:
            input_path = Path(td) / "observations.jsonl"
            output_path = Path(td) / "out.json"
            rows = []
            for i in range(3):
                rows.append({
                    "target_id": "e", "view_id": "v",
                    "captured_ns": (i + 1) * 500_000_000,
                    "now_ns": (i + 1) * 500_000_000,
                    "range_interval_km": [9.95, 10.05],
                    "aim_range_km": 10.0, "flight_s": 5.0,
                    "locked": True, "image_shift": [0, 0, 0.9],
                    "target_x": i * 2.0,
                })
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("locked_target_motion.py")),
                "--observations", str(input_path), "--out", str(output_path),
            ], check=True, capture_output=True, text=True)
            report = json.loads(output_path.read_text())
            self.assertFalse(report["fire_authorized"])
            self.assertEqual(3, len(report["solutions"]))
            self.assertEqual(64, len(report["input_sha256"]))
            self.assertEqual(64, len(report["implementation_sha256"]))


if __name__ == "__main__":
    unittest.main()
