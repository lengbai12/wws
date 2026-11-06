import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locked_aim_control import LockedAimControl


def solution(*, target_id="enemy-a", aim=10.0, error=0.0, lead=0.0, width=0.02):
    predicted = aim + error
    return dict(status="ready", target_id=target_id, horizontal_lead_px=lead,
                range_error_km=error, horizon_s=5.0,
                predicted_range_interval_km=[predicted-width/2, predicted+width/2])


def frame(index=0, **changes):
    values = dict(solution=solution(), target_x=160.0, reticle_x=100.0,
                  aim_range_km=10.0, captured_ns=1_000_000_000+index*100_000_000,
                  view_id="scope-a", pixel_hash="pixels-"+str(index),
                  image_shift=(0, 0, .95))
    values.update(changes)
    return values


class LockedAimControlTests(unittest.TestCase):
    def setUp(self):
        self.control = LockedAimControl()

    def baseline(self, **changes):
        self.assertEqual([0, 0], self.control.update(**frame(0, **changes))["move"])
        self.assertEqual([0, 0], self.control.update(**frame(1, **changes))["move"])
        return self.control.update(**frame(2, **changes))

    def assert_no_input(self, result, status=None):
        self.assertEqual([0, 0], result["move"])
        self.assertFalse(result["fire_authorized"])
        json.dumps(result, allow_nan=False)
        if status:
            self.assertEqual(status, result["status"])

    def test_first_frame_cannot_probe_without_no_input_baseline(self):
        out = self.control.update(**frame())
        self.assert_no_input(out, "waiting")
        self.assertEqual("awaiting_no_input_drift_baseline", out["reason"])
        self.assertIsNone(out["gain_x_px_per_unit"])

    def test_two_consecutive_baseline_intervals_allow_one_probe(self):
        out = self.baseline()
        self.assertEqual([20, 0], out["move"])
        self.assertEqual("correcting", out["status"])

    def test_natural_client_lock_drift_is_subtracted_from_first_response(self):
        self.baseline(image_shift=(4, 0, .95))
        out = self.control.update(**frame(3, image_shift=(-6, 0, .95)))
        self.assertAlmostEqual(-.5, out["gain_x_px_per_unit"])
        self.assertAlmostEqual(4, out["response"]["natural_drift_projection"])
        self.assertAlmostEqual(-10, out["response"]["isolated"])
        self.assert_no_input(out, "waiting")

    def test_drift_projection_uses_actual_elapsed_time(self):
        self.baseline(image_shift=(4, 0, .95))
        out = self.control.update(**frame(4, image_shift=(-2, 0, .95)))
        self.assertAlmostEqual(8, out["response"]["natural_drift_projection"])
        self.assertAlmostEqual(-.5, out["gain_x_px_per_unit"])

    def test_natural_drift_alone_never_calibrates(self):
        self.baseline(image_shift=(4, 0, .95))
        for i in (3, 4, 5):
            out = self.control.update(**frame(i, target_x=130+100*i,
                                               image_shift=(4, 0, .95)))
            self.assert_no_input(out, "rejected" if i == 5 else "waiting")
            self.assertIsNone(out["gain_x_px_per_unit"])

    def test_variable_baseline_bounds_uncertain_response(self):
        self.control.update(**frame(0))
        self.control.update(**frame(1, image_shift=(0, 0, .95)))
        self.control.update(**frame(2, image_shift=(8, 0, .95)))
        out = self.control.update(**frame(3, image_shift=(6, 0, .95)))
        self.assert_no_input(out, "waiting")
        self.assertIsNone(out["gain_x_px_per_unit"])

    def test_missing_flow_breaks_baseline(self):
        self.control.update(**frame(0))
        self.control.update(**frame(1))
        out = self.control.update(**frame(2, image_shift=None))
        self.assert_no_input(out, "waiting")
        self.assertEqual(0, out["no_input_baseline_intervals"])

    def test_missing_response_flow_cannot_recover_false_sum(self):
        self.baseline()
        self.control.update(**frame(3, image_shift=None))
        out = self.control.update(**frame(4, image_shift=(-20, 0, .95)))
        self.assert_no_input(out, "waiting")
        self.assertIsNone(out["gain_x_px_per_unit"])

    def test_subpixel_response_accumulates_without_more_inputs(self):
        self.baseline()
        self.assert_no_input(self.control.update(**frame(3, image_shift=(-.6, 0, .95))), "waiting")
        out = self.control.update(**frame(4, image_shift=(-.6, 0, .95)))
        self.assertAlmostEqual(-1.2/20, out["gain_x_px_per_unit"])
        self.assert_no_input(out, "waiting")

    def test_response_deadline_begins_at_delivery_not_input_frame(self):
        self.baseline()
        self.control.input_completed(1_500_000_000)
        out = self.control.update(**frame(4, image_shift=(-100, 0, .95)))
        self.assert_no_input(out, 'waiting')
        self.assertIsNone(out['gain_x_px_per_unit'])
        # Old capture is 900 ms ago; delivery is only 600 ms ago.
        out = self.control.update(**frame(11, image_shift=(-20, 0, .95)))
        self.assert_no_input(out, 'waiting')
        self.assertAlmostEqual(out['gain_x_px_per_unit'], -1.0)

    def test_delivery_does_not_extend_response_beyond_750ms(self):
        self.baseline()
        self.control.input_completed(1_500_000_000)
        self.control.update(**frame(8, image_shift=(0, 0, .95)))
        out = self.control.update(**frame(13, image_shift=(-20, 0, .95)))
        self.assert_no_input(out, 'rejected')
        self.assertIsNone(out['gain_x_px_per_unit'])

    def test_delivery_timestamp_cannot_rearm_an_already_delivered_move(self):
        self.baseline()
        self.control.input_completed(1_500_000_000)
        with self.assertRaises(ValueError):
            self.control.input_completed(1_600_000_000)
        self.assertIsNone(self.control.gain_x_px_per_unit)

    def test_signed_response_and_horizontal_bounds(self):
        for shift, expected in ((-10, 120), (10, -120)):
            self.control.reset()
            self.baseline()
            out = self.control.update(**frame(3, image_shift=(shift, 0, .95)))
            self.assertAlmostEqual(shift/20, out["gain_x_px_per_unit"])
            self.control.update(**frame(4))
            out = self.control.update(**frame(5))
            self.assertEqual([expected, 0], out["move"])

    def test_y_gain_uses_reticle_range_and_subtracts_natural_range_drift(self):
        self.control.update(**frame(0, target_x=100, solution=solution(error=.8)))
        self.control.update(**frame(1, target_x=100, aim_range_km=10.01,
                                    solution=solution(aim=10.01, error=.8)))
        probe = self.control.update(**frame(2, target_x=100, aim_range_km=10.02,
                                            solution=solution(aim=10.02, error=.8)))
        self.assertEqual([0, 3], probe["move"])
        out = self.control.update(**frame(3, target_x=100, aim_range_km=10.06,
                                          solution=solution(aim=10.06, error=1.0)))
        self.assertAlmostEqual(.01, out["gain_y_km_per_unit"])
        self.assertAlmostEqual(.01, out["response"]["natural_drift_projection"])

    def test_negative_reticle_response_keeps_vertical_limit_and_sign(self):
        self.baseline(target_x=100, solution=solution(error=.8))
        out = self.control.update(**frame(3, target_x=100, aim_range_km=9.97,
                                          solution=solution(aim=9.97, error=.8)))
        self.assertAlmostEqual(-.01, out["gain_y_km_per_unit"])
        self.control.update(**frame(4, target_x=100, aim_range_km=9.97,
                                    solution=solution(aim=9.97, error=.8)))
        out = self.control.update(**frame(5, target_x=100, aim_range_km=9.97,
                                          solution=solution(aim=9.97, error=.8)))
        self.assertEqual([0, -3], out["move"])

    def test_invalid_motion_solution_clears_pending(self):
        self.baseline()
        out = self.control.update(**frame(3, solution={"status": "rejected"}))
        self.assert_no_input(out, "rejected")
        self.assertIsNone(self.control._pending)
        self.assertIsNone(self.control.gain_x_px_per_unit)

    def test_target_radial_motion_cannot_calibrate_y(self):
        self.baseline(target_x=100, solution=solution(error=.8))
        for i in (3, 4, 5):
            out = self.control.update(**frame(i, target_x=100, solution=solution(error=i)))
            self.assert_no_input(out, "rejected" if i == 5 else "waiting")
            self.assertIsNone(out["gain_y_km_per_unit"])

    def test_y_quantization_uncertainty_cannot_become_precise_gain(self):
        kwargs = dict(target_x=100, solution=solution(error=.8),
                      aim_range_interval_km=[9.95, 10.05])
        self.baseline(**kwargs)
        out = self.control.update(**frame(3, target_x=100, aim_range_km=10.1,
                                          aim_range_interval_km=[10.05, 10.15],
                                          solution=solution(aim=10.1, error=.8)))
        self.assert_no_input(out, "waiting")
        self.assertIsNone(out["gain_y_km_per_unit"])

    def quantized_probe(self):
        for i in range(6):
            out = self.control.update(**frame(i, captured_ns=1_000_000_000+i*200_000_000,
                target_x=100, aim_range_interval_km=[9.95,10.05],
                solution=solution(error=2.0)))
        self.assertEqual(out['move'], [0,3])
        self.control.input_completed(2_020_000_000)

    def test_quantized_y_accumulates_only_after_each_measured_pixel_response(self):
        self.quantized_probe()
        out = self.control.update(**frame(20, captured_ns=2_220_000_000,
            target_x=100, aim_range_km=9.9, aim_range_interval_km=[9.85,9.95],
            image_shift=(0,4,.95), solution=solution(aim=9.9,error=2.1)))
        self.assertEqual(out['move'], [0,3])
        self.assertIsNone(out['gain_y_km_per_unit'])
        self.assertEqual(out['vertical_probe']['total_units'], 6)
        self.control.input_completed(2_240_000_000)
        out = self.control.update(**frame(21, captured_ns=2_440_000_000,
            target_x=100, aim_range_km=9.7, aim_range_interval_km=[9.65,9.75],
            image_shift=(0,4,.95), solution=solution(aim=9.7,error=2.3)))
        self.assert_no_input(out, 'waiting')
        self.assertAlmostEqual(out['gain_y_km_per_unit'], -.05)
        low, high = out['gain_y_interval_km_per_unit']
        self.assertLess(low, -.05)
        self.assertGreater(high, -.05)
        self.assertLess(high, 0)
        self.assertIsNone(out['vertical_probe'])

    def test_quantized_y_never_stacks_on_zero_or_missing_pixel_response(self):
        for shift in ((0,0,.95), None, (0,4,.1), (12,4,.95)):
            self.control.reset()
            self.quantized_probe()
            out = self.control.update(**frame(20, captured_ns=2_220_000_000,
                target_x=100, aim_range_km=9.9, aim_range_interval_km=[9.85,9.95],
                image_shift=shift, solution=solution(aim=9.9,error=2.1)))
            self.assert_no_input(out, 'waiting')
            self.assertIsNone(out['gain_y_km_per_unit'])

    def test_resolved_quantized_range_does_not_require_background_translation(self):
        for shift in ((0, 0, .95), None):
            self.control.reset()
            self.quantized_probe()
            out = self.control.update(**frame(20, captured_ns=2_220_000_000,
                target_x=100, aim_range_km=9.7, aim_range_interval_km=[9.65,9.75],
                image_shift=shift, solution=solution(aim=9.7,error=2.3)))
            self.assert_no_input(out, 'waiting')
            self.assertAlmostEqual(out['gain_y_km_per_unit'], -.1)
            low, high = out['gain_y_interval_km_per_unit']
            self.assertLess(low, -.1)
            self.assertGreater(high, -.1)
            self.assertLess(high, 0)
            self.assertIsNone(out['vertical_probe'])

    def test_quantized_y_pixel_direction_reversal_refuses_accumulation(self):
        self.quantized_probe()
        self.control.update(**frame(20, captured_ns=2_220_000_000,
            target_x=100, aim_range_km=9.9, aim_range_interval_km=[9.85,9.95],
            image_shift=(0,4,.95), solution=solution(aim=9.9,error=2.1)))
        self.control.input_completed(2_240_000_000)
        out = self.control.update(**frame(21, captured_ns=2_440_000_000,
            target_x=100, aim_range_km=9.9, aim_range_interval_km=[9.85,9.95],
            image_shift=(0,-4,.95), solution=solution(aim=9.9,error=2.1)))
        self.assert_no_input(out, 'rejected')

    def test_quantized_y_ends_after_four_visual_steps_without_range_resolution(self):
        self.quantized_probe()
        for i in range(4):
            stamp = 2_220_000_000+i*220_000_000
            out = self.control.update(**frame(20+i, captured_ns=stamp,
                target_x=100, aim_range_interval_km=[9.95,10.05],
                image_shift=(0,4,.95), solution=solution(error=2.0)))
            if i < 3:
                self.assertEqual(out['move'], [0,3])
                self.control.input_completed(stamp+20_000_000)
            else:
                self.assert_no_input(out, 'rejected')

    def test_quantized_y_alignment_margin_loss_never_emits_next_probe(self):
        self.quantized_probe()
        out = self.control.update(**frame(20, captured_ns=2_220_000_000,
            target_x=100, aim_range_interval_km=[9.95,10.05],
            image_shift=(0,4,.95), solution=solution(error=0)))
        self.assert_no_input(out, 'rejected')

    def test_duplicate_pixel_or_time_clears_pending_and_gain(self):
        for changes in (dict(pixel_hash="pixels-2"), dict(captured_ns=1_200_000_000),
                        dict(captured_ns=1_100_000_000), dict(view_id="new"),
                        dict(solution=solution(target_id="other")),
                        dict(now_ns=2_100_000_001)):
            self.control.reset()
            self.baseline()
            out = self.control.update(**frame(3, **changes))
            self.assert_no_input(out, "rejected")
            self.assertIsNone(out["gain_x_px_per_unit"])

    def test_pending_response_expires_without_stacking(self):
        self.baseline()
        out = self.control.update(**frame(10))
        self.assert_no_input(out, "rejected")

    def test_waiting_motion_warms_baseline_but_does_not_move(self):
        waiting = dict(status="waiting", target_id="enemy-a")
        for i in range(3):
            self.assert_no_input(self.control.update(**frame(i, solution=waiting)), "waiting")
        self.assertEqual([20, 0], self.control.update(**frame(3))["move"])

    def test_waiting_motion_publishes_current_interval_but_cannot_align(self):
        waiting = solution(width=1.2)
        waiting.update(status="waiting", reason="range_rate_uncertainty_high")
        out = self.control.update(**frame(target_x=100, solution=waiting))
        self.assert_no_input(out, "waiting")
        self.assertFalse(out["range_alignment"]["contained"])
        self.assertTrue(out["range_alignment"]["uncertain"])
        # An earlier diagnostic is not reused if the next producer has no fit.
        out = self.control.update(**frame(1, target_x=100,
                                          solution=dict(status="waiting", target_id="enemy-a")))
        self.assert_no_input(out, "waiting")
        self.assertIsNone(out["range_alignment"])

    def test_wide_interval_waits_when_overlapping_but_moves_when_all_ahead(self):
        out = self.control.update(**frame(target_x=100, solution=solution(width=1.2)))
        self.assert_no_input(out, "waiting")
        self.control.reset()
        out = self.baseline(target_x=100, solution=solution(error=.8, width=.2))
        self.assertEqual([0, 3], out["move"])
        self.assertTrue(out["range_alignment"]["uncertain"])

    def test_two_decimal_cells_can_prove_full_error_interval_alignment(self):
        out = self.control.update(**frame(target_x=95, solution=solution(lead=5, width=.02),
                                          aim_range_interval_km=[9.995, 10.005]))
        self.assert_no_input(out, "aligned")
        self.assertTrue(out["range_alignment"]["contained"])
        low, high = out["range_alignment"]["error_interval_km"]
        self.assertGreaterEqual(low, -.05)
        self.assertLessEqual(high, .05)

    def test_one_decimal_aim_cell_is_usable_with_relaxed_range_tolerance(self):
        out = self.control.update(**frame(target_x=100, solution=solution(width=.02),
                                          aim_range_interval_km=[9.95, 10.05]))
        self.assert_no_input(out, "aligned")
        self.assertTrue(out["range_alignment"]["contained"])

    def test_real_one_decimal_motion_can_align_without_input(self):
        from locked_target_motion import LockedTargetMotion
        motion = LockedTargetMotion()
        for i in range(20):
            stamp = 1_000_000_000+i*200_000_000
            predicted = motion.observe(target_id='enemy-a', view_id='scope-a',
                captured_ns=stamp, now_ns=stamp+20_000_000,
                range_interval_km=[9.95,10.05], aim_range_km=10., flight_s=5.,
                locked=True, image_shift=(0,0,.95), target_x=100.)
            out = self.control.update(**frame(i, solution=predicted, target_x=100,
                captured_ns=stamp, now_ns=stamp+20_000_000,
                aim_range_interval_km=[9.95,10.05]))
            self.assert_no_input(out)
            self.assertNotEqual(out['reason'], 'range_quantization_infeasible')
        self.assertEqual(predicted['status'], 'ready')
        self.assertEqual(out['status'], 'aligned')

    def test_positive_error_interval_can_correct_without_full_alignment(self):
        out = self.baseline(target_x=100, solution=solution(error=.495, width=.02))
        self.assertEqual([0, 3], out["move"])
        self.assertTrue(out["range_alignment"]["uncertain"])

    def test_prediction_centre_overlap_cannot_hide_wide_interval(self):
        out = self.control.update(**frame(target_x=100, solution=solution(width=1.2),
                                          aim_range_interval_km=[9.95, 10.05]))
        self.assert_no_input(out, "waiting")
        self.assertFalse(out["range_alignment"]["contained"])

    def test_horizontal_has_priority_and_never_combines_axes(self):
        out = self.baseline(solution=solution(error=.5))
        self.assertEqual([20, 0], out["move"])
        self.assertFalse(out["fire_authorized"])

    def test_relaxed_alignment_boundary_preserves_interval_uncertainty(self):
        for error, status in ((.4, 'aligned'), (.401, 'waiting'), (-.4, 'aligned')):
            self.control.reset()
            out = self.control.update(**frame(target_x=130, solution=solution(error=error, width=.1),
                                              aim_range_interval_km=[9.95,10.05]))
            self.assert_no_input(out, status)
            self.assertEqual(out['range_alignment']['tolerance_km'], .5)
        self.control.reset()
        out = self.control.update(**frame(target_x=131))
        self.assert_no_input(out, 'waiting')


if __name__ == "__main__":
    unittest.main()
