import unittest
import numpy as np

from bow_recognizer import decode_prediction, prepare_crop


class BowDecodeTests(unittest.TestCase):
    def outputs(self):
        mask = np.zeros((256,256),np.float32)
        mask[90:150,70:190] = .95
        bow, stern = np.zeros_like(mask), np.zeros_like(mask)
        bow[120,80], stern[120,180] = 1,1
        orientation = np.zeros(16,np.float32); orientation[4] = 1
        return mask,bow,stern,orientation

    def test_bow_and_stern_remain_distinct_from_orientation_and_fire(self):
        out = decode_prediction(*self.outputs())
        self.assertEqual(out['bow_point'],[80,120])
        self.assertEqual(out['stern_point'],[180,120])
        self.assertEqual(out['orientation_bin'],4)
        self.assertFalse(out['fire_authorized'])
        self.assertIsNone(out['world_velocity'])

    def test_uniform_heatmap_is_rejected_despite_visible_hull(self):
        mask,bow,stern,orient = self.outputs()
        bow[:] = 1/bow.size
        self.assertIsNone(decode_prediction(mask,bow,stern,orient))

    def test_two_hulls_or_keypoint_outside_support_reject(self):
        mask,bow,stern,orient = self.outputs()
        mask[10:30,10:30] = .95
        self.assertIsNone(decode_prediction(mask,bow,stern,orient))
        mask[10:30,10:30] = 0
        bow[:] = 0;bow[10,10] = 1
        self.assertIsNone(decode_prediction(mask,bow,stern,orient))

    def test_empty_mask_unknown_direction_and_nan_reject(self):
        mask,bow,stern,orient = self.outputs()
        self.assertIsNone(decode_prediction(mask*0,bow,stern,orient))
        self.assertIsNone(decode_prediction(mask,bow,stern,np.full(16,1/16)))
        bow[0,0] = np.nan
        self.assertIsNone(decode_prediction(mask,bow,stern,orient))

    def test_unsupervised_orientation_reports_unknown_without_rejecting_keypoints(self):
        mask, bow, stern, _ = self.outputs()
        orientation = np.full(16, 1/16)
        self.assertIsNone(decode_prediction(mask, bow, stern, orientation))
        result = decode_prediction(mask, bow, stern, orientation, orientation_supervised=False)
        self.assertEqual(result['bow_point'], [80, 120])
        self.assertEqual(result['orientation_bin'], 'unknown')
        self.assertIsNone(result['orientation_degrees'])
        self.assertIsNone(result['orientation_confidence'])
        self.assertIsNone(decode_prediction(mask*0, bow, stern, orientation, orientation_supervised=False))
        bow[:] = 1/bow.size
        self.assertIsNone(decode_prediction(mask, bow, stern, orientation, orientation_supervised=False))

    def test_source_crop_pads_without_distorting_roi(self):
        frame = np.full((400,800,3),64,np.uint8)
        crop,meta = prepare_crop(frame,[350,100,100,20])
        self.assertEqual(crop.shape,(256,256,3))
        self.assertEqual(meta['source_roi'],[0,120,800,280])
        self.assertEqual(int(crop[-1].max()),0)
        self.assertEqual(int(crop[0].min()),64)

    def test_evaluation_keeps_rejected_positives_and_false_outputs(self):
        from verify_bow_model import summarize
        rows = [dict(positive=True,prediction=None),
                dict(positive=True,prediction={},bow_error_px=2.,stern_error_px=4.,orientation_correct=True),
                dict(positive=False,prediction={})]
        result = summarize(rows)
        self.assertEqual(result['positive_output_coverage'], .5)
        self.assertEqual(result['negative_false_positives'], 1)
        self.assertEqual(result['bow_error_crop_px_mean'], 2.)
        self.assertIsNone(summarize([rows[0]])['bow_error_crop_px_mean'])


if __name__=='__main__':unittest.main()
