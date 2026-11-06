"""Synthetic fixtures exercise rejection/accounting, not semantic model accuracy."""
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from annotate_bow import save_sample
from train_bow_model import BowNet, metadata
import verify_bow_model as V


class IndependentEvaluation(unittest.TestCase):
    def fixture(self, root):
        image = np.random.default_rng(8).integers(0,256,(256,256,3),dtype=np.uint8)
        cv2.imwrite(str(root/'source.png'),image)
        save_sample(root/'reviewed.jsonl','synthetic-test',root/'source.png',image,
                    [0,0,256,256],'synthetic-heldout','test-fixture-not-real-human',
                    [[30,70],[210,70],[210,160],[30,160]],[40,110],[200,110],4,reviewed=True)
        model=BowNet(1)
        for parameter in model.parameters():parameter.data.zero_()
        torch.save(dict(metadata(),base=1,state_dict=model.state_dict()),root/'model.pt')
        manifest=dict(model=dict(path='model.pt',sha256=hashlib.sha256((root/'model.pt').read_bytes()).hexdigest()),
                      dataset=dict(samples=[dict(session='synthetic-training',source_pixels_sha256='other',image_pixels_sha256='other')]))
        (root/'manifest.json').write_text(json.dumps(manifest))
        return manifest

    def test_actual_flat_model_is_measured_as_rejected_positive_not_success(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);self.fixture(root)
            self.assertEqual(V.main(['--annotations',str(root/'reviewed.jsonl'),
                '--training-manifest',str(root/'manifest.json'),'--out',str(root/'result')]),0)
            report=json.loads((root/'result/report.json').read_text())
            self.assertEqual(report['summary']['positive_output_coverage'],0)
            self.assertIsNone(report['summary']['bow_error_crop_px_mean'])
            self.assertFalse(report['accepted'])
            self.assertFalse(report['fire_authorized'])
            self.assertEqual(report['coverage']['missing_orientation_bins'],[i for i in range(16) if i!=4])

    def test_shared_session_refuses_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            root=Path(directory);manifest=self.fixture(root)
            manifest['dataset']['samples'][0]['session']='synthetic-heldout'
            (root/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(SystemExit):
                V.main(['--annotations',str(root/'reviewed.jsonl'),
                    '--training-manifest',str(root/'manifest.json'),'--out',str(root/'result')])
            self.assertFalse((root/'result').exists())


if __name__=='__main__':unittest.main()
