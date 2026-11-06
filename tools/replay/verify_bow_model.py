#!/usr/bin/env python3
"""Measure image-only BowRecognizer on reviewed independent sessions; never accepts fire."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2

from bow_dataset import load_annotations, annotation_summary
from bow_recognizer import BowRecognizer


def summarize(rows):
    positives = [r for r in rows if r['positive']]
    emitted = [r for r in rows if r['prediction'] is not None]
    matched = [r for r in positives if r['prediction'] is not None]
    return dict(samples=len(rows), positives=len(positives), outputs=len(emitted),
                rejected=len(rows)-len(emitted),
                negative_false_positives=sum(not r['positive'] for r in emitted),
                positive_output_coverage=len(matched)/len(positives) if positives else None,
                bow_error_crop_px_mean=sum(r['bow_error_px'] for r in matched)/len(matched) if matched else None,
                stern_error_crop_px_mean=sum(r['stern_error_px'] for r in matched)/len(matched) if matched else None,
                orientation_correct=sum(r['orientation_correct'] is True for r in matched),
                orientation_evaluated=sum(r['orientation_correct'] is not None for r in matched),
                # Errors on accepted outputs must always be read with coverage.
                semantics='conditional errors; rejected positives remain misses')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--training-manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    rows = load_annotations(a.annotations)
    manifest = json.loads(a.training_manifest.read_text())
    model = a.training_manifest.parent / manifest['model']['path']
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest['model']['sha256']:
        p.error('model identity changed')
    inventory = manifest['dataset']['samples']
    seen_sessions = {r['session'] for r in inventory}
    seen_sources = {r['source_pixels_sha256'] for r in inventory}
    seen_pixels = {r['image_pixels_sha256'] for r in inventory}
    for row in rows:
        if (row['session'] in seen_sessions or row['source_sha256'] in seen_sources
                or row['image_sha256'] in seen_pixels):
            p.error('independent evaluation overlaps training/validation/test inventory')
    recognizer = BowRecognizer(model)
    a.out.mkdir(parents=True, exist_ok=False)
    records = []
    with (a.out/'detections.jsonl').open('x') as stream:
        for row in rows:
            bgr = cv2.imread(str(a.annotations.parent/row['image']))
            result = recognizer.predict_crop(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            positive = row['label_status']=='positive'
            record = dict(image=row['image'], session=row['session'], positive=positive,
                          annotation_source=row['annotation_source'], reviewer=row['reviewer'],
                          orientation_bin=row['orientation_bin'], prediction=result,
                          bow_error_px=None, stern_error_px=None, orientation_correct=None)
            if result is not None and positive:
                record['bow_error_px'] = math.dist(result['bow_point'],row['bow_point'])
                record['stern_error_px'] = math.dist(result['stern_point'],row['stern_point'])
                if isinstance(row['orientation_bin'],int) and isinstance(result['orientation_bin'],int):
                    record['orientation_correct'] = result['orientation_bin']==row['orientation_bin']
            records.append(record)
            stream.write(json.dumps(record,allow_nan=False)+'\n')
    report = dict(summary=summarize(records), coverage=annotation_summary(rows),
                  by_orientation={str(i):summarize([r for r in records if r['orientation_bin']==i]) for i in range(16)},
                  model_sha256=recognizer.model_sha256,
                  orientation_supervised=recognizer.orientation_supervised,
                  annotations_sha256=hashlib.sha256(a.annotations.read_bytes()).hexdigest(),
                  training_manifest_sha256=hashlib.sha256(a.training_manifest.read_bytes()).hexdigest(),
                  implementation_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in
                      (Path(__file__),Path(__file__).with_name('bow_recognizer.py'),Path(__file__).with_name('bow_dataset.py'))},
                  accepted=False, fire_authorized=False,
                  reason='measurement only; no calibrated operational acceptance threshold')
    (a.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,allow_nan=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
