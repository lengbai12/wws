#!/usr/bin/env python3
"""Run raw RGB enemy detection + native icon species on an explicit corpus split.

Species references come from packet roster shipParamsId, never inference inputs.
Metrics are conditional on one-to-one enemy localization matches; unmatched detections
and unresolved reference types are reported separately. Thresholds are fixed before test.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

import enemy_marker_classifier as E
import train_minimap_heatmap as T


def species_metrics(confusion, populations, unmatched_species, detection_count, truth_count, species):
    """Separate localization, conditional classification, and end-to-end coverage."""
    known = sum(n for (truth, pred), n in confusion.items() if pred != 'unknown')
    correct = sum(n for (truth, pred), n in confusion.items() if pred == truth)
    total = sum(confusion.values())
    matched = populations.get('matched', 0)
    per_class = {}
    for name in species:
        count = sum(n for (truth, pred), n in confusion.items() if truth == name)
        accepted = sum(n for (truth, pred), n in confusion.items()
                       if truth == name and pred != 'unknown')
        right = confusion.get((name, name), 0)
        per_class[name] = dict(reference=count, classified=accepted, correct=right,
                               accuracy_when_classified=right/accepted if accepted else None)
    return dict(populations=dict(populations), species_reference=total,
                classified=known, correct=correct, coverage=known/total if total else None,
                accuracy_when_classified=correct/known if known else None,
                unknown=total-known, per_species=per_class,
                confusion={truth: {pred: n for (t, pred), n in confusion.items() if t == truth}
                           for truth, _ in confusion},
                unmatched_species=dict(unmatched_species),
                detection_precision=matched/detection_count if detection_count else None,
                detection_recall=matched/truth_count if truth_count else None,
                correct_species_recall=correct/truth_count if truth_count else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package', required=True)
    ap.add_argument('--split', choices=('validation', 'test'), default='test')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.stride < 1 or args.out.exists():
        raise SystemExit('positive stride and unused output path required')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    recognizer = E.EnemyRecognizer(args.package, str(T.pick_device('auto')))
    package = recognizer.package
    archive = np.load(package['dataset']['heatmaps'], allow_pickle=False)
    sessions, splits = archive['session_ids'], archive['splits']
    grid = int(archive['grid'][0])
    radius = T.match_radius(dict(crop_rect=package['input']['crop_rect'], grid=grid))
    full = np.load(T.twin_path(package['dataset']['images'], 'images'), mmap_mode='r', allow_pickle=False)
    index = np.flatnonzero(splits == args.split)[::args.stride]
    if not len(index):
        raise SystemExit('empty split')
    references = {}; inventories = {}; local = {}
    for session in sorted(set(sessions[index])):
        replay = next((root / (str(session)+'.wowsreplay') for root in
                       (Path('replays'), Path('replays/steam159'))
                       if (root / (str(session)+'.wowsreplay')).exists()), None)
        if replay is None:
            raise SystemExit('missing replay for '+str(session))
        ref = subprocess.run([sys.executable, str(Path(__file__).with_name('export_ship_species.py')),
                              str(replay)], check=True, capture_output=True, text=True)
        references[str(session)] = json.loads(ref.stdout)
        rows = [json.loads(line) for line in Path('replays/parsed/inventory', str(session)+'.inventory.jsonl').open()]
        ids = np.flatnonzero(sessions == session)
        if len(rows) != len(ids):
            raise SystemExit('inventory/corpus sample count mismatch: '+str(session))
        inventories[str(session)] = rows
        local.update({int(sample): pos for pos, sample in enumerate(ids)})
    previews = []; elapsed = []; confusion = Counter(); populations = Counter()
    unmatched_species = Counter()
    detection_count = truth_count = 0
    with (args.out/'detections.jsonl').open('w') as output:
        for number, sample in enumerate(index):
            E.guard()
            session = str(sessions[sample]); row = inventories[session][local[int(sample)]]
            markers = [m for m in row['markers'] if m['class'] == 'enemy_visible'
                       and 0 <= m['u'] <= 1 and 0 <= m['v'] <= 1]
            rgb = np.asarray(full[sample]); start = time.perf_counter()
            detections = recognizer.detect(rgb)
            elapsed.append((time.perf_counter()-start)*1000)
            detection_count += len(detections); truth_count += len(markers)
            used = set()
            for detection in detections:
                distances = sorted((float(np.hypot((detection['u']-m['u'])*(grid-1),
                                                  (detection['v']-m['v'])*(grid-1))), k)
                                   for k, m in enumerate(markers) if k not in used)
                truth = None; vehicle = None
                if distances and distances[0][0] <= radius:
                    k = distances[0][1]; used.add(k); vehicle = markers[k]['vehicle']
                    truth = references[session]['vehicles'].get(str(vehicle), {}).get('species')
                    populations['matched'] += 1
                    if truth is None:
                        populations['unresolved_species'] += 1
                else:
                    populations['unmatched'] += 1
                    unmatched_species[detection['species']] += 1
                    if detection['species'] != 'unknown':
                        populations['unmatched_classified'] += 1
                if truth is not None:
                    confusion[truth, detection['species']] += 1
                item = dict(sample=int(sample), session=session, vehicle=vehicle,
                            truth_species=truth, **detection)
                output.write(json.dumps(item)+'\n')
                if len(previews) < 64:
                    x=round(detection['u']*719); y=round(detection['v']*722)
                    padded = np.pad(rgb[max(0,y-24):min(723,y+24),max(0,x-24):min(720,x+24)],
                                    ((max(0,24-y),max(0,y+24-723)),(max(0,24-x),max(0,x+24-720)),(0,0)))
                    previews.append((padded, truth or '?', detection['species']))
            if number % 100 == 0:
                print(number, '/', len(index), dict(populations), flush=True)
    metrics = species_metrics(confusion, populations, unmatched_species,
                              detection_count, truth_count, recognizer.ship_icons.hashes)
    report = dict(schema_version=2, package=args.package, package_sha256=T.sha256(args.package), split=args.split,
                  stride=args.stride, frames=len(index), detections=detection_count, truth=truth_count,
                  **metrics,
                  matcher=dict(min_score=recognizer.ship_icons.min_score,
                               min_margin=recognizer.ship_icons.min_margin,
                               svg_sha256=recognizer.ship_icons.hashes),
                  timing_ms=dict(mean=float(np.mean(elapsed)),p95=float(np.percentile(elapsed,95))),
                  limits='Repeated frames are correlated; species accuracy excludes unmatched detections. '
                         'Correct species recall includes missed, unknown, and misclassified enemies. '
                         'Missing species are unverified; template score is not calibrated probability.',
                  references=references)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    canvas=Image.new('RGB',(8*128,8*80)); draw=ImageDraw.Draw(canvas)
    for i,(patch,truth,pred) in enumerate(previews):
        x=(i%8)*128; y=(i//8)*80
        canvas.paste(Image.fromarray(patch),(x,y));draw.text((x,y+49),truth,fill='white')
        draw.text((x,y+63),pred,fill='yellow')
    canvas.save(args.out/'preview.png')
    print(json.dumps({k:v for k,v in report.items() if k!='references'},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
