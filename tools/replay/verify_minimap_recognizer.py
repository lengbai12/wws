#!/usr/bin/env python3
"""Exercise raw-frame minimap consumer, own localization and enemy dedup together."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

import enemy_marker_classifier as E
from enemy_glyph_dedup import deduplicate
from minimap_recognizer import MinimapRecognizer
import train_minimap_heatmap as T


def enemy_counts(detections, truth, width, height):
    used = set(); matched = 0
    for detection in detections:
        candidates = [(float(np.hypot(detection['u']*(width-1)-x,
                                     detection['v']*(height-1)-y)), i)
                      for i, (x, y) in enumerate(truth) if i not in used]
        if candidates:
            distance, identity = min(candidates)
            if distance <= 24:
                used.add(identity); matched += 1
    return dict(predicted=len(detections), matched=matched, truth=len(truth))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package', required=True)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.stride < 1:
        raise SystemExit('stride must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    consumer = MinimapRecognizer(args.package, str(T.pick_device('auto')))
    package = consumer.enemy.package
    archive = np.load(package['dataset']['heatmaps'], allow_pickle=False)
    truth, splits, sessions = archive['truth'], archive['splits'], archive['session_ids']
    full = np.load(T.twin_path(package['dataset']['images'], 'images'), mmap_mode='r', allow_pickle=False)
    indices = np.flatnonzero(splits == 'test')[::args.stride]
    counts = {}; times = []; errors = []; previews = []; removed = []
    chosen = set(indices[np.linspace(0, len(indices)-1,6,dtype=int)])
    before = []

    def observe(image, detections, matcher):
        before[:] = detections
        return deduplicate(image, detections, matcher)

    original = E.deduplicate
    E.deduplicate = observe
    try:
        with (args.out/'detections.jsonl').open('w') as output:
            for number, sample in enumerate(indices):
                E.guard()
                rgb = np.asarray(full[sample]); height, width = rgb.shape[:2]
                start = time.perf_counter(); result = consumer.detect(rgb)
                times.append((time.perf_counter()-start)*1000)
                channels = {}
                for marker in truth[sample]:
                    if np.isfinite(marker[0]):
                        channels.setdefault(int(marker[0]), []).append((float(marker[1])*(width-1), float(marker[2])*(height-1)))
                own_truth = channels.get(0, [])
                enemy_truth = channels.get(package['channel'], [])
                row = counts.setdefault(str(sessions[sample]), dict(frames=0, own_truth=0,
                    own_predicted=0, own_matched=0, before_predicted=0, before_matched=0,
                    after_predicted=0, after_matched=0, enemy_truth=0))
                row['frames'] += 1; row['own_truth'] += len(own_truth)
                own = result['own']; error = None
                if own is not None:
                    row['own_predicted'] += 1
                    if own_truth:
                        error = min(float(np.hypot(own['pixel_x']-x, own['pixel_y']-y)) for x,y in own_truth)
                        if error <= 24:
                            row['own_matched'] += 1; errors.append(error)
                pre = enemy_counts(before, enemy_truth, width, height)
                post = enemy_counts(result['enemies'], enemy_truth, width, height)
                row['enemy_truth'] += len(enemy_truth)
                for prefix, metrics in (('before', pre), ('after', post)):
                    row[prefix+'_predicted'] += metrics['predicted']
                    row[prefix+'_matched'] += metrics['matched']
                if len(before) != len(result['enemies']):
                    removed.append(dict(sample=int(sample), before=pre, after=post))
                output.write(json.dumps(dict(sample=int(sample), session=str(sessions[sample]),
                    own_truth=own_truth, enemy_truth=enemy_truth, before=before,
                    **result, own_error_crop_px=error))+'\n')
                if sample in chosen:
                    image = Image.fromarray(rgb).resize((360,361)); draw = ImageDraw.Draw(image)
                    for x,y in own_truth:
                        draw.ellipse((x/2-6,y/2-6,x/2+6,y/2+6),outline='cyan',width=2)
                    if own:
                        x,y=own['pixel_x']/2,own['pixel_y']/2
                        draw.ellipse((x-3,y-3,x+3,y+3),outline='white',width=2)
                    for detection in result['enemies']:
                        x,y=detection['u']*359,detection['v']*360
                        draw.ellipse((x-4,y-4,x+4,y+4),outline='yellow')
                    previews.append(image)
                if number % 200 == 0:
                    print(number, '/', len(indices), flush=True)
    finally:
        E.deduplicate = original
    totals = {key: sum(row[key] for row in counts.values()) for key in next(iter(counts.values()))}
    for row in [totals, *counts.values()]:
        row['own_precision'] = row['own_matched']/row['own_predicted'] if row['own_predicted'] else None
        row['own_recall'] = row['own_matched']/row['own_truth'] if row['own_truth'] else None
        for prefix in ('before', 'after'):
            row[prefix+'_precision'] = row[prefix+'_matched']/row[prefix+'_predicted'] if row[prefix+'_predicted'] else None
            row[prefix+'_recall'] = row[prefix+'_matched']/row['enemy_truth'] if row['enemy_truth'] else None
    report = dict(schema_version=1, package=args.package, package_sha256=T.sha256(args.package),
        tool_sha256=T.sha256(__file__), split='test', stride=args.stride, **totals,
        by_session=counts, changed_frames=removed,
        identity=dict(base_sha256=package['base_sha256'], classifier_sha256=package['classifier_sha256'],
                      dataset=package['dataset'], own_svg_sha256=consumer.own.icon_sha256,
                      enemy_svg_sha256=consumer.enemy.ship_icons.hashes,
                      source_sha256={name:T.sha256(Path(__file__).with_name(name+'.py')) for name in
                                     ('own_ship_locator','enemy_glyph_dedup','enemy_marker_classifier','minimap_recognizer')}),
        own_error_crop_px=dict(mean=float(np.mean(errors)) if errors else None,
                               p95=float(np.percentile(errors,95)) if errors else None),
        timing_ms=dict(mean=float(np.mean(times)),p95=float(np.percentile(times,95))),
        limits='Raw-frame offline perception only; repeated frames are correlated. '
               'Ambiguous/occluded own arrows return null. Enemy ambiguity can retain duplicates.')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    canvas = Image.new('RGB',(720,1083))
    for i,image in enumerate(previews):
        canvas.paste(image,((i%2)*360,(i//2)*361))
    canvas.save(args.out/'preview.png')
    print(json.dumps({k:v for k,v in report.items() if k!='changed_frames'},indent=2),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
