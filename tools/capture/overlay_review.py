#!/usr/bin/env python3
"""Summarise an annotator run so identity claims can be scored by eye.

Reads detections.jsonl from recognition_overlay.py and reports, per label, how
often it was claimed, its confidence spread, and how many frames were in battle.
It does not decide correctness: a claim is correct only against what the frame
shows, which is why the annotations themselves are kept.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import overlay_verdicts as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', help='write the summary JSON here')
    args = ap.parse_args()
    directory = Path(args.dir)
    frames = [json.loads(line) for line in (directory / 'detections.jsonl').read_text().splitlines() if line.strip()]
    per_label = defaultdict(lambda: dict(frames=0, claims=0, confidences=[], km=[]))
    per_frame_counts = Counter()
    battle_frames = sum(1 for row in frames if row.get('battle') is True)
    for row in frames:
        labels = []
        for claim in row.get('claims') or []:
            entry = per_label[claim['label']]
            entry['frames'] += 1
            entry['claims'] += 1
            entry['confidences'].append(round(float(claim['confidence']), 3))
            if claim.get('km') is not None:
                entry['km'].append(float(claim['km']))
            labels.append(claim['label'])
        per_frame_counts[tuple(sorted(labels))] += 1
    summary = dict(dir=str(directory), frames=len(frames), battle_frames=battle_frames,
                   labels={label: dict(
                       frames=value['frames'], claims=value['claims'],
                       confidence_min=min(value['confidences']),
                       confidence_median=sorted(value['confidences'])[len(value['confidences']) // 2],
                       confidence_max=max(value['confidences']),
                       km_min=(min(value['km']) if value['km'] else None),
                       km_max=(max(value['km']) if value['km'] else None))
                   for label, value in sorted(per_label.items())},
                   frame_shapes={','.join(k) or 'none': v for k, v in per_frame_counts.most_common(10)},
                   verdicts=V.score_verdicts(directory))
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
