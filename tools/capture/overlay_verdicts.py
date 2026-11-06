#!/usr/bin/env python3
"""Record and score a human verdict on one annotated frame (stdlib only).

The viewer shows what the model claims; a person looking at that frame is the
only source of truth about whether the claim was right. One key press appends a
verdict, and the same file is what the scoring reads, so "accuracy" means
"frames the operator judged correct", not a number the detector produced about
itself.

Verdicts: correct, wrong, ambiguous (an unreadable or borderline claim is not
silently counted as either).
"""
import json
import time
from pathlib import Path

VERDICTS = ('correct', 'wrong', 'ambiguous')


def verdict_path(directory):
    return Path(directory) / 'verdicts.jsonl'


def record_verdict(directory, payload, verdict, note=None):
    """Append one verdict for the frame described by ``payload`` (boxes.json)."""
    if verdict not in VERDICTS:
        raise ValueError(f'verdict must be one of {VERDICTS}, got {verdict!r}')
    if not isinstance(payload, dict):
        raise ValueError('payload must be the parsed boxes.json dict')
    row = dict(t=time.time(), frame=payload.get('frame'), battle=payload.get('battle'),
               verdict=verdict, claims=payload.get('claims') or [])
    if note:
        row['note'] = str(note)
    path = verdict_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, default=str) + '\n')
        handle.flush()
    return row


def score_verdicts(directory):
    """Accuracy over recorded verdicts, overall and per label."""
    path = verdict_path(directory)
    rows = []
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    overall = dict(correct=0, wrong=0, ambiguous=0, frames=len(rows))
    per_label = {}
    for row in rows:
        overall[row['verdict']] = overall.get(row['verdict'], 0) + 1
        for claim in row.get('claims') or []:
            entry = per_label.setdefault(claim['label'], dict(correct=0, wrong=0, ambiguous=0))
            entry[row['verdict']] = entry.get(row['verdict'], 0) + 1
    decided = overall['correct'] + overall['wrong']
    overall['accuracy'] = round(overall['correct'] / decided, 4) if decided else None
    overall['decided'] = decided
    for label, entry in per_label.items():
        decidable = entry['correct'] + entry['wrong']
        entry['accuracy'] = round(entry['correct'] / decidable, 4) if decidable else None
    return dict(overall=overall, per_label=per_label, verdicts=str(path))
