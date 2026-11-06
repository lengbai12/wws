#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Export a manual-scope capture run into the bow review queue.

Only shots that carry a detected enemy nameplate are exported, and the scope
attestation is taken from the per-shot visual evidence recorded by the capture
tool — never assumed. Labels are still created by a human, never here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def run(run_dir: Path, session: str, out_dir: Path, *, require_scope: bool,
        attest_scope: bool = False):
    sys.path[:0] = [str(ROOT / 'tools/replay'), str(ROOT / 'tools/capture')]
    import annotate_bow

    report = json.loads((run_dir / 'report.json').read_text())
    shots = report.get('shots') or []
    if not shots:
        raise SystemExit('capture run has no shots')
    usable = []
    rejected = []
    for shot in shots:
        frame = ROOT / shot['file']
        if not frame.is_file():
            rejected.append((shot['index'], 'missing file'))
            continue
        if not shot.get('target_bbox'):
            rejected.append((shot['index'], 'no nameplate'))
            continue
        scope = bool((shot.get('scope_visual') or {}).get('scope_like'))
        if require_scope and not scope and not attest_scope:
            rejected.append((shot['index'], 'scope not evidenced'))
            continue
        if attest_scope:
            # A human asserts these frames are scoped; the exporter cannot verify
            # it, so the assertion is recorded instead of being implied.
            scope = True
        usable.append((frame, scope))
    if not usable:
        raise SystemExit(f'no usable shots ({len(rejected)} rejected)')

    # The exporter attests the scope observation once per batch; split the batch
    # so an unevidenced frame can never inherit a scoped attestation.
    exported = 0
    queue_dirs = []
    for scoped in (True, False):
        batch = [frame for frame, scope in usable if scope is scoped]
        if not batch:
            continue
        target = out_dir / ('scoped' if scoped else 'unscoped')
        result = annotate_bow.export_candidates(batch, session, target, scope_observed=scoped)
        exported += result['candidates']
        queue_dirs.append(dict(dir=str(target), scoped=scoped, **result))
    return dict(session=session, shots=len(shots), usable=len(usable), exported=exported,
                scope_source='operator_attestation' if attest_scope else 'image_evidence',
                batches=queue_dirs, rejected=rejected,
                labels_created=0, training_authorized=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-scope', action='store_true',
                        help='drop shots whose scope evidence is false')
    parser.add_argument('--attest-scope', action='store_true',
                        help='the operator confirms the shots are scoped (recorded as such)')
    args = parser.parse_args(argv)
    summary = run(args.run, args.session, args.out, require_scope=args.require_scope,
                  attest_scope=args.attest_scope)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
