#!/usr/bin/env python3
"""Offline benchmark: how often would the pipeline fire, and how fast is each stage.

Reads saved frames, runs the same stages the live entry runs, and reports:

  * verdict distribution (ready / refused) and the blocker histogram,
  * p50/p95 milliseconds per stage (sense, target, aim, gate),
  * the single number the goal is measured on: share of frames that end "ready".

It never touches the game: no capture, no input. ``--assume-runtime-ok`` models a
live run whose process identity and focus checks already passed, which is how the
runtime gate is meant to be satisfied (those are not per-frame evidence).
"""
import argparse
import glob
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/capture'))
sys.path.insert(0, str(ROOT / 'tools/replay'))

import aim_pipeline as P  # noqa: E402


def frames_from(patterns):
    files = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(str(p) for p in path.glob('frame-*.png')))
        else:
            files.extend(sorted(glob.glob(pattern)))
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('frames', nargs='+', help='directories or globs of PNG frames')
    parser.add_argument('--out', required=True)
    parser.add_argument('--allow-minimap', action='store_true')
    parser.add_argument('--assume-runtime-ok', action='store_true',
                        help='treat identity and focus as already verified (a live run does this once)')
    parser.add_argument('--fire-tolerance', type=float, default=140.0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)

    files = frames_from(args.frames)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print('no frames matched', args.frames)
        return 2

    timings = {k: [] for k in ('sense', 'target', 'aim', 'gate')}
    verdicts, blockers = Counter(), Counter()
    rows = []
    for path in files:
        image = np.asarray(Image.open(path).convert('RGB'))
        t0 = time.monotonic(); sense = P.stage_sense(image); t1 = time.monotonic()
        target = P.stage_target(sense, args.allow_minimap); t2 = time.monotonic()
        aim = P.stage_aim(sense, target); t3 = time.monotonic()
        decision = P.stage_gate(sense, target, aim, args.fire_tolerance,
                                identity_ok=True if args.assume_runtime_ok else None,
                                focus_ok=True if args.assume_runtime_ok else None)
        t4 = time.monotonic()
        timings['sense'].append((t1 - t0) * 1000)
        timings['target'].append((t2 - t1) * 1000)
        timings['aim'].append((t3 - t2) * 1000)
        timings['gate'].append((t4 - t3) * 1000)
        verdict = 'ready' if decision.get('allow') else 'refused'
        verdicts[verdict] += 1
        for b in decision.get('blockers') or []:
            blockers[b] += 1
        rows.append(dict(frame=Path(path).name,
                         basis=target['basis'], verdict=verdict,
                         blockers=decision.get('blockers') or [],
                         reload=(sense.get('hud') or {}).get('reload_ready'),
                         weapon=(sense.get('hud') or {}).get('weapon_kind'),
                         aim_km=sense.get('aim_km'), flight_s=sense.get('flight_s'),
                         ms=round((t4 - t0) * 1000, 1)))

    def pct(values, q):
        return round(statistics.quantiles(values, n=100)[q - 1], 1) if len(values) > 4 else (
            round(max(values), 1) if values else None)

    summary = dict(
        frames=len(files), verdicts=dict(verdicts),
        ready_share=round(verdicts['ready'] / len(files), 3),
        blockers=dict(blockers.most_common()),
        stage_ms={k: dict(p50=pct(v, 50), p95=pct(v, 95)) for k, v in timings.items()},
        per_frame_total_ms=round(sum(sum(v) for v in timings.values()) / len(files), 1),
        assume_runtime_ok=args.assume_runtime_ok, allow_minimap=args.allow_minimap,
    )
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
    (out / 'rows.json').write_text(json.dumps(rows, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
