#!/usr/bin/env python3
"""Describe recorded Operations routes so a waypoint policy can be judged offline.

This reads only products that already exist - `replays/parsed/routes/*.json`
(own 5 s route, enemy first appearances, minimap aim samples) and
`replays/parsed/spawn-patterns-18s.json` (which session belongs to which spawn
pattern) - and prints the tables `docs/operations-decision-plan.md` quotes.
It never starts the game, captures a frame, or injects input.

Reported per session: duration, path length in world units, mean moving speed,
stop count, key turns (RDP vertices), enemy-arrival profile.
Reported per map/pattern: how many sessions, and how far the sessions of one
pattern sit from each other (pairwise symmetric Hausdorff on 5 s route samples)
versus how far the patterns of one map sit from each other. That ratio is the
only evidence we have on whether "which pattern is this" is decidable from
geometry; it is a separability bound, not a classifier accuracy.

World coordinates are replay Position units. The metre scale is disputed
(docs/ops-playbooks.md section 0: 29.666 m/unit contradicts measured speed),
so every distance here is printed in world units and never converted.

    ./.venv/bin/python tools/replay/ops_route_report.py
    ./.venv/bin/python tools/replay/ops_route_report.py --json --out state/ops-route-report.json
"""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROUTES_DIR = ROOT / 'replays' / 'parsed' / 'routes'
PATTERNS_PATH = ROOT / 'replays' / 'parsed' / 'spawn-patterns-18s.json'
SIG_CORPUS = ROOT / 'replays' / 'parsed' / 'datasets' / 'heatmap-merged-18s-g176.npz'
SIG_OPENING = 60
SIG_COARSE = 8
SIG_THRESHOLD = 0.30

STEP_S = 5.0
TURN_MIN_DEG = 30.0
TURN_TOLERANCE = 40.0
MOVING_EPS = 0.30
STOP_EPS = 0.08
ARRIVAL_BINS = (60.0, 120.0, 240.0, 360.0)
EARLY_WINDOW_S = 180.0


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_patterns(path=PATTERNS_PATH):
    """map id -> list of {'recordings': [...], 'count': n}, order preserved."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError('pattern table must be a non-empty object')
    table = {}
    for map_id, entry in data.items():
        patterns = entry.get('patterns') if isinstance(entry, dict) else None
        if not isinstance(patterns, list) or not patterns:
            raise ValueError('map %r has no pattern list' % map_id)
        rows = []
        for pattern in patterns:
            recordings = pattern.get('recordings')
            if not isinstance(recordings, list) or not recordings:
                raise ValueError('map %r has an empty pattern' % map_id)
            if not all(isinstance(name, str) and name for name in recordings):
                raise ValueError('map %r has a non-string recording name' % map_id)
            count = pattern.get('count')
            if count is not None and count != len(recordings):
                raise ValueError('map %r pattern count %r != %d recordings'
                                 % (map_id, count, len(recordings)))
            rows.append(dict(recordings=list(recordings), count=len(recordings)))
        table[map_id] = rows
    return table


def load_routes(directory=ROUTES_DIR):
    """session name (file stem) -> {'map', 'route', 'arrivals', 'aim'}."""
    directory = Path(directory)
    sessions = {}
    for path in sorted(directory.glob('*.json')):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError('%s is not an object' % path.name)
        map_id = data.get('map')
        route = data.get('route')
        if not isinstance(map_id, str) or not map_id:
            raise ValueError('%s has no map id' % path.name)
        if not isinstance(route, list):
            raise ValueError('%s has no route list' % path.name)
        clean = []
        for point in route:
            if not all(finite(point.get(key)) for key in ('t', 'x', 'z')):
                raise ValueError('%s has a non-finite route point' % path.name)
            clean.append(dict(t=float(point['t']), x=float(point['x']),
                              z=float(point['z']), yaw=float(point.get('yaw') or 0.0)))
        clean.sort(key=lambda p: p['t'])
        if len(clean) > 1 and clean[-1]['t'] <= clean[0]['t']:
            raise ValueError('%s route has no time span' % path.name)
        arrivals = []
        for row in data.get('arrivals') or []:
            if all(finite(row.get(key)) for key in ('t', 'x', 'z')):
                arrivals.append(dict(t=float(row['t']), x=float(row['x']), z=float(row['z'])))
        arrivals.sort(key=lambda p: p['t'])
        sessions[path.stem] = dict(session=data.get('session') or path.stem, map=map_id,
                                   route=clean, arrivals=arrivals,
                                   aim=len(data.get('aim') or []), source=path.name)
    if not sessions:
        raise ValueError('%s holds no route tables' % directory)
    return sessions


def path_length(route):
    return sum(math.dist((a['x'], a['z']), (b['x'], b['z'])) for a, b in zip(route, route[1:]))


def bearing(a, b):
    return math.degrees(math.atan2(b['x'] - a['x'], b['z'] - a['z'])) % 360.0


def signed_delta(new, old):
    return (new - old + 180.0) % 360.0 - 180.0


def step_speeds(route):
    speeds = []
    for a, b in zip(route, route[1:]):
        dt = b['t'] - a['t']
        if dt <= 0:
            continue
        speeds.append(math.dist((a['x'], a['z']), (b['x'], b['z'])) / dt)
    return speeds


def moving_fraction(route):
    speeds = step_speeds(route)
    if not speeds:
        return 0.0
    return sum(1 for s in speeds if s >= MOVING_EPS) / len(speeds)


def stop_runs(route):
    """Longest run of consecutive near-stationary samples (possible stranding)."""
    speeds = step_speeds(route)
    longest = run = 0
    for speed in speeds:
        run = run + 1 if speed < STOP_EPS else 0
        longest = max(longest, run)
    return longest


def _perpendicular(p, a, b):
    ax, az, bx, bz = a['x'], a['z'], b['x'], b['z']
    px, pz = p['x'], p['z']
    dx, dz = bx - ax, bz - az
    norm = math.hypot(dx, dz)
    if norm < 1e-9:
        return math.hypot(px - ax, pz - az)
    return abs(dz * px - dx * pz + bx * az - bz * ax) / norm


def simplify(route, tolerance):
    """Ramer-Douglas-Peucker on (x, z); keeps the first and last samples."""
    if len(route) < 3:
        return list(route)
    keep = {0, len(route) - 1}
    stack = [(0, len(route) - 1)]
    while stack:
        first, last = stack.pop()
        worst, index = -1.0, None
        for i in range(first + 1, last):
            distance = _perpendicular(route[i], route[first], route[last])
            if distance > worst:
                worst, index = distance, i
        if index is not None and worst > tolerance:
            keep.add(index)
            stack.append((first, index))
            stack.append((index, last))
    return [route[i] for i in sorted(keep)]


def key_turns(route, min_turn_deg=TURN_MIN_DEG, tolerance=TURN_TOLERANCE):
    """Heading changes at simplified-polyline vertices, |deg| >= min_turn_deg.

    Simplification first is what makes the word "turn" mean something: raw 5 s
    samples carry a few degrees of paddle noise, and summing per-step deltas over
    a long window produced several hundred bogus degrees in an earlier revision.
    """
    simple = simplify(route, tolerance)
    events = []
    for a, b, c in zip(simple, simple[1:], simple[2:]):
        incoming, outgoing = bearing(a, b), bearing(b, c)
        if math.dist((b['x'], b['z']), (c['x'], c['z'])) < MOVING_EPS * max(c['t'] - b['t'], 1e-9):
            continue
        delta = signed_delta(outgoing, incoming)
        if abs(delta) < min_turn_deg:
            continue
        events.append(dict(t_start=b['t'], t_end=b['t'], t_peak=b['t'],
                           deg=round(delta, 1), peak_deg=round(delta, 1),
                           x=b['x'], z=b['z']))
    return events


def quantiles(values, qs=(0.5, 0.9, 0.95, 1.0)):
    if not values:
        return {('p%g' % (q * 100)): None for q in qs}
    ordered = sorted(values)
    out = {}
    for q in qs:
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        out['p%g' % (q * 100)] = round(ordered[index], 3)
    return out


def arrival_bins(arrivals, edges=ARRIVAL_BINS):
    bins, labels = [], []
    lower = 0.0
    for upper in edges:
        labels.append('%g-%gs' % (lower, upper))
        bins.append(sum(1 for a in arrivals if lower <= a['t'] < upper))
        lower = upper
    labels.append('>%gs' % edges[-1])
    bins.append(sum(1 for a in arrivals if a['t'] >= edges[-1]))
    return dict(zip(labels, bins))


def nearest_route_point(route, t):
    if not route:
        return None
    return min(route, key=lambda p: abs(p['t'] - t))


def session_stats(route, arrivals):
    speeds = step_speeds(route)
    moving = [s for s in speeds if s >= MOVING_EPS]
    turns = key_turns(route)
    big = [e for e in turns if abs(e['deg']) >= 45.0]
    distances = []
    for arrival in arrivals:
        own = nearest_route_point(route, arrival['t'])
        if own is not None:
            distances.append(math.dist((own['x'], own['z']), (arrival['x'], arrival['z'])))
    return dict(
        samples=len(route),
        duration_s=round(route[-1]['t'] - route[0]['t'], 1) if route else 0.0,
        path_units=round(path_length(route), 1),
        mean_speed_units_s=round(path_length(route) / max(route[-1]['t'] - route[0]['t'], 1e-9), 3),
        median_moving_speed=round(statistics.median(moving), 3) if moving else 0.0,
        moving_fraction=round(moving_fraction(route), 3),
        longest_stop_samples=stop_runs(route),
        turns=len(turns),
        turns_ge_45=len(big),
        top_turns=[dict(t=round(e['t_peak'], 1), x=round(e['x'], 1), z=round(e['z'], 1),
                        deg=e['deg']) for e in
                   sorted(big, key=lambda e: -abs(e['deg']))[:3]],
        arrivals=len(arrivals),
        arrival_first_s=round(arrivals[0]['t'], 1) if arrivals else None,
        arrival_median_s=round(statistics.median([a['t'] for a in arrivals]), 1) if arrivals else None,
        arrival_last_s=round(arrivals[-1]['t'], 1) if arrivals else None,
        arrival_bins=arrival_bins(arrivals),
        arrival_distance=quantiles(distances),
    )


def points(route):
    return [(p['x'], p['z']) for p in route]


def hausdorff(first, second):
    """Symmetric Hausdorff distance in world units between two point sets."""
    a, b = points(first), points(second)
    if not a or not b:
        raise ValueError('hausdorff needs two non-empty point sets')

    def one_way(source, target):
        worst = 0.0
        for x, z in source:
            best = min((x - tx) ** 2 + (z - tz) ** 2 for tx, tz in target)
            worst = max(worst, math.sqrt(best))
        return worst

    return max(one_way(a, b), one_way(b, a))


def truncate(route, limit):
    """Route cut at `limit` seconds, with the crossing point appended."""
    if not route:
        return []
    if route[-1]['t'] <= limit:
        return list(route)
    cut = [p for p in route if p['t'] <= limit]
    for a, b in zip(route, route[1:]):
        if a['t'] <= limit < b['t'] and b['t'] > a['t']:
            fraction = (limit - a['t']) / (b['t'] - a['t'])
            cut.append(dict(t=limit, x=a['x'] + fraction * (b['x'] - a['x']),
                            z=a['z'] + fraction * (b['z'] - a['z']), yaw=a['yaw']))
            break
    return cut


def common_ends(routes):
    return min(route[-1]['t'] for route in routes) if routes else 0.0


def consensus(routes, step=STEP_S):
    """Per-time median position across routes; sd reported where n >= 2."""
    if not routes:
        return []
    limit = min(route[-1]['t'] for route in routes)
    rows = []
    index = 0
    while index * step <= limit:
        t = index * step
        xs, zs = [], []
        for route in routes:
            nearest = min(route, key=lambda p: abs(p['t'] - t))
            xs.append(nearest['x'])
            zs.append(nearest['z'])
        row = dict(t=round(t, 1), n=len(xs), x=round(statistics.median(xs), 1),
                   z=round(statistics.median(zs), 1))
        if len(xs) >= 2:
            row['sd'] = round(math.hypot(statistics.pstdev(xs), statistics.pstdev(zs)), 1)
        rows.append(row)
        index += 1
    return rows


def dispersion(consensus_rows):
    sds = [row['sd'] for row in consensus_rows if 'sd' in row]
    if not sds:
        return dict(available=False)
    return dict(available=True, samples=len(sds),
                median_sd=round(statistics.median(sds), 1),
                p95_sd=quantiles(sds)['p95'],
                max_sd=round(max(sds), 1))


def load_signatures(path=SIG_CORPUS, opening=SIG_OPENING, coarse=SIG_COARSE):
    """Opening enemy-marker signature per session, exactly as the clusterer builds it.

    Reads only `truth.npy` and `session_ids.npy` out of the merged corpus zip (about
    15 MB of members); the 22 GB heatmap member is never touched. numpy is imported
    here so the rest of this report runs without it.
    """
    import io
    import zipfile

    import numpy as np

    archive = zipfile.ZipFile(Path(path))
    truth = np.load(io.BytesIO(archive.read('truth.npy')), allow_pickle=True)
    ids = np.load(io.BytesIO(archive.read('session_ids.npy')), allow_pickle=True).astype(str)
    signatures, firsts = {}, {}
    for session in np.unique(ids):
        rows = truth[ids == session][:opening]
        cells = np.zeros(coarse * coarse, dtype=np.float64)
        marker_samples = []
        for index, sample in enumerate(rows):
            count = 0
            for row in sample:
                if np.isnan(row[0]) or int(row[0]) != 2:
                    continue
                cell = (min(int(float(row[2]) * coarse), coarse - 1) * coarse
                        + min(int(float(row[1]) * coarse), coarse - 1))
                cells[cell] += 1.0
                count += 1
            if count:
                marker_samples.append((index, count))
        total = cells.sum()
        signatures[session] = cells / total if total else cells
        firsts[session] = dict(window_samples=len(rows),
                              samples_with_markers=len(marker_samples),
                              first_marker_sample=marker_samples[0][0] if marker_samples else None,
                              markers=sum(count for _, count in marker_samples),
                              first_sample_with_3=next((i for i, c in marker_samples if c >= 3), None),
                              first_sample_with_8=next((i for i, c in marker_samples if c >= 8), None))
    return signatures, firsts


def signature_report(patterns, signatures, threshold=SIG_THRESHOLD):
    """Per-map signature separation plus whether the published grouping reproduces."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cluster_spawn_patterns import cosine_distance, group

    distances = {(a, b): round(cosine_distance(signatures[a], signatures[b]), 3)
                 for a in signatures for b in signatures if a < b}
    reproduced = sorted(sorted(g) for map_id, rows in patterns.items()
                        for g in group({n: signatures[n] for row in rows for n in row['recordings']
                                        if n in signatures}, threshold))
    published = sorted(sorted(row['recordings']) for rows in patterns.values() for row in rows)
    per_map = {}
    for map_id, rows in patterns.items():
        groups = [[n for n in row['recordings'] if n in signatures] for row in rows]
        intra = [distances[tuple(sorted((a, b)))] for members in groups
                 for a, b in itertools.combinations(members, 2)]
        cross = [distances[tuple(sorted((a, b)))] for left, right in itertools.combinations(groups, 2)
                 for a in left for b in right]
        per_map[map_id] = dict(intra_max=max(intra) if intra else None,
                               cross_min=min(cross) if cross else None,
                               margin=round(min(cross) - max(intra), 3) if intra and cross else None,
                               clusters=len(groups),
                               members=[len(g) for g in groups])
    return dict(threshold=threshold, matches_published=reproduced == published,
                grouping=reproduced, published=published, per_map=per_map,
                distances={'%s|%s' % k: v for k, v in sorted(distances.items())})


def corridor_distance(route, x, z):
    """Shortest distance from a point to a polyline (any time on it)."""
    best = math.inf
    for a, b in zip(route, route[1:]):
        dx, dz = b['x'] - a['x'], b['z'] - a['z']
        length = dx * dx + dz * dz
        if length <= 0:
            best = min(best, math.hypot(x - a['x'], z - a['z']))
            continue
        fraction = max(0.0, min(1.0, ((x - a['x']) * dx + (z - a['z']) * dz) / length))
        px, pz = a['x'] + fraction * dx, a['z'] + fraction * dz
        best = min(best, math.hypot(x - px, z - pz))
    return best


def leave_one_out(sessions, patterns, step=STEP_S):
    """Two baselines a waypoint policy must beat, measured out of sample.

    For every member of a pattern with at least two sessions, the other members
    form the reference. The held-out session is then scored two ways:

      clock_aligned  distance to the reference position at the same elapsed time
      corridor       distance to the nearest point anywhere on the reference path

    A large gap between them means the sessions walk the same water at different
    times, so a clock-indexed route is the wrong shape for a waypoint policy even
    though the corridor is tight.
    """
    out = {}
    for map_id, rows in patterns.items():
        for index, row in enumerate(rows):
            present = [sessions[n] for n in row['recordings'] if n in sessions]
            if len(present) < 2:
                continue
            for held in present:
                others = [s for s in present if s is not held]
                reference = consensus([s['route'] for s in others], step)
                if len(reference) < 2:
                    continue
                aligned, corridor = [], []
                for point in held['route']:
                    if point['t'] > reference[-1]['t']:
                        continue
                    nearest = min(reference, key=lambda r: abs(r['t'] - point['t']))
                    if abs(nearest['t'] - point['t']) > step / 2 + 1e-9:
                        continue
                    aligned.append(math.hypot(point['x'] - nearest['x'], point['z'] - nearest['z']))
                    corridor.append(corridor_distance(
                        [dict(x=r['x'], z=r['z']) for r in reference], point['x'], point['z']))
                if not aligned:
                    continue
                out['%s/p%d/%s' % (map_id, index, held['source'])] = dict(
                    samples=len(aligned),
                    clock_aligned=quantiles(aligned),
                    corridor=quantiles(corridor),
                    within_50=round(sum(1 for v in aligned if v <= 50.0) / len(aligned), 3),
                    within_50_corridor=round(sum(1 for v in corridor if v <= 50.0) / len(corridor), 3))
    return out


def package(sessions, patterns):
    """Join sessions to patterns; sessions absent from the table stay unclustered."""
    owner = {}
    for map_id, rows in patterns.items():
        for index, row in enumerate(rows):
            for name in row['recordings']:
                if name in owner and owner[name][1] != index:
                    raise ValueError('%s is listed under two patterns of %s' % (name, map_id))
                owner[name] = (map_id, index)
    report = dict(schema_version=1, step_s=STEP_S, maps={}, unclustered=[],
                  missing_files=[], notes=[
                      'distances are replay world units; the metre scale is disputed',
                      'enemy arrivals are first Position-stream sightings, not spawn ticks',
                      'intra/cross pattern Hausdorff is a separability bound, not accuracy',
                  ])
    for name, session in sorted(sessions.items()):
        where = owner.get(name)
        if where is None:
            report['unclustered'].append(dict(session=name, map=session['map']))
            continue
        map_id, index = where
        if session['map'] != map_id:
            raise ValueError('%s is under %s in the pattern table but %s in its route file'
                             % (name, map_id, session['map']))
    for map_id, rows in sorted(patterns.items()):
        entry = dict(patterns=[], sessions_with_routes=[], missing=[])
        intra_full, intra_early = [], []
        consensuses, ends = [], []
        for index, row in enumerate(rows):
            present = [sessions[n] for n in row['recordings'] if n in sessions]
            absent = [n for n in row['recordings'] if n not in sessions]
            entry['missing'].extend(absent)
            routes = [s['route'] for s in present]
            end = common_ends(routes)
            ends.append(end)
            # Pairwise distances are measured over the members' common span, so a
            # session that simply ended earlier cannot inflate the spread.
            pairwise, pairwise_early = [], []
            for left in range(len(routes)):
                for right in range(left + 1, len(routes)):
                    pairwise.append(round(hausdorff(truncate(routes[left], end),
                                                    truncate(routes[right], end)), 1))
                    if end > EARLY_WINDOW_S:
                        pairwise_early.append(round(hausdorff(
                            truncate(routes[left], EARLY_WINDOW_S),
                            truncate(routes[right], EARLY_WINDOW_S)), 1))
            intra_full.extend(pairwise)
            intra_early.extend(pairwise_early)
            merged = consensus(routes)
            consensuses.append(merged)
            steps = [round(math.dist((a['x'], a['z']), (b['x'], b['z'])), 1)
                     for a, b in zip(merged, merged[1:])]
            entry['patterns'].append(dict(
                index=index, recordings=row['recordings'], with_routes=len(present),
                missing=absent, pairwise_hausdorff=pairwise,
                pairwise_hausdorff_early=pairwise_early,
                dispersion=dispersion(merged),
                median_step_units=round(statistics.median(steps), 1) if steps else None,
                consensus_samples=len(merged), consensus_end=end,
                consensus_turns=[dict(t=e['t_peak'], x=round(e['x'], 1), z=round(e['z'], 1),
                                      deg=e['deg']) for e in
                                 sorted(key_turns([dict(p) for p in merged]),
                                        key=lambda e: -abs(e['deg']))[:6]],
                sessions=[dict(session=s['session'] or s['source'],
                               path_units=round(path_length(s['route']), 1),
                               duration_s=round(s['route'][-1]['t'] - s['route'][0]['t'], 1))
                          for s in present]))
            entry['sessions_with_routes'].extend(s['source'] for s in present)
        cross, cross_early = [], []
        for left in range(len(consensuses)):
            for right in range(left + 1, len(consensuses)):
                if len(consensuses[left]) < 2 or len(consensuses[right]) < 2:
                    continue
                window = min(ends[left], ends[right])
                cross.append(round(hausdorff(
                    [dict(x=r['x'], z=r['z']) for r in consensuses[left] if r['t'] <= window],
                    [dict(x=r['x'], z=r['z']) for r in consensuses[right] if r['t'] <= window]), 1))
                early = min(window, EARLY_WINDOW_S)
                if early > STEP_S:
                    cross_early.append(round(hausdorff(
                        [dict(x=r['x'], z=r['z']) for r in consensuses[left] if r['t'] <= early],
                        [dict(x=r['x'], z=r['z']) for r in consensuses[right] if r['t'] <= early]), 1))
        def ratio(cross_values, intra_values):
            if not cross_values or not intra_values or max(intra_values) <= 0:
                return None
            return round(min(cross_values) / max(intra_values), 2)
        entry['intra_hausdorff'] = dict(
            values=intra_full,
            median=round(statistics.median(intra_full), 1) if intra_full else None,
            max=max(intra_full) if intra_full else None)
        entry['intra_hausdorff_early'] = dict(
            values=intra_early,
            median=round(statistics.median(intra_early), 1) if intra_early else None,
            max=max(intra_early) if intra_early else None)
        entry['cross_hausdorff'] = dict(values=cross, minimum=min(cross) if cross else None)
        entry['cross_hausdorff_early'] = dict(values=cross_early,
                                              minimum=min(cross_early) if cross_early else None)
        entry['separation_ratio'] = ratio(cross, intra_full)
        entry['separation_ratio_early'] = ratio(cross_early, intra_early)
        entry['pattern_dispersion'] = (max(p['dispersion']['median_sd']
                                          for p in entry['patterns']
                                          if p['dispersion'].get('available'))
                                      if any(p['dispersion'].get('available')
                                             for p in entry['patterns']) else None)
        report['maps'][map_id] = entry
    report['leave_one_out'] = leave_one_out(sessions, patterns)
    report['per_session'] = {name: dict(map=session['map'], **session_stats(session['route'],
                                                                            session['arrivals']))
                             for name, session in sorted(sessions.items())}
    used = {name for rows in patterns.values() for row in rows for name in row['recordings']}
    report['sessions_without_pattern'] = sorted(n for n in sessions if n not in used)
    report['sessions_without_file'] = sorted(n for n in used if n not in sessions)
    report['totals'] = dict(maps=len(patterns), patterns=sum(len(r) for r in patterns.values()),
                            listed_sessions=len(used), sessions_with_files=len(sessions),
                            route_files_matched=len(used & set(sessions)))
    return report


def table(rows, header):
    widths = [max(len(str(row[i])) for row in [header] + rows) for i in range(len(header))]
    line = '  '.join(str(cell).ljust(width) for cell, width in zip(header, widths))
    out = [line, '  '.join('-' * width for width in widths)]
    for row in rows:
        out.append('  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)))
    return '\n'.join(out)


def render(report):
    blocks = []
    totals = report['totals']
    blocks.append('sessions: %d files, %d listed in the pattern table, %d joined'
                  % (totals['sessions_with_files'], totals['listed_sessions'],
                     totals['route_files_matched']))
    rows = []
    for map_id, entry in report['maps'].items():
        rows.append([map_id, len(entry['patterns']),
                     sum(p['with_routes'] for p in entry['patterns']),
                     len(entry['missing']),
                     entry['intra_hausdorff']['median'],
                     entry['intra_hausdorff']['max'],
                     entry['cross_hausdorff']['minimum'],
                     entry['separation_ratio']])
    blocks.append(table(rows, ['map', 'patterns', 'sessions', 'missing', 'intra_p50',
                              'intra_max', 'cross_min', 'ratio']))
    rows = []
    for map_id, entry in report['maps'].items():
        for pattern in entry['patterns']:
            turns = [t['t'] for t in pattern['consensus_turns']][:4]
            rows.append([map_id, pattern['index'], pattern['with_routes'],
                         ','.join(pattern['recordings'])[:44],
                         pattern['dispersion'].get('median_sd'),
                         pattern['dispersion'].get('p95_sd'),
                         pattern['median_step_units'],
                         pattern['consensus_end'],
                         ','.join(str(int(v)) for v in turns)])
    blocks.append('per pattern (sd = spread of same-pattern sessions about their median):')
    blocks.append(table(rows, ['map', 'pat', 'n', 'recordings', 'sd_p50', 'sd_p95',
                               'step_u', 'end_s', 'turn_times']))
    if report.get('leave_one_out'):
        rows = []
        for key, value in sorted(report['leave_one_out'].items()):
            rows.append([key, value['samples'], value['clock_aligned']['p50'],
                         value['clock_aligned']['p95'], value['corridor']['p50'],
                         value['corridor']['p95'], value['within_50'],
                         value['within_50_corridor']])
        blocks.append('leave-one-out: held-out session vs the other sessions of its pattern')
        blocks.append(table(rows, ['held_out', 'n', 'clk_p50', 'clk_p95', 'cor_p50',
                                   'cor_p95', 'clk<=50', 'cor<=50']))
    rows = []
    for name, stats in report['per_session'].items():
        rows.append([name, stats['duration_s'], stats['path_units'],
                     stats['median_moving_speed'], stats['turns_ge_45'],
                     stats['longest_stop_samples'], stats['arrivals'],
                     stats['arrival_first_s'], stats['arrival_median_s']])
    blocks.append(table(rows, ['session', 'dur_s', 'path_u', 'v_med', 'turns45', 'stop',
                               'arrivals', 'first_s', 'med_s']))
    bins = list(arrival_bins([]).keys())
    rows = []
    for map_id in report['maps']:
        totals_row = {key: 0 for key in bins}
        count = 0
        for name, stats in report['per_session'].items():
            if stats['map'] != map_id:
                continue
            count += 1
            for key, value in stats['arrival_bins'].items():
                totals_row[key] += value
        rows.append([map_id, count] + [totals_row[key] for key in bins])
    blocks.append('enemy first sightings per map (count, sessions):')
    blocks.append(table(rows, ['map', 'sessions'] + bins))
    signatures = report.get('signatures')
    if signatures:
        rows = [[map_id, value['clusters'], value['intra_max'], value['cross_min'],
                 value['margin']] for map_id, value in signatures['per_map'].items()]
        blocks.append('opening signature separation (cosine, threshold %.2f, published table '
                      'reproduced: %s):' % (signatures['threshold'],
                                             signatures['matches_published']))
        blocks.append(table(rows, ['map', 'patterns', 'intra_max', 'cross_min', 'margin']))
        window = report.get('signature_window') or {}
        per_session = window.get('per_session') or {}
        if per_session:
            rows = [[name, value['markers'], value['samples_with_markers'],
                     value['first_marker_sample'], value['first_sample_with_3']]
                    for name, value in per_session.items()]
            blocks.append('enemy markers inside the %s-sample opening window:'
                          % window.get('opening_samples'))
            blocks.append(table(rows, ['session', 'markers', 'samples',
                                       'first_marker', 'first_3']))
    return '\n\n'.join(blocks)


def build(routes_dir=ROUTES_DIR, patterns_path=PATTERNS_PATH):
    return package(load_routes(routes_dir), load_patterns(patterns_path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--routes-dir', default=str(ROUTES_DIR))
    ap.add_argument('--patterns', default=str(PATTERNS_PATH))
    ap.add_argument('--json', action='store_true', help='print the report as JSON')
    ap.add_argument('--out', default=None, help='also write the JSON report here')
    ap.add_argument('--signatures', action='store_true',
                    help='also compare opening enemy-marker signatures (needs the merged corpus)')
    ap.add_argument('--opening', type=int, default=SIG_OPENING,
                    help='samples of the battle that define a signature (default %d)' % SIG_OPENING)
    ap.add_argument('--threshold', type=float, default=SIG_THRESHOLD,
                    help='cosine distance under which two sessions share a pattern (default %.2f)'
                         % SIG_THRESHOLD)
    args = ap.parse_args(argv)
    report = build(args.routes_dir, args.patterns)
    if args.signatures:
        signatures, firsts = load_signatures(opening=args.opening)
        report['signatures'] = signature_report(load_patterns(args.patterns), signatures,
                                                args.threshold)
        report['signature_window'] = dict(opening_samples=args.opening,
                                          coarse=SIG_COARSE, per_session=firsts)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) if args.json else render(report)
    print(text)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                             sort_keys=True) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
