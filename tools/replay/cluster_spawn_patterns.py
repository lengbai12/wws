#!/usr/bin/env python3
"""How many spawn patterns has each Operations map actually shown us?

The owner's premise is that a map's enemies arrive from nearly fixed places, and his
warning is that a map can have two or three patterns while our sample count may cover
only some of them. So the question this answers is not "what is the pattern" but "how
much of each map have we seen", and it answers it from data rather than assumption.

Signature per recording: the enemy markers of the opening window, quantised into a
coarse grid and normalised. Two recordings of the same map run the same opening if their
signatures are close; a different spawn pattern moves the mass somewhere else.

The distance matrix is printed rather than a forced clustering, because the honest
question is whether the groups are real. A threshold groups them, and every group is
reported with its sample count so an under-covered pattern is visible rather than folded
into a neat-looking summary.

    ./.venv/bin/python tools/replay/cluster_spawn_patterns.py
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_replay_meta as meta_reader

HEATMAPS = "replays/parsed/datasets/heatmap-merged-14s-g176.npz"
# Marker classes in the truth table, in the order the corpus declares them.
ENEMY_CLASS = 2
COARSE = 8          # the grid cells a signature is built from (8x8 over the 176 grid)


def replay_path(session):
    hits = sorted(Path("replays").glob("%s*.wowsreplay" % session.split("_")[0]))
    return hits[0] if hits else None


def map_of(session):
    path = replay_path(session)
    if path is None:
        return "?"
    try:
        return (meta_reader.read_meta(path).get("mapName") or "?").replace("spaces/", "")
    except Exception:
        return "?"


def signature(truth, ids, opening):
    """One normalised coarse-grid histogram of opening enemy markers per session."""
    out = {}
    for session in np.unique(ids):
        rows = truth[ids == session][:opening]
        cells = np.zeros(COARSE * COARSE, dtype=np.float64)
        for sample in rows:
            for row in sample:
                if np.isnan(row[0]) or int(row[0]) != ENEMY_CLASS:
                    continue
                u, v = float(row[1]), float(row[2])
                column = min(int(u * COARSE), COARSE - 1)
                index = min(int(v * COARSE), COARSE - 1)
                cells[index * COARSE + column] += 1.0
        total = cells.sum()
        out[str(session)] = cells / total if total else cells
    return out


def cosine_distance(a, b):
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 1.0
    return 1.0 - float(a @ b) / denominator


def group(signatures, threshold):
    """Single linkage: two recordings are the same pattern if they are that close."""
    names = sorted(signatures)
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            name = parent[name] = parent[parent[name]]
        return name

    for a, b in itertools.combinations(names, 2):
        if cosine_distance(signatures[a], signatures[b]) <= threshold:
            parent[find(a)] = find(b)
    groups = {}
    for name in names:
        groups.setdefault(find(name), []).append(name)
    return list(groups.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heatmaps", default=HEATMAPS)
    ap.add_argument("--opening", type=int, default=60,
                    help="samples at the start of a battle that define the pattern "
                         "(the corpus samples about once a second)")
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="cosine distance below which two recordings count as one pattern")
    ap.add_argument("--json-out", default="replays/parsed/spawn-patterns.json")
    ap.add_argument("--exclude", default="",
                    help="comma separated session ids to leave out, e.g. low-tier "
                         "recordings that are still inside the merged corpus")
    args = ap.parse_args()

    archive = np.load(args.heatmaps, allow_pickle=True)
    truth = archive["truth"]
    ids = archive["session_ids"]
    excluded = {name.strip() for name in args.exclude.split(",") if name.strip()}
    if excluded:
        keep = np.array([str(value) not in excluded for value in ids])
        truth = truth[keep]
        ids = ids[keep]
        print("excluded %d recording(s): %s\n" % (len(excluded), ", ".join(sorted(excluded))))
    print("corpus %d samples, %d sessions, opening window %d samples, threshold %.2f\n"
          % (truth.shape[0], len(np.unique(ids)), args.opening, args.threshold))

    signatures = signature(truth, ids, args.opening)
    per_map = {}
    for session in signatures:
        per_map.setdefault(map_of(session), []).append(session)

    report = {}
    thin = []
    for name in sorted(per_map):
        sessions = sorted(per_map[name])
        print("=== %s : %d recording(s)" % (name, len(sessions)))
        if len(sessions) > 1:
            print("    pairwise cosine distance:")
            for a, b in itertools.combinations(sessions, 2):
                print("      %-34s %-34s %.3f"
                      % (a, b, cosine_distance(signatures[a], signatures[b])))
        groups = group({s: signatures[s] for s in sessions}, args.threshold)
        groups.sort(key=lambda g: (-len(g), g[0]))
        for index, members in enumerate(groups, 1):
            mark = "sample enough" if len(members) >= 3 else "UNDER-COVERED"
            if len(members) < 3:
                thin.append((name, index, members))
            print("    pattern %d: %d recording(s)  [%s]  %s"
                  % (index, len(members), mark, ", ".join(members)))
        report[name] = {"patterns": [{"recordings": m, "count": len(m)} for m in groups]}
        print()

    Path(args.json_out).write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print("wrote %s" % args.json_out)
    if thin:
        print("\n%d pattern(s) below three recordings - more recordings needed, and the "
              "map to record is whichever appears here:" % len(thin))
        for name, index, members in thin:
            print("   %s pattern %d (%d): %s" % (name, index, len(members), ", ".join(members)))
    else:
        print("\nevery observed pattern has at least three recordings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
