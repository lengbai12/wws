#!/usr/bin/env python3
"""Build the per-map route tables the runtime will follow.

The owner's premise is that an Operations map's enemies arrive from nearly fixed places,
so a battle is close to scripted - with the caveat that a map can have two or three
patterns and our samples may cover only some of them. This turns each recorded session
into the three things a scripted run needs, in one coordinate system (the map's world
frame, so nothing depends on the camera or the minimap's rotation):

  route      the recording player's own path, sampled every few seconds
  arrivals   where each enemy first appears, and when
  aim        where the player was pointing on the minimap, sampled over time

Sessions are keyed by name; which pattern each one belongs to comes from
cluster_spawn_patterns.py, so these tables are joined to that result rather than
re-deriving it (a second implementation of the same rule only proves the second one).

    ./.venv/bin/python tools/replay/build_route_tables.py --corpus-config … \
        --sessions 236226 236284 …
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_replay_meta as meta_reader

GAMEDATA = "state/toalba/gamedata/data/scripts_entity/entity_defs"
try:
    from wows_replay_parser import parse_replay
except ImportError:
    raise SystemExit("run this with state/toalba/venv/bin/python: the replay parser "
                     "lives in its own environment")

ROUTE_STEP_S = 5.0
AIM_CLASS = "aim"


def own_route(session):
    """The recording player's own stream, sampled: it is what our export carries."""
    path = Path("replays/parsed") / ("position-%s.jsonl" % session)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    points = [r for r in rows if r.get("event") == "position"]
    if not points:
        return []
    first = points[0]["clock"]
    route = []
    for point in points:
        if point["clock"] - first >= len(route) * ROUTE_STEP_S:
            route.append({"t": round(point["clock"] - first, 1),
                          "x": round(point["x"], 1), "z": round(point["z"], 1),
                          "yaw": round(point.get("yaw") or 0.0, 3)})
    return route


def arrivals(session):
    """First appearance of each enemy vehicle, from the parser's position stream.

    The enemy set comes from our export's roster (team_id differs from the recording
    player's); the parser's Position events carry every other vehicle in exactly that
    roster's vehicle_id space (verified: 56 of 56 match on a test session).
    """
    path = Path("replays/parsed") / ("position-%s.jsonl" % session)
    hits = sorted(Path("replays").glob("%s*.wowsreplay" % session.split("_")[0]))
    if not path.exists() or not hits:
        return []
    roster, session_row = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "roster":
            roster.append(row)
        elif row.get("event") == "session":
            session_row = row
    player = session_row.get("recording_player")
    own = next((r for r in roster if r.get("name") == player), None)
    if own is None:
        return []
    enemies = {r["vehicle_id"] for r in roster
               if r.get("team_id") != own.get("team_id")}
    replay = parse_replay(replay_path=str(hits[0]), gamedata_path=GAMEDATA)
    seen = {}
    for event in replay.events:
        if type(event).__name__ != "PositionEvent" or event.entity_id not in enemies:
            continue
        if event.entity_id not in seen:
            seen[event.entity_id] = {"t": round(float(event.timestamp), 1),
                                     "x": round(float(event.x), 1),
                                     "z": round(float(event.z), 1)}
    return sorted(seen.values(), key=lambda entry: entry["t"])


def aim_points(corpus, session):
    """Where the recording player was pointing, from the corpus' own aim channel."""
    truth = corpus["truth"]
    ids = corpus["session_ids"]
    mask = np.asarray([str(value) == session for value in ids])
    if not mask.any():
        return []
    rows = truth[mask]
    channel = list(corpus["classes"]).index(AIM_CLASS)
    grid = int(np.asarray(corpus["grid"]).ravel()[0])
    points = []
    for sample in rows:
        for row in sample:
            if np.isnan(row[0]) or int(row[0]) != channel:
                continue
            points.append({"u": round(float(row[1]), 4), "v": round(float(row[2]), 4)})
            break
    return points


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heatmaps",
                    default="replays/parsed/datasets/heatmap-merged-18s-g176.npz",
                    help="the merged HEATMAP corpus: it carries truth, classes, grid, "
                         "splits and session_ids. The images npz carries only the crops, "
                         "so pointing this at it fails on the first member.")
    ap.add_argument("--sessions", nargs="*", default=[],
                    help="session names; default is every session in the corpus")
    ap.add_argument("--out", default="replays/parsed/routes")
    args = ap.parse_args()

    archive = np.load(args.heatmaps, allow_pickle=False)
    # Only the small members: the images member is 20 GB and reading it at all (even for
    # its shape) pulls it into memory, which is how a run was lost.
    corpus = {key: archive[key] for key in
              ("truth", "session_ids", "classes", "grid", "splits")}
    sessions = args.sessions or sorted({str(value) for value in corpus["session_ids"]})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        replay = sorted(Path("replays").glob("%s*.wowsreplay" % session.split("_")[0]))
        mapping = "?"
        if replay:
            try:
                mapping = (meta_reader.read_meta(replay[0]).get("mapName") or "?").replace(
                    "spaces/", "")
            except Exception:
                pass
        table = {"session": session, "map": mapping,
                 "route": own_route(session),
                 "arrivals": arrivals(session),
                 "aim": aim_points(corpus, session)}
        (out / ("%s.json" % session)).write_text(json.dumps(table, indent=1) + "\n",
                                                 encoding="utf-8")
        print("  %-34s %-22s route=%4d arrivals=%3d aim=%4d"
              % (session, mapping, len(table["route"]), len(table["arrivals"]),
                 len(table["aim"])))
    print("wrote %s (%d session table(s))" % (out, len(sessions)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
