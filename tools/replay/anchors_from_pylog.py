"""Build alignment anchors for a recording from the client's own death log.

A replay clock has to be tied to the video timeline before any label is placed.
The client writes one line per ship death while a replay plays:

    [Vehicle] id = 1448696, teamId = 1, typeDeath = 17;

and that line carries the wall-clock time of the rendered death. The replay file
carries the same deaths on its own clock, so pairing them gives anchors of the
form (video_time_s, replay_clock_s) that align_timeline.py turns into a linear
fit. No manual marking is involved, which is why a whole recording can be aligned.

Video time is stated in the CONTAINER timeline, because that is the timeline
ffmpeg seeks on:

    video_time_s = death_wall - wall_of_container_zero
    wall_of_container_zero = wall_start + since_start_s(first frame) - first_pts_s

Both terms come from the same recording directory, so the anchors are
reproducible from stored files alone.

Usage:
  anchors_from_pylog.py --pylog profile/python.log \
      --recording recordings/<session> --events replays/parsed/events.jsonl \
      --session <replay file name> --out replays/parsed/alignment/<session>.anchors.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

MIN_ANCHORS = 3

LOG_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}):")
PLAY_LINE = re.compile(r"\[BattleReplay\] play, \('([^']+)'")
DEATH_LINE = re.compile(
    r"\[Vehicle\] id = (\d+), teamId = (\d+), typeDeath = (\d+);")

ANCHOR_COLUMNS = ("video_time_s", "replay_clock_s", "anchor_kind",
                  "victim", "death_type", "source")


def parse_log_line_time(text):
    """Return the wall-clock epoch of a log line, or None when it has no stamp."""
    match = LOG_TIME.match(text)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()


def read_pylog(path):
    """Return (playback_starts, deaths) from a client log.

    ``playback_starts`` is a list of (wall_epoch, replay_path). ``deaths`` is a
    list of (wall_epoch, vehicle_id, team_id, death_type).
    """
    starts = []
    deaths = []
    for text in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stamp = parse_log_line_time(text)
        if stamp is None:
            continue
        play = PLAY_LINE.search(text)
        if play:
            starts.append((stamp, play.group(1)))
            continue
        death = DEATH_LINE.search(text)
        if death:
            deaths.append((stamp, int(death.group(1)), int(death.group(2)),
                           int(death.group(3))))
    return starts, deaths


def container_zero_wall(recording):
    """Return the wall-clock epoch of the video's t=0.

    ``first_pts_s`` is where the container places its first frame, which is not
    where the recorder's own clock started, so the first frame's arrival offset
    corrects for the difference.
    """
    session = json.loads((Path(recording) / "session.json").read_text(encoding="utf-8"))
    wall_start = session.get("wall_start")
    first_pts = session.get("first_pts_s")
    if wall_start is None or first_pts is None:
        raise SystemExit("%s/session.json needs wall_start and first_pts_s"
                         % recording)
    with (Path(recording) / "frames.jsonl").open(encoding="utf-8") as handle:
        first_frame = json.loads(handle.readline())
    return wall_start + first_frame["since_start_ns"] / 1e9 - first_pts, session


def candidate_windows(starts, replay_name, zero_wall, span_s, budget_s=180.0):
    """Return the playback windows that could belong to this recording.

    The play line sits after the video's t=0 when the recorder starts before the
    game, and before it when the recorder attaches to a game that already runs,
    so neither direction can be assumed. Every playback of this replay within
    ``budget_s`` of t=0 is offered, and the caller decides by how well the deaths
    pair.

    A window ends at the next playback start of any replay, because that is when
    the client stopped playing the first one.
    """
    matching = [(stamp, path) for stamp, path in starts if Path(path).name == replay_name]
    if not matching:
        raise SystemExit("no [BattleReplay] play line for %s in the log" % replay_name)
    near = [item for item in matching if abs(item[0] - zero_wall) <= budget_s]
    pool = near or [min(matching, key=lambda item: abs(item[0] - zero_wall))]
    every_start = [stamp for stamp, _ in starts]
    windows = []
    for stamp, path in pool:
        later = [other for other in every_start if other > stamp]
        end = min(later) if later else zero_wall + span_s + 1.0
        windows.append({"start": stamp, "end": end, "path": path,
                        "offset_from_zero": stamp - zero_wall})
    windows.sort(key=lambda item: abs(item["offset_from_zero"]))
    return windows


def session_deaths(events_path, replay_name):
    """Return [{'clock','victim','death_type'}] for one replay of an event file.

    Rows of the event file are grouped by a leading ``session`` row, so the
    current replay is tracked rather than assumed.
    """
    deaths = []
    current = None
    with Path(events_path).open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "session":
                current = row.get("replay")
                continue
            if current != replay_name or row.get("event") != "death":
                continue
            deaths.append({"clock": float(row["clock"]),
                           "victim": int(row["victim"]),
                           "death_type": int(row["death_type"])})
    if not deaths:
        raise SystemExit("no death rows for %s in %s" % (replay_name, events_path))
    return deaths


def pair_deaths(log_deaths, replay_deaths):
    """Pair logged deaths with replay deaths by identity, in order.

    Returns (anchors, unpaired_log, unpaired_replay). A logged death takes the
    earliest unconsumed replay death with the same victim and death type, so a
    repeated identity cannot silently shift every later anchor.
    """
    by_key = {}
    for index, death in enumerate(replay_deaths):
        by_key.setdefault((death["victim"], death["death_type"]), []).append(index)
    consumed = set()
    anchors = []
    unpaired_log = []
    for stamp, victim, _team, death_type in log_deaths:
        queue = by_key.get((victim, death_type)) or []
        match = next((index for index in queue if index not in consumed), None)
        if match is None:
            unpaired_log.append((stamp, victim, death_type))
            continue
        consumed.add(match)
        anchors.append((stamp, victim, death_type, replay_deaths[match]["clock"]))
    unpaired_replay = [death for index, death in enumerate(replay_deaths)
                       if index not in consumed]
    return anchors, unpaired_log, unpaired_replay


def build_anchors(pylog, recording, events, session, span_s=None, budget_s=180.0):
    """Return (rows, report) for one recording.

    Several playbacks of the same replay can sit near one recording, so every
    candidate window is paired and the one that explains the most deaths wins. A
    tie goes to the playback nearest the video's t=0.
    """
    zero_wall, session_meta = container_zero_wall(recording)
    if span_s is None:
        span_s = float(session_meta.get("elapsed_s") or session_meta.get("duration_s") or 0.0)
    starts, all_deaths = read_pylog(pylog)
    replay_deaths = session_deaths(events, session)
    windows = candidate_windows(starts, session, zero_wall, span_s, budget_s)
    evaluations = []
    for window in windows:
        window_deaths = [death for death in all_deaths
                         if window["start"] <= death[0] <= window["end"]]
        anchors, unpaired_log, unpaired_replay = pair_deaths(window_deaths, replay_deaths)
        evaluations.append({
            "window": window,
            "logged_deaths_in_window": len(window_deaths),
            "anchors": anchors,
            "unpaired_logged": unpaired_log,
            "unpaired_replay": unpaired_replay,
        })
    best = max(evaluations, key=lambda item: (len(item["anchors"]),
                                              -abs(item["window"]["offset_from_zero"])))
    rows = []
    for stamp, victim, death_type, clock in best["anchors"]:
        rows.append({
            "video_time_s": stamp - zero_wall,
            "replay_clock_s": clock,
            "anchor_kind": "death",
            "victim": victim,
            "death_type": death_type,
            "source": Path(pylog).name,
        })
    rows.sort(key=lambda row: row["video_time_s"])
    report = {
        "session": session,
        "container_zero_wall": zero_wall,
        "playback_window_wall": [best["window"]["start"], best["window"]["end"]],
        "playback_offset_from_zero_s": best["window"]["offset_from_zero"],
        "candidate_windows": [
            {"start": item["window"]["start"],
             "offset_from_zero_s": item["window"]["offset_from_zero"],
             "logged_deaths": item["logged_deaths_in_window"],
             "anchors": len(item["anchors"])}
            for item in evaluations
        ],
        "logged_deaths_in_window": best["logged_deaths_in_window"],
        "replay_deaths": len(replay_deaths),
        "anchors": len(rows),
        "unpaired_logged": [{"wall": stamp, "victim": victim, "death_type": kind}
                            for stamp, victim, kind in best["unpaired_logged"]],
        "unpaired_replay": best["unpaired_replay"],
    }
    return rows, report


def write_anchors(path, rows):
    """Write the anchor CSV that align_timeline.py consumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(ANCHOR_COLUMNS)]
    for row in rows:
        lines.append("%.6f,%.3f,%s,%d,%d,%s" % (
            row["video_time_s"], row["replay_clock_s"], row["anchor_kind"],
            row["victim"], row["death_type"], row["source"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pylog", type=Path, required=True,
                        help="client log that holds the death lines")
    parser.add_argument("--recording", type=Path, required=True,
                        help="recordings/<session> directory with session.json and frames.jsonl")
    parser.add_argument("--events", type=Path, required=True,
                        help="exported event stream that holds the replay deaths")
    parser.add_argument("--session", required=True,
                        help="replay file name, as it appears in the log and events")
    parser.add_argument("--out", type=Path, help="anchor CSV to write")
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    parser.add_argument("--min-anchors", type=int, default=MIN_ANCHORS,
                        help="refuse to write fewer anchors than this")
    args = parser.parse_args(argv)

    rows, report = build_anchors(args.pylog, args.recording, args.events, args.session)
    print("container t=0 at wall %.6f" % report["container_zero_wall"])
    print("playback window %.3f..%.3f  logged deaths %d  replay deaths %d  paired %d"
          % (report["playback_window_wall"][0], report["playback_window_wall"][1],
             report["logged_deaths_in_window"], report["replay_deaths"],
             report["anchors"]))
    if report["unpaired_logged"]:
        print("unpaired logged deaths: %d (first %s)"
              % (len(report["unpaired_logged"]), report["unpaired_logged"][0]))
    if report["unpaired_replay"]:
        print("unpaired replay deaths: %d" % len(report["unpaired_replay"]))
    if report["anchors"] < args.min_anchors:
        print("refusing to write: %d anchors is below --min-anchors %d"
              % (report["anchors"], args.min_anchors))
        return 1
    if args.out:
        print("wrote %s (%d anchors)" % (write_anchors(args.out, rows), len(rows)))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
