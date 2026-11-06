#!/usr/bin/env python3
"""Build event-centered WoWS candidate windows from aligned JSONL events."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from fsutil import write_jsonl


PRE = {"damage": 3.0, "death": 8.0, "consumable": 5.0, "target_change": 4.0}
POST = {"damage": 5.0, "death": 8.0, "consumable": 8.0, "target_change": 6.0}
WEIGHT = {"damage": 1.0, "death": 6.0, "consumable": 3.0, "target_change": 2.0}
KINDS = frozenset(PRE)


def read_events(path: Path) -> list[dict]:
    rows = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as src:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("event") in KINDS and isinstance(record.get("video_time_s"), (int, float)):
                identity = json.dumps(record, sort_keys=True, separators=(",", ":"))
                if identity in seen:
                    continue
                seen.add(identity)
                record["_line"] = line_number
                rows.append(record)
    return sorted(rows, key=lambda r: (float(r["video_time_s"]), r["_line"]))


def make_windows(events: list[dict], duration_s: float, max_duration_s: float = 18.0,
                 merge_gap_s: float = 1.0) -> list[dict]:
    intervals = []
    for event in events:
        kind = event["event"]
        time_s = float(event["video_time_s"])
        intervals.append({"start": max(0.0, time_s - PRE[kind]),
                          "end": min(duration_s, time_s + POST[kind]),
                          "event": event})
    windows = []
    current = None
    for interval in intervals:
        if current is None:
            current = {"start": interval["start"], "end": interval["end"],
                       "events": [interval["event"]]}
            continue
        end = max(current["end"], interval["end"])
        if interval["start"] <= current["end"] + merge_gap_s and end - current["start"] <= max_duration_s:
            current["end"] = end
            current["events"].append(interval["event"])
        else:
            windows.append(current)
            current = {"start": interval["start"], "end": interval["end"],
                       "events": [interval["event"]]}
    if current is not None:
        windows.append(current)
    output = []
    for index, window in enumerate(windows):
        counts = collections.Counter(e["event"] for e in window["events"])
        output.append({
            "candidate_id": f"candidate-{index:04d}",
            "start_s": round(window["start"], 3),
            "end_s": round(window["end"], 3),
            "duration_s": round(window["end"] - window["start"], 3),
            "score": round(sum(WEIGHT[e["event"]] for e in window["events"]), 3),
            "event_counts": dict(sorted(counts.items())),
            "event_count": len(window["events"]),
            "trigger_events": window["events"],
        })
    return output


def select_preview(windows: list[dict], limit: int, min_separation_s: float = 20.0) -> list[dict]:
    chosen = []
    for candidate in sorted(windows, key=lambda row: (-row["score"], row["start_s"])):
        center = (candidate["start_s"] + candidate["end_s"]) / 2
        if all(abs(center - (row["start_s"] + row["end_s"]) / 2) >= min_separation_s
                   for row in chosen):
            chosen.append(candidate)
            if len(chosen) == limit:
                break
    return sorted(chosen, key=lambda row: row["start_s"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview-out", type=Path, required=True)
    parser.add_argument("--preview-limit", type=int, default=12)
    parser.add_argument("--max-duration-s", type=float, default=18.0)
    parser.add_argument("--merge-gap-s", type=float, default=1.0)
    args = parser.parse_args()
    events = read_events(args.events)
    windows = make_windows(events, args.duration_s, args.max_duration_s, args.merge_gap_s)
    preview = select_preview(windows, args.preview_limit)
    write_jsonl(args.out, windows)
    write_jsonl(args.preview_out, preview)
    summary = {
        "source": str(args.events), "duration_s": args.duration_s,
        "input_events": len(events),
        "input_event_counts": dict(collections.Counter(e["event"] for e in events)),
        "candidate_windows": len(windows), "preview_windows": len(preview),
        "max_duration_s": args.max_duration_s, "merge_gap_s": args.merge_gap_s,
        "context_pre_s": PRE, "context_post_s": POST,
        "preview_ids": [row["candidate_id"] for row in preview],
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
