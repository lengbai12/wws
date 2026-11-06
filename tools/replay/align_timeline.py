#!/usr/bin/env python3
"""Fit replay-clock to video-time alignment from manually verified anchors.

Anchor CSV columns: video_time_s,replay_clock_s. Use visible, unambiguous
events in the recording (for example battle timer transitions) and the same
event's replay clock. Two anchors determine offset and rate; three or more make
residual/drift diagnostics useful.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def fit(anchors: list[tuple[float, float]]) -> dict:
    if len(anchors) < 2:
        raise ValueError("at least two anchors are required")
    xs = [x for x, _ in anchors]
    ys = [y for _, y in anchors]
    xbar, ybar = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("video_time_s anchors must not all be identical")
    rate = sum((x - xbar) * (y - ybar) for x, y in anchors) / denom
    offset = ybar - rate * xbar
    residuals = [y - (rate * x + offset) for x, y in anchors]
    rmse = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    r_squared = 1.0 if ss_tot == 0 and rmse == 0 else (
        1.0 - sum(r * r for r in residuals) / ss_tot if ss_tot else 0.0)
    return {
        "model": "replay_clock_s = rate * video_time_s + offset_s",
        "rate": rate, "offset_s": offset, "anchors": len(anchors),
        "rmse_s": rmse, "max_abs_residual_s": max(map(abs, residuals)),
        "r_squared": r_squared,
        "video_to_replay": {"rate": rate, "offset_s": offset},
        "replay_to_video": {"rate": 1.0 / rate, "offset_s": -offset / rate}
            if rate else None,
    }


def read_anchors(path: Path) -> list[tuple[float, float]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        required = {"video_time_s", "replay_clock_s"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError("anchor CSV needs video_time_s,replay_clock_s columns")
        return [(float(row["video_time_s"]), float(row["replay_clock_s"]))
                for row in rows]


def iter_event_records(path: Path, replay: str | None):
    """Yield event records, optionally selecting one replay from a combined JSONL.

    Exported replay streams put the replay filename on the session row and keep
    the following roster/event rows compact.  Track that session boundary here
    so alignment cannot accidentally apply one fit to several replays.
    """
    active: str | None = None
    with path.open(encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            if record.get("event") == "session":
                active = record.get("replay")
            if replay is None or active == replay:
                yield record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anchors", type=Path, help="CSV of paired video/replay timestamps")
    parser.add_argument("--out", type=Path, required=True, help="alignment JSON output")
    parser.add_argument("--events", type=Path, help="optional replay events JSONL to map onto video time")
    parser.add_argument("--aligned-out", type=Path, help="aligned JSONL (requires --events)")
    parser.add_argument("--replay", help="replay filename to select from a combined events JSONL")
    args = parser.parse_args()
    if bool(args.events) != bool(args.aligned_out):
        parser.error("--events and --aligned-out must be provided together")
    model = fit(read_anchors(args.anchors))
    model["source"] = {"anchors_csv": str(args.anchors)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    if args.events:
        assert args.aligned_out is not None
        rate, offset = model["rate"], model["offset_s"]
        args.aligned_out.parent.mkdir(parents=True, exist_ok=True)
        with args.aligned_out.open("w", encoding="utf-8") as dst:
            for record in iter_event_records(args.events, args.replay):
                if isinstance(record.get("clock"), (int, float)) and rate:
                    record["video_time_s"] = round((record["clock"] - offset) / rate, 3)
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"rate={model['rate']:.9f} offset={model['offset_s']:.3f}s "
          f"anchors={model['anchors']} rmse={model['rmse_s']:.3f}s "
          f"max_residual={model['max_abs_residual_s']:.3f}s")


if __name__ == "__main__":
    main()
