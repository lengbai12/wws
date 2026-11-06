#!/usr/bin/env python3
"""Build a non-destructive, auditable WoWS replay/video dataset manifest.

This command deliberately keeps replay-derived labels separate from human visual
labels.  It can therefore run when the full recording is unavailable: the
manifest and review queue are still useful, while training remains blocked until
real frames have been reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from cut_candidates import make_windows, read_events, select_preview
from fsutil import sha256_or_none, load_jsonl, write_jsonl, write_json



def fold_rows(rows: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    output = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output


def focus_target_changes(event_path: Path, owner: str) -> list[dict]:
    """Emit target-focus changes from the owner's damage stream.

    This is replay provenance for selecting review windows, not a visual target
    label.  It is preferable to using the track endpoint here because the
    exported track currently contains only the recording player's own entity.
    """
    rows = fold_rows(load_jsonl(event_path))
    damage = [
        row for row in rows
        if row.get("event") == "damage"
        and row.get("killer_name") == owner
        and isinstance(row.get("victim"), int)
        and isinstance(row.get("video_time_s"), (int, float))
    ]
    damage.sort(key=lambda row: (float(row["video_time_s"]), row.get("victim", 0)))
    changes = []
    previous = None
    for row in damage:
        victim = row["victim"]
        if victim == previous:
            continue
        previous = victim
        changes.append({
            "event": "target_change",
            "change": "focus_target",
            "entity": victim,
            "target_name": row.get("victim_name"),
            "owner": owner,
            "clock": float(row.get("clock", 0.0)),
            "video_time_s": float(row["video_time_s"]),
            "_source": str(event_path),
        })
    return changes


def session_meta(summary_path: Path) -> dict[str, dict]:
    result = {}
    for row in load_jsonl(summary_path):
        result[row["file"]] = row["meta"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("replays/parsed"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-video", type=Path)
    args = parser.parse_args()
    root = args.root
    out = args.out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset artifact: {out}")
    out.mkdir(parents=True)

    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    split_by_session = splits["sessions"]
    metadata = session_meta(root / "summary.jsonl")
    session_ids = sorted(split_by_session)
    sessions = []
    manifest = []
    review_queue = []
    candidate_summary = {}

    aligned_dir = root / "alignment"
    old_candidate_dir = root / "candidates" / "236066_random_Kurama_XI"
    video_exists = bool(args.source_video and args.source_video.is_file())
    target_session = "236066_random_Kurama_XI.wowsreplay"
    for filename in session_ids:
        meta = metadata.get(filename, {})
        session_id = Path(filename).stem
        split = split_by_session[filename]
        has_alignment = (aligned_dir / f"{session_id}.json").is_file()
        session_video = str(args.source_video) if filename == target_session and video_exists else None
        sessions.append({
            "session_id": session_id,
            "replay_file": filename,
            "source_type": "replay",
            "source_path": str(Path("replays") / filename),
            "split": split,
            "match_group": meta.get("match_group"),
            "game_build": meta.get("build"),
            "resolution": [1920, 1080] if has_alignment else None,
            "ui_scale": None,
            "map": meta.get("map"),
            "ship_class": None,
            "recording_player": meta.get("recording_player"),
            "video_path": session_video,
            "video_available": bool(session_video),
            "alignment_available": has_alignment,
            "labels": {
                "replay_events": has_alignment,
                "human_visual": False,
            },
        })

    # Candidate windows are generated only for the session with a verified fit.
    if (aligned_dir / f"{Path(target_session).stem}.events.aligned.jsonl").is_file():
        event_path = aligned_dir / f"{Path(target_session).stem}.events.aligned.jsonl"
        consumable_path = aligned_dir / f"{Path(target_session).stem}.consumables.aligned.jsonl"
        track_path = aligned_dir / f"{Path(target_session).stem}.track.aligned.jsonl"
        events = read_events(event_path)
        # export_events.py already folds consumables into events.jsonl.  Only use
        # the dedicated stream as a fallback for older sessions that lack them.
        if not any(row.get("event") == "consumable" for row in events):
            events.extend(read_events(consumable_path))
        owner = metadata.get(target_session, {}).get("recording_player", "")
        events.extend(focus_target_changes(event_path, owner))
        events = sorted(fold_rows(events), key=lambda row: (float(row["video_time_s"]), row["event"]))
        windows = make_windows(events, duration_s=1499.20149)
        preview = select_preview(windows, limit=12)
        candidate_dir = out / "candidates" / Path(target_session).stem
        write_jsonl(candidate_dir / "windows.jsonl", windows)
        write_jsonl(candidate_dir / "preview.jsonl", preview)
        counts = Counter(row["event"] for row in events)
        candidate_summary[target_session] = {
            "input_events": len(events),
            "input_event_counts": dict(sorted(counts.items())),
            "candidate_windows": len(windows),
            "preview_windows": len(preview),
            "source_video_available": video_exists,
        }
        for candidate in windows:
            candidate_id = candidate["candidate_id"]
            sample_id = f"{Path(target_session).stem}:{candidate_id}"
            manifest.append({
                "sample_id": sample_id,
                "session_id": Path(target_session).stem,
                "source_type": "event_window",
                "source_path": str(event_path),
                "frame_index": None,
                "timestamp_ns": round(((candidate["start_s"] + candidate["end_s"]) / 2) * 1e9),
                "game_build": "15.8.0.13187581",
                "resolution": [1920, 1080],
                "ui_scale": None,
                "map": "spaces/41_Conquest",
                "ship_class": None,
                "split": split_by_session[target_session],
                "labels_path": str(candidate_dir / "review-labels.jsonl"),
                "sha256": None,
                "event_counts": candidate["event_counts"],
                "start_s": candidate["start_s"],
                "end_s": candidate["end_s"],
                "label_status": "replay_provenance_only",
            })
        # Existing retained previews are included as pixel evidence, never as full-video evidence.
        preview_dir = old_candidate_dir / "balanced-preview"
        for image in sorted(preview_dir.glob("candidate-*.jpg")):
            candidate_id = image.stem
            candidate = next((row for row in preview if row["candidate_id"] == candidate_id), None)
            if candidate is None:
                candidate = next((row for row in windows if row["candidate_id"] == candidate_id), None)
            if candidate is None:
                continue
            frame_id = f"{Path(target_session).stem}:{candidate_id}:center"
            record = {
                "sample_id": frame_id,
                "session_id": Path(target_session).stem,
                "source_type": "retained_preview_frame",
                "source_path": str(image),
                "frame_index": None,
                "timestamp_ns": round(((candidate["start_s"] + candidate["end_s"]) / 2) * 1e9),
                "game_build": "15.8.0.13187581",
                "resolution": [1920, 1080],
                "ui_scale": None,
                "map": "spaces/41_Conquest",
                "ship_class": None,
                "split": split_by_session[target_session],
                "labels_path": str(out / "review" / "labels.jsonl"),
                "sha256": sha256_or_none(image),
                "event_counts": candidate["event_counts"],
                "timestamp_kind": "window_center",
                "label_status": "pending_human_review",
            }
            manifest.append(record)
            review_queue.append({
                "sample_id": frame_id,
                "image_path": str(image),
                "video_context": str(preview_dir / f"{candidate_id}.mp4") if (preview_dir / f"{candidate_id}.mp4").is_file() else None,
                "session_id": Path(target_session).stem,
                "split": split_by_session[target_session],
                "candidate_id": candidate_id,
                "event_counts": candidate["event_counts"],
                "labels": {"hud": None, "ship_box": None, "health_bar": None, "target": None, "minimap": None},
                "status": "pending",
            })

    write_json(out / "session_manifest.json", {"schema_version": 1, "sessions": sessions})
    write_jsonl(out / "manifest.jsonl", manifest)
    write_jsonl(out / "review" / "queue.jsonl", review_queue)
    write_json(out / "candidate_summary.json", candidate_summary)

    split_sets = {
        name: {session["session_id"] for session in sessions if session["split"] == name}
        for name in ("fit", "validation", "holdout")
    }
    overlap = {f"{left}:{right}": sorted(split_sets[left] & split_sets[right]) for left in split_sets for right in split_sets if left < right}
    hashes: dict[str, list[str]] = {}
    for row in manifest:
        if row.get("sha256"):
            hashes.setdefault(row["sha256"], []).append(row["sample_id"])
    duplicate_hashes = {key: value for key, value in hashes.items() if len(value) > 1}
    leakage = {
        "session_split_overlap": overlap,
        "duplicate_content_hashes": duplicate_hashes,
        "cross_split_duplicate_hashes": {
            key: value for key, value in duplicate_hashes.items()
            if len({next(row["split"] for row in manifest if row["sample_id"] == sample) for sample in value}) > 1
        },
        "status": "pass" if not any(overlap.values()) and not any(
            len({next(row["split"] for row in manifest if row["sample_id"] == sample) for sample in value}) > 1
            for value in duplicate_hashes.values()
        ) else "fail",
    }
    write_json(out / "leakage_report.json", leakage)
    training = {
        "status": "blocked",
        "reason": "no full source video and no human visual labels; replay events are not HUD ground truth",
        "available_pixel_samples": len(review_queue),
        "required_before_training": ["human_review_labels", "fit/validation visual coverage", "holdout evaluation"],
    }
    write_json(out / "training_readiness.json", training)
    write_json(out / "manifest_summary.json", {
        "schema_version": 1,
        "sessions": len(sessions),
        "manifest_rows": len(manifest),
        "review_queue_rows": len(review_queue),
        "sample_splits": {name: sum(row["split"] == name for row in manifest) for name in split_sets},
        "session_splits": {name: len(split_sets[name]) for name in split_sets},
        "pixel_coverage": {
            name: sum(row["split"] == name and row["source_type"] == "retained_preview_frame" for row in manifest)
            for name in split_sets
        },
        "source_video_available": video_exists,
        "candidate_summary": candidate_summary,
        "leakage_status": leakage["status"],
        "training_status": training["status"],
    })
    print(json.dumps({"out": str(out), "manifest_rows": len(manifest), "review_rows": len(review_queue), "leakage": leakage["status"], "training": training["status"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
