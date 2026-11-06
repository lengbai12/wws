#!/usr/bin/env python3
"""Strict validation for the generated WoWS dataset manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from fsutil import sha256, load_json, load_jsonl



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset
    errors: list[str] = []
    notes: list[str] = []
    required = ["session_manifest.json", "manifest.jsonl", "leakage_report.json", "training_readiness.json"]
    for name in required:
        if not (root / name).is_file():
            errors.append(f"missing required artifact: {name}")
    if errors:
        print(json.dumps({"status": "fail", "errors": errors}, ensure_ascii=False))
        raise SystemExit(1)
    sessions_doc = load_json(root / "session_manifest.json")
    rows = load_jsonl(root / "manifest.jsonl")
    pixel_manifest = root / "review" / "frames" / "frames.jsonl"
    pixel_rows = load_jsonl(pixel_manifest) if pixel_manifest.is_file() else []
    sessions = {row["session_id"]: row for row in sessions_doc["sessions"]}
    seen_ids: set[str] = set()
    seen_hashes: dict[str, list[str]] = {}
    required_fields = {
        "sample_id", "session_id", "source_type", "source_path", "frame_index",
        "timestamp_ns", "game_build", "resolution", "split", "labels_path",
        "sha256", "label_status",
    }
    for index, row in enumerate(rows, 1):
        missing = required_fields - row.keys()
        if missing:
            errors.append(f"manifest row {index}: missing {sorted(missing)}")
        sample_id = row.get("sample_id")
        if sample_id in seen_ids:
            errors.append(f"manifest row {index}: duplicate sample_id {sample_id}")
        seen_ids.add(sample_id)
        session = sessions.get(row.get("session_id"))
        if session is None:
            errors.append(f"manifest row {index}: unknown session {row.get('session_id')}")
        elif row.get("split") != session.get("split"):
            errors.append(f"manifest row {index}: split disagrees with session manifest")
        if row.get("split") not in {"fit", "validation", "holdout"}:
            errors.append(f"manifest row {index}: invalid split {row.get('split')!r}")
        if row.get("resolution") != [1920, 1080]:
            errors.append(f"manifest row {index}: resolution contract is not [1920,1080]")
        source = Path(row.get("source_path", ""))
        if not source.is_absolute():
            source = Path.cwd() / source
        if not source.is_file():
            errors.append(f"manifest row {index}: source missing: {row.get('source_path')}")
        expected_hash = row.get("sha256")
        if expected_hash:
            if not source.is_file():
                continue
            actual = sha256(source)
            if actual != expected_hash:
                errors.append(f"manifest row {index}: sha256 mismatch")
            seen_hashes.setdefault(actual, []).append(sample_id)
        elif row.get("source_type") == "retained_preview_frame":
            errors.append(f"manifest row {index}: pixel sample has no sha256")
        if row.get("label_status") == "human_verified":
            notes.append(f"human verified row: {sample_id}")

    split_sessions = {name: {sid for sid, value in sessions.items() if value.get("split") == name} for name in ("fit", "validation", "holdout")}
    for left, right in (("fit", "validation"), ("fit", "holdout"), ("validation", "holdout")):
        overlap = split_sessions[left] & split_sessions[right]
        if overlap:
            errors.append(f"session split overlap {left}/{right}: {sorted(overlap)}")
    duplicate_hashes = {key: value for key, value in seen_hashes.items() if len(value) > 1}
    for digest, sample_ids in duplicate_hashes.items():
        sample_splits = {next(row["split"] for row in rows if row["sample_id"] == sid) for sid in sample_ids}
        if len(sample_splits) > 1:
            errors.append(f"cross-split duplicate content {digest}: {sample_ids}")
    readiness = load_json(root / "training_readiness.json")
    if readiness.get("status") == "ready" and not notes:
        errors.append("training marked ready without any human_verified labels")
    pixel_errors = []
    pixel_ids: set[str] = set()
    for index, row in enumerate(pixel_rows, 1):
        if row.get("sample_id") in pixel_ids:
            pixel_errors.append(f"pixel row {index}: duplicate sample_id")
        pixel_ids.add(row.get("sample_id"))
        source = Path(row.get("source_path", ""))
        if not source.is_file():
            pixel_errors.append(f"pixel row {index}: source missing")
        if row.get("source_resolution") != [1920, 1080]:
            pixel_errors.append(f"pixel row {index}: source resolution contract mismatch")
        if source.is_file() and sha256(source) != row.get("sha256"):
            pixel_errors.append(f"pixel row {index}: sha256 mismatch")
    errors.extend(pixel_errors)
    report = {
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "notes": notes,
        "manifest_rows": len(rows),
        "pixel_manifest_rows": len(pixel_rows),
        "sessions": len(sessions),
        "duplicate_content_hashes": duplicate_hashes,
        "training_status": readiness.get("status"),
    }
    (root / "strict_validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
