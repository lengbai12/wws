#!/usr/bin/env python3
"""Offline human bow annotation using the existing OpenCV dependency.

Example:
  python tools/replay/annotate_bow.py frame.png dataset/annotations.jsonl \
      --sample-id match03-frame012 --session match03 --reviewer your-name \
      --roi 800 350 600 220

No game is opened and no labels are generated automatically. The optional ROI
is raw source [x,y,width,height]; the default is the whole source. It is padded
black at bottom/right to a square then resized to 256, exactly like inference.
The original source pixels are copied into the dataset for future audit.

Export unlabelled scope candidates for later human review (never creates labels):
  python tools/replay/annotate_bow.py export --image frame.png --session match03 \
      --output-dir dataset/pending-match03 --scope-observed
The queue includes exact per-candidate annotation commands. Run each command,
adding --reviewer YOUR_NAME, and manually review the polygon/points/viewpoint.

Mouse left: add polygon vertex, or set bow/stern in selected mode.
P: polygon; B: bow; T: stern; Z/right mouse: undo polygon vertex.
C: clear; N: cycle negative reasons; +/-: camera bin; U: explicitly unknown.
Positive mode requires an explicit hull polygon, both visible points, and an
orientation choice (unknown is a valid explicit choice). For an occluded or
unidentifiable bow, choose a negative reason rather than guessing points.
R: attest manual review; W: save reviewed sample and exit; Esc: cancel.
Any annotation change revokes review. No output file is overwritten.

Camera bins are NOT image-plane direction: 0 bow-on, 4 ship's STARBOARD/right,
8 stern-on, 12 ship's PORT/left. Centers advance 22.5 degrees, with half-open
+/-11.25-degree circular intervals. A 2D rotation never proves a 3D viewpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

import cv2
import numpy as np

from bow_dataset import (
    CROP_CONVENTION, ORIENTATION_CONVENTION, REJECTION_REASONS, SIZE,
    load_annotations, pixel_sha256, source_crop, validate_row, read_source,
)


def save_sample(annotations: Path, sample_id: str, source_path: Path,
                source: np.ndarray, roi, session: str, reviewer: str,
                polygon, bow, stern, orientation, negative_reason=None,
                reviewed=False, annotation_source="manual") -> dict:
    """Persist an explicitly reviewed annotation with its truthful label source."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", sample_id):
        raise ValueError("sample-id must be 1..120 ASCII letters/digits/_/-")
    if reviewed is not True:
        raise ValueError("explicit review acknowledgement is required")
    if annotation_source not in ("manual", "assistant_visual"):
        raise ValueError("annotation_source must be manual or assistant_visual")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer must identify the actual reviewer")
    annotations = Path(annotations)
    root = annotations.parent
    root.mkdir(parents=True, exist_ok=True)
    lock = annotations.with_name(annotations.name + ".lock")
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    created = []
    try:
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("source must be uint8 BGR")
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4
                or any(isinstance(v, bool) or not isinstance(v, int) for v in roi)):
            raise ValueError("roi must be integer [x,y,width,height]")
        x, y, width, height = roi
        if min(x, y) < 0 or min(width, height) <= 0 or x + width > source.shape[1] or y + height > source.shape[0]:
            raise ValueError("roi outside source image")
        crop = source_crop(source, roi)
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        if negative_reason is None:
            if (not isinstance(polygon, (list, tuple)) or len(polygon) < 3):
                raise ValueError("positive requires at least three manual polygon vertices")
            from bow_dataset import _point
            vertices = [_point(point, "polygon vertex") for point in polygon]
            cv2.fillPoly(mask, [np.rint(vertices).astype(np.int32)], 255)
        else:
            if polygon or bow is not None or stern is not None:
                raise ValueError("negative must not retain polygon or point labels")
        names = {
            "image": f"{sample_id}.png", "mask": f"{sample_id}.mask.png",
            "source_image": f"{sample_id}.source.png",
        }
        row = dict(
            **names, sample_id=sample_id, source_original_path=str(Path(source_path).resolve()),
            source_roi=list(roi), source_sha256=pixel_sha256(source),
            image_sha256=pixel_sha256(crop), crop_transform=CROP_CONVENTION,
            session=session, reviewer=reviewer, reviewed=True, annotation_source=annotation_source,
            label_status="negative" if negative_reason is not None else "positive",
            bow_point=bow, stern_point=stern, orientation_bin=orientation,
        )
        if negative_reason is not None:
            row["rejection_reason"] = negative_reason
        existing = load_annotations(annotations) if annotations.exists() else []
        if any(old.get("sample_id") == sample_id or old["image_sha256"] == row["image_sha256"] for old in existing):
            raise ValueError("sample-id or crop pixels already annotated; refusing duplicate/overwrite")
        if any(old["source_sha256"] == row["source_sha256"] and old["session"] != session for old in existing):
            raise ValueError("source frame already belongs to a different session")
        for key, pixels in (("image", crop), ("mask", mask), ("source_image", source)):
            target = root / names[key]
            encoded, png = cv2.imencode(".png", pixels)
            if not encoded:
                raise ValueError(f"cannot encode {key}")
            with target.open("xb") as stream:
                created.append(target)
                stream.write(png.tobytes())
        row = validate_row(row, root)
        # Appending never rewrites earlier reviewed records. The lock serializes
        # writers using this entrypoint; exclusive asset creation is a second guard.
        if annotations.exists() and annotations.stat().st_size:
            with annotations.open("rb") as stream:
                stream.seek(-1, os.SEEK_END)
                needs_newline = stream.read(1) != b"\n"
        else:
            needs_newline = False
        with annotations.open("a", encoding="utf-8") as stream:
            stream.write(("\n" if needs_newline else "") + json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return row
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        os.close(lock_fd)
        lock.unlink()


class AnnotationUI:
    def __init__(self, crop):
        self.crop = crop
        self.polygon = []
        self.bow = self.stern = None
        self.orientation = "unknown"
        self.orientation_selected = False
        self.negative_reason = None
        self.mode = "polygon"
        self.reviewed = False
        self.message = "Draw a hull polygon; select bow/stern and viewpoint."

    def mouse(self, event, x, y, flags, param):
        if not (0 <= x < 768 and 0 <= y < 768):
            return
        if event == cv2.EVENT_RBUTTONDOWN and self.polygon:
            self.polygon.pop()
            self.reviewed = False
        if event != cv2.EVENT_LBUTTONDOWN or self.negative_reason is not None:
            return
        point = [min(SIZE - 1.0, x / 3), min(SIZE - 1.0, y / 3)]
        if self.mode == "polygon":
            self.polygon.append(point)
        elif self.mode == "bow":
            self.bow = point
        else:
            self.stern = point
        self.reviewed = False

    def render(self):
        crop = cv2.resize(self.crop, (768, 768), interpolation=cv2.INTER_NEAREST)
        if len(self.polygon) >= 3:
            vertices = np.rint(np.array(self.polygon) * 3).astype(np.int32)
            overlay = crop.copy()
            cv2.fillPoly(overlay, [vertices], (40, 170, 40))
            crop = cv2.addWeighted(crop, .7, overlay, .3, 0)
            cv2.polylines(crop, [vertices], True, (0, 255, 0), 2)
        for point in self.polygon:
            cv2.circle(crop, tuple(np.rint(np.array(point) * 3).astype(int)), 4, (0, 255, 0), -1)
        for point, label, color in ((self.bow, "BOW", (0, 255, 255)), (self.stern, "STERN", (255, 180, 0))):
            if point is not None:
                position = tuple(np.rint(np.array(point) * 3).astype(int))
                cv2.circle(crop, position, 7, color, 2)
                cv2.putText(crop, label, position, cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
        canvas = np.zeros((768, 1360, 3), dtype=np.uint8)
        canvas[:, :768] = crop
        lines = [
            "OFFLINE HUMAN ANNOTATION", "P polygon | B bow | T stern",
            "Left click: add/set | Z/right: undo", "C clear | N cycle negative reason",
            "+/- select viewpoint (including unknown)", "R reviewed | W save | Esc cancel", "",
            f"Mode: {self.mode}", f"Status: {self.negative_reason or 'positive'}",
            f"Camera orientation: {self.orientation}",
            f"Explicit choice: {self.orientation_selected}", f"Reviewed: {self.reviewed}", "",
            "0 = BOW-ON", "4 = STARBOARD (ship's RIGHT side)",
            "8 = STERN-ON", "12 = PORT (ship's LEFT side)",
            "Centers: k * 22.5 degrees", "Intervals: [center-11.25, center+11.25)",
            "Wrap at 360 degrees.", "NOT the image-plane bow/stern angle!", "",
            "Occluded/invisible/ambiguous bow:", "choose negative, never guess points.",
            "U: explicitly choose unknown view.", "", self.message,
        ]
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (783, 26 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, .48, (230, 230, 230), 1)
        return canvas

    def key(self, key):
        if key in (ord("p"), ord("b"), ord("t")):
            self.mode = {ord("p"): "polygon", ord("b"): "bow", ord("t"): "stern"}[key]
        elif key == ord("z") and self.polygon:
            self.polygon.pop()
            self.reviewed = False
        elif key == ord("c"):
            self.polygon.clear()
            self.bow = self.stern = None
            self.negative_reason = None
            self.orientation = "unknown"
            self.orientation_selected = self.reviewed = False
        elif key == ord("n"):
            choices = [None, *REJECTION_REASONS]
            self.negative_reason = choices[(choices.index(self.negative_reason) + 1) % len(choices)]
            self.polygon.clear()
            self.bow = self.stern = None
            self.orientation = "unknown"
            self.orientation_selected = self.negative_reason is not None
            self.reviewed = False
        elif key in (ord("+"), ord("="), ord("-")) and self.negative_reason is None:
            choices = ["unknown", *range(16)]
            delta = -1 if key == ord("-") else 1
            self.orientation = choices[(choices.index(self.orientation) + delta) % len(choices)]
            self.orientation_selected = True
            self.reviewed = False
        elif key == ord("u") and self.negative_reason is None:
            self.orientation = "unknown"
            self.orientation_selected = True
            self.reviewed = False
        elif key == ord("r"):
            if not self.orientation_selected:
                self.message = "Select a viewpoint, or cycle to unknown first."
            elif self.negative_reason is None and (len(self.polygon) < 3 or self.bow is None or self.stern is None):
                self.message = "Positive needs polygon, bow, and stern."
            else:
                self.reviewed = True
                self.message = "Human review attested. W saves; changes revoke review."

def export_candidates(images, session: str, output_dir: Path, *, scope_observed: bool) -> dict:
    """Export real detector crops and provenance, with NO training annotations."""
    from main_view_perception import FRAME_H, FRAME_W, detect_nameplates
    from bow_recognizer import crop_box, PREPROCESS
    if not isinstance(session, str) or not session.strip():
        raise ValueError("session must be an actual non-empty recording identity")
    if scope_observed is not True:
        raise ValueError("scope observation must be explicitly confirmed for export")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    created = []
    queue = []
    source_hashes = set()
    frame_count = 0
    try:
        for value in images:
            original = Path(value).resolve()
            source = read_source(original)
            if source.shape[:2] != (FRAME_H, FRAME_W):
                raise ValueError(
                    f"scope source must be native {FRAME_W}x{FRAME_H}; got "
                    f"{source.shape[1]}x{source.shape[0]}: {original}; no image rescaling")
            source_hash = pixel_sha256(source)
            if source_hash in source_hashes:
                raise ValueError(f"duplicate source pixels: {original}")
            source_hashes.add(source_hash)
            frame_count += 1
            source_name = f"{source_hash}.source.png"
            source_copy = output_dir / source_name
            ok, encoded = cv2.imencode(".png", source)
            if not ok:
                raise ValueError("cannot encode source image")
            with source_copy.open("xb") as stream:
                created.append(source_copy)
                stream.write(encoded.tobytes())
            plates = detect_nameplates(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), scope_observed=True)
            for index, plate in enumerate(plates):
                box = plate["nameplate_bbox"]
                x0, y0, x1, y1 = crop_box(box, source.shape)
                if min(x1 - x0, y1 - y0) < 16:
                    continue
                roi = [x0, y0, x1 - x0, y1 - y0]
                crop = source_crop(source, roi)
                sample_id = f"{source_hash[:24]}-plate-{index:03d}"
                crop_path = output_dir / f"{sample_id}.pending.png"
                ok, encoded = cv2.imencode(".png", crop)
                if not ok:
                    raise ValueError("cannot encode candidate crop")
                with crop_path.open("xb") as stream:
                    created.append(crop_path)
                    stream.write(encoded.tobytes())
                command = [sys.executable, str(Path(__file__).resolve()), str(source_copy),
                           str(output_dir / "reviewed.jsonl"), "--sample-id", sample_id,
                           "--session", session, "--roi", *map(str, roi)]
                queue.append(dict(
                    sample_id=sample_id, review_status="pending", reviewed=False,
                    candidate_origin="nameplate_healthbar_scope", scope_observed=True,
                    session=session, source_image=source_name,
                    source_original_path=str(original), source_sha256=source_hash,
                    source_roi=roi, crop_transform=CROP_CONVENTION,
                    preprocessing=PREPROCESS, image=crop_path.name,
                    image_sha256=pixel_sha256(crop), nameplate_bbox=list(box),
                    annotation_command=command,
                    annotation_command_note="Append --reviewer NAME; manually label all fields."
                ))
        queue_path = output_dir / "queue.jsonl"
        with queue_path.open("x", encoding="utf-8") as stream:
            created.append(queue_path)
            for row in queue:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        return dict(status="pending_human_review", source_frames=frame_count,
                    candidates=len(queue), queue=str(queue_path), labels_created=0,
                    known_orientation_coverage=[], orientation_convention=ORIENTATION_CONVENTION)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        output_dir.rmdir()
        raise


def export_main(argv):
    parser = argparse.ArgumentParser(description="Export offline scope nameplate crops; never infer bow labels.")
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope-observed", action="store_true", required=True,
                        help="Attest these saved frames were observed in binocular scope")
    args = parser.parse_args(argv)
    try:
        result = export_candidates(args.image, args.session, args.output_dir,
                                   scope_observed=args.scope_observed)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result))
    return 0



def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        return export_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--session", required=True, help="Actual recorded game/session identity; never invent split sessions")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args(argv)
    try:
        source = read_source(args.source)
    except ValueError as error:
        parser.error(str(error))
    roi = args.roi if args.roi is not None else [0, 0, source.shape[1], source.shape[0]]
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > source.shape[1] or y + height > source.shape[0]:
        parser.error("ROI outside source")
    ui = AnnotationUI(source_crop(source, roi))
    title = "Offline bow annotation - " + args.sample_id
    try:
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, ui.mouse)
        while True:
            cv2.imshow(title, ui.render())
            key = cv2.waitKey(50) & 0xff
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return 1
            if key == ord("w"):
                try:
                    row = save_sample(args.annotations, args.sample_id, args.source, source, roi,
                                      args.session, args.reviewer, ui.polygon, ui.bow, ui.stern,
                                      ui.orientation, ui.negative_reason, ui.reviewed)
                except (ValueError, OSError) as error:
                    ui.message = str(error)
                    print(f"Not saved: {error}")
                else:
                    print(json.dumps({"saved": row["sample_id"], "annotations": str(args.annotations),
                                      "orientation_convention": ORIENTATION_CONVENTION}))
                    return 0
            else:
                ui.key(key)
    except cv2.error as error:
        parser.exit(2, f"OpenCV GUI unavailable; use a desktop OpenCV build: {error}\n")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
