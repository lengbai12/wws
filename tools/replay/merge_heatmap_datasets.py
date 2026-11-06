"""Merge per-session minimap datasets into one corpus with SESSION-level splits.

Splitting one battle into blocks still shares a trajectory, a map, a ship and a UI
setting between its parts, so a block split cannot show whether the model works on
a battle it has never seen. Only a session-level split can, and that needs several
recorded sessions, which is what this merge exists to feed.

The split itself is delegated to ``splits.make_splits`` so the corpus uses the same
rule as the rest of the project: holdout and validation are chosen per SESSION,
stratified by match group, from a fixed seed, and the result is checked for overlap.

The corpus does not fit in RAM: five sessions are 14.7 GB of images and heatmaps on a
30 GB machine with no swap, so this never holds more than one session. It sizes the
output from the npz headers first, streams each big array into a plain ``.npy`` twin
one session at a time, and then streams those twins into the npz in chunks. Each big
array is therefore written twice -- as a plain ``.npy``, which is the copy the trainer
memory-maps, and inside the npz, which is the dataset contract. ``load_all`` in
train_minimap_heatmap.py explains why the twin is needed at all: a zipped member
cannot be memory-mapped by numpy, stored or deflated.

Neither pass memory-maps anything. A write into a memmap counts its pages as the
process's own RSS for as long as they are dirty, so mapping both output arrays at
once measured 13.2 GB for THREE sessions (9 GB of output) -- the two arrays are not
one session's worth of memory, they are the corpus. Copying through a chunk buffer
instead keeps the peak at one session of source plus one chunk (measured below one
gigabyte for five sessions).

Usage:
  merge_heatmap_datasets.py \
      --pair heatmap-A.npz:minimap-A.npz --pair heatmap-B.npz:minimap-B.npz ... \
      --summary replays/parsed/summary.jsonl \
      --out-heatmaps replays/parsed/datasets/heatmap-merged.npz \
      --out-images replays/parsed/datasets/minimap-merged.npz \
      --out-summary replays/parsed/datasets/heatmap-merged.summary.json \
      [--holdout 2] [--validation 2] [--seed wws-layer1]
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from fsutil import twin_path


import numpy as np

import splits as split_tools

# The project's layer-1 names are fit / validation / holdout; the trainer speaks
# train / validation / test. One mapping, stated once.
SPLIT_NAMES = {"fit": "train", "validation": "validation", "holdout": "test"}

# Small enough to hold for every session at once: the whole corpus of these is ~20 MB.
SMALL_KEYS = ("rings", "truth", "counts", "negatives", "negative_counts",
              "valid", "ring_valid")
# Big enough that one session at a time is the only option.
BIG_KEYS = ("heatmaps", "images", "images_half")
# 半分辨率成员按"每次处理多少样本"分块，峰值只是几块而不是整个语料 ✓
HALF_CHUNK = 16
COPY_CHUNK = 1 << 25


def read_npy_header(stream):
    """Return (shape, dtype) of an open .npy, leaving it positioned on the data.

    The magic has to be consumed first: ``read_array_header_1_0`` starts at the
    header-length field, not at the magic.
    """
    version = np.lib.format.read_magic(stream)
    read = (np.lib.format.read_array_header_1_0 if version == (1, 0)
            else np.lib.format.read_array_header_2_0)
    shape, fortran_order, dtype = read(stream, max_header_size=1 << 24)
    if fortran_order:
        raise SystemExit("a Fortran-ordered array cannot be streamed into the corpus")
    return tuple(shape), np.dtype(dtype)


def read_headers(path):
    """Return {member: (shape, dtype)} for an npz, reading only the npy headers.

    A header never touches the member's data, so this pass over five sessions costs
    kilobytes instead of the 15 GB the arrays hold.
    """
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with archive.open(info) as stream:
                headers[info.filename[:-len(".npy")]] = read_npy_header(stream)
    return headers


def join_rows(arrays, fill=np.nan):
    """Concatenate per-session arrays along axis 0, padding a narrower axis 1.

    ``truth`` and ``negatives`` are ragged across sessions (60 / 35 / 43 / 27 / 44
    markers), so a plain concatenate raises. The pad is NaN, which every consumer
    already skips: truth rows are filtered by ``not np.isnan(row[0])`` and negatives
    are sliced by ``negative_counts``, so padding to the widest keeps it out of both.
    """
    width = max(array.shape[1] for array in arrays)
    if all(array.shape[1] == width for array in arrays):
        return np.concatenate(arrays)
    padded = []
    for array in arrays:
        if array.shape[1] == width:
            padded.append(array)
            continue
        room = np.full((array.shape[0], width - array.shape[1]) + array.shape[2:],
                       fill, dtype=array.dtype)
        padded.append(np.concatenate([array, room], axis=1))
    return np.concatenate(padded)


def copy_corpus(members, target, shape, dtype):
    """Stream per-session npz members into one standalone .npy, in order.

    A DEFLATE member has to be decompressed to be read at all, so one session is
    resident no matter how the copy is written; what this avoids is holding the
    OUTPUT as well. Writing through a chunk buffer keeps the peak at one session of
    source instead of the whole corpus, and the file it leaves behind is a plain .npy
    because that is the layout ``np.load(..., mmap_mode="r")`` can actually map.
    """
    written = 0
    with open(target, "wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle, dict(descr=dtype.str, fortran_order=False, shape=shape))
        for source, member in members:
            with zipfile.ZipFile(source) as archive:
                with archive.open(member + ".npy") as stream:
                    member_shape, member_dtype = read_npy_header(stream)
                    if (member_shape[1:], member_dtype) != (tuple(shape[1:]), dtype):
                        raise SystemExit("%s: %s is shaped %s %s while the corpus holds "
                                         "%s %s" % (source, member, member_shape,
                                                    member_dtype, shape[1:], dtype))
                    remaining = int(np.prod(member_shape)) * member_dtype.itemsize
                    while remaining:
                        block = stream.read(min(COPY_CHUNK, remaining))
                        if not block:
                            raise SystemExit("%s: %s ended early" % (source, member))
                        handle.write(block)
                        remaining -= len(block)
                        written += len(block)
    expected = int(np.prod(shape)) * dtype.itemsize
    if written != expected:
        raise SystemExit("%s holds %d bytes but %d were expected"
                         % (target, written, expected))
    return target


def stream_npy(stream, source):
    """Copy a standalone .npy into an open archive member, one chunk at a time."""
    with open(source, "rb") as handle:
        shape, dtype = read_npy_header(handle)
        np.lib.format.write_array_header_1_0(
            stream, dict(descr=dtype.str, fortran_order=False, shape=shape))
        remaining = int(np.prod(shape)) * dtype.itemsize
        while remaining:
            block = handle.read(min(COPY_CHUNK, remaining))
            if not block:
                raise SystemExit("%s ended early" % source)
            stream.write(block)
            remaining -= len(block)


def write_half_twin(source, target, factor=2):
    """Stream a 2x2-downsampled twin of the image member.

    What the GPU waits on is the loader's gather, not the model: 76.5 ms of reads
    against 42.5 ms of compute, because one crop is 1.5 MB and the corpus does not fit
    in the page cache. A half-resolution copy is four times less data per epoch.

    Averaging four uint8 values in uint16 and rounding half up stays within half a step
    (1/510) of the exact mean, and the same rule is used wherever the model input is
    built, so training and the runtime see the same numbers. Chunks are whole samples,
    so no sample is ever averaged across a boundary.
    """
    array = np.load(source, allow_pickle=False, mmap_mode="r")
    samples, height, width, channels = array.shape
    shape = (samples, height // factor, width // factor, channels)
    with target.open("wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle, {"descr": "|u1", "fortran_order": False, "shape": shape})
        for start in range(0, samples, HALF_CHUNK):
            block = np.asarray(array[start:start + HALF_CHUNK])
            rows, columns = block.shape[1] // factor, block.shape[2] // factor
            trimmed = block[:, :rows * factor, :columns * factor, :]
            summed = trimmed.astype(np.uint16).reshape(
                block.shape[0], rows, factor, columns, factor, channels).sum(axis=(2, 4))
            half = ((summed + (factor * factor) // 2) // (factor * factor)).astype(np.uint8)
            handle.write(np.ascontiguousarray(half).tobytes())
    del array, block, summed, half
    return target


def write_npz(path, members):
    """Write an npz whose members are STORED, streaming the big ones from disk.

    STORED (never DEFLATE) keeps every member a plain contiguous .npy inside the zip,
    and the big members must be opened with ``force_zip64`` because they exceed 4 GB.
    A big member is named by the .npy file that holds it, not by an in-memory array:
    that file is the same bytes the trainer memory-maps, so the npz and the twin can
    never disagree about shape, dtype or order.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            with archive.open(name + ".npy", "w", force_zip64=True) as stream:
                if name in BIG_KEYS:
                    stream_npy(stream, payload)
                else:
                    np.lib.format.write_array(stream, payload, allow_pickle=False)


def read_match_groups(path):
    """Return {replay file stem: match group} from the parsed corpus summary."""
    groups = {}
    if path and Path(path).exists():
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                meta = row.get("meta") or {}
                stem = Path(row.get("file", "")).stem
                if stem:
                    groups[stem] = meta.get("match_group") or "unknown"
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", action="append", required=True,
                        help="HEATMAP_NPZ:IMAGE_NPZ, repeat once per session")
    parser.add_argument("--summary", type=Path,
                        help="parsed summary.jsonl, used for match_group stratification")
    parser.add_argument("--out-heatmaps", type=Path, required=True)
    parser.add_argument("--out-images", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path)
    parser.add_argument("--holdout", type=int, default=2)
    parser.add_argument("--validation", type=int, default=2)
    parser.add_argument("--seed", default="wws-layer1")
    args = parser.parse_args(argv)

    sessions = []
    per_session_availability = {}
    rects = []
    pooled = {key: [] for key in SMALL_KEYS}
    sources = []
    classes = ring_labels = grid = sigma = None
    heat_shape = image_shape = heat_dtype = image_dtype = None
    for pair in args.pair:
        heat_path, _, image_path = pair.partition(":")
        heat_headers = read_headers(heat_path)
        image_headers = read_headers(image_path)
        with np.load(heat_path, allow_pickle=False) as heat:
            heat_classes = [str(v) for v in heat["classes"]]
            heat_ring_labels = [float(v) for v in heat["ring_labels"]]
            heat_grid = [int(v) for v in heat["grid"]]
            heat_sigma = float(heat["sigma"][0])
            heat_sessions = [str(v) for v in heat["session_ids"]]
            # Fail loudly rather than treat a missing mask as "everything is valid":
            # a session built before the validity contract would then be merged as if
            # its unmeasurable channels (no zone calibration, no lock log) were
            # confirmed absences, which is the defect the mask exists to remove.
            missing = [key for key in SMALL_KEYS if key not in heat]
            if missing:
                raise SystemExit(
                    "%s carries no %s: it was built before the validity contract, and "
                    "merging it would silently mark an unmeasurable channel as "
                    "supervised. Rebuild that session's heatmap first "
                    "(auto_learn.py --force heatmap)."
                    % (heat_path, ", ".join(missing)))
            small = {key: heat[key] for key in SMALL_KEYS}
        count = heat_headers["heatmaps"][0][0]
        if classes is None:
            classes = heat_classes
            ring_labels = heat_ring_labels
            grid = heat_grid
            sigma = heat_sigma
        elif heat_classes != classes:
            raise SystemExit("%s has a different class list; the corpus must agree"
                             % heat_path)
        if heat_grid[0] != grid[0]:
            raise SystemExit("%s uses grid %d while the corpus uses %d"
                             % (heat_path, heat_grid[0], grid[0]))

        # The output is sized from these and written into with memmaps, so a session
        # that disagrees about the shape has to be refused here, before anything is
        # allocated, and not halfway through a copy.
        heat_row = heat_headers["heatmaps"][0]
        image_row = image_headers["images"][0]
        if heat_shape is None:
            heat_shape, heat_dtype = heat_row, heat_headers["heatmaps"][1]
            image_shape, image_dtype = image_row, image_headers["images"][1]
        else:
            if heat_row[1:] != heat_shape[1:]:
                raise SystemExit("%s holds heatmaps shaped %s while the corpus holds %s"
                                 % (heat_path, heat_row[1:], heat_shape[1:]))
            if image_row[1:] != image_shape[1:]:
                raise SystemExit("%s holds images shaped %s while the corpus holds %s"
                                 % (image_path, image_row[1:], image_shape[1:]))
        if not (len(heat_sessions) == count == image_row[0]
                and all(array.shape[0] == count for array in small.values())):
            raise SystemExit("%s: %d heatmaps, %d images, %d session ids and %s are not "
                             "the same number of samples"
                             % (heat_path, count, image_row[0], len(heat_sessions),
                                {key: array.shape[0] for key, array in small.items()}))

        with np.load(image_path, allow_pickle=False) as image:
            rect = [int(v) for v in image["minimap_rect"]]
        if rects and rect != rects[0]:
            raise SystemExit("%s was recorded with minimap rect %s while the corpus uses %s; "
                             "the crop rectangle is part of the input contract, so sessions "
                             "recorded at a different minimap size cannot be merged"
                             % (heat_path, rect, rects[0]))
        rects.append(rect)

        sources.append({"heat": heat_path, "image": image_path, "count": count})
        sessions.extend(heat_sessions)
        for key, array in small.items():
            pooled[key].append(array)
        session_summary = Path(str(heat_path).replace(".npz", ".summary.json"))
        if session_summary.exists():
            try:
                recorded = json.loads(session_summary.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                recorded = {}
            availability = recorded.get("class_availability") or {}
            if availability:
                per_session_availability[Path(heat_path).name] = {
                    "map": recorded.get("map"),
                    "zone": availability.get("zone"),
                    "range_ring": availability.get("range_ring"),
                }
        print("  %-28s %4d samples  session %s"
              % (Path(heat_path).name, count, sorted(set(heat_sessions))))

    unique = sorted(set(sessions))
    groups = read_match_groups(args.summary)
    session_rows = [{"replay": name, "match_group": groups.get(name, "unknown")}
                    for name in unique]
    print("sessions: %s" % unique)
    print("match groups: %s" % {row["replay"]: row["match_group"] for row in session_rows})
    if len(unique) < args.holdout + args.validation + 1:
        raise SystemExit("only %d sessions; a session-level split needs at least %d "
                         "(holdout %d + validation %d + fit 1)"
                         % (len(unique), args.holdout + args.validation + 1,
                            args.holdout, args.validation))

    split_record = split_tools.make_splits(session_rows, holdout=args.holdout,
                                           validation=args.validation, seed=args.seed,
                                           stratify="match_group")
    # check_disjoint reports STRUCTURAL problems as a list of strings (an empty list
    # means the split file is usable), so it is a list to report, not a pair to unpack.
    problems = split_tools.check_disjoint(split_record)
    if problems:
        raise SystemExit("session splits are not usable: %s" % "; ".join(problems))
    per_session = split_record["sessions"]
    print("session splits: %s" % per_session)
    print("counts: %s" % split_record["counts"])

    sample_splits = np.asarray([SPLIT_NAMES[per_session[name]] for name in sessions])
    # A channel that a session cannot supply (no zone calibration for its map, or no
    # measured ring scale) is zeros there. Saying so keeps a model that learned to
    # output nothing for that channel from looking like a modelling failure.
    unavailable = {}
    for channel in ("zone", "range_ring"):
        missing = [name for name, entry in per_session_availability.items()
                   if entry.get(channel) is False]
        if missing:
            unavailable[channel] = missing

    total_samples = sum(entry["count"] for entry in sources)
    args.out_heatmaps.parent.mkdir(parents=True, exist_ok=True)
    args.out_images.parent.mkdir(parents=True, exist_ok=True)
    # One array at a time, and inside it one session at a time: the peak here is a
    # chunk plus a session's decompressed source (~3 GB), not the corpus and its copies.
    heat_twin = copy_corpus([(entry["heat"], "heatmaps") for entry in sources],
                            twin_path(args.out_heatmaps, "heatmaps"),
                            (total_samples,) + tuple(heat_shape[1:]), heat_dtype)
    image_twin = copy_corpus([(entry["image"], "images") for entry in sources],
                             twin_path(args.out_images, "images"),
                             (total_samples,) + tuple(image_shape[1:]), image_dtype)
    # Same data at half resolution: this is the member the trainer reads every epoch,
    # so its cost is worth paying once here.
    image_half_twin = write_half_twin(image_twin,
                                      twin_path(args.out_images, "images_half"))

    merged = {
        "heatmaps": heat_twin,
        "rings": np.concatenate(pooled["rings"]),
        "valid": np.concatenate(pooled["valid"]),
        "ring_valid": np.concatenate(pooled["ring_valid"]),
        "truth": join_rows(pooled["truth"]),
        "counts": np.concatenate(pooled["counts"]),
        "negatives": join_rows(pooled["negatives"]),
        "negative_counts": np.concatenate(pooled["negative_counts"]),
        "splits": sample_splits,
        "session_ids": np.asarray(sessions),
        "classes": np.asarray(classes),
        "ring_labels": np.asarray(ring_labels),
        "grid": np.asarray(grid),
        "sigma": np.asarray([sigma]),
    }
    write_npz(args.out_heatmaps, merged)
    image_merged = {
        "images": image_twin,
        "images_half": image_half_twin,
        "minimap_rect": np.load(args.pair[0].partition(":")[2], allow_pickle=False)["minimap_rect"],
        "splits": sample_splits,
        "session_ids": np.asarray(sessions),
    }
    write_npz(args.out_images, image_merged)
    print("wrote %s (%.1f MB) and %s (%.1f MB)"
          % (args.out_heatmaps, args.out_heatmaps.stat().st_size / 1e6,
             args.out_images, args.out_images.stat().st_size / 1e6))
    print("memmap-able twins: %s and %s (plus %s, %.1f MB)"
          % (heat_twin, image_twin, image_half_twin,
             image_half_twin.stat().st_size / 1e6))
    print("sample split counts: %s"
          % {name: int((sample_splits == name).sum())
             for name in ("train", "validation", "test")})

    if args.out_summary:
        summary = {
            "sessions": unique,
            "session_splits": per_session,
            "session_groups": {row["replay"]: row["match_group"] for row in session_rows},
            "split_rule": ("session-level, stratified by match group, seed %s, from "
                           "splits.make_splits" % args.seed),
            "sample_split_counts": {name: int((sample_splits == name).sum())
                                    for name in ("train", "validation", "test")},
            "samples": int(len(sample_splits)),
            "minimap_rect": rects[0] if rects else None,
            "classes": classes,
            "grid": grid,
            "sigma": sigma,
            "pairs": args.pair,
            "per_session_class_availability": per_session_availability,
            "channels_unavailable_in_some_session": unavailable,
            # What the trainer will actually supervise: derived from the mask members,
            # so the corpus describes its own supervision instead of leaving a reader to
            # infer it from per-session summaries.
            "supervised_channels": {
                name: bool((merged["valid"][:, index] > 0).any())
                for index, name in enumerate(classes)},
            "samples_with_any_unsupervised_channel": int(
                (merged["valid"].min(axis=1) == 0).sum()),
            "ring_supervised": bool((merged["ring_valid"] > 0).any()),
        }
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                    encoding="utf-8")
        print("wrote %s" % args.out_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
