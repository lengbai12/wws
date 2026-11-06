"""One implementation of the small file helpers every stage needs.

These used to be copied into eight modules (``sha256``), two (``twin_path``) and
three (``load_jsonl``/``write_jsonl``); the copies drifted in block size and in
how a missing file is reported. Kept here so there is exactly one behaviour.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

BLOCK = 1 << 20


def sha256(path):
    """Streaming SHA-256 of a file; a missing file is an error, not a value."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_or_none(path):
    """``sha256`` for an optional file: absent means no digest to report."""
    path = Path(path)
    return sha256(path) if path.is_file() else None


def twin_path(path, member):
    """The memmap-able ``.npy`` twin of one member of the npz at ``path``.

    This naming is the interface between the merge and the trainer: the merge
    writes the twin, the trainer memory-maps it instead of reading the member.
    """
    path = Path(path)
    return path.with_name("%s.%s.npy" % (path.stem, member))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + chr(10))


def write_json(path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + chr(10), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
