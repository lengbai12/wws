#!/usr/bin/env python3
"""Minimal World of Warships replay container decoder.

Owns only the container format: header blocks, the Blowfish/zlib packet stream,
and 12-byte packet framing. Entity/property semantics come from the schema layer.
"""
from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from Crypto.Cipher import Blowfish

SIGNATURE = b"\x12\x32\x34\x11"
WOWS_KEY = bytes([0x29, 0xB7, 0xC9, 0x09, 0x38, 0x3F, 0x84, 0x88,
                  0xFA, 0x98, 0xEC, 0x4E, 0x13, 0x19, 0x79, 0xFB])
PACKET_HEADER = 12


@dataclass(frozen=True)
class Packet:
    type: int
    clock: float
    payload: bytes


@dataclass
class Replay:
    meta: dict
    extra: list
    stream: bytes
    source: Path


def _decrypt(ciphertext: bytes) -> bytes:
    """ECB-decrypt each 8-byte block, then XOR-chain with the previous plaintext."""
    cipher = Blowfish.new(WOWS_KEY, Blowfish.MODE_ECB)
    usable = len(ciphertext) - len(ciphertext) % 8
    out = bytearray()
    previous = None
    for offset in range(0, usable, 8):
        plain = cipher.decrypt(ciphertext[offset:offset + 8])
        if previous is not None:
            plain = bytes(a ^ b for a, b in zip(plain, previous))
        previous = plain
        out += plain
    return bytes(out)


def read(path: str | Path) -> Replay:
    """Decode one .wowsreplay into engine metadata plus the decompressed packet stream."""
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] != SIGNATURE:
        raise ValueError(f"{path.name}: not a replay (signature {raw[:4].hex(' ')})")

    blocks_count = struct.unpack_from("<i", raw, 4)[0]
    offset = 8
    blocks = []
    for _ in range(blocks_count):
        size = struct.unpack_from("<i", raw, offset)[0]
        offset += 4
        blob = raw[offset:offset + size]
        offset += size
        if not blob:
            blocks.append(None)
            continue
        try:
            blocks.append(json.loads(blob))
        except (ValueError, UnicodeDecodeError):
            blocks.append(blob)

    # The first 8 bytes of the ciphertext are a seed field, skipped by convention.
    stream = zlib.decompress(_decrypt(raw[offset + 8:]))
    return Replay(meta=blocks[0], extra=blocks[1:], stream=stream, source=path)


def walk(stream: bytes) -> list[Packet]:
    """Split the decompressed stream into packets. Consumes the buffer exactly."""
    packets = []
    offset = 0
    total = len(stream)
    while offset + PACKET_HEADER <= total:
        size, ptype, clock = struct.unpack_from("<IIf", stream, offset)
        body = offset + PACKET_HEADER
        if body + size > total:
            break
        packets.append(Packet(type=ptype, clock=clock, payload=stream[body:body + size]))
        offset = body + size
    return packets
