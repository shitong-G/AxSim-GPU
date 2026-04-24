#!/usr/bin/env python3
"""Shared AXPI010 primary-input pattern file (CPU Verilog + GPU axsim)."""

from __future__ import annotations

import random
import struct
from pathlib import Path
from typing import List, Tuple

MAGIC = b"AXPI010\0"


def generate_pi_packed(seed: int, num_pi_bits: int, num_patterns: int) -> List[int]:
    """Same RNG as build_tb; packed row k = LSB-index-k PI bit (matches GPU PI node k)."""
    rng = random.Random(seed)
    num_blocks = (num_patterns + 63) // 64
    packed = [0] * (num_pi_bits * num_blocks)
    for p in range(num_patterns):
        bits = "".join("1" if rng.randint(0, 1) else "0" for _ in range(num_pi_bits))
        for k in range(num_pi_bits):
            bit = 1 if bits[num_pi_bits - 1 - k] == "1" else 0
            blk = p >> 6
            pos = p & 63
            packed[k * num_blocks + blk] |= (bit << pos)
    return packed


def truncate_packed_pi(
    packed: List[int], num_pi_bits: int, num_patterns_full: int, num_patterns_use: int
) -> List[int]:
    """Keep only the first num_patterns_use rows; file may contain more (e.g. 4M AXPI, 100k eval)."""
    if num_patterns_use <= 0:
        raise ValueError("num_patterns_use must be positive")
    if num_patterns_use > num_patterns_full:
        raise ValueError(
            f"need {num_patterns_use} patterns but file has only {num_patterns_full}"
        )
    if num_patterns_use == num_patterns_full:
        return packed
    nb_full = (num_patterns_full + 63) // 64
    nb_use = (num_patterns_use + 63) // 64
    out: List[int] = []
    for k in range(num_pi_bits):
        base = k * nb_full
        out.extend(packed[base : base + nb_use])
    return out


def read_axpi010(path: Path) -> Tuple[int, int, List[int]]:
    raw = path.read_bytes()
    if len(raw) < 8 + 4 + 8:
        raise ValueError("AXPI file too small")
    if raw[:8] != MAGIC:
        raise ValueError("bad AXPI magic")
    num_pi_bits = struct.unpack_from("<I", raw, 8)[0]
    num_patterns = struct.unpack_from("<Q", raw, 12)[0]
    payload = raw[20:]
    need = num_pi_bits * ((num_patterns + 63) // 64) * 8
    if len(payload) < need:
        raise ValueError(f"AXPI payload short: need {need} got {len(payload)}")
    nwords = need // 8
    packed = list(struct.unpack(f"<{nwords}Q", payload[:need]))
    return num_pi_bits, int(num_patterns), packed


def write_axpi010(path: Path, seed: int, num_pi_bits: int, num_patterns: int) -> None:
    packed = generate_pi_packed(seed, num_pi_bits, num_patterns)
    num_blocks = (num_patterns + 63) // 64
    need = num_pi_bits * num_blocks
    if len(packed) != need:
        raise RuntimeError("internal: packed size mismatch")
    buf = bytearray(MAGIC)
    buf.extend(struct.pack("<I", num_pi_bits))
    buf.extend(struct.pack("<Q", num_patterns))
    for w in packed:
        buf.extend(struct.pack("<Q", w))
    path.write_bytes(buf)


def write_axpi010_counting_pi(path: Path, num_pi_bits: int, num_patterns: int) -> None:
    """Each pattern index p uses PI unsigned value p (exhaustive sweep for small adders/muls).

    Requires num_patterns <= 2**num_pi_bits so every value fits. Matches GPU PI bit order in
    pattern_integer_at (bit k = (value >> k) & 1).
    """
    if num_patterns <= 0:
        raise ValueError("num_patterns must be positive")
    if num_patterns > (1 << num_pi_bits):
        raise ValueError("num_patterns exceeds 2**num_pi_bits")
    num_blocks = (num_patterns + 63) // 64
    packed = [0] * (num_pi_bits * num_blocks)
    for p in range(num_patterns):
        val = p
        for k in range(num_pi_bits):
            bit = (val >> k) & 1
            blk = p >> 6
            pos = p & 63
            packed[k * num_blocks + blk] |= (bit << pos)
    buf = bytearray(MAGIC)
    buf.extend(struct.pack("<I", num_pi_bits))
    buf.extend(struct.pack("<Q", num_patterns))
    for w in packed:
        buf.extend(struct.pack("<Q", w))
    path.write_bytes(buf)


def pattern_integer_at(p: int, packed: List[int], num_pi_bits: int, num_blocks: int) -> int:
    """Reconstruct unsigned PI bus value (LSB = bit 0) for pattern index p."""
    val = 0
    for b in range(num_pi_bits):
        w = packed[b * num_blocks + (p >> 6)]
        bit = (w >> (p & 63)) & 1
        val |= bit << b
    return val


def write_stim_hex_from_packed(
    path: Path, packed: List[int], num_pi_bits: int, num_patterns: int
) -> None:
    """Stream one hex word per line for $readmemh (avoids multi-GB inline Verilog)."""
    if num_patterns <= 0:
        raise ValueError("num_patterns must be positive")
    num_blocks = (num_patterns + 63) // 64
    nd = (num_pi_bits + 3) // 4
    with path.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        for p in range(num_patterns):
            val = pattern_integer_at(p, packed, num_pi_bits, num_blocks)
            f.write(f"{val:0{nd}x}\n")
