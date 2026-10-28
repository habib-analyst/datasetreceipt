"""Perceptual and exact hashing for duplicate detection.

The perceptual hash is the classic pHash construction: grayscale, shrink to
32x32, take a 2D DCT-II, keep the top-left 8x8 low-frequency block, and emit
one bit per coefficient (above or below the block median, DC excluded from
the median). It is implemented with NumPy only, so DatasetReceipt needs no
scipy and no imagehash dependency.

Two images that are re-encodings, mild resizes, or small brightness edits of
each other land within a few bits of Hamming distance; unrelated images sit
near 32 bits apart (half of 64, as expected for random bits).
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

_HASH_SIZE = 8  # 8x8 low-frequency block -> 64-bit hash
_SAMPLE_SIZE = 32  # images are reduced to 32x32 before the DCT


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis matrix of size ``n``."""
    k = np.arange(n)[:, None]
    x = np.arange(n)[None, :]
    basis = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    basis[0, :] *= 1.0 / np.sqrt(2.0)
    return basis * np.sqrt(2.0 / n)


_DCT32 = _dct_matrix(_SAMPLE_SIZE)


def perceptual_hash(image: Image.Image) -> int:
    """Return the 64-bit perceptual hash of a PIL image as a Python int."""
    gray = image.convert("L").resize((_SAMPLE_SIZE, _SAMPLE_SIZE), Image.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    dct = _DCT32 @ pixels @ _DCT32.T
    block = dct[:_HASH_SIZE, :_HASH_SIZE]
    median = float(np.median(block.flat[1:]))  # exclude the DC coefficient
    bits = (block > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def file_hash(data: bytes) -> str:
    """SHA-256 hex digest, used only for exact-duplicate detection."""
    return hashlib.sha256(data).hexdigest()


def pixels_hash(image: Image.Image) -> str:
    """Digest of the decoded RGB pixels, for images without source bytes."""
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    return (a ^ b).bit_count()


def find_near_duplicate_pairs(
    hashes: list[int], max_distance: int
) -> list[tuple[int, int, int]]:
    """All index pairs ``(i, j, distance)`` with Hamming distance <= threshold.

    Uses banding so the search stays practical on large splits: the 64-bit
    hash is split into ``max_distance + 1`` bands, and any pair within the
    threshold is guaranteed to agree on at least one whole band. Candidate
    pairs that share a band bucket are verified with the exact distance.
    """
    if max_distance < 0:
        raise ValueError("max_distance must be >= 0")
    n = len(hashes)
    if n < 2:
        return []

    n_bands = max_distance + 1
    band_bits = [64 // n_bands + (1 if b < 64 % n_bands else 0) for b in range(n_bands)]
    shifts: list[int] = []
    masks: list[int] = []
    shift = 0
    for width in band_bits:
        shifts.append(shift)
        masks.append((1 << width) - 1)
        shift += width

    candidates: set[tuple[int, int]] = set()
    for band in range(n_bands):
        buckets: dict[int, list[int]] = {}
        for idx, value in enumerate(hashes):
            key = (value >> shifts[band]) & masks[band]
            buckets.setdefault(key, []).append(idx)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    i, j = members[a], members[b]
                    candidates.add((i, j) if i < j else (j, i))

    pairs = []
    for i, j in sorted(candidates):
        distance = hamming(hashes[i], hashes[j])
        if distance <= max_distance:
            pairs.append((i, j, distance))
    return pairs
