"""Finding near-duplicate images, so that no test image has a twin in the training set.

Each image is shrunk to 9x8 grey pixels and turned into a 64-bit fingerprint (a "difference
hash": one bit per pair of neighbouring pixels, set when the right one is brighter). Photos
that look the same get fingerprints differing in only a few bits, even after resizing,
recompression or small colour changes. The number of differing bits is the Hamming distance.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def dhash(path: str | Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"cannot read image {path}")
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def near_duplicate_pairs(hashes: list[int], max_distance: int) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with i < j and Hamming distance <= max_distance."""
    h = np.array(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h) - 1):
        distances = np.bitwise_count(np.bitwise_xor(h[i + 1:], h[i]))
        for j in np.nonzero(distances <= max_distance)[0]:
            pairs.append((i, i + 1 + int(j), int(distances[j])))
    return pairs


def groups_from_pairs(n: int, pairs: list[tuple[int, int, int]]) -> list[int]:
    """Group ids: images connected by any chain of near-duplicate pairs share a group."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    return [find(i) for i in range(n)]
