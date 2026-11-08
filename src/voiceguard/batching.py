"""Batches without padding: every batch holds clips of exactly the same length.

WavLM Base+ normalises its first convolution over time (group norm), so zero padding added to make a batch
rectangular changes the features of the shorter clips. Grouping equal lengths avoids padding altogether.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def equal_length_batches(lengths: Sequence[int], batch_size: int) -> List[List[int]]:
    groups: Dict[int, List[int]] = {}
    for i, n in enumerate(lengths):
        groups.setdefault(int(n), []).append(i)
    batches = []
    for n in sorted(groups, reverse=True):
        idx = groups[n]
        batches.extend(idx[k:k + batch_size] for k in range(0, len(idx), batch_size))
    return batches
