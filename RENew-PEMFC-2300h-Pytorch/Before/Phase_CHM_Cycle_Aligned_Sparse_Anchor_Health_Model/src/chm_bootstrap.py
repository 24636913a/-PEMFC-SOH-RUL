"""Final V2A time-axis-preserving CHM block-multiplicity bootstrap."""

from __future__ import annotations

import math

import numpy as np

from chm_data import FrozenCHMData


EXPECTED_SEGMENTS = ((1, 24, 24), (26, 36, 11), (38, 49, 12), (51, 53, 3), (55, 74, 20), (76, 100, 25), (103, 125, 23), (127, 133, 7), (135, 150, 16), (152, 156, 5))


def seed_tuple(stream: int, outer_fold_index: int, replicate_index: int) -> tuple[int, int, int, int]:
    return (90210, stream, outer_fold_index, replicate_index)


def _segments(data: FrozenCHMData) -> list[tuple[int, ...]]:
    cycles = data.hou.loc[data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU"), "raw_cycle_id"].astype(int).tolist()
    groups: list[list[int]] = []
    for cycle in cycles:
        if not groups or cycle != groups[-1][-1] + 1:
            groups.append([cycle])
        else:
            groups[-1].append(cycle)
    segments = [tuple(group) for group in groups]
    observed = tuple((group[0], group[-1], len(group)) for group in segments)
    if observed != EXPECTED_SEGMENTS:
        raise ValueError("Frozen complete-cycle segments mismatch.")
    return segments


def bootstrap_weights(data: FrozenCHMData, requested_block_length: int, stream: int, outer_fold_index: int, replicate_index: int) -> tuple[np.ndarray, dict[int, int]]:
    if requested_block_length not in (4, 8):
        raise ValueError("Only frozen block lengths 4 and 8 are legal.")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed_tuple(stream, outer_fold_index, replicate_index))))
    multiplicity: dict[int, int] = {}
    for segment in _segments(data):
        n = len(segment)
        block = min(requested_block_length, n)
        draws = math.ceil(n / block)
        starts = rng.integers(0, n - block + 1, size=draws)
        expanded: list[int] = []
        for start in starts.tolist():
            expanded.extend(segment[start : start + block])
        count = {cycle: 0 for cycle in segment}
        for cycle in expanded[:n]:
            count[cycle] += 1
        if sum(count.values()) != n:
            raise ValueError("Per-segment multiplicity sum mismatch.")
        multiplicity.update(count)
    weights = np.ones(len(data.hou), dtype=np.float64)
    complete = data.hou["hou_kind"].eq("COMPLETE_CYCLE_HOU")
    weights[complete.to_numpy()] = data.hou.loc[complete, "raw_cycle_id"].astype(int).map(multiplicity).to_numpy(dtype=float)
    if np.any(weights < 0) or not np.all(weights[~complete.to_numpy()] == 1):
        raise ValueError("Bootstrap weights violate partial-HOU contract.")
    return weights, multiplicity


def optimizer_seed(replicate_index: int) -> int:
    return (42, 2024, 3407)[replicate_index % 3]
