"""Shared exhaustive subset action definitions for hand play."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .constants import MAX_HAND_SIZE

HAND_SUBSETS: tuple[tuple[int, ...], ...] = tuple(
    combo
    for size in range(1, min(5, MAX_HAND_SIZE) + 1)
    for combo in combinations(range(MAX_HAND_SIZE), size)
)
NUM_HAND_SUBSETS = len(HAND_SUBSETS)

HAND_SUBSET_TO_INDEX = {subset: idx for idx, subset in enumerate(HAND_SUBSETS)}

HAND_SUBSET_MASKS = np.zeros((NUM_HAND_SUBSETS, MAX_HAND_SIZE), dtype=np.float32)
HAND_SUBSET_SIZES = np.zeros(NUM_HAND_SUBSETS, dtype=np.float32)
HAND_SUBSET_BITS = np.zeros(NUM_HAND_SUBSETS, dtype=np.uint16)
for idx, subset in enumerate(HAND_SUBSETS):
    HAND_SUBSET_SIZES[idx] = float(len(subset))
    bitmask = 0
    for slot in subset:
        HAND_SUBSET_MASKS[idx, slot] = 1.0
        bitmask |= 1 << slot
    HAND_SUBSET_BITS[idx] = bitmask


def subset_index(indices: tuple[int, ...] | list[int] | set[int]) -> int:
    """Return the canonical exhaustive subset index for sorted hand indices."""
    ordered = tuple(sorted(int(idx) for idx in indices))
    return HAND_SUBSET_TO_INDEX[ordered]


def subset_indices(index: int) -> tuple[int, ...]:
    """Return the hand-slot indices for an exhaustive subset id."""
    return HAND_SUBSETS[index]


def present_hand_bitmask(hand_size: int) -> int:
    """Return the bitmask for contiguous live hand slots [0, hand_size)."""
    if hand_size <= 0:
        return 0
    return (1 << hand_size) - 1


def forced_hand_bitmask(forced_slots: set[int] | tuple[int, ...] | list[int]) -> int:
    """Return the bitmask covering forced-selection hand slots."""
    bitmask = 0
    for slot in forced_slots:
        bitmask |= 1 << int(slot)
    return bitmask


def legal_subset_mask(hand_size: int, forced_slots: set[int] | tuple[int, ...] | list[int]) -> np.ndarray:
    """Return a boolean mask over exhaustive subsets legal for the current hand."""
    present_bits = present_hand_bitmask(hand_size)
    forced_bits = forced_hand_bitmask(forced_slots)
    all_bits = (1 << MAX_HAND_SIZE) - 1
    missing_bits = np.uint16(all_bits ^ present_bits)
    forced_bits_u16 = np.uint16(forced_bits)
    return ((HAND_SUBSET_BITS & missing_bits) == 0) & ((HAND_SUBSET_BITS & forced_bits_u16) == forced_bits_u16)
