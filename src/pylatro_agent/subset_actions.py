"""Shared exhaustive subset action definitions for hand play."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .constants import MAX_CONSUMABLE_HAND_TARGETS, MAX_HAND_SIZE, NUM_CONSUMABLE_HAND_SUBSETS

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

# Consumable targeting subsets are the size-1..MAX_CONSUMABLE_HAND_TARGETS
# prefix of HAND_SUBSETS, since combinations enumerate by ascending size,
# slicing the first NUM_CONSUMABLE_HAND_SUBSETS entries gives exactly the
# valid targeting subsets in the same canonical order.
CONSUMABLE_HAND_SUBSETS: tuple[tuple[int, ...], ...] = HAND_SUBSETS[:NUM_CONSUMABLE_HAND_SUBSETS]
assert all(len(s) <= MAX_CONSUMABLE_HAND_TARGETS for s in CONSUMABLE_HAND_SUBSETS)
CONSUMABLE_HAND_SUBSET_TO_INDEX = {
    subset: idx for idx, subset in enumerate(CONSUMABLE_HAND_SUBSETS)
}
CONSUMABLE_HAND_SUBSET_SIZES = HAND_SUBSET_SIZES[:NUM_CONSUMABLE_HAND_SUBSETS].astype(np.int8)
CONSUMABLE_HAND_SUBSET_BITS = HAND_SUBSET_BITS[:NUM_CONSUMABLE_HAND_SUBSETS]
CONSUMABLE_HAND_SUBSET_MASKS = HAND_SUBSET_MASKS[:NUM_CONSUMABLE_HAND_SUBSETS]


def subset_index(indices: tuple[int, ...] | list[int] | set[int]) -> int:
    """Return the canonical exhaustive subset index for sorted hand indices."""
    ordered = tuple(sorted(int(idx) for idx in indices))
    return HAND_SUBSET_TO_INDEX[ordered]


def subset_indices(index: int) -> tuple[int, ...]:
    """Return the hand-slot indices for an exhaustive subset id."""
    return HAND_SUBSETS[index]


def consumable_subset_indices(index: int) -> tuple[int, ...]:
    """Return the hand-slot indices for a consumable-target subset id."""
    return CONSUMABLE_HAND_SUBSETS[index]


def consumable_subset_index(indices: tuple[int, ...] | list[int]) -> int:
    """Return the canonical consumable subset index for sorted hand indices."""
    ordered = tuple(sorted(int(idx) for idx in indices))
    return CONSUMABLE_HAND_SUBSET_TO_INDEX[ordered]


def legal_consumable_subset_mask(hand_size: int, min_size: int, max_size: int) -> np.ndarray:
    """Return a boolean mask over consumable subsets valid for this hand.

    A subset is valid iff all its card indices are live (< hand_size) and
    its size is in [min_size, max_size].
    """
    if hand_size <= 0 or max_size <= 0 or min_size > max_size:
        return np.zeros(NUM_CONSUMABLE_HAND_SUBSETS, dtype=bool)
    present_bits = present_hand_bitmask(hand_size)
    all_bits = (1 << MAX_HAND_SIZE) - 1
    missing_bits = np.uint16(all_bits ^ present_bits)
    size_ok = (min_size <= CONSUMABLE_HAND_SUBSET_SIZES) & (
        max_size >= CONSUMABLE_HAND_SUBSET_SIZES
    )
    indices_ok = (CONSUMABLE_HAND_SUBSET_BITS & missing_bits) == 0
    return size_ok & indices_ok


def present_hand_bitmask(hand_size: int) -> int:
    """Return the bitmask for contiguous live hand slots [0, hand_size).

    Hands larger than MAX_HAND_SIZE (possible via hand-size jokers/vouchers)
    are truncated: slots >= MAX_HAND_SIZE are not addressable by any subset
    action, matching the observation encoding.
    """
    if hand_size <= 0:
        return 0
    return (1 << min(hand_size, MAX_HAND_SIZE)) - 1


def forced_hand_bitmask(forced_slots: set[int] | tuple[int, ...] | list[int]) -> int:
    """Return the bitmask covering forced-selection hand slots.

    Forced slots beyond MAX_HAND_SIZE are dropped: they cannot be selected
    by any subset action, and requiring them would mask off every play.
    """
    bitmask = 0
    for slot in forced_slots:
        slot = int(slot)
        if 0 <= slot < MAX_HAND_SIZE:
            bitmask |= 1 << slot
    return bitmask


def legal_subset_mask(hand_size: int, forced_slots: set[int] | tuple[int, ...] | list[int]) -> np.ndarray:
    """Return a boolean mask over exhaustive subsets legal for the current hand."""
    present_bits = present_hand_bitmask(hand_size)
    forced_bits = forced_hand_bitmask(forced_slots)
    all_bits = (1 << MAX_HAND_SIZE) - 1
    missing_bits = np.uint16(all_bits ^ present_bits)
    forced_bits_u16 = np.uint16(forced_bits)
    return ((HAND_SUBSET_BITS & missing_bits) == 0) & ((HAND_SUBSET_BITS & forced_bits_u16) == forced_bits_u16)
