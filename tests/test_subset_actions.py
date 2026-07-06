"""Tests for exhaustive hand subset action helpers."""

from __future__ import annotations

import numpy as np

from pylatro_agent.constants import MAX_HAND_SIZE
from pylatro_agent.subset_actions import (
    legal_consumable_subset_mask,
    legal_subset_mask,
    subset_index,
    subset_indices,
)


def test_subset_index_roundtrip() -> None:
    indices = (0, 2, 4, 7)
    idx = subset_index(indices)
    assert subset_indices(idx) == indices


def test_legal_subset_mask_respects_hand_size_and_forced_slots() -> None:
    mask = legal_subset_mask(3, forced_slots={1})

    assert mask[subset_index((1,))]
    assert mask[subset_index((0, 1))]
    assert mask[subset_index((1, 2))]
    assert not mask[subset_index((0,))]
    assert not mask[subset_index((2,))]
    assert not mask[subset_index((3,))]
    assert not mask[subset_index((1, 3))]
    assert int(np.count_nonzero(mask)) == 4


def test_legal_subset_mask_hand_larger_than_max() -> None:
    # Hand-size stacking (Turtle Bean, Juggler, ...) can exceed MAX_HAND_SIZE;
    # slots beyond the window are truncated instead of overflowing uint16.
    oversized = MAX_HAND_SIZE + 1

    mask = legal_subset_mask(oversized, forced_slots=set())
    full = legal_subset_mask(MAX_HAND_SIZE, forced_slots=set())
    assert np.array_equal(mask, full)

    # A forced slot outside the addressable window is dropped, not required.
    mask_forced = legal_subset_mask(oversized, forced_slots={MAX_HAND_SIZE})
    assert np.array_equal(mask_forced, full)

    cons_mask = legal_consumable_subset_mask(oversized, 1, 3)
    cons_full = legal_consumable_subset_mask(MAX_HAND_SIZE, 1, 3)
    assert np.array_equal(cons_mask, cons_full)
