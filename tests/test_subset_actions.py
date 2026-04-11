"""Tests for exhaustive hand subset action helpers."""

from __future__ import annotations

import numpy as np

from pylatro_agent.subset_actions import legal_subset_mask, subset_index, subset_indices


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
