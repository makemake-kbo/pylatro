"""Tests for action masking."""

from __future__ import annotations

import numpy as np
import pytest

from pylatro import add_consumable, create_run_state, load_game_data, select_blind, start_blind
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import consumable_subset_index


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture
def hand_play_state(game_data):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    return state


def test_blind_select_mask(game_data):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)

    assert mask[ActionRange.BLIND_PLAY] == 1, "Play should always be valid"
    assert mask[ActionRange.BLIND_SKIP] == 1, "Skip should be valid for Small blind"
    assert mask.sum() >= 2


def test_choose_action_mask(hand_play_state):
    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    assert mask[ActionRange.PLAY_SUBSET_START] == 1, "Should expose at least one play subset"
    assert mask[ActionRange.DISCARD_SUBSET_START] == 1, "Should expose at least one discard subset"
    # Consumable depends on state
    assert mask.sum() >= 2


def test_choose_action_subset_ranges_are_bounded(hand_play_state):
    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    play_mask = mask[ActionRange.PLAY_SUBSET_START:ActionRange.PLAY_SUBSET_END + 1]
    discard_mask = mask[ActionRange.DISCARD_SUBSET_START:ActionRange.DISCARD_SUBSET_END + 1]
    assert play_mask.sum() >= 1
    assert discard_mask.sum() >= 1
    assert play_mask.sum() <= len(play_mask)
    assert discard_mask.sum() <= len(discard_mask)


def test_hand_targeted_consumable_actions_are_unmasked(hand_play_state):
    add_consumable(hand_play_state, "c_lovers")

    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)
    action = encode_action(
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        0,
        consumable_subset_index((0,)),
    )

    assert mask[action] == 1


def test_aura_consumable_uses_hand_target_fallback(hand_play_state):
    add_consumable(hand_play_state, "c_aura")

    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)
    action = encode_action(
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        0,
        consumable_subset_index((0,)),
    )

    assert mask[action] == 1


def test_heuristic_can_pick_hand_targeted_consumable(hand_play_state):
    add_consumable(hand_play_state, "c_lovers")
    mask = compute_action_mask(hand_play_state, SubPhase.CHOOSE_ACTION)

    action = HeuristicAgent().select_action(hand_play_state, SubPhase.CHOOSE_ACTION, mask)

    assert ActionRange.CONSUMABLE_FLAT_START <= action <= ActionRange.CONSUMABLE_FLAT_END


def test_select_cards_mask_is_legacy_noop(hand_play_state):
    mask = compute_action_mask(
        hand_play_state,
        SubPhase.SELECT_CARDS,
        selected_cards=set(),
        pending_action="play",
    )
    assert mask.sum() == 0


def test_shop_mask_leave_always_valid(hand_play_state):
    # Simulate being in shop phase
    from pylatro import populate_shop, cash_out
    # This is a simplified test — just check leave is always valid in shop mask
    mask = compute_action_mask(hand_play_state, SubPhase.SHOP)
    assert mask[ActionRange.SHOP_LEAVE] == 1
