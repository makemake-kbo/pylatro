"""Tests for action masking."""

from __future__ import annotations

import numpy as np
import pytest

from pylatro import create_run_state, load_game_data, select_blind, start_blind
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.masks import compute_action_mask


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

    assert mask[ActionRange.PLAY_HAND] == 1, "Should be able to play hand"
    assert mask[ActionRange.DISCARD] == 1, "Should be able to discard"
    # Consumable depends on state
    assert mask.sum() >= 2


def test_select_cards_mask_play(hand_play_state):
    selected = set()
    mask = compute_action_mask(
        hand_play_state, SubPhase.SELECT_CARDS,
        selected_cards=selected, pending_action="play",
    )

    # Should be able to toggle cards in hand
    hand_size = len(hand_play_state.hand_cards)
    for i in range(min(hand_size, 12)):
        assert mask[ActionRange.TOGGLE_CARD_START + i] == 1

    # Confirm should NOT be valid with 0 selected
    assert mask[ActionRange.SELECT_CONFIRM] == 0


def test_select_cards_mask_with_selection(hand_play_state):
    selected = {0, 1, 2}
    mask = compute_action_mask(
        hand_play_state, SubPhase.SELECT_CARDS,
        selected_cards=selected, pending_action="play",
    )

    # Confirm should be valid with 3 cards selected
    assert mask[ActionRange.SELECT_CONFIRM] == 1


def test_select_cards_max_5_for_play(hand_play_state):
    selected = {0, 1, 2, 3, 4}
    mask = compute_action_mask(
        hand_play_state, SubPhase.SELECT_CARDS,
        selected_cards=selected, pending_action="play",
    )

    # Can't select more (5 is max for play)
    for i in range(len(hand_play_state.hand_cards)):
        if i not in selected:
            assert mask[ActionRange.TOGGLE_CARD_START + i] == 0

    # Can deselect existing
    assert mask[ActionRange.TOGGLE_CARD_START + 0] == 1
    # Confirm is valid
    assert mask[ActionRange.SELECT_CONFIRM] == 1


def test_shop_mask_leave_always_valid(hand_play_state):
    # Simulate being in shop phase
    from pylatro import populate_shop, cash_out
    # This is a simplified test — just check leave is always valid in shop mask
    mask = compute_action_mask(hand_play_state, SubPhase.SHOP)
    assert mask[ActionRange.SHOP_LEAVE] == 1
