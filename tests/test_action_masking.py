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

    assert mask[ActionRange.PLAY_SELECTION] == 1, "Should expose play selection"
    assert mask[ActionRange.DISCARD_SELECTION] == 1, "Should expose discard selection"
    # Consumable depends on state
    assert mask.sum() >= 2


def test_select_cards_mask_exposes_toggles_and_confirm(hand_play_state):
    mask = compute_action_mask(
        hand_play_state,
        SubPhase.SELECT_CARDS,
        selected_cards={0},
        pending_action="play",
    )
    assert mask[ActionRange.SELECT_CARD_START] == 1
    assert mask[ActionRange.SELECTION_CONFIRM] == 1
    assert mask[ActionRange.SELECTION_CANCEL] == 1


def test_select_cards_requires_selection_before_confirm(hand_play_state):
    mask = compute_action_mask(
        hand_play_state,
        SubPhase.SELECT_CARDS,
        selected_cards=set(),
        pending_action="play",
    )
    assert mask[ActionRange.SELECTION_CONFIRM] == 0
    assert mask[ActionRange.SELECTION_CANCEL] == 1


def test_shop_mask_leave_always_valid(hand_play_state):
    # Simulate being in shop phase
    from pylatro import populate_shop, cash_out
    # This is a simplified test — just check leave is always valid in shop mask
    mask = compute_action_mask(hand_play_state, SubPhase.SHOP)
    assert mask[ActionRange.SHOP_LEAVE] == 1
