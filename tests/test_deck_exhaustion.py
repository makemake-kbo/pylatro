"""Exhausted games supply terminal labels instead of an empty policy mask."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pylatro import add_consumable, add_joker, load_game_data
from pylatro_agent.env import BalatroEnv
from pylatro_agent.reward import RewardConfig
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_cli.controller import GameController, GamePhase


@pytest.fixture(scope="module")
def data():
    return load_game_data()


@pytest.fixture
def last_card(data):
    ctrl = GameController(data=data)
    ctrl.new_run("LASTCARD", deck_key="b_blue")
    ctrl.select_blind("Small")
    ctrl.state.hand_cards = ctrl.state.hand_cards[:1]
    ctrl.state.draw_pile.clear()
    ctrl.state.current_round.hands_left = 3
    return ctrl


@pytest.mark.parametrize("action", ["play", "discard", "destroy"])
def test_empty_deck_and_hand_end_the_run(last_card, action):
    if action == "play":
        last_card.play_selected([0])
    elif action == "discard":
        last_card.discard_selected([0])
    else:
        add_consumable(last_card.state, "c_hanged_man")
        last_card.use_consumable_on(0, hand_targets=(0,))
    assert last_card.state.current_round.hands_left > 0
    assert not last_card.state.hand_cards and not last_card.state.draw_pile
    assert last_card.phase == GamePhase.GAME_OVER


def test_partial_last_hand_remains_playable(last_card):
    last_card.state.draw_pile = [last_card.state.deck_cards[-1]]
    last_card.play_selected([0])
    assert last_card.phase == GamePhase.HAND_PLAY
    assert last_card.state.hand_cards
    assert not last_card.state.draw_pile


def test_last_card_can_still_clear_the_blind(last_card):
    last_card.round_score = last_card.blind_target() - 1
    last_card.play_selected([0])
    assert last_card.blind_beaten()
    assert last_card.phase != GamePhase.GAME_OVER


def test_mr_bones_rescues_exhausted_round_without_an_extra_hand(last_card):
    add_joker(last_card.state, "j_mr_bones")
    last_card.round_score = last_card.blind_target() // 2
    hands = last_card.state.current_round.hands_left
    last_card.discard_selected([0])
    assert last_card.phase == GamePhase.SHOP
    assert "j_mr_bones" not in last_card.state.joker_keys
    # Small Blind pays $3 plus unused hands; no fictitious rescue hand pays out.
    assert last_card.state.current_round.round_dollars == hands + 3


def test_empty_deck_cannot_start_a_nonterminal_round(data):
    ctrl = GameController(data=data)
    ctrl.new_run("EMPTY")
    ctrl.state.draw_pile.clear()
    ctrl.state.deck_cards.clear()
    ctrl.select_blind("Small")
    assert ctrl.phase == GamePhase.GAME_OVER


def test_failed_training_seed_now_returns_a_complete_loss(data):
    fixture = json.loads((Path(__file__).parent / "fixtures/deck_exhaustion_8005054.json").read_text())
    reward = RewardConfig(objective="milestone", milestone_reward_budget=0, gamma=1)
    env = BalatroEnv(seed=fixture["seed"], deck_key="b_blue", data=data, reward_config=reward, enable_teacher=False)
    runner = FastRunner(fixture["seed"], data, deck_key="b_blue", raise_errors=True)
    try:
        obs, _ = env.reset(seed=fixture["seed"])
        for step, action in enumerate(fixture["actions"]):
            assert obs["action_mask"][action] and runner.compute_mask()[action]
            obs, returned_reward, terminated, truncated, info = env.step(action)
            runner.step(action)
            assert not truncated
            assert returned_reward == 0
            assert terminated == runner.done == (step == len(fixture["actions"]) - 1)
        assert not info["won"] and not runner.won
        assert info["ante"] == 5
        assert env.state.current_round.hands_left == 1
        assert env._controller.phase == runner.phase == GamePhase.GAME_OVER
    finally:
        env.close()
