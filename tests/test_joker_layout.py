"""Harness-owned joker ordering must be exact, side-effect free, and deterministic."""

from __future__ import annotations

import copy

from pylatro import add_joker, create_run_state, load_game_data, select_blind, start_blind
from pylatro.models import PlayingCard
from pylatro_agent import joker_layout


def _state_with(jokers, seed="joker_layout", data=None):
    data = data or load_game_data()
    state = create_run_state(seed, 1, "b_red", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    for key in jokers:
        add_joker(state, key)
    return state


def test_probe_does_not_disturb_engine_rng() -> None:
    """The scoring probe runs on a deepcopy; live RNG state must be untouched.

    If the probe ever leaked into the live ``PseudorandomState`` it would shift
    every subsequent card/shop/boss draw, silently changing runs for a given
    seed.
    """
    state = _state_with(("j_cavendish", "j_joker"))
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    before = copy.deepcopy(state.pseudorandom)
    joker_layout.apply_best_joker_order(state, (0,))
    after = state.pseudorandom

    assert after.values == before.values
    assert after.last_seed == before.last_seed
    assert after.draws_since_seed == before.draws_since_seed
    assert after.hashed_seed == before.hashed_seed


def test_probe_does_not_consume_hands_or_mutate_round() -> None:
    """Scoring a candidate order must not spend the hand it is evaluating."""
    state = _state_with(("j_cavendish", "j_joker"))
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    hands_left = state.current_round.hands_left
    hands_played = state.current_round.hands_played
    hand_size = len(state.hand_cards)

    joker_layout.apply_best_joker_order(state, (0,))

    assert state.current_round.hands_left == hands_left
    assert state.current_round.hands_played == hands_played
    assert len(state.hand_cards) == hand_size


def test_ordering_is_deterministic_and_idempotent() -> None:
    jokers = ("j_blueprint", "j_dusk", "j_hack", "j_idol")
    first = _state_with(jokers, seed="determinism")
    first.hand_cards = [PlayingCard(front_key="H_2", suit="Hearts", rank="2")]
    second = _state_with(jokers, seed="determinism")
    second.hand_cards = [PlayingCard(front_key="H_2", suit="Hearts", rank="2")]

    joker_layout.apply_best_joker_order(first, (0,))
    joker_layout.apply_best_joker_order(second, (0,))

    assert first.joker_keys == second.joker_keys
    # Reapplying finds nothing further to improve.
    assert joker_layout.apply_best_joker_order(first, (0,)) is False


def test_ordering_keeps_joker_keys_index_aligned() -> None:
    """``joker_keys`` is an index-aligned cache and must move with the instances."""
    state = _state_with(("j_cavendish", "j_joker", "j_blueprint"))
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    joker_layout.apply_best_joker_order(state, (0,))

    assert state.joker_keys == [joker.center_key for joker in state.jokers]


def test_single_joker_board_is_a_noop() -> None:
    state = _state_with(("j_joker",))
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    assert joker_layout.best_joker_order(state, (0,)) is None
    assert joker_layout.apply_best_joker_order(state, (0,)) is False


def test_empty_hand_selection_is_a_noop() -> None:
    state = _state_with(("j_cavendish", "j_joker"))
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    assert joker_layout.best_joker_order(state, ()) is None
