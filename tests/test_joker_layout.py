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
    assert joker_layout.apply_best_joker_order(first, (0,)).changed is False


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
    assert joker_layout.apply_best_joker_order(state, (0,)).changed is False


def test_empty_hand_selection_is_a_noop() -> None:
    state = _state_with(("j_cavendish", "j_joker"))
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    assert joker_layout.best_joker_order(state, ()) is None


def test_money_objective_never_sacrifices_a_clear() -> None:
    """The safety property: money is only taken from orders that still clear.

    Exhaustive over every candidate order for a copy+economy board, at every
    remaining target from trivial to unreachable. If the chosen order clears,
    the score-maximizing order must have cleared too - i.e. banking cash never
    turns a clearing play into a death.
    """
    state = _state_with(("j_blueprint", "j_bootstraps", "j_cavendish"), seed="money_safety")
    state.hand_cards = [
        PlayingCard(front_key="H_K", suit="Hearts", rank="K"),
        PlayingCard(front_key="H_Q", suit="Hearts", rank="Q"),
    ]
    count = min(len(state.jokers), 8)
    scored = [
        (order, *joker_layout._score_order(state, (0, 1), order))
        for order in joker_layout.order_candidates(state, count)
    ]
    best_chips = max(row[1] for row in scored)

    for target in (0, 1, best_chips // 2, best_chips, best_chips + 1, best_chips * 10):
        decision = joker_layout.plan_joker_order(state, (0, 1), remaining_target=target)
        assert decision.chips_forgone >= 0
        if decision.objective is joker_layout.OrderObjective.MONEY:
            # Taking money requires the chosen order to still clear...
            assert decision.chips >= target
            # ...and it is only ever offered when the score order cleared too.
            assert best_chips >= target
            assert decision.dollars_gained > 0


def test_unknown_remaining_target_forces_score_objective() -> None:
    """Without a target, whether a play clears is unknown - never guess."""
    state = _state_with(("j_blueprint", "j_bootstraps", "j_cavendish"), seed="no_target")
    state.hand_cards = [PlayingCard(front_key="H_K", suit="Hearts", rank="K")]

    decision = joker_layout.plan_joker_order(state, (0,), remaining_target=None)

    assert decision.objective is joker_layout.OrderObjective.SCORE
    assert decision.dollars_gained == 0
    assert decision.chips_forgone == 0


def test_unreachable_target_maximizes_score_not_money() -> None:
    state = _state_with(("j_blueprint", "j_bootstraps", "j_cavendish"), seed="unreachable")
    state.hand_cards = [PlayingCard(front_key="H_K", suit="Hearts", rank="K")]

    decision = joker_layout.plan_joker_order(state, (0,), remaining_target=10**9)

    assert decision.objective is joker_layout.OrderObjective.SCORE
    assert decision.clears is False
    assert decision.chips_forgone == 0


def test_decision_reports_no_change_when_current_order_is_chosen() -> None:
    state = _state_with(("j_joker", "j_sly"), seed="noop_decision")
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    decision = joker_layout.plan_joker_order(state, (0,), remaining_target=1.0)

    assert decision.changed is False
    assert decision.order is None


def test_money_objective_actually_fires_on_a_copy_plus_economy_board() -> None:
    """Proves the MONEY branch is reachable, so the safety test is not vacuous.

    Blueprint placed immediately before Business Card copies its per-face-card
    payout. That placement scores fewer chips than putting the xmult joker
    last, so the harness takes it only when the smaller total still clears.
    Surplus chips are worth nothing once a blind is cleared, which is why
    trading them for cash is strictly correct rather than merely acceptable.
    """
    state = _state_with(("j_blueprint", "j_business", "j_cavendish"), seed="money_fires")
    state.hand_cards = [
        PlayingCard(front_key="H_K", suit="Hearts", rank="K"),
        PlayingCard(front_key="S_K", suit="Spades", rank="K"),
        PlayingCard(front_key="D_Q", suit="Diamonds", rank="Q"),
    ]
    hand = (0, 1, 2)

    scored = [
        (order, *joker_layout._score_order(state, hand, order))
        for order in joker_layout.order_candidates(state, len(state.jokers))
    ]
    # Mirror the planner's own criteria rather than hard-coding chip totals.
    _, best_chips, score_order_dollars = max(scored, key=lambda row: row[1])
    _, money_chips, money_dollars = max(scored, key=lambda row: (row[2], row[1]))
    assert money_dollars > score_order_dollars, "fixture no longer trades chips for cash"
    assert money_chips < best_chips, "fixture no longer costs chips to take the cash"

    # Both orders clear -> take the cash.
    cheap = joker_layout.plan_joker_order(state, hand, remaining_target=money_chips)
    assert cheap.objective is joker_layout.OrderObjective.MONEY
    assert cheap.chips == money_chips
    assert cheap.dollars_gained == money_dollars - score_order_dollars
    assert cheap.chips_forgone == best_chips - money_chips

    # One chip more than the paying order can reach -> the clear wins outright.
    strict = joker_layout.plan_joker_order(state, hand, remaining_target=money_chips + 1)
    assert strict.objective is joker_layout.OrderObjective.SCORE
    assert strict.chips == best_chips
    assert strict.chips_forgone == 0
