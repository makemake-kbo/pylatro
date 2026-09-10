from copy import deepcopy

import pytest

from pylatro import add_joker, create_run_state, start_blind
from pylatro.runtime import apply_end_of_round
from pylatro_agent.heuristic_income import expected_passive_income


@pytest.mark.parametrize("blocker", ["j_burglar", "j_mystic_summit"])
def test_delayed_income_is_not_projected_when_policy_spends_all_discards(blocker):
    state = create_run_state("delayed_income")
    add_joker(state, "j_delayed_grat")
    assert expected_passive_income(state) > 0
    add_joker(state, blocker)
    assert expected_passive_income(state) == 0


@pytest.mark.parametrize("used,expected", [(0, 6), (1, 0)])
def test_delayed_payout_really_requires_zero_discard_actions(used, expected):
    state = create_run_state("delayed_payout", deck_key="b_blue")
    start_blind(state, "Small")
    state.current_round.discards_used = used
    state.current_round.discards_left = 3 - used
    trial = deepcopy(state, {id(state.data): state.data})
    add_joker(trial, "j_delayed_grat")
    assert apply_end_of_round(trial)["dollars"] - apply_end_of_round(state)["dollars"] == expected


def test_parking_income_tracks_faces_and_held_card_retriggers():
    state = create_run_state("parking_income")
    add_joker(state, "j_reserved_parking")
    normal = expected_passive_income(state)
    assert normal > 0
    state.deck_cards = [card for card in state.deck_cards if card.rank not in {"J", "Q", "K"}]
    assert expected_passive_income(state) == 0
    add_joker(state, "j_pareidolia")
    faces = expected_passive_income(state)
    assert faces > normal
    add_joker(state, "j_mime")
    assert expected_passive_income(state) > faces


@pytest.mark.parametrize("cost,worthwhile", [(10, False), (7, True)])
def test_late_interest_voucher_requires_profit_before_final_shop(cost, worthwhile):
    from pylatro.pool import create_card_spec
    from pylatro.shop import redeem_voucher
    from pylatro_agent.heuristic_income import interest_voucher_can_repay

    state = create_run_state("late_interest", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = "Small"
    state.dollars = 100
    item = create_card_spec(state, "Voucher", forced_key="v_seed_money")
    item.cost = cost
    state.shop.vouchers = [item]
    assert interest_voucher_can_repay(state, item) == worthwhile
    trial = deepcopy(state, {id(state.data): state.data})
    redeem_voucher(trial, item.center_key)
    # Only Small and Big payouts can fund another shop before winning Ante 8.
    for _ in range(2):
        apply_end_of_round(state)
        apply_end_of_round(trial)
    assert (trial.dollars > state.dollars) == worthwhile


def test_interest_multiplier_can_make_late_voucher_pay_back():
    from pylatro.pool import create_card_spec
    from pylatro_agent.heuristic_income import interest_voucher_can_repay

    state = create_run_state("late_moon_interest", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = "Small"
    item = create_card_spec(state, "Voucher", forced_key="v_seed_money")
    assert not interest_voucher_can_repay(state, item)
    add_joker(state, "j_to_the_moon")
    assert interest_voucher_can_repay(state, item)
    state.blind_on_deck = "Boss"
    assert not interest_voucher_can_repay(state, item)
