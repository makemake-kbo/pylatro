from pylatro import (
    add_consumable,
    add_joker,
    cash_out,
    close_pack,
    create_run_state,
    finish_shop,
    open_booster_pack,
    play_cards,
    populate_shop,
    score_hand,
    sell_owned_joker,
    start_blind,
)
from pylatro.models import PlayingCard


def test_setting_blind_and_pack_hooks_mutate_state() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_cartomancer")
    add_joker(state, "j_marble")
    add_joker(state, "j_burglar")
    add_joker(state, "j_riff_raff")

    start_blind(state)

    assert state.consumable_keys == ["c_tower"]
    assert len(state.deck_cards) == 53
    assert state.current_round.discards_left == 0
    assert state.current_round.hands_left == 7
    assert state.joker_keys == ["j_cartomancer", "j_marble", "j_burglar", "j_riff_raff", "j_juggler"]

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_red_card")
    populate_shop(state)
    open_booster_pack(state, 0)
    close_pack(state, skipped=True)
    assert state.jokers[0].mult == 3


def test_finish_shop_and_sell_hooks_apply_inventory_side_effects() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_perkeo")
    add_consumable(state, "c_mercury")
    created = finish_shop(state)
    assert created == ["c_mercury"]
    assert state.consumable_keys == ["c_mercury", "c_mercury"]
    assert [consumable.edition for consumable in state.consumables] == [None, {"negative": True}]

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_campfire")
    add_joker(state, "j_joker")
    sell_owned_joker(state, 1)
    assert state.dollars == 5
    assert state.jokers[0].x_mult == 1.25

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_diet_cola")
    sell_owned_joker(state, 0)
    assert state.tags == ["tag_double"]
    assert state.dollars == 7


def test_round_end_hooks_update_scalars_and_values() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_gift")
    add_joker(state, "j_joker")
    add_consumable(state, "c_mercury")
    state.round_resets.blind = {"boss": False, "chips": 100}
    cash_out(state)
    assert [joker.sell_cost for joker in state.jokers] == [4, 2]
    assert [consumable.sell_cost for consumable in state.consumables] == [2]

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_rocket")
    state.round_resets.blind = {"boss": True, "chips": 100}
    cash_out(state)
    assert state.dollars == 5
    assert state.jokers[0].extra == {"dollars": 3, "increase": 2}

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_turtle_bean")
    state.round_resets.blind = {"boss": False, "chips": 100}
    cash_out(state)
    assert state.starting_params.hand_size == 12
    assert state.jokers[0].extra == {"h_mod": 1, "h_size": 4}


def test_stateful_scoring_hooks_mutate_cards_inventory_and_hand_levels() -> None:
    state = create_run_state("AAAAAAAA")
    state.probabilities["normal"] = 100
    add_joker(state, "j_space")
    start_blind(state)
    play_cards(state, state.hand_cards)
    assert (state.hands["Pair"]["level"], state.hands["Pair"]["chips"], state.hands["Pair"]["mult"]) == (2, 25, 3)

    state = create_run_state("AAAAAAAA")
    state.probabilities["normal"] = 100
    add_joker(state, "j_8_ball")
    start_blind(state)
    idx = next(i for i, card in enumerate(state.hand_cards) if card.rank == "8")
    play_cards(state, [idx])
    assert state.consumable_keys == ["c_chariot"]

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_dna")
    start_blind(state)
    play_cards(state, [0])
    assert len(state.deck_cards) == 53
    assert state.hand_cards[-1].front_key == "D_A"

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_vampire")
    cards = [
        PlayingCard(front_key="S_J", suit="Spades", rank="J", center_key="m_mult"),
        PlayingCard(front_key="H_J", suit="Hearts", rank="J", center_key="m_bonus"),
    ]
    result = score_hand(state, cards)
    assert result.total == 72
    assert [card.center_key for card in cards] == ["c_base", "c_base"]
    assert state.jokers[0].x_mult == 1.2
