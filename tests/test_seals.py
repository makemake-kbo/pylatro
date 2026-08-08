from __future__ import annotations

from pylatro import add_consumable, create_run_state, discard_cards, load_game_data, start_blind
from pylatro.runtime import consumable_limit
from pylatro_cli.controller import GameController


def test_purple_seal_generates_tarot_when_discarded() -> None:
    data = load_game_data()
    state = create_run_state("purple_seal_effect", data=data)
    start_blind(state, "Small")
    state.hand_cards[0].seal = "Purple"

    result = discard_cards(state, [0])

    assert len(state.consumables) == 1
    assert data.centers[state.consumables[0].center_key]["set"] == "Tarot"
    assert result.purple_seals_activated == 1
    assert result.generated_consumables == [state.consumables[0].center_key]


def test_debuffed_purple_seal_does_not_generate_tarot_when_discarded() -> None:
    data = load_game_data()
    state = create_run_state("debuffed_purple_seal_effect", data=data)
    start_blind(state, "Small")
    state.hand_cards[0].seal = "Purple"
    state.hand_cards[0].debuff = True

    result = discard_cards(state, [0])

    assert state.consumables == []
    assert result.purple_seals_activated == 0
    assert result.generated_consumables == []


def test_purple_seal_reports_only_successful_generation() -> None:
    data = load_game_data()
    state = create_run_state("full_purple_seal_effect", data=data)
    start_blind(state, "Small")
    state.hand_cards[0].seal = "Purple"
    for _ in range(consumable_limit(state)):
        add_consumable(state, "c_fool")

    result = discard_cards(state, [0])

    assert result.purple_seals_activated == 1
    assert result.generated_consumables == []
    assert len(state.consumables) == consumable_limit(state)


def test_blue_seal_generates_winning_hands_planet() -> None:
    data = load_game_data()
    controller = GameController(data=data)
    controller.new_run("blue_seal_effect")
    controller.select_blind("Small")
    state = controller.state
    assert state is not None
    state.hand_cards[0].seal = "Blue"
    state.hands["High Card"]["chips"] = 1_000
    state.hands["High Card"]["mult"] = 10

    result = controller.play_selected([1])

    assert controller.blind_beaten()
    assert result.score.hand_name == "High Card"
    assert len(state.consumables) == 1
    planet = data.centers[state.consumables[0].center_key]
    assert planet["set"] == "Planet"
    assert planet["config"]["hand_type"] == "High Card"
    assert result.blue_seals_activated == 1
    assert result.blue_planets_generated == [state.consumables[0].center_key]


def test_blue_seal_activation_is_reported_when_inventory_blocks_generation() -> None:
    controller = _winning_controller("full_blue_seal_effect")
    state = controller.state
    assert state is not None
    state.hand_cards[0].seal = "Blue"
    for _ in range(consumable_limit(state)):
        add_consumable(state, "c_fool")

    result = controller.play_selected([1])

    assert result.blue_seals_activated == 1
    assert result.blue_planets_generated == []


def _winning_controller(seed: str) -> GameController:
    controller = GameController(data=load_game_data())
    controller.new_run(seed)
    controller.select_blind("Small")
    state = controller.state
    assert state is not None
    state.hands["High Card"]["chips"] = 1_000
    state.hands["High Card"]["mult"] = 10
    state.modifiers["no_blind_reward"] = {"Small": True}
    state.modifiers["money_per_hand"] = 0
    state.modifiers["no_interest"] = True
    return controller


def test_held_gold_card_pays_once_on_the_clearing_hand() -> None:
    controller = _winning_controller("held_gold_payout")
    state = controller.state
    assert state is not None
    state.hand_cards[0].center_key = "m_gold"
    dollars_before = state.dollars

    result = controller.play_selected([1])

    assert controller.blind_beaten()
    assert result.held_gold_count == 1
    assert result.held_gold_payout == 3
    assert state.dollars == dollars_before + 3
    controller.cash_out()
    assert state.dollars == dollars_before + 3


def test_held_gold_settlement_cannot_repeat_before_cash_out() -> None:
    controller = _winning_controller("held_gold_double_play")
    state = controller.state
    assert state is not None
    state.hand_cards[0].center_key = "m_gold"
    state.hand_cards[1].center_key = "m_gold"
    dollars_before = state.dollars

    first = controller.play_selected([2])
    second = controller.play_selected([3])

    assert first.held_gold_count == 2
    assert first.held_gold_payout == 6
    assert second.held_gold_count == 0
    assert second.held_gold_payout == 0
    assert state.dollars == dollars_before + 6


def test_played_or_debuffed_gold_cards_do_not_pay() -> None:
    played_controller = _winning_controller("played_gold_no_payout")
    played_state = played_controller.state
    assert played_state is not None
    played_state.hand_cards[0].center_key = "m_gold"
    played_result = played_controller.play_selected([0])

    debuffed_controller = _winning_controller("debuffed_gold_no_payout")
    debuffed_state = debuffed_controller.state
    assert debuffed_state is not None
    debuffed_state.hand_cards[0].center_key = "m_gold"
    debuffed_state.hand_cards[0].debuff = True
    debuffed_result = debuffed_controller.play_selected([1])

    assert played_result.held_gold_count == 0
    assert played_result.held_gold_payout == 0
    assert debuffed_result.held_gold_count == 0
    assert debuffed_result.held_gold_payout == 0
