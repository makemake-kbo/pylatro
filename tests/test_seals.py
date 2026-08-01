from __future__ import annotations

from pylatro import create_run_state, discard_cards, load_game_data, start_blind
from pylatro_cli.controller import GameController


def test_purple_seal_generates_tarot_when_discarded() -> None:
    data = load_game_data()
    state = create_run_state("purple_seal_effect", data=data)
    start_blind(state, "Small")
    state.hand_cards[0].seal = "Purple"

    discard_cards(state, [0])

    assert len(state.consumables) == 1
    assert data.centers[state.consumables[0].center_key]["set"] == "Tarot"


def test_debuffed_purple_seal_does_not_generate_tarot_when_discarded() -> None:
    data = load_game_data()
    state = create_run_state("debuffed_purple_seal_effect", data=data)
    start_blind(state, "Small")
    state.hand_cards[0].seal = "Purple"
    state.hand_cards[0].debuff = True

    discard_cards(state, [0])

    assert state.consumables == []


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
