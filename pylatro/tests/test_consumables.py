from pylatro import add_consumable, add_joker, create_run_state, score_hand, start_blind, use_consumable
from pylatro.models import PlayingCard


def test_targeted_consumables_mutate_cards_and_deck_state() -> None:
    state = create_run_state("AAAAAAAA")
    start_blind(state)

    add_consumable(state, "c_strength")
    result = use_consumable(state, 0, hand_targets=[0, 1])
    assert result.consumable_key == "c_strength"
    assert [card.front_key for card in state.hand_cards[:2]] == ["D_2", "S_A"]
    assert state.consumable_keys == []

    add_consumable(state, "c_death")
    use_consumable(state, 0, hand_targets=[0, 1])
    assert [card.front_key for card in state.hand_cards[:2]] == ["S_A", "S_A"]

    add_consumable(state, "c_cryptid")
    cryptid = use_consumable(state, 0, hand_targets=[0])
    assert len(state.deck_cards) == 54
    assert len(state.hand_cards) == 10
    assert [card.front_key for card in cryptid.created_cards] == ["S_A", "S_A"]


def test_hanged_man_and_planet_use_update_joker_and_usage_state() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_glass")
    start_blind(state)
    state.hand_cards[0].center_key = "m_glass"
    state.hand_cards[1].center_key = "m_glass"

    add_consumable(state, "c_hanged_man")
    use_consumable(state, 0, hand_targets=[0, 1])

    assert len(state.deck_cards) == 50
    assert state.jokers[0].x_mult == 2.5

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_constellation")
    add_consumable(state, "c_mercury")
    use_consumable(state, 0)

    assert state.jokers[0].x_mult == 1.1
    assert state.consumeable_usage_total == {"tarot": 0, "planet": 1, "spectral": 0, "tarot_planet": 1, "all": 1}
    assert state.last_tarot_planet == "c_mercury"


def test_generation_consumables_and_jokers_follow_seeded_pools() -> None:
    state = create_run_state("AAAAAAAA")
    add_consumable(state, "c_emperor")
    use_consumable(state, 0)
    assert state.consumable_keys == ["c_star"]

    state = create_run_state("AAAAAAAA")
    add_consumable(state, "c_judgement")
    use_consumable(state, 0)
    assert state.joker_keys == ["j_popcorn"]

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_joker")
    add_joker(state, "j_abstract")
    add_consumable(state, "c_ankh")
    use_consumable(state, 0)
    assert [state.data.centers[joker.center_key]["name"] for joker in state.jokers] == ["Joker", "Joker"]


def test_observatory_reads_held_planets_during_scoring() -> None:
    cards = [
        PlayingCard(front_key="D_3", suit="Diamonds", rank="3"),
        PlayingCard(front_key="H_3", suit="Hearts", rank="3"),
    ]

    state = create_run_state("AAAAAAAA")
    baseline = score_hand(state, cards)
    assert baseline.total == 32

    state = create_run_state("AAAAAAAA")
    state.used_vouchers["v_observatory"] = True
    add_consumable(state, "c_mercury")
    buffed = score_hand(state, cards)
    assert (buffed.chips, buffed.mult, buffed.total) == (16.0, 3.0, 48)
