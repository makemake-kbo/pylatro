from pylatro import create_run_state


def test_red_deck_seeded_run_fixture() -> None:
    state = create_run_state("AAAAAAAA")

    assert state.round_resets.blind_choices["Boss"] == "bl_manacle"
    assert state.current_voucher == "v_planet_merchant"
    assert state.round_resets.blind_tags == {"Small": "tag_skip", "Big": "tag_economy"}
    assert state.starting_params.hands == 4
    assert state.starting_params.discards == 4
    assert state.starting_params.joker_slots == 5
    assert len(state.deck_cards) == 52
    assert [card.front_key for card in state.deck_cards[:10]] == [
        "S_Q",
        "H_8",
        "D_8",
        "D_4",
        "H_Q",
        "D_Q",
        "D_5",
        "H_2",
        "S_T",
        "H_K",
    ]


def test_deck_variants_apply_expected_starting_effects() -> None:
    abandoned = create_run_state("AAAAAAAA", deck_key="b_abandoned")
    assert len(abandoned.deck_cards) == 40
    assert all(not card.is_face for card in abandoned.deck_cards)

    checkered = create_run_state("AAAAAAAA", deck_key="b_checkered")
    assert checkered.deck_cards[2].front_key == "H_8"
    assert all(card.suit in {"Spades", "Hearts"} for card in checkered.deck_cards)

    magic = create_run_state("AAAAAAAA", deck_key="b_magic")
    assert magic.consumable_keys == ["c_fool", "c_fool"]
    assert sorted(magic.used_vouchers) == ["v_crystal_ball"]

    black = create_run_state("AAAAAAAA", deck_key="b_black")
    assert black.starting_params.hands == 3
    assert black.starting_params.joker_slots == 6

    painted = create_run_state("AAAAAAAA", deck_key="b_painted")
    assert painted.starting_params.hand_size == 10
    assert painted.starting_params.joker_slots == 4

    plasma = create_run_state("AAAAAAAA", deck_key="b_plasma")
    assert plasma.starting_params.ante_scaling == 2

    erratic = create_run_state("AAAAAAAA", deck_key="b_erratic")
    assert [card.front_key for card in erratic.deck_cards[:5]] == [
        "S_T",
        "H_8",
        "D_7",
        "C_T",
        "H_9",
    ]


def test_stake_modifiers_apply_expected_flags() -> None:
    stake_1 = create_run_state("AAAAAAAA", stake=1)
    stake_6 = create_run_state("AAAAAAAA", stake=6)
    stake_8 = create_run_state("AAAAAAAA", stake=8)

    assert stake_1.modifiers == {}
    assert stake_6.modifiers == {
        "no_blind_reward": {"Small": True},
        "scaling": 3,
        "enable_eternals_in_shop": True,
    }
    assert stake_8.modifiers == {
        "no_blind_reward": {"Small": True},
        "scaling": 3,
        "enable_eternals_in_shop": True,
        "enable_perishables_in_shop": True,
        "enable_rentals_in_shop": True,
    }
