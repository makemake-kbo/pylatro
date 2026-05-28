from pylatro import create_run_state


def test_deck_variants_apply_expected_starting_effects() -> None:
    abandoned = create_run_state("AAAAAAAA", deck_key="b_abandoned")
    assert len(abandoned.deck_cards) == 40
    assert all(not card.is_face for card in abandoned.deck_cards)

    checkered = create_run_state("AAAAAAAA", deck_key="b_checkered")
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
