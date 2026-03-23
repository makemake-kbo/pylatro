from pylatro import (
    buy_shop_card,
    create_run_state,
    get_current_pool,
    open_booster_pack,
    populate_shop,
    redeem_voucher,
    refresh_shop,
    reroll_shop,
)


def test_shop_generation_and_reroll_fixture() -> None:
    state = create_run_state("AAAAAAAA")
    initial = refresh_shop(state)

    assert [(card.center_key, card.card_type, card.front_key, card.edition) for card in initial] == [
        ("j_bull", "Joker", None, None),
        ("j_faceless", "Joker", None, None),
    ]
    assert state.current_round.reroll_cost == 5
    assert state.dollars == 4

    rerolled = reroll_shop(state)
    assert [(card.center_key, card.card_type, card.front_key, card.edition) for card in rerolled] == [
        ("j_misprint", "Joker", None, None),
        ("c_sun", "Tarot", None, None),
    ]
    assert state.current_round.reroll_cost == 6
    assert state.dollars == -1


def test_buying_shop_cards_and_redeeming_vouchers_updates_run_state() -> None:
    state = create_run_state("AAAAAAAA")
    refresh_shop(state)

    bought = buy_shop_card(state, 0)
    assert bought.center_key == "j_bull"
    assert state.joker_keys == ["j_bull"]
    assert state.dollars == -2
    assert len(state.shop.cards) == 1

    redeem_voucher(state, "v_overstock_norm")
    assert state.shop.joker_max == 3
    assert state.starting_params.hands == 4
    assert state.starting_params.discards == 4


def test_seeded_shop_population_and_pack_generation_match_reference_fixture() -> None:
    state = create_run_state("AAAAAAAA")
    refresh_shop(state)
    shop = populate_shop(state)

    assert [(card.center_key, card.cost) for card in shop.vouchers] == [
        ("v_planet_merchant", 10),
    ]
    assert [(card.center_key, card.cost, card.booster_pos) for card in shop.boosters] == [
        ("p_buffoon_normal_1", 4, 1),
        ("p_celestial_jumbo_2", 6, 2),
    ]

    buffoon = open_booster_pack(state, 0)
    assert (buffoon.booster_key, buffoon.state_name, buffoon.choices_remaining) == (
        "p_buffoon_normal_1",
        "BUFFOON_PACK",
        1,
    )
    assert [(card.center_key, card.card_type, card.edition, card.eternal, card.perishable, card.rental) for card in buffoon.cards] == [
        ("j_gluttenous_joker", "Joker", None, False, False, False),
        ("j_zany", "Joker", None, False, False, False),
    ]

    state = create_run_state("AAAAAAAA")
    populate_shop(state)
    celestial = open_booster_pack(state, 1)
    assert (celestial.booster_key, celestial.state_name, celestial.choices_remaining) == (
        "p_celestial_jumbo_2",
        "PLANET_PACK",
        1,
    )
    assert [card.center_key for card in celestial.cards] == [
        "c_venus",
        "c_mercury",
        "c_mars",
        "c_uranus",
        "c_pluto",
    ]


def test_showman_and_live_shop_voucher_change_pool_availability() -> None:
    state = create_run_state("AAAAAAAA")
    pool, pool_key = get_current_pool(state, "Joker", rarity=0.1)
    assert pool_key == "Joker11"
    assert pool[0] == "j_joker"

    state.used_jokers["j_joker"] = True
    pool, _ = get_current_pool(state, "Joker", rarity=0.1)
    assert pool[0] == "UNAVAILABLE"

    state.joker_keys.append("j_ring_master")
    pool, _ = get_current_pool(state, "Joker", rarity=0.1)
    assert pool[0] == "j_joker"

    voucher_state = create_run_state("AAAAAAAA")
    populate_shop(voucher_state)
    voucher_pool, _ = get_current_pool(voucher_state, "Voucher")
    voucher_index = voucher_state.data.center_pools["Voucher"].index(
        voucher_state.data.centers[voucher_state.current_voucher]
    )
    assert voucher_pool[voucher_index] == "UNAVAILABLE"
