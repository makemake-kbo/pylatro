from pylatro import (
    create_run_state,
    get_current_pool,
    populate_shop,
)


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
