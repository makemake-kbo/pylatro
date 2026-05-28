from pylatro import (
    add_consumable,
    add_joker,
    create_run_state,
    finish_shop,
    sell_owned_joker,
)


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
