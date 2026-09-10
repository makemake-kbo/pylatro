"""Card pool occupancy follows live cards, as Balatro Card:remove does."""

from pylatro import add_consumable, add_joker, create_run_state, get_current_pool, use_consumable
from pylatro.instances import remove_joker
from pylatro.models import PackState
from pylatro.pool import create_card_spec
from pylatro.shop import claim_pack_card, close_pack, finish_shop, refresh_shop


def test_used_planet_returns_to_pool_and_can_scale_again():
    state = create_run_state("pool_planets")
    for expected_level in range(2, 6):
        planet = add_consumable(state, "c_mercury")
        assert "c_mercury" not in get_current_pool(state, "Planet")[0]
        use_consumable(state, planet)
        assert "c_mercury" in get_current_pool(state, "Planet")[0]
        assert state.hands["Pair"]["level"] == expected_level


def test_last_duplicate_removal_releases_joker():
    state = create_run_state("pool_duplicates")
    first = add_joker(state, "j_joker")
    second = add_joker(state, "j_joker")
    remove_joker(state, first)
    assert "j_joker" not in get_current_pool(state, "Joker", rarity=0.1)[0]
    remove_joker(state, second)
    assert "j_joker" in get_current_pool(state, "Joker", rarity=0.1)[0]


def test_reroll_releases_old_offers_but_keeps_owned_cards():
    state = create_run_state("pool_reroll")
    add_consumable(state, "c_mercury")
    state.shop.cards = [create_card_spec(state, "Planet", forced_key="c_venus")]
    state.shop.joker_max = 0
    refresh_shop(state)
    pool, _ = get_current_pool(state, "Planet")
    assert "c_venus" in pool
    assert "c_mercury" not in pool


def test_pack_claim_releases_used_and_unselected_cards():
    state = create_run_state("pool_pack")
    cards = [create_card_spec(state, "Planet", forced_key=key) for key in ("c_mercury", "c_venus")]
    state.pack = PackState(
        booster_key="p_celestial_normal_1", state_name="PLANET_PACK", choices_remaining=1, cards=cards
    )
    claim_pack_card(state, 0)
    assert state.pack is None
    pool, _ = get_current_pool(state, "Planet")
    assert "c_mercury" in pool and "c_venus" in pool
    assert state.hands["Pair"]["level"] == 2


def test_pack_skip_and_shop_exit_keep_other_live_copies():
    state = create_run_state("pool_leave")
    add_consumable(state, "c_mercury")
    state.shop.cards = [create_card_spec(state, "Planet", forced_key="c_venus")]
    cards = [create_card_spec(state, "Planet", forced_key=key) for key in ("c_mercury", "c_venus")]
    state.pack = PackState(
        booster_key="p_celestial_normal_1", state_name="PLANET_PACK", choices_remaining=1, cards=cards
    )
    close_pack(state, skipped=True)
    assert "c_venus" not in get_current_pool(state, "Planet")[0]
    finish_shop(state)
    pool, _ = get_current_pool(state, "Planet")
    assert "c_venus" in pool and "c_mercury" not in pool


def test_blue_white_runner_uses_extra_hand():
    from pylatro_agent.training.fast_runner import FastRunner

    state = create_run_state("deck_data")
    runner = FastRunner(0, state.data, deck_key="b_blue", stake=1)
    assert runner.state.deck_key == "b_blue"
    assert runner.state.stake == 1
    assert runner.state.round_resets.hands == 5
    assert runner.state.round_resets.discards == 3
