from __future__ import annotations

from pylatro import (
    add_consumable,
    add_joker,
    create_run_state,
    get_blind_amount,
    load_game_data,
    select_blind,
    start_blind,
)
from pylatro.models import PackState, PlayingCard, ShopCard
from pylatro_agent import joker_layout
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import subset_indices


def _shop_card(center_key: str, card_type: str, cost: int = 3) -> ShopCard:
    return ShopCard(center_key=center_key, card_type=card_type, cost=cost, base_cost=cost)


def test_ante_one_plays_small_blind_for_juggle_tag() -> None:
    data = load_game_data()
    state = create_run_state("juggle_skip", data=data)
    state.blind_on_deck = "Small"
    state.round_resets.blind_tags["Small"] = "tag_juggle"

    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)

    assert HeuristicAgent().select_action(state, SubPhase.BLIND_SELECT, mask) == ActionRange.BLIND_PLAY


def test_ante_one_skips_for_economy_tag() -> None:
    data = load_game_data()
    state = create_run_state("economy_skip", data=data)
    state.blind_on_deck = "Small"
    state.round_resets.blind_tags["Small"] = "tag_economy"

    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)

    assert HeuristicAgent().select_action(state, SubPhase.BLIND_SELECT, mask) == ActionRange.BLIND_SKIP


def test_ante_one_plays_big_blind_instead_of_juggle_skip() -> None:
    data = load_game_data()
    state = create_run_state("juggle_big_play", data=data)
    state.blind_on_deck = "Big"
    state.round_resets.blind_tags["Big"] = "tag_juggle"

    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)

    assert HeuristicAgent().select_action(state, SubPhase.BLIND_SELECT, mask) == ActionRange.BLIND_PLAY


def test_ante_one_prefers_riff_raff_to_buffoon_pack() -> None:
    data = load_game_data()
    state = create_run_state("riff_raff_engine", data=data)
    state.dollars = 10
    state.shop.cards = [_shop_card("j_riff_raff", "Joker", cost=6)]
    state.shop.boosters = [_shop_card("p_buffoon_normal_1", "Booster", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START


def test_ante_one_does_not_buy_inactive_vampire_before_basic_score() -> None:
    data = load_game_data()
    state = create_run_state("inactive_vampire", data=data)
    state.dollars = 10
    state.shop.cards = [_shop_card("j_vampire", "Joker", cost=7)]
    state.shop.boosters = [_shop_card("p_buffoon_normal_1", "Booster", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START + 1


def test_ante_one_buys_immediate_popcorn_instead_of_rerolling() -> None:
    data = load_game_data()
    state = create_run_state("popcorn_is_immediate", data=data)
    state.blind_on_deck = "Boss"
    state.dollars = 12
    state.shop.cards = [_shop_card("j_popcorn", "Joker", cost=7)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START


def test_full_board_does_not_sell_when_upgrade_remains_unaffordable() -> None:
    data = load_game_data()
    state = create_run_state("unaffordable_replacement", data=data)
    state.round_resets.ante = 2
    state.dollars = 2
    for center_key in ("j_mystic_summit", "j_runner", "j_supernova", "j_sly", "j_baseball"):
        add_joker(state, center_key)
    state.shop.cards = [_shop_card("j_odd_todd", "Joker", cost=7)]

    mask = compute_action_mask(state, SubPhase.SHOP)
    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action != ActionRange.SHOP_SELL_JOKER_START + 4


def test_ante_one_emergency_reroll_keeps_cash_buffer() -> None:
    data = load_game_data()
    state = create_run_state("early_reroll_buffer", data=data)
    state.dollars = state.current_round.reroll_cost

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_LEAVE


def test_ante_one_missing_core_can_reroll_to_three_dollars() -> None:
    data = load_game_data()
    state = create_run_state("early_three_dollar_reroll", data=data)
    state.dollars = state.current_round.reroll_cost + 3
    add_joker(state, "j_sly")

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_REROLL


def test_ante_one_bull_does_not_reroll_away_scoring_cash() -> None:
    data = load_game_data()
    state = create_run_state("early_bull_cash_floor", data=data)
    state.dollars = 8
    add_joker(state, "j_bull")

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_LEAVE


def test_final_ante_reroll_keeps_purchase_cash() -> None:
    data = load_game_data()
    state = create_run_state("final_reroll_purchase_buffer", data=data)
    state.round_resets.ante = state.win_ante
    state.dollars = state.current_round.reroll_cost
    add_joker(state, "j_joker")

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_LEAVE


def test_riff_raff_sells_disposable_roll_to_reopen_slot() -> None:
    data = load_game_data()
    state = create_run_state("riff_raff_sell_cycle", data=data)
    for center_key in ("j_riff_raff", "j_sly", "j_joker", "j_cavendish", "j_8_ball"):
        add_joker(state, center_key)

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_SELL_JOKER_START + 4


def test_full_build_does_not_buy_generic_standard_or_buffoon_pack() -> None:
    data = load_game_data()
    state = create_run_state("full_build_generic_packs", data=data)
    state.round_resets.ante = 5
    state.dollars = 40
    for center_key in ("j_joker", "j_sly", "j_odd_todd", "j_mystic_summit", "j_cavendish"):
        add_joker(state, center_key)
    for card, seal in zip(state.deck_cards[:3], ("Blue", "Purple", "Blue"), strict=True):
        card.seal = seal
    state.shop.boosters = [
        _shop_card("p_buffoon_normal_1", "Booster", cost=4),
        _shop_card("p_standard_normal_1", "Booster", cost=4),
    ]

    mask = compute_action_mask(state, SubPhase.SHOP)
    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action not in {ActionRange.SHOP_BUY_START, ActionRange.SHOP_BUY_START + 1}


def test_early_scoring_engine_preserves_economy_over_standard_pack() -> None:
    data = load_game_data()
    state = create_run_state("early_standard_deck_building", data=data)
    state.dollars = 8
    add_joker(state, "j_bull")
    state.shop.boosters = [_shop_card("p_standard_normal_1", "Booster", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) != ActionRange.SHOP_BUY_START


def test_midgame_surplus_cash_digs_for_priority_seals() -> None:
    data = load_game_data()
    state = create_run_state("midgame_standard_seal_hunt", data=data)
    state.round_resets.ante = 4
    state.dollars = 30
    add_joker(state, "j_bull")
    state.shop.boosters = [_shop_card("p_standard_normal_1", "Booster", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START


def test_midgame_missing_xmult_does_not_buy_paid_arcana() -> None:
    data = load_game_data()
    state = create_run_state("midgame_arcana_without_scaling", data=data)
    state.round_resets.ante = 4
    state.dollars = 40
    add_joker(state, "j_sly")
    add_joker(state, "j_joker")
    state.shop.cards = [_shop_card("c_magician", "Tarot", cost=3)]
    state.shop.boosters = [_shop_card("p_arcana_normal_1", "Booster", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)
    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action not in {ActionRange.SHOP_BUY_START, ActionRange.SHOP_BUY_START + 1}


def test_midgame_scaled_engine_can_buy_paid_arcana() -> None:
    data = load_game_data()
    state = create_run_state("midgame_arcana_with_scaling", data=data)
    state.round_resets.ante = 4
    state.dollars = 40
    add_joker(state, "j_sly")
    add_joker(state, "j_joker")
    add_joker(state, "j_cavendish")
    state.shop.boosters = [_shop_card("p_arcana_normal_1", "Booster", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START


def test_midgame_safe_score_margin_can_buy_arcana_without_xmult() -> None:
    data = load_game_data()
    state = create_run_state("midgame_arcana_safe_margin", data=data)
    state.round_resets.ante = 4
    state.dollars = 40
    add_joker(state, "j_sly")
    add_joker(state, "j_joker")
    state.shop.boosters = [_shop_card("p_arcana_normal_1", "Booster", cost=4)]
    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(
        state,
        SubPhase.SHOP,
        mask,
        round_score=100_000,
    )

    assert action == ActionRange.SHOP_BUY_START


def test_midgame_preserves_last_joker_slot_for_xmult() -> None:
    data = load_game_data()
    state = create_run_state("midgame_last_slot_for_xmult", data=data)
    state.round_resets.ante = 3
    state.dollars = 30
    for center_key in ("j_sly", "j_joker", "j_odd_todd", "j_walkie_talkie"):
        add_joker(state, center_key)
    state.shop.cards = [_shop_card("j_ride_the_bus", "Joker", cost=6)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) != ActionRange.SHOP_BUY_START


def test_midgame_missing_xmult_preserves_slot_over_economy_joker() -> None:
    data = load_game_data()
    state = create_run_state("midgame_no_economy_before_xmult", data=data)
    state.round_resets.ante = 4
    state.dollars = 30
    for center_key in ("j_sly", "j_joker", "j_odd_todd", "j_walkie_talkie"):
        add_joker(state, center_key)
    state.shop.cards = [_shop_card("j_egg", "Joker", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) != ActionRange.SHOP_BUY_START


def test_standard_pack_seal_priority_changes_with_ante() -> None:
    data = load_game_data()
    state = create_run_state("standard_seal_priority", data=data)
    center = data.centers["c_base"]
    agent = HeuristicAgent()

    early = {
        seal: agent._score_pack_card(state, center, seal=seal)
        for seal in (None, "Red", "Gold", "Blue", "Purple")
    }
    state.round_resets.ante = 7
    late_red = agent._score_pack_card(state, center, seal="Red")

    assert early["Purple"] > early["Blue"] > early["Gold"] > early["Red"] > early[None]
    assert early["Blue"] > late_red > early["Gold"]


def test_low_value_nonstandard_pack_is_skipped() -> None:
    data = load_game_data()
    state = create_run_state("skip_low_value_celestial", data=data)
    state.pack = PackState(
        booster_key="p_celestial_normal_1",
        state_name="PLANET_PACK",
        choices_remaining=1,
        cards=[_shop_card("c_earth", "Planet", cost=0)],
    )

    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)

    assert HeuristicAgent().select_action(state, SubPhase.BOOSTER_PACK, mask) == ActionRange.PACK_SKIP


def test_purple_seal_is_discarded_for_tarot_generation() -> None:
    data = load_game_data()
    state = create_run_state("purple_seal_discard", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.hand_cards[0].seal = "Purple"
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    action = HeuristicAgent().select_action(state, SubPhase.CHOOSE_ACTION, mask)

    assert ActionRange.DISCARD_SUBSET_START <= action <= ActionRange.DISCARD_SUBSET_END
    assert 0 in subset_indices(action - ActionRange.DISCARD_SUBSET_START)


def test_ante_one_banks_pair_when_it_is_close_to_required_pace() -> None:
    data = load_game_data()
    state = create_run_state("bank_early_pair", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.current_round.hands_left = 3
    state.hand_cards = [
        PlayingCard(front_key="S_A", suit="Spades", rank="A"),
        PlayingCard(front_key="H_J", suit="Hearts", rank="J"),
        PlayingCard(front_key="D_9", suit="Diamonds", rank="9"),
        PlayingCard(front_key="C_6", suit="Clubs", rank="6"),
        PlayingCard(front_key="S_6", suit="Spades", rank="6"),
        PlayingCard(front_key="H_5", suit="Hearts", rank="5"),
        PlayingCard(front_key="D_3", suit="Diamonds", rank="3"),
        PlayingCard(front_key="C_2", suit="Clubs", rank="2"),
    ]
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    action = HeuristicAgent().select_action(
        state,
        SubPhase.CHOOSE_ACTION,
        mask,
        round_score=124,
    )

    assert ActionRange.PLAY_SUBSET_START <= action <= ActionRange.PLAY_SUBSET_END
    played = subset_indices(action - ActionRange.PLAY_SUBSET_START)
    assert state.hand_cards[3].rank == state.hand_cards[4].rank
    assert {3, 4}.issubset(played)


def test_mystic_summit_exhausts_discards_before_banking_made_hand() -> None:
    data = load_game_data()
    state = create_run_state("activate_mystic_summit", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_mystic_summit")
    state.hand_cards = [
        PlayingCard(front_key="S_J", suit="Spades", rank="J"),
        PlayingCard(front_key="H_J", suit="Hearts", rank="J"),
        PlayingCard(front_key="D_J", suit="Diamonds", rank="J"),
        PlayingCard(front_key="S_8", suit="Spades", rank="8"),
        PlayingCard(front_key="H_8", suit="Hearts", rank="8"),
        PlayingCard(front_key="D_6", suit="Diamonds", rank="6"),
        PlayingCard(front_key="C_4", suit="Clubs", rank="4"),
        PlayingCard(front_key="S_2", suit="Spades", rank="2"),
    ]
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    action = HeuristicAgent().select_action(state, SubPhase.CHOOSE_ACTION, mask, round_score=0)

    assert ActionRange.DISCARD_SUBSET_START <= action <= ActionRange.DISCARD_SUBSET_END
    discarded = subset_indices(action - ActionRange.DISCARD_SUBSET_START)
    assert set(discarded).isdisjoint({0, 1, 2, 3, 4})


def test_square_joker_plays_exactly_four_cards_to_scale() -> None:
    data = load_game_data()
    state = create_run_state("scale_square", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_square")
    state.current_round.discards_left = 0
    state.hand_cards = [
        PlayingCard(front_key="S_A", suit="Spades", rank="A"),
        PlayingCard(front_key="H_A", suit="Hearts", rank="A"),
        PlayingCard(front_key="D_K", suit="Diamonds", rank="K"),
        PlayingCard(front_key="C_Q", suit="Clubs", rank="Q"),
        PlayingCard(front_key="S_J", suit="Spades", rank="J"),
        PlayingCard(front_key="H_8", suit="Hearts", rank="8"),
        PlayingCard(front_key="D_5", suit="Diamonds", rank="5"),
        PlayingCard(front_key="C_2", suit="Clubs", rank="2"),
    ]
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)

    action = HeuristicAgent().select_action(state, SubPhase.CHOOSE_ACTION, mask, round_score=0)

    assert ActionRange.PLAY_SUBSET_START <= action <= ActionRange.PLAY_SUBSET_END
    assert len(subset_indices(action - ActionRange.PLAY_SUBSET_START)) == 4


def test_runner_and_crazy_joker_form_dedicated_straight_engine() -> None:
    data = load_game_data()
    state = create_run_state("runner_crazy_straight", data=data)
    add_joker(state, "j_runner")
    add_joker(state, "j_crazy")

    assert HeuristicAgent()._get_main_hand_type(state) == "Straight"


def test_debuffed_runner_does_not_form_dedicated_straight_engine() -> None:
    data = load_game_data()
    state = create_run_state("debuffed_runner_crazy_straight", data=data)
    runner = add_joker(state, "j_runner")
    runner.debuff = True
    add_joker(state, "j_crazy")

    assert HeuristicAgent()._get_main_hand_type(state) == "Pair"


def test_blue_seal_is_held_while_playing_most_played_hand() -> None:
    data = load_game_data()
    state = create_run_state("blue_seal_planet", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_gros_michel")
    state.hands["Pair"]["played"] = 8
    state.hand_cards = [
        PlayingCard(front_key="S_A", suit="Spades", rank="A"),
        PlayingCard(front_key="H_A", suit="Hearts", rank="A"),
        PlayingCard(front_key="D_K", suit="Diamonds", rank="K", seal="Blue"),
        PlayingCard(front_key="C_T", suit="Clubs", rank="T"),
        PlayingCard(front_key="S_8", suit="Spades", rank="8"),
        PlayingCard(front_key="H_6", suit="Hearts", rank="6"),
        PlayingCard(front_key="D_4", suit="Diamonds", rank="4"),
        PlayingCard(front_key="C_2", suit="Clubs", rank="2"),
    ]
    mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    agent = HeuristicAgent()

    action = agent.select_action(state, SubPhase.CHOOSE_ACTION, mask)

    assert ActionRange.PLAY_SUBSET_START <= action <= ActionRange.PLAY_SUBSET_END
    played = subset_indices(action - ActionRange.PLAY_SUBSET_START)
    assert 2 not in played
    assert agent._quick_hand_quality(state, [state.hand_cards[i] for i in played]) == "Pair"


def test_rerolls_violet_vessel_when_cash_is_available() -> None:
    data = load_game_data()
    state = create_run_state("vessel_reroll", data=data)
    state.blind_on_deck = "Boss"
    state.dollars = 10
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"

    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)

    assert HeuristicAgent().select_action(state, SubPhase.BLIND_SELECT, mask) == ActionRange.BLIND_REROLL


def test_shop_target_ignores_defeated_boss_multiplier() -> None:
    data = load_game_data()
    state = create_run_state("stale_vessel_shop_target", data=data)
    state.round_resets.ante = 2
    state.blind_on_deck = "Small"
    state.round_resets.blind = data.blinds["bl_final_vessel"]
    state.round_resets.blind_choices["Boss"] = "bl_flint"
    base = get_blind_amount(state.round_resets.ante, min(state.stake, 3))

    assert HeuristicAgent()._near_term_shop_target(state) == base * data.blinds["bl_flint"]["mult"]


def test_rerolls_flint_with_midgame_cash() -> None:
    data = load_game_data()
    state = create_run_state("flint_reroll", data=data)
    state.blind_on_deck = "Boss"
    state.dollars = 15
    state.round_resets.blind_choices["Boss"] = "bl_flint"

    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)

    assert HeuristicAgent().select_action(state, SubPhase.BLIND_SELECT, mask) == ActionRange.BLIND_REROLL


def test_preflint_shop_preserves_boss_reroll_cash() -> None:
    data = load_game_data()
    state = create_run_state("flint_shop_reserve", data=data)
    state.round_resets.ante = 3
    state.blind_on_deck = "Boss"
    state.dollars = 12
    state.round_resets.blind_choices["Boss"] = "bl_flint"
    state.shop.boosters = [_shop_card("p_standard_normal_1", "Booster", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_LEAVE


def test_preneedle_shop_buys_missing_xmult_over_reroll_reserve() -> None:
    data = load_game_data()
    state = create_run_state("needle_xmult_override", data=data)
    state.round_resets.ante = 2
    state.blind_on_deck = "Boss"
    state.dollars = 13
    state.round_resets.blind_choices["Boss"] = "bl_needle"
    add_joker(state, "j_sly")
    add_joker(state, "j_joker")
    state.shop.cards = [_shop_card("j_campfire", "Joker", cost=9)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START


def test_auto_order_skips_build_without_xmult_or_copy(monkeypatch) -> None:
    data = load_game_data()
    state = create_run_state("reorder_search_gate", data=data)
    add_joker(state, "j_joker")
    add_joker(state, "j_sly")

    def unexpected_score(*_args, **_kwargs):
        raise AssertionError("ordinary builds should skip ordered hand scoring")

    monkeypatch.setattr(joker_layout, "_score_order", unexpected_score)

    assert joker_layout.best_joker_order(state, (0,)) is None


def test_auto_order_places_xmult_last_for_selected_hand() -> None:
    data = load_game_data()
    state = create_run_state("reorder_search_xmult", data=data)
    add_joker(state, "j_cavendish")
    add_joker(state, "j_joker")
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    assert joker_layout.apply_best_joker_order(state, (0,)) is True
    assert state.joker_keys == ["j_joker", "j_cavendish"]
    # A second application is a no-op: the roster is already optimal.
    assert joker_layout.apply_best_joker_order(state, (0,)) is False


def test_auto_order_candidate_search_has_hard_budget() -> None:
    data = load_game_data()
    state = create_run_state("reorder_candidate_budget", data=data)
    for center_key in (
        "j_blueprint",
        "j_blueprint",
        "j_brainstorm",
        "j_dusk",
        "j_hack",
        "j_idol",
        "j_joker",
        "j_cavendish",
    ):
        add_joker(state, center_key)

    candidates = joker_layout.order_candidates(state, len(state.jokers))

    assert len(candidates) <= 16
    assert len(candidates) == len(set(candidates))
    assert all(sorted(order) == list(range(len(state.jokers))) for order in candidates)


def test_auto_order_places_copy_before_target_without_xmult() -> None:
    data = load_game_data()
    state = create_run_state("reorder_search_copy", data=data)
    add_joker(state, "j_joker")
    add_joker(state, "j_blueprint")
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]

    assert joker_layout.apply_best_joker_order(state, (0,)) is True
    assert state.joker_keys == ["j_blueprint", "j_joker"]


def test_auto_order_balances_retrigger_and_idol_effects() -> None:
    data = load_game_data()
    state = create_run_state("retrigger_idol_balance", data=data)
    add_joker(state, "j_blueprint")
    add_joker(state, "j_dusk")
    add_joker(state, "j_hack")
    add_joker(state, "j_idol")
    state.current_round.hands_left = 1
    state.current_round.idol_card = {"rank": "2", "suit": "Hearts", "id": 2}
    state.hand_cards = [
        PlayingCard(front_key="H_2", suit="Hearts", rank="2", seal="Red"),
    ]

    joker_layout.apply_best_joker_order(state, (0,))

    blueprint_index = state.joker_keys.index("j_blueprint")
    assert state.joker_keys[blueprint_index + 1] == "j_idol"


def test_auto_order_copies_retrigger_when_it_repeats_more_effects() -> None:
    data = load_game_data()
    state = create_run_state("retrigger_multiple_effects", data=data)
    for center_key in (
        "j_blueprint",
        "j_sock_and_buskin",
        "j_smiley",
        "j_photograph",
        "j_triboulet",
    ):
        add_joker(state, center_key)
    state.hand_cards = [
        PlayingCard(front_key="H_K", suit="Hearts", rank="K"),
    ]

    joker_layout.apply_best_joker_order(state, (0,))

    blueprint_index = state.joker_keys.index("j_blueprint")
    assert state.joker_keys[blueprint_index + 1] == "j_sock_and_buskin"


def test_midgame_full_joker_slots_sell_weak_joker_for_xmult() -> None:
    data = load_game_data()
    state = create_run_state("midgame_xmult_replacement", data=data)
    state.round_resets.ante = 4
    state.starting_params.joker_slots = 1
    state.dollars = 3
    add_joker(state, "j_sly")
    state.shop.cards = [_shop_card("j_cavendish", "Joker", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action == ActionRange.SHOP_SELL_JOKER_START


def test_midgame_prefers_pair_planet_over_chip_joker() -> None:
    data = load_game_data()
    state = create_run_state("midgame_planet_scaling", data=data)
    state.round_resets.ante = 4
    state.dollars = 6
    state.hands["Pair"]["played"] = 1
    state.shop.cards = [
        _shop_card("j_sly", "Joker", cost=3),
        _shop_card("c_mercury", "Planet", cost=3),
    ]

    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action == ActionRange.SHOP_BUY_START + 1


def test_midgame_does_not_treat_incidental_two_pair_as_planet_plan() -> None:
    data = load_game_data()
    state = create_run_state("midgame_incidental_two_pair", data=data)
    state.round_resets.ante = 4
    state.dollars = 6
    state.hands["Two Pair"]["played"] = 5
    state.shop.cards = [
        _shop_card("j_sly", "Joker", cost=3),
        _shop_card("c_uranus", "Planet", cost=3),
    ]

    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action == ActionRange.SHOP_BUY_START


def test_generic_two_pair_joker_does_not_redirect_planet_plan() -> None:
    data = load_game_data()
    state = create_run_state("generic_two_pair_plan", data=data)
    state.round_resets.ante = 4
    state.hands["Two Pair"]["played"] = 20
    add_joker(state, "j_mad")

    assert HeuristicAgent()._get_main_hand_type(state) == "Pair"


def test_unscaled_runner_does_not_redirect_planet_plan() -> None:
    data = load_game_data()
    state = create_run_state("unscaled_runner_plan", data=data)
    add_joker(state, "j_runner")

    assert HeuristicAgent()._get_main_hand_type(state) == "Pair"


def test_scaled_runner_can_commit_to_straight_plan() -> None:
    data = load_game_data()
    state = create_run_state("scaled_runner_plan", data=data)
    runner = add_joker(state, "j_runner")
    runner.extra["chips"] = 45

    assert HeuristicAgent()._get_main_hand_type(state) == "Straight"


def test_wily_alone_does_not_redirect_planet_plan() -> None:
    data = load_game_data()
    state = create_run_state("wily_pair_plan", data=data)
    add_joker(state, "j_wily")

    assert HeuristicAgent()._get_main_hand_type(state) == "Pair"


def test_incidental_level_two_full_house_does_not_redirect_planet_plan() -> None:
    data = load_game_data()
    state = create_run_state("incidental_full_house_plan", data=data)
    state.hands["Full House"]["level"] = 2
    state.hands["Full House"]["played"] = 2

    assert HeuristicAgent()._get_main_hand_type(state) == "Pair"


def test_inactive_xmult_centers_do_not_end_xmult_hunt() -> None:
    data = load_game_data()
    constellation_state = create_run_state("inactive_constellation", data=data)
    add_joker(constellation_state, "j_constellation")

    baseball_state = create_run_state("inactive_baseball", data=data)
    add_joker(baseball_state, "j_baseball")
    add_joker(baseball_state, "j_joker")

    stencil_state = create_run_state("inactive_stencil", data=data)
    stencil_state.starting_params.joker_slots = 1
    add_joker(stencil_state, "j_stencil")

    loyalty_state = create_run_state("inactive_loyalty", data=data)
    loyalty = add_joker(loyalty_state, "j_loyalty_card")

    agent = HeuristicAgent()
    assert not agent._has_xmult_joker(constellation_state)
    assert not agent._has_xmult_joker(baseball_state)
    assert not agent._has_xmult_joker(stencil_state)
    assert not agent._has_xmult_joker(loyalty_state)

    loyalty_state.hands_played = loyalty.hands_played_at_create + 1
    assert not agent._has_xmult_joker(loyalty_state)

    loyalty_state.hands_played = loyalty.hands_played_at_create + 2
    assert agent._has_xmult_joker(loyalty_state)

    loyalty_state.hands_played = loyalty.hands_played_at_create + 4
    assert agent._has_xmult_joker(loyalty_state)


def test_midgame_xmult_hunt_rerolls_while_preserving_purchase_cash() -> None:
    data = load_game_data()
    state = create_run_state("midgame_xmult_hunt", data=data)
    state.round_resets.ante = 4
    state.dollars = state.current_round.reroll_cost + 8
    add_joker(state, "j_joker")
    add_joker(state, "j_sly")

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_REROLL


def test_live_constellation_counts_as_xmult() -> None:
    data = load_game_data()
    state = create_run_state("live_constellation", data=data)
    constellation = add_joker(state, "j_constellation")
    constellation.x_mult = 1.2

    assert HeuristicAgent()._has_xmult_joker(state)


def test_campfire_buys_disposable_planet_before_big_blind() -> None:
    data = load_game_data()
    state = create_run_state("campfire_big_blind_feed", data=data)
    state.round_resets.ante = 3
    state.blind_on_deck = "Big"
    state.dollars = 7
    add_joker(state, "j_campfire")
    state.shop.cards = [_shop_card("c_venus", "Planet", cost=3)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START


def test_suit_mult_engine_buys_droll_before_pair_planet() -> None:
    data = load_game_data()
    state = create_run_state("droll_suit_engine", data=data)
    state.round_resets.ante = 2
    state.dollars = 6
    add_joker(state, "j_lusty_joker")
    add_joker(state, "j_hologram")
    state.shop.cards = [
        _shop_card("j_droll", "Joker", cost=5),
        _shop_card("c_mercury", "Planet", cost=3),
    ]
    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(
        state,
        SubPhase.SHOP,
        mask,
        round_score=500,
    )

    assert action == ActionRange.SHOP_BUY_START


def test_baron_preserves_held_kings_while_drawing() -> None:
    data = load_game_data()
    state = create_run_state("baron_held_king_draw", data=data)
    add_joker(state, "j_baron")
    state.hand_cards = [
        PlayingCard(front_key="H_K", suit="Hearts", rank="K"),
        PlayingCard(front_key="S_Q", suit="Spades", rank="Q"),
        PlayingCard(front_key="D_9", suit="Diamonds", rank="9"),
        PlayingCard(front_key="C_8", suit="Clubs", rank="8"),
        PlayingCard(front_key="H_7", suit="Hearts", rank="7"),
        PlayingCard(front_key="S_4", suit="Spades", rank="4"),
    ]

    discarded = HeuristicAgent()._find_worst_cards(state, state.hand_cards, 5)

    assert 0 not in discarded


def test_spare_trousers_can_commit_to_two_pair_plan() -> None:
    data = load_game_data()
    state = create_run_state("trousers_two_pair_plan", data=data)
    add_joker(state, "j_trousers")

    assert HeuristicAgent()._get_main_hand_type(state) == "Two Pair"


def test_debuffed_spare_trousers_does_not_commit_to_two_pair_plan() -> None:
    data = load_game_data()
    state = create_run_state("debuffed_trousers_two_pair_plan", data=data)
    trousers = add_joker(state, "j_trousers")
    trousers.debuff = True

    assert HeuristicAgent()._get_main_hand_type(state) == "Pair"


def test_midgame_sells_consumable_for_pair_planet() -> None:
    data = load_game_data()
    state = create_run_state("midgame_planet_room", data=data)
    state.round_resets.ante = 4
    state.dollars = 6
    state.starting_params.consumable_slots = 1
    state.hands["Pair"]["played"] = 1
    add_consumable(state, "c_magician")
    state.shop.cards = [_shop_card("c_mercury", "Planet", cost=3)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action == ActionRange.SHOP_SELL_CONSUMABLE_START


def test_late_shop_does_not_sell_scaled_green_joker_for_chip_joker() -> None:
    data = load_game_data()
    state = create_run_state("late_scaled_green_protection", data=data)
    state.round_resets.ante = 6
    state.starting_params.joker_slots = 1
    state.dollars = 20
    green = add_joker(state, "j_green_joker")
    green.mult = 30
    state.shop.cards = [_shop_card("j_sly", "Joker", cost=3)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action != ActionRange.SHOP_SELL_JOKER_START


def test_late_shop_does_not_sell_other_scaling_jokers_for_chip_joker() -> None:
    data = load_game_data()
    scaling_cases = [
        ("j_runner", lambda joker: joker.extra.update({"chips": 120})),
        ("j_square", lambda joker: joker.extra.update({"chips": 80})),
        ("j_castle", lambda joker: joker.extra.update({"chips": 100})),
        ("j_trousers", lambda joker: setattr(joker, "mult", 24)),
        ("j_red_card", lambda joker: setattr(joker, "mult", 24)),
    ]

    for center_key, scale_joker in scaling_cases:
        state = create_run_state(f"late_scaled_{center_key}", data=data)
        state.round_resets.ante = 6
        state.starting_params.joker_slots = 1
        state.dollars = 20
        joker = add_joker(state, center_key)
        scale_joker(joker)
        state.shop.cards = [_shop_card("j_sly", "Joker", cost=3)]

        mask = compute_action_mask(state, SubPhase.SHOP)
        action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

        assert action != ActionRange.SHOP_SELL_JOKER_START


def test_late_shop_can_replace_an_unscaled_runner() -> None:
    data = load_game_data()
    state = create_run_state("late_unscaled_runner", data=data)
    state.round_resets.ante = 6
    state.starting_params.joker_slots = 1
    state.dollars = 20
    add_joker(state, "j_runner")
    state.shop.cards = [_shop_card("j_half", "Joker", cost=5)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action == ActionRange.SHOP_SELL_JOKER_START


def test_late_shop_preserves_immediate_conditional_xmult() -> None:
    data = load_game_data()
    state = create_run_state("late_baron_protection", data=data)
    state.round_resets.ante = 5
    state.starting_params.joker_slots = 1
    state.dollars = 40
    add_joker(state, "j_baron")
    state.shop.cards = [_shop_card("j_odd_todd", "Joker", cost=4)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) != ActionRange.SHOP_SELL_JOKER_START


def test_late_shop_buys_immediate_ancient_xmult() -> None:
    data = load_game_data()
    state = create_run_state("late_ancient_purchase", data=data)
    state.round_resets.ante = 5
    state.starting_params.joker_slots = 1
    state.dollars = 40
    add_joker(state, "j_odd_todd")
    state.shop.cards = [_shop_card("j_ancient", "Joker", cost=8)]

    mask = compute_action_mask(state, SubPhase.SHOP)

    assert HeuristicAgent().select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_SELL_JOKER_START


def test_ancient_joker_discards_toward_current_round_suit() -> None:
    data = load_game_data()
    state = create_run_state("ancient_suit_draw", data=data)
    add_joker(state, "j_ancient")
    state.current_round.ancient_card = {"suit": "Hearts"}
    state.hand_cards = [
        PlayingCard(front_key="H_A", suit="Hearts", rank="A"),
        PlayingCard(front_key="H_K", suit="Hearts", rank="K"),
        PlayingCard(front_key="S_Q", suit="Spades", rank="Q"),
        PlayingCard(front_key="S_J", suit="Spades", rank="J"),
        PlayingCard(front_key="S_T", suit="Spades", rank="T"),
        PlayingCard(front_key="S_9", suit="Spades", rank="9"),
    ]

    discarded = HeuristicAgent()._find_worst_cards(state, state.hand_cards, 4)

    assert discarded == {2, 3, 4, 5}


def test_hand_score_cache_tracks_live_joker_mult() -> None:
    data = load_game_data()
    state = create_run_state("live_joker_score_cache", data=data)
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]
    joker = add_joker(state, "j_joker")
    agent = HeuristicAgent()

    base_score = agent._estimate_hand_score(state, (0,))
    joker.mult += 20
    grown_score = agent._estimate_hand_score(state, (0,))

    assert grown_score > base_score


def test_card_sharp_estimate_requires_repeated_hand_this_round() -> None:
    data = load_game_data()
    state = create_run_state("card_sharp_condition", data=data)
    state.hand_cards = [
        PlayingCard(front_key="S_A", suit="Spades", rank="A"),
        PlayingCard(front_key="H_A", suit="Hearts", rank="A"),
    ]
    add_joker(state, "j_card_sharp")
    agent = HeuristicAgent()

    first_pair_score = agent._estimate_hand_score(state, (0, 1))
    state.hands["Pair"]["played_this_round"] = 2
    repeated_pair_score = agent._estimate_hand_score(state, (0, 1))

    assert repeated_pair_score > first_pair_score * 2
