import pickle
from copy import deepcopy

import pytest

from pylatro import add_joker, create_run_state
from pylatro.pool import create_card_spec
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_shop_search import ShopSearch
from pylatro_agent.masks import compute_action_mask


def test_shop_counterfactual_preserves_run_and_ignores_draw_order():
    state = create_run_state("public_deck", deck_key="b_blue")
    add_joker(state, "j_joker")
    before = pickle.dumps(state)
    search = ShopSearch()
    score = search.output(state, HeuristicAgent())
    assert pickle.dumps(state) == before
    shuffled = deepcopy(state, {id(state.data): state.data})
    shuffled.deck_cards.reverse()
    shuffled.draw_pile.reverse()
    assert search.output(shuffled, HeuristicAgent()) == score


def test_public_hand_samples_only_change_when_an_added_card_is_selected():
    state = create_run_state("stable_samples", deck_key="b_blue")
    search = ShopSearch()
    original, added = state.deck_cards[:-1], state.deck_cards[-1]
    for sample in range(12):
        before = search.sample_hand(original, 8, sample)
        after = search.sample_hand(state.deck_cards, 8, sample)
        assert after == search.sample_hand(list(reversed(state.deck_cards)), 8, sample)
        if added in after:
            assert [c for c in after if c is not added] == before[:7]
        else:
            assert after == before


def test_shop_forecast_does_not_depend_on_process_global_card_ids():
    first = create_run_state("repeatable_shop", deck_key="b_blue")
    create_run_state("unrelated_run")
    second = create_run_state("repeatable_shop", deck_key="b_blue")
    assert first.deck_cards[0].reward_uid != second.deck_cards[0].reward_uid
    for state in (first, second):
        add_joker(state, "j_half")
        add_joker(state, "j_blue_joker")
        add_joker(state, "j_constellation")
    agent = HeuristicAgent()
    assert ShopSearch().output(first, agent) == ShopSearch().output(second, agent)


def test_search_buys_immediate_multiplier_without_changing_state():
    state = create_run_state("shop_multiplier", deck_key="b_blue")
    state.dollars = 20
    add_joker(state, "j_joker")
    state.shop.cards = [create_card_spec(state, "Joker", forced_key="j_cavendish")]
    mask = compute_action_mask(state, SubPhase.SHOP)
    before = pickle.dumps(state)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.SHOP, mask)
    assert action == ActionRange.SHOP_BUY_START
    assert pickle.dumps(state) == before


def test_shop_sells_a_harmful_stencil_neighbor_without_a_replacement_offer():
    from pylatro.runtime import sell_joker

    state = create_run_state("stencil_pruning", deck_key="b_blue")
    state.round_resets.ante = 4
    state.dollars = 10
    for key in ("j_banner", "j_half", "j_stencil", "j_faceless", "j_credit_card"):
        add_joker(state, key)
    agent = HeuristicAgent(shop_policy="search")
    search = ShopSearch()
    score, _ = search.output(state, agent)
    before = pickle.dumps(state)
    action = agent.select_action(state, SubPhase.SHOP, compute_action_mask(state, SubPhase.SHOP))
    assert action in (ActionRange.SHOP_SELL_JOKER_START + 3, ActionRange.SHOP_SELL_JOKER_START + 4)
    assert pickle.dumps(state) == before
    sell_joker(state, action - ActionRange.SHOP_SELL_JOKER_START)
    improved, _ = search.output(state, agent)
    assert improved > score * 1.5


def test_stencil_sells_disabled_banner_when_needed_to_survive_water():
    from pylatro.runtime import sell_joker

    state = create_run_state("stencil_water_survival", deck_key="b_blue")
    state.round_resets.ante = 7
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_water"
    state.dollars = 8
    for key in ("j_banner", "j_even_steven", "j_stencil", "j_ramen"):
        add_joker(state, key)
    state.hands["Pair"].update(level=12, chips=175, mult=13)
    agent = HeuristicAgent(shop_policy="search")
    search = ShopSearch()
    score, _ = search.purchase_output(state, agent)
    assert score * search.round_hands(state, boss=True) < search.target(state, agent)
    action = agent.select_action(state, SubPhase.SHOP, compute_action_mask(state, SubPhase.SHOP))
    assert action == ActionRange.SHOP_SELL_JOKER_START
    sell_joker(state, 0)
    improved, _ = search.purchase_output(state, agent)
    assert improved * search.round_hands(state, boss=True) > search.target(state, agent) * 1.15


@pytest.mark.parametrize("shop", [False, True])
def test_mature_stencil_build_preserves_its_empty_slot_instead_of_using_judgement(shop):
    from pylatro import add_consumable, select_blind, start_blind
    from pylatro_agent.action import ActionType, decode_action

    state = create_run_state("stencil_judgement", deck_key="b_blue")
    for key in ("j_banner", "j_half", "j_stencil", "j_walkie_talkie"):
        add_joker(state, key)
    add_consumable(state, "c_judgement")
    select_blind(state, "Small")
    start_blind(state, "Small")
    phase = SubPhase.SHOP if shop else SubPhase.CHOOSE_ACTION
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(state, phase, compute_action_mask(state, phase))
    if shop:
        assert action == ActionRange.SHOP_SELL_CONSUMABLE_START
    else:
        assert decode_action(action).action_type != ActionType.USE_CONSUMABLE_NO_TARGET
    assert agent._score_tarot_value(state, state.data.centers["c_judgement"]) < 10


def test_search_buys_income_after_establishing_scoring():
    state = create_run_state("safe_income", deck_key="b_blue")
    state.dollars = 20
    add_joker(state, "j_blue_joker")
    add_joker(state, "j_half")
    state.shop.cards = [create_card_spec(state, "Joker", forced_key="j_golden")]
    mask = compute_action_mask(state, SubPhase.SHOP)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.SHOP, mask)
    assert action == ActionRange.SHOP_BUY_START


def test_search_does_not_buy_income_when_scoring_is_insufficient():
    state = create_run_state("unsafe_income", deck_key="b_blue")
    state.round_resets.ante = 2
    state.dollars = 20
    state.shop.cards = [create_card_spec(state, "Joker", forced_key="j_golden")]
    mask = compute_action_mask(state, SubPhase.SHOP)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.SHOP, mask)
    assert action != ActionRange.SHOP_BUY_START


@pytest.mark.parametrize(
    "joker,booster", [("j_hologram", "p_standard_normal_1"), ("j_constellation", "p_celestial_normal_1")],
)
def test_shop_funds_the_packs_that_develop_an_owned_scaler(joker, booster):
    state = create_run_state("scaler_pack", deck_key="b_blue")
    state.round_resets.ante = 3
    state.dollars = 12
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    add_joker(state, joker)
    state.shop.boosters = [create_card_spec(state, "Booster", forced_key=booster)]
    agent = HeuristicAgent(shop_policy="search")
    mask = compute_action_mask(state, SubPhase.SHOP)
    assert agent.select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START


def test_search_invests_in_early_constellation_with_scoring_already_secured():
    state = create_run_state("future_shop", deck_key="b_blue")
    state.round_resets.ante = 2
    state.dollars = 26
    add_joker(state, "j_blue_joker")
    add_joker(state, "j_half")
    state.shop.cards = [create_card_spec(state, "Joker", forced_key="j_constellation")]
    mask = compute_action_mask(state, SubPhase.SHOP)
    before = pickle.dumps(state)
    agent = HeuristicAgent(shop_policy="search")
    assert agent.select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START
    assert pickle.dumps(state) == before


def test_growth_valuation_keeps_survival_forecast_current_and_diminishes_with_progress():
    state = create_run_state("growth_value", deck_key="b_blue")
    add_joker(state, "j_blue_joker")
    add_joker(state, "j_half")
    constellation = add_joker(state, "j_constellation")
    agent = HeuristicAgent()
    search = ShopSearch()
    score, value = search.output(state, agent)
    current_score, current_value = search.output(state, agent, _project=False)
    assert score == current_score
    assert value > current_value
    first_bonus = value - current_value
    constellation.x_mult = 4
    _, grown = search.output(state, agent)
    _, grown_current = search.output(state, agent, _project=False)
    assert 0 < grown - grown_current < first_bonus
    state.round_resets.ante = 8
    assert search.output(state, agent) == search.output(state, agent, _project=False)


def test_constellation_claims_off_plan_planet_from_open_pack():
    from pylatro.models import PackState

    state = create_run_state("constellation_planet", deck_key="b_blue")
    add_joker(state, "j_constellation")
    state.pack = PackState(
        "p_celestial_normal_1", "CELESTIAL_PACK", 1,
        cards=[create_card_spec(state, "Planet", forced_key="c_saturn")],
    )
    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    agent = HeuristicAgent(shop_policy="search")
    assert agent.select_action(state, SubPhase.BOOSTER_PACK, mask) == ActionRange.PACK_CLAIM_START


def test_search_buys_luchador_to_counter_violet_vessel():
    state = create_run_state("luchador_shop", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    state.dollars = 10
    for key in ("j_half", "j_blue_joker", "j_banner", "j_abstract"):
        add_joker(state, key)
    state.shop.cards = [create_card_spec(state, "Joker", forced_key="j_luchador")]
    before = pickle.dumps(state)
    agent = HeuristicAgent(shop_policy="search")
    mask = compute_action_mask(state, SubPhase.SHOP)
    assert agent.select_action(state, SubPhase.SHOP, mask) == ActionRange.SHOP_BUY_START
    assert pickle.dumps(state) == before


def test_search_uses_an_owned_boss_reroll_when_eye_forecast_is_insufficient():
    state = create_run_state("eye_reroll", deck_key="b_blue")
    state.round_resets.ante = 7
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_eye"
    state.used_vouchers["v_directors_cut"] = True
    state.dollars = 25
    for key in ("j_half", "j_blue_joker", "j_card_sharp"):
        add_joker(state, key)
    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.BLIND_SELECT, mask)
    assert action == ActionRange.BLIND_REROLL


@pytest.mark.parametrize("boss,expected", [("bl_serpent", False), ("bl_eye", True), ("bl_needle", True)])
def test_boss_reroll_forecast_requires_a_meaningful_boss_penalty(boss, expected):
    state = create_run_state("reroll_penalty", deck_key="b_blue")
    state.round_resets.ante = 7
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = boss
    state.used_vouchers["v_directors_cut"] = True
    state.dollars = 25
    for key in ("j_half", "j_blue_joker", "j_card_sharp"):
        add_joker(state, key)
    before = pickle.dumps(state)
    assert ShopSearch().boss_reroll_improves_risk(state, HeuristicAgent()) == expected
    assert pickle.dumps(state) == before
    action = HeuristicAgent(shop_policy="search").select_action(
        state, SubPhase.BLIND_SELECT, compute_action_mask(state, SubPhase.BLIND_SELECT),
    )
    assert action == (ActionRange.BLIND_REROLL if expected else ActionRange.BLIND_PLAY)


def test_comfortable_vessel_forecast_does_not_spend_a_reroll():
    state = create_run_state("comfortable_vessel", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    add_joker(state, "j_half").extra["mult"] = 10000
    add_joker(state, "j_blue_joker")
    assert not ShopSearch().boss_reroll_improves_risk(state, HeuristicAgent())


def test_boss_counter_avoids_spending_on_an_unnecessary_vessel_reroll():
    state = create_run_state("chicot_reroll", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    state.used_vouchers["v_directors_cut"] = True
    state.dollars = 25
    add_joker(state, "j_chicot")
    mask = compute_action_mask(state, SubPhase.BLIND_SELECT)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.BLIND_SELECT, mask)
    assert action == ActionRange.BLIND_PLAY


def test_final_ante_can_spend_last_reroll_budget_to_find_an_upgrade():
    state = create_run_state("last_shop_reroll", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    state.dollars = 8
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    mask = compute_action_mask(state, SubPhase.SHOP)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.SHOP, mask)
    assert action == ActionRange.SHOP_REROLL


@pytest.mark.parametrize("blind", ["Small", "Big", "Boss"])
def test_final_ante_keeps_purchase_cash_when_the_build_is_already_safe(blind):
    state = create_run_state("safe_last_reroll", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = blind
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    state.dollars = 8
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    scaler = add_joker(state, "j_constellation")
    scaler.x_mult = 100
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(state, SubPhase.SHOP, compute_action_mask(state, SubPhase.SHOP))
    assert action == ActionRange.SHOP_LEAVE


def test_shop_consumables_are_usable_without_exposing_old_hand_targets():
    from pylatro import add_consumable
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    runner = FastRunner(0, create_run_state("shop_use").data, deck_key="b_blue", raise_errors=True)
    state = runner.state
    runner.step(ActionRange.BLIND_PLAY)
    # Simulate the shop with leftover cards from the completed blind.
    runner._sub_phase = SubPhase.SHOP
    runner._ctrl.phase = GamePhase.SHOP
    state.dollars = 15
    add_consumable(state, "c_hermit")
    add_consumable(state, "c_death")
    mask = runner.compute_mask()
    assert (mask == compute_action_mask(state, SubPhase.SHOP)).all()
    for index, valid in enumerate(mask):
        if valid:
            assert decode_action(index).action_type != ActionType.USE_CONSUMABLE_HAND_SUBSET
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(state, runner.sub_phase, mask)
    assert decode_action(action).action_type == ActionType.USE_CONSUMABLE_NO_TARGET
    runner.step(action)
    assert state.dollars == 30
    assert runner.sub_phase == SubPhase.SHOP
    assert [c.center_key for c in state.consumables] == ["c_death"]


def test_shop_planet_is_applied_before_the_next_purchase_forecast():
    from pylatro import add_consumable
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    runner = FastRunner(0, create_run_state("shop_planet").data, deck_key="b_blue", raise_errors=True)
    state = runner.state
    runner._sub_phase = SubPhase.SHOP
    runner._ctrl.phase = GamePhase.SHOP
    constellation = add_joker(state, "j_constellation")
    add_consumable(state, "c_mercury")
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(state, runner.sub_phase, runner.compute_mask())
    assert decode_action(action).action_type == ActionType.USE_CONSUMABLE_NO_TARGET
    runner.step(action)
    assert state.hands["Pair"]["level"] == 2
    assert constellation.x_mult == 1.1
    assert not state.consumables
    assert runner.sub_phase == SubPhase.SHOP


def test_shop_buys_and_uses_fool_to_repeat_hermit_before_spending():
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    runner = FastRunner(0, create_run_state("fool_shop").data, deck_key="b_blue", raise_errors=True)
    state = runner.state
    runner._sub_phase = SubPhase.SHOP
    runner._ctrl.phase = GamePhase.SHOP
    state.dollars = 20
    state.last_tarot_planet = "c_hermit"
    state.shop.cards = [create_card_spec(state, "Tarot", forced_key="c_fool")]
    cost = state.shop.cards[0].cost
    agent = HeuristicAgent(shop_policy="search")
    assert agent.select_action(state, runner.sub_phase, runner.compute_mask()) == ActionRange.SHOP_BUY_START
    for _ in range(3):
        runner.step(agent.select_action(state, runner.sub_phase, runner.compute_mask()))
    assert not state.consumables
    assert state.dollars == (20 - cost) * 2
    assert runner.sub_phase == SubPhase.SHOP


def test_shop_forecast_values_extra_hands_and_last_hand_multipliers():
    state = create_run_state("round_output", deck_key="b_blue")
    add_joker(state, "j_half")
    agent = HeuristicAgent()
    search = ShopSearch()
    _, initial = search.output(state, agent)
    add_joker(state, "j_acrobat")
    _, acrobat = search.output(state, agent)
    assert acrobat > initial
    add_joker(state, "j_burglar")
    _, burglar = search.output(state, agent)
    assert burglar > acrobat
    state.round_resets.blind_choices["Boss"] = "bl_needle"
    assert search.round_hands(state, boss=True) == 4


def test_needle_forecast_does_not_activate_card_sharp_before_any_hand_is_played():
    state = create_run_state("needle_card_sharp", deck_key="b_blue")
    state.round_resets.blind_choices["Boss"] = "bl_needle"
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    agent = HeuristicAgent()
    baseline, _ = ShopSearch().output(state, agent, boss=True)
    add_joker(state, "j_card_sharp")
    score, _ = ShopSearch().output(state, agent, boss=True)
    assert score == pytest.approx(baseline)


@pytest.mark.parametrize("joker", ["j_blue_joker", "j_ice_cream"])
def test_later_forecast_hands_account_for_depleted_chips(joker):
    state = create_run_state("depleting_forecast", deck_key="b_blue")
    add_joker(state, "j_half")
    add_joker(state, joker)
    agent = HeuristicAgent()
    search = ShopSearch()
    # Identical public hands isolate the change in the scoring resource.
    search.sample_hand = lambda cards, size, sample: cards[:size]
    state.round_resets.hands = 1
    opening, _ = search.output(state, agent, _project=False)
    state.round_resets.hands = 5
    before = pickle.dumps(state)
    average, _ = search.output(state, agent, _project=False)
    assert average < opening
    assert pickle.dumps(state) == before


def test_madness_forecast_accounts_for_destroying_a_scoring_neighbor_before_normal_blinds():
    state = create_run_state("madness_cost", deck_key="b_blue")
    madness = add_joker(state, "j_madness")
    madness.x_mult = 3
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    search = ShopSearch()
    agent = HeuristicAgent()
    before = pickle.dumps(state)
    unchanged, _ = search.output(state, agent, _activated=True)
    realistic, _ = search.output(state, agent)
    assert realistic < unchanged * 0.8
    assert pickle.dumps(state) == before
    assert search.output(state, agent, boss=True) == search.output(state, agent, boss=True, _activated=True)


def test_lone_madness_forecast_includes_its_free_growth_without_mutating_the_joker():
    state = create_run_state("madness_alone", deck_key="b_blue")
    madness = add_joker(state, "j_madness")
    search = ShopSearch()
    agent = HeuristicAgent()
    unchanged, _ = search.output(state, agent, _activated=True)
    grown, _ = search.output(state, agent)
    assert grown > unchanged * 1.4
    assert madness.x_mult == 1


def test_leaf_forecast_can_sell_a_non_scoring_joker_instead_of_assuming_permanent_debuffs():
    state = create_run_state("leaf_sale_forecast", deck_key="b_blue")
    state.round_resets.ante = 8
    state.round_resets.blind_choices["Boss"] = "bl_final_leaf"
    for key in ("j_half", "j_blue_joker", "j_faceless"):
        add_joker(state, key)
    agent = HeuristicAgent()
    search = ShopSearch()
    before = pickle.dumps(state)
    ordinary, _ = search.output(state, agent)
    leaf, _ = search.output(state, agent, boss=True)
    assert leaf == pytest.approx(ordinary)
    assert pickle.dumps(state) == before
    for joker in state.jokers:
        joker.eternal = True
    # Eternal flags affect which sale counterfactuals are legal.
    blocked, _ = ShopSearch().output(state, agent, boss=True)
    assert blocked < ordinary


def test_buffoon_search_compares_complete_scoring_portfolios():
    from pylatro.models import PackState

    state = create_run_state("buffoon_search", deck_key="b_blue")
    add_joker(state, "j_blue_joker")
    add_joker(state, "j_half")
    state.pack = PackState(
        "p_buffoon_normal_1", "BUFFOON_PACK", 1,
        cards=[create_card_spec(state, "Joker", forced_key=key) for key in ("j_abstract", "j_cavendish")],
    )
    before = pickle.dumps(state)
    mask = compute_action_mask(state, SubPhase.BOOSTER_PACK)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.BOOSTER_PACK, mask)
    assert action == ActionRange.PACK_CLAIM_START + 1
    assert pickle.dumps(state) == before


def test_full_roster_can_replace_a_joker_from_a_buffoon_pack():
    from pylatro.models import PackState
    from pylatro_agent.training.fast_runner import FastRunner

    data = create_run_state("buffoon_data").data
    runner = FastRunner(0, data, deck_key="b_blue", raise_errors=True)
    state = runner.state
    for key in ("j_blue_joker", "j_half", "j_abstract", "j_banner", "j_joker"):
        add_joker(state, key)
    state.pack = PackState(
        "p_buffoon_normal_1", "BUFFOON_PACK", 1,
        cards=[create_card_spec(state, "Joker", forced_key="j_cavendish")],
    )
    runner._sub_phase = SubPhase.BOOSTER_PACK
    agent = HeuristicAgent(shop_policy="search")
    mask = runner.compute_mask()
    assert (mask == compute_action_mask(state, SubPhase.BOOSTER_PACK)).all()
    assert not mask[ActionRange.PACK_CLAIM_START]
    action = agent.select_action(state, SubPhase.BOOSTER_PACK, mask)
    assert action == ActionRange.SHOP_SELL_JOKER_START + 4
    runner.step(action)
    action = agent.select_action(state, runner.sub_phase, runner.compute_mask())
    assert action == ActionRange.PACK_CLAIM_START
    runner.step(action)
    assert "j_cavendish" in state.joker_keys
    assert "j_joker" not in state.joker_keys


@pytest.mark.parametrize("phase", [SubPhase.SHOP, SubPhase.BOOSTER_PACK])
@pytest.mark.parametrize("negative_offer", [False, True])
def test_full_roster_keeps_negative_joker_when_sale_cannot_free_a_slot(phase, negative_offer):
    from pylatro.models import PackState

    state = create_run_state("negative_replacement", deck_key="b_blue")
    state.dollars = 20
    for _ in range(5):
        add_joker(state, "j_joker").eternal = True
    add_joker(state, "j_joker", edition={"negative": True})
    offer = create_card_spec(state, "Joker", forced_key="j_cavendish")
    offer.edition = {"negative": True} if negative_offer else None
    if phase == SubPhase.SHOP:
        state.shop.cards = [offer]
        acquire = ActionRange.SHOP_BUY_START
    else:
        state.pack = PackState("p_buffoon_normal_1", "BUFFOON_PACK", 1, cards=[offer])
        acquire = ActionRange.PACK_CLAIM_START
    mask = compute_action_mask(state, phase)
    assert bool(mask[acquire]) == negative_offer
    assert mask[ActionRange.SHOP_SELL_JOKER_START + 5]
    before = pickle.dumps(state)
    action = HeuristicAgent(shop_policy="search").select_action(state, phase, mask)
    assert action != ActionRange.SHOP_SELL_JOKER_START + 5
    if negative_offer:
        assert action == acquire
    elif phase == SubPhase.BOOSTER_PACK:
        assert action == ActionRange.PACK_SKIP
    assert pickle.dumps(state) == before


def test_blue_generation_keeps_configuration_in_both_passes(monkeypatch):
    from pylatro_agent.tokenizer import Tokenizer
    from pylatro_agent.training import fast_generate
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_agent.vocab import build_vocab

    data = create_run_state("generation_data").data
    seen = []

    def one_step(seed, data, **kwargs):
        kwargs["max_steps"] = 1
        runner = FastRunner(seed, data, **kwargs)
        seen.append((runner.state.deck_key, runner.state.stake, runner.state.round_resets.hands))
        return runner

    monkeypatch.setattr(fast_generate, "FastRunner", one_step)
    agent = HeuristicAgent(shop_policy="search")
    fast_generate._run_game_fast_no_obs(0, data, agent, deck_key="b_blue", stake=1)
    records, _, _ = fast_generate._run_game_single_pass(
        0,
        data,
        Tokenizer(vocab=build_vocab(data)),
        agent,
        0.997,
        deck_key="b_blue",
        stake=1,
    )
    assert seen == [("b_blue", 1, 5), ("b_blue", 1, 5)]
    assert records[0]["deck_key"] == "b_blue"
    assert records[0]["stake"] == 1


@pytest.mark.parametrize("blind_rollout,confirm_early", [(None, False), ("all", False), ("all", True)])
def test_generation_chunks_do_not_repeat_episode_seeds(monkeypatch, blind_rollout, confirm_early):
    from pylatro_agent.training import fast_generate

    calls = []

    def batch(num_games, workers, min_ante, gamma, keep, win_ante, reward, excluded, game_options):
        calls.append((excluded, game_options))
        seed = next(seed for seed in range(20) if seed not in excluded)
        return [{"seed": seed}]

    monkeypatch.setattr(fast_generate, "_generate_batch", batch)
    records = fast_generate.generate_training_data(
        3,
        num_workers=1,
        chunk_size=1,
        excluded_seeds=(0,),
        deck_key="b_blue",
        stake=1,
        shop_policy="search",
        blind_rollout=blind_rollout, blind_confirm_early=confirm_early,
    )
    assert [record["seed"] for record in records] == [1, 2, 3]
    assert all(
        options == {"deck_key": "b_blue", "stake": 1, "shop_policy": "search",
                    "grow_scalers": False, "blind_rollout": blind_rollout, "blind_confirm_early": confirm_early}
        for _, options in calls
    )


def test_shop_blue_joker_forecast_uses_fresh_sampled_draw_pile():
    state = create_run_state("blue_joker_public_deck", deck_key="b_blue")
    add_joker(state, "j_blue_joker")
    state.draw_pile = list(state.deck_cards)
    full = ShopSearch().output(state, HeuristicAgent())
    state.draw_pile = []
    empty = ShopSearch().output(state, HeuristicAgent())
    assert full == empty


def test_card_sharp_establishes_unleveled_pair_without_wasting_discards():
    from pylatro import get_poker_hand_info, select_blind, start_blind
    from pylatro.models import PlayingCard
    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.subset_actions import subset_indices

    state = create_run_state("sharp_unleveled", deck_key="b_blue")
    state.round_resets.ante = 2
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.hand_cards = [
        PlayingCard(front_key=f"{suit[0]}_{rank}", suit=suit, rank=rank)
        for suit, rank in [
            ("Spades", "K"),
            ("Hearts", "K"),
            ("Clubs", "Q"),
            ("Diamonds", "Q"),
            ("Hearts", "9"),
            ("Clubs", "7"),
            ("Spades", "4"),
            ("Diamonds", "2"),
        ]
    ]
    add_joker(state, "j_joker")
    add_joker(state, "j_card_sharp")
    action = HeuristicAgent().select_action(
        state, SubPhase.CHOOSE_ACTION, compute_action_mask(state, SubPhase.CHOOSE_ACTION), round_score=0
    )
    decoded = decode_action(action)
    assert decoded.action_type == ActionType.PLAY_SUBSET
    played = [state.hand_cards[i] for i in subset_indices(decoded.index)]
    assert get_poker_hand_info(state, played)[0] == "Pair"


def test_boss_forecast_accounts_for_pillar_card_debuffs_without_mutation():
    state = create_run_state("pillar_forecast", deck_key="b_blue")
    state.round_resets.blind_choices["Boss"] = "bl_pillar"
    for card in state.deck_cards:
        card.played_this_ante = True
    original = pickle.dumps(state)
    search = ShopSearch()
    normal, _ = search.output(state, HeuristicAgent())
    boss, _ = search.output(state, HeuristicAgent(), boss=True)
    assert boss < normal
    assert pickle.dumps(state) == original


def test_eye_forecast_does_not_assume_repeated_card_sharp_hands():
    state = create_run_state("eye_forecast", deck_key="b_blue")
    add_joker(state, "j_half")
    add_joker(state, "j_card_sharp")
    state.round_resets.blind_choices["Boss"] = "bl_eye"
    search = ShopSearch()
    agent = HeuristicAgent()
    normal, _ = search.output(state, agent)
    boss, _ = search.output(state, agent, boss=True)
    assert boss < normal * 0.75


def test_search_keeps_the_planned_boss_reroll_budget():
    state = create_run_state("needle_reserve", deck_key="b_blue")
    state.used_vouchers["v_directors_cut"] = True
    state.dollars = 12
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_needle"
    add_joker(state, "j_joker")
    state.shop.cards = [create_card_spec(state, "Joker", forced_key="j_cavendish")]
    mask = compute_action_mask(state, SubPhase.SHOP)
    action = HeuristicAgent(shop_policy="search").select_action(state, SubPhase.SHOP, mask)
    assert action == ActionRange.SHOP_LEAVE


def test_search_buys_directors_cut_before_spending_its_known_needle_escape_budget():
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_cli.controller import GamePhase

    runner = FastRunner(0, create_run_state("early_needle_voucher").data, deck_key="b_blue", raise_errors=True)
    state = runner.state
    runner._sub_phase = SubPhase.SHOP
    runner._ctrl.phase = GamePhase.SHOP
    state.round_resets.ante = 6
    state.blind_on_deck = "Small"
    state.round_resets.blind_choices["Boss"] = "bl_needle"
    state.dollars = 25
    add_joker(state, "j_half")
    add_joker(state, "j_sly")
    state.shop.vouchers = [create_card_spec(state, "Voucher", forced_key="v_directors_cut")]
    state.shop.boosters = [create_card_spec(state, "Booster", forced_key="p_celestial_jumbo_1")]
    state.shop.vouchers[0].cost = 10
    state.shop.boosters[0].cost = 6
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(state, SubPhase.SHOP, runner.compute_mask())
    assert action == ActionRange.SHOP_BUY_START
    runner.step(action)
    assert state.used_vouchers.get("v_directors_cut")
    assert state.dollars == 15
    action = agent.select_action(state, SubPhase.SHOP, runner.compute_mask())
    assert action == ActionRange.SHOP_LEAVE


def test_conditional_chip_joker_does_not_force_a_flush_plan():
    state = create_run_state("crafty_plan", deck_key="b_blue")
    add_joker(state, "j_crafty", edition={"holo": True})
    assert HeuristicAgent()._get_main_hand_type(state) == "Pair"


def test_green_joker_does_not_discard_away_growth_for_unneeded_summit():
    from pylatro import select_blind, start_blind
    from pylatro.models import PlayingCard
    from pylatro_agent.action import ActionType, decode_action

    state = create_run_state("green_summit", deck_key="b_blue")
    select_blind(state, "Small")
    start_blind(state, "Small")
    state.hand_cards = [
        PlayingCard(front_key=f"{s[0]}_{r}", suit=s, rank=r)
        for s, r in [
            ("Spades", "K"),
            ("Hearts", "K"),
            ("Clubs", "Q"),
            ("Diamonds", "Q"),
            ("Hearts", "9"),
            ("Clubs", "7"),
            ("Spades", "4"),
            ("Diamonds", "2"),
        ]
    ]
    green = add_joker(state, "j_green_joker")
    green.mult = 10
    add_joker(state, "j_mystic_summit")
    action = HeuristicAgent().select_action(
        state, SubPhase.CHOOSE_ACTION, compute_action_mask(state, SubPhase.CHOOSE_ACTION), round_score=0
    )
    assert decode_action(action).action_type == ActionType.PLAY_SUBSET


def test_weak_suit_joker_preserves_a_four_card_flush_draw():
    from pylatro import select_blind, start_blind
    from pylatro.models import PlayingCard

    state = create_run_state("weak_suit_draw", deck_key="b_blue")
    state.round_resets.ante = 2
    select_blind(state, "Small")
    start_blind(state, "Small")
    add_joker(state, "j_greedy_joker")
    state.hand_cards = [
        PlayingCard(front_key=f"{s[0]}_{r}", suit=s, rank=r)
        for s, r in [
            ("Diamonds", "2"), ("Diamonds", "5"), ("Diamonds", "8"), ("Diamonds", "Q"),
            ("Spades", "Q"), ("Clubs", "7"), ("Hearts", "9"), ("Clubs", "K"),
        ]
    ]
    agent = HeuristicAgent(shop_policy="search")
    discard = agent._should_discard_for_draw(
        state, state.hand_cards, compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    )
    assert discard == (4, 5, 6, 7)


def test_weak_poor_build_plays_instead_of_skipping_for_economy():
    state = create_run_state("weak_economy_skip", deck_key="b_blue")
    state.blind_on_deck = "Big"
    state.round_resets.blind_tags["Big"] = "tag_economy"
    state.dollars = 7
    action = HeuristicAgent().select_action(
        state, SubPhase.BLIND_SELECT, compute_action_mask(state, SubPhase.BLIND_SELECT)
    )
    assert action == ActionRange.BLIND_PLAY


def test_forecast_reuses_irrelevant_history_but_matches_an_uncached_result():
    state = create_run_state("forecast_history_cache", deck_key="b_blue")
    for key in ("j_half", "j_blue_joker", "j_card_sharp"):
        add_joker(state, key)
    agent = HeuristicAgent()
    search = ShopSearch()
    expected = search.output(state, agent)
    entries = len(search.cache)
    state.hands["Pair"]["played"] = 10
    state.hands["Pair"]["played_this_round"] = 4
    state.hands_played += 10
    state.current_round.ancient_card = {"suit": "Hearts"}
    state.current_round.idol_card = {"suit": "Spades", "rank": "K", "id": 13}
    assert search.output(state, agent) == expected
    assert len(search.cache) == entries
    assert ShopSearch().output(state, agent) == expected


@pytest.mark.parametrize("joker", ["j_supernova", "j_obelisk", "j_ancient", "j_idol", "j_loyalty_card"])
def test_forecast_cache_keeps_scoring_sensitive_history(joker):
    state = create_run_state("forecast_sensitive_cache", deck_key="b_blue")
    add_joker(state, "j_half")
    instance = add_joker(state, joker)
    agent = HeuristicAgent()
    search = ShopSearch()
    search.output(state, agent)
    entries = len(search.cache)
    if joker in {"j_supernova", "j_obelisk"}:
        state.hands["Pair"]["played"] += 10
    elif joker == "j_ancient":
        state.current_round.ancient_card = {"suit": "Hearts"}
    elif joker == "j_idol":
        state.current_round.idol_card = {"suit": "Spades", "rank": "K", "id": 13}
    else:
        state.hands_played += 5
        instance.hands_played_at_create = 2
    assert search.output(state, agent) == ShopSearch().output(state, agent)
    assert len(search.cache) > entries


@pytest.mark.parametrize("blind,target", [("Small", 50000), ("Big", 75000), ("Boss", 300000)])
def test_search_spending_target_is_the_next_blind(blind, target):
    state = create_run_state("next_blind_budget", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = blind
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    assert ShopSearch.target(state, HeuristicAgent()) == target


@pytest.mark.parametrize("counter,target", [(False, 300000), (True, 100000)])
def test_future_boss_target_ignores_the_previous_blinds_disabled_flag(counter, target):
    state = create_run_state("future_disabled_target", deck_key="b_blue")
    state.round_resets.ante = 8
    state.blind_on_deck = "Boss"
    state.round_resets.blind_choices["Boss"] = "bl_final_vessel"
    state.blind_disabled = True
    if counter:
        add_joker(state, "j_chicot")
    assert ShopSearch.target(state, HeuristicAgent()) == target


def test_safe_small_blind_budget_does_not_depend_on_a_later_wall():
    state = create_run_state("later_wall_budget", deck_key="b_blue")
    state.round_resets.ante = 5
    state.blind_on_deck = "Small"
    state.round_resets.blind_choices["Boss"] = "bl_wall"
    state.dollars = 17
    for key in ("j_half", "j_blue_joker", "j_abstract"):
        add_joker(state, key)
    state.hands["Pair"].update(level=2, chips=25, mult=3)
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(state, SubPhase.SHOP, compute_action_mask(state, SubPhase.SHOP))
    assert agent._shop_search.last_decision["safe"]
    # Investment for later antes can still justify a reroll. A known Wall
    # should not change this safe Small blind into an immediate emergency.
    state.round_resets.blind_choices["Boss"] = "bl_head"
    ordinary = HeuristicAgent(shop_policy="search")
    assert ordinary.select_action(state, SubPhase.SHOP, compute_action_mask(state, SubPhase.SHOP)) == action
    assert ordinary._shop_search.last_decision["safe"]


@pytest.mark.parametrize("joker,levels", [("j_space", 4), ("j_burnt", 6)])
def test_hand_level_scalers_have_bounded_projected_growth(joker, levels):
    state = create_run_state("hand_level_projection", deck_key="b_blue")
    state.round_resets.ante = 2
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    add_joker(state, joker)
    agent = HeuristicAgent(shop_policy="search")
    before = pickle.dumps(state)
    projected = ShopSearch.projected_growth(state, agent)
    assert projected.hands["Pair"]["level"] == state.hands["Pair"]["level"] + levels
    assert projected.hands["Pair"]["chips"] > state.hands["Pair"]["chips"]
    assert pickle.dumps(state) == before


def test_burglar_removes_burnts_projected_discard_growth():
    state = create_run_state("burnt_burglar_projection", deck_key="b_blue")
    add_joker(state, "j_burnt")
    add_joker(state, "j_burglar")
    assert ShopSearch.projected_growth(state, HeuristicAgent(shop_policy="search")) is None


@pytest.mark.parametrize("joker", ["j_space", "j_burnt"])
def test_safe_build_can_buy_a_hand_level_scaler(joker):
    state = create_run_state("buy_hand_level_scaler", deck_key="b_blue")
    state.round_resets.ante = 2
    state.dollars = 40
    add_joker(state, "j_half")
    add_joker(state, "j_blue_joker")
    state.shop.cards = [create_card_spec(state, "Joker", forced_key=joker)]
    agent = HeuristicAgent(shop_policy="search")
    action = agent.select_action(state, SubPhase.SHOP, compute_action_mask(state, SubPhase.SHOP))
    assert action == ActionRange.SHOP_BUY_START


@pytest.mark.parametrize("joker", ["j_space", "j_burnt"])
def test_projected_levels_change_purchase_value_but_not_immediate_capacity(joker):
    state = create_run_state("level_value_not_survival", deck_key="b_blue")
    state.round_resets.ante = 3
    for key in ("j_half", "j_blue_joker", joker):
        add_joker(state, key)
    agent = HeuristicAgent(shop_policy="search")
    search = ShopSearch()
    now, immediate_value = search.output(state, agent, _project=False)
    projected_now, future_value = search.output(state, agent)
    assert projected_now == now
    assert future_value > immediate_value
