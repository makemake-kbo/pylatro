from __future__ import annotations

import numpy as np

from pylatro import add_consumable, add_joker, create_run_state, load_game_data
from pylatro.instances import move_joker
from pylatro.models import PlayingCard, ShopCard
from pylatro_agent.action import decode_action
from pylatro_agent.constants import NUM_ACTIONS, ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask
from pylatro_agent.subset_actions import subset_index


def _shop_card(center_key: str, card_type: str, cost: int = 3) -> ShopCard:
    return ShopCard(center_key=center_key, card_type=card_type, cost=cost, base_cost=cost)


def test_joker_move_search_skips_build_without_xmult_or_copy(monkeypatch) -> None:
    data = load_game_data()
    state = create_run_state("reorder_search_gate", data=data)
    add_joker(state, "j_joker")
    add_joker(state, "j_sly")
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)

    def unexpected_score(*_args, **_kwargs):
        raise AssertionError("ordinary builds should skip ordered hand scoring")

    monkeypatch.setattr(HeuristicAgent, "_score_joker_move_for_hand", unexpected_score)

    assert HeuristicAgent()._best_joker_move(state, mask, (0,)) is None


def test_joker_move_search_orders_xmult_for_selected_hand() -> None:
    data = load_game_data()
    state = create_run_state("reorder_search_xmult", data=data)
    add_joker(state, "j_cavendish")
    add_joker(state, "j_joker")
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)

    action = HeuristicAgent()._best_joker_move(state, mask, (0,))
    decoded = decode_action(action)
    move_joker(state, decoded.index, decoded.detail)

    assert state.joker_keys == ["j_joker", "j_cavendish"]


def test_selected_play_emits_move_then_play_for_bc() -> None:
    data = load_game_data()
    state = create_run_state("reorder_then_play", data=data)
    add_joker(state, "j_cavendish")
    add_joker(state, "j_joker")
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)
    play_action = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    agent = HeuristicAgent()

    first_action = agent._play_or_reorder_jokers(state, mask, play_action)
    decoded = decode_action(first_action)
    move_joker(state, decoded.index, decoded.detail)
    second_action = agent._play_or_reorder_jokers(state, mask, play_action)

    assert decode_action(first_action).action_type.value == "move_joker"
    assert second_action == play_action


def test_selected_play_reuses_cached_joker_order_plan(monkeypatch) -> None:
    data = load_game_data()
    state = create_run_state("reorder_plan_cache", data=data)
    add_joker(state, "j_cavendish")
    add_joker(state, "j_joker")
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)
    play_action = ActionRange.PLAY_SUBSET_START + subset_index((0,))
    agent = HeuristicAgent()
    original_score = agent._score_joker_move_for_hand
    score_calls = 0

    def counted_score(*args, **kwargs):
        nonlocal score_calls
        score_calls += 1
        return original_score(*args, **kwargs)

    monkeypatch.setattr(agent, "_score_joker_move_for_hand", counted_score)

    first_action = agent._play_or_reorder_jokers(state, mask, play_action)
    first_search_calls = score_calls
    decoded = decode_action(first_action)
    move_joker(state, decoded.index, decoded.detail)
    second_action = agent._resume_joker_order_plan(state, mask)

    assert first_search_calls == 2
    assert score_calls == first_search_calls
    assert second_action == play_action


def test_joker_order_candidate_search_has_hard_budget() -> None:
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

    candidates = HeuristicAgent()._joker_order_candidates(state, len(state.jokers))

    assert len(candidates) <= 16
    assert len(candidates) == len(set(candidates))
    assert all(sorted(order) == list(range(len(state.jokers))) for order in candidates)


def test_joker_move_search_orders_copy_without_xmult() -> None:
    data = load_game_data()
    state = create_run_state("reorder_search_copy", data=data)
    add_joker(state, "j_joker")
    add_joker(state, "j_blueprint")
    state.hand_cards = [PlayingCard(front_key="S_A", suit="Spades", rank="A")]
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)

    action = HeuristicAgent()._best_joker_move(state, mask, (0,))
    decoded = decode_action(action)
    move_joker(state, decoded.index, decoded.detail)

    assert state.joker_keys == ["j_blueprint", "j_joker"]


def test_joker_move_balances_retrigger_and_idol_effects() -> None:
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
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)
    agent = HeuristicAgent()

    for _ in range(len(state.jokers)):
        action = agent._best_joker_move(state, mask, (0,))
        if action is None:
            break
        decoded = decode_action(action)
        move_joker(state, decoded.index, decoded.detail)

    blueprint_index = state.joker_keys.index("j_blueprint")
    assert state.joker_keys[blueprint_index + 1] == "j_idol"


def test_joker_move_copies_retrigger_when_it_repeats_more_effects() -> None:
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
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)
    agent = HeuristicAgent()

    for _ in range(len(state.jokers)):
        action = agent._best_joker_move(state, mask, (0,))
        if action is None:
            break
        decoded = decode_action(action)
        move_joker(state, decoded.index, decoded.detail)

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


def test_midgame_prefers_played_planet_over_chip_joker() -> None:
    data = load_game_data()
    state = create_run_state("midgame_planet_scaling", data=data)
    state.round_resets.ante = 4
    state.dollars = 6
    state.hands["Two Pair"]["played"] = 1
    state.shop.cards = [
        _shop_card("j_sly", "Joker", cost=3),
        _shop_card("c_uranus", "Planet", cost=3),
    ]

    mask = compute_action_mask(state, SubPhase.SHOP)

    action = HeuristicAgent().select_action(state, SubPhase.SHOP, mask)

    assert action == ActionRange.SHOP_BUY_START + 1


def test_midgame_sells_consumable_for_played_planet() -> None:
    data = load_game_data()
    state = create_run_state("midgame_planet_room", data=data)
    state.round_resets.ante = 4
    state.dollars = 6
    state.starting_params.consumable_slots = 1
    state.hands["Two Pair"]["played"] = 1
    add_consumable(state, "c_magician")
    state.shop.cards = [_shop_card("c_uranus", "Planet", cost=3)]

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
