from __future__ import annotations

from pylatro import add_consumable, add_joker, create_run_state, load_game_data
from pylatro.models import PlayingCard, ShopCard
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.masks import compute_action_mask


def _shop_card(center_key: str, card_type: str, cost: int = 3) -> ShopCard:
    return ShopCard(center_key=center_key, card_type=card_type, cost=cost, base_cost=cost)


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
