from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from pylatro import add_joker, create_run_state
from pylatro.models import PlayingCard, ShopCard
from pylatro.scoring import score_hand
from pylatro_agent.build_value import estimate_build_value
from pylatro_agent.shop_eval import capture_build_features


def joker(key: str, **over: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "key": key,
        "name": key,
        "effect": "",
        "type": "",
        "config": {},
        "base_mult": 0.0,
        "base_x_mult": 1.0,
        "base_t_mult": 0.0,
        "base_t_chips": 0.0,
        "mult": 0.0,
        "t_mult": 0.0,
        "t_chips": 0.0,
        "x_mult": 1.0,
        "h_mult": 0.0,
        "h_x_mult": 0.0,
        "extra": None,
        "dollars": 0.0,
        "edition": {},
        "eternal": False,
        "perishable": False,
        "perish_tally": None,
        "rental": False,
        "debuffed": False,
        "is_economy": False,
        "is_scaling": False,
        "is_scaling_xmult": False,
        "is_retrigger": False,
        "blueprint_compat": True,
        "copy_compatible": True,
        "sell_cost": 2,
    }
    value.update(over)
    return value


def card(rank: str, suit: str, enhancement: str = "", seal: str = "") -> dict[str, Any]:
    return {"rank": rank, "suit": suit, "enhancement": enhancement, "seal": seal, "edition": ""}


def info(cards: list[dict[str, Any]], jokers: list[dict[str, Any]] | None = None, slots: int = 5) -> dict[str, Any]:
    owned = jokers or []
    return {
        "ante": 2,
        "dollars": 12,
        "interest_cap_cash": 25,
        "reroll_cost": 5,
        "joker_details": tuple(owned),
        "joker_slots_used": len(owned),
        "joker_slots_limit": slots,
        "joker_slots_left": max(0, slots - len(owned)),
        "hand_play_counts": {"Pair": 8, "High Card": 1},
        "hand_levels": {"Pair": 1, "High Card": 1},
        "hand_details": {
            "Pair": {"chips": 10, "mult": 2, "level": 1, "played": 8},
            "High Card": {"chips": 5, "mult": 1, "level": 1, "played": 1},
        },
        "hands_available": 4,
        "hand_size": 8,
        "blind_target": 300,
        "idol_card": {"rank": "2", "suit": "Hearts", "id": 2},
        "deck_stats": {"size": len(cards), "cards": tuple(cards)},
        "shop_cards": (),
    }


def test_representative_hand_uses_play_count_then_level() -> None:
    snapshot = info([card("8", "Clubs") for _ in range(52)])
    snapshot["hand_details"]["High Card"] = {"chips": 25, "mult": 3, "level": 3, "played": 8}
    snapshot["hand_play_counts"] = {"Pair": 8, "High Card": 8}

    estimate = estimate_build_value(snapshot)
    assert estimate.representative_hand_type == "High Card"
    assert estimate.required_score_per_hand == 75


@pytest.mark.parametrize(
    ("enhancement", "center_key"),
    [
        ("", "c_base"),
        ("Mult Card", "m_mult"),
        ("Glass Card", "m_glass"),
    ],
)
def test_representative_pair_card_math_matches_real_engine(
    enhancement: str,
    center_key: str,
) -> None:
    cards = [card("8", "Clubs", enhancement) for _ in range(52)]
    analytic = estimate_build_value(info(cards))

    state = create_run_state("AAAAAAAA")
    played = [
        PlayingCard(front_key="C_8", suit="Clubs", rank="8", center_key=center_key),
        PlayingCard(front_key="C_8", suit="Clubs", rank="8", center_key=center_key),
    ]
    exact = score_hand(state, played, held_hand=[]).total

    assert analytic.representative_score_per_hand == exact


def test_direct_hand_chips_and_mult_use_score_context() -> None:
    pair_chips = joker("j_pair_chips", type="Pair", t_chips=50)
    pair_mult = joker("j_pair_mult", type="Pair", t_mult=4)
    offhand = joker("j_flush_chips", type="Flush", t_chips=100)
    snapshot = info([card("8", "Clubs") for _ in range(52)], [pair_chips, pair_mult, offhand])

    estimate = estimate_build_value(snapshot)
    assert estimate.joker_marginal_score_ratios[0] > 1.0
    assert estimate.joker_marginal_score_ratios[1] > 1.0
    assert estimate.joker_marginal_score_ratios[2] == pytest.approx(1.0)


def test_typed_joker_matches_nested_real_engine_hand() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_duo")
    full_house = [
        PlayingCard("S_2", "Spades", "2"),
        PlayingCard("H_2", "Hearts", "2"),
        PlayingCard("D_2", "Diamonds", "2"),
        PlayingCard("S_3", "Spades", "3"),
        PlayingCard("H_3", "Hearts", "3"),
    ]
    exact_with = score_hand(deepcopy(state), deepcopy(full_house), held_hand=[]).total
    without = deepcopy(state)
    without.jokers.clear()
    without.joker_keys.clear()
    exact_without = score_hand(without, deepcopy(full_house), held_hand=[]).total

    snapshot = capture_build_features(state)
    snapshot.update({"ante": 2, "blind_target": 300, "hands_available": 4, "hand_size": 8})
    for detail in snapshot["hand_details"].values():
        detail["played"] = 0
    snapshot["hand_details"]["Full House"]["played"] = 8
    analytic = estimate_build_value(snapshot)

    assert analytic.representative_hand_type == "Full House"
    assert analytic.joker_marginal_score_ratios[0] == pytest.approx(exact_with / exact_without)


@pytest.mark.parametrize(
    ("copy_key", "ordered_keys", "copy_index"),
    [
        ("j_blueprint", ("j_blueprint", "j_joker"), 0),
        ("j_brainstorm", ("j_joker", "j_brainstorm"), 1),
    ],
)
def test_copy_joker_matches_simple_real_engine_target(
    copy_key: str,
    ordered_keys: tuple[str, str],
    copy_index: int,
) -> None:
    state = create_run_state("AAAAAAAA")
    for key in ordered_keys:
        add_joker(state, key)
    pair = [
        PlayingCard("S_2", "Spades", "2"),
        PlayingCard("H_2", "Hearts", "2"),
    ]
    exact_with = score_hand(deepcopy(state), deepcopy(pair), held_hand=[]).total
    without = deepcopy(state)
    without.jokers.pop(copy_index)
    without.joker_keys.pop(copy_index)
    exact_without = score_hand(without, deepcopy(pair), held_hand=[]).total

    snapshot = capture_build_features(state)
    uniform_cards = tuple(card("2", "Spades") for _ in range(52))
    snapshot["deck_stats"]["cards"] = uniform_cards
    snapshot["deck_stats"]["card_descriptors"] = uniform_cards
    snapshot["deck_stats"]["size"] = len(uniform_cards)
    snapshot.update({"ante": 2, "blind_target": 300, "hands_available": 4, "hand_size": 8})
    for detail in snapshot["hand_details"].values():
        detail["played"] = 0
    snapshot["hand_details"]["Pair"]["played"] = 8
    analytic = estimate_build_value(snapshot)

    assert analytic.joker_marginal_score_ratios[copy_index] == pytest.approx(exact_with / exact_without)
    assert analytic.joker_marginals[copy_index].modeled_effect_fraction == 1.0


def test_mime_retriggers_plain_baron_and_shoot_the_moon_effects() -> None:
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_baron")
    add_joker(state, "j_shoot_the_moon")
    add_joker(state, "j_mime")
    pair = [
        PlayingCard("S_2", "Spades", "2"),
        PlayingCard("H_2", "Hearts", "2"),
    ]
    held = [
        PlayingCard("S_K", "Spades", "K"),
        PlayingCard("H_K", "Hearts", "K"),
        PlayingCard("D_K", "Diamonds", "K"),
        PlayingCard("S_Q", "Spades", "Q"),
        PlayingCard("H_Q", "Hearts", "Q"),
        PlayingCard("D_Q", "Diamonds", "Q"),
    ]
    exact_with = score_hand(deepcopy(state), deepcopy(pair), held_hand=deepcopy(held)).total
    without = deepcopy(state)
    without.jokers.pop(2)
    without.joker_keys.pop(2)
    exact_without = score_hand(without, deepcopy(pair), held_hand=deepcopy(held)).total

    cards = [card("K", "Spades") for _ in range(3)]
    cards += [card("Q", "Hearts") for _ in range(3)]
    cards += [card("2", "Clubs") for _ in range(46)]
    snapshot = info(cards)
    snapshot["joker_details"] = capture_build_features(state)["joker_details"]
    snapshot["joker_slots_used"] = 3
    snapshot["joker_slots_left"] = 2
    analytic = estimate_build_value(snapshot)

    # Wiki activation sequence: Mime repeats Joker-provided held abilities too.
    assert (exact_with, exact_without) == (1410, 640)
    assert analytic.joker_marginal_score_ratios[2] > 1.0


def test_suit_joker_uses_deck_concentration() -> None:
    lusty = joker(
        "j_lusty_joker",
        name="Lusty Joker",
        effect="Suit Mult",
        extra={"suit": "Hearts", "s_mult": 3},
    )
    hearts = info([card("7", "Hearts") for _ in range(40)] + [card("7", "Diamonds") for _ in range(12)], [lusty])
    diamonds = info([card("7", "Diamonds") for _ in range(40)] + [card("7", "Hearts") for _ in range(12)], [lusty])

    assert (
        estimate_build_value(hearts).representative_score_per_hand
        > estimate_build_value(diamonds).representative_score_per_hand
    )


def test_smeared_and_wild_cards_support_suit_effects() -> None:
    lusty = joker(
        "j_lusty_joker",
        name="Lusty Joker",
        effect="Suit Mult",
        extra={"suit": "Hearts", "s_mult": 3},
    )
    smeared = joker("j_smeared", name="Smeared Joker")
    red = info([card("7", "Diamonds") for _ in range(52)], [lusty, smeared])
    wild = info([card("7", "Clubs", "Wild Card") for _ in range(52)], [lusty])
    plain = info([card("7", "Clubs") for _ in range(52)], [lusty])

    assert (
        estimate_build_value(red).representative_score_per_hand
        > estimate_build_value(plain).representative_score_per_hand
    )
    assert (
        estimate_build_value(wild).representative_score_per_hand
        > estimate_build_value(plain).representative_score_per_hand
    )


def test_hack_targets_ranks_two_through_five() -> None:
    hack = joker("j_hack", name="Hack", extra=1, is_retrigger=True)
    low = estimate_build_value(info([card("2", "Hearts") for _ in range(52)], [hack]))
    faces = estimate_build_value(info([card("K", "Hearts") for _ in range(52)], [hack]))

    assert low.joker_marginal_score_ratios[0] > 1.0
    assert faces.joker_marginal_score_ratios[0] == pytest.approx(1.0)


def test_hack_retriggers_glass_low_cards() -> None:
    hack = joker("j_hack", name="Hack", extra=1, is_retrigger=True)
    plain = estimate_build_value(info([card("2", "Hearts") for _ in range(52)], [hack]))
    glass = estimate_build_value(info([card("2", "Hearts", "Glass Card") for _ in range(52)], [hack]))

    assert glass.joker_marginal_score_ratios[0] > plain.joker_marginal_score_ratios[0]
    assert glass.channels.retriggers > 0


def test_idol_only_uses_current_exact_rank_and_suit() -> None:
    idol = joker("j_idol", name="The Idol", extra=2)
    exact = estimate_build_value(
        info([card("2", "Hearts") for _ in range(8)] + [card("K", "Clubs") for _ in range(44)], [idol])
    )
    unrelated_glass = estimate_build_value(
        info([card("2", "Diamonds", "Glass Card") for _ in range(8)] + [card("K", "Clubs") for _ in range(44)], [idol])
    )

    assert exact.joker_marginal_score_ratios[0] > 1.0
    assert unrelated_glass.joker_marginal_score_ratios[0] == pytest.approx(1.0)
    assert unrelated_glass.representative_score_per_hand > unrelated_glass.no_joker_baseline_score - 1


def test_live_hologram_xmult_is_used() -> None:
    fresh = estimate_build_value(
        info([card("8", "Clubs") for _ in range(52)], [joker("j_hologram", name="Hologram", x_mult=1.0)])
    )
    scaled = estimate_build_value(
        info([card("8", "Clubs") for _ in range(52)], [joker("j_hologram", name="Hologram", x_mult=1.75)])
    )

    assert scaled.representative_score_per_hand > fresh.representative_score_per_hand
    assert scaled.joker_marginal_score_ratios[0] > 1.7


def test_static_base_xmult_fallback_preserves_joker_order() -> None:
    cavendish = joker("j_cavendish", name="Cavendish", base_x_mult=3.0, x_mult=1.0)
    additive = joker("j_joker", name="Joker", mult=4.0)
    early_xmult = estimate_build_value(
        info([card("8", "Clubs") for _ in range(52)], [cavendish, additive])
    )
    late_xmult = estimate_build_value(
        info([card("8", "Clubs") for _ in range(52)], [additive, cavendish])
    )

    assert late_xmult.representative_score_per_hand > early_xmult.representative_score_per_hand


def test_joker_edition_order_and_debuff_are_modeled() -> None:
    poly = joker("j_poly", edition={"polychrome": True})
    additive = joker("j_add", mult=10)
    early_poly = estimate_build_value(info([card("8", "Clubs") for _ in range(52)], [poly, additive]))
    late_poly = estimate_build_value(info([card("8", "Clubs") for _ in range(52)], [additive, poly]))
    debuffed = estimate_build_value(
        info([card("8", "Clubs") for _ in range(52)], [joker("j_hologram", x_mult=3.0, debuffed=True)])
    )

    assert late_poly.representative_score_per_hand > early_poly.representative_score_per_hand
    assert debuffed.representative_score_per_hand == debuffed.no_joker_baseline_score


def test_unknown_conditional_effect_has_no_static_fallback() -> None:
    unknown = joker("j_unknown", name="Unknown scaler", effect="Mystery", is_scaling=True)
    estimate = estimate_build_value(info([card("8", "Clubs") for _ in range(52)], [unknown]))

    assert estimate.representative_score_per_hand == estimate.no_joker_baseline_score
    assert estimate.joker_marginal_score_ratios == (1.0,)
    assert estimate.joker_marginals[0].unmodeled_effects == ("conditional_effect",)


def test_capture_copies_live_context_and_shop_stickers() -> None:
    state = create_run_state("AAAAAAAA")
    hologram = add_joker(state, "j_hologram", edition={"foil": True})
    hologram.x_mult = 1.75
    state.current_round.idol_card["rank"] = "2"
    state.shop.cards = [
        ShopCard(
            center_key="j_lusty_joker",
            card_type="Joker",
            cost=5,
            base_cost=5,
            edition={"polychrome": True},
            eternal=True,
            rental=True,
        ),
        ShopCard(center_key="c_jupiter", card_type="Planet", cost=3, base_cost=3),
    ]
    snapshot = capture_build_features(state)
    state.current_round.idol_card["rank"] = "A"
    hologram.edition["foil"] = False

    assert snapshot["joker_details"][0]["x_mult"] == 1.75
    assert snapshot["joker_details"][0]["edition"] == {"foil": True}
    assert snapshot["idol_card"]["rank"] != "A"
    offered = snapshot["shop_cards"][0]
    assert offered["joker"]["edition"] == {"polychrome": True}
    assert offered["joker"]["eternal"] is True
    assert offered["joker"]["rental"] is True
    assert snapshot["shop_cards"][1]["hand_type"] == "Flush"
    assert snapshot["deck_stats"]["cards"]
    assert snapshot["hand_details"]["Pair"]["chips"] == state.hands["Pair"]["chips"]
