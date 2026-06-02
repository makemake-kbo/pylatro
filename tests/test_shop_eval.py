from __future__ import annotations

from typing import Any

from pylatro_agent.shop_eval import (
    evaluate_build,
    evaluate_shop_opportunity,
    interest_tiers,
    score_shop_item,
)

# ───────────────────────── synthetic info builders ─────────────────────────


def mk_joker(key: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "key": key,
        "name": key,
        "rarity": 1,
        "type": "",
        "base_mult": 0.0,
        "base_x_mult": 1.0,
        "base_t_mult": 0.0,
        "base_t_chips": 0.0,
        "dollars": 0.0,
        "h_size": 0.0,
        "d_size": 0.0,
        "is_economy": False,
        "is_scaling": False,
        "is_scaling_xmult": False,
        "is_retrigger": False,
        "sell_cost": 3,
        "edition": {},
        "eternal": False,
        "perishable": False,
        "rental": False,
        "debuffed": False,
        "mult": 0.0,
        "t_mult": 0.0,
        "t_chips": 0.0,
        "x_mult": 1.0,
    }
    base.update(over)
    return base


def mk_info(
    *,
    jokers: list[dict[str, Any]] | None = None,
    shop: list[dict[str, Any]] | None = None,
    dollars: int = 10,
    ante: int = 1,
    interest_cap_cash: int = 25,
    joker_slots_limit: int = 5,
    reroll_cost: int = 5,
    hand_play_counts: dict[str, int] | None = None,
    deck_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    jokers = jokers or []
    return {
        "ante": ante,
        "dollars": dollars,
        "reroll_cost": reroll_cost,
        "interest_cap_cash": interest_cap_cash,
        "interest_amount": 1,
        "joker_slots_used": len(jokers),
        "joker_slots_limit": joker_slots_limit,
        "joker_slots_left": max(0, joker_slots_limit - len(jokers)),
        "joker_details": tuple(jokers),
        "consumable_details": (),
        "hand_play_counts": hand_play_counts or {},
        "hand_levels": {},
        "deck_stats": deck_stats or {"size": 52, "suit_counts": {}, "enhancement_counts": {}, "seal_counts": {}},
        "shop_cards": tuple(shop or ()),
    }


def joker_card(summary: dict[str, Any], cost: int = 5) -> dict[str, Any]:
    return {"index": 0, "key": summary["key"], "set": "Joker", "cost": cost, "joker": summary}


def consumable_card(key: str, cset: str, cost: int = 3, hand_type: str = "") -> dict[str, Any]:
    return {"index": 0, "key": key, "set": cset, "cost": cost, "type": hand_type}


# ───────────────────────────── interest tiers ─────────────────────────────


def test_interest_tiers_default_cap():
    assert interest_tiers(50, 25) == 5
    assert interest_tiers(20, 25) == 4
    assert interest_tiers(50, 50) == 10


# ───────────────────────────── build eval ─────────────────────────────


def test_empty_build_has_no_scoring_joker():
    b = evaluate_build(mk_info())
    assert not b.has_scoring_joker
    assert not b.has_xmult_joker
    assert b.survival_margin <= 0.5


def test_additive_mult_joker_is_scoring():
    b = evaluate_build(mk_info(jokers=[mk_joker("j_joker", mult=4.0)]))
    assert b.has_scoring_joker
    assert b.score_power > 0


def test_xmult_joker_detected():
    b = evaluate_build(mk_info(jokers=[mk_joker("j_cavendish", x_mult=3.0)]))
    assert b.has_xmult_joker
    assert b.xmult_power > 0


# ───────────────────────── first scoring joker ─────────────────────────


def test_first_scoring_joker_strongly_positive():
    info = mk_info(jokers=[], shop=[])
    b = evaluate_build(info)
    card = joker_card(mk_joker("j_joker", mult=4.0), cost=4)
    value = score_shop_item(info, card, b)
    # First reasonable scoring joker should clear a healthy threshold.
    assert value > 0.5


def test_scoring_joker_less_urgent_once_owned():
    owned = mk_info(jokers=[mk_joker("j_existing", mult=8.0)])
    b_owned = evaluate_build(owned)
    card = joker_card(mk_joker("j_joker", mult=4.0), cost=4)
    value_owned = score_shop_item(owned, card, b_owned)

    empty = mk_info(jokers=[])
    b_empty = evaluate_build(empty)
    value_empty = score_shop_item(empty, joker_card(mk_joker("j_joker", mult=4.0), cost=4), b_empty)

    # The same joker is worth more when we have no scoring engine yet.
    assert value_empty > value_owned


# ───────────────────────── xmult acquisition value ─────────────────────────


def test_xmult_worth_more_with_additive_base():
    no_base = mk_info(jokers=[])
    with_base = mk_info(jokers=[mk_joker("j_base", mult=12.0)])
    card = joker_card(mk_joker("j_cavendish", x_mult=3.0), cost=6)
    v_no_base = score_shop_item(no_base, card, evaluate_build(no_base))
    v_with_base = score_shop_item(with_base, card, evaluate_build(with_base))
    assert v_with_base > v_no_base


# ───────────────────────── economy joker context ─────────────────────────


def test_economy_joker_scored():
    info = mk_info(jokers=[], ante=2)
    card = joker_card(mk_joker("j_rocket", is_economy=True, dollars=2.0), cost=6)
    assert score_shop_item(info, card, evaluate_build(info)) > 0


# ───────────────────────── hand-specific synergy ─────────────────────────


def test_hand_specific_joker_matches_main_hand():
    counts = {"Flush": 10, "Pair": 1}
    matched = mk_info(jokers=[mk_joker("j_base", mult=10.0)], hand_play_counts=counts)
    b = evaluate_build(matched)
    assert b.main_hand_type == "Flush"
    flush_card = joker_card(mk_joker("j_flush", t_mult=4.0, type="Flush"), cost=5)
    offhand_card = joker_card(mk_joker("j_straight", t_mult=4.0, type="Straight"), cost=5)
    v_match = score_shop_item(matched, flush_card, b)
    v_off = score_shop_item(matched, offhand_card, b)
    assert v_match > v_off


# ───────────────────────── shop opportunity / reroll ─────────────────────────


def test_reroll_desirable_when_no_useful_buy():
    # Empty slots, money, but only a weak off-hand joker visible -> reroll desirable.
    info = mk_info(
        jokers=[],
        dollars=40,
        reroll_cost=5,
        shop=[joker_card(mk_joker("j_weak"), cost=4)],
    )
    op = evaluate_shop_opportunity(info)
    assert op.reroll_desirable
    assert op.can_reroll_above_interest_cap  # 40 - 5 = 35 >= 25


def test_critical_upgrade_flagged_when_no_engine():
    info = mk_info(
        jokers=[],
        dollars=10,
        shop=[joker_card(mk_joker("j_joker", mult=4.0), cost=4)],
    )
    op = evaluate_shop_opportunity(info)
    assert op.has_critical_upgrade
    assert op.has_affordable_upgrade


def test_no_critical_upgrade_once_engine_exists():
    info = mk_info(
        jokers=[mk_joker("j_existing", mult=8.0, x_mult=2.0)],
        dollars=10,
        shop=[joker_card(mk_joker("j_joker", mult=4.0), cost=4)],
    )
    op = evaluate_shop_opportunity(info)
    assert not op.has_critical_upgrade


def test_reroll_not_desirable_with_affordable_strong_buy():
    info = mk_info(
        jokers=[],
        dollars=20,
        reroll_cost=5,
        shop=[joker_card(mk_joker("j_cavendish", mult=4.0, x_mult=3.0), cost=6)],
    )
    op = evaluate_shop_opportunity(info)
    assert op.has_affordable_upgrade
    assert not op.reroll_desirable


# ───────────────────────── consumable scoring ─────────────────────────


def test_planet_aligned_to_main_hand_scores_higher():
    counts = {"Flush": 10}
    info = mk_info(jokers=[mk_joker("j_base", mult=10.0)], hand_play_counts=counts)
    b = evaluate_build(info)
    aligned = consumable_card("c_jupiter", "Planet", hand_type="Flush")
    off = consumable_card("c_mercury", "Planet", hand_type="Pair")
    assert score_shop_item(info, aligned, b) > score_shop_item(info, off, b)
