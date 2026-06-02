from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, default_reward_components


def _state(ante: int = 2, interest_cap: int = 25, win_ante: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        round_resets=SimpleNamespace(ante=ante),
        interest_cap=interest_cap,
        win_ante=win_ante,
    )


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


def joker_card(summary: dict[str, Any], cost: int = 5, index: int = 0) -> dict[str, Any]:
    return {"index": index, "key": summary["key"], "set": "Joker", "cost": cost, "joker": summary}


def base_info(
    *,
    jokers: list[dict[str, Any]] | None = None,
    shop: list[dict[str, Any]] | None = None,
    dollars: int = 12,
    ante: int = 2,
    interest_cap_cash: int = 25,
    joker_slots_limit: int = 5,
    reroll_cost: int = 5,
    in_shop: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    jokers = jokers or []
    info = {
        "ante": ante,
        "dollars": dollars,
        "round_score": 0,
        "blind_target": 300,
        "reroll_cost": reroll_cost,
        "in_shop": in_shop,
        "interest_cap_cash": interest_cap_cash,
        "interest_amount": 1,
        "joker_slots_used": len(jokers),
        "joker_slots_limit": joker_slots_limit,
        "joker_slots_left": max(0, joker_slots_limit - len(jokers)),
        "joker_details": tuple(jokers),
        "consumable_details": (),
        "hand_play_counts": {},
        "hand_levels": {},
        "deck_stats": {"size": 52, "suit_counts": {}, "enhancement_counts": {}, "seal_counts": {}},
        "shop_cards": tuple(shop or ()),
        "progress_made": True,
    }
    info.update(extra)
    return info


def comp(prev: dict, curr: dict, *, won: bool = False, terminated: bool = False) -> dict[str, float]:
    return default_reward_components(_state(), prev, curr, terminated, won, DEFAULT_REWARD_CONFIG)


# ───────────────────────── strategic activation ─────────────────────────


def test_strategic_inactive_without_build_info():
    # Minimal info (no joker_details) falls through to the legacy flat reroll.
    prev = {"ante": 1, "round_score": 0, "blind_target": 300}
    curr = {"round_score": 0, "blind_target": 300, "progress_made": True, "action_type": "shop_reroll"}
    c = comp(prev, curr)
    assert c["shop_reroll_good"] == 0.0 and c["shop_reroll_bad"] == 0.0
    assert c["shop_reroll_reward"] > 0.0  # legacy path


# ───────────────────────────── shop buy ─────────────────────────────


def test_buy_first_scoring_joker_rewarded():
    bought = mk_joker("j_joker", mult=4.0)
    prev = base_info(jokers=[], shop=[joker_card(bought, cost=4)], dollars=12)
    curr = base_info(jokers=[bought], shop=[], dollars=8, action_type="shop_buy", action_index=0)
    c = comp(prev, curr)
    assert c["shop_purchase_value"] > 0.0
    assert c["joker_slot_fill"] > 0.0


def test_buy_xmult_with_base_triggers_acquisition():
    base = mk_joker("j_base", mult=12.0)
    xj = mk_joker("j_cavendish", x_mult=3.0)
    prev = base_info(jokers=[base], shop=[joker_card(xj, cost=6)], dollars=12)
    curr = base_info(jokers=[base, xj], shop=[], dollars=6, action_type="shop_buy", action_index=0)
    c = comp(prev, curr)
    assert c["xmult_acquisition"] > 0.0


def test_buy_below_threshold_joker_penalized():
    weak = mk_joker("j_weak")  # no scoring stats, already have an engine
    engine = mk_joker("j_engine", mult=10.0, x_mult=2.0)
    prev = base_info(jokers=[engine], shop=[joker_card(weak, cost=6)], dollars=12)
    curr = base_info(jokers=[engine, weak], shop=[], dollars=6, action_type="shop_buy", action_index=0)
    c = comp(prev, curr)
    assert c["shop_bad_buy_penalty"] < 0.0


# ───────────────────────────── reroll ─────────────────────────────


def test_reroll_good_above_cap_no_useful_buy():
    weak = mk_joker("j_weak")
    prev = base_info(jokers=[], shop=[joker_card(weak, cost=4)], dollars=40, reroll_cost=5)
    curr = base_info(jokers=[], shop=[joker_card(weak, cost=4)], dollars=35, action_type="shop_reroll")
    c = comp(prev, curr)
    assert c["shop_reroll_good"] > 0.0
    assert c["shop_reroll_bad"] == 0.0


def test_reroll_away_from_strong_buy_penalized():
    strong = mk_joker("j_cavendish", mult=4.0, x_mult=3.0)
    prev = base_info(jokers=[], shop=[joker_card(strong, cost=6)], dollars=20, reroll_cost=5)
    curr = base_info(jokers=[], shop=[joker_card(strong, cost=6)], dollars=15, action_type="shop_reroll")
    c = comp(prev, curr)
    assert c["shop_reroll_bad"] < 0.0


# ───────────────────────────── leave ─────────────────────────────


def test_leave_with_affordable_critical_joker_penalized():
    joker = mk_joker("j_joker", mult=4.0)
    prev = base_info(jokers=[], shop=[joker_card(joker, cost=4)], dollars=12)
    curr = base_info(jokers=[], shop=[joker_card(joker, cost=4)], dollars=12, action_type="shop_leave")
    c = comp(prev, curr)
    assert c["shop_leave_missed_upgrade_penalty"] < 0.0


def test_leave_good_when_at_cap_and_no_useful_buy():
    engine = mk_joker("j_engine", mult=10.0, x_mult=2.0)
    weak = mk_joker("j_weak")
    prev = base_info(jokers=[engine], shop=[joker_card(weak, cost=4)], dollars=25, joker_slots_limit=1)
    curr = base_info(jokers=[engine], shop=[joker_card(weak, cost=4)], dollars=25, joker_slots_limit=1,
                     action_type="shop_leave")
    c = comp(prev, curr)
    assert c["shop_leave_good"] > 0.0
    assert c["shop_leave_missed_upgrade_penalty"] == 0.0


# ───────────────────────────── sell ─────────────────────────────


def test_sell_scaling_joker_penalized():
    scaling = mk_joker("j_ride_the_bus", is_scaling=True, mult=5.0)
    prev = base_info(jokers=[scaling], dollars=10)
    curr = base_info(jokers=[], dollars=13, action_type="shop_sell_joker", action_index=0)
    c = comp(prev, curr)
    assert c["joker_sell_bad"] < 0.0
    assert c["shop_sell_penalty"] == 0.0  # strategic path supersedes the flat penalty


def test_sell_weak_joker_to_fund_upgrade_ok():
    weak = mk_joker("j_weak")
    strong = mk_joker("j_cavendish", mult=4.0, x_mult=3.0)
    prev = base_info(jokers=[weak], shop=[joker_card(strong, cost=6)], dollars=12, joker_slots_limit=1)
    curr = base_info(jokers=[], shop=[joker_card(strong, cost=6)], dollars=15, joker_slots_limit=1,
                     action_type="shop_sell_joker", action_index=0)
    c = comp(prev, curr)
    assert c["joker_sell_good"] >= 0.0
    assert c["joker_sell_bad"] == 0.0


# ───────────────────────────── economy ─────────────────────────────


def test_interest_breakpoint_rewarded_on_reaching_cap():
    engine = mk_joker("j_engine", mult=10.0, x_mult=2.0)
    prev = base_info(jokers=[engine], dollars=20, action_type="shop_reroll", in_shop=True)
    # money grows past the $25 cap between steps (e.g. cash-out / sell)
    curr = base_info(jokers=[engine], dollars=25, action_type="shop_reroll", in_shop=True)
    c = comp(prev, curr)
    assert c["economy_interest_progress"] > 0.0
    assert c["economy_interest_breakpoint"] > 0.0


def test_overspend_below_cap_without_value_penalized():
    engine = mk_joker("j_engine", mult=10.0, x_mult=2.0)
    # Spend $10 (2 tiers) on nothing of build value while well below cap-blocking.
    prev = base_info(jokers=[engine], dollars=20, action_type="shop_reroll")
    curr = base_info(jokers=[engine], dollars=10, action_type="shop_reroll")
    c = comp(prev, curr)
    assert c["economy_overspend_penalty"] < 0.0


# ───────────────────────────── caps ─────────────────────────────


def test_strategic_contribution_is_capped():
    bought = mk_joker("j_joker", mult=40.0, x_mult=5.0)
    prev = base_info(jokers=[], shop=[joker_card(bought, cost=4)], dollars=40)
    curr = base_info(jokers=[bought], shop=[], dollars=36, action_type="shop_buy", action_index=0)
    c = comp(prev, curr)
    strategic_total = (
        c["shop_engine_delta"]
        + c["shop_purchase_value"]
        + c["joker_slot_fill"]
        + c["xmult_acquisition"]
        + c["economy_interest_progress"]
    )
    assert strategic_total <= DEFAULT_REWARD_CONFIG.max_single_shop_reward + 1e-6
