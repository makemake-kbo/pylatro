from __future__ import annotations

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest

from pylatro_agent.hand_plan import PLAN_HAND_TYPES, estimate_hand_plans
from pylatro_agent.reward import (
    RewardConfig,
    default_reward_components,
    state_potential_breakdown,
)
from pylatro_agent.risk import best_confident_joker_rescue, estimate_clear_risk

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
SUITS = ("Spades", "Hearts", "Clubs", "Diamonds")


def _card(rank: str, suit: str, *, seal: str = "", enhancement: str = "") -> dict[str, Any]:
    return {
        "rank": rank,
        "suit": suit,
        "seal": seal,
        "enhancement": enhancement,
        "edition": "",
    }


def _snapshot(cards: list[dict[str, Any]], *, blind_target: float = 400.0) -> dict[str, Any]:
    ranks = Counter(str(card["rank"]) for card in cards)
    suits = Counter(str(card["suit"]) for card in cards)
    exact = Counter((str(card["rank"]), str(card["suit"])) for card in cards)
    seals = Counter(str(card["seal"]) for card in cards if card.get("seal"))
    hand_details = {
        "High Card": {"chips": 5, "mult": 1, "level": 1, "played": 2},
        "Pair": {"chips": 10, "mult": 2, "level": 1, "played": 8},
        "Two Pair": {"chips": 20, "mult": 2, "level": 1, "played": 5},
        "Three of a Kind": {"chips": 30, "mult": 3, "level": 1, "played": 0},
        "Flush": {"chips": 35, "mult": 4, "level": 1, "played": 0},
        "Full House": {"chips": 40, "mult": 4, "level": 1, "played": 6},
        "Four of a Kind": {"chips": 60, "mult": 7, "level": 1, "played": 0},
        "Five of a Kind": {"chips": 120, "mult": 12, "level": 1, "played": 0},
        "Flush Five": {"chips": 160, "mult": 16, "level": 1, "played": 0},
    }
    return {
        "ante": 3,
        "dollars": 12,
        "blind_target": blind_target,
        "hands_available": 4,
        "discards_available": 3,
        "round_discard_capacity": 3,
        "hand_size": 8,
        "joker_limit": 5,
        "consumable_limit": 2,
        "joker_details": (),
        "consumable_details": (),
        "hand_details": hand_details,
        "deck_stats": {
            "size": len(cards),
            "cards": tuple(cards),
            "rank_counts": dict(ranks),
            "suit_counts": dict(suits),
            "rank_suit_counts": dict(exact),
            "seal_counts": dict(seals),
            "gold_count": sum(card.get("enhancement") == "Gold Card" for card in cards),
        },
        "shop_cards": (),
        "pack_card_details": (),
    }


def _joker(
    key: str,
    *,
    mult: float = 0.0,
    x_mult: float = 1.0,
    hand_type: str = "",
    eternal: bool = False,
    debuffed: bool = False,
) -> dict[str, Any]:
    return {
        "key": key,
        "name": key,
        "effect": "",
        "type": hand_type,
        "config": {},
        "mult": mult,
        "base_mult": mult,
        "x_mult": x_mult,
        "base_x_mult": x_mult,
        "t_mult": 0.0,
        "base_t_mult": 0.0,
        "t_chips": 0.0,
        "base_t_chips": 0.0,
        "edition": {},
        "eternal": eternal,
        "perishable": False,
        "perish_tally": None,
        "debuffed": debuffed,
        "copy_compatible": True,
    }


def _standard_deck() -> list[dict[str, Any]]:
    return [_card(rank, suit) for suit in SUITS for rank in RANKS]


def test_normal_deck_only_enables_scalable_fallback_plans() -> None:
    plans = estimate_hand_plans(_snapshot(_standard_deck()))

    assert plans is not None
    assert tuple(plan.hand_type for plan in plans.plans) == PLAN_HAND_TYPES
    assert plans.for_hand("Pair").draw_reliability > 0.8
    assert plans.for_hand("High Card").draw_reliability == 1.0
    assert plans.for_hand("Flush").deck_fix_gate == 0.0
    assert plans.for_hand("Three of a Kind").deck_fix_gate == 0.0
    assert plans.for_hand("Four of a Kind").deck_fix_gate == 0.0
    assert plans.for_hand("Five of a Kind").deck_fix_gate == 0.0
    assert plans.for_hand("Flush Five").deck_fix_gate == 0.0
    assert plans.for_hand("Two Pair") is None
    assert plans.for_hand("Full House") is None


def test_two_pair_is_low_priority_only_with_dedicated_joker_synergy() -> None:
    base = _snapshot(_standard_deck())
    trousers = deepcopy(base)
    trousers["joker_details"] = (_joker("j_trousers", mult=8.0),)
    typed = deepcopy(base)
    typed["joker_details"] = (_joker("j_mad", hand_type="Two Pair", mult=10.0),)
    square = deepcopy(base)
    square["joker_details"] = (_joker("j_square", mult=10.0),)
    debuffed = deepcopy(base)
    debuffed["joker_details"] = (_joker("j_trousers", mult=8.0, debuffed=True),)

    trousers_plans = estimate_hand_plans(trousers)
    typed_plans = estimate_hand_plans(typed)
    square_plans = estimate_hand_plans(square)
    debuffed_plans = estimate_hand_plans(debuffed)
    assert trousers_plans is not None and typed_plans is not None
    assert square_plans is not None and debuffed_plans is not None
    assert trousers_plans.for_hand("Two Pair").draw_reliability > 0.5
    assert typed_plans.for_hand("Two Pair").draw_reliability > 0.4
    assert trousers_plans.for_hand("Two Pair").utility < trousers_plans.for_hand("Pair").utility
    assert square_plans.for_hand("Two Pair") is None
    assert debuffed_plans.for_hand("Two Pair") is None


def test_rank_fixed_deck_unlocks_kind_plan() -> None:
    cards = _standard_deck() + [_card("A", "Spades") for _ in range(5)]
    plans = estimate_hand_plans(_snapshot(cards))

    assert plans is not None
    assert plans.for_hand("Three of a Kind").draw_reliability > 0.9
    assert plans.for_hand("Four of a Kind").draw_reliability > 0.9
    assert plans.for_hand("Five of a Kind").draw_reliability > 0.8
    assert plans.best.hand_type in {"Four of a Kind", "Five of a Kind", "Flush Five"}


def test_suit_fixed_deck_unlocks_flush_but_not_kind_hands() -> None:
    cards = _standard_deck()
    for card in cards:
        if card["suit"] == "Diamonds" and card["rank"] in RANKS[:8]:
            card["suit"] = "Hearts"
    plans = estimate_hand_plans(_snapshot(cards))

    assert plans is not None
    assert plans.for_hand("Flush").deck_fix_gate > 0.5
    assert plans.for_hand("Flush").draw_reliability > 0.5
    assert plans.for_hand("Three of a Kind").deck_fix_gate == 0.0


def test_blue_and_purple_seals_are_large_persistent_potential() -> None:
    base_info = _snapshot(_standard_deck())
    sealed_info = deepcopy(base_info)
    sealed_info["deck_stats"]["cards"] = tuple(
        [
            {**sealed_info["deck_stats"]["cards"][0], "seal": "Blue"},
            {**sealed_info["deck_stats"]["cards"][1], "seal": "Purple"},
            *sealed_info["deck_stats"]["cards"][2:],
        ]
    )
    sealed_info["deck_stats"]["seal_counts"] = {"Blue": 1, "Purple": 1}
    config = RewardConfig(enable_score_build_potential=True)

    base = state_potential_breakdown(base_info, config)
    sealed = state_potential_breakdown(sealed_info, config)

    assert base["seal_value"] == 0.0
    assert sealed["seal_value"] > config.potential_w_economy
    assert sealed["total"] > base["total"]


def test_purple_seal_potential_is_stable_after_using_a_discard_or_filling_inventory() -> None:
    before = _snapshot(_standard_deck())
    before["deck_stats"]["seal_counts"] = {"Purple": 1}
    after = deepcopy(before)
    after["discards_available"] = 2
    after["consumable_details"] = (
        {"key": "c_death", "set": "Tarot"},
        {"key": "c_hermit", "set": "Tarot"},
    )
    config = RewardConfig(enable_score_build_potential=True)

    before_seal = state_potential_breakdown(before, config)["seal_value"]
    after_seal = state_potential_breakdown(after, config)["seal_value"]

    assert after_seal == pytest.approx(before_seal)


def test_generated_planet_only_has_value_for_a_reliable_plan() -> None:
    pair_planet = _snapshot(_standard_deck(), blind_target=100.0)
    pair_planet["consumable_details"] = (
        {"key": "c_mercury", "set": "Planet", "hand_type": "Pair"},
    )
    two_pair_planet = deepcopy(pair_planet)
    two_pair_planet["consumable_details"] = (
        {"key": "c_uranus", "set": "Planet", "hand_type": "Two Pair"},
    )
    config = RewardConfig(enable_score_build_potential=True)

    assert state_potential_breakdown(pair_planet, config)["planet_option_value"] > 0.0
    assert state_potential_breakdown(two_pair_planet, config)["planet_option_value"] == 0.0

    synergized = deepcopy(two_pair_planet)
    synergized["joker_details"] = (_joker("j_trousers", mult=8.0),)
    assert state_potential_breakdown(synergized, config)["planet_option_value"] > 0.0


def test_generated_tarot_creates_option_value_until_converted() -> None:
    base = _snapshot(_standard_deck())
    tarot = deepcopy(base)
    tarot["consumable_details"] = ({"key": "c_death", "set": "Tarot"},)
    config = RewardConfig(enable_score_build_potential=True)

    assert state_potential_breakdown(base, config)["tarot_option_value"] == 0.0
    assert state_potential_breakdown(tarot, config)["tarot_option_value"] > 0.0


def test_hermit_and_temperance_receive_attributable_cash_reward() -> None:
    prev = _snapshot(_standard_deck())
    prev["dollars"] = 10
    config = RewardConfig(enable_score_build_potential=True)
    state = SimpleNamespace(win_ante=4, round_resets=SimpleNamespace(ante=3))

    hermit = {
        **prev,
        "dollars": 20,
        "consumable_use_key": "c_hermit",
        "strategic_attributable_cash_payout": 10,
        "progress_made": True,
    }
    temperance = {
        **prev,
        "dollars": 18,
        "pack_claim_set": "Tarot",
        "pack_claim_key": "c_temperance",
        "strategic_attributable_cash_payout": 8,
        "progress_made": True,
    }
    no_payout = {
        **prev,
        "consumable_use_key": "c_temperance",
        "progress_made": True,
    }

    hermit_reward = default_reward_components(state, prev, hermit, False, False, config)
    temperance_reward = default_reward_components(state, prev, temperance, False, False, config)
    no_payout_reward = default_reward_components(state, prev, no_payout, False, False, config)
    assert hermit_reward["strategic_cash_payout"] == 0.30
    assert temperance_reward["strategic_cash_payout"] == 0.24
    assert no_payout_reward["strategic_cash_payout"] == 0.0


def test_deck_fixing_and_gold_cards_raise_persistent_value() -> None:
    base = _snapshot(_standard_deck(), blind_target=100.0)
    rank_fixed = _snapshot(
        _standard_deck() + [_card("A", "Spades") for _ in range(5)],
        blind_target=100.0,
    )
    gold = deepcopy(base)
    gold["deck_stats"]["gold_count"] = 3
    config = RewardConfig(enable_score_build_potential=True)

    base_value = state_potential_breakdown(base, config)
    fixed_value = state_potential_breakdown(rank_fixed, config)
    gold_value = state_potential_breakdown(gold, config)
    assert fixed_value["realized_build_quality"] > base_value["realized_build_quality"]
    assert gold_value["economy"] > base_value["economy"]


def test_planet_reward_rejects_two_pair_and_unfixed_flush() -> None:
    prev = _snapshot(_standard_deck(), blind_target=100.0)
    pair_use = {**prev, "planet_use_observed": True, "planet_use_hand_type": "Pair", "progress_made": True}
    two_pair_use = {
        **prev,
        "planet_use_observed": True,
        "planet_use_hand_type": "Two Pair",
        "progress_made": True,
    }
    flush_use = {**prev, "planet_use_observed": True, "planet_use_hand_type": "Flush", "progress_made": True}
    config = RewardConfig(enable_planet_match_rewards=True)
    state = SimpleNamespace(win_ante=4, round_resets=SimpleNamespace(ante=3))

    pair = default_reward_components(state, prev, pair_use, False, False, config)
    two_pair = default_reward_components(state, prev, two_pair_use, False, False, config)
    flush = default_reward_components(state, prev, flush_use, False, False, config)
    assert pair["planet_match_bonus"] > 0.0
    assert two_pair["planet_match_bonus"] == 0.0
    assert flush["planet_match_bonus"] == 0.0


def test_two_pair_planet_gets_smaller_bonus_with_dedicated_synergy() -> None:
    prev = _snapshot(_standard_deck(), blind_target=100.0)
    prev["joker_details"] = (_joker("j_trousers", mult=8.0),)
    pair_use = {**prev, "planet_use_observed": True, "planet_use_hand_type": "Pair", "progress_made": True}
    two_pair_use = {
        **prev,
        "planet_use_observed": True,
        "planet_use_hand_type": "Two Pair",
        "progress_made": True,
    }
    config = RewardConfig(enable_planet_match_rewards=True)
    state = SimpleNamespace(win_ante=4, round_resets=SimpleNamespace(ante=3))

    pair = default_reward_components(state, prev, pair_use, False, False, config)
    two_pair = default_reward_components(state, prev, two_pair_use, False, False, config)
    assert 0.0 < two_pair["planet_match_bonus"] < pair["planet_match_bonus"]


def test_planet_reward_unlocks_flush_after_suit_fixing() -> None:
    cards = _standard_deck()
    for card in cards:
        if card["suit"] == "Diamonds" and card["rank"] in RANKS[:8]:
            card["suit"] = "Hearts"
    prev = _snapshot(cards, blind_target=100.0)
    curr = {**prev, "planet_use_observed": True, "planet_use_hand_type": "Flush", "progress_made": True}
    config = RewardConfig(enable_planet_match_rewards=True)
    state = SimpleNamespace(win_ante=4, round_resets=SimpleNamespace(ante=3))

    components = default_reward_components(state, prev, curr, False, False, config)
    assert components["planet_match_bonus"] > 0.0


def test_standard_pack_search_requires_midgame_scoring_readiness() -> None:
    ready = _snapshot(_standard_deck(), blind_target=100.0)
    ready["ante"] = 3
    ready["shop_cards"] = ({"key": "p_standard_normal_1", "set": "Booster", "cost": 4},)
    behind = deepcopy(ready)
    behind["blind_target"] = 100_000.0
    early = deepcopy(ready)
    early["ante"] = 1
    config = RewardConfig(enable_score_build_potential=True, potential_win_ante=4)

    assert state_potential_breakdown(ready, config)["standard_pack_search_option"] > 0.0
    assert state_potential_breakdown(behind, config)["standard_pack_search_option"] == 0.0
    assert state_potential_breakdown(early, config)["standard_pack_search_option"] == 0.0


def test_joker_search_only_values_score_improving_offers() -> None:
    base = _snapshot(_standard_deck(), blind_target=800.0)
    good = deepcopy(base)
    good["shop_cards"] = (
        {"key": "j_strong", "set": "Joker", "cost": 5, "joker": _joker("j_strong", x_mult=4.0)},
    )
    bad = deepcopy(base)
    bad["shop_cards"] = (
        {"key": "j_blank", "set": "Joker", "cost": 5, "joker": _joker("j_blank")},
    )
    good_rescue = best_confident_joker_rescue(good)
    bad_rescue = best_confident_joker_rescue(bad)

    assert good_rescue is not None and good_rescue.clear_probability_delta > 0.0
    assert bad_rescue is None or bad_rescue.clear_probability_delta == 0.0
    # Visible offers stay diagnostic-only so leaving or rerolling cannot create
    # an implicit negative potential reward.
    config = RewardConfig(enable_score_build_potential=True)
    assert state_potential_breakdown(good, config)["joker_search_option"] == 0.0


def test_full_joker_slots_value_replacing_a_weak_joker() -> None:
    info = _snapshot(_standard_deck(), blind_target=800.0)
    info["joker_limit"] = 1
    info["joker_details"] = (_joker("j_weak", mult=1.0),)
    info["shop_cards"] = (
        {"key": "j_strong", "set": "Joker", "cost": 5, "joker": _joker("j_strong", x_mult=4.0)},
    )
    rescue = best_confident_joker_rescue(info)

    assert rescue is not None
    assert rescue.clear_probability_delta > 0.0
    assert rescue.removed_key == "j_weak"


def test_buying_score_improving_joker_beats_bad_purchase() -> None:
    before = _snapshot(_standard_deck(), blind_target=800.0)
    good_joker = _joker("j_strong", x_mult=4.0)
    bad_joker = _joker("j_blank")
    before["shop_cards"] = (
        {"key": "j_strong", "set": "Joker", "cost": 5, "joker": good_joker},
    )
    good_after = deepcopy(before)
    good_after.update(
        {
            "dollars": 7,
            "shop_cards": (),
            "joker_details": (good_joker,),
            "shop_bought_joker_id": "j_strong",
            "progress_made": True,
            "action_type": "shop_buy",
        }
    )
    bad_after = deepcopy(before)
    bad_after.update(
        {
            "dollars": 7,
            "shop_cards": (),
            "joker_details": (bad_joker,),
            "shop_bought_joker_id": "j_blank",
            "progress_made": True,
            "action_type": "shop_buy",
        }
    )
    config = RewardConfig(enable_score_build_potential=True)
    state = SimpleNamespace(win_ante=4, round_resets=SimpleNamespace(ante=3))

    good = default_reward_components(state, before, good_after, False, False, config)
    bad = default_reward_components(state, before, bad_after, False, False, config)
    assert good["joker_upgrade_bonus"] > 0.0
    assert bad["joker_upgrade_bonus"] == 0.0
    assert good["total"] > bad["total"]


def test_two_step_joker_replacement_uses_the_pre_sale_roster_baseline() -> None:
    config = RewardConfig(enable_score_build_potential=True)
    state = SimpleNamespace(win_ante=4, round_resets=SimpleNamespace(ante=3))
    empty = _snapshot(_standard_deck(), blind_target=400.0)

    old = deepcopy(empty)
    old["joker_details"] = (_joker("j_old", x_mult=2.0),)
    upgrade = deepcopy(empty)
    upgrade.update(
        {
            "joker_details": (_joker("j_upgrade", x_mult=2.5),),
            "shop_bought_joker_id": "j_upgrade",
            "joker_upgrade_baseline_clear_probability": estimate_clear_risk(old).clear_probability,
            "progress_made": True,
            "action_type": "shop_buy",
        }
    )

    strong_old = deepcopy(empty)
    strong_old["joker_details"] = (_joker("j_strong_old", x_mult=2.5),)
    downgrade = deepcopy(empty)
    downgrade.update(
        {
            "joker_details": (_joker("j_downgrade", x_mult=2.0),),
            "shop_bought_joker_id": "j_downgrade",
            "joker_upgrade_baseline_clear_probability": estimate_clear_risk(strong_old).clear_probability,
            "progress_made": True,
            "action_type": "shop_buy",
        }
    )

    upgrade_reward = default_reward_components(state, empty, upgrade, False, False, config)
    downgrade_reward = default_reward_components(state, empty, downgrade, False, False, config)
    assert upgrade_reward["joker_upgrade_bonus"] > 0.0
    assert downgrade_reward["joker_upgrade_bonus"] == 0.0


def test_potential_remains_bounded_with_multiple_strategic_assets() -> None:
    info = _snapshot(_standard_deck(), blind_target=100.0)
    info["deck_stats"]["seal_counts"] = {"Blue": 3, "Purple": 3, "Gold": 3, "Red": 3}
    info["consumable_details"] = (
        {"key": "c_death", "set": "Tarot"},
        {"key": "c_hermit", "set": "Tarot"},
    )
    info["shop_cards"] = ({"key": "p_standard_mega_1", "set": "Booster", "cost": 4},)
    config = RewardConfig(enable_score_build_potential=True, potential_win_ante=4)
    breakdown = state_potential_breakdown(info, config)
    strategic_sum = sum(
        breakdown[name]
        for name in (
            "realized_build_quality",
            "scaling_option_value",
            "readiness",
            "economy",
            "tarot_option_value",
            "planet_option_value",
            "seal_value",
            "joker_search_option",
            "standard_pack_search_option",
        )
    )

    assert strategic_sum <= config.potential_build_cap + 1e-12
