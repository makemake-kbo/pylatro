"""Bounded strategic state values used by potential-based reward shaping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log1p
from typing import Any

from .hand_plan import HandPlanEstimate, estimate_hand_plans


@dataclass(frozen=True)
class StrategyValue:
    plans: HandPlanEstimate
    hand_plan_quality: float
    readiness: float
    economy: float
    tarot_option: float
    planet_option: float
    seals: float
    joker_search: float
    pack_search: float


_TAROT_OPTION_UNITS = {
    # Direct value generation.
    "c_hermit": 1.00,
    "c_temperance": 1.00,
    "c_emperor": 0.85,
    "c_high_priestess": 0.85,
    "c_judgement": 0.75,
    # Persistent deck fixing.
    "c_hanged_man": 1.00,
    "c_death": 1.00,
    "c_star": 0.85,
    "c_moon": 0.85,
    "c_sun": 0.85,
    "c_world": 0.85,
    "c_magician": 0.75,
    "c_empress": 0.75,
    "c_heirophant": 0.72,
    "c_chariot": 0.72,
    "c_justice": 0.72,
    "c_devil": 0.72,
}


def _clip01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _hand_plans(
    info: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]] | None = None,
) -> HandPlanEstimate | None:
    if jokers is None:
        cached = info.get("_hand_plan_estimate") or info.get("hand_plan_estimate")
        if isinstance(cached, HandPlanEstimate):
            return cached
    return estimate_hand_plans(info, jokers)


def _plan_strength(plans: HandPlanEstimate) -> float:
    # Utility already combines explicit-hand score, reliability, and a small
    # strategic priority. Normalize the maximum possible priority to one.
    return _clip01(plans.best.utility / 1.32)


def _tarot_option(info: Mapping[str, Any]) -> float:
    units = 0.0
    for consumable in info.get("consumable_details") or ():
        if not isinstance(consumable, Mapping) or consumable.get("set") != "Tarot":
            continue
        units += _TAROT_OPTION_UNITS.get(str(consumable.get("key") or ""), 0.55)
    return _clip01(units / 2.0)


def _planet_option(info: Mapping[str, Any], plans: HandPlanEstimate) -> float:
    units = 0.0
    for consumable in info.get("consumable_details") or ():
        if not isinstance(consumable, Mapping) or consumable.get("set") != "Planet":
            continue
        plan = plans.for_hand(str(consumable.get("hand_type") or ""))
        if plan is None or plan.draw_reliability < 0.20:
            continue
        best_bonus = 0.25 if plan.hand_type == plans.best.hand_type else 0.0
        units += plan.draw_reliability * (0.50 + best_bonus + 0.25 * _clip01(plan.readiness_ratio))
    return _clip01(units / 2.0)


def _seal_value(info: Mapping[str, Any], plans: HandPlanEstimate) -> float:
    deck = _mapping(info.get("deck_stats"))
    counts = _mapping(deck.get("seal_counts"))
    blue = max(int(counts.get("Blue", 0) or 0), 0)
    purple = max(int(counts.get("Purple", 0) or 0), 0)
    gold = max(int(counts.get("Gold", 0) or 0), 0)
    red = max(int(counts.get("Red", 0) or 0), 0)

    best = plans.best
    winning_plan = best.draw_reliability * _clip01(best.readiness_ratio)
    discards = max(int(info.get("discards_available") or info.get("discards_left") or 0), 0)
    consumables = sum(1 for item in (info.get("consumable_details") or ()) if isinstance(item, Mapping))
    capacity = max(int(info.get("consumable_limit", 2) or 2), 0)
    room = max(capacity - consumables, 0)
    purple_throughput = _clip01(discards / 2.0) * (0.55 + 0.45 * _clip01(room))

    # Blue/Purple deliberately dwarf Red/Gold. They create a repeatable Planet
    # or Tarot engine and are the strongest non-joker deck assets in this model.
    units = 0.0
    units += min(blue, 2) * (0.62 + 0.38 * winning_plan)
    units += min(purple, 2) * (0.65 + 0.35 * purple_throughput)
    units += min(gold, 3) * 0.22
    units += min(red, 3) * 0.28
    return _clip01(units / 1.55)


def _economy_value(info: Mapping[str, Any]) -> float:
    dollars = max(float(info.get("dollars", 0) or 0), 0.0)
    deck = _mapping(info.get("deck_stats"))
    gold_cards = max(int(deck.get("gold_count", 0) or 0), 0)
    cash = log1p(min(dollars, 50.0)) / log1p(50.0)
    persistent_gold = _clip01(gold_cards / 3.0)
    return _clip01(0.72 * cash + 0.28 * persistent_gold)


def _joker_search_value(info: Mapping[str, Any], current: HandPlanEstimate) -> float:
    dollars = max(float(info.get("dollars", 0) or 0), 0.0)
    owned = tuple(joker for joker in (info.get("joker_details") or ()) if isinstance(joker, Mapping))
    limit = max(int(info.get("joker_limit", 5) or 5), 0)
    current_strength = max(_plan_strength(current), 1e-4)
    underpowered = _clip01((1.15 - current.best.readiness_ratio) / 0.65)
    if underpowered <= 0.0:
        return 0.0

    best_gain = 0.0
    for card in info.get("shop_cards") or ():
        if not isinstance(card, Mapping) or card.get("set") != "Joker":
            continue
        if float(card.get("cost", 0) or 0) > dollars:
            continue
        offer = card.get("joker")
        if not isinstance(offer, Mapping):
            continue
        rosters: list[tuple[Mapping[str, Any], ...]] = []
        if len(owned) < limit or bool(_mapping(offer.get("edition")).get("negative")):
            rosters.append((*owned, offer))
        else:
            for index, joker in enumerate(owned):
                if joker.get("eternal"):
                    continue
                rosters.append((*owned[:index], offer, *owned[index + 1 :]))
        for roster in rosters:
            candidate = _hand_plans(info, roster)
            if candidate is None:
                continue
            gain = (_plan_strength(candidate) - current_strength) / current_strength
            best_gain = max(best_gain, gain)
    return underpowered * _clip01(best_gain / 0.75)


def _standard_pack_search_value(
    info: Mapping[str, Any],
    plans: HandPlanEstimate,
    *,
    win_ante: int,
) -> float:
    ante = max(int(info.get("ante", 1) or 1), 1)
    # Search after the opening ante, while there is runway to exploit a seal,
    # and only when the current scoring plan is already credible.
    if ante < 2 or ante >= max(int(win_ante), 2):
        return 0.0
    scoring_ready = _clip01((plans.best.readiness_ratio - 0.80) / 0.35) * plans.best.draw_reliability
    if scoring_ready <= 0.0:
        return 0.0

    dollars = max(float(info.get("dollars", 0) or 0), 0.0)
    visible_standard = any(
        isinstance(card, Mapping)
        and "standard" in str(card.get("key") or "").lower()
        and float(card.get("cost", 0) or 0) <= dollars
        for card in (info.get("shop_cards") or ())
    )
    pack_name = " ".join(
        str(info.get(key) or "") for key in ("pack_state_name", "pack_booster_key")
    ).lower()
    open_standard = "standard" in pack_name
    if not visible_standard and not open_standard:
        return 0.0

    # Preserve expected search value while the pack is open, then strongly
    # privilege revealed Purple/Blue cards. Claiming one transfers this value
    # into the persistent seal component instead of creating a farmable bonus.
    value = 0.28
    if open_standard:
        revealed = tuple(card for card in (info.get("pack_card_details") or ()) if isinstance(card, Mapping))
        seals = {str(card.get("seal") or "") for card in revealed}
        if "Purple" in seals or "Blue" in seals:
            value = 1.0
        elif "Red" in seals or "Gold" in seals:
            value = 0.48
    return scoring_ready * value


def estimate_strategy_value(info: Mapping[str, Any], *, win_ante: int) -> StrategyValue | None:
    plans = _hand_plans(info)
    if plans is None:
        return None
    best = plans.best
    return StrategyValue(
        plans=plans,
        hand_plan_quality=_plan_strength(plans),
        readiness=best.draw_reliability * _clip01(best.readiness_ratio / 1.25),
        economy=_economy_value(info),
        tarot_option=_tarot_option(info),
        planet_option=_planet_option(info, plans),
        seals=_seal_value(info, plans),
        joker_search=_joker_search_value(info, plans),
        pack_search=_standard_pack_search_value(info, plans, win_ante=win_ante),
    )
