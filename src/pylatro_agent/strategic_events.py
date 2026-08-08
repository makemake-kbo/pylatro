"""Attributable strategic events and contextual Tarot deck-fixing quality."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import comb
from typing import Any

from .action import ActionType
from .hand_plan import HandPlan, estimate_hand_plans
from .strategy_context import SUIT_TARGET_ORDER, suit_target_utilities

_RANK_PLANS = {"Pair": 2, "Three of a Kind": 3, "Four of a Kind": 4, "Five of a Kind": 5}
_SUIT_TAROTS = {
    "c_star": "Diamonds",
    "c_moon": "Clubs",
    "c_sun": "Hearts",
    "c_world": "Spades",
}
_TAROT_ACTIONS = {
    ActionType.USE_CONSUMABLE_NO_TARGET,
    ActionType.USE_CONSUMABLE_HAND_SUBSET,
    ActionType.USE_CONSUMABLE_JOKER,
}
_CASH_TAROTS = {"c_hermit", "c_temperance"}
_TAROT_FAMILIES = {
    "c_hermit": "cash",
    "c_temperance": "cash",
    "c_devil": "gold",
    "c_hanged_man": "deck_cut",
    "c_death": "rank_fix",
    "c_strength": "rank_fix",
    "c_star": "suit_fix",
    "c_moon": "suit_fix",
    "c_sun": "suit_fix",
    "c_world": "suit_fix",
    "c_fool": "creation",
    "c_emperor": "creation",
    "c_high_priestess": "creation",
    "c_judgement": "joker",
    "c_wheel_of_fortune": "joker",
}


@dataclass(frozen=True)
class StrategicEvent:
    tarot_source: str = ""
    tarot_uses: int = 0
    tarot_key: str = ""
    tarot_family: str = ""
    pack_auto_use: bool = False
    planet_uses: int = 0
    planet_pack_auto_uses: int = 0
    tarot_pack_auto_uses: int = 0
    planet_acquired: int = 0
    tarot_acquired: int = 0
    planet_sold: int = 0
    tarot_sold: int = 0
    planet_overwritten: int = 0
    tarot_overwritten: int = 0
    purple_tarots_generated: int = 0
    blue_planets_generated: int = 0
    gold_created_tarot: int = 0
    gold_created_midas: int = 0
    held_gold_count: int = 0
    held_gold_payout: int = 0
    attributable_cash_payout: int = 0
    blue_seals_claimed: int = 0
    purple_seals_claimed: int = 0
    blue_seals_activated: int = 0
    purple_seals_activated: int = 0
    tarot_fix_pre_quality: float = 0.0
    tarot_fix_post_quality: float = 0.0
    tarot_fix_pre_reliability: float = 0.0
    tarot_fix_post_reliability: float = 0.0
    tarot_fix_reliability_delta: float = 0.0
    tarot_fix_reward: float = 0.0

    def as_info(self) -> dict[str, Any]:
        return {f"strategic_{key}": value for key, value in asdict(self).items()}


def _cards(info: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    deck = info.get("deck_stats")
    deck = deck if isinstance(deck, Mapping) else {}
    return tuple(card for card in (deck.get("cards") or ()) if isinstance(card, Mapping))


def _card_map(info: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(card.get("identity", 0) or 0): card
        for card in _cards(info)
        if int(card.get("identity", 0) or 0)
    }


def _consumable_set_counts(info: Mapping[str, Any]) -> dict[str, int]:
    counts = {"Planet": 0, "Tarot": 0}
    for detail in info.get("consumable_details") or ():
        if isinstance(detail, Mapping) and detail.get("set") in counts:
            counts[str(detail["set"])] += 1
    return counts


def tarot_effect_family(tarot_key: str) -> str:
    """Return one bounded Tarot telemetry family."""

    return _TAROT_FAMILIES.get(str(tarot_key or ""), "enhancement") if tarot_key else ""


def _is_protected(card: Mapping[str, Any]) -> bool:
    return bool(
        str(card.get("seal") or "")
        or card.get("edition")
        or str(card.get("enhancement") or "")
        or str(card.get("center_key") or "c_base") != "c_base"
        or float(card.get("perma_bonus", 0) or 0) > 0.0
        or int(card.get("times_played", 0) or 0) > 0
    )


def _protected_value(cards: tuple[Mapping[str, Any], ...]) -> float:
    value = 0.0
    for card in cards:
        seal = str(card.get("seal") or "")
        value += 2.0 if seal else 0.0
        value += 2.0 if card.get("edition") else 0.0
        value += 1.0 if str(card.get("enhancement") or "") else 0.0
        value += min(float(card.get("perma_bonus", 0) or 0) / 10.0, 1.0)
        value += min(int(card.get("times_played", 0) or 0) / 3.0, 1.0)
    return value


def _protected_identities_preserved(
    pre_map: Mapping[int, Mapping[str, Any]],
    post_map: Mapping[int, Mapping[str, Any]],
) -> bool:
    """Require every protected physical asset to survive on that identity."""

    for identity, before in pre_map.items():
        after = post_map.get(identity)
        if after is None:
            if _is_protected(before):
                return False
            continue
        before_seal = str(before.get("seal") or "")
        if before_seal and str(after.get("seal") or "") != before_seal:
            return False
        before_edition = before.get("edition")
        if before_edition and after.get("edition") != before_edition:
            return False
        before_center = str(before.get("center_key") or "c_base")
        before_enhancement = str(before.get("enhancement") or "")
        if (
            (before_center != "c_base" or before_enhancement)
            and (
                str(after.get("center_key") or "c_base") != before_center
                or str(after.get("enhancement") or "") != before_enhancement
            )
        ):
            return False
        if float(after.get("perma_bonus", 0) or 0) < float(before.get("perma_bonus", 0) or 0):
            return False
    return True


def _at_least(population: int, successes: int, draws: int, needed: int) -> float:
    population = max(int(population), 0)
    successes = max(0, min(int(successes), population))
    draws = max(0, min(int(draws), population))
    if needed <= 0:
        return 1.0
    if population <= 0 or successes < needed or draws < needed:
        return 0.0
    denominator = comb(population, draws)
    if denominator <= 0:
        return 0.0
    low = max(needed, draws - (population - successes))
    high = min(successes, draws)
    numerator = sum(
        comb(successes, hits) * comb(population - successes, draws - hits)
        for hits in range(low, high + 1)
    )
    return max(0.0, min(numerator / denominator, 1.0))


def _draw_budget(info: Mapping[str, Any], deck_size: int) -> int:
    hand_size = max(int(info.get("hand_size", 8) or 8), 1)
    discards = max(int(info.get("discards_available", 0) or 0), 0)
    hands = max(int(info.get("hands_available", 1) or 1), 1)
    return min(deck_size, hand_size + min(hand_size, 5) * (discards + max(hands - 1, 0)))


def _anchor_for_plan(
    plan: HandPlan,
    info: Mapping[str, Any],
    cards: tuple[Mapping[str, Any], ...],
) -> tuple[str, str]:
    def score(field: str, value: str) -> tuple[float, int, str]:
        family = [card for card in cards if str(card.get(field) or "") == value]
        played = sum(int(card.get("times_played", 0) or 0) for card in family)
        protected = sum(_is_protected(card) for card in family)
        return (len(family) + 0.75 * played + 0.25 * protected, len(family), value)

    if plan.hand_type == "Flush Five":
        suit_utilities = dict(zip(SUIT_TARGET_ORDER, suit_target_utilities(info), strict=True))
        pairs = {(str(card.get("rank") or ""), str(card.get("suit") or "")) for card in cards}
        return max(
            pairs,
            key=lambda pair: (
                suit_utilities.get(pair[1], 0.0),
                sum(
                    1.0 + 0.75 * int(card.get("times_played", 0) or 0)
                    for card in cards
                    if (str(card.get("rank") or ""), str(card.get("suit") or "")) == pair
                ),
                pair,
            ),
            default=("", ""),
        )
    if plan.hand_type == "Flush":
        suits = {str(card.get("suit") or "") for card in cards if card.get("suit")}
        suit_utilities = dict(zip(SUIT_TARGET_ORDER, suit_target_utilities(info), strict=True))
        preferred = max(
            suits,
            key=lambda suit: (suit_utilities.get(suit, 0.0), score("suit", suit)),
            default="",
        )
        return "", preferred if suit_utilities.get(preferred, 0.0) > 0.0 else ""
    if plan.hand_type in _RANK_PLANS:
        ranks = {str(card.get("rank") or "") for card in cards if card.get("rank")}
        return max(ranks, key=lambda rank: score("rank", rank), default=""), ""
    return "", ""


def _anchor_metrics(
    info: Mapping[str, Any],
    hand_type: str,
    rank: str,
    suit: str,
) -> tuple[float, float]:
    cards = _cards(info)
    size = len(cards)
    if size <= 0:
        return 0.0, 0.0
    draws = _draw_budget(info, size)
    if hand_type == "Flush Five":
        successes = sum(card.get("rank") == rank and card.get("suit") == suit for card in cards)
        completion = _at_least(size, successes, draws, 5)
        gate = max(0.0, min((successes - 1) / 5.0, 1.0))
    elif hand_type == "Flush":
        successes = sum(card.get("suit") == suit for card in cards)
        completion = _at_least(size, successes, draws, 5)
        gate = max(0.0, min((successes / size - 0.25) / 0.15, 1.0))
    elif hand_type in _RANK_PLANS:
        successes = sum(card.get("rank") == rank for card in cards)
        completion = _at_least(size, successes, draws, _RANK_PLANS[hand_type])
        if hand_type == "Pair":
            gate = 1.0
        elif hand_type == "Three of a Kind":
            gate = max(0.0, min((successes - 4) / 3.0, 1.0))
        else:
            gate = max(0.0, min((successes - 4) / 4.0, 1.0))
    else:
        return 0.0, 0.0
    reliability = gate * completion
    cleanliness = successes / size
    quality = 0.55 * completion + 0.30 * reliability + 0.15 * cleanliness
    return quality, reliability


def _contextual_tarot_fix_metrics(
    prev_info: Mapping[str, Any],
    curr_info: Mapping[str, Any],
    tarot_key: str,
) -> tuple[float, float, float, float, float]:
    """Score a real deck delta against one reliable pre-action plan anchor."""
    if not tarot_key or tarot_key in {"c_fool", "c_devil"}:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    pre_cards = _cards(prev_info)
    post_cards = _cards(curr_info)
    pre_map = _card_map(prev_info)
    post_map = _card_map(curr_info)
    if not pre_map or pre_map == post_map:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    plans = estimate_hand_plans(prev_info)
    if plans is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    anchor_plan = plans.best
    rank, suit = _anchor_for_plan(anchor_plan, prev_info, pre_cards)
    if rank or suit:
        pre_quality, pre_reliability = _anchor_metrics(
            prev_info,
            anchor_plan.hand_type,
            rank,
            suit,
        )
        post_quality, post_reliability = _anchor_metrics(
            curr_info,
            anchor_plan.hand_type,
            rank,
            suit,
        )
    else:
        pre_quality = post_quality = 0.0
        pre_reliability = post_reliability = 0.0

    # A known suit-debuff boss makes converting cards *away* from that suit a
    # measurable reliability improvement even before a Flush plan is mature.
    # This is directional and cannot be farmed by converting the cards back.
    target_suit = _SUIT_TAROTS.get(tarot_key)
    boss_suit = str(prev_info.get("boss_debuff_suit", "") or "")
    boss_delta = 0.0
    if target_suit and boss_suit and target_suit != boss_suit:
        changed = [
            (before, post_map.get(identity))
            for identity, before in pre_map.items()
            if post_map.get(identity) is not None
            and str(post_map[identity].get("suit") or "") != str(before.get("suit") or "")
        ]
        removed_from_boss = sum(
            str(before.get("suit") or "") == boss_suit
            and str(after.get("suit") or "") != boss_suit
            for before, after in changed
        )
        moved_into_boss = sum(
            str(before.get("suit") or "") != boss_suit
            and str(after.get("suit") or "") == boss_suit
            for before, after in changed
        )
        boss_delta = (removed_from_boss - moved_into_boss) / max(len(pre_cards), 1)
        if boss_delta > 0.0:
            post_quality += boss_delta
            post_reliability += boss_delta

    # High Card has no rank/suit anchor. Cutting never-played, unprotected
    # cards is still useful when it concentrates physical assets the build has
    # demonstrated it wants to draw.
    if tarot_key == "c_hanged_man" and not rank and not suit:
        def asset_count(cards: tuple[Mapping[str, Any], ...]) -> int:
            return sum(_is_protected(card) for card in cards)

        assets = asset_count(pre_cards)
        if assets > 0:
            pre_quality = assets / max(len(pre_cards), 1)
            post_quality = assets / max(len(post_cards), 1)
            pre_reliability = _at_least(len(pre_cards), assets, _draw_budget(prev_info, len(pre_cards)), 1)
            post_reliability = _at_least(len(post_cards), assets, _draw_budget(curr_info, len(post_cards)), 1)

    if not rank and not suit and boss_delta <= 0.0 and tarot_key != "c_hanged_man":
        return 0.0, pre_quality, post_quality, pre_reliability, post_reliability
    played = int((prev_info.get("hand_play_counts") or {}).get(anchor_plan.hand_type, 0) or 0)
    total_played = sum(int(value or 0) for value in (prev_info.get("hand_play_counts") or {}).values())
    share = played / max(total_played, 1)
    play_evidence = min(played / 3.0, 1.0) * min(share / 0.5, 1.0)
    plan_evidence = max(0.0, min(pre_reliability, 1.0))
    # Deck composition can establish a plan before three repetitions have been
    # played. Treat either source as evidence instead of multiplying them into
    # an early-run zero gate.
    evidence = max(play_evidence, plan_evidence, min(max(boss_delta * 8.0, 0.0), 1.0))
    delta = post_quality - pre_quality
    if abs(delta) <= 0.002:
        return 0.0, pre_quality, post_quality, pre_reliability, post_reliability

    removed = [card for identity, card in pre_map.items() if identity not in post_map]
    if tarot_key == "c_hanged_man":
        def on_family(card: Mapping[str, Any]) -> bool:
            return bool(
                (rank and card.get("rank") == rank)
                or (suit and card.get("suit") == suit)
            )

        eligible_cut = bool(removed) and all(
            not _is_protected(card)
            and int(card.get("times_played", 0) or 0) == 0
            and not on_family(card)
            for card in removed
        )
        if delta > 0.0 and not eligible_cut:
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability

    if tarot_key == "c_strength" and rank:
        pre_rank = sum(card.get("rank") == rank for card in pre_cards)
        post_rank = sum(card.get("rank") == rank for card in post_cards)
        if delta > 0.0 and post_rank <= pre_rank:
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability

    if target_suit:
        if delta > 0.0 and target_suit != suit and boss_delta <= 0.0:
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability
        if target_suit == boss_suit and delta > 0.0:
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability

    if delta > 0.0:
        if post_reliability < pre_reliability - 0.01:
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability
        if float(curr_info.get("clear_probability", 0.0) or 0.0) < float(
            prev_info.get("clear_probability", 0.0) or 0.0
        ) - 0.02:
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability
        if not _protected_identities_preserved(pre_map, post_map):
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability
        if _protected_value(post_cards) + 1e-9 < _protected_value(pre_cards):
            return 0.0, pre_quality, post_quality, pre_reliability, post_reliability
        return (
            min(4.0 * evidence * delta, 0.20),
            pre_quality,
            post_quality,
            pre_reliability,
            post_reliability,
        )
    return (
        -min(4.0 * evidence * (-delta), 0.20),
        pre_quality,
        post_quality,
        pre_reliability,
        post_reliability,
    )


def contextual_tarot_fix_reward(
    prev_info: Mapping[str, Any],
    curr_info: Mapping[str, Any],
    tarot_key: str,
) -> tuple[float, float, float]:
    """Return reward and pre/post quality while keeping the public API stable."""

    reward, pre_quality, post_quality, _pre_reliability, _post_reliability = (
        _contextual_tarot_fix_metrics(prev_info, curr_info, tarot_key)
    )
    return reward, pre_quality, post_quality


def derive_strategic_event(
    prev_info: Mapping[str, Any],
    curr_info: Mapping[str, Any],
    decoded,
    action_result: Any,
    rewarded_gold_identities: set[int],
) -> StrategicEvent:
    """Derive one post-transition event record from attributable engine data."""
    tarot_delta = max(
        int(curr_info.get("tarot_usage_total", 0) or 0)
        - int(prev_info.get("tarot_usage_total", 0) or 0),
        0,
    )
    planet_delta = max(
        int(curr_info.get("planet_usage_total", 0) or 0)
        - int(prev_info.get("planet_usage_total", 0) or 0),
        0,
    )
    tarot_key = ""
    tarot_source = ""
    pack_auto_use = False
    use_result = action_result
    if decoded.action_type in _TAROT_ACTIONS:
        details = tuple(prev_info.get("consumable_details") or ())
        if 0 <= decoded.index < len(details):
            detail = details[decoded.index]
            if isinstance(detail, Mapping) and detail.get("set") == "Tarot":
                tarot_key = str(detail.get("key") or "")
                tarot_source = "inventory"
    elif decoded.action_type == ActionType.PACK_CLAIM and getattr(action_result, "auto_used", False):
        pack_auto_use = True
        use_result = getattr(action_result, "use_result", None)
        center_key = str(getattr(action_result, "center_key", "") or "")
        pack_details = tuple(prev_info.get("pack_card_details") or ())
        detail = pack_details[decoded.index] if 0 <= decoded.index < len(pack_details) else {}
        if isinstance(detail, Mapping) and detail.get("set") == "Tarot":
            tarot_key = center_key
            tarot_source = "pack_auto_use"

    pack_auto_set = ""
    if pack_auto_use:
        pack_details = tuple(prev_info.get("pack_card_details") or ())
        detail = pack_details[decoded.index] if 0 <= decoded.index < len(pack_details) else {}
        if isinstance(detail, Mapping):
            pack_auto_set = str(detail.get("set") or "")

    cash_payout = (
        max(int(getattr(use_result, "dollars_delta", 0) or 0), 0)
        if tarot_key in _CASH_TAROTS
        else 0
    )
    held_gold_count = max(int(getattr(action_result, "held_gold_count", 0) or 0), 0)
    held_gold_payout = max(int(getattr(action_result, "held_gold_payout", 0) or 0), 0)

    before_consumables = _consumable_set_counts(prev_info)
    after_consumables = _consumable_set_counts(curr_info)
    previous_details = tuple(prev_info.get("consumable_details") or ())
    inventory_use_set = ""
    if decoded.action_type in _TAROT_ACTIONS and 0 <= decoded.index < len(previous_details):
        detail = previous_details[decoded.index]
        if isinstance(detail, Mapping):
            inventory_use_set = str(detail.get("set") or "")
    sold_set = ""
    if decoded.action_type == ActionType.SHOP_SELL_CONSUMABLE and 0 <= decoded.index < len(previous_details):
        detail = previous_details[decoded.index]
        if isinstance(detail, Mapping):
            sold_set = str(detail.get("set") or "")

    acquired: dict[str, int] = {}
    overwritten: dict[str, int] = {}
    for consumable_set in ("Planet", "Tarot"):
        explicit_use = int(inventory_use_set == consumable_set)
        explicit_sale = int(sold_set == consumable_set)
        auto_use = int(pack_auto_set == consumable_set)
        acquired[consumable_set] = max(
            after_consumables[consumable_set]
            - before_consumables[consumable_set]
            + explicit_use
            + explicit_sale,
            0,
        ) + auto_use
        overwritten[consumable_set] = max(
            before_consumables[consumable_set]
            - after_consumables[consumable_set]
            - explicit_use
            - explicit_sale,
            0,
        )

    pre_map = _card_map(prev_info)
    post_map = _card_map(curr_info)
    gold_created_tarot = 0
    gold_created_midas = 0
    midas_active = any(
        isinstance(joker, Mapping)
        and joker.get("key") == "j_midas_mask"
        and not joker.get("debuffed")
        for joker in (prev_info.get("joker_details") or ())
    )
    for identity, after in post_map.items():
        before = pre_map.get(identity)
        if before is None or identity in rewarded_gold_identities:
            continue
        if before.get("center_key") == "m_gold" or after.get("center_key") != "m_gold":
            continue
        if tarot_delta > 0 and tarot_source:
            gold_created_tarot += 1
            rewarded_gold_identities.add(identity)
        elif decoded.action_type == ActionType.PLAY_SUBSET and midas_active:
            gold_created_midas += 1
            rewarded_gold_identities.add(identity)

    blue_claimed = 0
    purple_claimed = 0
    if decoded.action_type == ActionType.PACK_CLAIM:
        details = tuple(prev_info.get("pack_card_details") or ())
        detail = details[decoded.index] if 0 <= decoded.index < len(details) else {}
        if isinstance(detail, Mapping) and detail.get("set") in {"Default", "Enhanced"}:
            seal = str(detail.get("seal") or "")
            blue_claimed = int(seal == "Blue")
            purple_claimed = int(seal == "Purple")

    fix_reward, pre_quality, post_quality, pre_reliability, post_reliability = (
        _contextual_tarot_fix_metrics(
            prev_info,
            curr_info,
            tarot_key if tarot_delta > 0 else "",
        )
    )
    return StrategicEvent(
        tarot_source=tarot_source,
        tarot_uses=tarot_delta,
        tarot_key=tarot_key,
        tarot_family=tarot_effect_family(tarot_key) if tarot_delta > 0 else "",
        pack_auto_use=pack_auto_use,
        planet_uses=planet_delta,
        planet_pack_auto_uses=int(pack_auto_use and planet_delta > 0),
        tarot_pack_auto_uses=int(pack_auto_use and tarot_delta > 0),
        planet_acquired=acquired["Planet"],
        tarot_acquired=acquired["Tarot"],
        planet_sold=int(sold_set == "Planet"),
        tarot_sold=int(sold_set == "Tarot"),
        planet_overwritten=overwritten["Planet"],
        tarot_overwritten=overwritten["Tarot"],
        purple_tarots_generated=len(getattr(action_result, "generated_consumables", ()) or ()),
        blue_planets_generated=len(getattr(action_result, "blue_planets_generated", ()) or ()),
        gold_created_tarot=gold_created_tarot,
        gold_created_midas=gold_created_midas,
        held_gold_count=held_gold_count,
        held_gold_payout=held_gold_payout,
        attributable_cash_payout=cash_payout,
        blue_seals_claimed=blue_claimed,
        purple_seals_claimed=purple_claimed,
        blue_seals_activated=max(int(getattr(action_result, "blue_seals_activated", 0) or 0), 0),
        purple_seals_activated=max(int(getattr(action_result, "purple_seals_activated", 0) or 0), 0),
        tarot_fix_pre_quality=pre_quality,
        tarot_fix_post_quality=post_quality,
        tarot_fix_pre_reliability=pre_reliability,
        tarot_fix_post_reliability=post_reliability,
        tarot_fix_reliability_delta=post_reliability - pre_reliability,
        tarot_fix_reward=fix_reward,
    )
