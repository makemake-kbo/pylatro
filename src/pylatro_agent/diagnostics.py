"""Action, build, and joker diagnostics for agent rollouts.

The action-quality fields are shared by BalatroEnv and fast_generate because
reward shaping reads them on both paths. Build, shop-event, Hologram, and exact
counterfactual diagnostics are emitted by BalatroEnv for PPO metrics only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import log
from typing import Any

from pylatro.flow import play_cards
from pylatro.instances import remove_joker

from .action import ActionType
from .constants import MAX_CONSUMABLE_SLOTS, ActionRange, SubPhase
from .hand_candidates import generate_hand_candidates
from .hand_plan import estimate_hand_plans
from .masks import compute_action_mask
from .risk import best_confident_joker_rescue
from .shop_eval import BuildEval, capture_build_features, evaluate_build
from .subset_actions import subset_index, subset_indices

MAX_DIAGNOSTIC_JOKERS = 12
MAX_DIAGNOSTIC_EVENTS = 12
MAX_DIAGNOSTIC_SHOP_JOKERS = 8

_POTENTIAL_COMPONENTS = (
    "blind_progress",
    "ante_progress",
    "realized_build_quality",
    "scaling_option_value",
    "readiness",
    "economy",
    "tarot_option_value",
    "planet_option_value",
    "seal_value",
    "joker_search_option",
    "standard_pack_search_option",
    "total",
)


def _center_key(card) -> str:
    return str(getattr(card, "center_key", "") or "")


def _center_set(state, center_key: str) -> str:
    center = state.data.centers.get(center_key, {})
    return str(center.get("set", "") or "")


def _planet_hand_type(state, center_key: str) -> str:
    center = state.data.centers.get(center_key, {})
    config = center.get("config", {}) or {}
    return str(config.get("hand_type", "") or "")


def _main_hand_proxy(state) -> str:
    # rank by (times played, hand level, chips*mult) descending; name breaks final ties
    rows: list[tuple[int, int, float, str]] = []
    for name, hand in state.hands.items():
        rows.append((
            int(hand.get("played", 0) or 0),
            int(hand.get("level", 1) or 1),
            float(hand.get("chips", 0) or 0) * float(hand.get("mult", 1) or 1),
            str(name),
        ))
    rows.sort(reverse=True)
    if not rows or rows[0][0] <= 0:
        return ""
    return rows[0][3]


def _planet_alignment_rank(
    state, center_key: str, main_hand: str, max_played: int
) -> tuple[int, float]:
    """How well a planet fits the run: (matches main hand, hand play share)."""
    hand_type = _planet_hand_type(state, center_key)
    played = (
        int(state.hands.get(hand_type, {}).get("played", 0) or 0) if hand_type else 0
    )
    share = (played / max_played) if max_played > 0 else 0.0
    return (1 if hand_type and hand_type == main_hand else 0, share)


def _is_best_planet_in_pack(state, chosen_index: int) -> bool:
    """True when no other planet in the open pack better fits the run.

    Claiming the best planet a matchless pack offers (even Pluto) is
    legitimate — the pack is already paid for and skipping wastes it — so
    the unmatched-claim penalty exempts these claims. Only planet cards
    compete in the ranking; ties keep the claim exempt.
    """
    plan_ranks: dict[str, tuple[int, float, float]] = {}
    try:
        plans = estimate_hand_plans(capture_build_features(state))
        if plans is not None:
            for plan in plans.plans:
                plan_ranks[plan.hand_type] = (
                    int(plan.draw_reliability >= 0.20),
                    float(plan.utility),
                    float(plan.draw_reliability),
                )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        plan_ranks = {}
    if plan_ranks:
        chosen_type = _planet_hand_type(state, _center_key(state.pack.cards[chosen_index]))
        chosen_rank = plan_ranks.get(chosen_type, (0, 0.0, 0.0))
        return not any(
            _center_set(state, _center_key(card)) == "Planet"
            and plan_ranks.get(_planet_hand_type(state, _center_key(card)), (0, 0.0, 0.0)) > chosen_rank
            for index, card in enumerate(state.pack.cards)
            if index != chosen_index
        )

    main_hand = _main_hand_proxy(state)
    max_played = max(
        (int(hand.get("played", 0) or 0) for hand in state.hands.values()),
        default=0,
    )
    chosen_rank = _planet_alignment_rank(
        state, _center_key(state.pack.cards[chosen_index]), main_hand, max_played
    )
    for index, card in enumerate(state.pack.cards):
        if index == chosen_index:
            continue
        other_key = _center_key(card)
        if _center_set(state, other_key) != "Planet":
            continue
        if _planet_alignment_rank(state, other_key, main_hand, max_played) > chosen_rank:
            return False
    return True


def _planet_diagnostics(state, center_key: str, *, prefix: str) -> dict[str, Any]:
    hand_type = _planet_hand_type(state, center_key)
    main_hand = _main_hand_proxy(state)
    hand_info = state.hands.get(hand_type, {}) if hand_type else {}
    played = int(hand_info.get("played", 0) or 0) if hand_info else 0
    max_played = max(
        (int(hand.get("played", 0) or 0) for hand in state.hands.values()),
        default=0,
    )
    diagnostics = {
        f"{prefix}_observed": True,
        f"{prefix}_key": center_key,
        f"{prefix}_hand_type": hand_type,
        f"{prefix}_played_hand": played > 0,
        # Play count of the planet's hand relative to the most-played hand this
        # run, 1.0 for the workhorse hand, near 0 for a hand played once.
        f"{prefix}_play_share": (played / max_played) if max_played > 0 else 0.0,
        f"{prefix}_main_hand": main_hand,
        f"{prefix}_main_hand_match": bool(hand_type and hand_type == main_hand),
    }
    try:
        plans = estimate_hand_plans(capture_build_features(state))
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        plans = None
    plan = plans.for_hand(hand_type) if plans is not None else None
    diagnostics[f"{prefix}_plan_supported"] = bool(
        plan is not None and plan.draw_reliability >= 0.20
    )
    diagnostics[f"{prefix}_active_plan_match"] = bool(
        plan is not None
        and plans is not None
        and plan.draw_reliability >= 0.20
        and hand_type == plans.best.hand_type
    )
    return diagnostics


def consumable_funnel_state_diagnostics(
    info: Mapping[str, Any],
    action_mask,
) -> dict[str, Any]:
    """Describe ownership and exact legal-use opportunities in one pre-state."""

    details = tuple(info.get("consumable_details") or ())
    choose_action = str(info.get("sub_phase", "")) == str(SubPhase.CHOOSE_ACTION)
    blocks = None
    if choose_action and action_mask is not None:
        blocks = action_mask[
            int(ActionRange.CONSUMABLE_FLAT_START) : int(ActionRange.CONSUMABLE_FLAT_END) + 1
        ].reshape(MAX_CONSUMABLE_SLOTS, -1)

    diagnostics: dict[str, Any] = {}
    legal_slots: dict[str, set[int]] = {"Planet": set(), "Tarot": set()}
    owned_slots: dict[str, set[int]] = {"Planet": set(), "Tarot": set()}
    for slot, detail in enumerate(details[:MAX_CONSUMABLE_SLOTS]):
        if not isinstance(detail, Mapping) or detail.get("set") not in owned_slots:
            continue
        consumable_set = str(detail["set"])
        owned_slots[consumable_set].add(slot)
        if blocks is not None and bool(blocks[slot].any()):
            legal_slots[consumable_set].add(slot)

    for consumable_set in ("Planet", "Tarot"):
        prefix = f"consumable_{consumable_set.lower()}"
        diagnostics[f"{prefix}_owned_count"] = len(owned_slots[consumable_set])
        diagnostics[f"{prefix}_owned_state"] = bool(owned_slots[consumable_set])
        diagnostics[f"{prefix}_legal_use_opportunity"] = bool(legal_slots[consumable_set])

        offer_details = ()
        action_start = None
        sub_phase = str(info.get("sub_phase", ""))
        if sub_phase == str(SubPhase.SHOP):
            offer_details = tuple(info.get("shop_cards") or ())
            action_start = int(ActionRange.SHOP_BUY_START)
        elif sub_phase == str(SubPhase.BOOSTER_PACK):
            offer_details = tuple(info.get("pack_card_details") or ())
            action_start = int(ActionRange.PACK_CLAIM_START)
        diagnostics[f"{prefix}_eligible_offer_opportunity"] = bool(
            action_start is not None
            and action_mask is not None
            and any(
                isinstance(detail, Mapping)
                and detail.get("set") == consumable_set
                and action_start + index < len(action_mask)
                and bool(action_mask[action_start + index])
                for index, detail in enumerate(offer_details)
            )
        )

    plans = None
    if owned_slots["Planet"]:
        try:
            plans = estimate_hand_plans(info)
        except (KeyError, OverflowError, TypeError, ValueError):
            plans = None
    active_slots: set[int] = set()
    if plans is not None:
        for slot in owned_slots["Planet"]:
            detail = details[slot]
            hand_type = str(detail.get("hand_type") or "")
            plan = plans.for_hand(hand_type)
            if (
                plan is not None
                and plan.draw_reliability >= 0.20
                and hand_type == plans.best.hand_type
            ):
                active_slots.add(slot)
    diagnostics["consumable_planet_active_plan_owned"] = bool(active_slots)
    diagnostics["consumable_planet_active_plan_legal"] = bool(active_slots & legal_slots["Planet"])
    return diagnostics


def _legal_play_candidates(state, play_candidates, action_mask=None):
    """Filter generated play candidates through the operative action grammar."""

    if action_mask is None:
        action_mask = compute_action_mask(state, SubPhase.CHOOSE_ACTION)
    play_mask = action_mask[
        int(ActionRange.PLAY_SUBSET_START) : int(ActionRange.PLAY_SUBSET_END) + 1
    ]
    legal = []
    for candidate in play_candidates:
        try:
            candidate_index = subset_index(candidate.indices)
        except KeyError:
            # Hand slots outside the bounded action grammar are not playable.
            continue
        if play_mask[candidate_index]:
            legal.append(candidate)
    return tuple(legal)


def action_diagnostics(
    state,
    decoded,
    *,
    action_mask=None,
    round_score: float = 0.0,
    blind_target: float = 0.0,
) -> dict[str, Any]:
    """Return policy-quality diagnostics for the pre-action state.

    Mirrors the env's _action_diagnostics, when this returns fields,
    default_reward_components reads them to award candidate / planet /
    pack-skip shaping bonuses. fast_generate calls this so its value-
    head BC targets see the same reward signal as PPO.
    """
    if state is None:
        return {}

    if decoded.action_type == ActionType.PLAY_SUBSET:
        diagnostics: dict[str, Any] = {"hand_play_observed": True}
        indices = tuple(subset_indices(decoded.index))
        selected = set(indices)
        diagnostics["blue_seal_held_count"] = sum(
            card.seal == "Blue" and not card.debuff
            for index, card in enumerate(state.hand_cards)
            if index not in selected
        )
        if any(index >= len(state.hand_cards) for index in indices):
            diagnostics["hand_play_not_in_candidates"] = True
            return diagnostics
        play_candidates, _discard_candidates = generate_hand_candidates(state)
        if not play_candidates:
            diagnostics["hand_play_not_in_candidates"] = True
            return diagnostics

        legal_play_candidates = _legal_play_candidates(
            state,
            play_candidates,
            action_mask=action_mask,
        )
        if not legal_play_candidates:
            diagnostics["hand_play_not_in_candidates"] = True
            diagnostics["hand_play_no_legal_candidates"] = True
            return diagnostics

        if int(state.round_resets.ante) == 1:
            chip_best = max(
                legal_play_candidates,
                key=lambda candidate: candidate.raw_score,
            )
            diagnostics.update(
                {
                    "ante1_chip_best_score": float(chip_best.raw_score),
                    "ante1_chip_best_hand": chip_best.hand_name,
                }
            )

        chosen = next(
            (candidate for candidate in play_candidates if candidate.indices == indices),
            None,
        )
        if chosen is None:
            diagnostics["hand_play_not_in_candidates"] = True
            diagnostics["hand_play_best_hand"] = legal_play_candidates[0].hand_name
            return diagnostics

        best = legal_play_candidates[0]
        chosen_is_legal = any(
            candidate.indices == chosen.indices for candidate in legal_play_candidates
        )
        if not chosen_is_legal:
            diagnostics.update(
                {
                    "hand_play_in_candidates": True,
                    "hand_play_illegal_candidate": True,
                    "hand_play_best_hand": best.hand_name,
                }
            )
            return diagnostics

        value_ratio = float(chosen.estimated_score / max(best.estimated_score, 1e-9))
        diagnostics.update({
            "hand_play_in_candidates": True,
            "hand_play_top1": chosen.indices == best.indices,
            "hand_play_top3": any(
                c.indices == chosen.indices for c in legal_play_candidates[:3]
            ),
            "hand_play_candidate_value_ratio": value_ratio,
            # Explicit aliases make the corrected denominator discoverable;
            # legacy fields above retain their names for existing dashboards.
            "hand_play_legal_top1": chosen.indices == best.indices,
            "hand_play_legal_candidate_value_ratio": value_ratio,
            "hand_play_chosen_hand": chosen.hand_name,
            "hand_play_best_hand": best.hand_name,
        })
        if int(state.round_resets.ante) == 1:
            conservative_best = max(
                legal_play_candidates,
                key=lambda candidate: candidate.raw_score,
            )
            conservative_ratio = float(
                chosen.raw_score / max(conservative_best.raw_score, 1e-9)
            )
            diagnostics.update(
                {
                    "ante1_play_observed": True,
                    "ante1_chip_chosen_score": float(chosen.raw_score),
                    "ante1_conservative_chosen_best_ratio": conservative_ratio,
                }
            )
            if blind_target > 0.0:
                remaining_target = max(float(blind_target) - float(round_score), 0.0)
                proxy_available = conservative_best.raw_score >= remaining_target
                proxy_chosen = proxy_available and chosen.raw_score >= remaining_target
                diagnostics.update(
                    {
                        "ante1_one_hand_clear_proxy_observed": True,
                        "ante1_one_hand_clear_proxy_available": proxy_available,
                        "ante1_one_hand_clear_proxy_chosen": proxy_chosen,
                        "ante1_one_hand_clear_proxy_missed": proxy_available
                        and not proxy_chosen,
                    }
                )
        return diagnostics

    if decoded.action_type == ActionType.DISCARD_SUBSET:
        indices = tuple(subset_indices(decoded.index))
        diagnostics = {
            "purple_seal_discarded_count": sum(
                index < len(state.hand_cards)
                and state.hand_cards[index].seal == "Purple"
                and not state.hand_cards[index].debuff
                for index in indices
            )
        }
        if int(state.round_resets.ante) == 1:
            play_candidates, _discard_candidates = generate_hand_candidates(state)
            if play_candidates:
                legal_play_candidates = _legal_play_candidates(
                    state,
                    play_candidates,
                    action_mask=action_mask,
                )
                if not legal_play_candidates:
                    return diagnostics
                chip_best = max(
                    legal_play_candidates,
                    key=lambda candidate: candidate.raw_score,
                )
                diagnostics.update(
                    {
                        "ante1_chip_best_score": float(chip_best.raw_score),
                        "ante1_chip_best_hand": chip_best.hand_name,
                    }
                )
        return diagnostics

    if decoded.action_type in {
        ActionType.USE_CONSUMABLE_NO_TARGET,
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        ActionType.USE_CONSUMABLE_JOKER,
    }:
        if decoded.index >= len(state.consumables):
            return {}
        center_key = _center_key(state.consumables[decoded.index])
        center_set = _center_set(state, center_key)
        diagnostics = {
            "consumable_use_set": center_set,
            "consumable_use_key": center_key,
        }
        if center_set == "Planet":
            diagnostics.update(_planet_diagnostics(state, center_key, prefix="planet_use"))
        return diagnostics

    if decoded.action_type == ActionType.PACK_CLAIM:
        if state.pack is None or decoded.index >= len(state.pack.cards):
            return {}
        center_key = _center_key(state.pack.cards[decoded.index])
        center_set = _center_set(state, center_key)
        diagnostics = {
            "pack_claim_set": center_set,
            "pack_claim_key": center_key,
            "pack_claim_seal": str(getattr(state.pack.cards[decoded.index], "seal", "") or ""),
        }
        if center_set == "Planet":
            diagnostics.update(_planet_diagnostics(state, center_key, prefix="planet_claim"))
            diagnostics["planet_claim_best_available"] = _is_best_planet_in_pack(
                state, decoded.index
            )
        return diagnostics

    if decoded.action_type == ActionType.PACK_SKIP and state.pack is not None:
        return {
            "pack_skip_state_name": state.pack.state_name,
            "planet_pack_skip": state.pack.state_name == "PLANET_PACK",
        }

    return {}


def _ordered_counter_diff(before: tuple[str, ...], after: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Return acquired and removed IDs while preserving roster order."""
    before_remaining = Counter(before)
    after_remaining = Counter(after)
    acquired_remaining = after_remaining - before_remaining
    removed_remaining = before_remaining - after_remaining

    acquired: list[str] = []
    for key in after:
        if acquired_remaining[key] > 0:
            acquired.append(key)
            acquired_remaining[key] -= 1

    removed: list[str] = []
    for key in before:
        if removed_remaining[key] > 0:
            removed.append(key)
            removed_remaining[key] -= 1
    return acquired, removed


def _add_indexed_ids(
    diagnostics: dict[str, Any],
    prefix: str,
    values: list[str] | tuple[str, ...],
    *,
    limit: int,
) -> None:
    diagnostics[f"{prefix}_count"] = len(values)
    emitted = min(len(values), limit)
    diagnostics[f"{prefix}_emitted_count"] = emitted
    for index, value in enumerate(values[:limit]):
        diagnostics[f"{prefix}_{index}_id"] = value


def _shop_joker_ids(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(card.get("key") or "")
        for card in (info.get("shop_cards") or ())
        if isinstance(card, dict) and card.get("set") == "Joker" and card.get("key")
    )


def _record_consumable_offer_surface(
    diagnostics: dict[str, Any],
    curr_info: Mapping[str, Any],
    *,
    surface: str,
    action_mask=None,
) -> None:
    """Record one newly visible shop/pack decision surface with bounded tags."""

    details_key = "shop_cards" if surface == "shop" else "pack_card_details"
    details = tuple(curr_info.get(details_key) or ())
    inventory = tuple(curr_info.get("consumable_details") or ())
    inventory_count = len(inventory)
    capacity = max(int(curr_info.get("consumable_limit", 0) or 0), 0)
    dollars = max(float(curr_info.get("dollars", 0) or 0), 0.0)

    for consumable_set in ("Planet", "Tarot"):
        offered_indices = [
            index
            for index, detail in enumerate(details)
            if isinstance(detail, Mapping) and detail.get("set") == consumable_set
        ]
        if surface == "shop":
            claimable_indices = [
                index
                for index in offered_indices
                if float(details[index].get("cost", 0) or 0) <= dollars
                and inventory_count < capacity
            ]
            full_blocked = sum(
                float(details[index].get("cost", 0) or 0) <= dollars
                and inventory_count >= capacity
                for index in offered_indices
            )
        else:
            claimable_indices = []
            for index in offered_indices:
                action_id = int(ActionRange.PACK_CLAIM_START) + index
                if action_mask is not None and action_id < len(action_mask) and bool(action_mask[action_id]):
                    claimable_indices.append(index)
            full_blocked = sum(
                inventory_count >= capacity and index not in claimable_indices
                for index in offered_indices
            )
        prefix = f"consumable_{consumable_set.lower()}"
        diagnostics[f"{prefix}_offered_count"] = len(offered_indices)
        diagnostics[f"{prefix}_claimable_count"] = len(claimable_indices)
        diagnostics[f"{prefix}_inventory_full_blocked_count"] = int(full_blocked)

    if surface == "pack":
        for seal in ("Blue", "Purple"):
            offered_indices = [
                index
                for index, detail in enumerate(details)
                if isinstance(detail, Mapping) and str(detail.get("seal") or "") == seal
            ]
            prefix = f"seal_{seal.lower()}"
            diagnostics[f"{prefix}_offered_count"] = len(offered_indices)


def step_event_diagnostics(
    prev_info: dict[str, Any],
    curr_info: dict[str, Any],
    decoded,
    *,
    next_action_mask=None,
) -> dict[str, Any]:
    """Describe bounded joker/shop events produced by one environment step."""
    diagnostics: dict[str, Any] = {}
    before_keys = tuple(str(key) for key in (prev_info.get("joker_keys") or ()))
    after_keys = tuple(str(key) for key in (curr_info.get("joker_keys") or ()))
    acquired, removed = _ordered_counter_diff(before_keys, after_keys)
    if acquired or removed:
        diagnostics["joker_roster_changed"] = True
        _add_indexed_ids(diagnostics, "joker_acquired", acquired, limit=MAX_DIAGNOSTIC_EVENTS)
        _add_indexed_ids(diagnostics, "joker_removed", removed, limit=MAX_DIAGNOSTIC_EVENTS)
        diagnostics["joker_turnover_count"] = len(acquired) + len(removed)
        diagnostics["joker_churn_count"] = len(removed)
        diagnostics["joker_replacement_event"] = bool(acquired and removed)

    if decoded.action_type == ActionType.SHOP_BUY:
        shop_cards = tuple(prev_info.get("shop_cards") or ())
        if 0 <= decoded.index < len(shop_cards):
            card = shop_cards[decoded.index]
            if isinstance(card, dict) and card.get("set") == "Joker":
                diagnostics["shop_bought_joker_id"] = str(card.get("key") or "")
            if isinstance(card, dict) and card.get("set") in {"Planet", "Tarot"}:
                diagnostics["shop_bought_consumable_set"] = str(card.get("set") or "")
                diagnostics["shop_bought_consumable_id"] = str(card.get("key") or "")

    if decoded.action_type == ActionType.SHOP_SELL_JOKER:
        joker_details = tuple(prev_info.get("joker_details") or ())
        if 0 <= decoded.index < len(joker_details):
            joker = joker_details[decoded.index]
            if isinstance(joker, dict):
                diagnostics["shop_sold_joker_id"] = str(joker.get("key") or "")

    shop_surface = bool(curr_info.get("in_shop")) and (
        not bool(prev_info.get("in_shop"))
        or decoded.action_type
        in {
            ActionType.SHOP_BUY,
            ActionType.SHOP_REROLL,
            ActionType.SHOP_SELL_CONSUMABLE,
        }
    )
    pack_surface = bool(curr_info.get("pack_card_details")) and (
        not bool(prev_info.get("pack_card_details")) or decoded.action_type == ActionType.PACK_CLAIM
    )
    if shop_surface:
        _record_consumable_offer_surface(diagnostics, curr_info, surface="shop")
    elif pack_surface:
        _record_consumable_offer_surface(
            diagnostics,
            curr_info,
            surface="pack",
            action_mask=next_action_mask,
        )

    before_consumables = Counter(
        (str(item.get("key") or ""), str(item.get("set") or ""))
        for item in (prev_info.get("consumable_details") or ())
        if isinstance(item, dict)
    )
    after_consumables = Counter(
        (str(item.get("key") or ""), str(item.get("set") or ""))
        for item in (curr_info.get("consumable_details") or ())
        if isinstance(item, dict)
    )
    generated = after_consumables - before_consumables
    if decoded.action_type == ActionType.DISCARD_SUBSET:
        diagnostics["purple_seal_tarot_generated_count"] = sum(
            count for (_key, card_set), count in generated.items() if card_set == "Tarot"
        )
    elif decoded.action_type == ActionType.PLAY_SUBSET:
        diagnostics["blue_seal_planet_generated_count"] = sum(
            count for (_key, card_set), count in generated.items() if card_set == "Planet"
        )

    after_offers = _shop_joker_ids(curr_info)
    entered_shop = bool(curr_info.get("in_shop")) and not bool(prev_info.get("in_shop"))
    if after_offers and (entered_shop or decoded.action_type == ActionType.SHOP_REROLL):
        diagnostics["shop_joker_offer_observed"] = True
        _add_indexed_ids(
            diagnostics,
            "shop_offered_joker",
            list(after_offers),
            limit=MAX_DIAGNOSTIC_SHOP_JOKERS,
        )

    if decoded.action_type == ActionType.SHOP_LEAVE:
        clear_probability = float(prev_info.get("clear_probability", 0.0) or 0.0)
        unsafe = clear_probability < 0.65
        reroll_cost = max(float(prev_info.get("reroll_cost", 0) or 0), 0.0)
        can_reroll = bool(prev_info.get("free_rerolls", 0)) or float(prev_info.get("dollars", 0) or 0) >= reroll_cost
        rescue = best_confident_joker_rescue(prev_info)
        diagnostics.update(
            {
                "shop_leave_observed": True,
                "shop_unsafe_leave": unsafe,
                "shop_unsafe_can_reroll": bool(unsafe and can_reroll),
                "shop_missed_confident_upgrade": bool(
                    rescue is not None and rescue.clear_probability_delta >= 0.10
                ),
                "shop_leave_clear_probability": clear_probability,
                "shop_leave_joker_full_weak": bool(
                    prev_info.get("joker_full") and prev_info.get("weak_confident_joker")
                ),
            }
        )
        if rescue is not None:
            diagnostics["shop_best_confident_upgrade_delta"] = rescue.clear_probability_delta

    return diagnostics


def _modeled_fraction(build: BuildEval) -> float:
    marginals = build.build_value.joker_marginals
    if not marginals:
        return 0.0
    return sum(float(m.modeled_effect_fraction) for m in marginals) / len(marginals)


def _add_build_summary(diagnostics: dict[str, Any], prefix: str, build: BuildEval) -> None:
    estimate = build.build_value
    diagnostics[f"{prefix}_estimated_score"] = float(build.estimated_score)
    diagnostics[f"{prefix}_required_score"] = float(build.required_score_per_hand)
    diagnostics[f"{prefix}_readiness"] = float(build.readiness_ratio)
    diagnostics[f"{prefix}_score_gain_ratio"] = float(
        build.estimated_score / max(build.no_joker_baseline_score, 1.0)
    )
    diagnostics[f"{prefix}_modeled_fraction"] = _modeled_fraction(build)

    marginals = estimate.joker_marginals
    diagnostics[f"{prefix}_joker_count"] = len(marginals)
    diagnostics[f"{prefix}_joker_emitted_count"] = min(len(marginals), MAX_DIAGNOSTIC_JOKERS)
    for marginal in marginals[:MAX_DIAGNOSTIC_JOKERS]:
        index = marginal.index
        diagnostics[f"{prefix}_joker_{index}_id"] = marginal.key
        diagnostics[f"{prefix}_joker_{index}_marginal_ratio"] = float(marginal.score_ratio)
        diagnostics[f"{prefix}_joker_{index}_modeled_fraction"] = float(
            marginal.modeled_effect_fraction
        )


def _hologram_x_mults(info: dict[str, Any]) -> tuple[float, ...]:
    values: list[float] = []
    for joker in info.get("joker_details") or ():
        if isinstance(joker, dict) and joker.get("key") == "j_hologram":
            values.append(float(joker.get("x_mult", 1.0) or 1.0))
    return tuple(sorted(values))


def build_step_diagnostics(
    prev_info: dict[str, Any],
    curr_info: dict[str, Any],
    reward_config,
    *,
    win_ante: int,
) -> dict[str, Any]:
    """Evaluate pre/post build and potential values for a relevant step."""
    from dataclasses import replace

    from .reward import state_potential_breakdown

    try:
        pre_build = evaluate_build(prev_info)
        post_build = evaluate_build(curr_info)
    except (KeyError, OverflowError, TypeError, ValueError):
        return {"build_diagnostics_failed": True}

    # Private in-process cache only. The reward evaluator already recognizes
    # these keys, so diagnostics do not force a second leave-one-out build pass.
    # They are never copied into the vector info payload below.
    prev_info["_build_value_estimate"] = pre_build.build_value
    curr_info["_build_value_estimate"] = post_build.build_value
    try:
        pre_plan = estimate_hand_plans(prev_info)
        post_plan = estimate_hand_plans(curr_info)
    except (KeyError, OverflowError, TypeError, ValueError):
        pre_plan = post_plan = None
    if pre_plan is not None:
        prev_info["_hand_plan_estimate"] = pre_plan
    if post_plan is not None:
        curr_info["_hand_plan_estimate"] = post_plan

    diagnostics: dict[str, Any] = {"build_diagnostics_observed": True}
    _add_build_summary(diagnostics, "build_pre", pre_build)
    _add_build_summary(diagnostics, "build_post", post_build)
    diagnostics["build_estimated_score_delta"] = float(
        post_build.estimated_score - pre_build.estimated_score
    )
    if pre_plan is not None and post_plan is not None:
        diagnostics.update(
            {
                "hand_plan_pre_type": pre_plan.best.hand_type,
                "hand_plan_post_type": post_plan.best.hand_type,
                "hand_plan_pre_reliability": float(pre_plan.best.draw_reliability),
                "hand_plan_post_reliability": float(post_plan.best.draw_reliability),
                "hand_plan_pre_readiness": float(pre_plan.best.readiness_ratio),
                "hand_plan_post_readiness": float(post_plan.best.readiness_ratio),
            }
        )
    pre_seals = prev_info.get("deck_stats", {}).get("seal_counts", {})
    post_seals = curr_info.get("deck_stats", {}).get("seal_counts", {})
    for seal in ("Blue", "Purple", "Gold", "Red"):
        diagnostics[f"seal_pre_{seal.lower()}_count"] = int(pre_seals.get(seal, 0) or 0)
        diagnostics[f"seal_post_{seal.lower()}_count"] = int(post_seals.get(seal, 0) or 0)

    pre_hologram = _hologram_x_mults(prev_info)
    post_hologram = _hologram_x_mults(curr_info)
    matched = len(pre_hologram) if len(pre_hologram) == len(post_hologram) else 0
    changed = sum(
        abs(post_hologram[index] - pre_hologram[index]) > 1e-12
        for index in range(matched)
    )
    if changed:
        prev_total = sum(pre_hologram[:matched])
        current_total = sum(post_hologram[:matched])
        diagnostics.update(
            {
                "hologram_scaling_count": changed,
                "hologram_x_mult_prev": float(prev_total),
                "hologram_x_mult_current": float(current_total),
                "hologram_x_mult_delta": float(current_total - prev_total),
                "hologram_build_score_delta": float(
                    post_build.estimated_score - pre_build.estimated_score
                ),
            }
        )

    try:
        potential_config = replace(reward_config, potential_win_ante=max(int(win_ante), 2))
        pre_for_potential = dict(prev_info)
        curr_for_potential = dict(curr_info)
        pre_for_potential["_build_value_estimate"] = pre_build.build_value
        curr_for_potential["_build_value_estimate"] = post_build.build_value
        pre_potential = state_potential_breakdown(pre_for_potential, potential_config)
        post_potential = state_potential_breakdown(curr_for_potential, potential_config)
        for component in _POTENTIAL_COMPONENTS:
            before = float(pre_potential[component])
            after = float(post_potential[component])
            diagnostics[f"potential_pre_{component}"] = before
            diagnostics[f"potential_post_{component}"] = after
            diagnostics[f"potential_delta_{component}"] = after - before
    except (KeyError, OverflowError, TypeError, ValueError):
        diagnostics["potential_diagnostics_failed"] = True
    return diagnostics


@dataclass
class ExactPlayCounterfactual:
    """One copied pre-play state with one focal joker removed."""

    state: Any
    indices: tuple[int, ...]
    focal_joker_id: str
    representative_ratio: float


def prepare_exact_play_counterfactual(
    state,
    indices: tuple[int, ...],
    captured_info: dict[str, Any],
    *,
    sample_index: int,
) -> tuple[ExactPlayCounterfactual | None, dict[str, Any]]:
    """Prepare one exact leave-one-out replay without touching the live state."""
    try:
        build = evaluate_build(captured_info)
    except (KeyError, OverflowError, TypeError, ValueError):
        return None, {}

    eligible = [
        marginal
        for marginal in build.build_value.joker_marginals
        if marginal.modeled_effect_fraction > 0.0
    ]
    if not eligible:
        return None, {}

    focal = eligible[sample_index % len(eligible)]
    diagnostics: dict[str, Any] = {
        "counterfactual_call": True,
        "counterfactual_focal_joker_id": focal.key,
        "counterfactual_representative_ratio": float(focal.score_ratio),
    }
    try:
        copied_state = deepcopy(state, {id(state.data): state.data})
        if focal.index >= len(copied_state.jokers):
            raise IndexError("focal joker index missing from copied state")
        remove_joker(copied_state, copied_state.jokers[focal.index])
    except Exception as exc:
        diagnostics["counterfactual_failure"] = True
        diagnostics["counterfactual_failure_reason"] = type(exc).__name__
        return None, diagnostics

    return (
        ExactPlayCounterfactual(
            state=copied_state,
            indices=indices,
            focal_joker_id=focal.key,
            representative_ratio=float(focal.score_ratio),
        ),
        diagnostics,
    )


def finish_exact_play_counterfactual(
    probe: ExactPlayCounterfactual,
    *,
    actual_score: float,
) -> dict[str, Any]:
    """Replay the selected cards once and compare exact vs analytic ratios."""
    try:
        selected = [probe.state.hand_cards[index] for index in sorted(probe.indices)]
        exact_without = float(play_cards(probe.state, selected).score.total)
        exact_ratio = float(actual_score) / max(exact_without, 1.0)
        signed_gap = log(max(exact_ratio, 1e-9)) - log(
            max(probe.representative_ratio, 1e-9)
        )
        return {
            "counterfactual_realized_score_with": float(actual_score),
            "counterfactual_realized_score_without": exact_without,
            "counterfactual_realized_ratio": exact_ratio,
            "counterfactual_representative_vs_realized_log_ratio_gap": signed_gap,
            "counterfactual_representative_vs_realized_abs_log_ratio_gap": abs(signed_gap),
            "counterfactual_failure": False,
        }
    except Exception as exc:
        return {
            "counterfactual_failure": True,
            "counterfactual_failure_reason": type(exc).__name__,
        }
