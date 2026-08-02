"""Conservative, boss-aware estimates of clearing the immediate blind.

These estimates are deliberately bounded and pessimistic.  They are used as
policy inputs, economy gates, and confidence-gated positive shop rewards; they
are not an oracle and never punish the policy for declining an offer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .build_value import BuildValueEstimate, estimate_build_value
from .hand_plan import HandPlanEstimate, estimate_hand_plans


@dataclass(frozen=True)
class ClearRiskEstimate:
    clear_probability: float
    immediate_death_probability: float
    hand_type: str
    draw_reliability: float
    score_margin: float
    model_confidence: float


@dataclass(frozen=True)
class JokerRescue:
    clear_probability_delta: float
    offered_key: str
    removed_key: str
    model_confidence: float


# Target multipliers and current hand/discard counts already cover Wall,
# Needle, Water, and Manacle once their effects are active.  These factors
# reserve additional margin for constraints the representative score model
# cannot faithfully simulate before entering the boss.
_BOSS_SAFETY_FACTORS = {
    "bl_hook": 0.82,
    "bl_mouth": 0.78,
    "bl_psychic": 0.82,
    "bl_eye": 0.85,
    "bl_flint": 0.72,
    "bl_house": 0.88,
    "bl_fish": 0.88,
    "bl_wheel": 0.90,
    "bl_mark": 0.90,
    "bl_goad": 0.86,
    "bl_head": 0.86,
    "bl_club": 0.86,
    "bl_window": 0.86,
    "bl_plant": 0.82,
    "bl_pillar": 0.88,
}


def _clip01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _active_boss_key(info: Mapping[str, Any]) -> str:
    blind = str(info.get("blind_on_deck") or "").lower()
    if blind == "boss":
        return str(info.get("boss_key") or "")
    return ""


def _roster_confidence(build: BuildValueEstimate) -> float:
    marginals = tuple(build.joker_marginals)
    if not marginals:
        return 1.0
    return _clip01(sum(float(item.modeled_effect_fraction) for item in marginals) / len(marginals))


def _risk_info(info: Mapping[str, Any]) -> dict[str, Any]:
    adjusted = dict(info)
    target = max(float(info.get("blind_target", 0) or 0), 0.0)
    score = max(float(info.get("round_score", 0) or 0), 0.0)
    adjusted["blind_target"] = max(target - score, 1.0)
    return adjusted


def estimate_clear_risk(
    info: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]] | None = None,
    *,
    plans: HandPlanEstimate | None = None,
) -> ClearRiskEstimate:
    """Estimate the probability of clearing the current/upcoming blind.

    The hand planner supplies exact draw reliability for supported plans.  A
    buffered score margin maps to a bounded success probability: merely
    matching the deterministic target is treated as a coin flip, while 1.25x
    modeled margin is required for full score confidence.  Boss constraints
    then reserve additional safety margin.
    """

    adjusted = _risk_info(info)
    owned = tuple(jokers if jokers is not None else (info.get("joker_details") or ()))
    try:
        resolved_plans = plans if plans is not None else estimate_hand_plans(adjusted, owned)
        build = estimate_build_value(adjusted, owned)
    except (KeyError, OverflowError, TypeError, ValueError):
        resolved_plans = None
        build = None
    if resolved_plans is None:
        return ClearRiskEstimate(0.0, 1.0, "", 0.0, 0.0, 0.0)

    boss_factor = _BOSS_SAFETY_FACTORS.get(_active_boss_key(info), 1.0)
    best_probability = 0.0
    best_plan = resolved_plans.best
    best_margin = max(float(best_plan.readiness_ratio), 0.0)
    for plan in resolved_plans.plans:
        margin = max(float(plan.readiness_ratio), 0.0)
        # 0.75x modeled score is unsalvageable, 1.0x is deliberately only a
        # 50/50 estimate, and 1.25x is the minimum fully-safe modeled margin.
        score_probability = _clip01((margin - 0.75) / 0.50)
        probability = _clip01(float(plan.draw_reliability) * score_probability * boss_factor)
        if probability > best_probability:
            best_probability = probability
            best_plan = plan
            best_margin = margin

    confidence = _roster_confidence(build) if build is not None else 0.0
    # Unknown Joker effects should make the risk estimate more conservative,
    # but not collapse it to zero and hide all useful danger information.
    confidence_factor = 0.70 + 0.30 * confidence
    clear_probability = _clip01(best_probability * confidence_factor)
    return ClearRiskEstimate(
        clear_probability=clear_probability,
        immediate_death_probability=1.0 - clear_probability,
        hand_type=best_plan.hand_type,
        draw_reliability=float(best_plan.draw_reliability),
        score_margin=best_margin,
        model_confidence=confidence,
    )


def weakest_confident_joker(info: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return (marginal ratio, modeled confidence) for the weakest known Joker."""

    try:
        build = estimate_build_value(info)
    except (KeyError, OverflowError, TypeError, ValueError):
        return None
    known = [
        (float(item.score_ratio), float(item.modeled_effect_fraction))
        for item in build.joker_marginals
        if float(item.modeled_effect_fraction) >= 0.75
    ]
    return min(known, key=lambda item: item[0]) if known else None


def best_confident_joker_rescue(info: Mapping[str, Any]) -> JokerRescue | None:
    """Return the best affordable, confidently modeled visible Joker rescue."""

    owned = tuple(item for item in (info.get("joker_details") or ()) if isinstance(item, Mapping))
    limit = max(int(info.get("joker_limit", 5) or 5), 0)
    dollars = max(float(info.get("dollars", 0) or 0), 0.0)
    before = estimate_clear_risk(info, owned)
    best: JokerRescue | None = None

    for card in info.get("shop_cards") or ():
        if not isinstance(card, Mapping) or card.get("set") != "Joker":
            continue
        offer = card.get("joker")
        if not isinstance(offer, Mapping):
            continue
        cost = max(float(card.get("cost", 0) or 0), 0.0)
        rosters: list[tuple[tuple[Mapping[str, Any], ...], str]] = []
        negative = bool((offer.get("edition") or {}).get("negative"))
        if len(owned) < limit or negative:
            if cost <= dollars:
                rosters.append(((*owned, offer), ""))
        else:
            for index, current in enumerate(owned):
                if current.get("eternal"):
                    continue
                proceeds = max(float(current.get("sell_cost", 0) or 0), 0.0)
                if cost <= dollars + proceeds:
                    rosters.append(((*owned[:index], offer, *owned[index + 1 :]), str(current.get("key") or "")))

        for roster, removed_key in rosters:
            try:
                build = estimate_build_value(info, roster)
            except (KeyError, OverflowError, TypeError, ValueError):
                continue
            acquired = [item for item in build.joker_marginals if str(item.key) == str(offer.get("key") or "")]
            confidence = max((float(item.modeled_effect_fraction) for item in acquired), default=0.0)
            if confidence < 0.75:
                continue
            after = estimate_clear_risk(info, roster)
            rescue = JokerRescue(
                clear_probability_delta=max(after.clear_probability - before.clear_probability, 0.0),
                offered_key=str(card.get("key") or ""),
                removed_key=removed_key,
                model_confidence=confidence,
            )
            if best is None or rescue.clear_probability_delta > best.clear_probability_delta:
                best = rescue
    return best


def capture_state_risk(state, round_score: int = 0) -> ClearRiskEstimate:
    """Build the serialized strategy snapshot needed for observation risk."""

    from .shop_eval import capture_build_features

    blind_on_deck = str(state.blind_on_deck or "")
    info: dict[str, Any] = {
        "ante": int(state.round_resets.ante),
        "round_score": max(float(round_score), 0.0),
        "blind_on_deck": blind_on_deck,
        "boss_key": str(state.round_resets.blind_choices.get("Boss", "") or ""),
        "dollars": float(state.dollars),
    }
    info.update(capture_build_features(state))
    return estimate_clear_risk(info)
