"""Action, build, and joker diagnostics for agent rollouts.

The action-quality fields are shared by BalatroEnv and fast_generate because
reward shaping reads them on both paths. Build, shop-event, Hologram, and exact
counterfactual diagnostics are emitted by BalatroEnv for PPO metrics only.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from math import log
from typing import Any

from pylatro.flow import play_cards
from pylatro.instances import remove_joker

from .action import ActionType
from .hand_candidates import generate_hand_candidates
from .shop_eval import BuildEval, evaluate_build
from .subset_actions import subset_indices

MAX_DIAGNOSTIC_JOKERS = 12
MAX_DIAGNOSTIC_EVENTS = 12
MAX_DIAGNOSTIC_SHOP_JOKERS = 8

_POTENTIAL_COMPONENTS = (
    "blind_progress",
    "ante_progress",
    "realized_build_quality",
    "scaling_option_value",
    "readiness",
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
    return {
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


def action_diagnostics(state, decoded) -> dict[str, Any]:
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
        if any(index >= len(state.hand_cards) for index in indices):
            diagnostics["hand_play_not_in_candidates"] = True
            return diagnostics
        play_candidates, _discard_candidates = generate_hand_candidates(state)
        if not play_candidates:
            diagnostics["hand_play_not_in_candidates"] = True
            return diagnostics

        chosen = next(
            (candidate for candidate in play_candidates if candidate.indices == indices),
            None,
        )
        if chosen is None:
            diagnostics["hand_play_not_in_candidates"] = True
            diagnostics["hand_play_best_hand"] = play_candidates[0].hand_name
            return diagnostics

        best = play_candidates[0]
        diagnostics.update({
            "hand_play_in_candidates": True,
            "hand_play_top1": chosen.indices == best.indices,
            "hand_play_top3": any(c.indices == chosen.indices for c in play_candidates[:3]),
            "hand_play_candidate_value_ratio": float(
                chosen.estimated_score / max(best.estimated_score, 1e-9)
            ),
            "hand_play_chosen_hand": chosen.hand_name,
            "hand_play_best_hand": best.hand_name,
        })
        return diagnostics

    if decoded.action_type == ActionType.USE_CONSUMABLE_NO_TARGET:
        if decoded.index >= len(state.consumables):
            return {}
        center_key = _center_key(state.consumables[decoded.index])
        if _center_set(state, center_key) != "Planet":
            return {}
        return _planet_diagnostics(state, center_key, prefix="planet_use")

    if decoded.action_type == ActionType.PACK_CLAIM:
        if state.pack is None or decoded.index >= len(state.pack.cards):
            return {}
        center_key = _center_key(state.pack.cards[decoded.index])
        if _center_set(state, center_key) != "Planet":
            return {}
        diagnostics = _planet_diagnostics(state, center_key, prefix="planet_claim")
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


def step_event_diagnostics(
    prev_info: dict[str, Any],
    curr_info: dict[str, Any],
    decoded,
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

    if decoded.action_type == ActionType.SHOP_SELL_JOKER:
        joker_details = tuple(prev_info.get("joker_details") or ())
        if 0 <= decoded.index < len(joker_details):
            joker = joker_details[decoded.index]
            if isinstance(joker, dict):
                diagnostics["shop_sold_joker_id"] = str(joker.get("key") or "")

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

    diagnostics: dict[str, Any] = {"build_diagnostics_observed": True}
    _add_build_summary(diagnostics, "build_pre", pre_build)
    _add_build_summary(diagnostics, "build_post", post_build)
    diagnostics["build_estimated_score_delta"] = float(
        post_build.estimated_score - pre_build.estimated_score
    )

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
