"""The reward model shared by supervised generation and PPO."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

from .build_value import BuildValueEstimate, estimate_build_value
from .hand_plan import estimate_hand_plans
from .heuristic import _SCALING_JOKER_KEYS
from .risk import best_confident_joker_rescue, estimate_clear_risk
from .strategy_value import estimate_strategy_value

if TYPE_CHECKING:
    from pylatro.models import RunState


@dataclass
class RewardConfig:
    """Configuration for the sole active reward model.

    Terminal outcome and potential shaping are always enabled. The optional
    planet-alignment and score/build terms are explicit additions to that
    baseline, not alternate reward versions.
    """

    gamma: float = 0.997
    dense_reward_scale: float = 1.0
    consumable_reward_scale: float = 1.0

    enable_planet_match_rewards: bool = False
    planet_unmatched_use_penalty_coeff: float = 0.0
    planet_unmatched_claim_penalty_coeff: float = 0.0

    enable_score_build_potential: bool = False
    potential_w_blind: float = 0.5
    potential_w_ante: float = 2.0
    potential_win_ante: int = 8
    potential_w_build_quality: float = 0.88
    potential_w_readiness: float = 0.40
    potential_w_scaling_option: float = 0.22
    potential_w_economy: float = 0.30
    potential_w_tarot_option: float = 0.20
    potential_w_planet_option: float = 0.25
    potential_w_seals: float = 1.25
    # Visible shop rescue is diagnostic-only in the potential.  A positive
    # option potential would become an implicit penalty when the policy leaves
    # or rerolls, and our counterfactual model is not complete enough to punish
    # a declined offer.  Realized, confidence-gated upgrades are rewarded below.
    potential_w_joker_search: float = 0.0
    potential_w_standard_pack_search: float = 0.50
    # Potential allocated only to the unsafe region. It rises linearly with
    # clear probability until the build reaches the 65% safety threshold, then
    # saturates so PPO is paid for rescue rather than needless over-preparation.
    potential_w_survival_safety: float = 0.40
    potential_build_cap: float = 2.0
    potential_readiness_saturation: float = 1.5


# Increment whenever reward semantics change without a RewardConfig field
# change. It participates in the checkpoint fingerprint.
REWARD_MODEL_VERSION = 12


def reward_config_snapshot(config: RewardConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete plain-data reward configuration."""
    if isinstance(config, RewardConfig):
        return asdict(config)
    return dict(config)


def reward_config_fingerprint(
    config: RewardConfig | Mapping[str, Any],
    *,
    reward_model_version: int | None = None,
) -> str:
    """Return a canonical SHA-256 fingerprint of reward code and configuration."""
    version = REWARD_MODEL_VERSION if reward_model_version is None else reward_model_version
    payload = {
        "reward_config": reward_config_snapshot(config),
        "reward_model_version": int(version),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def reward_checkpoint_metadata(config: RewardConfig) -> dict[str, Any]:
    """Build the reward metadata persisted in full PPO checkpoints."""
    return {
        "reward_config": reward_config_snapshot(config),
        "reward_model_version": REWARD_MODEL_VERSION,
        "reward_fingerprint": reward_config_fingerprint(config),
    }


DEFAULT_REWARD_CONFIG = RewardConfig()

# Terminal values are deliberately O(1-10), large enough to dominate the
# bounded potential while fitting the default critic support [-8, 12].
WIN_VALUE = 10.0
LOSS_BASE = -5.0
ANTE_PROGRESS_VALUE = 0.5
STALL_EXTRA_PENALTY = 1.0
EARLY_DEATH_PENALTIES = {1: 2.0, 2: 1.0}

IDLE_PENALTY_BASE = 0.001
IDLE_PENALTY_RAMP = 0.0005
IDLE_PENALTY_CAP = 0.02
JOKER_MOVE_NON_IMPROVING_PENALTY = 0.10
JOKER_MOVE_REPEAT_PENALTY = 0.02
JOKER_MOVE_PENALTY_CAP = 0.50

# Ante 1 has little build scaling, so actual blind-score progress carries a
# small, bounded potential. Its signed difference is segmentation invariant.
ANTE1_CHIP_TEMPO_BONUS = 0.40

PLANET_MATCH_BONUS = 0.5
PLANET_PLAYED_HAND_BONUS = 0.25
PLANET_UNMATCHED_MIN_PROGRESS = 0.25
JOKER_UPGRADE_BONUS = 0.40
DANGER_REROLL_BONUS = 0.20
VALUE_TAROT_BONUS = 1.20

_BLIND_INDEX = {"small": 0, "big": 1, "boss": 2}
_FOOL_PROTECT_TARGETS = {"c_death", "c_hermit", "c_temperance"}

REWARD_COMPONENT_NAMES = (
    "terminal",
    "potential_shaping",
    "idle_penalty",
    "planet_match_bonus",
    "planet_played_hand_bonus",
    "planet_unmatched_use_penalty",
    "planet_unmatched_claim_penalty",
    "tarot_value_bonus",
    "joker_upgrade_bonus",
    "danger_reroll_bonus",
    "survival_shaping",
    "joker_move",
    "ante1_chip_tempo",
)
REWARD_INFO_KEYS = tuple(f"reward_{name}" for name in ("total", *REWARD_COMPONENT_NAMES))

# Used only to aggregate TensorBoard reward diagnostics.
_COMPONENT_GROUP = {
    "potential_shaping": "potential",
    "idle_penalty": "idle",
    "planet_match_bonus": "consumable",
    "planet_played_hand_bonus": "consumable",
    "planet_unmatched_use_penalty": "consumable",
    "planet_unmatched_claim_penalty": "consumable",
    "tarot_value_bonus": "consumable",
    "joker_upgrade_bonus": "shop",
    "danger_reroll_bonus": "shop",
    "survival_shaping": "potential",
    "joker_move": "joker_move",
    "ante1_chip_tempo": "hand_quality",
}


def outcome_value(
    *,
    won: bool,
    ante: int,
    win_ante: int = 8,
    stalled: bool = False,
) -> float:
    """Return the terminal value used by supervised targets and PPO."""
    if won:
        return WIN_VALUE

    capped_ante = min(max(int(ante), 1), max(int(win_ante), 1))
    value = LOSS_BASE + ANTE_PROGRESS_VALUE * capped_ante
    value -= EARLY_DEATH_PENALTIES.get(capped_ante, 0.0)
    if stalled:
        value -= STALL_EXTRA_PENALTY
    return value


def _build_value_estimate(info: Mapping[str, Any]) -> BuildValueEstimate | None:
    """Return a cached build estimate or derive one from a complete snapshot."""
    for key in ("build_value_estimate", "_build_value_estimate"):
        captured = info.get(key)
        if isinstance(captured, BuildValueEstimate):
            return captured

    hand_details = info.get("hand_details")
    deck_stats = info.get("deck_stats")
    if "joker_details" not in info or not isinstance(hand_details, Mapping):
        return None
    if not isinstance(deck_stats, Mapping):
        return None
    cards = deck_stats.get("cards") or deck_stats.get("card_descriptors")
    if not isinstance(cards, Sequence) or isinstance(cards, (str, bytes)) or not cards:
        return None
    if float(info.get("blind_target", 0) or 0) <= 0.0:
        return None
    if info.get("hands_available") is None and info.get("hands_left") is None:
        return None

    try:
        return estimate_build_value(info)
    except (OverflowError, TypeError, ValueError):
        return None


def _early_chip_marginal_multiplier(score_ratio: float, ante: int) -> float:
    marginal_strength = max(0.0, min((score_ratio - 1.0) / 0.5, 1.0))
    if ante <= 3:
        phase_strength = 1.0
    elif ante == 4:
        phase_strength = 0.5
    else:
        phase_strength = 0.0
    return 1.0 + phase_strength * marginal_strength


def _is_active_scaler(joker: Mapping[str, Any]) -> bool:
    key = str(joker.get("key") or "")
    if key not in _SCALING_JOKER_KEYS or joker.get("debuffed"):
        return False
    if joker.get("perishable") and joker.get("perish_tally") is not None:
        return int(joker.get("perish_tally") or 0) > 0
    return True


def _visible_scaling_opportunity(
    info: Mapping[str, Any],
    joker: Mapping[str, Any],
    estimate: BuildValueEstimate,
) -> float:
    """Return observable near-term trigger support for a scaling joker."""
    key = str(joker.get("key") or "")
    shop_cards = tuple(card for card in (info.get("shop_cards") or ()) if isinstance(card, Mapping))
    consumables = tuple(card for card in (info.get("consumable_details") or ()) if isinstance(card, Mapping))
    pack_name = str(info.get("pack_state_name") or info.get("pack_booster_key") or "").lower()

    if key == "j_hologram":
        visible_standard = "standard" in pack_name or any(
            "standard"
            in " ".join(
                str(card.get(field) or "") for field in ("key", "name", "pack_state_name", "pack_booster_key")
            ).lower()
            for card in shop_cards
        )
        return 1.0 if visible_standard else 0.0

    if key == "j_constellation":
        planet_count = sum(card.get("set") == "Planet" for card in (*shop_cards, *consumables))
        if "planet" in pack_name or "celestial" in pack_name:
            planet_count += 1
        return min(planet_count / 2.0, 1.0)

    if key == "j_campfire":
        sellable = len(consumables) + sum(card.get("set") in {"Tarot", "Planet"} for card in shop_cards)
        return min(sellable / 3.0, 1.0)

    deck = info.get("deck_stats")
    deck = deck if isinstance(deck, Mapping) else {}
    deck_size = max(int(deck.get("size", 0) or 0), 1)
    if key == "j_vampire":
        enhanced = sum(int(value or 0) for value in (deck.get("enhancement_counts") or {}).values())
        return min(2.0 * enhanced / deck_size, 1.0)
    if key == "j_steel_joker":
        return min(3.0 * int(deck.get("steel_count", 0) or 0) / deck_size, 1.0)
    if key == "j_glass":
        return min(3.0 * int(deck.get("glass_count", 0) or 0) / deck_size, 1.0)
    if key == "j_wee":
        rank_counts = deck.get("rank_counts") or {}
        return min(4.0 * int(rank_counts.get("2", 0) or 0) / deck_size, 1.0)
    if key == "j_castle":
        target = info.get("castle_card") or {}
        suit = str(target.get("suit") or "") if isinstance(target, Mapping) else ""
        suit_counts = deck.get("suit_counts") or {}
        return min(2.0 * int(suit_counts.get(suit, 0) or 0) / deck_size, 1.0) if suit else 0.0
    if key == "j_runner":
        return 1.0 if estimate.representative_hand_type in {"Straight", "Straight Flush"} else 0.0
    if key == "j_square":
        return 1.0 if estimate.representative_hand_type in {"Four of a Kind", "Two Pair"} else 0.0
    if key in {"j_green_joker", "j_ride_the_bus", "j_supernova"}:
        hands = int(info.get("hands_available") or info.get("hands_left") or 0)
        return min(hands / 4.0, 1.0)
    return 0.0


def _build_potential_components(
    info: Mapping[str, Any],
    config: RewardConfig,
) -> dict[str, float]:
    names = (
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
    if not config.enable_score_build_potential:
        return dict.fromkeys(names, 0.0)
    estimate = _build_value_estimate(info)
    if estimate is None:
        return dict.fromkeys(names, 0.0)

    ante = max(int(info.get("ante", 1) or 1), 1)
    baseline = max(float(estimate.no_joker_baseline_score), 1.0)
    score = max(float(estimate.representative_score_per_hand), 0.0)
    score_gain = max(math.log(max(score / baseline, 1.0)), 0.0)

    chip_extra_gain = 0.0
    for marginal in estimate.joker_marginals:
        ratio = max(float(marginal.score_ratio), 1.0)
        if marginal.channels.chips <= 0.0 or ratio <= 1.0:
            continue
        multiplier = _early_chip_marginal_multiplier(ratio, ante)
        chip_extra_gain += math.log(ratio) * (multiplier - 1.0)

    try:
        strategy = estimate_strategy_value(info, win_ante=max(int(config.potential_win_ante), 2))
    except (KeyError, OverflowError, TypeError, ValueError):
        strategy = None
    if strategy is None:
        readiness_ratio = max(float(estimate.readiness_ratio), 0.0)
        readiness_gate = min(readiness_ratio, 1.0)
        quality_fraction = (1.0 - math.exp(-(score_gain + chip_extra_gain))) * readiness_gate
        saturation = max(config.potential_readiness_saturation, 1e-6)
        readiness_fraction = min(readiness_ratio / saturation, 1.0)
        economy_fraction = tarot_fraction = planet_fraction = seal_fraction = joker_search_fraction = 0.0
        pack_search_fraction = 0.0
    else:
        # Only draw-reliable plans participate. Two Pair requires a dedicated
        # synergy Joker, Full House stays absent, and kind hands require fixing.
        quality_fraction = strategy.hand_plan_quality
        readiness_fraction = strategy.readiness
        economy_fraction = strategy.economy
        tarot_fraction = strategy.tarot_option
        planet_fraction = strategy.planet_option
        seal_fraction = strategy.seals
        joker_search_fraction = strategy.joker_search
        pack_search_fraction = strategy.pack_search

    realized_quality = max(config.potential_w_build_quality, 0.0) * quality_fraction
    readiness = max(config.potential_w_readiness, 0.0) * readiness_fraction
    economy = max(config.potential_w_economy, 0.0) * economy_fraction
    tarot_option = max(config.potential_w_tarot_option, 0.0) * tarot_fraction
    planet_option = max(config.potential_w_planet_option, 0.0) * planet_fraction
    seal_value = max(config.potential_w_seals, 0.0) * seal_fraction
    joker_search = max(config.potential_w_joker_search, 0.0) * joker_search_fraction
    pack_search = max(config.potential_w_standard_pack_search, 0.0) * pack_search_fraction

    win_ante = max(int(config.potential_win_ante), 2)
    remaining_antes = max(win_ante - ante, 0)
    runway = min(remaining_antes / max(win_ante - 1, 1), 1.0)
    option_units = 0.0
    jokers = tuple(joker for joker in (info.get("joker_details") or ()) if isinstance(joker, Mapping))
    if runway > 0.0:
        for joker in jokers:
            if _is_active_scaler(joker):
                opportunity = _visible_scaling_opportunity(info, joker, estimate)
                option_units += runway * (0.25 + 0.75 * opportunity)
    option_value = max(config.potential_w_scaling_option, 0.0) * min(option_units, 1.0)

    components = {
        "realized_build_quality": realized_quality,
        "scaling_option_value": option_value,
        "readiness": readiness,
        "economy": economy,
        "tarot_option_value": tarot_option,
        "planet_option_value": planet_option,
        "seal_value": seal_value,
        "joker_search_option": joker_search,
        "standard_pack_search_option": pack_search,
    }
    build_sum = sum(components.values())
    build_cap = max(config.potential_build_cap, 0.0)
    if build_sum > build_cap and build_sum > 0.0:
        scale = build_cap / build_sum
        components = {name: value * scale for name, value in components.items()}
    return components


def state_potential_breakdown(info: dict, config: RewardConfig) -> dict[str, float]:
    """Return the bounded components of the state potential."""
    w_blind = max(config.potential_w_blind, 0.0)
    w_ante = max(config.potential_w_ante, 0.0)
    win_ante = max(config.potential_win_ante, 2)

    blind_target = max(float(info.get("blind_target", 0) or 0), 1.0)
    round_score = max(float(info.get("round_score", 0) or 0), 0.0)
    sub_phase = info.get("sub_phase")
    if bool(info.get("in_shop")) or (sub_phase is not None and str(sub_phase) != "choose_action"):
        # Between blinds, round_score still belongs to the completed blind
        # while blind_target belongs to the upcoming one.  Macro blind/ante
        # progress already records the clear; carrying chips forward here both
        # overvalues the shop state and creates a spurious penalty when the next
        # blind resets the score to zero.
        round_score = 0.0
    blind_progress = w_blind * min(round_score / blind_target, 1.0)

    ante = max(int(info.get("ante", 1) or 1), 1)
    blind_on_deck = str(info.get("blind_on_deck", "small")).lower()
    blind_index = _BLIND_INDEX.get(blind_on_deck, 0)
    macro_denom = max(win_ante - 1, 1)
    macro_fraction = max(0.0, min((ante - 1 + blind_index / 3.0) / macro_denom, 1.0))
    ante_progress = w_ante * macro_fraction

    build = _build_potential_components(info, config)
    total = blind_progress + ante_progress + sum(build.values())
    return {
        "blind_progress": blind_progress,
        "ante_progress": ante_progress,
        **build,
        "total": total,
    }


def state_potential(info: dict, config: RewardConfig) -> float:
    """Return the bounded progress and build potential Phi(s)."""
    return state_potential_breakdown(info, config)["total"]


def potential_shaping_reward(prev_info: dict, curr_info: dict, config: RewardConfig) -> float:
    """Return gamma*Phi(s') - Phi(s), with zero terminal potential."""
    phi_curr = 0.0 if curr_info.get("_potential_terminal", False) else state_potential(curr_info, config)
    return config.gamma * phi_curr - state_potential(prev_info, config)


def _survival_safety_potential(info: Mapping[str, Any], config: RewardConfig) -> float:
    """Bounded safety potential focused below the 65% clear threshold."""

    if not config.enable_score_build_potential or config.potential_w_survival_safety <= 0.0:
        return 0.0
    if info.get("_potential_terminal", False):
        return 0.0
    clear_probability = info.get("clear_probability")
    if clear_probability is None:
        try:
            clear_probability = estimate_clear_risk(info).clear_probability
        except (KeyError, OverflowError, TypeError, ValueError):
            return 0.0
    safe_probability = 1.0 - 0.35
    safety_fraction = _clip(float(clear_probability) / safe_probability, 0.0, 1.0)
    return max(float(config.potential_w_survival_safety), 0.0) * safety_fraction


def survival_shaping_reward(prev_info: dict, curr_info: dict, config: RewardConfig) -> float:
    """Potential-based credit for actions that move an unsafe state toward safety."""

    return config.gamma * _survival_safety_potential(curr_info, config) - _survival_safety_potential(
        prev_info,
        config,
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _keys(value) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


def _has_protected_fool(info: dict) -> bool:
    last_tarot_planet = str(info.get("last_tarot_planet", "") or "")
    return "c_fool" in _keys(info.get("consumable_keys")) and last_tarot_planet in _FOOL_PROTECT_TARGETS


def _apply_planet_match_rewards(
    prev_info: dict,
    curr_info: dict,
    config: RewardConfig,
    components: dict[str, float],
    *,
    win_ante: int,
) -> None:
    try:
        plans = estimate_hand_plans(prev_info)
    except (KeyError, OverflowError, TypeError, ValueError):
        plans = None
    for prefix in ("planet_use", "planet_claim"):
        if not curr_info.get(f"{prefix}_observed", False):
            continue
        hand_type = str(curr_info.get(f"{prefix}_hand_type", "") or "")
        plan = plans.for_hand(hand_type) if plans is not None else None
        if plan is not None and plan.draw_reliability >= 0.20:
            viability = 0.4 + 0.6 * min(max(plan.readiness_ratio, 0.0) / 0.8, 1.0)
            components["planet_match_bonus"] += PLANET_MATCH_BONUS * plan.draw_reliability * viability
            if plans is not None and hand_type == plans.best.hand_type:
                components["planet_played_hand_bonus"] += PLANET_PLAYED_HAND_BONUS
            continue

        # Backward-compatible fallback for old serialized diagnostics without a
        # deck snapshot. New rollouts always take the plan-aware branch above.
        if plans is None and curr_info.get(f"{prefix}_main_hand_match", False):
            components["planet_match_bonus"] += PLANET_MATCH_BONUS
            if curr_info.get(f"{prefix}_played_hand", False):
                share = float(curr_info.get(f"{prefix}_play_share", 1.0) or 0.0)
                components["planet_played_hand_bonus"] += PLANET_PLAYED_HAND_BONUS * share
            continue

        share_weight = (
            1.0 - float(curr_info.get(f"{prefix}_play_share", 0.0) or 0.0)
            if plans is None
            else 1.0 - (plan.draw_reliability if plan is not None else 0.0)
        )
        ante = int(curr_info.get("ante", 1) or 1)
        progress = max(
            _clip((ante - 1) / max(win_ante - 1, 1), 0.0, 1.0),
            PLANET_UNMATCHED_MIN_PROGRESS,
        )
        if prefix == "planet_use":
            coeff = config.planet_unmatched_use_penalty_coeff
            component = "planet_unmatched_use_penalty"
            weight = share_weight * progress
        else:
            coeff = config.planet_unmatched_claim_penalty_coeff
            component = "planet_unmatched_claim_penalty"
            best_available = bool(curr_info.get("planet_claim_best_available", False))
            weight = 0.0 if best_available and not _has_protected_fool(prev_info) else share_weight
        if coeff > 0.0 and weight > 0.0:
            components[component] -= coeff * weight


def _apply_joker_upgrade_reward(
    prev_info: dict,
    curr_info: dict,
    components: dict[str, float],
) -> None:
    if not curr_info.get("shop_bought_joker_id"):
        return
    try:
        after_build = _build_value_estimate(curr_info)
        before_risk = estimate_clear_risk(prev_info)
        after_risk = estimate_clear_risk(curr_info)
    except (KeyError, OverflowError, TypeError, ValueError):
        return
    if after_build is None:
        return
    acquired_key = str(curr_info.get("shop_bought_joker_id") or "")
    acquired = [item for item in after_build.joker_marginals if str(item.key) == acquired_key]
    confidence = max((float(item.modeled_effect_fraction) for item in acquired), default=0.0)
    if confidence < 0.75:
        return
    baseline_probability = float(
        curr_info.get("joker_upgrade_baseline_clear_probability", before_risk.clear_probability)
    )
    probability_gain = after_risk.clear_probability - baseline_probability
    if probability_gain > 0.0:
        components["joker_upgrade_bonus"] = JOKER_UPGRADE_BONUS * min(probability_gain / 0.25, 1.0)


def _apply_danger_reroll_reward(
    prev_info: dict,
    curr_info: dict,
    components: dict[str, float],
) -> None:
    """Give a small positive-only search signal when the next blind is unsafe."""

    if str(curr_info.get("action_type", "")) != "shop_reroll":
        return
    risk = estimate_clear_risk(prev_info)
    if risk.immediate_death_probability <= 0.35:
        return
    rescue = best_confident_joker_rescue(prev_info)
    if rescue is not None and rescue.clear_probability_delta >= 0.10:
        # A known rescue was already visible.  Do not punish the reroll, but do
        # not pay the blind-search bonus either.
        return
    urgency = min((risk.immediate_death_probability - 0.35) / 0.65, 1.0)
    components["danger_reroll_bonus"] = DANGER_REROLL_BONUS * urgency


def _apply_value_tarot_reward(
    prev_info: dict,
    curr_info: dict,
    components: dict[str, float],
) -> None:
    key = str(curr_info.get("consumable_use_key") or "")
    if not key and curr_info.get("pack_claim_set") == "Tarot":
        key = str(curr_info.get("pack_claim_key") or "")
    if key not in {"c_hermit", "c_temperance"}:
        return
    payout = max(float(curr_info.get("dollars", 0) or 0) - float(prev_info.get("dollars", 0) or 0), 0.0)
    if payout > 0.0:
        components["tarot_value_bonus"] = VALUE_TAROT_BONUS * min(payout / 10.0, 1.0)


def _apply_ante1_chip_tempo_reward(
    prev_info: dict,
    curr_info: dict,
    components: dict[str, float],
) -> None:
    """Apply a signed difference of the bounded ante-one score potential.

    ``Phi(score) = bonus * clamp(score / blind_target, 0, 1)``. Both scores
    use the *previous* blind target, so a play that completes the blind or the
    run is valued against the blind it actually faced. The signed difference
    telescopes within a blind: splitting a score across plays cannot create
    reward, and a Mr. Bones-style reset pays back earlier progress. Starting a
    new blind establishes a fresh zero baseline rather than clawing back the
    completed blind's score.
    """
    if int(prev_info.get("ante", 0) or 0) != 1:
        return
    action_type = str(curr_info.get("action_type", ""))
    if action_type != "play_subset":
        return
    blind_target = float(prev_info.get("blind_target", 0.0) or 0.0)
    if blind_target <= 0.0:
        return
    prev_score = float(prev_info.get("round_score", 0.0) or 0.0)
    curr_score = float(curr_info.get("round_score", 0.0) or 0.0)
    prev_progress = min(max(prev_score / blind_target, 0.0), 1.0)
    curr_progress = min(max(curr_score / blind_target, 0.0), 1.0)
    components["ante1_chip_tempo"] = ANTE1_CHIP_TEMPO_BONUS * (curr_progress - prev_progress)


def default_reward_components(
    state: RunState,
    prev_info: dict,
    curr_info: dict,
    terminated: bool,
    won: bool,
    config: RewardConfig | None = None,
) -> dict[str, float]:
    """Return the active reward broken down into named components."""
    config = config or DEFAULT_REWARD_CONFIG
    components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}
    win_ante = int(getattr(state, "win_ante", config.potential_win_ante) or config.potential_win_ante)
    active_config = replace(config, potential_win_ante=win_ante)

    if terminated or curr_info.get("stalled", False):
        terminal_info = dict(curr_info)
        terminal_info["_potential_terminal"] = True
        components["potential_shaping"] = potential_shaping_reward(prev_info, terminal_info, active_config)
        components["survival_shaping"] = survival_shaping_reward(prev_info, terminal_info, active_config)
        death_ante = int(curr_info.get("ante", state.round_resets.ante))
        components["terminal"] = outcome_value(
            won=won,
            ante=death_ante,
            win_ante=win_ante,
            stalled=bool(curr_info.get("stalled", False)),
        )
        _apply_ante1_chip_tempo_reward(prev_info, curr_info, components)
        components["ante1_chip_tempo"] *= config.dense_reward_scale
        components["total"] = sum(components.values())
        return components

    # Reordering is scored by an exact build-layout diagnostic and must not
    # also collect generic potential shaping.  A neutral or harmful move is
    # still an idle action, however: exempting every reorder from the idle
    # penalty lets PPO cycle through the many legal permutations until the
    # distant no-progress truncation, whose credit is effectively lost over a
    # long GAE horizon.
    if str(curr_info.get("action_type", "")) == "move_joker":
        move_reward = float(curr_info.get("joker_move_reward", 0.0) or 0.0)
        components["joker_move"] = move_reward
        if move_reward <= 1e-6:
            idle_streak = max(int(curr_info.get("steps_since_progress", 1)), 1)
            move_penalty = min(
                JOKER_MOVE_NON_IMPROVING_PENALTY + max(idle_streak - 1, 0) * JOKER_MOVE_REPEAT_PENALTY,
                JOKER_MOVE_PENALTY_CAP,
            )
            components["joker_move"] -= move_penalty
            idle_penalty = IDLE_PENALTY_BASE + max(idle_streak - 8, 0) * IDLE_PENALTY_RAMP
            components["idle_penalty"] = -min(idle_penalty, IDLE_PENALTY_CAP) * config.dense_reward_scale
        components["total"] = sum(components.values())
        return components

    components["potential_shaping"] = potential_shaping_reward(prev_info, curr_info, active_config)
    components["survival_shaping"] = survival_shaping_reward(prev_info, curr_info, active_config)
    _apply_ante1_chip_tempo_reward(prev_info, curr_info, components)

    if config.enable_planet_match_rewards:
        _apply_planet_match_rewards(
            prev_info,
            curr_info,
            config,
            components,
            win_ante=win_ante,
        )

    if config.enable_score_build_potential:
        _apply_value_tarot_reward(prev_info, curr_info, components)
        _apply_joker_upgrade_reward(prev_info, curr_info, components)
        _apply_danger_reroll_reward(prev_info, curr_info, components)

    if not curr_info.get("progress_made", False):
        idle_streak = max(int(curr_info.get("steps_since_progress", 1)), 1)
        idle_penalty = IDLE_PENALTY_BASE + max(idle_streak - 8, 0) * IDLE_PENALTY_RAMP
        components["idle_penalty"] = -min(idle_penalty, IDLE_PENALTY_CAP)

    dense_scale = config.dense_reward_scale
    components["idle_penalty"] *= dense_scale
    consumable_scale = dense_scale * config.consumable_reward_scale
    for name in (
        "planet_match_bonus",
        "planet_played_hand_bonus",
        "planet_unmatched_use_penalty",
        "planet_unmatched_claim_penalty",
        "tarot_value_bonus",
    ):
        components[name] *= consumable_scale
    components["joker_upgrade_bonus"] *= dense_scale
    components["danger_reroll_bonus"] *= dense_scale
    components["ante1_chip_tempo"] *= dense_scale

    components["total"] = sum(components.values())
    return components


def default_reward(
    state: RunState,
    prev_info: dict,
    curr_info: dict,
    terminated: bool,
    won: bool,
    config: RewardConfig | None = None,
) -> float:
    """Return the scalar reward for one transition."""
    return default_reward_components(state, prev_info, curr_info, terminated, won, config)["total"]
