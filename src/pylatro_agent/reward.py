"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol

from .build_value import BuildValueEstimate, estimate_build_value
from .heuristic import (
    _CHIPS_PROFILE_JOKER_KEYS,
    _MULT_PROFILE_JOKER_KEYS,
    _RETRIGGER_JOKER_KEYS,
    _SCALING_JOKER_KEYS,
    _XMULT_PROFILE_JOKER_KEYS,
)
from .shop_eval import (
    BuildEval,
    evaluate_build,
    evaluate_shop_opportunity,
    interest_tiers,
    score_shop_item,
)

if TYPE_CHECKING:
    from pylatro.models import RunState


@dataclass
class RewardConfig:
    enable_hand_candidate_rewards: bool = True
    enable_planet_match_rewards: bool = True
    enable_shop_reroll_reward: bool = True
    enable_consumable_targeted_reward: bool = True
    enable_blind_clear_reward: bool = True
    enable_ante_advance_reward: bool = True
    enable_score_progress: bool = True
    enable_pressure_progress: bool = True

    # ── Strategic shop / build / economy shaping ──
    # These fire only when the step ``info`` carries the build features added by
    # shop_eval.capture_build_features (i.e. real env / training rollouts). When
    # active they replace the flat reroll/sell shaping with context-aware,
    # build-delta rewards. Unit tests that hand-build minimal info dicts fall
    # through to the legacy flat path so they stay unchanged.
    enable_shop_strategy_rewards: bool = True
    enable_economy_strategy_rewards: bool = True
    enable_joker_context_rewards: bool = True

    shop_engine_delta_coeff: float = 0.35
    shop_purchase_value_coeff: float = 0.25
    shop_bad_buy_penalty_coeff: float = 0.25
    shop_reroll_good_coeff: float = 0.08
    shop_reroll_bad_coeff: float = 0.15
    shop_leave_good_coeff: float = 0.08
    shop_leave_missed_upgrade_coeff: float = 0.25

    economy_interest_progress_coeff: float = 0.025
    economy_interest_breakpoint_coeff: float = 0.06
    economy_overspend_penalty_coeff: float = 0.05

    joker_slot_fill_coeff: float = 0.12
    joker_sell_good_coeff: float = 0.06
    joker_sell_bad_coeff: float = 0.30
    xmult_acquisition_coeff: float = 0.20
    consumable_improvement_coeff: float = 0.50

    # Per-step clamp on the aggregate strategic shop/joker/economy contribution
    # (pre dense-scale) so a single shop step cannot dominate terminal reward.
    max_single_shop_reward: float = 0.50
    max_single_shop_penalty: float = 0.50

    # Multiplier applied to all dense/local shaping components at the common
    # exit of default_reward_components. Terminal win/loss reward is NOT
    # affected by this, it is emitted on its own early-return path. Set < 1.0
    # to shrink shaping while preserving the supervised terminal value scale,
    # so the policy optimizes *winning* rather than farming bounded shaping.
    dense_reward_scale: float = 1.0

    # Per-group dense scales layered on top of dense_reward_scale. Default 1.0
    # keeps reward identical to the pre-split behavior; lower local_hand to stop
    # already-solved hand play from drowning out strategic shop/economy signal
    # (the plan's "Option B" recommends ~0.10 local / 1.0 shop / 0.5 economy).
    local_hand_reward_scale: float = 1.0
    progression_reward_scale: float = 1.0
    shop_strategy_reward_scale: float = 1.0
    joker_strategy_reward_scale: float = 1.0
    economy_reward_scale: float = 1.0
    consumable_reward_scale: float = 1.0

    # Optional per-step penalties for engaging with planets that do NOT match an
    # already-played hand type. Default 0.0 (disabled) so the next run's first
    # intervention is lowering consumable shaping, not adding new penalties.
    planet_unmatched_use_penalty_coeff: float = 0.0
    planet_unmatched_claim_penalty_coeff: float = 0.0

    # ── Build-curve shaping ──
    # In the legacy reward this keeps the static acquisition/removal component.
    # In V2 it is a compatibility switch for the contextual build potential below;
    # the static component stays zero so build changes are paid exactly once.
    enable_build_curve_rewards: bool = False
    build_curve_coeff: float = 0.35

    # ── Potential-based shaping (Phase 2) ──
    # When enabled, replaces all killed heuristic-agreement shaping with a
    # single potential-based term F(s,s') = gamma*Phi(s') - Phi(s), the unique
    # form that does not change the optimal policy under discounting (Ng et al.
    # 1999). Gamma is plumbed from PPOConfig so the telescope matches the
    # return the value function regresses.
    enable_potential_shaping: bool = False
    gamma: float = 0.997
    # Potential weights. Progress is bounded by w_blind + w_ante. Contextual
    # build/readiness terms share a separate 1.5-unit cap.
    potential_w_blind: float = 0.5
    potential_w_ante: float = 2.0
    potential_win_ante: int = 8
    potential_w_build_quality: float = 0.88
    potential_w_readiness: float = 0.40
    potential_w_scaling_option: float = 0.22
    potential_build_cap: float = 1.5
    potential_readiness_saturation: float = 1.5


# Increment this whenever reward semantics change without a corresponding
# RewardConfig field change. The version participates in the fingerprint, so a
# strict PPO resume cannot silently restore a critic trained on older targets.
REWARD_MODEL_VERSION = 2


def reward_config_snapshot(config: RewardConfig | Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete plain-data RewardConfig snapshot used in checkpoints."""
    if isinstance(config, RewardConfig):
        return asdict(config)
    return dict(config)


def reward_config_fingerprint(
    config: RewardConfig | Mapping[str, Any],
    *,
    reward_model_version: int | None = None,
) -> str:
    """Return the canonical SHA-256 fingerprint for reward code and configuration."""
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
    """Build the reward metadata persisted in every full PPO checkpoint."""
    return {
        "reward_config": reward_config_snapshot(config),
        "reward_model_version": REWARD_MODEL_VERSION,
        "reward_fingerprint": reward_config_fingerprint(config),
    }


DEFAULT_REWARD_CONFIG = RewardConfig()
PPO_SPARSE_CONFIG = RewardConfig(
    enable_hand_candidate_rewards=False,
    enable_planet_match_rewards=False,
    enable_shop_reroll_reward=False,
    enable_consumable_targeted_reward=False,
    enable_blind_clear_reward=True,
    enable_ante_advance_reward=True,
    enable_score_progress=True,
    enable_pressure_progress=True,
    enable_shop_strategy_rewards=False,
    enable_economy_strategy_rewards=False,
    enable_joker_context_rewards=False,
)


def PPO_V2_REWARD_CONFIG(
    *,
    gamma: float = 0.997,
    win_ante: int = 8,
    planet_match_shaping: bool = False,
    planet_unmatched_use_penalty_coeff: float = 0.0,
    planet_unmatched_claim_penalty_coeff: float = 0.0,
    build_curve_shaping: bool = False,
    dense_reward_scale: float = 1.0,
    progression_reward_scale: float = 1.0,
    consumable_reward_scale: float = 1.0,
) -> RewardConfig:
    """Phase 2 reward config: potential-based shaping + terminal-dominated return.

    Kills every prescriptive heuristic-agreement component (hand bonuses, planet
    matching, shop strategy, economy) and replaces them with a single
    policy-invariant potential-based shaping term. Terminal reward uses the halved
    v2 scale so outcome is ~2/3 of achievable return magnitude. Gamma is plumbed
    from PPOConfig so the potential telescope matches the value function's return.

    ``planet_match_shaping`` re-enables just the planet-alignment component on top
    of the v2 baseline: the sparse win signal cannot credit-assign which planet to
    level, and without it agents settle into ~90% unmatched planet use. The
    bounded bonuses (PLANET_MATCH_BONUS main-hand match, PLANET_PLAYED_HAND_BONUS
    played-hand) and optional unmatched-use/claim penalties stay small relative
    to the ~10 terminal reward.

    ``build_curve_shaping`` is the compatibility switch for contextual build
    potential. It values realized representative score, readiness, early chip
    marginals, and a capped option value for active recognized scalers. V2 does
    not emit the legacy static acquisition/removal bonus.
    """
    return RewardConfig(
        # Kill all prescriptive shaping (planet matching optionally retained ,
        # see docstring).
        enable_hand_candidate_rewards=False,
        enable_planet_match_rewards=planet_match_shaping,
        planet_unmatched_use_penalty_coeff=planet_unmatched_use_penalty_coeff,
        planet_unmatched_claim_penalty_coeff=planet_unmatched_claim_penalty_coeff,
        enable_build_curve_rewards=build_curve_shaping,
        dense_reward_scale=dense_reward_scale,
        progression_reward_scale=progression_reward_scale,
        consumable_reward_scale=consumable_reward_scale,
        enable_shop_reroll_reward=False,
        enable_consumable_targeted_reward=False,
        enable_shop_strategy_rewards=False,
        enable_economy_strategy_rewards=False,
        enable_joker_context_rewards=False,
        # Enable potential-based shaping (the only dense signal besides terminal).
        enable_potential_shaping=True,
        gamma=gamma,
        potential_w_blind=0.5,
        potential_w_ante=2.0,
        potential_win_ante=win_ante,
        potential_w_build_quality=0.88,
        potential_w_readiness=0.40,
        potential_w_scaling_option=0.22,
        potential_build_cap=1.5,
        potential_readiness_saturation=1.5,
    )



# All dense reward components are multiplied by REWARD_SCALE at the exit of
# default_reward_components. Terminal outcomes are divided by REWARD_SCALE
# before that common exit path, so changing this value changes only dense
# shaping strength while preserving the supervised value-target scale for
# wins/losses. For run-to-run dense shaping ablations prefer
# RewardConfig.dense_reward_scale (a per-config multiplier layered on top of
# this global) so terminal reward stays bit-for-bit identical.
REWARD_SCALE = 1.0  # default dense shaping multiplier

# PPO terminal targets stay close to the supervised value-head scale for losses,
# while wins get a larger positive value so rare successes survive rollout noise.
PRETRAIN_WIN_VALUE = 20.0
PRETRAIN_LOSS_BASE = -10.0
PRETRAIN_ANTE_PROGRESS_VALUE = 1.0
PRETRAIN_STALL_EXTRA_PENALTY = 2.0

# Phase 2 v2 terminal values: halved from the supervised scale so terminal
# reward (~+10 win, ~-5 to 0 loss) dominates return alongside the bounded
# potential shaping (w_blind + w_ante = 0.5 + 2.0 = 2.5). This makes the value
# head regress an O(1-10) return without normalize_returns.
V2_WIN_VALUE = 10.0
V2_LOSS_BASE = -5.0
V2_ANTE_PROGRESS_VALUE = 0.5
V2_STALL_EXTRA_PENALTY = 1.0
# The flat 0.5/ante loss slope makes early deaths disproportionately cheap
# (ante-1 death -4.5 vs ante-5 -2.5 spans just 2.0 on the 15-point win/loss
# scale), leaving a thin gradient against ante-1 wipes such as playing into
# The Hook junk-first. Steepen the bottom of the curve: deaths before ante 3
# pay an extra surcharge (ante 1: -6.5 total, ante 2: -5.0, ante 3+: unchanged).
V2_EARLY_DEATH_PENALTIES = {1: 2.0, 2: 1.0}

# Blind index mapping for the macro-progress potential.
_BLIND_INDEX = {"small": 0, "big": 1, "boss": 2}

SCORE_PROGRESS_SCALE = 0.5
PRESSURE_PROGRESS_SCALE = 1.0
DISCARD_RESOURCE_WEIGHT = 0.5
BLIND_CLEAR_REWARD = 0.8
HANDS_LEFT_BONUS_SCALE = 0.08
ANTE_ADVANCE_REWARD = 1.0
ANTE_ADVANCE_EXPONENT = 1.0
INTEREST_BONUS_SCALE = 0.08

IDLE_PENALTY_BASE = 0.001
IDLE_PENALTY_RAMP = 0.0005
IDLE_PENALTY_CAP = 0.02
# Flat reward for committing an atomic consumable use that touches a
# targeting consumable (hand subset or joker target). The cold-head
# problem that motivated the previous CONSUMABLE_TARGET / CONSUMABLE_SLOT
# / CONSUMABLE_CONFIRM shaping cluster is gone, atomic actions keep
# every projection in ConsumableFlatHead receiving gradient every time
# any consumable is used, so we keep a single small pull toward
# engaging with targeting consumables at all while the BC prior warms up.
CONSUMABLE_TARGETED_USE_REWARD = 0.12
# Flat penalty for selling jokers or consumables in the shop. The policy
# found it could cash out inventory every shop for free dollars without
# ever engaging with scaling mechanics; a small friction makes that
# pattern unprofitable while still letting legitimate sells through if
# follow-up shaping dominates.
SHOP_SELL_PENALTY = 0.08
# Flat reward for rerolling the shop. Reroll is the main engine-building
# lever (swap junk for jokers that actually scale) but costs $5+, so the
# policy avoided it in favor of buying whatever was on the shelf. Action
# is only valid when the agent can afford it, so this can't trigger when
# cash-starved.
SHOP_REROLL_REWARD = 0.12
# Penalty for skipping a Tarot pack when it contains at least one real
# deck-fixing/economy target. Standard packs are handled separately below:
# once the deck is already over 52 cards, adding random playing cards is a
# liability unless the deck is already fixed.
TAROT_SKIP_FIXING_PENALTY = 0.2
PLANET_SKIP_PENALTY = 0.15
PLANET_FOOL_OVERWRITE_PENALTY = 0.2
STANDARD_OVERFULL_CARD_BASE_PENALTY = 0.03
STANDARD_OVERFULL_CARD_EXPONENT = 0.35
STANDARD_OVERFULL_CARD_PENALTY_CAP = 0.75

# Per-step shaping bonuses driven by env action diagnostics. Asymmetric
# (positive-only) so the agent can't reduce expected reward by
# terminating sooner, staying alive and playing well is the only path
# to accumulating the bonus. Negative shaping created a die-fast
# pathology in the v1 run; positive shaping flips the incentive.
HAND_SUBSET_BONUS_SCALE = 0.3
HAND_TOP1_BONUS = 0.35
HAND_TOP3_BONUS = 0.12
PLANET_MATCH_BONUS = 0.5
PLANET_PLAYED_HAND_BONUS = 0.25
# Floor on the ante-progress discount applied to unmatched planet penalties.
# The pre-floor discount made unmatched claims/uses literally free before
# ante 2, and the sil2/env32 leg farmed exactly that seam: ~70% of planet
# engagement unmatched, Pluto ~1/3 of claims vs <1% High Card played. Early
# leniency for pivot stockpiling survives, but nothing is ever free.
PLANET_UNMATCHED_MIN_PROGRESS = 0.25

REWARD_COMPONENT_NAMES = (
    "terminal",
    "score_progress",
    "pressure_progress",
    "blind_clear",
    "hands_bonus",
    "ante_bonus",
    "interest_bonus",
    "idle_penalty",
    "consumable_targeted_use",
    "shop_sell_penalty",
    "shop_reroll_reward",
    "tarot_skip_penalty",
    "planet_skip_penalty",
    "planet_fool_overwrite_penalty",
    "standard_overfull_penalty",
    "hand_subset_bonus",
    "hand_top1_bonus",
    "hand_top3_bonus",
    "planet_match_bonus",
    "planet_played_hand_bonus",
    "planet_unmatched_use_penalty",
    "planet_unmatched_claim_penalty",
    # Strategic shop / build / economy shaping (gated on enriched info).
    "shop_engine_delta",
    "shop_purchase_value",
    "shop_bad_buy_penalty",
    "shop_reroll_good",
    "shop_reroll_bad",
    "shop_leave_good",
    "shop_leave_missed_upgrade_penalty",
    "economy_interest_progress",
    "economy_interest_breakpoint",
    "economy_overspend_penalty",
    "joker_slot_fill",
    "joker_sell_good",
    "joker_sell_bad",
    "xmult_acquisition",
    "consumable_improvement",
    "build_curve_bonus",
    # Phase 2: potential-based shaping (policy-invariant; replaces killed
    # heuristic-agreement components when enable_potential_shaping is set).
    "potential_shaping",
)
REWARD_INFO_KEYS = tuple(f"reward_{name}" for name in ("total", *REWARD_COMPONENT_NAMES))

# Component → dense-scale group. Components not listed default to the
# "progression" group (scale 1.0), preserving legacy behavior. The terminal
# component is scaled on its own early-return path and is intentionally absent.
_COMPONENT_GROUP = {
    "score_progress": "progression",
    "pressure_progress": "progression",
    "blind_clear": "progression",
    "hands_bonus": "progression",
    "ante_bonus": "progression",
    "interest_bonus": "progression",
    "idle_penalty": "progression",
    "consumable_targeted_use": "consumable",
    "planet_match_bonus": "consumable",
    "planet_played_hand_bonus": "consumable",
    "planet_unmatched_use_penalty": "consumable",
    "planet_unmatched_claim_penalty": "consumable",
    "consumable_improvement": "consumable",
    "hand_subset_bonus": "local_hand",
    "hand_top1_bonus": "local_hand",
    "hand_top3_bonus": "local_hand",
    "shop_sell_penalty": "shop_strategy",
    "shop_reroll_reward": "shop_strategy",
    "tarot_skip_penalty": "shop_strategy",
    "planet_skip_penalty": "shop_strategy",
    "planet_fool_overwrite_penalty": "shop_strategy",
    "standard_overfull_penalty": "shop_strategy",
    "shop_engine_delta": "shop_strategy",
    "shop_purchase_value": "shop_strategy",
    "shop_bad_buy_penalty": "shop_strategy",
    "shop_reroll_good": "shop_strategy",
    "shop_reroll_bad": "shop_strategy",
    "shop_leave_good": "shop_strategy",
    "shop_leave_missed_upgrade_penalty": "shop_strategy",
    "economy_interest_progress": "economy",
    "economy_interest_breakpoint": "economy",
    "economy_overspend_penalty": "economy",
    "joker_slot_fill": "joker_strategy",
    "joker_sell_good": "joker_strategy",
    "joker_sell_bad": "joker_strategy",
    "xmult_acquisition": "joker_strategy",
    "build_curve_bonus": "joker_strategy",
}

# Strategic components clamped together per step (pre dense-scale).
_STRATEGIC_SHOP_COMPONENTS = (
    "shop_engine_delta",
    "shop_purchase_value",
    "shop_bad_buy_penalty",
    "shop_reroll_good",
    "shop_reroll_bad",
    "shop_leave_good",
    "shop_leave_missed_upgrade_penalty",
    "economy_interest_progress",
    "economy_interest_breakpoint",
    "economy_overspend_penalty",
    "joker_slot_fill",
    "joker_sell_good",
    "joker_sell_bad",
    "xmult_acquisition",
    "consumable_improvement",
)

# Per-tier interest opportunity cost expressed in reward units. Matches the
# accounting used inside shop_eval so build/shop value and lost-interest are
# comparable. One interest tier ≈ this much shaped reward.
_INTEREST_TIER_VALUE = 0.05
# A "good" joker buy must clear this raw build-value threshold.
_GOOD_BUY_THRESHOLD = 0.20

_FIXED_DECK_SIGNATURE_SHARE = 0.70
_FIXED_DECK_MIN_SIGNATURE_COUNT = 8
_TAROT_FIXING_TARGETS = {
    "c_death",
    "c_hanged_man",
    "c_strength",
    "c_chariot",
    "c_justice",
    "c_magician",
    "c_lovers",
    "c_star",
    "c_moon",
    "c_sun",
    "c_world",
    "c_tower",
}
_TAROT_ECONOMY_TARGETS = {"c_hermit", "c_temperance"}
_FOOL_PROTECT_TARGETS = {"c_death", "c_hermit", "c_temperance"}


class RewardFn(Protocol):
    def __call__(
        self,
        state: RunState,
        prev_info: dict,
        curr_info: dict,
        terminated: bool,
        won: bool,
    ) -> float: ...


def pretraining_outcome_value(
    *,
    won: bool,
    ante: int,
    win_ante: int = 8,
    stalled: bool = False,
) -> float:
    """Terminal value target used by both supervised fallback and PPO rewards."""
    if won:
        return PRETRAIN_WIN_VALUE

    capped_ante = min(max(int(ante), 1), max(int(win_ante), 1))
    value = PRETRAIN_LOSS_BASE + PRETRAIN_ANTE_PROGRESS_VALUE * capped_ante
    if stalled:
        value -= PRETRAIN_STALL_EXTRA_PENALTY
    return value


def v2_outcome_value(
    *,
    won: bool,
    ante: int,
    win_ante: int = 8,
    stalled: bool = False,
) -> float:
    """Phase 2 v2 terminal value: halved scale so outcome dominates return."""
    if won:
        return V2_WIN_VALUE

    capped_ante = min(max(int(ante), 1), max(int(win_ante), 1))
    value = V2_LOSS_BASE + V2_ANTE_PROGRESS_VALUE * capped_ante
    value -= V2_EARLY_DEATH_PENALTIES.get(capped_ante, 0.0)
    if stalled:
        value -= V2_STALL_EXTRA_PENALTY
    return value


def _build_value_estimate(info: Mapping[str, Any]) -> BuildValueEstimate | None:
    """Return a captured estimate or build one from a complete snapshot."""
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
    """Early-game multiplier for a chip joker's leave-one-out score gain."""
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
    """Observable near-term trigger support for a recognized scaling joker."""
    key = str(joker.get("key") or "")
    shop_cards = tuple(card for card in (info.get("shop_cards") or ()) if isinstance(card, Mapping))
    consumables = tuple(
        card for card in (info.get("consumable_details") or ()) if isinstance(card, Mapping)
    )
    pack_name = str(info.get("pack_state_name") or info.get("pack_booster_key") or "").lower()

    if key == "j_hologram":
        visible_standard = "standard" in pack_name or any(
            "standard"
            in " ".join(
                str(card.get(field) or "")
                for field in ("key", "name", "pack_state_name", "pack_booster_key")
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
) -> tuple[float, float, float]:
    if not config.enable_build_curve_rewards:
        return 0.0, 0.0, 0.0
    estimate = _build_value_estimate(info)
    if estimate is None:
        return 0.0, 0.0, 0.0

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

    readiness_ratio = max(float(estimate.readiness_ratio), 0.0)
    readiness_gate = min(readiness_ratio, 1.0)
    quality_fraction = (1.0 - math.exp(-(score_gain + chip_extra_gain))) * readiness_gate
    realized_quality = max(config.potential_w_build_quality, 0.0) * quality_fraction

    saturation = max(config.potential_readiness_saturation, 1e-6)
    readiness_fraction = min(readiness_ratio / saturation, 1.0)
    readiness = max(config.potential_w_readiness, 0.0) * readiness_fraction

    win_ante = max(int(config.potential_win_ante), 2)
    remaining_antes = max(win_ante - ante, 0)
    runway = min(remaining_antes / max(win_ante - 1, 1), 1.0)
    option_units = 0.0
    jokers = tuple(joker for joker in (info.get("joker_details") or ()) if isinstance(joker, Mapping))
    if runway > 0.0:
        for joker in jokers:
            if not _is_active_scaler(joker):
                continue
            opportunity = _visible_scaling_opportunity(info, joker, estimate)
            option_units += runway * (0.25 + 0.75 * opportunity)
    option_value = max(config.potential_w_scaling_option, 0.0) * min(option_units, 1.0)

    build_sum = realized_quality + option_value + readiness
    build_cap = max(config.potential_build_cap, 0.0)
    if build_sum > build_cap and build_sum > 0.0:
        scale = build_cap / build_sum
        realized_quality *= scale
        option_value *= scale
        readiness *= scale
    return realized_quality, option_value, readiness


def state_potential_breakdown(info: dict, config: RewardConfig) -> dict[str, float]:
    """Return the bounded components of the V2 state potential."""
    w_blind = max(config.potential_w_blind, 0.0)
    w_ante = max(config.potential_w_ante, 0.0)
    win_ante = max(config.potential_win_ante, 2)

    blind_target = max(float(info.get("blind_target", 0) or 0), 1.0)
    round_score = max(float(info.get("round_score", 0) or 0), 0.0)
    blind_progress = w_blind * min(round_score / blind_target, 1.0)

    ante = max(int(info.get("ante", 1) or 1), 1)
    blind_on_deck = str(info.get("blind_on_deck", "small")).lower()
    blind_index = _BLIND_INDEX.get(blind_on_deck, 0)
    macro_denom = max(win_ante - 1, 1)
    macro_fraction = max(0.0, min((ante - 1 + blind_index / 3.0) / macro_denom, 1.0))
    ante_progress = w_ante * macro_fraction

    realized_quality, option_value, readiness = _build_potential_components(info, config)
    total = blind_progress + ante_progress + realized_quality + option_value + readiness
    return {
        "blind_progress": blind_progress,
        "ante_progress": ante_progress,
        "realized_build_quality": realized_quality,
        "scaling_option_value": option_value,
        "readiness": readiness,
        "total": total,
    }


def state_potential(info: dict, config: RewardConfig) -> float:
    """Bounded progress and build potential Phi(s)."""
    return state_potential_breakdown(info, config)["total"]


def potential_shaping_reward(prev_info: dict, curr_info: dict, config: RewardConfig) -> float:
    """Potential-based shaping: F(s,s') = gamma*Phi(s') - Phi(s).

    This is the unique form guaranteed not to change the optimal policy under
    discounting (Ng, Harada, Russell 1999). The gamma is taken from the live
    config (plumbed from PPOConfig) so the telescope matches the return the
    value function regresses. At terminal states Phi(s')=0 so the accumulated
    potential is paid back (standard treatment).
    """
    gamma = config.gamma
    phi_curr = state_potential(curr_info, config)
    # Terminal states have Phi(s') = 0: detect via terminated/stalled flags the
    # caller sets on curr_info, or by the blind being cleared with no further
    # progress available.
    if curr_info.get("_potential_terminal", False):
        phi_curr = 0.0
    phi_prev = state_potential(prev_info, config)
    return gamma * phi_curr - phi_prev


def _blind_pressure(info: dict, *, cleared_blind: bool = False) -> float | None:
    """Return a normalized measure of how hard the current blind is to finish.

    Lower is better. The metric compares the fraction of blind score still
    needed against the number of scoring resources left. This gives PPO a
    signal for whether a discard or weak hand improved the situation, instead
    of only paying for raw score deltas after the fact.
    """
    if cleared_blind:
        return 0.0

    blind_target = float(info.get("blind_target", 0))
    if blind_target <= 0.0:
        return None

    phase = info.get("phase", "")
    if str(phase) != "hand_play":
        return None

    hands_left = info.get("hands_left")
    discards_left = info.get("discards_left")
    if hands_left is None or discards_left is None:
        return None

    remaining_fraction = max(blind_target - float(info.get("round_score", 0)), 0.0) / blind_target
    effective_resources = max(float(hands_left) + DISCARD_RESOURCE_WEIGHT * float(discards_left), 1.0)
    return remaining_fraction / effective_resources


# ───────────────────────── strategic shop shaping ─────────────────────────


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _has_build_info(info: dict) -> bool:
    return "joker_details" in info


def _bought_shop_card(prev_info: dict, curr_info: dict) -> dict | None:
    """Map a shop_buy action_index onto the prev-step shop_cards entry."""
    action_index = curr_info.get("action_index")
    try:
        index = int(action_index)
    except (TypeError, ValueError):
        return None
    cards = tuple(prev_info.get("shop_cards", ()))
    if 0 <= index < len(cards) and isinstance(cards[index], dict):
        return cards[index]
    return None


def _emergency(build: BuildEval) -> bool:
    return build.survival_margin < 1.0 or not build.has_scoring_joker


def _reward_shop_buy(
    prev_info: dict,
    curr_info: dict,
    prev_build: BuildEval,
    curr_build: BuildEval,
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    card = _bought_shop_card(prev_info, curr_info)
    if card is None:
        return
    cset = card.get("set", "")
    value = score_shop_item(prev_info, card, prev_build)

    dollars = int(prev_info.get("dollars", 0) or 0)
    cap = prev_build.interest_cap_cash
    cost = int(card.get("cost", 0) or 0)
    lost_tiers = interest_tiers(dollars, cap) - interest_tiers(dollars - cost, cap)
    opp_cost = lost_tiers * (_INTEREST_TIER_VALUE * (0.25 if _emergency(prev_build) else 1.0))
    net = value - opp_cost

    if cset == "Joker":
        if value > _GOOD_BUY_THRESHOLD:
            components["shop_purchase_value"] += config.shop_purchase_value_coeff * _clip(net, 0.0, 1.0)
        else:
            components["shop_bad_buy_penalty"] -= config.shop_bad_buy_penalty_coeff * _clip(-net, 0.0, 1.0)
        if prev_build.joker_slots_left > 0 and curr_build.joker_slots_used > prev_build.joker_slots_used:
            components["joker_slot_fill"] += config.joker_slot_fill_coeff
        if not prev_build.has_xmult_joker and curr_build.has_xmult_joker:
            components["xmult_acquisition"] += config.xmult_acquisition_coeff
    elif cset in ("Tarot", "Planet", "Spectral"):
        components["consumable_improvement"] += config.consumable_improvement_coeff * _clip(value, -1.0, 1.0)


def _reward_shop_reroll(
    prev_info: dict,
    prev_build: BuildEval,
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    opp = evaluate_shop_opportunity(prev_info, prev_build)
    if opp.has_critical_upgrade:
        components["shop_reroll_bad"] -= config.shop_reroll_bad_coeff
        return
    if opp.has_affordable_upgrade and opp.best_visible_net_value > _GOOD_BUY_THRESHOLD:
        components["shop_reroll_bad"] -= config.shop_reroll_bad_coeff * _clip(opp.best_visible_net_value, 0.0, 1.0)
        return

    ante = int(prev_info.get("ante", 1) or 1)
    emergency_hunt = _emergency(prev_build) or (not prev_build.has_xmult_joker and ante >= 3)
    if opp.can_reroll_above_interest_cap and opp.reroll_desirable:
        components["shop_reroll_good"] += config.shop_reroll_good_coeff
    elif emergency_hunt and opp.reroll_desirable:
        components["shop_reroll_good"] += config.shop_reroll_good_coeff * 0.5
    else:
        components["shop_reroll_bad"] -= config.shop_reroll_bad_coeff * (1 + opp.lost_interest_tiers_if_reroll)


def _reward_shop_leave(
    prev_info: dict,
    prev_build: BuildEval,
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    opp = evaluate_shop_opportunity(prev_info, prev_build)
    if opp.has_critical_upgrade:
        components["shop_leave_missed_upgrade_penalty"] -= config.shop_leave_missed_upgrade_coeff * 1.6
        return
    if opp.has_affordable_upgrade and opp.best_visible_net_value > _GOOD_BUY_THRESHOLD:
        components["shop_leave_missed_upgrade_penalty"] -= config.shop_leave_missed_upgrade_coeff * _clip(
            opp.best_visible_net_value, 0.0, 1.0
        )
        return
    dollars = int(prev_info.get("dollars", 0) or 0)
    reroll_cost = int(prev_info.get("reroll_cost", 0) or 0)
    if opp.reroll_desirable and dollars - reroll_cost >= prev_build.interest_cap_cash:
        components["shop_leave_missed_upgrade_penalty"] -= config.shop_leave_missed_upgrade_coeff * 0.6
        return
    if dollars >= prev_build.interest_cap_cash:
        components["shop_leave_good"] += config.shop_leave_good_coeff
    if prev_build.survival_margin >= 1.5:
        components["shop_leave_good"] += config.shop_leave_good_coeff * 0.5


def _reward_joker_sell(
    prev_info: dict,
    curr_info: dict,
    prev_build: BuildEval,
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    # The sold joker occupied curr_info's action_index slot in the prev build.
    prev_jokers = list(prev_info.get("joker_details", ()))
    try:
        idx = int(curr_info.get("action_index"))
    except (TypeError, ValueError):
        idx = -1
    sold = prev_jokers[idx] if 0 <= idx < len(prev_jokers) else None
    opp = evaluate_shop_opportunity(prev_info, prev_build)
    if sold is None:
        components["joker_sell_bad"] -= config.joker_sell_bad_coeff * 0.25
        return
    # Selling a scaling / xmult engine is bad; selling a weak joker to fund a
    # strictly better visible upgrade is fine.
    if sold.get("is_scaling") or sold.get("x_mult", 1.0) > 1.0:
        components["joker_sell_bad"] -= config.joker_sell_bad_coeff
    elif opp.best_visible_net_value > _GOOD_BUY_THRESHOLD:
        components["joker_sell_good"] += config.joker_sell_good_coeff
    else:
        components["joker_sell_bad"] -= config.joker_sell_bad_coeff * 0.25


def _reward_economy(
    prev_info: dict,
    curr_info: dict,
    prev_build: BuildEval,
    curr_build: BuildEval,
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    cap = prev_build.interest_cap_cash
    prev_d = int(prev_info.get("dollars", 0) or 0)
    curr_d = int(curr_info.get("dollars", 0) or 0)
    prev_tiers = interest_tiers(prev_d, cap)
    curr_tiers = interest_tiers(curr_d, cap)
    tier_delta = curr_tiers - prev_tiers

    if tier_delta > 0:
        gate = 1.0 if (curr_build.has_scoring_joker and curr_build.survival_margin >= 1.0) else 0.25
        components["economy_interest_progress"] += config.economy_interest_progress_coeff * tier_delta * gate
        if curr_d >= cap and prev_d < cap:
            components["economy_interest_breakpoint"] += config.economy_interest_breakpoint_coeff
    elif tier_delta < 0:
        lost = -tier_delta
        # Justified if the spend bought real build power or we were in trouble.
        purchase_value = max(0.0, curr_build.total - prev_build.total)
        justified = purchase_value > lost * _INTEREST_TIER_VALUE or _emergency(prev_build)
        if not justified:
            components["economy_overspend_penalty"] -= config.economy_overspend_penalty_coeff * lost


def _apply_contextual_build_delta(
    prev_info: dict,
    curr_info: dict,
    config: RewardConfig,
    components: dict[str, float],
) -> tuple[BuildEval, BuildEval]:
    """Apply the pure pre/post build-power delta for any build-mutating action."""
    prev_build = evaluate_build(prev_info)
    curr_build = evaluate_build(curr_info)
    delta = curr_build.total - prev_build.total
    if abs(delta) > 1e-6:
        components["shop_engine_delta"] += config.shop_engine_delta_coeff * _clip(
            delta, -1.0, 1.0
        )
    return prev_build, curr_build


def _clamp_strategic_components(
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    """Clamp aggregate contextual contributions while preserving component logs."""
    pos = sum(max(0.0, components[k]) for k in _STRATEGIC_SHOP_COMPONENTS)
    neg = -sum(min(0.0, components[k]) for k in _STRATEGIC_SHOP_COMPONENTS)
    if pos > config.max_single_shop_reward and pos > 0:
        factor = config.max_single_shop_reward / pos
        for key in _STRATEGIC_SHOP_COMPONENTS:
            if components[key] > 0:
                components[key] *= factor
    if neg > config.max_single_shop_penalty and neg > 0:
        factor = config.max_single_shop_penalty / neg
        for key in _STRATEGIC_SHOP_COMPONENTS:
            if components[key] < 0:
                components[key] *= factor


def _apply_strategic_shop_rewards(
    prev_info: dict,
    curr_info: dict,
    action_type: str,
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    """Context-aware shop/build/economy shaping; replaces flat reroll/sell.

    Active only when both infos carry build features (real env / training).
    Caps the aggregate strategic contribution so a single shop step cannot
    dominate the terminal win/loss reward.
    """
    prev_build, curr_build = _apply_contextual_build_delta(
        prev_info, curr_info, config, components
    )

    if action_type == "shop_buy":
        _reward_shop_buy(prev_info, curr_info, prev_build, curr_build, config, components)
    elif action_type == "shop_reroll":
        _reward_shop_reroll(prev_info, prev_build, config, components)
    elif action_type == "shop_leave":
        _reward_shop_leave(prev_info, prev_build, config, components)
    elif action_type == "shop_sell_joker" and config.enable_joker_context_rewards:
        _reward_joker_sell(prev_info, curr_info, prev_build, config, components)

    if config.enable_economy_strategy_rewards:
        _reward_economy(prev_info, curr_info, prev_build, curr_build, config, components)

    _clamp_strategic_components(config, components)


def _apply_planet_match_rewards(
    prev_info: dict,
    curr_info: dict,
    config: RewardConfig,
    components: dict[str, float],
    win_ante: int | None = None,
) -> None:
    """Add planet-alignment bonuses/penalties for this step into ``components``.

    Shared by the legacy dense path and the v2 potential-shaping path (where
    ``planet_match_shaping`` re-enables just this component). ``prev_info``
    carries the pre-action consumable inventory for the Fool-protection check
    on claim exemptions.
    """
    effective_win_ante = max(
        int(win_ante if win_ante is not None else config.potential_win_ante), 2
    )
    for prefix in ("planet_use", "planet_claim"):
        if not curr_info.get(f"{prefix}_observed", False):
            continue
        if curr_info.get(f"{prefix}_played_hand", False):
            # Scale by the hand's play share so "technically played once" hands
            # (High Card in nearly every run) can't farm the full bonus, the
            # update-1200..1600 run learned to claim Pluto 3x over uniform for
            # exactly this reason. Missing key (older infos/tests) keeps 1.0.
            share = float(curr_info.get(f"{prefix}_play_share", 1.0) or 0.0)
            components["planet_played_hand_bonus"] += PLANET_PLAYED_HAND_BONUS * share
        main_match = curr_info.get(f"{prefix}_main_hand_match", False)
        if main_match:
            components["planet_match_bonus"] += PLANET_MATCH_BONUS
        else:
            # Weight the penalty by how far the planet's hand is from the
            # workhorse: leveling a strong #2 hand (play share ~0.9) is nearly
            # free, leveling a never-played hand pays the full penalty. Missing
            # key (older infos/tests) keeps the original flat penalty.
            share_weight = 1.0 - float(curr_info.get(f"{prefix}_play_share", 0.0) or 0.0)
            # Play share is backward-looking and cannot see a planned pivot
            # (leveling Flush Five while still playing Bloodstone flushes), so
            # discount by ante progress: unplayed-hand claims are cheap early,
            # when stockpiling levels for a pivot is legitimate planning, and
            # pay in full near win_ante, when a still-never-played hand is
            # farming. Floored — a fully free window gets farmed (the
            # sil2/env32 leg claimed ~70% unmatched through it). Missing ante
            # (older infos/tests) keeps full weight.
            ante_raw = curr_info.get("ante")
            progress = (
                1.0
                if ante_raw is None
                else max(
                    _clip((int(ante_raw) - 1) / (effective_win_ante - 1), 0.0, 1.0),
                    PLANET_UNMATCHED_MIN_PROGRESS,
                )
            )
            if prefix == "planet_use":
                coeff = config.planet_unmatched_use_penalty_coeff
                component = "planet_unmatched_use_penalty"
                weight = share_weight * progress
            else:
                coeff = config.planet_unmatched_claim_penalty_coeff
                component = "planet_unmatched_claim_penalty"
                best_available = curr_info.get("planet_claim_best_available")
                if best_available is None:
                    # Older infos/tests without pack-ranking diagnostics.
                    weight = share_weight * progress
                elif best_available and not _has_protected_fool(prev_info):
                    # Best planet a matchless pack offered: the pack is paid
                    # for and skipping wastes it, so taking even Pluto is
                    # fine — unless a held Fool stores a protected target the
                    # claim would overwrite (then skipping is the free,
                    # unpenalized move via _should_penalize_planet_skip).
                    weight = 0.0
                else:
                    # A better-aligned planet was available (or a protected
                    # Fool gets clobbered): no pivot excuse at any ante, pay
                    # the full share-weighted penalty.
                    weight = share_weight
            if coeff > 0.0 and weight > 0.0:
                components[component] -= coeff * weight


def _build_curve_weight(joker: dict, ante: int) -> float:
    """Ante-phase fit of a joker's scoring profile, in [0, 2].

    Desired curve: chip scaling carries antes 1-3, additive mult is online by
    ante 4, xmult is the ante-6+ engine that is welcome at any earlier point.
    A joker with several profiles takes the best one (chips+mult is good early
    via chips AND good late via mult). Economy/utility jokers score 0, this
    component only shapes the scoring curve.

    Multiplicative engines (xmult, retriggers) score 2.0 — strictly above the
    1.0 ceiling for additive mult/chips. Death analysis at win_ante 6 showed a
    systematic ~0.75 score/target deficit driven by additive-only builds (only
    ~15% of ante-5/6 deaths had any xmult): additive mult scales linearly while
    blind targets scale exponentially, so the reward must prefer the xmult
    engine over another +mult joker, not merely tie it.
    """
    key = str(joker.get("key", ""))
    weights = [0.0]
    x_mult = float(joker.get("x_mult", 1.0) or 1.0)
    if x_mult > 1.0 or joker.get("is_scaling_xmult", False) or key in _XMULT_PROFILE_JOKER_KEYS:
        weights.append(2.0)
    # Retriggers amplify whatever the build already scores (they multiply the
    # engine), phase-neutral, top-priority pickups: same 2.0 as xmult.
    if joker.get("is_retrigger", False) or key in _RETRIGGER_JOKER_KEYS:
        weights.append(2.0)
    if float(joker.get("t_chips", 0) or 0) > 0.0 or key in _CHIPS_PROFILE_JOKER_KEYS:
        weights.append(1.0 if ante <= 3 else (0.5 if ante <= 5 else 0.25))
    if (
        float(joker.get("mult", 0) or 0) + float(joker.get("t_mult", 0) or 0) > 0.0
        or key in _MULT_PROFILE_JOKER_KEYS
    ):
        weights.append(1.0 if ante >= 4 else 0.5)
    return max(weights)


def _joker_diff(prev_info: dict, curr_info: dict) -> tuple[list[dict], list[dict]]:
    """(acquired, removed) joker summaries between two infos (multiset by key)."""
    prev_counts: dict[str, int] = {}
    prev_by_key: dict[str, dict] = {}
    for joker in prev_info.get("joker_details", ()):
        if isinstance(joker, dict):
            key = str(joker.get("key", ""))
            prev_counts[key] = prev_counts.get(key, 0) + 1
            prev_by_key[key] = joker
    acquired: list[dict] = []
    for joker in curr_info.get("joker_details", ()):
        if not isinstance(joker, dict):
            continue
        key = str(joker.get("key", ""))
        if prev_counts.get(key, 0) > 0:
            prev_counts[key] -= 1
        else:
            acquired.append(joker)
    removed = [
        prev_by_key[key]
        for key, count in prev_counts.items()
        for _ in range(count)
    ]
    return acquired, removed


def _apply_build_curve_rewards(
    prev_info: dict,
    curr_info: dict,
    config: RewardConfig,
    components: dict[str, float],
) -> None:
    """Delta of ante-phase build fit for jokers gained/lost this step.

    Acquisitions add +coeff*weight, removals (sells, destroyed jokers) subtract
    it at the CURRENT ante's weight, so the component telescopes to the net
    build change: buy→sell→rebuy nets one bonus, not three. The acquisition-only
    version was churn-farmable, sell_joker_fraction rose ~35% over updates
    1200..1600. Upgrades stay rewarded (sell 0.25-weight chip joker for a
    2.0-weight xmult at ante 6 nets a large positive delta). Both infos must carry joker_details;
    the terminal path returns before this so deaths never pay a removal bill.
    """
    if "joker_details" not in prev_info or "joker_details" not in curr_info:
        return
    ante = max(int(curr_info.get("ante", 1) or 1), 1)
    acquired, removed = _joker_diff(prev_info, curr_info)
    for joker in acquired:
        components["build_curve_bonus"] += config.build_curve_coeff * _build_curve_weight(joker, ante)
    for joker in removed:
        components["build_curve_bonus"] -= config.build_curve_coeff * _build_curve_weight(joker, ante)


def default_reward_components(
    state: RunState,
    prev_info: dict,
    curr_info: dict,
    terminated: bool,
    won: bool,
    config: RewardConfig | None = None,
) -> dict[str, float]:
    """Return the default reward broken down into named components."""
    if config is None:
        config = DEFAULT_REWARD_CONFIG
    components = {name: 0.0 for name in REWARD_COMPONENT_NAMES}

    if terminated or curr_info.get("stalled", False):
        death_ante = int(curr_info.get("ante", state.round_resets.ante))
        win_ante = int(getattr(state, "win_ante", 8) or 8)
        # Store in raw component units because this terminal exit path applies
        # REWARD_SCALE (and only REWARD_SCALE, not dense_reward_scale) to every
        # component, leaving the terminal value at its supervised scale.
        if config.enable_potential_shaping:
            outcome_fn = v2_outcome_value
            # Use the env's win_ante for potential normalization without
            # dropping any reward fields added to the live config.
            pot_config = replace(config, potential_win_ante=win_ante)
            # Terminal potential Phi(s') = 0; pay the final shaping transition.
            curr_info_terminal = dict(curr_info)
            curr_info_terminal["_potential_terminal"] = True
            components["potential_shaping"] += potential_shaping_reward(
                prev_info, curr_info_terminal, pot_config
            )
            components["terminal"] += outcome_fn(
                won=won,
                ante=death_ante,
                win_ante=win_ante,
                stalled=bool(curr_info.get("stalled", False)),
            ) / REWARD_SCALE
        else:
            components["terminal"] += pretraining_outcome_value(
                won=won,
                ante=death_ante,
                win_ante=win_ante,
                stalled=bool(curr_info.get("stalled", False)),
            ) / REWARD_SCALE
        for key in components:
            components[key] *= REWARD_SCALE
        components["total"] = sum(components.values())
        return components

    # ── Phase 2 potential-based shaping path ──
    # When enabled, replace all killed heuristic-agreement shaping with a single
    # policy-invariant potential term F(s,s') = gamma*Phi(s') - Phi(s), plus the
    # small idle_penalty. Every prescriptive component is skipped.
    if config.enable_potential_shaping:
        win_ante = int(getattr(state, "win_ante", config.potential_win_ante) or config.potential_win_ante)
        pot_config = replace(config, potential_win_ante=win_ante)
        components["potential_shaping"] += potential_shaping_reward(prev_info, curr_info, pot_config)

        # Planet-alignment shaping is the one prescriptive component that can be
        # re-enabled on top of v2 (planet_match_shaping): the sparse win signal
        # cannot credit-assign which planet to level. Bounded per-event, scaled
        # by dense_scale below like idle_penalty.
        if config.enable_planet_match_rewards:
            _apply_planet_match_rewards(
                prev_info, curr_info, config, components, win_ante=win_ante
            )

        if not curr_info.get("progress_made", False):
            idle_streak = max(int(curr_info.get("steps_since_progress", 1)), 1)
            idle_penalty = IDLE_PENALTY_BASE + max(idle_streak - 8, 0) * IDLE_PENALTY_RAMP
            idle_penalty = min(idle_penalty, IDLE_PENALTY_CAP)
            components["idle_penalty"] -= idle_penalty

        # potential_shaping is scaled by REWARD_SCALE only, the same factor the
        # terminal exit path applies, so the telescoping sum stays exact for any
        # dense_reward_scale. Scaling the potential term differently across steps
        # would leave a per-step residual that breaks the policy-invariance
        # guarantee (the entire point of potential-based shaping).
        dense_scale = REWARD_SCALE * config.dense_reward_scale
        group_scales = {
            "progression": config.progression_reward_scale,
            "consumable": config.consumable_reward_scale,
        }
        for key in components:
            if key == "potential_shaping":
                components[key] *= REWARD_SCALE
            else:
                group = _COMPONENT_GROUP.get(key, "progression")
                components[key] *= dense_scale * group_scales.get(group, 1.0)
        components["total"] = sum(components.values())
        return components

    prev_ante = prev_info.get("ante", 1)
    curr_ante = state.round_resets.ante

    prev_score = float(prev_info.get("round_score", 0))
    curr_score = float(curr_info.get("round_score", 0))
    blind_target = max(float(prev_info.get("blind_target", curr_info.get("blind_target", 0))), 1.0)
    prev_progress = min(prev_score / blind_target, 1.0)
    curr_progress = min(curr_score / blind_target, 1.0)
    if config.enable_score_progress and curr_progress > prev_progress:
        components["score_progress"] += SCORE_PROGRESS_SCALE * (curr_progress - prev_progress)

    prev_pressure = _blind_pressure(prev_info)
    curr_pressure = _blind_pressure(
        curr_info,
        cleared_blind=bool(curr_info.get("blind_just_beaten", False)),
    )
    if config.enable_pressure_progress and prev_pressure is not None and curr_pressure is not None:
        components["pressure_progress"] += PRESSURE_PROGRESS_SCALE * (prev_pressure - curr_pressure)

    if config.enable_blind_clear_reward and curr_info.get("blind_just_beaten", False):
        components["blind_clear"] += BLIND_CLEAR_REWARD
        hands_left = curr_info.get("hands_left", 0)
        components["hands_bonus"] += HANDS_LEFT_BONUS_SCALE * hands_left

    if config.enable_ante_advance_reward and curr_ante > prev_ante:
        components["ante_bonus"] += ANTE_ADVANCE_REWARD * (curr_ante ** ANTE_ADVANCE_EXPONENT)
        interest_tier = min(prev_info.get("dollars", 0) // 5, state.interest_cap // 5)
        components["interest_bonus"] += INTEREST_BONUS_SCALE * min(interest_tier, 5)

    action_type = curr_info.get("action_type", "")

    if config.enable_consumable_targeted_reward and action_type in (
        "use_consumable_hand_subset",
        "use_consumable_joker",
    ):
        components["consumable_targeted_use"] += CONSUMABLE_TARGETED_USE_REWARD

    # Hand-subset bonus: reward in-candidates plays scaled by their value
    # ratio. Out-of-candidates plays get nothing (no penalty), so the
    # agent's only path to accumulating reward is to play well rather
    # than terminating early.
    if (
        config.enable_hand_candidate_rewards
        and action_type == "play_subset"
        and not curr_info.get("hand_play_not_in_candidates", False)
    ):
        ratio = curr_info.get("hand_play_candidate_value_ratio")
        if ratio is not None:
            components["hand_subset_bonus"] += HAND_SUBSET_BONUS_SCALE * float(ratio)
        if curr_info.get("hand_play_top1", False):
            components["hand_top1_bonus"] += HAND_TOP1_BONUS
        elif curr_info.get("hand_play_top3", False):
            components["hand_top3_bonus"] += HAND_TOP3_BONUS

    # Planet alignment bonus: rewards using or claiming planets that match
    # already-played hand types, with extra weight for the main hand. This is
    # deliberately state-conditional so random planet use does not get paid.
    # Optional penalties for *unmatched* planet engagement are layered on top
    # (default coeff 0.0 so they are off unless explicitly enabled).
    if config.enable_planet_match_rewards:
        _apply_planet_match_rewards(
            prev_info,
            curr_info,
            config,
            components,
            win_ante=int(getattr(state, "win_ante", 8) or 8),
        )

    # Build-curve bonus (off by default): joker scoring profile vs ante phase.
    if config.enable_build_curve_rewards:
        _apply_build_curve_rewards(prev_info, curr_info, config, components)

    # Strategic shop/build/economy shaping is active only when the step info
    # carries the build features (real env / training rollouts). When active it
    # supersedes the legacy flat reroll/sell shaping below.
    strategic = (
        config.enable_shop_strategy_rewards
        and _has_build_info(prev_info)
        and _has_build_info(curr_info)
    )
    if strategic and action_type in ("shop_buy", "shop_reroll", "shop_leave", "shop_sell_joker"):
        _apply_strategic_shop_rewards(prev_info, curr_info, action_type, config, components)
    elif strategic and action_type in (
        "pack_claim",
        "use_consumable_no_target",
        "use_consumable_hand_subset",
        "use_consumable_joker",
    ):
        _apply_contextual_build_delta(prev_info, curr_info, config, components)
        _clamp_strategic_components(config, components)
    elif strategic and prev_info.get("in_shop") and config.enable_economy_strategy_rewards:
        # Interest accrual / overspend on shop steps without a dedicated handler.
        _apply_strategic_shop_rewards(prev_info, curr_info, action_type, config, components)

    if action_type in ("shop_sell_joker", "shop_sell_consumable") and not (
        strategic and config.enable_joker_context_rewards and action_type == "shop_sell_joker"
    ):
        components["shop_sell_penalty"] -= SHOP_SELL_PENALTY

    if action_type == "shop_reroll" and not strategic and config.enable_shop_reroll_reward:
        components["shop_reroll_reward"] += SHOP_REROLL_REWARD

    if action_type == "pack_skip":
        if _should_penalize_tarot_skip(state, prev_info, curr_info):
            components["tarot_skip_penalty"] -= TAROT_SKIP_FIXING_PENALTY
        if _should_penalize_planet_skip(prev_info):
            components["planet_skip_penalty"] -= PLANET_SKIP_PENALTY

    if (
        action_type == "shop_buy"
        and _is_buying_planet_pack(prev_info, curr_info)
        and _should_penalize_planet_pack_open(prev_info)
    ):
        components["planet_fool_overwrite_penalty"] -= PLANET_FOOL_OVERWRITE_PENALTY

    if _is_standard_overfull_action(prev_info, curr_info):
        components["standard_overfull_penalty"] -= _standard_overfull_penalty(state)

    if not curr_info.get("progress_made", False):
        idle_streak = max(int(curr_info.get("steps_since_progress", 1)), 1)
        idle_penalty = IDLE_PENALTY_BASE + max(idle_streak - 8, 0) * IDLE_PENALTY_RAMP
        idle_penalty = min(idle_penalty, IDLE_PENALTY_CAP)
        components["idle_penalty"] -= idle_penalty

    # This branch only ever accumulates dense/local shaping, the terminal
    # component is emitted on the early-return path above and never reaches
    # here, so layering dense_reward_scale on top of REWARD_SCALE shrinks
    # shaping without touching terminal win/loss reward. Each component is
    # additionally scaled by its group multiplier (default 1.0 → identical to
    # the pre-split behavior) so local hand play can be annealed independently
    # of strategic shop/economy signal.
    group_scales = {
        "progression": config.progression_reward_scale,
        "local_hand": config.local_hand_reward_scale,
        "shop_strategy": config.shop_strategy_reward_scale,
        "joker_strategy": config.joker_strategy_reward_scale,
        "economy": config.economy_reward_scale,
        "consumable": config.consumable_reward_scale,
    }
    dense_scale = REWARD_SCALE * config.dense_reward_scale
    for key in components:
        group = _COMPONENT_GROUP.get(key, "progression")
        components[key] *= dense_scale * group_scales[group]
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
    """Default reward shaping function.

    prev_info / curr_info contain:
        ante, round_score, blind_beaten, hands_left, dollars, in_shop
    curr_info also has:
        blind_just_beaten: bool, whether a blind was beaten this step
        progress_made: bool, whether the environment state changed meaningfully
        steps_since_progress: int, idle streak length after the action
        action_type: ActionType, action type taken this step
    """
    return default_reward_components(state, prev_info, curr_info, terminated, won, config)["total"]


def _should_penalize_tarot_skip(state: RunState, prev_info: dict, curr_info: dict) -> bool:
    if _ante(curr_info, state) > 4 and "j_red_card" in _keys(prev_info.get("joker_keys", ())):
        return False

    if not _is_tarot_pack(prev_info):
        return False

    pack_cards = tuple(prev_info.get("pack_card_details", ()))
    if not pack_cards:
        return False

    deck_cards = tuple(getattr(state, "deck_cards", ()) or ())
    if not deck_cards:
        return False

    if _deck_is_fixed(deck_cards):
        return False

    return _tarot_pack_has_legit_target(pack_cards, prev_info)


def _is_standard_pack(info: dict) -> bool:
    state_name = str(info.get("pack_state_name", ""))
    if state_name == "STANDARD_PACK":
        return True
    booster_key = str(info.get("pack_booster_key", "")).lower()
    return "standard" in booster_key


def _is_tarot_pack(info: dict) -> bool:
    state_name = str(info.get("pack_state_name", ""))
    if state_name == "TAROT_PACK":
        return True
    booster_key = str(info.get("pack_booster_key", "")).lower()
    return "arcana" in booster_key


def _is_planet_pack(info: dict) -> bool:
    state_name = str(info.get("pack_state_name", ""))
    if state_name == "PLANET_PACK":
        return True
    booster_key = str(info.get("pack_booster_key", "")).lower()
    return "celestial" in booster_key


def _ante(info: dict, state: RunState) -> int:
    fallback = getattr(getattr(state, "round_resets", None), "ante", 1)
    return int(info.get("ante", fallback) or fallback)


def _keys(value) -> tuple[str, ...]:
    return tuple(str(item) for item in (value or ()))


def _deck_is_fixed(deck_cards: tuple) -> bool:
    active_cards = tuple(card for card in deck_cards if not getattr(card, "destroyed", False))
    signatures = Counter(_card_signature(card) for card in active_cards)
    if not signatures:
        return False
    signature, count = signatures.most_common(1)[0]
    share = count / max(sum(signatures.values()), 1)
    if count >= _FIXED_DECK_MIN_SIGNATURE_COUNT and share >= _FIXED_DECK_SIGNATURE_SHARE:
        rank, suit, center_key, seal = signature
        has_identity = bool(rank) and bool(suit)
        has_modifier = center_key != "c_base" or bool(seal)
        if has_identity and (has_modifier or share >= 0.85):
            return True

    total = max(len(active_cards), 1)
    fixed_dimensions = 0
    fixed_dimensions += _dominant_share(active_cards, "rank") >= 0.70
    fixed_dimensions += _dominant_share(active_cards, "suit") >= 0.85
    center_key, center_share = _dominant_nonbase(active_cards, "center_key", empty_values={"", "c_base"})
    seal, seal_share = _dominant_nonbase(active_cards, "seal", empty_values={"", None})
    center_fixed = bool(center_key) and center_share >= 0.70
    seal_fixed = bool(seal) and seal_share >= 0.70
    fixed_dimensions += center_fixed
    fixed_dimensions += seal_fixed
    return total >= _FIXED_DECK_MIN_SIGNATURE_COUNT and fixed_dimensions >= 3 and (center_fixed or seal_fixed)


def _card_signature(card) -> tuple[str, str, str, str]:
    return (
        str(getattr(card, "rank", "") or ""),
        str(getattr(card, "suit", "") or ""),
        str(getattr(card, "center_key", "c_base") or "c_base"),
        str(getattr(card, "seal", "") or ""),
    )


def _dominant_share(cards: tuple, attr: str) -> float:
    counts = Counter(str(getattr(card, attr, "") or "") for card in cards)
    counts.pop("", None)
    if not counts:
        return 0.0
    return counts.most_common(1)[0][1] / max(len(cards), 1)


def _dominant_nonbase(cards: tuple, attr: str, *, empty_values: set) -> tuple[str, float]:
    counts = Counter(str(getattr(card, attr, "") or "") for card in cards)
    for value in empty_values:
        counts.pop(str(value or ""), None)
    if not counts:
        return "", 0.0
    value, count = counts.most_common(1)[0]
    return value, count / max(len(cards), 1)


def _tarot_pack_has_legit_target(pack_cards: tuple, prev_info: dict) -> bool:
    for card in pack_cards:
        key = str(card.get("center_key", "") or "")
        if key in _TAROT_FIXING_TARGETS or key in _TAROT_ECONOMY_TARGETS:
            return True
        if key == "c_fool" and _last_tarot_planet(prev_info) in (_TAROT_FIXING_TARGETS | _TAROT_ECONOMY_TARGETS):
            return True
    return False


def _should_penalize_planet_skip(prev_info: dict) -> bool:
    return _is_planet_pack(prev_info) and not _has_protected_fool(prev_info)


def _should_penalize_planet_pack_open(prev_info: dict) -> bool:
    if not _has_protected_fool(prev_info):
        return False
    inventory = set(_keys(prev_info.get("consumable_keys", ())))
    return inventory.isdisjoint(_FOOL_PROTECT_TARGETS)


def _has_protected_fool(info: dict) -> bool:
    return "c_fool" in _keys(info.get("consumable_keys", ())) and _last_tarot_planet(info) in _FOOL_PROTECT_TARGETS


def _last_tarot_planet(info: dict) -> str:
    return str(info.get("last_tarot_planet", "") or "")


def _is_buying_planet_pack(prev_info: dict, curr_info: dict) -> bool:
    item = _bought_shop_item(prev_info, curr_info)
    return item is not None and str(item.get("pack_state_name", "")) == "PLANET_PACK"


def _is_standard_overfull_action(prev_info: dict, curr_info: dict) -> bool:
    action_type = curr_info.get("action_type", "")
    if action_type == "pack_claim":
        return _is_standard_pack(prev_info)
    if action_type == "shop_buy":
        item = _bought_shop_item(prev_info, curr_info)
        return item is not None and str(item.get("pack_state_name", "")) == "STANDARD_PACK"
    return False


def _bought_shop_item(prev_info: dict, curr_info: dict) -> dict | None:
    action_index = curr_info.get("action_index")
    if action_index is None:
        return None
    try:
        index = int(action_index)
    except (TypeError, ValueError):
        return None
    shop_items = tuple(prev_info.get("shop_item_details", ()))
    if index < 0 or index >= len(shop_items):
        return None
    item = shop_items[index]
    return item if isinstance(item, dict) else None


def _standard_overfull_penalty(state: RunState) -> float:
    deck_cards = tuple(getattr(state, "deck_cards", ()) or ())
    active_cards = tuple(card for card in deck_cards if not getattr(card, "destroyed", False))
    overfull = max(len(active_cards) - 52, 0)
    if overfull <= 0 or _deck_is_fixed(tuple(active_cards)):
        return 0.0
    penalty = STANDARD_OVERFULL_CARD_BASE_PENALTY * math.expm1(STANDARD_OVERFULL_CARD_EXPONENT * overfull)
    return min(penalty, STANDARD_OVERFULL_CARD_PENALTY_CAP)
