"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

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
)


# All dense reward components are multiplied by REWARD_SCALE at the exit of
# default_reward_components. Terminal outcomes are divided by REWARD_SCALE
# before that common exit path, so changing this value changes only dense
# shaping strength while preserving the supervised value-target scale for
# wins/losses.
REWARD_SCALE = 1.0

# PPO terminal targets stay close to the supervised value-head scale for losses,
# while wins get a larger positive value so rare successes survive rollout noise.
PRETRAIN_WIN_VALUE = 20.0
PRETRAIN_LOSS_BASE = -10.0
PRETRAIN_ANTE_PROGRESS_VALUE = 1.0
PRETRAIN_STALL_EXTRA_PENALTY = 2.0

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
# / CONSUMABLE_CONFIRM shaping cluster is gone — atomic actions keep
# every projection in ConsumableFlatHead receiving gradient every time
# any consumable is used — so we keep a single small pull toward
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
# terminating sooner — staying alive and playing well is the only path
# to accumulating the bonus. Negative shaping created a die-fast
# pathology in the v1 run; positive shaping flips the incentive.
HAND_SUBSET_BONUS_SCALE = 0.3
HAND_TOP1_BONUS = 0.35
HAND_TOP3_BONUS = 0.12
PLANET_MATCH_BONUS = 0.25
PLANET_PLAYED_HAND_BONUS = 0.12

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
)
REWARD_INFO_KEYS = tuple(f"reward_{name}" for name in ("total", *REWARD_COMPONENT_NAMES))

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
        # Store in raw component units because the common exit path applies
        # REWARD_SCALE to every component.
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
    if config.enable_hand_candidate_rewards:
        if action_type == "play_subset" and not curr_info.get(
            "hand_play_not_in_candidates", False
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
    if config.enable_planet_match_rewards:
        for prefix in ("planet_use", "planet_claim"):
            if not curr_info.get(f"{prefix}_observed", False):
                continue
            if curr_info.get(f"{prefix}_played_hand", False):
                components["planet_played_hand_bonus"] += PLANET_PLAYED_HAND_BONUS
            if curr_info.get(f"{prefix}_main_hand_match", False):
                components["planet_match_bonus"] += PLANET_MATCH_BONUS

    if action_type in ("shop_sell_joker", "shop_sell_consumable"):
        components["shop_sell_penalty"] -= SHOP_SELL_PENALTY

    if config.enable_shop_reroll_reward and action_type == "shop_reroll":
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

    for key in components:
        components[key] *= REWARD_SCALE
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
        blind_just_beaten: bool — whether a blind was beaten this step
        progress_made: bool — whether the environment state changed meaningfully
        steps_since_progress: int — idle streak length after the action
        action_type: ActionType — action type taken this step
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
