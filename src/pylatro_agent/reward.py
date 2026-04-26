"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pylatro.models import RunState


# All reward components are multiplied by REWARD_SCALE at the exit of
# default_reward_components. The BC pretrain supervised the value head
# on targets in the ±10 range; PPO reward magnitudes of 40 / -16+ made
# the critic chase a 4x miscalibration, which showed up as a flat
# value_loss and rollouts where shaping dominated terminal signal. The
# constants below keep their "natural" units so the shaping math stays
# readable; only the final sum gets scaled.
REWARD_SCALE = 0.25

# Previous shaping (ANTE_ADVANCE_REWARD=1.5, BLIND_CLEAR=1.25,
# HANDS_LEFT_BONUS=0.1, LOSS_PENALTY_BASE=-16, LOSS_PER_UNFINISHED_ANTE=1.5)
# made early deaths net positive: at a death ante of ~3 the cumulative
# dense payout (ante_bonus + blind_clear + hands_bonus) already exceeded
# the terminal penalty, so ppo_resume5 converged to ep_reward_mean ≈ +2.6
# with a 0% win rate and no pressure to survive. The values below shrink
# the stacked-per-ante payouts and amplify the terminal penalty so the
# net raw reward stays strictly negative until roughly ante 7 while the
# per-ante gradient (ante_bonus exponent 1.5 vs linear loss penalty)
# remains monotone in favour of pushing deeper.
WIN_REWARD = 60.0
LOSS_PENALTY_BASE = -30.0
STALL_EXTRA_PENALTY = 5.0
# Per-ante penalty for every ante between death and win_ante. Pushes the
# policy to survive deeper instead of settling for a shallow-death local
# optimum where dense shaping dominates the flat loss penalty.
LOSS_PER_UNFINISHED_ANTE = 3.0

SCORE_PROGRESS_SCALE = 0.25
PRESSURE_PROGRESS_SCALE = 1.5
DISCARD_RESOURCE_WEIGHT = 0.5
BLIND_CLEAR_REWARD = 0.5
HANDS_LEFT_BONUS_SCALE = 0.05
# Bonus is super-linear in the ante just reached so deeper antes give a
# strictly steeper gradient than the per-unfinished-ante loss penalty can
# cancel out. Growth is curr_ante ** ANTE_ADVANCE_EXPONENT.
ANTE_ADVANCE_REWARD = 0.5
ANTE_ADVANCE_EXPONENT = 1.5
INTEREST_BONUS_SCALE = 0.05

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
CONSUMABLE_TARGETED_USE_REWARD = 0.1
# Flat penalty for selling jokers or consumables in the shop. The policy
# found it could cash out inventory every shop for free dollars without
# ever engaging with scaling mechanics; a small friction makes that
# pattern unprofitable while still letting legitimate sells through if
# follow-up shaping dominates.
SHOP_SELL_PENALTY = 0.05
# Flat reward for rerolling the shop. Reroll is the main engine-building
# lever (swap junk for jokers that actually scale) but costs $5+, so the
# policy avoided it in favor of buying whatever was on the shelf. Action
# is only valid when the agent can afford it, so this can't trigger when
# cash-starved.
SHOP_REROLL_REWARD = 0.08
# Penalty for skipping a Tarot pack when it contains at least one real
# deck-fixing/economy target. Standard packs are handled separately below:
# once the deck is already over 52 cards, adding random playing cards is a
# liability unless the deck is already fixed.
TAROT_SKIP_FIXING_PENALTY = 0.12
PLANET_SKIP_PENALTY = 0.08
PLANET_FOOL_OVERWRITE_PENALTY = 0.12
STANDARD_OVERFULL_CARD_BASE_PENALTY = 0.03
STANDARD_OVERFULL_CARD_EXPONENT = 0.35
STANDARD_OVERFULL_CARD_PENALTY_CAP = 0.75

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
) -> dict[str, float]:
    """Return the default reward broken down into named components."""
    components = {
        "terminal": 0.0,
        "score_progress": 0.0,
        "pressure_progress": 0.0,
        "blind_clear": 0.0,
        "hands_bonus": 0.0,
        "ante_bonus": 0.0,
        "interest_bonus": 0.0,
        "idle_penalty": 0.0,
        "consumable_targeted_use": 0.0,
        "shop_sell_penalty": 0.0,
        "shop_reroll_reward": 0.0,
        "tarot_skip_penalty": 0.0,
        "planet_skip_penalty": 0.0,
        "planet_fool_overwrite_penalty": 0.0,
        "standard_overfull_penalty": 0.0,
    }

    if terminated or curr_info.get("stalled", False):
        if won:
            components["terminal"] += WIN_REWARD
        else:
            # Keep terminal outcomes larger than the dense shaping terms so
            # PPO cannot maximize local progress while still losing every run.
            # Scale penalty by how many antes short of the win target we died
            # at, so an early death is strictly worse than pushing deeper.
            loss_penalty = LOSS_PENALTY_BASE
            death_ante = int(curr_info.get("ante", state.round_resets.ante))
            antes_unfinished = max(0, state.win_ante - death_ante)
            loss_penalty -= LOSS_PER_UNFINISHED_ANTE * antes_unfinished
            if curr_info.get("stalled", False):
                loss_penalty -= STALL_EXTRA_PENALTY
            components["terminal"] += loss_penalty
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
    if curr_progress > prev_progress:
        components["score_progress"] += SCORE_PROGRESS_SCALE * (curr_progress - prev_progress)

    prev_pressure = _blind_pressure(prev_info)
    curr_pressure = _blind_pressure(
        curr_info,
        cleared_blind=bool(curr_info.get("blind_just_beaten", False)),
    )
    if prev_pressure is not None and curr_pressure is not None:
        components["pressure_progress"] += PRESSURE_PROGRESS_SCALE * (prev_pressure - curr_pressure)

    if curr_info.get("blind_just_beaten", False):
        components["blind_clear"] += BLIND_CLEAR_REWARD
        hands_left = curr_info.get("hands_left", 0)
        components["hands_bonus"] += HANDS_LEFT_BONUS_SCALE * hands_left

    if curr_ante > prev_ante:
        components["ante_bonus"] += ANTE_ADVANCE_REWARD * (curr_ante ** ANTE_ADVANCE_EXPONENT)
        interest_tier = min(prev_info.get("dollars", 0) // 5, state.interest_cap // 5)
        components["interest_bonus"] += INTEREST_BONUS_SCALE * min(interest_tier, 5)

    action_type = curr_info.get("action_type", "")

    if action_type in ("use_consumable_hand_subset", "use_consumable_joker"):
        components["consumable_targeted_use"] += CONSUMABLE_TARGETED_USE_REWARD

    if action_type in ("shop_sell_joker", "shop_sell_consumable"):
        components["shop_sell_penalty"] -= SHOP_SELL_PENALTY

    if action_type == "shop_reroll":
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
    return default_reward_components(state, prev_info, curr_info, terminated, won)["total"]


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


def sparse_reward(
    state: RunState,
    prev_info: dict,
    curr_info: dict,
    terminated: bool,
    won: bool,
) -> float:
    """Sparse reward: only win/loss."""
    if terminated:
        return 10.0 if won else -10.0
    return 0.0
