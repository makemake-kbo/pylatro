"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pylatro.models import RunState


# All reward components are multiplied by REWARD_SCALE at the exit of
# default_reward_components. The BC pretrain supervised the value head
# on targets in the ±10 range; PPO reward magnitudes of 40 / -16+ made
# the critic chase a 4× miscalibration, which showed up as a flat
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
