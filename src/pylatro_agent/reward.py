"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pylatro.models import RunState


WIN_REWARD = 15.0
LOSS_PENALTY_BASE = -10.0
LOSS_ANTE_RECOVERY = 1.0
STALL_EXTRA_PENALTY = 1.5

SCORE_PROGRESS_SCALE = 1.0
BLIND_CLEAR_REWARD = 0.5
HANDS_LEFT_BONUS_SCALE = 0.05
ANTE_ADVANCE_REWARD = 0.5
INTEREST_BONUS_SCALE = 0.05

IDLE_PENALTY_BASE = 0.001
IDLE_PENALTY_RAMP = 0.0005
IDLE_PENALTY_CAP = 0.02


class RewardFn(Protocol):
    def __call__(
        self,
        state: RunState,
        prev_info: dict,
        curr_info: dict,
        terminated: bool,
        won: bool,
    ) -> float: ...


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
        "blind_clear": 0.0,
        "hands_bonus": 0.0,
        "ante_bonus": 0.0,
        "interest_bonus": 0.0,
        "idle_penalty": 0.0,
    }

    if terminated or curr_info.get("stalled", False):
        if won:
            components["terminal"] += WIN_REWARD
        else:
            ante = state.round_resets.ante
            # Keep terminal outcomes larger than the dense shaping terms so
            # PPO cannot maximize local progress while still losing every run.
            loss_penalty = LOSS_PENALTY_BASE + min(ante - 1, 5) * LOSS_ANTE_RECOVERY
            if curr_info.get("stalled", False):
                loss_penalty -= STALL_EXTRA_PENALTY
            components["terminal"] += loss_penalty
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

    if curr_info.get("blind_just_beaten", False):
        components["blind_clear"] += BLIND_CLEAR_REWARD
        hands_left = curr_info.get("hands_left", 0)
        components["hands_bonus"] += HANDS_LEFT_BONUS_SCALE * hands_left

    if curr_ante > prev_ante:
        components["ante_bonus"] += ANTE_ADVANCE_REWARD
        interest_tier = min(prev_info.get("dollars", 0) // 5, state.interest_cap // 5)
        components["interest_bonus"] += INTEREST_BONUS_SCALE * min(interest_tier, 5)

    # Only penalize idle steps outside mandatory selection phases
    # (card toggles and consumable targeting are necessary, not idle)
    in_selection = curr_info.get("sub_phase", "") in ("select_cards", "consumable_target")
    if not curr_info.get("progress_made", False) and not in_selection:
        idle_streak = max(int(curr_info.get("steps_since_progress", 1)), 1)
        idle_penalty = IDLE_PENALTY_BASE + max(idle_streak - 8, 0) * IDLE_PENALTY_RAMP
        components["idle_penalty"] -= min(idle_penalty, IDLE_PENALTY_CAP)

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
