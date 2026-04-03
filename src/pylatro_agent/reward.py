"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pylatro.models import RunState


class RewardFn(Protocol):
    def __call__(
        self,
        state: RunState,
        prev_info: dict,
        curr_info: dict,
        terminated: bool,
        won: bool,
    ) -> float: ...


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
    reward = 0.0

    if terminated or curr_info.get("stalled", False):
        if won:
            reward += 10.0
        else:
            # Scale loss penalty by progress — dying at ante 3 is better than ante 1.
            # Stalling is an avoidable policy failure, so it must stay strictly worse
            # than taking a decisive losing line.
            ante = state.round_resets.ante
            loss_penalty = -10.0 + min(ante - 1, 5) * 1.0  # -10 at ante 1, -5 at ante 6+
            if curr_info.get("stalled", False):
                reward += loss_penalty - 2.0
            else:
                reward += loss_penalty
        return reward

    prev_ante = prev_info.get("ante", 1)
    curr_ante = state.round_resets.ante

    prev_score = float(prev_info.get("round_score", 0))
    curr_score = float(curr_info.get("round_score", 0))
    blind_target = max(float(prev_info.get("blind_target", curr_info.get("blind_target", 0))), 1.0)
    prev_progress = min(prev_score / blind_target, 1.0)
    curr_progress = min(curr_score / blind_target, 1.0)
    if curr_progress > prev_progress:
        # Reward score progress toward clearing the blind without letting overscore dominate.
        reward += 1.5 * (curr_progress - prev_progress)

    # Beat a blind → enter shop (most important intermediate signal)
    if curr_info.get("blind_just_beaten", False):
        reward += 0.5
        # Bonus for hands remaining (efficiency)
        hands_left = curr_info.get("hands_left", 0)
        reward += 0.1 * hands_left

    # Ante advanced (boss beaten) — extra bonus on top of blind-beat
    if curr_ante > prev_ante:
        reward += 1.0
        # Interest bonus
        interest_tier = min(prev_info.get("dollars", 0) // 5, state.interest_cap // 5)
        reward += 0.1 * min(interest_tier, 5)

    # Allow a short grace window for normal multi-click sequences, then ramp up.
    # The cap is intentionally large enough that surviving only by idling remains
    # much worse than advancing the run.
    if curr_info.get("progress_made", False):
        reward -= 0.001
    else:
        idle_streak = max(int(curr_info.get("steps_since_progress", 1)), 1)
        idle_penalty = 0.001 + max(idle_streak - 8, 0) * 0.0005
        reward -= min(idle_penalty, 0.02)

    return reward


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
