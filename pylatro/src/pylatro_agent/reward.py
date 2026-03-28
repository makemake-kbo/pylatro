"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

from typing import Protocol

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
    """
    reward = 0.0

    if terminated:
        if won:
            reward += 10.0
        else:
            # Scale loss penalty by progress — dying at ante 3 is better than ante 1
            ante = state.round_resets.ante
            reward += -10.0 + min(ante - 1, 5) * 1.0  # -10 at ante 1, -5 at ante 6+
        return reward

    prev_ante = prev_info.get("ante", 1)
    curr_ante = state.round_resets.ante

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
