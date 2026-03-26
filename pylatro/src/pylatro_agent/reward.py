"""Reward shaping functions for the Balatro environment."""

from __future__ import annotations

from typing import Protocol

from pylatro.models import RunState


class RewardFn(Protocol):
    def __call__(
        self,
        state: RunState,
        prev_state_info: dict,
        terminated: bool,
        won: bool,
    ) -> float: ...


def default_reward(
    state: RunState,
    prev_state_info: dict,
    terminated: bool,
    won: bool,
) -> float:
    """Default reward shaping function.

    prev_state_info should contain:
        ante: int — ante before this step
        round_score: int — round_score before this step
        blind_beaten: bool — whether blind was beaten before this step
        hands_left: int — hands_left before this step
        dollars: int — dollars before this step
    """
    reward = 0.0

    # Per-step cost
    reward -= 0.001

    if terminated:
        if won:
            reward += 10.0
        else:
            reward -= 10.0
        return reward

    # Beat a blind (transition from not beaten to beaten)
    prev_beaten = prev_state_info.get("blind_beaten", False)
    curr_blind = state.round_resets.blind
    if curr_blind is not None:
        from pylatro import get_blind_amount
        from math import floor
        ante = state.round_resets.ante
        scaling = min(state.stake, 3)
        base = get_blind_amount(ante, scaling)
        mult = curr_blind.get("mult", 1)
        target = floor(base * mult)
    else:
        target = 0

    # Check if we just beat the blind via round_score exceeding target
    # We use the prev_state_info to detect transitions
    prev_ante = prev_state_info.get("ante", 1)
    curr_ante = state.round_resets.ante

    # Ante advanced (boss beaten)
    if curr_ante > prev_ante:
        reward += 0.5

    # Hands remaining bonus after beating blind
    prev_hands = prev_state_info.get("hands_left", 0)
    curr_hands = state.current_round.hands_left
    if not prev_beaten and prev_state_info.get("round_score", 0) < target:
        # Did not just beat a blind
        pass

    # Interest earned bonus
    prev_dollars = prev_state_info.get("dollars", 0)
    dollar_gain = state.dollars - prev_dollars
    if dollar_gain > 0:
        interest_tier = min(state.dollars // 5, state.interest_cap // 5)
        reward += 0.05 * interest_tier

    return reward


def sparse_reward(
    state: RunState,
    prev_state_info: dict,
    terminated: bool,
    won: bool,
) -> float:
    """Sparse reward: only win/loss."""
    if terminated:
        return 10.0 if won else -10.0
    return 0.0
