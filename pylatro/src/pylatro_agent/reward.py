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
        in_shop: bool — whether we were in shop phase before this step
    """
    reward = 0.0

    if terminated:
        if won:
            reward += 10.0
        else:
            reward -= 10.0
        return reward

    prev_ante = prev_state_info.get("ante", 1)
    curr_ante = state.round_resets.ante

    # Ante advanced (boss beaten) — biggest non-terminal signal
    if curr_ante > prev_ante:
        reward += 1.0

    # Beat a blind (transition into shop from hand play)
    prev_in_shop = prev_state_info.get("in_shop", False)
    curr_in_shop = not bool(state.round_resets.blind) or state.round_resets.blind_states.get(
        state.blind_on_deck or "Small", ""
    ) in ("Defeated", "Skipped")
    prev_beaten = prev_state_info.get("blind_beaten", False)
    if not prev_beaten and not prev_in_shop:
        # We were playing a blind. Check if we just beat it.
        # Detect via hands_left — if prev had hands and now we're in shop, we beat it
        if prev_state_info.get("hands_left", 0) > 0 and curr_ante == prev_ante:
            # Hands remaining efficiency bonus (only on blind-beat transition)
            # Can't easily detect the exact transition here, so we skip mid-blind bonuses
            pass

    # Interest earned — only reward *actual* interest at cash_out (shop entry).
    # Detect by checking if we just entered shop (dollars jumped from interest).
    # We approximate: reward at ante advance based on savings tier.
    if curr_ante > prev_ante:
        interest_tier = min(prev_state_info.get("dollars", 0) // 5, state.interest_cap // 5)
        reward += 0.1 * min(interest_tier, 5)

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
