"""Helpers for changing owned-joker order from the interactive TUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pylatro.instances import sync_all_jokers

if TYPE_CHECKING:
    from pylatro.models import RunState


def move_owned_joker(state: RunState, index: int, offset: int) -> int:
    """Move one joker and return its new index.

    ``joker_keys`` is an index-aligned engine cache, so it must be reordered
    with the live instances. Re-syncing also updates Blueprint and Brainstorm's
    compatibility display immediately.
    """
    target = index + offset
    if not 0 <= index < len(state.jokers) or not 0 <= target < len(state.jokers):
        return index

    state.jokers[index], state.jokers[target] = state.jokers[target], state.jokers[index]
    state.joker_keys = [joker.center_key for joker in state.jokers]
    sync_all_jokers(state)
    return target
