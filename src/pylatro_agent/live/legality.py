"""Intersect Pylatro's action mask with legality observed in Balatro."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pylatro_agent.constants import (
    CONSUMABLE_ACTIONS_PER_SLOT,
    MAX_CONSUMABLE_SLOTS,
    NUM_ACTIONS,
    ActionRange,
)
from pylatro_agent.subset_actions import subset_indices

from .protocol import ProtocolError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from .adapter import LiveState


def _flag(legal: Mapping[str, object], key: str) -> bool:
    value = legal.get(key, False)
    if not isinstance(value, bool):
        raise ProtocolError(f"legal.{key} must be a boolean")
    return value


def _id_set(legal: Mapping[str, object], key: str) -> set[str]:
    value = legal.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProtocolError(f"legal.{key} must be an array of ids")
    return set(value)


def _restrict_subset_range(
    mask: np.ndarray,
    start: int,
    end: int,
    hand_ids: list[str],
    allowed_ids: set[str],
    exact_sets: object,
) -> None:
    exact: set[frozenset[str]] | None = None
    if exact_sets is not None:
        if not isinstance(exact_sets, list):
            raise ProtocolError("legal exact card sets must be an array")
        exact = set()
        for item in exact_sets:
            if not isinstance(item, list) or any(not isinstance(card_id, str) for card_id in item):
                raise ProtocolError("each legal card set must be an array of ids")
            exact.add(frozenset(item))
    for action_id in range(start, end + 1):
        if not mask[action_id]:
            continue
        selected = {hand_ids[index] for index in subset_indices(action_id - start) if index < len(hand_ids)}
        if not selected.issubset(allowed_ids) or (exact is not None and frozenset(selected) not in exact):
            mask[action_id] = 0


def intersect_live_legality(
    local_mask: np.ndarray,
    live: LiveState,
    legal: Mapping[str, object],
) -> np.ndarray:
    """Return the safe intersection; an empty result is an error."""
    if local_mask.shape != (NUM_ACTIONS,):
        raise ValueError(f"expected mask shape {(NUM_ACTIONS,)}, got {local_mask.shape}")
    mask = local_mask.astype(np.int8, copy=True)
    ar = ActionRange

    if not _flag(legal, "blind_play"):
        mask[ar.BLIND_PLAY] = 0
    if not _flag(legal, "blind_skip"):
        mask[ar.BLIND_SKIP] = 0
    if not _flag(legal, "blind_reroll"):
        mask[ar.BLIND_REROLL] = 0

    play_ids = _id_set(legal, "play_card_ids")
    discard_ids = _id_set(legal, "discard_card_ids")
    if not _flag(legal, "play"):
        mask[ar.PLAY_SUBSET_START : ar.PLAY_SUBSET_END + 1] = 0
    else:
        _restrict_subset_range(
            mask,
            int(ar.PLAY_SUBSET_START),
            int(ar.PLAY_SUBSET_END),
            live.hand_ids,
            play_ids,
            legal.get("play_card_sets"),
        )
    if not _flag(legal, "discard"):
        mask[ar.DISCARD_SUBSET_START : ar.DISCARD_SUBSET_END + 1] = 0
    else:
        _restrict_subset_range(
            mask,
            int(ar.DISCARD_SUBSET_START),
            int(ar.DISCARD_SUBSET_END),
            live.hand_ids,
            discard_ids,
            legal.get("discard_card_sets"),
        )

    usable_consumables = _id_set(legal, "use_consumable_ids")
    for slot in range(MAX_CONSUMABLE_SLOTS):
        item_id = live.consumable_ids[slot] if slot < len(live.consumable_ids) else None
        if item_id not in usable_consumables:
            start = int(ar.CONSUMABLE_FLAT_START) + slot * CONSUMABLE_ACTIONS_PER_SLOT
            mask[start : start + CONSUMABLE_ACTIONS_PER_SLOT] = 0

    buy_ids = _id_set(legal, "shop_buy_ids")
    for index, item_id in enumerate(live.shop_ids):
        if item_id not in buy_ids:
            mask[int(ar.SHOP_BUY_START) + index] = 0
    if not _flag(legal, "shop_reroll"):
        mask[ar.SHOP_REROLL] = 0
    sell_jokers = _id_set(legal, "shop_sell_joker_ids")
    for index, item_id in enumerate(live.joker_ids):
        if item_id not in sell_jokers:
            mask[int(ar.SHOP_SELL_JOKER_START) + index] = 0
    sell_consumables = _id_set(legal, "shop_sell_consumable_ids")
    for index, item_id in enumerate(live.consumable_ids):
        if item_id not in sell_consumables:
            mask[int(ar.SHOP_SELL_CONSUMABLE_START) + index] = 0
    if not _flag(legal, "shop_leave"):
        mask[ar.SHOP_LEAVE] = 0

    claim_ids = _id_set(legal, "pack_claim_ids")
    for index, item_id in enumerate(live.pack_ids):
        if item_id not in claim_ids:
            mask[int(ar.PACK_CLAIM_START) + index] = 0
    if not _flag(legal, "pack_skip"):
        mask[ar.PACK_SKIP] = 0

    if not mask.any():
        raise ProtocolError("no action remains after intersecting Pylatro and live legality")
    return mask


def restrict_to_actions(mask: np.ndarray, action_ids: Iterable[int]) -> np.ndarray:
    restricted = np.zeros_like(mask)
    for action_id in action_ids:
        if 0 <= action_id < len(mask):
            restricted[action_id] = mask[action_id]
    return restricted
