"""Action encoding/decoding between flat action IDs and game operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .constants import (
    CONSUMABLE_ACTIONS_PER_SLOT,
    CONSUMABLE_HAND_SUBSET_OFFSET,
    CONSUMABLE_JOKER_OFFSET,
    CONSUMABLE_NO_TARGET_OFFSET,
    MAX_JOKER_SLOTS,
    NUM_CONSUMABLE_HAND_SUBSETS,
    ActionRange,
)


class ActionType(StrEnum):
    BLIND_PLAY = "blind_play"
    BLIND_SKIP = "blind_skip"
    BLIND_REROLL = "blind_reroll"
    PLAY_SUBSET = "play_subset"
    DISCARD_SUBSET = "discard_subset"
    USE_CONSUMABLE_NO_TARGET = "use_consumable_no_target"
    USE_CONSUMABLE_HAND_SUBSET = "use_consumable_hand_subset"
    USE_CONSUMABLE_JOKER = "use_consumable_joker"
    SHOP_BUY = "shop_buy"
    SHOP_REROLL = "shop_reroll"
    SHOP_SELL_JOKER = "shop_sell_joker"
    SHOP_SELL_CONSUMABLE = "shop_sell_consumable"
    SHOP_LEAVE = "shop_leave"
    PACK_CLAIM = "pack_claim"
    PACK_SKIP = "pack_skip"


@dataclass(slots=True)
class DecodedAction:
    action_type: ActionType
    index: int = 0  # primary index, slot/card/shop offset depending on action
    # For USE_CONSUMABLE_HAND_SUBSET: hand-subset id inside the slot block.
    # For USE_CONSUMABLE_JOKER: joker target index.
    # For USE_CONSUMABLE_NO_TARGET: unused.
    detail: int = 0


def decode_action(action_id: int) -> DecodedAction:
    """Convert flat action ID to a DecodedAction."""
    AR = ActionRange

    if action_id == AR.BLIND_PLAY:
        return DecodedAction(ActionType.BLIND_PLAY)
    if action_id == AR.BLIND_SKIP:
        return DecodedAction(ActionType.BLIND_SKIP)
    if action_id == AR.BLIND_REROLL:
        return DecodedAction(ActionType.BLIND_REROLL)
    if AR.PLAY_SUBSET_START <= action_id <= AR.PLAY_SUBSET_END:
        return DecodedAction(ActionType.PLAY_SUBSET, action_id - AR.PLAY_SUBSET_START)
    if AR.DISCARD_SUBSET_START <= action_id <= AR.DISCARD_SUBSET_END:
        return DecodedAction(ActionType.DISCARD_SUBSET, action_id - AR.DISCARD_SUBSET_START)

    if AR.CONSUMABLE_FLAT_START <= action_id <= AR.CONSUMABLE_FLAT_END:
        rel = action_id - int(AR.CONSUMABLE_FLAT_START)
        slot, within = divmod(rel, CONSUMABLE_ACTIONS_PER_SLOT)
        if within == CONSUMABLE_NO_TARGET_OFFSET:
            return DecodedAction(ActionType.USE_CONSUMABLE_NO_TARGET, slot)
        if (
            CONSUMABLE_HAND_SUBSET_OFFSET
            <= within
            < CONSUMABLE_HAND_SUBSET_OFFSET + NUM_CONSUMABLE_HAND_SUBSETS
        ):
            return DecodedAction(
                ActionType.USE_CONSUMABLE_HAND_SUBSET,
                slot,
                within - CONSUMABLE_HAND_SUBSET_OFFSET,
            )
        if CONSUMABLE_JOKER_OFFSET <= within < CONSUMABLE_JOKER_OFFSET + MAX_JOKER_SLOTS:
            return DecodedAction(
                ActionType.USE_CONSUMABLE_JOKER,
                slot,
                within - CONSUMABLE_JOKER_OFFSET,
            )
        raise ValueError(f"Invalid consumable flat offset: {within}")

    if AR.SHOP_BUY_START <= action_id <= AR.SHOP_BUY_END:
        return DecodedAction(ActionType.SHOP_BUY, action_id - AR.SHOP_BUY_START)
    if action_id == AR.SHOP_REROLL:
        return DecodedAction(ActionType.SHOP_REROLL)
    if AR.SHOP_SELL_JOKER_START <= action_id <= AR.SHOP_SELL_JOKER_END:
        return DecodedAction(ActionType.SHOP_SELL_JOKER, action_id - AR.SHOP_SELL_JOKER_START)
    if AR.SHOP_SELL_CONSUMABLE_START <= action_id <= AR.SHOP_SELL_CONSUMABLE_END:
        return DecodedAction(ActionType.SHOP_SELL_CONSUMABLE, action_id - AR.SHOP_SELL_CONSUMABLE_START)
    if action_id == AR.SHOP_LEAVE:
        return DecodedAction(ActionType.SHOP_LEAVE)

    if AR.PACK_CLAIM_START <= action_id <= AR.PACK_CLAIM_END:
        return DecodedAction(ActionType.PACK_CLAIM, action_id - AR.PACK_CLAIM_START)
    if action_id == AR.PACK_SKIP:
        return DecodedAction(ActionType.PACK_SKIP)

    raise ValueError(f"Invalid action ID: {action_id}")


def _consumable_slot_base(slot: int) -> int:
    return int(ActionRange.CONSUMABLE_FLAT_START) + slot * CONSUMABLE_ACTIONS_PER_SLOT


def encode_action(action_type: ActionType, index: int = 0, detail: int = 0) -> int:
    """Convert ActionType + index(+detail) to flat action ID.

    For USE_CONSUMABLE_HAND_SUBSET and USE_CONSUMABLE_JOKER, `index` is the
    consumable slot and `detail` is the subset/joker offset.
    """
    AR = ActionRange

    match action_type:
        case ActionType.BLIND_PLAY:
            return AR.BLIND_PLAY
        case ActionType.BLIND_SKIP:
            return AR.BLIND_SKIP
        case ActionType.BLIND_REROLL:
            return AR.BLIND_REROLL
        case ActionType.PLAY_SUBSET:
            return AR.PLAY_SUBSET_START + index
        case ActionType.DISCARD_SUBSET:
            return AR.DISCARD_SUBSET_START + index
        case ActionType.USE_CONSUMABLE_NO_TARGET:
            return _consumable_slot_base(index) + CONSUMABLE_NO_TARGET_OFFSET
        case ActionType.USE_CONSUMABLE_HAND_SUBSET:
            return _consumable_slot_base(index) + CONSUMABLE_HAND_SUBSET_OFFSET + detail
        case ActionType.USE_CONSUMABLE_JOKER:
            return _consumable_slot_base(index) + CONSUMABLE_JOKER_OFFSET + detail
        case ActionType.SHOP_BUY:
            return AR.SHOP_BUY_START + index
        case ActionType.SHOP_REROLL:
            return AR.SHOP_REROLL
        case ActionType.SHOP_SELL_JOKER:
            return AR.SHOP_SELL_JOKER_START + index
        case ActionType.SHOP_SELL_CONSUMABLE:
            return AR.SHOP_SELL_CONSUMABLE_START + index
        case ActionType.SHOP_LEAVE:
            return AR.SHOP_LEAVE
        case ActionType.PACK_CLAIM:
            return AR.PACK_CLAIM_START + index
        case ActionType.PACK_SKIP:
            return AR.PACK_SKIP
    raise ValueError(f"Unknown action type: {action_type}")
