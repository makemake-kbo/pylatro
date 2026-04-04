"""Action encoding/decoding between flat action IDs and game operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .constants import ActionRange, SubPhase


class ActionType(StrEnum):
    BLIND_PLAY = "blind_play"
    BLIND_SKIP = "blind_skip"
    BLIND_REROLL = "blind_reroll"
    PLAY_CANDIDATE = "play_candidate"
    DISCARD_CANDIDATE = "discard_candidate"
    USE_CONSUMABLE = "use_consumable"
    CONSUMABLE_SLOT = "consumable_slot"
    CONSUMABLE_HAND_TARGET = "consumable_hand_target"
    CONSUMABLE_JOKER_TARGET = "consumable_joker_target"
    CONSUMABLE_CONFIRM = "consumable_confirm"
    CONSUMABLE_CANCEL = "consumable_cancel"
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
    index: int = 0  # slot/card index where applicable


def decode_action(action_id: int) -> DecodedAction:
    """Convert flat action ID to a DecodedAction."""
    AR = ActionRange

    if action_id == AR.BLIND_PLAY:
        return DecodedAction(ActionType.BLIND_PLAY)
    if action_id == AR.BLIND_SKIP:
        return DecodedAction(ActionType.BLIND_SKIP)
    if action_id == AR.BLIND_REROLL:
        return DecodedAction(ActionType.BLIND_REROLL)
    if AR.PLAY_CANDIDATE_START <= action_id <= AR.PLAY_CANDIDATE_END:
        return DecodedAction(ActionType.PLAY_CANDIDATE, action_id - AR.PLAY_CANDIDATE_START)
    if AR.DISCARD_CANDIDATE_START <= action_id <= AR.DISCARD_CANDIDATE_END:
        return DecodedAction(ActionType.DISCARD_CANDIDATE, action_id - AR.DISCARD_CANDIDATE_START)
    if action_id == AR.USE_CONSUMABLE:
        return DecodedAction(ActionType.USE_CONSUMABLE)

    if AR.CONSUMABLE_SLOT_START <= action_id <= AR.CONSUMABLE_SLOT_END:
        return DecodedAction(ActionType.CONSUMABLE_SLOT, action_id - AR.CONSUMABLE_SLOT_START)
    if AR.CONSUMABLE_HAND_TARGET_START <= action_id <= AR.CONSUMABLE_HAND_TARGET_END:
        return DecodedAction(ActionType.CONSUMABLE_HAND_TARGET, action_id - AR.CONSUMABLE_HAND_TARGET_START)
    if AR.CONSUMABLE_JOKER_TARGET_START <= action_id <= AR.CONSUMABLE_JOKER_TARGET_END:
        return DecodedAction(ActionType.CONSUMABLE_JOKER_TARGET, action_id - AR.CONSUMABLE_JOKER_TARGET_START)
    if action_id == AR.CONSUMABLE_CONFIRM:
        return DecodedAction(ActionType.CONSUMABLE_CONFIRM)
    if action_id == AR.CONSUMABLE_CANCEL:
        return DecodedAction(ActionType.CONSUMABLE_CANCEL)

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


def encode_action(action_type: ActionType, index: int = 0) -> int:
    """Convert ActionType + index to flat action ID."""
    AR = ActionRange

    match action_type:
        case ActionType.BLIND_PLAY:
            return AR.BLIND_PLAY
        case ActionType.BLIND_SKIP:
            return AR.BLIND_SKIP
        case ActionType.BLIND_REROLL:
            return AR.BLIND_REROLL
        case ActionType.PLAY_CANDIDATE:
            return AR.PLAY_CANDIDATE_START + index
        case ActionType.DISCARD_CANDIDATE:
            return AR.DISCARD_CANDIDATE_START + index
        case ActionType.USE_CONSUMABLE:
            return AR.USE_CONSUMABLE
        case ActionType.CONSUMABLE_SLOT:
            return AR.CONSUMABLE_SLOT_START + index
        case ActionType.CONSUMABLE_HAND_TARGET:
            return AR.CONSUMABLE_HAND_TARGET_START + index
        case ActionType.CONSUMABLE_JOKER_TARGET:
            return AR.CONSUMABLE_JOKER_TARGET_START + index
        case ActionType.CONSUMABLE_CONFIRM:
            return AR.CONSUMABLE_CONFIRM
        case ActionType.CONSUMABLE_CANCEL:
            return AR.CONSUMABLE_CANCEL
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
