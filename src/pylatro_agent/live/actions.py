"""Translate between flat internal actions and semantic wire commands."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pylatro_agent.action import ActionType, DecodedAction, decode_action
from pylatro_agent.subset_actions import consumable_subset_indices, subset_indices

from .protocol import ProtocolError

if TYPE_CHECKING:
    from .adapter import LiveState


def _indexed(items: list[str], index: int, kind: str) -> str:
    if index < 0 or index >= len(items):
        raise ProtocolError(f"{kind} index {index} is absent from the live snapshot")
    return items[index]


def semantic_action(action_id: int, live: LiveState) -> dict[str, Any]:
    decoded = decode_action(action_id)
    action_type = decoded.action_type
    if action_type == ActionType.BLIND_PLAY:
        return {"type": "blind_play"}
    if action_type == ActionType.BLIND_SKIP:
        return {"type": "blind_skip"}
    if action_type == ActionType.BLIND_REROLL:
        return {"type": "blind_reroll"}
    if action_type in (ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET):
        ids = [_indexed(live.hand_ids, index, "hand card") for index in subset_indices(decoded.index)]
        return {
            "type": "play" if action_type == ActionType.PLAY_SUBSET else "discard",
            "card_ids": ids,
        }
    if action_type in (
        ActionType.USE_CONSUMABLE_NO_TARGET,
        ActionType.USE_CONSUMABLE_HAND_SUBSET,
        ActionType.USE_CONSUMABLE_JOKER,
    ):
        result: dict[str, Any] = {
            "type": "use_consumable",
            "consumable_id": _indexed(live.consumable_ids, decoded.index, "consumable"),
        }
        result.update(semantic_targets(decoded, live))
        return result
    if action_type == ActionType.SHOP_BUY:
        return {"type": "shop_buy", "item_id": _indexed(live.shop_ids, decoded.index, "shop item")}
    if action_type == ActionType.SHOP_REROLL:
        return {"type": "shop_reroll"}
    if action_type == ActionType.SHOP_SELL_JOKER:
        return {"type": "shop_sell", "item_id": _indexed(live.joker_ids, decoded.index, "joker")}
    if action_type == ActionType.SHOP_SELL_CONSUMABLE:
        return {
            "type": "shop_sell",
            "item_id": _indexed(live.consumable_ids, decoded.index, "consumable"),
        }
    if action_type == ActionType.SHOP_LEAVE:
        return {"type": "shop_leave"}
    if action_type == ActionType.PACK_CLAIM:
        return {"type": "pack_claim", "item_id": _indexed(live.pack_ids, decoded.index, "pack card")}
    if action_type == ActionType.PACK_SKIP:
        return {"type": "pack_skip"}
    raise ProtocolError(f"cannot encode action {action_type}")


def semantic_targets(decoded: DecodedAction, live: LiveState) -> dict[str, Any]:
    if decoded.action_type == ActionType.USE_CONSUMABLE_NO_TARGET:
        return {}
    if decoded.action_type == ActionType.USE_CONSUMABLE_HAND_SUBSET:
        return {
            "card_ids": [
                _indexed(live.hand_ids, index, "hand card") for index in consumable_subset_indices(decoded.detail)
            ]
        }
    if decoded.action_type == ActionType.USE_CONSUMABLE_JOKER:
        return {"joker_ids": [_indexed(live.joker_ids, decoded.detail, "joker")]}
    raise ProtocolError(f"{decoded.action_type} is not a consumable target action")
