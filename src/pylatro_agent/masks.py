"""Compute valid action masks from RunState and SubPhase."""

from __future__ import annotations

import numpy as np

from pylatro import can_use_consumable
from pylatro.models import RunState

from .constants import (
    MAX_CONSUMABLE_SLOTS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    ActionRange,
    SubPhase,
)
from .subset_actions import legal_subset_mask


def compute_action_mask(
    state: RunState,
    sub_phase: SubPhase,
    selected_cards: set[int] | None = None,
    pending_action: str | None = None,
    pending_consumable_slot: int | None = None,
    pending_consumable_targets_hand: tuple[int, ...] = (),
    pending_consumable_targets_joker: tuple[int, ...] = (),
) -> np.ndarray:
    """Return a binary mask of shape (NUM_ACTIONS,) where 1 = valid."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    AR = ActionRange

    if selected_cards is None:
        selected_cards = set()

    if sub_phase == SubPhase.BLIND_SELECT:
        _mask_blind_select(mask, state)

    elif sub_phase == SubPhase.CHOOSE_ACTION:
        _mask_choose_action(mask, state)

    elif sub_phase == SubPhase.SELECT_CARDS:
        _mask_select_cards(mask, state, selected_cards, pending_action)

    elif sub_phase == SubPhase.SHOP:
        _mask_shop(mask, state)

    elif sub_phase == SubPhase.BOOSTER_PACK:
        _mask_booster_pack(mask, state)

    elif sub_phase == SubPhase.CONSUMABLE_TARGET:
        _mask_consumable_target(
            mask, state, pending_consumable_slot,
            pending_consumable_targets_hand, pending_consumable_targets_joker,
        )

    return mask


def _mask_blind_select(mask: np.ndarray, state: RunState) -> None:
    AR = ActionRange
    # Play is always valid
    mask[AR.BLIND_PLAY] = 1

    # Skip only for Small/Big
    blind_on_deck = state.blind_on_deck or "Small"
    if blind_on_deck in ("Small", "Big"):
        mask[AR.BLIND_SKIP] = 1

    # Reroll boss if $>=10 and not already rerolled
    if blind_on_deck == "Boss" and state.dollars >= 10 and not state.round_resets.boss_rerolled:
        mask[AR.BLIND_REROLL] = 1


def _mask_choose_action(mask: np.ndarray, state: RunState) -> None:
    AR = ActionRange
    hand_size = len(state.hand_cards)
    forced_slots = {idx for idx, card in enumerate(state.hand_cards) if card.forced_selection}
    legal_subsets = legal_subset_mask(hand_size, forced_slots)

    if state.current_round.hands_left > 0 and hand_size > 0:
        mask[AR.PLAY_SUBSET_START:AR.PLAY_SUBSET_END + 1] = legal_subsets.astype(np.int8)

    if state.current_round.discards_left > 0 and hand_size > 0:
        mask[AR.DISCARD_SUBSET_START:AR.DISCARD_SUBSET_END + 1] = legal_subsets.astype(np.int8)

    # Use consumable if any is usable
    for i, cons in enumerate(state.consumables):
        if can_use_consumable(state, cons):
            mask[AR.USE_CONSUMABLE] = 1
            break


def _mask_select_cards(
    mask: np.ndarray,
    state: RunState,
    selected_cards: set[int],
    pending_action: str | None,
) -> None:
    # The legacy select_cards phase is intentionally left empty. Hand selection
    # now happens in one shot via exhaustive play/discard subset actions.
    _ = (mask, state, selected_cards, pending_action)


def _mask_shop(mask: np.ndarray, state: RunState) -> None:
    AR = ActionRange

    # Buy shop items (cards + vouchers + boosters flattened)
    all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
    from pylatro.runtime import consumable_limit, joker_limit

    for i, item in enumerate(all_items[:MAX_SHOP_ITEMS]):
        if item.cost <= state.dollars:
            # Check capacity
            if item.card_type == "Joker":
                if len(state.jokers) < joker_limit(state):
                    mask[AR.SHOP_BUY_START + i] = 1
            elif item.card_type in ("Tarot", "Planet", "Spectral"):
                if len(state.consumables) < consumable_limit(state):
                    mask[AR.SHOP_BUY_START + i] = 1
            else:
                # Vouchers, boosters, playing cards
                mask[AR.SHOP_BUY_START + i] = 1

    # Reroll if affordable
    if state.current_round.reroll_cost <= state.dollars:
        mask[AR.SHOP_REROLL] = 1

    # Sell jokers (not eternal)
    for i, joker in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
        if not joker.eternal:
            mask[AR.SHOP_SELL_JOKER_START + i] = 1

    # Sell consumables
    for i in range(min(len(state.consumables), MAX_CONSUMABLE_SLOTS)):
        mask[AR.SHOP_SELL_CONSUMABLE_START + i] = 1

    # Leave always valid
    mask[AR.SHOP_LEAVE] = 1


def _mask_booster_pack(mask: np.ndarray, state: RunState) -> None:
    AR = ActionRange
    pack = state.pack

    if pack and pack.choices_remaining > 0:
        for i in range(min(len(pack.cards), MAX_PACK_CARDS)):
            mask[AR.PACK_CLAIM_START + i] = 1

    # Skip/close always valid
    mask[AR.PACK_SKIP] = 1


def _mask_consumable_target(
    mask: np.ndarray,
    state: RunState,
    pending_slot: int | None,
    hand_targets: tuple[int, ...],
    joker_targets: tuple[int, ...],
) -> None:
    AR = ActionRange

    if pending_slot is None:
        # Need to select which consumable to use
        for i, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            if can_use_consumable(state, cons):
                mask[AR.CONSUMABLE_SLOT_START + i] = 1
    else:
        # Consumable selected, need targets
        cons = state.consumables[pending_slot]
        center = state.data.centers[cons.center_key]
        config = center.get("config") or {}
        max_highlighted = config.get("max_highlighted")

        if max_highlighted is not None:
            # Need card targets
            required_min = int(config.get("min_highlighted", 1) or 1)
            required_max = int(max_highlighted or 0)
            current_count = len(hand_targets)

            if current_count < required_max:
                for i in range(min(len(state.hand_cards), MAX_HAND_SIZE)):
                    if i not in hand_targets:
                        mask[AR.CONSUMABLE_HAND_TARGET_START + i] = 1

            # Confirm if we have enough targets
            if required_min <= current_count <= required_max:
                if can_use_consumable(state, cons, hand_targets=hand_targets, joker_targets=joker_targets):
                    mask[AR.CONSUMABLE_CONFIRM] = 1
        else:
            # No targeting needed (planets, hermit, etc.) — auto-confirm
            if can_use_consumable(state, cons, hand_targets=hand_targets, joker_targets=joker_targets):
                mask[AR.CONSUMABLE_CONFIRM] = 1

        # Joker targets for certain consumables
        name = center.get("name", "")
        if name in ("The Wheel of Fortune", "Ectoplasm", "Hex", "Ankh"):
            for i in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
                if i not in joker_targets:
                    mask[AR.CONSUMABLE_JOKER_TARGET_START + i] = 1

    # Cancel always valid
    mask[AR.CONSUMABLE_CANCEL] = 1
