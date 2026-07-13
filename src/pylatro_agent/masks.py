"""Compute valid action masks from RunState and SubPhase."""

from __future__ import annotations

import numpy as np

from pylatro import can_use_consumable
from pylatro.flow import _debuff_hand
from pylatro.models import RunState
from pylatro.scoring import get_poker_hand_info

from .constants import (
    CONSUMABLE_ACTIONS_PER_SLOT,
    CONSUMABLE_HAND_SUBSET_OFFSET,
    CONSUMABLE_JOKER_OFFSET,
    CONSUMABLE_NO_TARGET_OFFSET,
    HAND_TARGET_CONSUMABLE_LIMITS,
    JOKER_TARGET_CONSUMABLE_NAMES,
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    ActionRange,
    SubPhase,
)
from .subset_actions import (
    consumable_subset_indices,
    legal_consumable_subset_mask,
    legal_subset_mask,
    subset_indices,
)


def compute_action_mask(
    state: RunState,
    sub_phase: SubPhase,
    selected_cards: set[int] | None = None,
    pending_action: str | None = None,
) -> np.ndarray:
    """Return a binary mask of shape (NUM_ACTIONS,) where 1 = valid."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)

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
        play_subsets = _mask_debuffed_plays(state, legal_subsets.astype(np.int8))
        mask[AR.PLAY_SUBSET_START:AR.PLAY_SUBSET_END + 1] = play_subsets

    if state.current_round.discards_left > 0 and hand_size > 0:
        mask[AR.DISCARD_SUBSET_START:AR.DISCARD_SUBSET_END + 1] = legal_subsets.astype(np.int8)

    _mask_consumable_flat(mask, state)


def _mask_debuffed_plays(state: RunState, play_subsets: np.ndarray) -> np.ndarray:
    """Zero out play subsets that the active boss debuff would score as 0.

    A hand-level debuff (The Psychic's must-play-5, The Eye's no-repeat hand
    type, The Mouth's single hand type, showdown "hand" patterns) makes the
    whole play score 0 chips while still consuming a hand — the play is
    provably wasted at decision time, so it is masked rather than left for the
    policy to discover. Card-level debuffs (suit/rank) only zero individual
    cards and are not touched. Semantics are delegated to flow._debuff_hand
    with check=True so the mask can never disagree with scoring.
    """
    if state.blind_disabled:
        return play_subsets
    blind = state.round_resets.blind or {}
    debuff = blind.get("debuff") or {}
    if not isinstance(debuff, dict):
        debuff = {}

    h_size_ge = int(debuff.get("h_size_ge") or 0)
    h_size_le = int(debuff.get("h_size_le") or 0)
    # The Eye/Mouth zero plays by boss name with an empty debuff table, so
    # they are gated on the name, not the dict (The Arm/Ox never zero plays).
    needs_hand_eval = bool(debuff.get("hand")) or str(blind.get("name", "")) in (
        "The Eye",
        "The Mouth",
    )
    if not (h_size_ge or h_size_le or needs_hand_eval):
        return play_subsets

    filtered = play_subsets.copy()
    # _debuff_hand flips state.blind_triggered even in check mode, and Matador
    # reads that flag during real scoring — restore it after probing.
    saved_triggered = state.blind_triggered
    try:
        for idx in np.where(filtered)[0]:
            positions = subset_indices(int(idx))
            if h_size_ge and len(positions) < h_size_ge:
                filtered[idx] = 0
                continue
            if h_size_le and len(positions) > h_size_le:
                filtered[idx] = 0
                continue
            if not needs_hand_eval:
                continue
            cards = [state.hand_cards[i] for i in positions]
            hand_name, _, poker_hands, _ = get_poker_hand_info(state, cards)
            if _debuff_hand(state, cards, hand_name, poker_hands, check=True):
                filtered[idx] = 0
    finally:
        state.blind_triggered = saved_triggered

    # Never mask the phase into a dead end: with e.g. The Psychic and fewer
    # than 5 cards remaining every play scores 0, but the game still demands
    # one — keep the unfiltered subsets in that case.
    if not filtered.any():
        return play_subsets
    return filtered


def _mask_consumable_flat(mask: np.ndarray, state: RunState) -> None:
    """Enable atomic consumable actions for every usable slot.

    For each slot the policy picks slot+target in one step; the layout is
    [no_target, hand_subset_0..695, joker_0..7] contiguous per slot.
    """
    base = int(ActionRange.CONSUMABLE_FLAT_START)
    num_jokers = min(len(state.jokers), MAX_JOKER_SLOTS)
    hand_size = min(len(state.hand_cards), MAX_HAND_SIZE)

    for slot in range(min(len(state.consumables), MAX_CONSUMABLE_SLOTS)):
        cons = state.consumables[slot]
        center = state.data.centers[cons.center_key]
        config = center.get("config") or {}
        max_highlighted = config.get("max_highlighted")
        name = center.get("name", "")
        needs_joker_target = name in JOKER_TARGET_CONSUMABLE_NAMES
        fallback_hand_limits = HAND_TARGET_CONSUMABLE_LIMITS.get(name)

        slot_base = base + slot * CONSUMABLE_ACTIONS_PER_SLOT

        if max_highlighted is None and fallback_hand_limits is None and not needs_joker_target:
            if can_use_consumable(state, cons, hand_targets=(), joker_targets=()):
                mask[slot_base + CONSUMABLE_NO_TARGET_OFFSET] = 1
            continue

        if max_highlighted is not None or fallback_hand_limits is not None:
            if fallback_hand_limits is not None:
                min_size, max_size = fallback_hand_limits
            else:
                min_size = int(config.get("min_highlighted", 1) or 1)
                max_size = int(max_highlighted)
            max_size = min(max_size, MAX_CONSUMABLE_HAND_TARGETS)
            subset_mask = legal_consumable_subset_mask(hand_size, min_size, max_size)
            start = slot_base + CONSUMABLE_HAND_SUBSET_OFFSET
            for subset_idx in np.where(subset_mask)[0]:
                hand_targets = consumable_subset_indices(int(subset_idx))
                if can_use_consumable(state, cons, hand_targets=hand_targets, joker_targets=()):
                    mask[start + int(subset_idx)] = 1

        if needs_joker_target:
            start = slot_base + CONSUMABLE_JOKER_OFFSET
            for j in range(num_jokers):
                if can_use_consumable(state, cons, hand_targets=(), joker_targets=(j,)):
                    mask[start + j] = 1


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
            if item.card_type == "Joker":
                is_negative = bool(item.edition and item.edition.get("negative"))
                if len(state.jokers) < joker_limit(state) or is_negative:
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
        from pylatro.runtime import consumable_limit, joker_limit

        for i in range(min(len(pack.cards), MAX_PACK_CARDS)):
            card = pack.cards[i]
            center = state.data.centers.get(card.center_key, {})
            card_type = _pack_card_type(center)

            if card_type == "Joker":
                is_negative = bool(card.edition and card.edition.get("negative"))
                if len(state.jokers) < joker_limit(state) or is_negative:
                    mask[AR.PACK_CLAIM_START + i] = 1
            elif card_type in ("Tarot", "Planet", "Spectral"):
                if len(state.consumables) < consumable_limit(state):
                    mask[AR.PACK_CLAIM_START + i] = 1
            else:
                mask[AR.PACK_CLAIM_START + i] = 1

    # Skip/close always valid
    mask[AR.PACK_SKIP] = 1


def _pack_card_type(center: dict) -> str:
    if not center:
        return ""
    if center.get("set") == "Joker":
        return "Joker"
    if center.get("consumeable"):
        return str(center.get("set", ""))
    if center.get("set") in ("Default", "Enhanced"):
        return "Playing"
    return ""
