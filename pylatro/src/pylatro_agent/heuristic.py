"""Rule-based heuristic agent for generating supervised pretraining data."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from pylatro import can_use_consumable, evaluate_poker_hand
from pylatro.models import PlayingCard, RunState
from pylatro.scoring import RANK_TO_NOMINAL

from .action import ActionType, encode_action
from .constants import ActionRange, SubPhase


class HeuristicAgent:
    """Simple rule-based agent that plays highest-scoring hands and buys jokers."""

    def select_action(self, state: RunState, sub_phase: SubPhase, action_mask: np.ndarray, **kwargs) -> int:
        """Select an action given the current state and valid action mask.

        kwargs may include:
            selected_cards: set[int] — current card selection state
            pending_action: str — "play" or "discard"
            pending_consumable_slot: int | None
        """
        if sub_phase == SubPhase.BLIND_SELECT:
            return self._blind_select(state, action_mask)
        elif sub_phase == SubPhase.CHOOSE_ACTION:
            return self._choose_action(state, action_mask)
        elif sub_phase == SubPhase.SELECT_CARDS:
            return self._select_cards(state, action_mask, kwargs.get("selected_cards", set()), kwargs.get("pending_action"))
        elif sub_phase == SubPhase.SHOP:
            return self._shop(state, action_mask)
        elif sub_phase == SubPhase.BOOSTER_PACK:
            return self._booster_pack(state, action_mask)
        elif sub_phase == SubPhase.CONSUMABLE_TARGET:
            return self._consumable_target(state, action_mask, kwargs.get("pending_consumable_slot"))
        return self._random_valid(action_mask)

    def _blind_select(self, state: RunState, mask: np.ndarray) -> int:
        # Always play the blind
        if mask[ActionRange.BLIND_PLAY]:
            return ActionRange.BLIND_PLAY
        return self._random_valid(mask)

    def _choose_action(self, state: RunState, mask: np.ndarray) -> int:
        # Use planet consumables if available
        if mask[ActionRange.USE_CONSUMABLE]:
            for cons in state.consumables:
                center = state.data.centers[cons.center_key]
                if center.get("set") == "Planet" and can_use_consumable(state, cons):
                    return ActionRange.USE_CONSUMABLE

        # Check if current hand has a real poker hand (pair or better)
        hand = state.hand_cards
        has_good_hand = False
        if hand:
            best = self._find_best_hand(state, hand)
            cards = [hand[i] for i in best]
            try:
                result = evaluate_poker_hand(state, cards)
                for hname in ["Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
                              "Four of a Kind", "Full House", "Flush", "Straight",
                              "Three of a Kind", "Two Pair", "Pair"]:
                    if result.get(hname) and any(result[hname]):
                        has_good_hand = True
                        break
            except Exception:
                pass

        # Discard first if hand is weak and we have discards
        if not has_good_hand and mask[ActionRange.DISCARD] and state.current_round.discards_left > 0:
            return ActionRange.DISCARD

        # Play if we can
        if mask[ActionRange.PLAY_HAND]:
            return ActionRange.PLAY_HAND
        # Otherwise discard
        if mask[ActionRange.DISCARD]:
            return ActionRange.DISCARD
        return self._random_valid(mask)

    def _select_cards(self, state: RunState, mask: np.ndarray, selected: set[int], pending: str | None) -> int:
        """Pick cards forming the best poker hand, then confirm."""
        hand = state.hand_cards
        if not hand:
            if mask[ActionRange.SELECT_CONFIRM]:
                return ActionRange.SELECT_CONFIRM
            return self._random_valid(mask)

        if pending == "play":
            best_cards = self._find_best_hand(state, hand)
        else:
            # For discard, select lowest-value cards (up to 5)
            best_cards = self._find_worst_cards(state, hand, max_discard=min(5, len(hand)))

        best_indices = set(best_cards)

        # Toggle cards to match target selection
        for idx in range(len(hand)):
            if idx in best_indices and idx not in selected:
                action = ActionRange.TOGGLE_CARD_START + idx
                if mask[action]:
                    return action
            elif idx not in best_indices and idx in selected:
                action = ActionRange.TOGGLE_CARD_START + idx
                if mask[action]:
                    return action

        # If selection matches, confirm
        if mask[ActionRange.SELECT_CONFIRM]:
            return ActionRange.SELECT_CONFIRM

        return self._random_valid(mask)

    def _find_best_hand(self, state: RunState, hand: list[PlayingCard]) -> set[int]:
        """Find indices of cards forming the best 5-card poker hand."""
        best_hand_name = "High Card"
        best_indices: set[int] = set()
        best_score = -1

        hand_order = [
            "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
            "Four of a Kind", "Full House", "Flush", "Straight",
            "Three of a Kind", "Two Pair", "Pair", "High Card",
        ]
        hand_rank = {name: i for i, name in enumerate(hand_order)}

        max_cards = min(5, len(hand))
        for size in range(max_cards, 0, -1):
            for combo in combinations(range(len(hand)), size):
                cards = [hand[i] for i in combo]
                try:
                    result = evaluate_poker_hand(state, cards)
                except Exception:
                    continue

                for hand_name in hand_order:
                    if result.get(hand_name) and any(result[hand_name]):
                        rank = hand_rank[hand_name]
                        # Lower rank = better hand
                        score = -rank * 10000 + sum(RANK_TO_NOMINAL.get(c.rank, 0) for c in cards)
                        if score > best_score:
                            best_score = score
                            best_indices = set(combo)
                            best_hand_name = hand_name
                        break

        # If only High Card, play the 5 highest-value cards for max chips
        if best_hand_name == "High Card" or not best_indices:
            ranked = sorted(range(len(hand)), key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
            best_indices = set(ranked[:max_cards])

        return best_indices

    def _find_worst_cards(self, state: RunState, hand: list[PlayingCard], max_discard: int) -> set[int]:
        """Find indices of least valuable cards to discard."""
        ranked = sorted(range(len(hand)), key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
        return set(ranked[:max_discard])

    def _shop(self, state: RunState, mask: np.ndarray) -> int:
        # Buy best joker if affordable and have slots
        all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Joker":
                return action

        # Save money for interest ($5 increments) — leave shop
        if mask[ActionRange.SHOP_LEAVE]:
            return ActionRange.SHOP_LEAVE
        return self._random_valid(mask)

    def _booster_pack(self, state: RunState, mask: np.ndarray) -> int:
        # Claim first available card
        for i in range(5):
            action = ActionRange.PACK_CLAIM_START + i
            if mask[action]:
                return action
        if mask[ActionRange.PACK_SKIP]:
            return ActionRange.PACK_SKIP
        return self._random_valid(mask)

    def _consumable_target(self, state: RunState, mask: np.ndarray, pending_slot: int | None) -> int:
        if pending_slot is None:
            # Select first usable planet, then tarot
            for i, cons in enumerate(state.consumables):
                action = ActionRange.CONSUMABLE_SLOT_START + i
                if mask[action]:
                    center = state.data.centers[cons.center_key]
                    if center.get("set") == "Planet":
                        return action
            # Select any usable consumable
            for i in range(5):
                action = ActionRange.CONSUMABLE_SLOT_START + i
                if mask[action]:
                    return action

        # Confirm if possible
        if mask[ActionRange.CONSUMABLE_CONFIRM]:
            return ActionRange.CONSUMABLE_CONFIRM
        # Cancel otherwise
        if mask[ActionRange.CONSUMABLE_CANCEL]:
            return ActionRange.CONSUMABLE_CANCEL
        return self._random_valid(mask)

    def _random_valid(self, mask: np.ndarray) -> int:
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))
