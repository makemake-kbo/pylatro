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

        hand = state.hand_cards
        can_discard = mask[ActionRange.DISCARD] and state.current_round.discards_left > 0
        hand_quality = self._evaluate_hand_quality(state, hand) if hand else 0

        # Strong hands (two pair+, flush, straight): play immediately
        # Weak hands with discards left: try to improve
        # Pair only: play if no discards, otherwise discard to try for better
        if hand_quality < 2 and can_discard:
            return ActionRange.DISCARD

        # Play if we can
        if mask[ActionRange.PLAY_HAND]:
            return ActionRange.PLAY_HAND
        if mask[ActionRange.DISCARD]:
            return ActionRange.DISCARD
        return self._random_valid(mask)

    def _evaluate_hand_quality(self, state: RunState, hand: list[PlayingCard]) -> int:
        """Rate the best hand quality: 0=high card, 1=pair, 2=two pair, 3+=better."""
        quality_map = {
            "Flush Five": 8, "Flush House": 7, "Five of a Kind": 6,
            "Straight Flush": 5, "Four of a Kind": 4, "Full House": 3,
            "Flush": 3, "Straight": 3, "Three of a Kind": 2, "Two Pair": 2,
            "Pair": 1, "High Card": 0,
        }
        best = self._find_best_hand(state, hand)
        cards = [hand[i] for i in best]
        try:
            result = evaluate_poker_hand(state, cards)
            for hname, quality in quality_map.items():
                if result.get(hname) and any(result[hname]):
                    return quality
        except Exception:
            pass
        return 0

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
            # For discard, find cards NOT in our best potential hand and discard those
            keep = self._find_best_hand(state, hand)
            discard = set(range(len(hand))) - keep
            # Cap at 5 discards, prioritize discarding worst cards
            if len(discard) > 5:
                worst = self._find_worst_cards(state, hand, max_discard=5)
                discard = discard & worst
                if not discard:
                    discard = worst
            best_cards = discard if discard else self._find_worst_cards(state, hand, max_discard=min(3, len(hand)))

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
        best_score = float("-inf")

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
        """Find indices of cards to discard — keep cards that contribute to potential hands."""
        # Score each card by how useful it is for building hands
        from collections import Counter

        suits = [c.suit for c in hand]
        ranks = [c.rank for c in hand]
        suit_counts = Counter(suits)
        rank_counts = Counter(ranks)

        # Score each card: higher = more worth keeping
        keep_scores: list[float] = []
        for i, card in enumerate(hand):
            score = 0.0
            # Base value from rank
            score += RANK_TO_NOMINAL.get(card.rank, 0) * 0.1

            # Flush potential: cards in the most common suit get a big bonus
            suit_n = suit_counts[card.suit]
            if suit_n >= 4:
                score += 20.0  # near-flush — strongly keep
            elif suit_n >= 3:
                score += 8.0

            # Pair/trips potential: cards with matching ranks
            rank_n = rank_counts[card.rank]
            if rank_n >= 3:
                score += 25.0  # trips or better — always keep
            elif rank_n >= 2:
                score += 15.0  # pair — keep

            # Straight potential: check for connected cards
            nominal = RANK_TO_NOMINAL.get(card.rank, 0)
            nearby = sum(1 for r in ranks if abs(RANK_TO_NOMINAL.get(r, 0) - nominal) <= 4 and r != card.rank)
            if nearby >= 3:
                score += 5.0

            keep_scores.append(score)

        # Discard the lowest-scored cards
        ranked = sorted(range(len(hand)), key=lambda i: keep_scores[i])
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
