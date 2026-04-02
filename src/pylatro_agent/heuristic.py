"""Rule-based heuristic agent for generating supervised pretraining data."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from pylatro import can_use_consumable, evaluate_poker_hand
from pylatro.scoring import RANK_TO_ID, RANK_TO_NOMINAL

from .constants import ActionRange, SubPhase

if TYPE_CHECKING:
    from pylatro.models import PlayingCard, RunState


class HeuristicAgent:
    _hand_cache_key: tuple
    _hand_cache_val: set[int]

    def __init__(self) -> None:
        self._hand_cache_key = ()
        self._hand_cache_val = set()

    def _cached_best_hand(self, state: RunState, hand: list[PlayingCard]) -> set[int]:
        key = (
            tuple(
                (
                    id(card),
                    card.rank,
                    card.suit,
                    card.center_key,
                    card.debuff,
                    card.face_down,
                )
                for card in hand
            ),
            tuple(joker.center_key for joker in state.jokers),
        )
        if key == self._hand_cache_key:
            return set(self._hand_cache_val)
        result = set(self._find_best_hand(state, hand))
        self._hand_cache_key = key
        self._hand_cache_val = result
        return set(result)

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
            return self._select_cards(
                state,
                action_mask,
                kwargs.get("selected_cards", set()),
                kwargs.get("pending_action"),
            )
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
        quality_map = {
            "Flush Five": 8, "Flush House": 7, "Five of a Kind": 6,
            "Straight Flush": 5, "Four of a Kind": 4, "Full House": 3,
            "Flush": 3, "Straight": 3, "Three of a Kind": 2, "Two Pair": 2,
            "Pair": 1, "High Card": 0,
        }
        best = self._cached_best_hand(state, hand)
        cards = [hand[i] for i in best]
        quality = self._quick_hand_quality(state, cards)
        return quality_map.get(quality, 0)

    def _quick_hand_quality(self, state: RunState, cards: list[PlayingCard]) -> str:
        n = len(cards)
        if n == 0:
            return "High Card"
        ranks = [RANK_TO_ID[c.rank] for c in cards]
        suits = [c.suit for c in cards]
        rank_counts: dict[int, int] = {}
        for r in ranks:
            rank_counts[r] = rank_counts.get(r, 0) + 1
        counts = sorted(rank_counts.values(), reverse=True)

        is_flush = n >= 5 and len(set(suits)) == 1
        sorted_r = sorted(set(ranks))
        is_straight = False
        if len(sorted_r) >= 5 and len(sorted_r) == n:
            is_straight = sorted_r[-1] - sorted_r[0] == n - 1
            if not is_straight and 14 in sorted_r:
                low = sorted(set(1 if r == 14 else r for r in ranks))
                if len(low) >= 5 and len(low) == n and low[-1] - low[0] == n - 1:
                    is_straight = True

        if is_flush and is_straight:
            if counts[0] >= 4:
                return "Flush Five"
            if counts[0] >= 3 and len(counts) >= 2 and counts[1] >= 2:
                return "Flush House"
            return "Straight Flush"
        if counts[0] >= 5:
            return "Five of a Kind"
        if counts[0] >= 4:
            return "Four of a Kind"
        if counts[0] >= 3 and len(counts) >= 2 and counts[1] >= 2:
            return "Full House"
        if is_flush:
            return "Flush"
        if is_straight:
            return "Straight"
        if counts[0] >= 3:
            return "Three of a Kind"
        if len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2:
            return "Two Pair"
        if counts[0] >= 2:
            return "Pair"
        return "High Card"

    def _select_cards(self, state: RunState, mask: np.ndarray, selected: set[int], pending: str | None) -> int:
        """Pick cards forming the best poker hand, then confirm."""
        hand = state.hand_cards
        if not hand:
            if mask[ActionRange.SELECT_CONFIRM]:
                return ActionRange.SELECT_CONFIRM
            return self._random_valid(mask)

        if pending == "play":
            best_cards = self._cached_best_hand(state, hand)
        else:
            keep = self._cached_best_hand(state, hand)
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
            if (idx in best_indices and idx not in selected) or (idx not in best_indices and idx in selected):
                action = ActionRange.TOGGLE_CARD_START + idx
                if mask[action]:
                    return action

        # If selection matches, confirm
        if mask[ActionRange.SELECT_CONFIRM]:
            return ActionRange.SELECT_CONFIRM

        return self._random_valid(mask)

    def _find_best_hand(self, state: RunState, hand: list[PlayingCard]) -> set[int]:
        if not hand:
            return set()

        max_cards = min(5, len(hand))
        has_stone = False

        centers = state.data.centers
        card_ids: list[int] = []
        by_rank: dict[int, list[int]] = {}
        by_suit: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            center = centers.get(card.center_key)
            effect = center.get("effect", "") if center else ""
            if effect == "Stone Card":
                has_stone = True
                cid = -id(card)
            else:
                cid = RANK_TO_ID[card.rank]
            card_ids.append(cid)
            if cid > 0:
                by_rank.setdefault(cid, []).append(i)

            if effect != "Stone Card":
                is_wild = effect == "Wild Card"
                if is_wild:
                    for s in ("Spades", "Hearts", "Clubs", "Diamonds"):
                        by_suit.setdefault(s, []).append(i)
                else:
                    by_suit.setdefault(card.suit, []).append(i)
                    if state.has_joker("Smeared Joker"):
                        if card.suit in ("Hearts", "Diamonds"):
                            other = "Diamonds" if card.suit == "Hearts" else "Hearts"
                        else:
                            other = "Clubs" if card.suit == "Spades" else "Spades"
                        by_suit.setdefault(other, []).append(i)

        four_fingers = state.has_joker("Four Fingers")
        flush_req = 4 if four_fingers else 5
        straight_req = 4 if four_fingers else 5

        # If stone cards or exotic jokers present, fall back to brute force
        # (rare case — doesn't affect typical performance)
        if has_stone or state.has_joker("Shortcut") or state.has_joker("Pareidolia"):
            return self._find_best_hand_brute(state, hand, max_cards)

        # --- Rank-based groups (sorted high to low) ---
        groups = sorted(by_rank.items(), key=lambda x: x[0], reverse=True)
        groups_by_size: dict[int, list[tuple[int, list[int]]]] = {}
        for cid, indices in groups:
            n = len(indices)
            for sz in range(1, n + 1):
                groups_by_size.setdefault(sz, []).append((cid, indices))

        def _best_of(indices_set: set[int]) -> float:
            """Score for tiebreaking: sum of nominals."""
            return sum(RANK_TO_NOMINAL.get(hand[i].rank, 0) for i in indices_set)

        # --- Flush detection ---
        flush_suit: str | None = None
        flush_indices: list[int] | None = None
        for suit, idxs in by_suit.items():
            # Deduplicate (wild cards may appear multiple times)
            unique = list(dict.fromkeys(idxs))
            if len(unique) >= flush_req:
                # Pick highest-value cards
                unique.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                flush_indices = unique[:5]
                flush_suit = suit
                break

        # --- Straight detection ---
        def _find_straight() -> set[int] | None:
            # Check descending windows of card IDs
            sorted_ranks = sorted(by_rank.keys(), reverse=True)
            # Ace-low: also add 1 if 14 exists
            rank_set = set(sorted_ranks)
            if 14 in rank_set:
                rank_set.add(1)

            for high in range(14, 0, -1):
                run: list[int] = []
                for r in range(high, high - straight_req - 1, -1):
                    if r < 1:
                        break
                    actual = 14 if r == 1 else r
                    if actual in by_rank:
                        run.append(actual)
                    else:
                        break
                if len(run) >= straight_req:
                    result: set[int] = set()
                    for r in run[:5]:
                        result.add(by_rank[r][0])  # pick one card per rank
                    return result
            return None

        straight_indices = _find_straight()

        # --- Try hands from best to worst ---

        # Five of a Kind (rare without special jokers, but check)
        if 5 in groups_by_size:
            cid, idxs = groups_by_size[5][0]
            chosen = set(idxs[:5])
            if flush_indices and chosen <= set(flush_indices):
                return chosen  # Flush Five
            return chosen  # Five of a Kind

        # Four of a Kind
        if 4 in groups_by_size:
            best_four: set[int] | None = None
            best_four_score = -1.0
            for _cid, idxs in groups_by_size[4]:
                s = set(idxs[:4])
                sc = _best_of(s)
                if sc > best_four_score:
                    best_four = s
                    best_four_score = sc

            # Check for Straight Flush first (ranks higher than Four of a Kind)
            if flush_indices and straight_indices:
                sf_set = set(flush_indices) & straight_indices
                if len(sf_set) >= straight_req:
                    return set(list(sf_set)[:5])
                # Try to build straight flush from flush cards
                flush_set = set(flush_indices) if flush_indices else set()
                if straight_indices and len(flush_set & straight_indices) >= straight_req:
                    return set(list(flush_set & straight_indices)[:5])

            # Full House (four of a kind + any pair makes full house available,
            # but four of a kind beats full house, so just return four)
            # Actually check: Full House (3+2) might lose to Four of a Kind
            # Four of a Kind > Full House, so return four
            if best_four is not None:
                return best_four

        # Straight Flush
        if flush_indices and straight_indices and flush_suit:
            flush_only_by_rank: dict[int, int] = {}
            for i in dict.fromkeys(by_suit.get(flush_suit, [])):
                cid = card_ids[i]
                if cid > 0 and cid not in flush_only_by_rank:
                    flush_only_by_rank[cid] = i
            if 14 in flush_only_by_rank:
                flush_only_by_rank.setdefault(1, flush_only_by_rank[14])
            for high in range(14, 0, -1):
                run: list[int] = []
                for r in range(high, high - straight_req - 1, -1):
                    if r < 1:
                        break
                    actual = 14 if r == 1 else r
                    if actual in flush_only_by_rank:
                        run.append(flush_only_by_rank[actual])
                    else:
                        break
                if len(run) >= straight_req:
                    return set(run[:5])

        # Full House (3 + 2)
        if 3 in groups_by_size and 2 in groups_by_size:
            trips = groups_by_size[3]
            pairs = groups_by_size[2]
            best_trip_cid, best_trip_idxs = trips[0]
            # Find a pair that doesn't overlap with the trips
            for pair_cid, pair_idxs in pairs:
                if pair_cid != best_trip_cid:
                    return set(best_trip_idxs[:3]) | set(pair_idxs[:2])
            # Two sets of trips — use second as pair
            if len(trips) >= 2:
                return set(best_trip_idxs[:3]) | set(trips[1][1][:2])

        # Flush
        if flush_indices:
            return set(flush_indices[:5])

        # Straight
        if straight_indices:
            return straight_indices

        # Three of a Kind
        if 3 in groups_by_size:
            return set(groups_by_size[3][0][1][:3])

        # Two Pair
        if 2 in groups_by_size and len(groups_by_size[2]) >= 2:
            p1 = groups_by_size[2][0][1][:2]
            p2 = groups_by_size[2][1][1][:2]
            return set(p1) | set(p2)

        # Pair
        if 2 in groups_by_size:
            return set(groups_by_size[2][0][1][:2])

        # High Card — play the highest-value cards
        ranked = sorted(range(len(hand)), key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
        return set(ranked[:max_cards])

    def _find_best_hand_brute(self, state: RunState, hand: list[PlayingCard], max_cards: int) -> set[int]:
        """Brute-force fallback for hands with Stone Cards or exotic jokers."""
        hand_order = [
            "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
            "Four of a Kind", "Full House", "Flush", "Straight",
            "Three of a Kind", "Two Pair", "Pair", "High Card",
        ]
        hand_rank = {name: i for i, name in enumerate(hand_order)}
        best_hand_name = "High Card"
        best_indices: set[int] = set()
        best_score = float("-inf")

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
                        score = -rank * 10000 + sum(RANK_TO_NOMINAL.get(c.rank, 0) for c in cards)
                        if score > best_score:
                            best_score = score
                            best_indices = set(combo)
                            best_hand_name = hand_name
                        break

        if best_hand_name == "High Card" or not best_indices:
            ranked = sorted(range(len(hand)), key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
            best_indices = set(ranked[:max_cards])
        return best_indices

    def _find_worst_cards(self, state: RunState, hand: list[PlayingCard], max_discard: int) -> set[int]:
        suits = [c.suit for c in hand]
        ranks = [c.rank for c in hand]
        suit_counts = Counter(suits)
        rank_counts = Counter(ranks)
        nominals = [RANK_TO_NOMINAL.get(r, 0) for r in ranks]
        sorted_nominals = sorted(nominals)

        keep_scores: list[float] = []
        for i, card in enumerate(hand):
            score = nominals[i] * 0.1

            suit_n = suit_counts[card.suit]
            if suit_n >= 4:
                score += 20.0
            elif suit_n >= 3:
                score += 8.0

            rank_n = rank_counts[card.rank]
            if rank_n >= 3:
                score += 25.0
            elif rank_n >= 2:
                score += 15.0

            nom = nominals[i]
            lo = max(nom - 4, 0)
            hi = nom + 4
            lo_idx = 0
            for j in range(len(sorted_nominals)):
                if sorted_nominals[j] >= lo:
                    lo_idx = j
                    break
            nearby = 0
            for j in range(lo_idx, len(sorted_nominals)):
                if sorted_nominals[j] > hi:
                    break
                nearby += 1
            nearby -= 1  # exclude self
            if nearby >= 3:
                score += 5.0

            keep_scores.append(score)

        ranked = sorted(range(len(hand)), key=lambda idx: keep_scores[idx])
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
