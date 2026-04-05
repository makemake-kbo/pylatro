"""Rule-based heuristic agent for generating supervised pretraining data."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from pylatro import can_use_consumable, evaluate_poker_hand
from pylatro.runtime import consumable_limit, joker_limit
from pylatro.scoring import RANK_TO_ID, RANK_TO_NOMINAL

from .constants import ActionRange, SubPhase
from .subset_actions import subset_index

if TYPE_CHECKING:
    from pylatro.models import PlayingCard, RunState

_LOW_VALUE_JOKERS = frozenset(
    {
        "j_oops",
        "j_chaos",
        "j_credit_card",
        "j_egg",
        "j_diet_cola",
        "j_superposition",
        "j_luchador",
        "j_splash",
        "j_certificate",
        "j_cartomancer",
        "j_invisible",
    }
)


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
        if sub_phase == SubPhase.BLIND_SELECT:
            return self._blind_select(state, action_mask)
        elif sub_phase == SubPhase.CHOOSE_ACTION:
            return self._choose_action(state, action_mask)
        elif sub_phase == SubPhase.SELECT_CARDS:
            return self._random_valid(action_mask)
        elif sub_phase == SubPhase.SHOP:
            return self._shop(state, action_mask)
        elif sub_phase == SubPhase.BOOSTER_PACK:
            return self._booster_pack(state, action_mask)
        elif sub_phase == SubPhase.CONSUMABLE_TARGET:
            return self._consumable_target(state, action_mask, kwargs.get("pending_consumable_slot"))
        return self._random_valid(action_mask)

    def _blind_select(self, state: RunState, mask: np.ndarray) -> int:
        if mask[ActionRange.BLIND_PLAY]:
            return ActionRange.BLIND_PLAY
        return self._random_valid(mask)

    def _choose_action(self, state: RunState, mask: np.ndarray) -> int:
        if mask[ActionRange.USE_CONSUMABLE]:
            for cons in state.consumables:
                center = state.data.centers[cons.center_key]
                cset = center.get("set", "")
                if cset == "Planet" and can_use_consumable(state, cons):
                    return ActionRange.USE_CONSUMABLE
            for cons in state.consumables:
                center = state.data.centers[cons.center_key]
                cset = center.get("set", "")
                if cset == "Tarot" and can_use_consumable(state, cons):
                    return ActionRange.USE_CONSUMABLE

        hand = state.hand_cards
        best_play = tuple(sorted(self._cached_best_hand(state, hand)))
        best_discard = tuple(sorted(self._find_worst_cards(state, hand, max_discard=min(5, len(hand)))))
        can_discard = (
            bool(best_discard)
            and state.current_round.discards_left > 0
            and mask[ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)]
        )
        hand_quality = self._evaluate_hand_quality(state, hand) if hand else 0

        if hand_quality < 1 and can_discard and state.current_round.hands_left > 1:
            return ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)

        if best_play:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return play_action

        if can_discard:
            return ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)

        valid_play = np.where(mask[ActionRange.PLAY_SUBSET_START:ActionRange.PLAY_SUBSET_END + 1] == 1)[0]
        if len(valid_play) > 0:
            return ActionRange.PLAY_SUBSET_START + int(valid_play[0])
        valid_discard = np.where(mask[ActionRange.DISCARD_SUBSET_START:ActionRange.DISCARD_SUBSET_END + 1] == 1)[0]
        if len(valid_discard) > 0:
            return ActionRange.DISCARD_SUBSET_START + int(valid_discard[0])
        return self._random_valid(mask)

    def _evaluate_hand_quality(self, state: RunState, hand: list[PlayingCard]) -> int:
        quality_map = {
            "Flush Five": 8,
            "Flush House": 7,
            "Five of a Kind": 6,
            "Straight Flush": 5,
            "Four of a Kind": 4,
            "Full House": 3,
            "Flush": 3,
            "Straight": 3,
            "Three of a Kind": 2,
            "Two Pair": 2,
            "Pair": 1,
            "High Card": 0,
        }
        best = self._cached_best_hand(state, hand)
        cards = [hand[i] for i in best]
        quality_name = self._quick_hand_quality(state, cards)
        base_quality = quality_map.get(quality_name, 0)

        for j in state.jokers:
            jc = state.data.centers.get(j.center_key, {})
            jcfg = jc.get("config")
            if not isinstance(jcfg, dict):
                continue
            jtype = jcfg.get("type", "")
            if jtype == quality_name:
                base_quality += 2
                t_mult = jcfg.get("t_mult", 0) or 0
                if t_mult >= 8:
                    base_quality += 1
                xmult = jcfg.get("Xmult", 0) or 0
                if xmult > 1:
                    base_quality += 2

        return base_quality

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
            is_straight = sorted_r[len(sorted_r) - 1] - sorted_r[0] == n - 1
            if not is_straight and 14 in sorted_r:
                low = sorted(set(1 if r == 14 else r for r in ranks))
                if len(low) >= 5 and len(low) == n and low[len(low) - 1] - low[0] == n - 1:
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

    def _score_joker_value(self, state: RunState, center_key: str) -> float:
        center = state.data.centers.get(center_key, {})
        config = center.get("config")
        if not config or isinstance(config, list):
            config = {}

        cost = center.get("cost", 5)
        score = 0.0

        if center_key in _LOW_VALUE_JOKERS:
            return -100.0

        mult = config.get("mult")
        if isinstance(mult, (int, float)) and mult:
            score += mult * 4.0

        t_mult = config.get("t_mult")
        hand_type = config.get("type", "")
        if isinstance(t_mult, (int, float)) and t_mult:
            if self._hand_type_synergy(state, hand_type) > 0:
                score += t_mult * 4.0
            else:
                score += t_mult * 1.5

        t_chips = config.get("t_chips")
        if isinstance(t_chips, (int, float)) and t_chips:
            if self._hand_type_synergy(state, hand_type) > 0:
                score += t_chips * 0.5
            else:
                score += t_chips * 0.15

        x_mult = config.get("Xmult")
        if isinstance(x_mult, (int, float)) and x_mult and x_mult > 1:
            total_add = sum(j.mult for j in state.jokers) + sum(j.t_mult for j in state.jokers)
            if total_add >= 8:
                score += (x_mult - 1) * 30.0
            elif total_add >= 4:
                score += (x_mult - 1) * 15.0
            else:
                score += (x_mult - 1) * 3.0

        extra = config.get("extra")
        if isinstance(extra, dict):
            s_mult = extra.get("s_mult")
            if isinstance(s_mult, (int, float)) and s_mult:
                score += s_mult * 2.5

            chip_mod = extra.get("chip_mod")
            if isinstance(chip_mod, (int, float)) and chip_mod:
                score += chip_mod * 2.0

            hand_add = extra.get("hand_add")
            if isinstance(hand_add, (int, float)) and hand_add:
                score += hand_add * 3.0

            ex_mult = extra.get("Xmult")
            if isinstance(ex_mult, (int, float)) and ex_mult:
                score += ex_mult * 5.0

            mult_val = extra.get("mult")
            if isinstance(mult_val, (int, float)) and mult_val:
                score += mult_val * 3.0

            chips_val = extra.get("chips")
            if isinstance(chips_val, (int, float)) and chips_val:
                score += chips_val * 0.3
        elif isinstance(extra, (int, float)) and extra and extra > 0:
            effect = center.get("effect", "")
            if "Mult" in effect:
                score += extra * 3.0
            elif "Chip" in effect:
                score += extra * 0.5
            elif "Card Buff" in effect:
                score += extra * 2.0
            elif extra >= 10:
                score += extra * 1.0
            else:
                score += extra * 2.0

        h_size = config.get("h_size")
        if isinstance(h_size, (int, float)) and h_size and h_size > 0:
            score += h_size * 12.0

        d_size = config.get("d_size")
        if isinstance(d_size, (int, float)) and d_size and d_size > 0:
            score += d_size * 8.0

        if center_key == "j_four_fingers":
            score += 10.0
        elif center_key in ("j_blueprint", "j_brainstorm"):
            score += 15.0 if len(state.jokers) >= 2 else 3.0

        if cost <= 3:
            score *= 1.2
        elif cost >= 8:
            score *= 0.8

        return score

    def _hand_type_synergy(self, state: RunState, hand_type: str) -> float:
        if not hand_type:
            return 0.0
        synergy = 0.0
        for j in state.jokers:
            jc = state.data.centers.get(j.center_key, {})
            jcfg = jc.get("config")
            if not isinstance(jcfg, dict):
                continue
            jtype = jcfg.get("type", "")
            if jtype == hand_type:
                synergy += jcfg.get("t_mult", 0) or 0
                synergy += (jcfg.get("t_chips", 0) or 0) * 0.1
                xm = jcfg.get("Xmult", 0) or 0
                if xm > 1:
                    synergy += (xm - 1) * 10
        return synergy

    def _select_cards(self, state: RunState, mask: np.ndarray, selected: set[int], pending: str | None) -> int:
        _ = (state, selected, pending)
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

        four_fingers_flag = state.has_joker("Four Fingers")
        flush_req = 4 if four_fingers_flag else 5
        straight_req = 4 if four_fingers_flag else 5

        if has_stone or state.has_joker("Shortcut") or state.has_joker("Pareidolia"):
            return self._find_best_hand_brute(state, hand, max_cards)

        groups = sorted(by_rank.items(), key=lambda x: x[0], reverse=True)
        groups_by_size: dict[int, list[tuple[int, list[int]]]] = {}
        for cid, indices in groups:
            n = len(indices)
            for sz in range(1, n + 1):
                groups_by_size.setdefault(sz, []).append((cid, indices))

        def _best_of(indices_set: set[int]) -> float:
            return sum(RANK_TO_NOMINAL.get(hand[i].rank, 0) for i in indices_set)

        flush_suit: str | None = None
        flush_indices: list[int] | None = None
        for suit, idxs in by_suit.items():
            unique = list(dict.fromkeys(idxs))
            if len(unique) >= flush_req:
                unique.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                flush_indices = unique[:5]
                flush_suit = suit
                break

        def _find_straight() -> set[int] | None:
            sorted_ranks = sorted(by_rank.keys(), reverse=True)
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
                        result.add(by_rank[r][0])
                    return result
            return None

        straight_indices = _find_straight()

        if 5 in groups_by_size:
            cid, idxs = groups_by_size[5][0]
            chosen = set(idxs[:5])
            if flush_indices and chosen <= set(flush_indices):
                return chosen
            return chosen

        if 4 in groups_by_size:
            best_four: set[int] | None = None
            best_four_score = -1.0
            for _cid, idxs in groups_by_size[4]:
                s = set(idxs[:4])
                sc = _best_of(s)
                if sc > best_four_score:
                    best_four = s
                    best_four_score = sc

            if flush_indices and straight_indices:
                sf_set = set(flush_indices) & straight_indices
                if len(sf_set) >= straight_req:
                    return set(list(sf_set)[:5])
                flush_set = set(flush_indices) if flush_indices else set()
                if straight_indices and len(flush_set & straight_indices) >= straight_req:
                    return set(list(flush_set & straight_indices)[:5])

            if best_four is not None:
                return best_four

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

        if 3 in groups_by_size and 2 in groups_by_size:
            trips = groups_by_size[3]
            pairs = groups_by_size[2]
            best_trip_cid, best_trip_idxs = trips[0]
            for pair_cid, pair_idxs in pairs:
                if pair_cid != best_trip_cid:
                    return set(best_trip_idxs[:3]) | set(pair_idxs[:2])
            if len(trips) >= 2:
                return set(best_trip_idxs[:3]) | set(trips[1][1][:2])

        if flush_indices:
            return set(flush_indices[:5])

        if straight_indices:
            return straight_indices

        if 3 in groups_by_size:
            return set(groups_by_size[3][0][1][:3])

        if 2 in groups_by_size and len(groups_by_size[2]) >= 2:
            p1 = groups_by_size[2][0][1][:2]
            p2 = groups_by_size[2][1][1][:2]
            return set(p1) | set(p2)

        if 2 in groups_by_size:
            return set(groups_by_size[2][0][1][:2])

        ranked = sorted(range(len(hand)), key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
        return set(ranked[:max_cards])

    def _find_best_hand_brute(self, state: RunState, hand: list[PlayingCard], max_cards: int) -> set[int]:
        hand_order = [
            "Flush Five",
            "Flush House",
            "Five of a Kind",
            "Straight Flush",
            "Four of a Kind",
            "Full House",
            "Flush",
            "Straight",
            "Three of a Kind",
            "Two Pair",
            "Pair",
            "High Card",
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
            nearby -= 1
            if nearby >= 3:
                score += 5.0

            keep_scores.append(score)

        ranked = sorted(range(len(hand)), key=lambda idx: keep_scores[idx])
        return set(ranked[:max_discard])

    def _shop(self, state: RunState, mask: np.ndarray) -> int:
        all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
        joker_slots_left = joker_limit(state) - len(state.jokers)

        best_joker_action = -1
        best_joker_score = -1e9

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Joker":
                jscore = self._score_joker_value(state, item.center_key)
                if jscore > best_joker_score:
                    best_joker_score = jscore
                    best_joker_action = action

        if best_joker_action >= 0 and joker_slots_left > 0 and best_joker_score > 0:
            return best_joker_action

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Voucher":
                return action

        if joker_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster":
                    name = center.get("name", "")
                    if "Buffoon" in name:
                        return action

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Booster":
                name = center.get("name", "")
                if "Celestial" in name:
                    return action

        cons_slots_left = consumable_limit(state) - len(state.consumables)
        if cons_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster":
                    name = center.get("name", "")
                    if "Arcana" in name:
                        return action

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Booster":
                return action

        if best_joker_action >= 0 and joker_slots_left > 0 and best_joker_score > -5:
            return best_joker_action

        if mask[ActionRange.SHOP_REROLL] and state.dollars >= 8 and joker_slots_left > 0:
            return ActionRange.SHOP_REROLL

        if mask[ActionRange.SHOP_LEAVE]:
            return ActionRange.SHOP_LEAVE
        return self._random_valid(mask)

    def _booster_pack(self, state: RunState, mask: np.ndarray) -> int:
        pack = state.pack
        if pack and pack.cards:
            best_score = -1e9
            best_action = -1
            for i, card in enumerate(pack.cards):
                action = ActionRange.PACK_CLAIM_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(card.center_key, {})
                cscore = self._score_pack_card(state, center)
                if cscore > best_score:
                    best_score = cscore
                    best_action = action
            if best_action >= 0:
                return best_action

        for i in range(5):
            action = ActionRange.PACK_CLAIM_START + i
            if mask[action]:
                return action
        if mask[ActionRange.PACK_SKIP]:
            return ActionRange.PACK_SKIP
        return self._random_valid(mask)

    def _score_pack_card(self, state: RunState, center: dict) -> float:
        cset = center.get("set", "")
        if cset == "Joker":
            jslots = joker_limit(state) - len(state.jokers)
            if jslots > 0:
                return self._score_joker_value(state, center.get("key", ""))
            return -100.0
        elif cset == "Planet":
            return 5.0
        elif cset == "Tarot":
            return 4.0
        elif cset == "Spectral":
            return 3.0
        return 1.0

    def _consumable_target(self, state: RunState, mask: np.ndarray, pending_slot: int | None) -> int:
        if pending_slot is None:
            for i, cons in enumerate(state.consumables):
                action = ActionRange.CONSUMABLE_SLOT_START + i
                if mask[action]:
                    center = state.data.centers[cons.center_key]
                    cset = center.get("set", "")
                    if cset == "Planet":
                        return action
            for i, cons in enumerate(state.consumables):
                action = ActionRange.CONSUMABLE_SLOT_START + i
                if mask[action]:
                    return action

        if mask[ActionRange.CONSUMABLE_HAND_TARGET_START]:
            hand = state.hand_cards
            for i in range(min(len(hand), 5)):
                action = ActionRange.CONSUMABLE_HAND_TARGET_START + i
                if mask[action]:
                    return action

        if mask[ActionRange.CONSUMABLE_CONFIRM]:
            return ActionRange.CONSUMABLE_CONFIRM
        if mask[ActionRange.CONSUMABLE_CANCEL]:
            return ActionRange.CONSUMABLE_CANCEL
        return self._random_valid(mask)

    def _random_valid(self, mask: np.ndarray) -> int:
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))
