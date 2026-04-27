"""Rule-based heuristic agent for generating supervised pretraining data."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from pylatro import can_use_consumable, evaluate_poker_hand, get_blind_amount
from pylatro.runtime import consumable_limit, joker_limit
from pylatro.scoring import RANK_TO_ID, RANK_TO_NOMINAL

from .action import ActionType, encode_action
from .constants import (
    HAND_TARGET_CONSUMABLE_LIMITS,
    JOKER_TARGET_CONSUMABLE_NAMES,
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_JOKER_SLOTS,
    ActionRange,
    SubPhase,
)
from .subset_actions import consumable_subset_index, subset_index

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
        "j_faceless",
        "j_delayed_grat",
        "j_matador",
        "j_troubadour",
        "j_turtle_bean",
        "j_space",
        "j_seance",
        "j_midas_mask",
        "j_marble",
        "j_hallucination",
        "j_ramen",
        "j_vagabond",
        "j_sixth_sense",
        "j_satellite",
        "j_gift",
        "j_trading",
        "j_ring_master",
        "j_dna",
        "j_raised_fist",
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

    def _get_blind_target(self, state: RunState) -> int:
        blind = state.round_resets.blind
        if blind is None:
            return 0
        ante = state.round_resets.ante
        scaling = min(state.stake, 3)
        base = get_blind_amount(ante, scaling)
        mult = blind.get("mult", 1)
        return int(base * mult)

    def _estimate_hand_score(self, state: RunState, hand_indices: tuple[int, ...]) -> int:
        if not hand_indices:
            return 0
        hand = state.hand_cards
        cards = [hand[i] for i in hand_indices]
        hand_type = self._quick_hand_quality(state, cards)
        hand_info = state.hands.get(hand_type, {})
        base_chips = hand_info.get("chips", 0)
        base_mult = hand_info.get("mult", 0)

        _SCORING_COUNT = {
            "High Card": 1, "Pair": 2, "Two Pair": 4, "Three of a Kind": 3,
            "Straight": 5, "Flush": 5, "Full House": 5, "Four of a Kind": 4,
            "Five of a Kind": 5, "Straight Flush": 5, "Flush Five": 5, "Flush House": 5,
        }
        n_scoring = _SCORING_COUNT.get(hand_type, len(cards))

        card_chip_values = []
        for c in cards:
            center = state.data.centers.get(c.center_key)
            effect = center.get("effect", "") if center else ""
            if effect == "Stone Card":
                card_chip_values.append(50 + c.perma_bonus)
            else:
                card_chip_values.append(RANK_TO_NOMINAL.get(c.rank, 0) + c.perma_bonus)
        card_chip_values.sort(reverse=True)
        card_chips = sum(card_chip_values[:n_scoring])

        total_chips = base_chips + card_chips
        total_mult = base_mult
        x_mult_acc = 1.0

        for j in state.jokers:
            if j.debuff:
                continue

            if j.mult:
                total_mult += j.mult

            if j.t_mult and (not j.type or j.type == hand_type):
                total_mult += j.t_mult

            if j.t_chips and (not j.type or j.type == hand_type):
                total_chips += j.t_chips

            if j.x_mult and j.x_mult > 1 and (not j.type or j.type == hand_type):
                x_mult_acc *= j.x_mult

            jname = state.data.centers.get(j.center_key, {}).get("name", "")

            extra = j.extra
            if isinstance(extra, dict):
                em = extra.get("mult")
                if isinstance(em, (int, float)) and em:
                    total_mult += em
                ec = extra.get("chips")
                if isinstance(ec, (int, float)) and ec:
                    total_chips += ec
                exm = extra.get("Xmult")
                if isinstance(exm, (int, float)) and exm and exm > 1:
                    x_mult_acc *= exm

            if jname == "Fibonacci":
                for c in cards[:n_scoring]:
                    if c.rank in ("2", "3", "5", "8", "14", "Ace"):
                        total_mult += 8
            elif jname == "Even Steven":
                for c in cards[:n_scoring]:
                    if c.rank in ("2", "4", "6", "8", "10"):
                        total_mult += 4
            elif jname == "Odd Todd":
                for c in cards[:n_scoring]:
                    if c.rank in ("Ace", "3", "5", "7", "9"):
                        total_chips += 31
            elif jname == "Scary Face":
                for c in cards[:n_scoring]:
                    if c.rank in ("Jack", "Queen", "King"):
                        total_chips += 30
            elif jname == "Smiley Face":
                for c in cards[:n_scoring]:
                    if c.rank in ("Jack", "Queen", "King"):
                        total_mult += 5
            elif jname == "Scholar":
                for c in cards[:n_scoring]:
                    if c.rank == "Ace":
                        total_chips += 20
                        total_mult += 4
            elif jname == "Walkie Talkie":
                for c in cards[:n_scoring]:
                    if c.rank in ("4", "10"):
                        total_chips += 10
                        total_mult += 4
            elif jname == "Photograph":
                face_found = False
                for c in cards[:n_scoring]:
                    if c.rank in ("Jack", "Queen", "King") and not face_found:
                        x_mult_acc *= 2
                        face_found = True
            elif jname in ("Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker"):
                suit_map = {
                    "Greedy Joker": "Diamonds",
                    "Lusty Joker": "Hearts",
                    "Wrathful Joker": "Spades",
                    "Gluttonous Joker": "Clubs",
                }
                target_suit = suit_map[jname]
                sv = 3 if isinstance(j.extra, dict) else (j.extra or 3)
                for c in cards[:n_scoring]:
                    if c.suit == target_suit:
                        total_mult += sv
            elif jname == "Onyx Agate":
                for c in cards[:n_scoring]:
                    if c.suit == "Clubs":
                        total_mult += 7
            elif jname == "Arrowhead":
                for c in cards[:n_scoring]:
                    if c.suit == "Spades":
                        total_chips += 50
            elif jname == "Shoot the Moon":
                held = [hand[i] for i in range(len(hand)) if i not in hand_indices]
                for c in held:
                    if c.rank == "Queen":
                        total_mult += 13
            elif jname == "Baron":
                held = [hand[i] for i in range(len(hand)) if i not in hand_indices]
                for c in held:
                    if c.rank == "King":
                        x_mult_acc *= 1.5
            elif jname == "Bootstraps":
                total_mult += (state.dollars // 5) * 2
            elif jname == "Half Joker":
                if len(cards) <= 3:
                    total_mult += 20
            elif jname == "Mystic Summit":
                if state.current_round.discards_left == 0:
                    total_mult += 15
            elif jname == "Banner":
                total_chips += state.current_round.discards_left * 30
            elif jname == "Abstract Joker":
                total_mult += len([jj for jj in state.jokers if not jj.debuff])
            elif jname == "Supernova":
                played = state.hands.get(hand_type, {}).get("played", 0)
                total_mult += played

        return int(total_chips * total_mult * x_mult_acc)

    def _get_main_hand_type(self, state: RunState) -> str:
        best = "Pair"
        best_score = 0.0
        for ht in ("Pair", "Two Pair", "Three of a Kind", "Full House",
                    "Flush", "Straight", "High Card", "Four of a Kind",
                    "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"):
            synergy = self._hand_type_synergy(state, ht)
            played = state.hands.get(ht, {}).get("played", 0)
            level = state.hands.get(ht, {}).get("level", 1)
            score = synergy * 2 + played * 0.5 + level * 3
            if score > best_score:
                best_score = score
                best = ht
        return best

    def _find_type_hand(self, state: RunState, hand: list[PlayingCard], type_name: str) -> tuple[int, ...] | None:
        if not hand:
            return None

        by_rank: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            by_rank.setdefault(card.rank, []).append(i)

        if type_name == "Pair":
            for rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 2:
                    pair = indices[:2]
                    kickers = sorted([i for i in range(len(hand)) if i not in pair], key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                    return tuple(sorted(pair + kickers[:3]))

        elif type_name == "Two Pair":
            pairs: list[int] = []
            for rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 2 and len(pairs) < 4:
                    pairs.extend(indices[:2])
            if len(pairs) >= 4:
                kickers = sorted([i for i in range(len(hand)) if i not in pairs], key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                return tuple(sorted(pairs + kickers[:1]))

        elif type_name == "Three of a Kind":
            for rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 3:
                    trip = indices[:3]
                    kickers = sorted([i for i in range(len(hand)) if i not in trip], key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                    return tuple(sorted(trip + kickers[:2]))

        elif type_name == "Four of a Kind":
            for rank, indices in sorted(by_rank.items(), key=lambda x: RANK_TO_NOMINAL.get(x[0], 0), reverse=True):
                if len(indices) >= 4:
                    quad = indices[:4]
                    kickers = sorted([i for i in range(len(hand)) if i not in quad], key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                    return tuple(sorted(quad + kickers[:1]))

        elif type_name == "Flush":
            by_suit: dict[str, list[int]] = {}
            centers = state.data.centers
            for i, card in enumerate(hand):
                center = centers.get(card.center_key)
                effect = center.get("effect", "") if center else ""
                if effect == "Wild Card":
                    for s in ("Spades", "Hearts", "Clubs", "Diamonds"):
                        by_suit.setdefault(s, []).append(i)
                else:
                    by_suit.setdefault(card.suit, []).append(i)
            for suit, indices in by_suit.items():
                unique = list(dict.fromkeys(indices))
                if len(unique) >= 5:
                    unique.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                    return tuple(sorted(unique[:5]))

        return None

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
        return self._random_valid(action_mask)

    def _blind_select(self, state: RunState, mask: np.ndarray) -> int:
        if mask[ActionRange.BLIND_PLAY]:
            return ActionRange.BLIND_PLAY
        return self._random_valid(mask)

    def _choose_action(self, state: RunState, mask: np.ndarray) -> int:
        hand = state.hand_cards
        if not hand:
            return self._random_valid(mask)

        _MONEY_TAROTS = frozenset({"The Hermit", "Temperance"})
        for slot, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            center = state.data.centers[cons.center_key]
            if center.get("set", "") != "Tarot":
                continue
            name = center.get("name", "")
            if name in _MONEY_TAROTS:
                action = self._atomic_consumable_action(state, slot, mask)
                if action is not None:
                    return action

        best_play = tuple(sorted(self._cached_best_hand(state, hand)))

        if best_play and len(best_play) < 5 and len(hand) > len(best_play):
            remaining = sorted(
                [i for i in range(len(hand)) if i not in best_play],
                key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0),
                reverse=True,
            )
            padded = tuple(sorted(list(best_play) + remaining[: 5 - len(best_play)]))
            orig_score = self._estimate_hand_score(state, best_play)
            pad_score = self._estimate_hand_score(state, padded)
            if pad_score >= orig_score:
                padded_action = ActionRange.PLAY_SUBSET_START + subset_index(padded)
                if mask[padded_action]:
                    best_play = padded

        best_play_cards = [hand[i] for i in best_play] if best_play else []
        best_hand_quality_now = self._quick_hand_quality(state, best_play_cards) if best_play_cards else ""

        est_score = self._estimate_hand_score(state, best_play) if best_play else 0

        main_type = self._get_main_hand_type(state)
        main_synergy = self._hand_type_synergy(state, main_type)

        alt_types = ["Pair", "Two Pair", "Three of a Kind", "Four of a Kind", "Flush"]
        if main_type in alt_types:
            alt_types.remove(main_type)
            alt_types.insert(0, main_type)

        if main_type in ("Pair", "Two Pair", "Three of a Kind"):
            main_hand = self._find_type_hand(state, hand, main_type)
            if main_hand:
                main_score = self._estimate_hand_score(state, main_hand)
                if main_score >= est_score * 0.8:
                    best_play = main_hand
                    est_score = main_score
                    best_play_cards = [hand[i] for i in best_play]
                    best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)

        for alt_type in alt_types:
            if best_hand_quality_now == alt_type:
                continue
            alt_hand = self._find_type_hand(state, hand, alt_type)
            if alt_hand:
                alt_score = self._estimate_hand_score(state, alt_hand)
                if alt_type == main_type and main_synergy > 0:
                    if alt_score > est_score * 0.7:
                        best_play = alt_hand
                        est_score = alt_score
                        best_play_cards = [hand[i] for i in best_play]
                        best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)
                elif alt_score > est_score:
                    best_play = alt_hand
                    est_score = alt_score
                    best_play_cards = [hand[i] for i in best_play]
                    best_hand_quality_now = self._quick_hand_quality(state, best_play_cards)
        blind_target = self._get_blind_target(state)
        hands_left = state.current_round.hands_left
        discards_left = state.current_round.discards_left

        best_planet_score = -1
        best_planet_action = None
        for slot, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            center = state.data.centers[cons.center_key]
            if center.get("set", "") != "Planet":
                continue
            action = self._atomic_consumable_action(state, slot, mask)
            if action is None:
                continue
            planet_hand_type = center.get("config", {}).get("hand_type", "")
            score = 0
            main_type = self._get_main_hand_type(state)
            if planet_hand_type == best_hand_quality_now:
                score += 10
            if planet_hand_type == main_type:
                score += 8
            score += self._hand_type_synergy(state, planet_hand_type)
            played_count = state.hands.get(planet_hand_type, {}).get("played", 0)
            if played_count > 0:
                score += min(played_count, 10)
            if score > best_planet_score:
                best_planet_score = score
                best_planet_action = action
        if best_planet_action is not None:
            return best_planet_action

        if best_play and est_score >= blind_target:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return play_action

        _BUFF_TAROTS = frozenset({
            "The Magician", "The Empress", "The Hierophant", "The Lovers",
            "The Chariot", "Justice", "Strength", "The Star", "The Moon",
            "The Sun", "The World", "The Wheel of Fortune",
        })
        _DESTROY_TAROTS = frozenset({
            "Death", "The Hanged Man", "The Devil", "The Tower", "Judgement",
        })

        for slot, cons in enumerate(state.consumables[:MAX_CONSUMABLE_SLOTS]):
            center = state.data.centers[cons.center_key]
            if center.get("set", "") != "Tarot":
                continue
            name = center.get("name", "")
            preferred = None
            if name in _BUFF_TAROTS:
                preferred = best_play if best_play else None
            elif name in _DESTROY_TAROTS:
                worst = tuple(sorted(self._find_worst_cards(state, hand, max_discard=2)))
                preferred = worst if worst else None
            action = self._atomic_consumable_action(state, slot, mask, preferred_indices=preferred)
            if action is not None:
                return action

        if best_play and est_score * hands_left >= blind_target:
            play_action = ActionRange.PLAY_SUBSET_START + subset_index(best_play)
            if mask[play_action]:
                return play_action

        draw_discard = self._should_discard_for_draw(state, hand, mask)
        if draw_discard is not None:
            return ActionRange.DISCARD_SUBSET_START + subset_index(draw_discard)

        best_discard = tuple(sorted(self._find_worst_cards(state, hand, max_discard=min(5, len(hand))))) if hand else ()
        can_discard = (
            bool(best_discard)
            and discards_left > 0
            and mask[ActionRange.DISCARD_SUBSET_START + subset_index(best_discard)]
        )

        if can_discard and hands_left > 1 and est_score * hands_left < blind_target:
            if not self._has_scaling_jokers(state):
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
        jname = center.get("name", "")

        if center_key in _LOW_VALUE_JOKERS:
            return -100.0

        main_type = self._get_main_hand_type(state)
        has_main_synergy = len(state.jokers) > 0 and self._hand_type_synergy(state, main_type) > 0

        mult = config.get("mult")
        if isinstance(mult, (int, float)) and mult:
            score += mult * 5.0

        t_mult = config.get("t_mult")
        hand_type = config.get("type", "")
        _COMMITTED_TYPES = {"Pair", "High Card", "Two Pair", "Three of a Kind"}
        if isinstance(t_mult, (int, float)) and t_mult:
            if hand_type == main_type:
                score += t_mult * 8.0
            elif self._hand_type_synergy(state, hand_type) > 0:
                score += t_mult * 6.0
            elif hand_type in _COMMITTED_TYPES and not has_main_synergy:
                score += t_mult * 5.0
            elif hand_type in _COMMITTED_TYPES:
                score += t_mult * 2.0
            else:
                score += t_mult * 0.5

        t_chips = config.get("t_chips")
        if isinstance(t_chips, (int, float)) and t_chips:
            if hand_type == main_type:
                score += t_chips * 1.5
            elif self._hand_type_synergy(state, hand_type) > 0:
                score += t_chips * 0.8
            elif hand_type in _COMMITTED_TYPES and not has_main_synergy:
                score += t_chips * 0.4
            else:
                score += t_chips * 0.1

        x_mult = config.get("Xmult")
        if isinstance(x_mult, (int, float)) and x_mult and x_mult > 1:
            total_add = sum(j.mult for j in state.jokers) + sum(j.t_mult for j in state.jokers)
            base_xmult_score = (x_mult - 1) * 20.0
            if hand_type == main_type:
                base_xmult_score *= 4.0
            elif not hand_type:
                base_xmult_score *= 3.0
            elif hand_type and hand_type != main_type:
                base_xmult_score *= 0.5
            if total_add >= 8:
                base_xmult_score *= 2.5
            elif total_add >= 4:
                base_xmult_score *= 2.0
            score += base_xmult_score

        extra = config.get("extra")
        if isinstance(extra, dict):
            s_mult = extra.get("s_mult")
            if isinstance(s_mult, (int, float)) and s_mult:
                score += s_mult * 6.0

            chip_mod = extra.get("chip_mod")
            if isinstance(chip_mod, (int, float)) and chip_mod:
                score += chip_mod * 4.0

            hand_add = extra.get("hand_add")
            if isinstance(hand_add, (int, float)) and hand_add:
                score += hand_add * 8.0

            ex_mult = extra.get("Xmult")
            if isinstance(ex_mult, (int, float)) and ex_mult:
                score += ex_mult * 10.0

            mult_val = extra.get("mult")
            if isinstance(mult_val, (int, float)) and mult_val:
                score += mult_val * 3.0

            chips_val = extra.get("chips")
            if isinstance(chips_val, (int, float)) and chips_val:
                score += chips_val * 0.5

            dollars_extra = extra.get("dollars")
            if isinstance(dollars_extra, (int, float)) and dollars_extra:
                score += dollars_extra * 10.0
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

        _PER_CARD_JOKERS = {
            "j_fibonacci": 5.0, "j_even_steven": 4.0, "j_odd_todd": 2.5,
            "j_smiley": 4.0, "j_scary_face": 2.0, "j_scholar": 4.0,
            "j_walkie_talkie": 3.0, "j_photograph": 8.0,
            "j_greedy_joker": 4.0, "j_lusty_joker": 4.0,
            "j_wrathful_joker": 4.0, "j_gluttenous_joker": 4.0,
            "j_onyx_agate": 5.0, "j_arrowhead": 3.0,
        }
        if center_key in _PER_CARD_JOKERS:
            score += _PER_CARD_JOKERS[center_key]

        _HELD_CARD_JOKERS = {
            "j_baron": 8.0, "j_shoot_the_moon": 6.0, "j_raised_fist": 3.0,
        }
        if center_key in _HELD_CARD_JOKERS:
            score += _HELD_CARD_JOKERS[center_key]

        if jname == "Bootstraps":
            score += 6.0
        elif jname == "Banner":
            score += 3.0
        elif jname == "Abstract Joker":
            score += 2.0
        elif jname == "Mystic Summit":
            score += 5.0
        elif jname == "Half Joker":
            score += 2.0
        elif jname == "Supernova":
            score += 3.0
        elif jname == "Cavendish":
            score += 8.0
        elif jname == "Gros Michel":
            score += 6.0
        elif jname == "Misprint":
            score += 4.0
        elif jname == "Hanging Chad":
            score += 5.0
        elif jname == "Sock and Buskin":
            score += 6.0
        elif jname == "Seltzer":
            score += 5.0
        elif jname == "Mime":
            score += 5.0
        elif jname == "Mr. Bones":
            score += 4.0
        elif jname == "Burglar":
            score += 6.0

        if cost <= 3:
            score *= 1.2
        elif cost >= 8:
            x_mult = config.get("Xmult")
            if isinstance(x_mult, (int, float)) and x_mult and x_mult > 1:
                score *= 0.95
            else:
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

        suit_bonus = {}
        suit_map = {
            "j_greedy_joker": ("Diamonds", 3),
            "j_lusty_joker": ("Hearts", 3),
            "j_wrathful_joker": ("Spades", 3),
            "j_gluttenous_joker": ("Clubs", 3),
            "j_onyx_agate": ("Clubs", 7),
            "j_arrowhead": ("Spades", 50),
        }
        for key, (suit, bonus) in suit_map.items():
            for j in state.jokers:
                if j.center_key == key:
                    suit_bonus[suit] = suit_bonus.get(suit, 0) + bonus

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

        def _suit_value(indices_set: set[int]) -> float:
            v = 0.0
            for i in indices_set:
                for suit, bonus in suit_bonus.items():
                    if hand[i].suit == suit:
                        v += bonus
            return v

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

        if flush_indices and suit_bonus:
            flush_suit_best = None
            flush_best_sv = -1.0
            for suit, idxs in by_suit.items():
                unique = list(dict.fromkeys(idxs))
                if len(unique) >= flush_req:
                    sv = sum(suit_bonus.get(suit, 0) for i in unique[:5])
                    if sv > flush_best_sv:
                        flush_best_sv = sv
                        flush_suit_best = suit
            if flush_suit_best and flush_best_sv > 0:
                idxs = by_suit[flush_suit_best]
                unique = list(dict.fromkeys(idxs))
                unique.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0), reverse=True)
                return set(unique[:5])

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
            pair_idxs = groups_by_size[2][0][1][:2]
            if suit_bonus and len(hand) > 2:
                remaining = sorted(
                    [i for i in range(len(hand)) if i not in pair_idxs],
                    key=lambda i: (suit_bonus.get(hand[i].suit, 0), RANK_TO_NOMINAL.get(hand[i].rank, 0)),
                    reverse=True,
                )
                return set(pair_idxs) | set(remaining[:3])
            return set(pair_idxs)

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

        main_type = self._get_main_hand_type(state)
        main_synergy = self._hand_type_synergy(state, main_type)

        keep_scores: list[float] = []
        for i, card in enumerate(hand):
            score = nominals[i] * 0.1

            rank_n = rank_counts[card.rank]
            if main_type in ("Pair", "Two Pair", "Three of a Kind", "Full House", "Four of a Kind") and main_synergy > 0:
                if rank_n >= 3:
                    score += 30.0
                elif rank_n >= 2:
                    score += 20.0
                else:
                    score += 0.0
            else:
                if rank_n >= 3:
                    score += 25.0
                elif rank_n >= 2:
                    score += 15.0

            suit_n = suit_counts[card.suit]
            if main_type == "Flush" and main_synergy > 0:
                if suit_n >= 4:
                    score += 25.0
                elif suit_n >= 3:
                    score += 12.0
            else:
                if suit_n >= 4:
                    score += 20.0
                elif suit_n >= 3:
                    score += 8.0

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

            if card.rank in ("Ace", "King", "Queen", "Jack", "10"):
                score += 2.0

            keep_scores.append(score)

        ranked = sorted(range(len(hand)), key=lambda idx: keep_scores[idx])
        return set(ranked[:max_discard])

    def _has_scaling_jokers(self, state: RunState) -> bool:
        _SCALING_KEYS = {
            "j_runner", "j_square", "j_castle", "j_trousers", "j_ride_the_bus",
            "j_green_joker", "j_flash", "j_fortune_teller", "j_supernova",
            "j_obelisk", "j_steel_joker", "j_constellation", "j_hologram",
            "j_campfire", "j_hit_the_road", "j_cavendish",
        }
        for j in state.jokers:
            if j.center_key in _SCALING_KEYS:
                return True
        return False

    def _should_discard_for_draw(self, state: RunState, hand: list[PlayingCard], mask: np.ndarray) -> tuple[int, ...] | None:
        if state.current_round.discards_left <= 0 or state.current_round.hands_left < 2:
            return None
        best = self._cached_best_hand(state, hand)
        best_cards = [hand[i] for i in best]
        best_quality = self._quick_hand_quality(state, best_cards)
        quality_above_sf = {"Flush Five", "Flush House", "Five of a Kind", "Straight Flush"}

        main_type = self._get_main_hand_type(state)
        main_synergy = self._hand_type_synergy(state, main_type)

        by_rank: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            by_rank.setdefault(card.rank, []).append(i)

        if main_type in ("Pair", "Two Pair", "Three of a Kind"):
            non_group_cards = []
            for i, card in enumerate(hand):
                if len(by_rank.get(card.rank, [])) < 2:
                    non_group_cards.append(i)
            non_group_cards.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
            to_discard = tuple(sorted(non_group_cards[:5]))
            if to_discard and self._discard_mask_ok(state, mask, to_discard):
                return to_discard

        centers = state.data.centers
        by_suit: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            center = centers.get(card.center_key)
            effect = center.get("effect", "") if center else ""
            if effect == "Wild Card":
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

        # Rule 1: Flush draw
        if best_quality not in quality_above_sf:
            for suit, idxs in by_suit.items():
                unique = list(dict.fromkeys(idxs))
                if len(unique) >= 4:
                    suit_set = set(unique)
                    off_suit = [i for i in range(len(hand)) if i not in suit_set]
                    off_suit.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                    to_discard = tuple(sorted(off_suit[:5]))
                    if to_discard and self._discard_mask_ok(state, mask, to_discard):
                        return to_discard

        # Rule 2: Straight draw
        if not state.has_joker("Shortcut"):
            rank_ids = [(i, RANK_TO_ID[hand[i].rank]) for i in range(len(hand))]
            all_ranks = set()
            rank_to_indices: dict[int, list[int]] = {}
            for i, rid in rank_ids:
                all_ranks.add(rid)
                rank_to_indices.setdefault(rid, []).append(i)
                if rid == 14:
                    all_ranks.add(1)
                    rank_to_indices.setdefault(1, []).append(i)
            for low in range(1, 11):
                window = set(range(low, low + 5))
                covered = window & all_ranks
                if len(covered) >= 4:
                    window_indices: set[int] = set()
                    for r in window:
                        if r in rank_to_indices:
                            for idx in rank_to_indices[r]:
                                window_indices.add(idx)
                                break
                    outside = [i for i in range(len(hand)) if i not in window_indices]
                    outside.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                    to_discard = tuple(sorted(outside[:5]))
                    if to_discard and self._discard_mask_ok(state, mask, to_discard):
                        return to_discard

        # Rule 3: Pair -> Trips/Two-Pair
        if best_quality == "Pair" and state.current_round.hands_left >= 2:
            pair_ranks: dict[int, list[int]] = {}
            for i in range(len(hand)):
                rid = RANK_TO_ID[hand[i].rank]
                pair_ranks.setdefault(rid, []).append(i)
            pair_rank_id = None
            for rid, idxs in pair_ranks.items():
                if len(idxs) >= 2:
                    pair_rank_id = rid
                    break
            if pair_rank_id is not None:
                non_pair = [i for i in range(len(hand)) if RANK_TO_ID[hand[i].rank] != pair_rank_id]
                non_pair.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                to_discard = tuple(sorted(non_pair[:3]))
                if to_discard and self._discard_mask_ok(state, mask, to_discard):
                    return to_discard

        # Rule 4: Trips -> Quad/Full House
        if best_quality == "Three of a Kind":
            trip_ranks: dict[int, list[int]] = {}
            for i in range(len(hand)):
                rid = RANK_TO_ID[hand[i].rank]
                trip_ranks.setdefault(rid, []).append(i)
            trip_rank_id = None
            for rid, idxs in trip_ranks.items():
                if len(idxs) >= 3:
                    trip_rank_id = rid
                    break
            if trip_rank_id is not None:
                non_trip = [i for i in range(len(hand)) if RANK_TO_ID[hand[i].rank] != trip_rank_id]
                non_trip.sort(key=lambda i: RANK_TO_NOMINAL.get(hand[i].rank, 0))
                to_discard = tuple(sorted(non_trip[:2]))
                if to_discard and self._discard_mask_ok(state, mask, to_discard):
                    return to_discard

        return None

    def _discard_mask_ok(self, state: RunState, mask: np.ndarray, indices: tuple[int, ...]) -> bool:
        return bool(mask[ActionRange.DISCARD_SUBSET_START + subset_index(indices)])

    def _shop(self, state: RunState, mask: np.ndarray) -> int:
        all_items = list(state.shop.cards) + list(state.shop.vouchers) + list(state.shop.boosters)
        joker_slots_left = joker_limit(state) - len(state.jokers)
        cons_slots_left = consumable_limit(state) - len(state.consumables)
        dollars = state.dollars
        reroll_cost = state.current_round.reroll_cost
        ante = state.round_resets.ante
        main_type = self._get_main_hand_type(state)

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

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Voucher":
                return action

        _SHOP_MONEY_TAROTS = frozenset({"The Hermit", "Temperance"})
        if cons_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                name = center.get("name", "")
                if name in _SHOP_MONEY_TAROTS and item.cost <= 4:
                    return action

        if joker_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Joker" and item.center_key == "j_joker":
                    return action

        n_jokers = len(state.jokers)
        joker_threshold = -5 if ante <= 2 else (0 if ante <= 4 else 5)
        if best_joker_action >= 0 and joker_slots_left > 0 and best_joker_score > joker_threshold:
            item_idx = best_joker_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                item_cost = all_items[item_idx].cost
                post_buy = dollars - item_cost
                if post_buy >= 4 or best_joker_score > 15:
                    return best_joker_action

        if len(state.jokers) > 0 and best_joker_score > 0:
            worst_score = 1e9
            worst_sell_action = -1
            for i, j in enumerate(state.jokers[:MAX_JOKER_SLOTS]):
                if j.eternal:
                    continue
                jscore = self._score_joker_value(state, j.center_key)
                if jscore < worst_score:
                    worst_score = jscore
                    worst_sell_action = ActionRange.SHOP_SELL_JOKER_START + i
            if worst_sell_action >= 0 and mask[worst_sell_action]:
                if best_joker_action >= 0:
                    best_item = all_items[best_joker_action - ActionRange.SHOP_BUY_START]
                    best_center = state.data.centers.get(best_item.center_key, {})
                    best_cfg = best_center.get("config", {})
                    if not isinstance(best_cfg, dict):
                        best_cfg = {}
                    best_xmult = best_cfg.get("Xmult", 0)
                    best_jtype = best_cfg.get("type", "")
                    if best_xmult and best_xmult > 1 and best_jtype == main_type:
                        if worst_score < best_joker_score:
                            return worst_sell_action
                if best_xmult and best_xmult > 1 and best_jtype == main_type:
                    if worst_score < best_joker_score:
                        return worst_sell_action
                if len(state.jokers) >= 5 and worst_score < best_joker_score - 10:
                    return worst_sell_action
                if worst_score < -50 and dollars < 5:
                    return worst_sell_action

        if n_jokers < 3 and joker_slots_left > 0 and mask[ActionRange.SHOP_REROLL] and dollars >= reroll_cost + 6 and ante <= 3:
            return ActionRange.SHOP_REROLL

        if cons_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Planet":
                    planet_type = center.get("config", {}).get("hand_type", "")
                    if planet_type == main_type:
                        return action
                    if self._hand_type_synergy(state, planet_type) > 0:
                        return action

        if joker_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster":
                    name = center.get("name", "")
                    if "Buffoon" in name and item.cost <= 4:
                        return action

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Booster":
                name = center.get("name", "")
                if "Celestial" in name and item.cost <= 4:
                    return action

        if cons_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Booster":
                    name = center.get("name", "")
                    if "Arcana" in name and item.cost <= 4:
                        return action

        if cons_slots_left > 0:
            for i, item in enumerate(all_items):
                action = ActionRange.SHOP_BUY_START + i
                if not mask[action]:
                    continue
                center = state.data.centers.get(item.center_key, {})
                if center.get("set") == "Planet":
                    return action

        for i, item in enumerate(all_items):
            action = ActionRange.SHOP_BUY_START + i
            if not mask[action]:
                continue
            center = state.data.centers.get(item.center_key, {})
            if center.get("set") == "Booster":
                return action

        if best_joker_action >= 0 and joker_slots_left > 0 and best_joker_score > -5:
            item_idx = best_joker_action - ActionRange.SHOP_BUY_START
            if item_idx < len(all_items):
                item_cost = all_items[item_idx].cost
                if dollars - item_cost >= 3 or best_joker_score > 10:
                    return best_joker_action

        max_rerolls = 2 if ante <= 3 else 1
        should_reroll = (
            mask[ActionRange.SHOP_REROLL]
            and joker_slots_left > 0
            and best_joker_score < 10
            and dollars >= reroll_cost + 5
            and state.current_round.reroll_cost_increase < max_rerolls
        )
        if should_reroll:
            return ActionRange.SHOP_REROLL

        if (mask[ActionRange.SHOP_REROLL]
            and joker_slots_left > 0
            and best_joker_score < 5
            and dollars >= 20
            and state.current_round.reroll_cost_increase < 3):
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
            planet_type = center.get("config", {}).get("hand_type", "")
            main_type = self._get_main_hand_type(state)
            if planet_type == main_type:
                return 15.0
            if self._hand_type_synergy(state, planet_type) > 0:
                return 10.0
            if state.hands.get(planet_type, {}).get("played", 0) > 0:
                return 8.0
            return 5.0
        elif cset == "Tarot":
            cons_slots = consumable_limit(state) - len(state.consumables)
            if cons_slots <= 0:
                return -100.0
            return 4.0
        elif cset == "Spectral":
            return 3.0
        return 1.0

    def _atomic_consumable_action(
        self, state: RunState, slot: int, mask: np.ndarray, *, preferred_indices: tuple[int, ...] | None = None
    ) -> int | None:
        """Return the flat action id for using the consumable in `slot`, or None.

        Picks the targeting variant that matches the consumable's config
        and targets the first k hand cards (k = max_highlighted clamped
        to hand size) for hand-targeted consumables — replacing the old
        greedy multi-step sequence that walked CONSUMABLE_TARGET.
        """
        if slot >= len(state.consumables):
            return None
        cons = state.consumables[slot]
        center = state.data.centers[cons.center_key]
        config = center.get("config") or {}
        max_highlighted = config.get("max_highlighted")
        name = center.get("name", "")
        fallback_hand_limits = HAND_TARGET_CONSUMABLE_LIMITS.get(name)

        if name in JOKER_TARGET_CONSUMABLE_NAMES:
            for joker_idx in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
                action = encode_action(ActionType.USE_CONSUMABLE_JOKER, slot, joker_idx)
                if mask[action]:
                    return action
            return None

        if max_highlighted is not None or fallback_hand_limits is not None:
            if fallback_hand_limits is not None:
                min_size, max_size = fallback_hand_limits
            else:
                min_size = int(config.get("min_highlighted", 1) or 1)
                max_size = int(max_highlighted)
            max_size = min(max_size, MAX_CONSUMABLE_HAND_TARGETS)
            hand_size = len(state.hand_cards)
            max_target_size = min(max_size, hand_size)

            if preferred_indices is not None:
                pref_set = set(preferred_indices)
                pref_and_valid = [i for i in preferred_indices if i < hand_size]
                pref_max = min(len(pref_and_valid), max_target_size)
                for target_size in range(pref_max, min_size - 1, -1):
                    for subset in combinations(pref_and_valid, target_size):
                        if not can_use_consumable(state, cons, hand_targets=subset, joker_targets=()):
                            continue
                        action = encode_action(
                            ActionType.USE_CONSUMABLE_HAND_SUBSET,
                            slot,
                            consumable_subset_index(subset),
                        )
                        if mask[action]:
                            return action

            for target_size in range(max_target_size, min_size - 1, -1):
                for subset in combinations(range(hand_size), target_size):
                    if preferred_indices is not None and set(subset).issubset(set(preferred_indices)):
                        continue
                    if not can_use_consumable(state, cons, hand_targets=subset, joker_targets=()):
                        continue
                    action = encode_action(
                        ActionType.USE_CONSUMABLE_HAND_SUBSET,
                        slot,
                        consumable_subset_index(subset),
                    )
                    if mask[action]:
                        return action
            return None

        if not can_use_consumable(state, cons, hand_targets=(), joker_targets=()):
            return None
        action = encode_action(ActionType.USE_CONSUMABLE_NO_TARGET, slot)
        return action if mask[action] else None

    def _random_valid(self, mask: np.ndarray) -> int:
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            return 0
        return int(np.random.choice(valid))
