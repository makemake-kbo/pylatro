"""One-shot play/discard candidate generation for hand-play decisions."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

from pylatro import get_poker_hand_info
from pylatro.scoring import RANK_TO_NOMINAL

from .constants import MAX_DISCARD_CANDIDATES, MAX_PLAY_CANDIDATES, POKER_HAND_NAMES

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pylatro.models import PlayingCard, RunState

HAND_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(POKER_HAND_NAMES)}


@dataclass(frozen=True, slots=True)
class HandCandidate:
    kind: str
    indices: tuple[int, ...]
    hand_name: str = ""
    estimated_score: float = 0.0
    blind_ratio: float = 0.0
    # A deliberately Joker-free estimate of the chips this hand can bank.
    # Keep this separate from estimated_score: the latter ranks structural
    # build potential and is part of the existing candidate/token semantics.
    raw_score: float = 0.0


def generate_hand_candidates(
    state: RunState,
) -> tuple[tuple[HandCandidate, ...], tuple[HandCandidate, ...]]:
    play_candidates = _generate_play_candidates(state)
    discard_candidates = _generate_discard_candidates(state, play_candidates)
    return tuple(play_candidates[:MAX_PLAY_CANDIDATES]), tuple(discard_candidates[:MAX_DISCARD_CANDIDATES])


def _generate_play_candidates(state: RunState) -> list[HandCandidate]:
    hand = state.hand_cards
    if not hand:
        return []

    forced = {idx for idx, card in enumerate(hand) if card.forced_selection}
    min_size = max(1, len(forced))
    max_size = min(5, len(hand))
    if min_size > max_size:
        return []

    # Generate a small set of promising combos structurally instead of
    # brute-forcing all C(n,1..5) combinations.
    candidate_indices = _structural_candidates(state, hand, forced, min_size, max_size)

    ranked: list[HandCandidate] = []
    seen: set[tuple[int, ...]] = set()

    for indices in candidate_indices:
        if indices in seen:
            continue
        seen.add(indices)
        cards = [hand[i] for i in indices]
        hand_name, _display, _poker_hands, scoring_hand = get_poker_hand_info(state, cards)
        estimated_score = _estimate_play_value(state, cards, hand_name, scoring_hand)
        raw_score = _estimate_raw_score(state, cards, hand_name, scoring_hand)
        blind_ratio = estimated_score / max(_blind_target(state), 1)
        ranked.append(
            HandCandidate(
                kind="play",
                indices=indices,
                hand_name=hand_name,
                estimated_score=estimated_score,
                blind_ratio=blind_ratio,
                raw_score=raw_score,
            )
        )

    ranked.sort(
        key=lambda c: (
            c.estimated_score,
            len(c.indices),
            tuple(_card_nominal(hand[i]) for i in c.indices),
        ),
        reverse=True,
    )
    return ranked


def _structural_candidates(
    state: RunState,
    hand: list[PlayingCard],
    forced: set[int],
    min_size: int,
    max_size: int,
) -> list[tuple[int, ...]]:
    """Build promising hand combos using rank/suit grouping instead of brute force."""
    n = len(hand)

    # Index cards by rank and suit
    by_rank: dict[int, list[int]] = {}
    by_suit: dict[str, list[int]] = {}
    nominals: list[float] = []
    for i, card in enumerate(hand):
        rid = RANK_TO_NOMINAL.get(card.rank, 0)
        nominals.append(rid)
        center = state.data.centers.get(card.center_key, {})
        effect = center.get("effect", "") if center else ""
        if effect != "Stone Card":
            cid = _rank_to_id(card.rank)
            by_rank.setdefault(cid, []).append(i)
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
        else:
            by_rank.setdefault(-id(card), []).append(i)

    four_fingers = state.has_joker("Four Fingers")
    flush_req = 4 if four_fingers else 5
    straight_req = 4 if four_fingers else 5

    results: list[tuple[int, ...]] = []

    def _add(indices) -> None:
        indices = set(indices)
        if not forced.issubset(indices):
            return
        if not (min_size <= len(indices) <= max_size):
            return
        results.append(tuple(sorted(indices)))

    # --- N-of-a-kind combos (pairs, trips, quads, fives) ---
    groups_sorted = sorted(by_rank.items(), key=lambda x: (len(x[1]), x[0]), reverse=True)
    for cid, idxs in groups_sorted:
        if cid < 0:
            continue
        for sz in range(min(len(idxs), 5), 1, -1):
            _add(set(idxs[:sz]))

    # --- Full houses (3 + 2) ---
    trips = [(cid, idxs) for cid, idxs in groups_sorted if len(idxs) >= 3 and cid > 0]
    pairs = [(cid, idxs) for cid, idxs in groups_sorted if len(idxs) >= 2 and cid > 0]
    for t_cid, t_idxs in trips[:3]:
        for p_cid, p_idxs in pairs[:4]:
            if p_cid != t_cid:
                _add(set(t_idxs[:3]) | set(p_idxs[:2]))

    # --- Flushes ---
    for _suit, idxs in by_suit.items():
        unique = list(dict.fromkeys(idxs))
        if len(unique) >= flush_req:
            unique.sort(key=lambda i: nominals[i], reverse=True)
            _add(set(unique[:5]))
            # Also try the worst flush (for diversity)
            if len(unique) > 5:
                _add(set(unique[:4] + unique[5:6]))

    # --- Straights ---
    rank_set = set(by_rank.keys())
    positive_ranks = {r for r in rank_set if r > 0}
    if 14 in positive_ranks:
        positive_ranks.add(1)
    for high in range(14, 0, -1):
        run_ranks: list[int] = []
        for r in range(high, high - straight_req - 1, -1):
            if r < 1:
                break
            actual = 14 if r == 1 else r
            if actual in by_rank and actual > 0:
                run_ranks.append(actual)
            elif state.has_joker("Shortcut") and len(run_ranks) > 0:
                continue
            else:
                break
        if len(run_ranks) >= straight_req:
            straight_indices = set()
            for r in run_ranks[:5]:
                straight_indices.add(by_rank[r][0])
            _add(straight_indices)

    # --- Straight flushes ---
    for _suit, idxs in by_suit.items():
        unique = list(dict.fromkeys(idxs))
        if len(unique) < straight_req:
            continue
        suit_ranks: dict[int, int] = {}
        for i in unique:
            cid = _rank_to_id(hand[i].rank)
            if cid > 0 and cid not in suit_ranks:
                suit_ranks[cid] = i
        if 14 in suit_ranks:
            suit_ranks.setdefault(1, suit_ranks[14])
        for high in range(14, 0, -1):
            run: list[int] = []
            for r in range(high, high - straight_req - 1, -1):
                if r < 1:
                    break
                actual = 14 if r == 1 else r
                if actual in suit_ranks:
                    run.append(suit_ranks[actual])
                else:
                    break
            if len(run) >= straight_req:
                _add(set(run[:5]))
                break

    # --- Two pairs ---
    pair_groups = [(cid, idxs) for cid, idxs in groups_sorted if len(idxs) >= 2 and cid > 0]
    for i in range(min(len(pair_groups), 3)):
        for j in range(i + 1, min(len(pair_groups), 4)):
            _add(set(pair_groups[i][1][:2]) | set(pair_groups[j][1][:2]))

    # --- Single high cards ---
    ranked_indices = sorted(range(n), key=lambda i: nominals[i], reverse=True)
    for i in ranked_indices[:3]:
        _add({i})

    # --- Best 5-card hand (all highest nominals) ---
    _add(set(ranked_indices[:min(5, n)]))

    # --- Forced-card combos if we have forced cards ---
    if forced:
        remaining = [i for i in ranked_indices if i not in forced]
        for fill_size in range(min(5 - len(forced), len(remaining)) + 1):
            if fill_size == 0:
                _add(set(forced))
            else:
                for fill_indices in combinations(remaining[:6], fill_size):
                    _add(set(forced) | set(fill_indices))

    return results


# PlayingCard.rank uses single-character codes ("T", "J", "Q", "K", "A"),
# matching pylatro.scoring.RANK_TO_ID.
_RANK_IDS = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}


def _rank_to_id(rank: str) -> int:
    return _RANK_IDS.get(rank, 0)


def _generate_discard_candidates(state: RunState, play_candidates: Iterable[HandCandidate]) -> list[HandCandidate]:
    hand = state.hand_cards
    if not hand:
        return []

    forced = {idx for idx, card in enumerate(hand) if card.forced_selection}
    keep_scores = _card_keep_scores(state, hand)
    worst_first = sorted(range(len(hand)), key=lambda idx: keep_scores[idx])
    candidates: dict[tuple[int, ...], HandCandidate] = {}

    def add(indices: Iterable[int]) -> None:
        chosen = set(indices)
        if not chosen:
            return
        chosen |= forced
        if len(chosen) > 5:
            return
        key = tuple(sorted(chosen))
        if key in candidates:
            return
        value = _estimate_discard_value(state, keep_scores, key)
        candidates[key] = HandCandidate(kind="discard", indices=key, estimated_score=value)

    for size in range(1, min(5, len(hand)) + 1):
        add(worst_first[:size])

    for idx in worst_first[:5]:
        add((idx,))

    for combo in combinations(worst_first[:5], 2):
        add(combo)
    for combo in combinations(worst_first[:5], 3):
        add(combo)

    best_play = set(play_candidates[0].indices) if play_candidates else set()
    if best_play:
        add(set(range(len(hand))) - best_play)

    for cand in list(play_candidates)[:4]:
        add(set(range(len(hand))) - set(cand.indices))

    low_singletons = [
        idx for idx in worst_first
        if Counter(card.rank for card in hand)[hand[idx].rank] == 1
    ]
    add(low_singletons[:5])

    ranked = sorted(
        candidates.values(),
        key=lambda c: (c.estimated_score, len(c.indices)),
        reverse=True,
    )
    return ranked


def _estimate_play_value(
    state: RunState,
    cards: list[PlayingCard],
    hand_name: str,
    scoring_hand: list[PlayingCard],
) -> float:
    rank_bonus = float(len(POKER_HAND_NAMES) - POKER_HAND_NAMES.index(hand_name))
    hand_meta = state.hands.get(hand_name, {})
    base_chips = float(hand_meta.get("chips", 0) or 0)
    base_mult = float(hand_meta.get("mult", 1) or 1)
    level = float(hand_meta.get("level", 1) or 1)

    score = rank_bonus * 100_000.0
    score += level * 10_000.0
    score += base_chips * max(base_mult, 1.0) * 25.0
    score += sum(_card_play_value(state, card) for card in scoring_hand) * 200.0
    score += sum(_card_hold_value(card) for card in cards) * 20.0
    score += _joker_synergy_value(state, hand_name) * 50.0
    score += min(
        (base_chips * max(base_mult, 1.0)) / max(_blind_target(state), 1),
        2.0,
    ) * 20_000.0
    return score


def _estimate_raw_score(
    state: RunState,
    cards: list[PlayingCard],
    hand_name: str,
    scoring_hand: list[PlayingCard],
) -> float:
    """Conservative score proxy for banking chips without Joker scaling.

    This covers the deterministic hand/card contributions that matter most in
    ante 1. It intentionally excludes Jokers and held-card effects so the
    tempo signal does not teach the policy to play a low-chip hand merely
    because a later-game build could carry it.
    """
    hand_meta = state.hands.get(hand_name, {})
    chips = float(hand_meta.get("chips", 0) or 0)
    mult = float(hand_meta.get("mult", 1) or 1)

    scoring_cards = list(scoring_hand)
    scoring_ids = {id(card) for card in scoring_cards}
    for card in cards:
        center = state.data.centers.get(card.center_key, {})
        if center.get("effect", "") == "Stone Card" and id(card) not in scoring_ids:
            scoring_cards.append(card)

    for card in scoring_cards:
        if card.debuff:
            continue
        center = state.data.centers.get(card.center_key, {})
        config = center.get("config") or {}
        is_stone = center.get("effect", "") == "Stone Card"
        repetitions = 2 if card.seal == "Red" else 1
        for _ in range(repetitions):
            chips += (
                (0.0 if is_stone else _card_nominal(card))
                + float(card.perma_bonus)
                + float(config.get("bonus", 0) or 0)
            )
            # Lucky Card's multiplier is stochastic, so stay conservative.
            if center.get("effect", "") != "Lucky Card":
                mult += float(config.get("mult", 0) or 0)
            x_mult = float(config.get("Xmult", 1) or 1)
            if x_mult > 1.0:
                mult *= x_mult
            if card.edition_key == "foil":
                chips += 50.0
            elif card.edition_key == "holo":
                mult += 10.0
            elif card.edition_key == "polychrome":
                mult *= 1.5

    return max(chips, 0.0) * max(mult, 0.0)


def _estimate_discard_value(
    state: RunState,
    keep_scores: list[float],
    indices: tuple[int, ...],
) -> float:
    if not indices:
        return -math.inf
    removed = sum(10.0 - keep_scores[idx] for idx in indices)
    tempo = len(indices) * 5.0
    discards_left = float(state.current_round.discards_left)
    hands_left = float(state.current_round.hands_left)
    return removed * 100.0 + tempo * 10.0 + discards_left * 3.0 - hands_left * 2.0


def _card_play_value(state: RunState, card: PlayingCard) -> float:
    center = state.data.centers.get(card.center_key, {})
    config = center.get("config") or {}
    bonus = float(config.get("bonus", 0) or 0)
    mult = float(config.get("mult", 0) or 0)
    h_mult = float(config.get("h_mult", 0) or 0)
    x_mult = float(config.get("Xmult", 1) or 1)
    edition_bonus = 10.0 if (card.edition_key or "") == "holo" else 50.0 if (card.edition_key or "") == "foil" else 0.0
    return (
        _card_nominal(card)
        + float(card.perma_bonus)
        + bonus
        + mult * 1.5
        + h_mult * 1.5
        + max(x_mult - 1.0, 0.0) * 10.0
        + edition_bonus
    )


def _card_hold_value(card: PlayingCard) -> float:
    return _card_nominal(card) + float(card.perma_bonus) * 0.2


def _card_keep_scores(state: RunState, hand: list[PlayingCard]) -> list[float]:
    suits = [card.suit for card in hand]
    ranks = [card.rank for card in hand]
    suit_counts = Counter(suits)
    rank_counts = Counter(ranks)
    scores: list[float] = []

    for card in hand:
        score = _card_nominal(card) * 0.1
        if suit_counts[card.suit] >= 4:
            score += 20.0
        elif suit_counts[card.suit] >= 3:
            score += 8.0
        if rank_counts[card.rank] >= 3:
            score += 25.0
        elif rank_counts[card.rank] >= 2:
            score += 15.0
        score += float(card.perma_bonus) * 0.2
        if card.forced_selection:
            score += 100.0
        scores.append(score)
    return scores


def _joker_synergy_value(state: RunState, hand_name: str) -> float:
    total = 0.0
    for joker in state.jokers:
        if joker.debuff:
            continue
        center = state.data.centers.get(joker.center_key, {})
        config = center.get("config")
        if not isinstance(config, dict):
            config = {}
        total += float(config.get("mult", 0) or 0)
        total += float(config.get("t_mult", 0) or 0)
        total += float(config.get("t_chips", 0) or 0) * 0.2
        total += max(float(config.get("Xmult", 1) or 1) - 1.0, 0.0) * 10.0
        if config.get("type", "") == hand_name:
            total += 20.0
    return total


def _blind_target(state: RunState) -> int:
    blind = state.round_resets.blind
    if blind is None:
        return 0
    from pylatro import get_blind_amount

    base = get_blind_amount(state.round_resets.ante, min(state.stake, 3))
    return math.floor(base * float(blind.get("mult", 1) or 1))


def _card_nominal(card: PlayingCard) -> float:
    return float(RANK_TO_NOMINAL.get(card.rank, 0))
