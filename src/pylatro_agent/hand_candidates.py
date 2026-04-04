"""One-shot play/discard candidate generation for hand-play decisions."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from pylatro import get_poker_hand_info
from pylatro.models import PlayingCard, RunState
from pylatro.scoring import RANK_TO_NOMINAL

from .constants import MAX_DISCARD_CANDIDATES, MAX_PLAY_CANDIDATES, POKER_HAND_NAMES

HAND_NAME_TO_ID = {name: idx + 1 for idx, name in enumerate(POKER_HAND_NAMES)}


@dataclass(frozen=True, slots=True)
class HandCandidate:
    kind: str
    indices: tuple[int, ...]
    hand_name: str = ""
    estimated_score: float = 0.0
    blind_ratio: float = 0.0


def candidate_signature(state: RunState) -> tuple:
    """Stable-enough cache key for the current hand-play choice set."""
    hand_sig = tuple(
        (
            id(card),
            card.rank,
            card.suit,
            card.center_key,
            card.edition_key or "",
            card.seal or "",
            int(card.perma_bonus),
            int(card.debuff),
            int(card.face_down),
            int(card.forced_selection),
        )
        for card in state.hand_cards
    )
    joker_sig = tuple(
        (
            joker.center_key,
            int(joker.mult),
            int(joker.t_mult),
            int(joker.t_chips),
            int(joker.x_mult * 10),
            int(joker.debuff),
        )
        for joker in state.jokers
    )
    hand_level_sig = tuple(
        (
            name,
            int(state.hands[name]["level"]),
            int(state.hands[name]["chips"]),
            int(state.hands[name]["mult"]),
            int(state.hands[name].get("played", 0)),
        )
        for name in POKER_HAND_NAMES
    )
    blind = state.round_resets.blind or {}
    return (
        hand_sig,
        joker_sig,
        hand_level_sig,
        int(state.current_round.hands_left),
        int(state.current_round.discards_left),
        int(state.current_round.hand_size),
        state.blind_on_deck or "",
        int(blind.get("mult", 1)),
    )


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

    ranked: list[HandCandidate] = []
    seen: set[tuple[int, ...]] = set()

    for size in range(min_size, max_size + 1):
        for combo in combinations(range(len(hand)), size):
            combo_set = set(combo)
            if not forced.issubset(combo_set):
                continue
            indices = tuple(sorted(combo))
            if indices in seen:
                continue
            seen.add(indices)
            cards = [hand[i] for i in indices]
            hand_name, _display, _poker_hands, scoring_hand = get_poker_hand_info(state, cards)
            estimated_score = _estimate_play_value(state, cards, hand_name, scoring_hand)
            blind_ratio = estimated_score / max(_blind_target(state), 1)
            ranked.append(
                HandCandidate(
                    kind="play",
                    indices=indices,
                    hand_name=hand_name,
                    estimated_score=estimated_score,
                    blind_ratio=blind_ratio,
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
    return int(math.floor(base * float(blind.get("mult", 1) or 1)))


def _card_nominal(card: PlayingCard) -> float:
    return float(RANK_TO_NOMINAL.get(card.rank, 0))
