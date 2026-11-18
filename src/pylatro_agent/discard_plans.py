"""Discards that keep a draw plan, with the chance the redraw completes it.

A discard usually throws away everything outside a plan: a suit chasing a
Flush, a rank group chasing more of a kind, four ranks of a Straight window,
or a strong play kept for later. The worst-card discard candidates rarely
list such a complement, so the hand head had to build it card by card.

Each plan discard names the hand it chases and carries the exact chance that
the redraw completes it, from the draw pile's composition. The composition is
public (the game's deck view shows it); the order is not used.
"""

from __future__ import annotations

from collections import defaultdict
from enum import IntEnum
from math import comb
from typing import TYPE_CHECKING

from pylatro.scoring import _card_id, _is_suit

from .constants import MAX_HAND_SIZE
from .hand_candidates import HandCandidate, _card_keep_scores

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pylatro.models import PlayingCard, RunState

MAX_DISCARD_SIZE = 5
SUITS = ("Spades", "Hearts", "Clubs", "Diamonds")
# Kept plays: the strongest few play candidates, held while the rest redraws.
KEPT_PLAYS = 4
OF_A_KIND = {2: "Three of a Kind", 3: "Four of a Kind"}


class PlanKind(IntEnum):
    NONE = 0
    SUIT = 1
    RANK = 2
    STRAIGHT = 3
    KEEP_PLAY = 4


def hit_chance(outs: int, pool: int, draws: int, need: int) -> float:
    """P(at least ``need`` of ``outs`` marked cards among ``draws`` cards drawn from ``pool``)."""
    if need <= 0:
        return 1.0
    draws = min(draws, pool)
    if draws < need or outs < need:
        return 0.0
    total = comb(pool, draws)
    return sum(comb(outs, k) * comb(pool - outs, draws - k) for k in range(need, min(outs, draws) + 1)) / total


def _redraw_count(state: RunState, discarded: int) -> int:
    room = state.current_round.hand_size - (len(state.hand_cards) - discarded)
    return max(0, min(len(state.draw_pile), room))


def _plans(
    state: RunState, plays: Sequence[HandCandidate]
) -> list[tuple[frozenset[int], PlanKind, str, list[PlayingCard] | None, int]]:
    """(kept cards, kind, hand chased, draw cards that advance it or None, cards still needed)."""
    hand = state.hand_cards[:MAX_HAND_SIZE]
    pile = state.draw_pile
    required = 4 if state.has_joker("Four Fingers") else 5
    plans: list[tuple[frozenset[int], PlanKind, str, list[PlayingCard] | None, int]] = []

    for suit in SUITS:
        group = frozenset(i for i, card in enumerate(hand) if _is_suit(state, card, suit, flush_calc=True))
        if len(group) >= 3:
            outs = [card for card in pile if _is_suit(state, card, suit, flush_calc=True)]
            plans.append((group, PlanKind.SUIT, "Flush", outs, required - len(group)))

    by_rank: dict[int, set[int]] = defaultdict(set)
    for i, card in enumerate(hand):
        if (rank := _card_id(state, card)) > 0:
            by_rank[rank].add(i)
    for rank, group in by_rank.items():
        if len(group) >= 2:
            outs = [card for card in pile if _card_id(state, card) == rank]
            plans.append((frozenset(group), PlanKind.RANK, OF_A_KIND.get(len(group), "Five of a Kind"), outs, 1))

    for low in range(1, 11):
        held: dict[int, int] = {}
        for i, card in enumerate(hand):
            rank = _card_id(state, card)
            rank = 1 if rank == 14 and low == 1 else rank
            if low <= rank < low + 5 and not card.debuff:
                held.setdefault(rank, i)
        if len(held) >= 4:
            missing = {14 if rank == 1 else rank for rank in range(low, low + 5) if rank not in held}
            outs = [card for card in pile if _card_id(state, card) in missing]
            plans.append((frozenset(held.values()), PlanKind.STRAIGHT, "Straight", outs, required - len(held)))

    for play in plays[:KEPT_PLAYS]:
        plans.append((frozenset(play.indices), PlanKind.KEEP_PLAY, play.hand_name, None, 0))
    return plans


def plan_discards(
    state: RunState, plays: Sequence[HandCandidate]
) -> list[tuple[HandCandidate, PlanKind, float]]:
    """Discards of everything outside each plan, most likely to complete first.

    A complement larger than five cards discards its five least useful cards.
    Forced cards ride along as every discard must include them; Blue-seal
    cards are kept. Kept-play discards come last with a chance of 0, as they
    complete nothing.
    """
    if state.current_round.discards_left <= 0:
        return []
    hand = state.hand_cards[:MAX_HAND_SIZE]
    forced = {i for i, card in enumerate(hand) if card.forced_selection}
    if len(forced) > MAX_DISCARD_SIZE:
        return []
    keep_scores = _card_keep_scores(state, state.hand_cards)
    best: dict[tuple[int, ...], tuple[float, int, HandCandidate, PlanKind]] = {}
    for order, (kept, kind, chased, outs, need) in enumerate(_plans(state, plays)):
        rest = sorted(
            (i for i in range(len(hand)) if i not in kept and i not in forced and hand[i].seal != "Blue"),
            key=lambda i: keep_scores[i],
        )
        indices = tuple(sorted(forced | set(rest[: MAX_DISCARD_SIZE - len(forced)])))
        if not rest or not indices:
            continue
        chance = 0.0
        if outs is not None:
            chance = hit_chance(len(outs), len(state.draw_pile), _redraw_count(state, len(indices)), need)
        if indices not in best or (chance, -order) > (best[indices][0], -best[indices][1]):
            best[indices] = (chance, order, HandCandidate(kind="discard", indices=indices, hand_name=chased), kind)
    ranked = sorted(best.values(), key=lambda entry: (entry[3] == PlanKind.KEEP_PLAY, -entry[0], entry[1]))
    return [(candidate, kind, chance) for chance, _, candidate, kind in ranked]
