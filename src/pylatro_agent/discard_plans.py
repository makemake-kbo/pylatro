"""Discard intents: what a discard keeps, and the chance the redraw completes it.

A discard usually throws away cards outside a plan: a suit chasing a Flush, a
rank group chasing more of a kind, four ranks of a Straight window, a strong
play kept for later, or Purple-seal cards discarded for their Tarot. The
policy picks the plan and how many cards go; code picks which cards (the
least useful outside the plan), so model capacity is not spent on drawing
odds or card bookkeeping.

Each intent carries the exact chance that the redraw completes its plan, from
the draw pile's composition. The composition is public (the game's deck view
shows it); the order is not used.
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
    PURPLE = 5  # discard Purple-seal cards for their Tarot


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


def discard_intents(
    state: RunState, plays: Sequence[HandCandidate]
) -> list[tuple[HandCandidate, PlanKind, tuple[float, ...]]]:
    """Discard intents: a plan and the order its cards leave the hand (v18).

    The policy picks an intent and a count k; the discard is the first k cards
    of the intent's list. The list holds the forced cards first (every discard
    must include them), then, for the Purple-seal intent, the Purple-seal cards,
    then the cards outside the plan from least to most useful, five at most.
    Blue-seal cards stay. ``chances[k - 1]`` is the chance that a k-card discard
    completes the plan (0 for kept plays and Purple seals, which complete
    nothing). Intents that could complete something come first, most likely
    first, then Purple seals, then kept plays.
    """
    if state.current_round.discards_left <= 0:
        return []
    hand = state.hand_cards[:MAX_HAND_SIZE]
    forced = [i for i, card in enumerate(hand) if card.forced_selection]
    if len(forced) > MAX_DISCARD_SIZE:
        return []
    keep_scores = _card_keep_scores(state, state.hand_cards)
    pile = len(state.draw_pile)

    def order(kept: frozenset[int], lead: Sequence[int] = ()) -> tuple[int, ...]:
        skip = set(kept) | set(forced) | set(lead)
        rest = sorted(
            (i for i in range(len(hand)) if i not in skip and hand[i].seal != "Blue"), key=lambda i: keep_scores[i]
        )
        return tuple([*forced, *lead, *rest][:MAX_DISCARD_SIZE])

    def chances(cards: tuple[int, ...], outs: list[PlayingCard] | None, need: int) -> tuple[float, ...]:
        return tuple(
            hit_chance(len(outs), pile, _redraw_count(state, k), need)
            if outs is not None and len(forced) <= k <= len(cards)
            else 0.0
            for k in range(1, MAX_DISCARD_SIZE + 1)
        )

    best: dict[tuple[int, ...], tuple[int, float, int, HandCandidate, PlanKind, tuple[float, ...]]] = {}

    def add(cards: tuple[int, ...], kind: PlanKind, chased: str, odds: tuple[float, ...], order_: int) -> None:
        if len(cards) <= len(forced):
            return  # nothing to choose beyond the forced cards
        group = 2 if kind == PlanKind.KEEP_PLAY else 1 if kind == PlanKind.PURPLE else 0
        key = (group, -max(odds), order_)
        if cards not in best or key < best[cards][:3]:
            best[cards] = (*key, HandCandidate(kind="discard", indices=cards, hand_name=chased), kind, odds)

    for order_, (kept, kind, chased, outs, need) in enumerate(_plans(state, plays)):
        cards = order(kept)
        add(cards, kind, chased, chances(cards, outs, need), order_)
    purple = sorted(
        (i for i, card in enumerate(hand) if card.seal == "Purple" and i not in forced), key=lambda i: keep_scores[i]
    )
    if purple:
        add(order(frozenset(), purple), PlanKind.PURPLE, "", (0.0,) * MAX_DISCARD_SIZE, -1)
    return [(candidate, kind, odds) for *_, candidate, kind, odds in sorted(best.values(), key=lambda e: e[:3])]
