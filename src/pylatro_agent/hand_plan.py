"""Contextual poker-hand plans for reward and shop evaluation.

The planner deliberately excludes Full House and straights. Two Pair is only a
low-priority conditional plan when a dedicated synergy Joker is active. Those
hands may still be played when they are best in the current draw, but they are
not default deck-building targets. Difficult multiplicity hands only become
plans after the serialized deck shows real rank/card concentration.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

from .build_value import estimate_hand_score

PLAN_HAND_TYPES = (
    "Pair",
    "High Card",
    "Flush",
    "Three of a Kind",
    "Four of a Kind",
    "Five of a Kind",
    "Flush Five",
)

_HAND_PRIORITY = {
    "High Card": 0.72,
    "Pair": 1.00,
    "Two Pair": 0.58,
    "Flush": 1.05,
    "Three of a Kind": 1.08,
    "Four of a Kind": 1.16,
    "Five of a Kind": 1.24,
    "Flush Five": 1.32,
}


@dataclass(frozen=True)
class HandPlan:
    hand_type: str
    draw_reliability: float
    deck_fix_gate: float
    score_per_hand: float
    required_score_per_hand: float
    readiness_ratio: float
    utility: float


@dataclass(frozen=True)
class HandPlanEstimate:
    plans: tuple[HandPlan, ...]
    best: HandPlan

    def for_hand(self, hand_type: str) -> HandPlan | None:
        return next((plan for plan in self.plans if plan.hand_type == hand_type), None)


def _clip01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _counts(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        return ()
    return tuple(max(int(count or 0), 0) for count in value.values() if int(count or 0) > 0)


def _draw_budget(info: Mapping[str, Any], deck_size: int) -> int:
    """Cards observable before the final scoring attempt.

    Discards and spare hands may each cycle at most five cards.  This is an
    optimistic opportunity estimate, then the deck-fix gates below prevent a
    normal deck from masquerading as a reliable four/five-kind build.
    """
    hand_size = max(int(info.get("hand_size", 8) or 8), 1)
    discards = max(int(info.get("discards_available") or info.get("discards_left") or 0), 0)
    hands = max(int(info.get("hands_available") or info.get("hands_left") or 1), 1)
    cycle = min(hand_size, 5)
    return min(deck_size, hand_size + cycle * discards + cycle * max(hands - 1, 0))


def _hypergeom_at_least(population: int, successes: int, draws: int, needed: int) -> float:
    if needed <= 0:
        return 1.0
    population = max(int(population), 0)
    successes = max(0, min(int(successes), population))
    draws = max(0, min(int(draws), population))
    if successes < needed or draws < needed or population <= 0:
        return 0.0
    denominator = comb(population, draws)
    if denominator <= 0:
        return 0.0
    low = max(needed, draws - (population - successes))
    high = min(successes, draws)
    numerator = sum(comb(successes, hit) * comb(population - successes, draws - hit) for hit in range(low, high + 1))
    return _clip01(numerator / denominator)


def _any_pair_probability(rank_counts: Sequence[int], deck_size: int, draws: int) -> float:
    """Exact chance that the sampled cards contain at least one rank pair."""
    draws = min(max(int(draws), 0), deck_size)
    if draws < 2:
        return 0.0
    if draws > len(rank_counts):
        return 1.0
    # Coefficient of x**draws in product(1 + count[rank] * x): ways to
    # choose cards while taking at most one from every rank.
    distinct = [0] * (draws + 1)
    distinct[0] = 1
    for count in rank_counts:
        for size in range(draws, 0, -1):
            distinct[size] += distinct[size - 1] * count
    denominator = comb(deck_size, draws)
    return _clip01(1.0 - distinct[draws] / denominator) if denominator else 0.0


def _two_pair_probability(rank_counts: Sequence[int], deck_size: int, draws: int) -> float:
    """Exact chance that a sample contains pairs from two distinct ranks."""
    draws = min(max(int(draws), 0), deck_size)
    if draws < 4:
        return 0.0
    # dp[cards_used][paired_rank_count capped at 2] = combination count.
    dp = [[0, 0, 0] for _ in range(draws + 1)]
    dp[0][0] = 1
    for count in rank_counts:
        updated = [[0, 0, 0] for _ in range(draws + 1)]
        for used in range(draws + 1):
            for paired_ranks in range(3):
                ways = dp[used][paired_ranks]
                if not ways:
                    continue
                for taken in range(min(count, draws - used) + 1):
                    next_pairs = min(2, paired_ranks + int(taken >= 2))
                    updated[used + taken][next_pairs] += ways * comb(count, taken)
        dp = updated
    denominator = comb(deck_size, draws)
    return _clip01(dp[draws][2] / denominator) if denominator else 0.0


def _active_two_pair_synergy(jokers: Sequence[Mapping[str, Any]]) -> float:
    """Return a bounded gate for dedicated, active Two Pair Jokers."""
    units = 0.0
    for joker in jokers:
        if joker.get("debuffed"):
            continue
        if (
            joker.get("perishable")
            and joker.get("perish_tally") is not None
            and int(joker.get("perish_tally") or 0) <= 0
        ):
            continue
        key = str(joker.get("key") or "")
        restriction = str(joker.get("type") or "")
        if key == "j_trousers":
            units += 1.0
        elif restriction == "Two Pair":
            units += 0.55
    if units <= 0.0:
        return 0.0
    # Even multiple synergies keep Two Pair below the unconditional fallback
    # plans; mature scoring output can still make it the best actual plan.
    return min(0.35 + 0.25 * units, 0.70)


def _fixed_gate(hand_type: str, *, deck_size: int, max_rank: int, max_suit: int, max_exact: int) -> float:
    if hand_type in {"Pair", "High Card"}:
        return 1.0
    if hand_type == "Flush":
        # A stock deck is exactly 25% one suit and receives no Flush-plan
        # signal.  Full strength starts around 40% of the deck.
        share = max_suit / max(deck_size, 1)
        return _clip01((share - 0.25) / 0.15)
    if hand_type == "Three of a Kind":
        return _clip01((max_rank - 4) / 3.0)
    if hand_type == "Four of a Kind":
        return _clip01((max_rank - 4) / 4.0)
    if hand_type == "Five of a Kind":
        return _clip01((max_rank - 4) / 4.0)
    if hand_type == "Flush Five":
        return _clip01((max_exact - 1) / 5.0)
    return 0.0


def estimate_hand_plans(
    info: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]] | None = None,
) -> HandPlanEstimate | None:
    deck = _mapping(info.get("deck_stats"))
    deck_size = max(int(deck.get("size", 0) or 0), 0)
    rank_counts = _counts(deck.get("rank_counts"))
    suit_counts = _counts(deck.get("suit_counts"))
    exact_counts = _counts(deck.get("rank_suit_counts"))
    if not rank_counts or not suit_counts:
        cards = tuple(
            card
            for card in (deck.get("cards") or deck.get("card_descriptors") or ())
            if isinstance(card, Mapping)
        )
        if cards:
            rank_map = Counter(str(card.get("rank") or "") for card in cards if card.get("rank"))
            suit_map = Counter(str(card.get("suit") or "") for card in cards if card.get("suit"))
            exact_map = Counter(
                (str(card.get("rank") or ""), str(card.get("suit") or ""))
                for card in cards
                if card.get("rank") and card.get("suit")
            )
            rank_counts = rank_counts or tuple(rank_map.values())
            suit_counts = suit_counts or tuple(suit_map.values())
            exact_counts = exact_counts or tuple(exact_map.values())
            deck_size = deck_size or len(cards)
    blind_target = max(float(info.get("blind_target", 0) or 0), 0.0)
    hands = max(int(info.get("hands_available") or info.get("hands_left") or 1), 1)
    if deck_size <= 0 or not rank_counts or blind_target <= 0.0:
        return None

    draws = _draw_budget(info, deck_size)
    max_rank = max(rank_counts, default=0)
    max_suit = max(suit_counts, default=0)
    max_exact = max(exact_counts, default=0)
    required = blind_target / hands
    owned = tuple(jokers if jokers is not None else (info.get("joker_details") or ()))
    two_pair_gate = _active_two_pair_synergy(owned)
    hand_types = (*PLAN_HAND_TYPES, "Two Pair") if two_pair_gate > 0.0 else PLAN_HAND_TYPES
    plans: list[HandPlan] = []
    for hand_type in hand_types:
        gate = (
            two_pair_gate
            if hand_type == "Two Pair"
            else _fixed_gate(
                hand_type,
                deck_size=deck_size,
                max_rank=max_rank,
                max_suit=max_suit,
                max_exact=max_exact,
            )
        )
        if hand_type == "High Card":
            probability = 1.0
        elif hand_type == "Pair":
            probability = _any_pair_probability(rank_counts, deck_size, draws)
        elif hand_type == "Two Pair":
            probability = _two_pair_probability(rank_counts, deck_size, draws)
        elif hand_type == "Flush":
            probability = _hypergeom_at_least(deck_size, max_suit, draws, 5)
        elif hand_type == "Three of a Kind":
            probability = _hypergeom_at_least(deck_size, max_rank, draws, 3)
        elif hand_type == "Four of a Kind":
            probability = _hypergeom_at_least(deck_size, max_rank, draws, 4)
        elif hand_type == "Five of a Kind":
            probability = _hypergeom_at_least(deck_size, max_rank, draws, 5)
        else:
            probability = _hypergeom_at_least(deck_size, max_exact, draws, 5)

        reliability = gate * probability
        score = float(estimate_hand_score(info, hand_type, jokers))
        readiness = score / required if required > 0.0 else 0.0
        # Readiness above one still matters a little as future-blind margin,
        # but is bounded. Reliability is the dominant term.
        readiness_value = min(max(readiness, 0.0), 1.5) / 1.5
        utility = _HAND_PRIORITY[hand_type] * reliability * readiness_value
        plans.append(
            HandPlan(
                hand_type=hand_type,
                draw_reliability=reliability,
                deck_fix_gate=gate,
                score_per_hand=score,
                required_score_per_hand=required,
                readiness_ratio=readiness,
                utility=utility,
            )
        )

    best = max(plans, key=lambda plan: (plan.utility, plan.draw_reliability, plan.score_per_hand))
    return HandPlanEstimate(plans=tuple(plans), best=best)
