"""Shared, observable strategy context for policy features and rewards."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SUIT_TARGET_ORDER = ("Spades", "Hearts", "Clubs", "Diamonds")

_SUIT_JOKER_AFFINITY: dict[str, dict[str, float]] = {
    "j_arrowhead": {"Spades": 1.0},
    "j_bloodstone": {"Hearts": 1.0},
    "j_onyx_agate": {"Clubs": 1.0},
    "j_rough_gem": {"Diamonds": 1.0},
    "j_wrathful_joker": {"Spades": 0.65},
    "j_lusty_joker": {"Hearts": 0.65},
    "j_gluttenous_joker": {"Clubs": 0.65},
    "j_greedy_joker": {"Diamonds": 0.65},
    # These care about a suit family but less exclusively than the four
    # dedicated suit Jokers above.
    "j_seeing_double": {"Clubs": 0.55},
    "j_blackboard": {"Spades": 0.35, "Clubs": 0.35},
}


def _value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _clip01(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def compute_suit_target_utilities(
    cards: Sequence[Any],
    jokers: Sequence[Any] = (),
    *,
    hand_play_counts: Mapping[str, Any] | None = None,
    boss_debuff_suit: str = "",
    respect_card_debuff: bool = True,
) -> tuple[float, float, float, float]:
    """Return absolute consolidation confidence in canonical suit order.

    Every output is independently normalized to ``[0, 1]``. We deliberately
    do not softmax or emit only an argmax: an ordinary deck should expose a
    low-confidence four-way tie, while a fixed deck can expose one or more
    genuinely strong options.
    """

    live_cards = [
        card
        for card in cards
        if not respect_card_debuff
        or not bool(_value(card, "debuffed", _value(card, "debuff", False)))
    ]
    deck_size = max(len(live_cards), 1)
    suit_counts = {
        suit: sum(str(_value(card, "suit", "") or "") == suit for card in live_cards)
        for suit in SUIT_TARGET_ORDER
    }
    played_counts = {
        suit: sum(
            max(int(_value(card, "times_played", 0) or 0), 0)
            for card in live_cards
            if str(_value(card, "suit", "") or "") == suit
        )
        for suit in SUIT_TARGET_ORDER
    }
    total_played_cards = sum(played_counts.values())
    flush_plays = sum(
        max(int((hand_play_counts or {}).get(hand_type, 0) or 0), 0)
        for hand_type in ("Flush", "Straight Flush", "Flush House", "Flush Five")
    )
    flush_evidence = _clip01(flush_plays / 4.0)

    affinities = dict.fromkeys(SUIT_TARGET_ORDER, 0.0)
    for joker in jokers:
        if bool(_value(joker, "debuffed", _value(joker, "debuff", False))):
            continue
        if (
            bool(_value(joker, "perishable", False))
            and _value(joker, "perish_tally") is not None
            and int(_value(joker, "perish_tally", 0) or 0) <= 0
        ):
            continue
        key = str(_value(joker, "key", _value(joker, "center_key", "")) or "")
        for suit, value in _SUIT_JOKER_AFFINITY.get(key, {}).items():
            affinities[suit] += value

    utilities: list[float] = []
    for suit in SUIT_TARGET_ORDER:
        if suit == boss_debuff_suit:
            utilities.append(0.0)
            continue
        share = suit_counts[suit] / deck_size
        # A stock 13/52 suit produces only 0.10 total deck confidence. The
        # signal becomes strong around 40-45%, where a Flush plan is reliable.
        deck_confidence = _clip01((share - 0.20) / 0.25)
        played_confidence = (
            played_counts[suit] / total_played_cards if total_played_cards > 0 else 0.0
        )
        joker_confidence = _clip01(affinities[suit])
        utility = (
            0.50 * deck_confidence
            + 0.25 * played_confidence
            + 0.15 * flush_evidence * deck_confidence
            + 0.35 * joker_confidence
        )
        utilities.append(_clip01(utility))
    return utilities[0], utilities[1], utilities[2], utilities[3]


def suit_target_utilities(info: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Adapter for serialized pre/post state snapshots."""

    deck = info.get("deck_stats")
    deck = deck if isinstance(deck, Mapping) else {}
    cards = tuple(deck.get("cards") or deck.get("card_descriptors") or ())
    jokers = tuple(info.get("joker_details") or ())
    play_counts = info.get("hand_play_counts")
    return compute_suit_target_utilities(
        cards,
        jokers,
        hand_play_counts=play_counts if isinstance(play_counts, Mapping) else {},
        boss_debuff_suit=str(info.get("boss_debuff_suit", "") or ""),
        respect_card_debuff=bool(info.get("round_active", False))
        and not bool(info.get("blind_disabled", False)),
    )
