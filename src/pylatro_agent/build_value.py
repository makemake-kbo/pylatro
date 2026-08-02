"""Pure contextual score estimates for captured build snapshots.

The evaluator only consumes dictionaries produced by ``shop_eval``.  It does
not inspect or mutate a live ``RunState`` and never calls ``score_hand``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import floor, isclose
from typing import Any

_RANK_NOMINAL = {
    "2": 2.0,
    "3": 3.0,
    "4": 4.0,
    "5": 5.0,
    "6": 6.0,
    "7": 7.0,
    "8": 8.0,
    "9": 9.0,
    "T": 10.0,
    "J": 10.0,
    "Q": 10.0,
    "K": 10.0,
    "A": 11.0,
}
_RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
_RANK_ID = {rank: index for index, rank in enumerate(_RANKS, 2)}
_DEFAULT_HANDS: dict[str, tuple[float, float]] = {
    "Flush Five": (160.0, 16.0),
    "Flush House": (140.0, 14.0),
    "Five of a Kind": (120.0, 12.0),
    "Straight Flush": (100.0, 8.0),
    "Four of a Kind": (60.0, 7.0),
    "Full House": (40.0, 4.0),
    "Flush": (35.0, 4.0),
    "Straight": (30.0, 4.0),
    "Three of a Kind": (30.0, 3.0),
    "Two Pair": (20.0, 2.0),
    "Pair": (10.0, 2.0),
    "High Card": (5.0, 1.0),
}
_SCORING_CARD_COUNTS = {
    "Flush Five": 5,
    "Flush House": 5,
    "Five of a Kind": 5,
    "Straight Flush": 5,
    "Four of a Kind": 4,
    "Full House": 5,
    "Flush": 5,
    "Straight": 5,
    "Three of a Kind": 3,
    "Two Pair": 4,
    "Pair": 2,
    "High Card": 1,
}
_SUIT_JOKERS = {
    "j_greedy_joker": ("Diamonds", "mult"),
    "j_lusty_joker": ("Hearts", "mult"),
    "j_wrathful_joker": ("Spades", "mult"),
    "j_gluttenous_joker": ("Clubs", "mult"),
    "j_rough_gem": ("Diamonds", "economy"),
    "j_onyx_agate": ("Clubs", "mult"),
    "j_arrowhead": ("Spades", "chips"),
    "j_bloodstone": ("Hearts", "x_mult"),
}
_RETRIGGER_KEYS = {"j_hack", "j_sock_and_buskin", "j_hanging_chad", "j_dusk", "j_selzer", "j_mime"}
_SUPPORT_ONLY_KEYS = {"j_smeared"}
_COPY_KEYS = {"j_blueprint", "j_brainstorm"}

# The engine records every poker-hand category matched by the played cards, not
# just the highest display hand. Typed jokers therefore trigger on contained
# hands as well, for example The Duo on a Full House and The Tribe on a Straight
# Flush. These are the categories relevant to typed joker effects.
_MATCHED_HAND_TYPES: dict[str, frozenset[str]] = {
    "High Card": frozenset({"High Card"}),
    "Pair": frozenset({"Pair"}),
    "Two Pair": frozenset({"Two Pair", "Pair"}),
    "Three of a Kind": frozenset({"Three of a Kind", "Pair"}),
    "Straight": frozenset({"Straight"}),
    "Flush": frozenset({"Flush"}),
    "Full House": frozenset({"Full House", "Three of a Kind", "Two Pair", "Pair"}),
    "Four of a Kind": frozenset({"Four of a Kind", "Three of a Kind", "Pair"}),
    "Straight Flush": frozenset({"Straight Flush", "Straight", "Flush"}),
    "Five of a Kind": frozenset({"Five of a Kind", "Four of a Kind", "Three of a Kind", "Pair"}),
    "Flush House": frozenset({"Flush House", "Full House", "Flush", "Three of a Kind", "Two Pair", "Pair"}),
    "Flush Five": frozenset({"Flush Five", "Five of a Kind", "Four of a Kind", "Flush", "Three of a Kind", "Pair"}),
}


@dataclass(frozen=True)
class ScoreChannels:
    chips: float = 0.0
    additive_mult: float = 0.0
    x_mult: float = 1.0
    retriggers: float = 0.0
    option_value: float = 0.0


@dataclass(frozen=True)
class JokerMarginal:
    index: int
    key: str
    score_with: float
    score_without: float
    score_ratio: float
    modeled_effects: tuple[str, ...] = ()
    unmodeled_effects: tuple[str, ...] = ()
    modeled_effect_fraction: float = 0.0
    channels: ScoreChannels = ScoreChannels()


@dataclass(frozen=True)
class BuildValueEstimate:
    representative_hand_type: str
    representative_score_per_hand: float
    no_joker_baseline_score: float
    required_score_per_hand: float
    readiness_ratio: float
    joker_marginal_score_ratios: tuple[float, ...]
    joker_marginals: tuple[JokerMarginal, ...]
    modeled_effects: tuple[str, ...]
    unmodeled_effects: tuple[str, ...]
    channels: ScoreChannels


@dataclass
class _ScoreState:
    chips: float
    mult: float
    chips_added: float = 0.0
    mult_added: float = 0.0
    x_mult: float = 1.0
    retriggers: float = 0.0


@dataclass(frozen=True)
class _ScorePass:
    score: float
    channels: ScoreChannels
    modeled: tuple[tuple[int, str], ...]
    unmodeled: tuple[tuple[int, str], ...]


def _number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_active(joker: Mapping[str, Any]) -> bool:
    if joker.get("debuffed"):
        return False
    if joker.get("perishable") and joker.get("perish_tally") is not None:
        return int(joker.get("perish_tally") or 0) > 0
    return True


def _effective_joker(
    jokers: Sequence[Mapping[str, Any]],
    index: int,
    *,
    seen: frozenset[int] = frozenset(),
) -> Mapping[str, Any] | None:
    """Resolve Blueprint/Brainstorm to the compatible joker they copy."""
    if index < 0 or index >= len(jokers) or index in seen:
        return None
    joker = jokers[index]
    key = str(joker.get("key") or "")
    if key not in _COPY_KEYS:
        return joker

    target_index = 0 if key == "j_brainstorm" else index + 1
    if target_index == index or target_index < 0 or target_index >= len(jokers):
        return None
    target = jokers[target_index]
    if not (target.get("copy_compatible") or target.get("blueprint_compat") is True):
        return None
    return _effective_joker(jokers, target_index, seen=seen | {index})


def _representative_hand(info: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    details = _mapping(info.get("hand_details"))
    if not details:
        levels = _mapping(info.get("hand_levels"))
        counts = _mapping(info.get("hand_play_counts"))
        details = {
            name: {"level": levels.get(name, 1), "played": counts.get(name, 0)} for name in set(levels) | set(counts)
        }

    played = [(name, hand) for name, hand in details.items() if _number(_mapping(hand).get("played")) > 0]
    if played:
        name, detail = max(
            played,
            key=lambda item: (
                _number(_mapping(item[1]).get("played")),
                _number(_mapping(item[1]).get("level"), 1.0),
                -list(_DEFAULT_HANDS).index(item[0]) if item[0] in _DEFAULT_HANDS else -99,
            ),
        )
        return str(name), _mapping(detail)

    for fallback in ("Pair", "High Card"):
        if fallback in details:
            return fallback, _mapping(details[fallback])
    return "Pair", {}


def _hand_base(hand_type: str, detail: Mapping[str, Any]) -> tuple[float, float]:
    default_chips, default_mult = _DEFAULT_HANDS.get(hand_type, _DEFAULT_HANDS["Pair"])
    return _number(detail.get("chips"), default_chips), _number(detail.get("mult"), default_mult)


def _cards(info: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    deck = _mapping(info.get("deck_stats"))
    raw = deck.get("cards") or deck.get("card_descriptors") or ()
    cards = tuple(card for card in raw if isinstance(card, Mapping))
    if cards:
        return cards

    # Older synthetic snapshots only contain independent rank/suit counters.
    # These descriptors retain concentration information without inventing exact
    # rank+suit targets.
    result: list[Mapping[str, Any]] = []
    for suit, count in _mapping(deck.get("suit_counts")).items():
        result.extend({"rank": "", "suit": str(suit), "enhancement": "", "seal": ""} for _ in range(int(count)))
    if result:
        return tuple(result)
    size = max(1, int(deck.get("size", 52) or 52))
    return tuple({"rank": "", "suit": "", "enhancement": "", "seal": ""} for _ in range(size))


def _enhancement(card: Mapping[str, Any]) -> str:
    return str(card.get("enhancement") or card.get("effect") or "")


def _is_wild(card: Mapping[str, Any]) -> bool:
    return _enhancement(card) == "Wild Card"


def _supports_suit(card: Mapping[str, Any], suit: str, smeared: bool) -> bool:
    if _enhancement(card) == "Stone Card" or card.get("debuffed"):
        return False
    if _is_wild(card):
        return True
    actual = str(card.get("suit") or "")
    if actual == suit:
        return True
    if not smeared:
        return False
    return (actual in {"Hearts", "Diamonds"} and suit in {"Hearts", "Diamonds"}) or (
        actual in {"Spades", "Clubs"} and suit in {"Spades", "Clubs"}
    )


def _card_weight(scoring_count: int, deck_size: int) -> float:
    return scoring_count / max(1, deck_size)


def _edition(state: _ScoreState, edition: Any, *, weight: float = 1.0) -> None:
    ed = _mapping(edition)
    if ed.get("foil"):
        amount = 50.0 * weight
        state.chips += amount
        state.chips_added += amount
    if ed.get("holo") or ed.get("holographic"):
        amount = 10.0 * weight
        state.mult += amount
        state.mult_added += amount
    if ed.get("polychrome"):
        factor = 1.5**weight
        state.mult *= factor
        state.x_mult *= factor


def _joker_extra(joker: Mapping[str, Any]) -> Any:
    if "extra" in joker:
        return joker.get("extra")
    return _mapping(joker.get("config")).get("extra")


def _repetitions(
    card: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]],
    scoring_count: int,
    hands_available: int,
) -> tuple[float, list[tuple[int, str]]]:
    repetitions = 1.0 + (1.0 if card.get("seal") == "Red" else 0.0)
    modeled: list[tuple[int, str]] = []
    rank = str(card.get("rank") or "")
    rank_id = _RANK_ID.get(rank, 0)
    for index, _owned in enumerate(jokers):
        joker = _effective_joker(jokers, index)
        if joker is None or not _is_active(joker):
            continue
        key = str(joker.get("key") or "")
        extra = _number(_joker_extra(joker), 1.0)
        if key == "j_hack" and rank_id in {2, 3, 4, 5}:
            repetitions += extra
            modeled.append((index, "hack_2_to_5"))
        elif key == "j_sock_and_buskin" and rank_id in {11, 12, 13}:
            repetitions += extra
            modeled.append((index, "face_retrigger"))
        elif key == "j_hanging_chad":
            repetitions += extra / max(1, scoring_count)
            modeled.append((index, "first_card_retrigger"))
        elif key == "j_dusk":
            repetitions += extra / max(1, hands_available)
            modeled.append((index, "last_hand_retrigger"))
        elif key == "j_selzer":
            repetitions += 1.0
            modeled.append((index, "all_card_retrigger"))
    return repetitions, modeled


def _individual_joker_effect(
    state: _ScoreState,
    card: Mapping[str, Any],
    joker: Mapping[str, Any],
    *,
    index: int,
    smeared: bool,
    idol: Mapping[str, Any],
    weight: float = 1.0,
) -> tuple[int, str] | None:
    if not _is_active(joker):
        return None
    key = str(joker.get("key") or "")
    name = str(joker.get("name") or "")
    effect = str(joker.get("effect") or "")
    extra = _joker_extra(joker)

    suit_spec = _SUIT_JOKERS.get(key)
    if effect == "Suit Mult":
        cfg = _mapping(extra)
        suit_spec = (str(cfg.get("suit") or ""), "mult")
    if suit_spec is not None:
        suit, channel = suit_spec
        if _supports_suit(card, suit, smeared):
            cfg = _mapping(extra)
            if channel == "mult":
                amount = _number(cfg.get("s_mult"), _number(extra, 3.0)) * weight
                state.mult += amount
                state.mult_added += amount
            elif channel == "chips":
                amount = _number(extra, _number(cfg.get("chips"))) * weight
                state.chips += amount
                state.chips_added += amount
            elif channel == "x_mult":
                odds = max(1.0, _number(cfg.get("odds"), 1.0))
                factor = _number(cfg.get("Xmult"), 1.0)
                expected_factor = 1.0 + (factor - 1.0) / odds
                weighted_factor = expected_factor**weight
                state.mult *= weighted_factor
                state.x_mult *= weighted_factor
            return index, f"suit_{channel}"
        return index, f"suit_{channel}"

    rank = str(card.get("rank") or "")
    rank_id = _RANK_ID.get(rank, 0)
    cfg = _mapping(extra)
    if name == "Scary Face" and rank_id in {11, 12, 13}:
        amount = _number(extra) * weight
        state.chips += amount
        state.chips_added += amount
        return index, "face_chips"
    if name == "Smiley Face" and rank_id in {11, 12, 13}:
        amount = _number(extra) * weight
        state.mult += amount
        state.mult_added += amount
        return index, "face_mult"
    if name == "Scholar" and rank_id == 14:
        chips = _number(cfg.get("chips")) * weight
        mult = _number(cfg.get("mult")) * weight
        state.chips += chips
        state.mult += mult
        state.chips_added += chips
        state.mult_added += mult
        return index, "ace_chips_mult"
    if name == "Walkie Talkie" and rank_id in {4, 10}:
        chips = _number(cfg.get("chips")) * weight
        mult = _number(cfg.get("mult")) * weight
        state.chips += chips
        state.mult += mult
        state.chips_added += chips
        state.mult_added += mult
        return index, "rank_4_10"
    if name == "Fibonacci" and rank_id in {2, 3, 5, 8, 14}:
        amount = _number(extra) * weight
        state.mult += amount
        state.mult_added += amount
        return index, "fibonacci_ranks"
    if name == "Even Steven" and 2 <= rank_id <= 10 and rank_id % 2 == 0:
        amount = _number(extra) * weight
        state.mult += amount
        state.mult_added += amount
        return index, "even_ranks"
    if name == "Odd Todd" and (rank_id == 14 or (rank_id <= 10 and rank_id % 2 == 1)):
        amount = _number(extra) * weight
        state.chips += amount
        state.chips_added += amount
        return index, "odd_ranks"
    if key == "j_idol" or name == "The Idol":
        target_rank = str(idol.get("rank") or "")
        target_suit = str(idol.get("suit") or "")
        if rank == target_rank and target_suit and _supports_suit(card, target_suit, smeared):
            factor = _number(extra, 1.0) ** weight
            state.mult *= factor
            state.x_mult *= factor
        return index, "idol_exact_target"
    return None


def _held_effects(
    state: _ScoreState,
    cards: Sequence[Mapping[str, Any]],
    jokers: Sequence[Mapping[str, Any]],
    held_count: int,
) -> list[tuple[int, str]]:
    if held_count <= 0 or not cards:
        return []
    weight = held_count / len(cards)
    mime_indices: list[int] = []
    mime_extra = 0.0
    for index in range(len(jokers)):
        effective = _effective_joker(jokers, index)
        if effective is None or not _is_active(effective):
            continue
        if str(effective.get("key") or "") == "j_mime":
            mime_indices.append(index)
            mime_extra += _number(_joker_extra(effective), 1.0)

    modeled: list[tuple[int, str]] = []
    mime_triggered = False
    for card in cards:
        if card.get("debuffed"):
            continue
        enhancement = _enhancement(card)
        base_h_mult = _number(card.get("h_mult"))
        base_h_x_mult = _number(card.get("h_x_mult"), 1.0)
        if enhancement == "Steel Card" and base_h_x_mult <= 1.0:
            base_h_x_mult = 1.5
        has_base_held_effect = base_h_mult != 0.0 or base_h_x_mult > 1.0
        repetitions = 1.0 + mime_extra if has_base_held_effect else 1.0
        reps = weight * repetitions
        mime_triggered = mime_triggered or (has_base_held_effect and mime_extra > 0.0)

        if base_h_x_mult > 1.0:
            factor = base_h_x_mult**reps
            state.mult *= factor
            state.x_mult *= factor
        held_mult = base_h_mult * reps
        if held_mult:
            state.mult += held_mult
            state.mult_added += held_mult

        for index in range(len(jokers)):
            joker = _effective_joker(jokers, index)
            if joker is None or not _is_active(joker):
                continue
            name = str(joker.get("name") or "")
            rank = str(card.get("rank") or "")
            if name == "Shoot the Moon" and rank == "Q":
                amount = 13.0 * reps
                state.mult += amount
                state.mult_added += amount
                modeled.append((index, "held_queens"))
            elif name == "Baron" and rank == "K":
                factor = _number(_joker_extra(joker), 1.0) ** reps
                state.mult *= factor
                state.x_mult *= factor
                modeled.append((index, "held_kings"))
    if mime_triggered:
        modeled.extend((index, "held_retrigger") for index in mime_indices)
    return modeled


def _main_effect(
    state: _ScoreState,
    joker: Mapping[str, Any],
    *,
    index: int,
    hand_type: str,
) -> tuple[int, str] | None:
    if not _is_active(joker):
        return None
    key = str(joker.get("key") or "")
    if key in _SUIT_JOKERS or key in _RETRIGGER_KEYS or key in _SUPPORT_ONLY_KEYS or key == "j_idol":
        return None

    restriction = str(joker.get("type") or "")
    matched_types = _MATCHED_HAND_TYPES.get(hand_type, frozenset({hand_type}))
    matches = not restriction or restriction in matched_types
    if not matches:
        return index, "hand_condition"

    x_mult = _number(joker.get("x_mult"), _number(joker.get("base_x_mult"), 1.0))
    if x_mult > 1.0:
        state.mult *= x_mult
        state.x_mult *= x_mult
        return index, "live_x_mult"
    t_mult = _number(joker.get("t_mult"), _number(joker.get("base_t_mult")))
    if t_mult > 0.0 and restriction:
        state.mult += t_mult
        state.mult_added += t_mult
        return index, "hand_additive_mult"
    t_chips = _number(joker.get("t_chips"), _number(joker.get("base_t_chips")))
    if t_chips > 0.0 and restriction:
        state.chips += t_chips
        state.chips_added += t_chips
        return index, "hand_chips"
    mult = _number(joker.get("mult"), _number(joker.get("base_mult")))
    if mult > 0.0:
        state.mult += mult
        state.mult_added += mult
        return index, "additive_mult"

    extra = _mapping(_joker_extra(joker))
    name = str(joker.get("name") or "")
    if name == "Stuntman":
        amount = _number(extra.get("chip_mod"))
        state.chips += amount
        state.chips_added += amount
        return index, "direct_chips"
    if name in {"Wee Joker", "Castle", "Square Joker", "Runner", "Ice Cream"}:
        amount = _number(extra.get("chips"))
        if amount:
            state.chips += amount
            state.chips_added += amount
            return index, "live_chips"
    if name == "Canio":
        factor = _number(joker.get("caino_xmult"), 1.0)
        if factor > 1.0:
            state.mult *= factor
            state.x_mult *= factor
            return index, "live_x_mult"
    return None


def _has_unknown_effect(joker: Mapping[str, Any], modeled: set[str]) -> bool:
    if not _is_active(joker) or modeled:
        return False
    if str(joker.get("key") or "") in _SUPPORT_ONLY_KEYS:
        return False
    if _number(joker.get("dollars")) or joker.get("is_economy"):
        return False
    return bool(joker.get("effect") or joker.get("name") or joker.get("key"))


def _score_pass(
    info: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]],
    *,
    hand_type: str,
    hand_detail: Mapping[str, Any],
) -> _ScorePass:
    base_chips, base_mult = _hand_base(hand_type, hand_detail)
    state = _ScoreState(chips=base_chips, mult=base_mult)
    cards = _cards(info)
    scoring_count = _SCORING_CARD_COUNTS.get(hand_type, 2)
    card_weight = _card_weight(scoring_count, len(cards))
    hands_available = max(1, int(info.get("hands_available") or info.get("hands_left") or 4))
    smeared = any(str(joker.get("key") or "") == "j_smeared" and _is_active(joker) for joker in jokers)
    idol = _mapping(info.get("idol_card") or _mapping(info.get("dynamic_targets")).get("idol_card"))
    modeled: list[tuple[int, str]] = []

    sorted_cards = sorted(cards, key=lambda card: _RANK_ID.get(str(card.get("rank") or ""), 0), reverse=True)
    for card in sorted_cards:
        if card.get("debuffed"):
            continue
        repetitions, retrigger_markers = _repetitions(card, jokers, scoring_count, hands_available)
        expected_repetitions = repetitions * card_weight
        state.retriggers += max(0.0, expected_repetitions - card_weight)
        modeled.extend(retrigger_markers)
        for _ in range(int(expected_repetitions)):
            _score_card_once(state, card, jokers, smeared=smeared, idol=idol, modeled=modeled)
        fractional = expected_repetitions % 1.0
        if fractional:
            _score_card_fraction(state, card, jokers, fractional, smeared=smeared, idol=idol, modeled=modeled)

    hand_size = max(scoring_count, int(info.get("hand_size") or 8))
    modeled.extend(_held_effects(state, cards, jokers, max(0, hand_size - scoring_count)))

    for index, owned in enumerate(jokers):
        _edition(state, owned.get("edition"))
        if any(_mapping(owned.get("edition")).get(name) for name in ("foil", "holo", "holographic", "polychrome")):
            modeled.append((index, "edition"))
        joker = _effective_joker(jokers, index)
        if joker is None:
            continue
        marker = _main_effect(state, joker, index=index, hand_type=hand_type)
        if marker:
            modeled.append(marker)
    if smeared:
        modeled.extend(
            (i, "smeared_suits") for i, joker in enumerate(jokers) if str(joker.get("key") or "") == "j_smeared"
        )
    for index in range(len(jokers)):
        joker = _effective_joker(jokers, index)
        if joker is None or not _is_active(joker):
            continue
        key = str(joker.get("key") or "")
        if key in _RETRIGGER_KEYS:
            modeled.append((index, "retrigger_condition"))
        if key == "j_hologram" or joker.get("is_scaling_xmult"):
            modeled.append((index, "live_x_mult"))

    modeled_by_index: dict[int, set[str]] = {}
    for index, effect in modeled:
        modeled_by_index.setdefault(index, set()).add(effect)
    unmodeled = [
        (index, "conditional_effect")
        for index, joker in enumerate(jokers)
        if _has_unknown_effect(joker, modeled_by_index.get(index, set()))
    ]
    raw_score = state.chips * max(0.0, state.mult)
    nearest_integer = round(raw_score)
    if isclose(raw_score, nearest_integer, rel_tol=1e-12, abs_tol=1e-9):
        raw_score = float(nearest_integer)
    score = float(max(0, floor(raw_score)))
    return _ScorePass(
        score=score,
        channels=ScoreChannels(
            chips=state.chips_added,
            additive_mult=state.mult_added,
            x_mult=state.x_mult,
            retriggers=state.retriggers,
            option_value=0.0,
        ),
        modeled=tuple(sorted(set(modeled))),
        unmodeled=tuple(unmodeled),
    )


def _score_card_once(
    state: _ScoreState,
    card: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]],
    *,
    smeared: bool,
    idol: Mapping[str, Any],
    modeled: list[tuple[int, str]],
    weight: float = 1.0,
) -> None:
    enhancement = _enhancement(card)
    perma_bonus = _number(card.get("perma_bonus"))
    if enhancement == "Stone Card":
        chips = (_number(card.get("bonus")) if "bonus" in card else 50.0) + perma_bonus
    else:
        bonus = _number(card.get("bonus"))
        if enhancement == "Bonus Card" and "bonus" not in card:
            bonus = 30.0
        chips = _RANK_NOMINAL.get(str(card.get("rank") or ""), 0.0) + bonus + perma_bonus
    weighted_chips = chips * weight
    state.chips += weighted_chips
    state.chips_added += weighted_chips
    if enhancement == "Mult Card":
        amount = _number(card.get("mult"), 4.0) * weight
        state.mult += amount
        state.mult_added += amount
    elif enhancement == "Lucky Card":
        amount = (_number(card.get("mult"), 20.0) / 5.0) * weight
        state.mult += amount
        state.mult_added += amount
    elif enhancement == "Glass Card":
        factor = _number(card.get("x_mult"), 2.0) ** weight
        state.mult *= factor
        state.x_mult *= factor

    for index in range(len(jokers)):
        joker = _effective_joker(jokers, index)
        if joker is None:
            continue
        marker = _individual_joker_effect(
            state,
            card,
            joker,
            index=index,
            smeared=smeared,
            idol=idol,
            weight=weight,
        )
        if marker:
            modeled.append(marker)
    edition = card.get("edition") or card.get("edition_key")
    if isinstance(edition, str):
        edition = {edition: True}
    _edition(state, edition, weight=weight)


def _score_card_fraction(
    state: _ScoreState,
    card: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]],
    weight: float,
    *,
    smeared: bool,
    idol: Mapping[str, Any],
    modeled: list[tuple[int, str]],
) -> None:
    _score_card_once(
        state,
        card,
        jokers,
        smeared=smeared,
        idol=idol,
        modeled=modeled,
        weight=weight,
    )


def _effects_for_index(effects: Iterable[tuple[int, str]], index: int) -> tuple[str, ...]:
    return tuple(sorted({effect for effect_index, effect in effects if effect_index == index}))


def estimate_build_value(
    info: Mapping[str, Any],
    jokers: Sequence[Mapping[str, Any]] | None = None,
) -> BuildValueEstimate:
    """Estimate representative score and ordered joker marginal ratios."""
    owned = tuple(jokers if jokers is not None else (info.get("joker_details") or ()))
    hand_type, hand_detail = _representative_hand(info)
    baseline = _score_pass(info, (), hand_type=hand_type, hand_detail=hand_detail)
    full = _score_pass(info, owned, hand_type=hand_type, hand_detail=hand_detail)

    marginals: list[JokerMarginal] = []
    ratios: list[float] = []
    for index, joker in enumerate(owned):
        without = _score_pass(info, owned[:index] + owned[index + 1 :], hand_type=hand_type, hand_detail=hand_detail)
        ratio = full.score / max(1.0, without.score)
        ratios.append(ratio)
        modeled = _effects_for_index(full.modeled, index)
        unmodeled = _effects_for_index(full.unmodeled, index)
        modeled_fraction = 1.0 if modeled else 0.0
        if modeled and unmodeled:
            modeled_fraction = len(modeled) / (len(modeled) + len(unmodeled))
        marginals.append(
            JokerMarginal(
                index=index,
                key=str(joker.get("key") or ""),
                score_with=full.score,
                score_without=without.score,
                score_ratio=ratio,
                modeled_effects=modeled,
                unmodeled_effects=unmodeled,
                modeled_effect_fraction=modeled_fraction,
                channels=ScoreChannels(
                    chips=full.channels.chips - without.channels.chips,
                    additive_mult=full.channels.additive_mult - without.channels.additive_mult,
                    x_mult=full.channels.x_mult / max(without.channels.x_mult, 1e-9),
                    retriggers=full.channels.retriggers - without.channels.retriggers,
                    option_value=0.0,
                ),
            )
        )

    blind_target = _number(info.get("blind_target"))
    hands_available = max(1, int(info.get("hands_available") or info.get("hands_left") or 4))
    required = blind_target / hands_available if blind_target > 0 else 0.0
    readiness = full.score / required if required > 0 else 0.0
    modeled_effects = tuple(sorted({effect for _, effect in full.modeled}))
    unmodeled_effects = tuple(sorted({effect for _, effect in full.unmodeled}))
    return BuildValueEstimate(
        representative_hand_type=hand_type,
        representative_score_per_hand=full.score,
        no_joker_baseline_score=baseline.score,
        required_score_per_hand=required,
        readiness_ratio=readiness,
        joker_marginal_score_ratios=tuple(ratios),
        joker_marginals=tuple(marginals),
        modeled_effects=modeled_effects,
        unmodeled_effects=unmodeled_effects,
        channels=full.channels,
    )


def estimate_hand_score(
    info: Mapping[str, Any],
    hand_type: str,
    jokers: Sequence[Mapping[str, Any]] | None = None,
) -> float:
    """Estimate one explicit poker hand instead of the historical main hand.

    Strategic reward shaping uses this public wrapper to compare only hands the
    captured deck can draw reliably.  Keeping the score model here ensures shop
    counterfactuals and the ordinary build evaluator use identical joker logic.
    """
    details = _mapping(info.get("hand_details"))
    detail = _mapping(details.get(hand_type))
    owned = tuple(jokers if jokers is not None else (info.get("joker_details") or ()))
    return _score_pass(
        info,
        owned,
        hand_type=str(hand_type),
        hand_detail=detail,
    ).score
