from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any

try:
    import cython
except ImportError:  # pragma: no cover - dependency-free core install without Cython
    from ._cyshadow import cython

from ._helpers import _as_dict
from .instances import remove_joker, sync_all_jokers
from .models import ConsumableInstance, JokerInstance, PlayingCard, RunState
from .runtime import add_generated_consumable, add_playing_cards, copy_playing_card

RANK_TO_ID = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}

RANK_TO_NOMINAL = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 10,
    "Q": 10,
    "K": 10,
    "A": 11,
}

SUIT_TO_NOMINAL = {"Diamonds": 0.01, "Clubs": 0.02, "Hearts": 0.03, "Spades": 0.04}
POKER_HAND_ORDER = (
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
)


@dataclass(slots=True)
class ScoreResult:
    hand_name: str
    display_name: str
    poker_hands: dict[str, list[list[PlayingCard]]]
    scoring_cards: list[PlayingCard]
    held_cards: list[PlayingCard]
    chips: float
    mult: float
    total: int
    dollars: int = 0


def _center(state: RunState, key: str) -> dict[str, Any]:
    return state.data.centers[key]


def _card_center(state: RunState, card: PlayingCard) -> dict[str, Any]:
    return _center(state, card.center_key)


def _joker_center(state: RunState, joker: JokerInstance) -> dict[str, Any]:
    return _center(state, joker.center_key)


def _edition_dict_from_key(edition_key: str | None) -> dict[str, bool] | None:
    return {edition_key: True} if edition_key else None


# Editions contribute fixed scoring bonuses: foil = +50 chips, holo = +10 mult,
# polychrome = x1.5 mult. Each only fires for its own edition, returning the
# additive/multiplicative identity otherwise.
def _edition_chip_mod(edition: dict[str, bool] | None) -> float:
    if not edition:
        return 0
    return 50 if edition.get("foil") else 0


def _edition_mult_mod(edition: dict[str, bool] | None) -> float:
    if not edition:
        return 0
    return 10 if edition.get("holo") else 0


def _edition_x_mult_mod(edition: dict[str, bool] | None) -> float:
    if not edition:
        return 1
    return 1.5 if edition.get("polychrome") else 1


def _mod_chips(state: RunState, chips: float) -> float:
    if state.modifiers.get("chips_dollar_cap"):
        return min(chips, max(state.dollars, 0))
    return chips


def _mod_mult(mult: float) -> float:
    # Symmetry counterpart to _mod_chips; mult is never capped, so this is the
    # identity. Kept as a hook so all mult writes route through one place.
    return mult


def _card_name(state: RunState, card: PlayingCard) -> str:
    return str(_card_center(state, card)["name"])


# A center key's "effect" string is static within loaded game data, so cache it
# globally. Assumes a single GameData per process: the cache is not keyed by
# state/data instance and is never cleared.
_card_effect_cache: dict[str, str] = {}


def _card_effect(state: RunState, card: PlayingCard) -> str:
    ck = card.center_key
    cached = _card_effect_cache.get(ck)
    if cached is not None:
        return cached
    result = str(_card_center(state, card).get("effect", ""))
    _card_effect_cache[ck] = result
    return result


def _card_id(state: RunState, card: PlayingCard) -> int:
    # Stone Cards have no rank, so give each a unique negative id (from object
    # stable identity), that way they never group into pairs/straights with each other.
    if _card_effect(state, card) == "Stone Card":
        return -card.reward_uid
    return RANK_TO_ID[card.rank]


@cython.locals(base=cython.double, face_nominal=cython.double, suit_nominal=cython.double, suit_mult=cython.int)
def _card_nominal(state: RunState, card: PlayingCard) -> float:
    # A sortable scalar that ranks cards rank-first, then suit, used to pick which
    # cards "score" within a hand. The tiny suit (1e-4) and creation-order (1e-6)
    # terms preserve ordering across copied states; Stone Cards get suit_mult=-1000 so
    # they always sort last (they shouldn't contribute to rank/suit hands).
    base = RANK_TO_NOMINAL[card.rank]
    face_nominal = 0.1 if card.rank == "J" else 0.2 if card.rank == "Q" else 0.3 if card.rank == "K" else 0.4 if card.rank == "A" else 0
    suit_nominal = SUIT_TO_NOMINAL[card.suit]
    suit_mult = -1000 if _card_effect(state, card) == "Stone Card" else 1
    return base + suit_nominal * suit_mult + suit_nominal * 0.0001 * suit_mult + face_nominal + 0.000001 * (1 - card.reward_uid / 1_603_301)


def _is_face(state: RunState, card: PlayingCard, *, from_boss: bool = False) -> bool:
    if card.debuff and not from_boss:
        return False
    return _card_id(state, card) in {11, 12, 13} or state.has_joker("Pareidolia")


def _is_suit(
    state: RunState,
    card: PlayingCard,
    suit: str,
    *,
    bypass_debuff: bool = False,
    flush_calc: bool = False,
) -> bool:
    if not flush_calc and card.debuff and not bypass_debuff:
        return False
    if _card_effect(state, card) == "Stone Card":
        return False
    if _card_name(state, card) == "Wild Card" and (flush_calc is False or not card.debuff):
        return True
    if state.has_joker("Smeared Joker"):
        both_red = card.suit in {"Hearts", "Diamonds"} and suit in {"Hearts", "Diamonds"}
        both_black = card.suit in {"Spades", "Clubs"} and suit in {"Spades", "Clubs"}
        if both_red or both_black:
            return True
    return card.suit == suit


def _chip_bonus(state: RunState, card: PlayingCard) -> float:
    if card.debuff:
        return 0
    center = _card_center(state, card)
    bonus = int(_as_dict(center.get("config")).get("bonus", 0) or 0)
    if _card_effect(state, card) == "Stone Card":
        return bonus + card.perma_bonus
    return RANK_TO_NOMINAL[card.rank] + bonus + card.perma_bonus


def _chip_mult(state: RunState, card: PlayingCard) -> float:
    if card.debuff:
        return 0
    center = _card_center(state, card)
    if center.get("set") == "Joker":
        return 0
    if _card_effect(state, card) == "Lucky Card":
        if float(state.pseudorandom.pseudorandom("lucky_mult")) < state.probabilities["normal"] / 5:
            return float(_as_dict(center.get("config"))["mult"])
        return 0
    return float(_as_dict(center.get("config")).get("mult", 0) or 0)


def _chip_x_mult(state: RunState, card: PlayingCard) -> float:
    if card.debuff:
        return 0
    center = _card_center(state, card)
    if center.get("set") == "Joker":
        return 0
    value = float(_as_dict(center.get("config")).get("Xmult", 1) or 1)
    return value if value > 1 else 0


def _chip_h_mult(state: RunState, card: PlayingCard) -> float:
    if card.debuff:
        return 0
    return float(_as_dict(_card_center(state, card).get("config")).get("h_mult", 0) or 0)


def _chip_h_x_mult(state: RunState, card: PlayingCard) -> float:
    if card.debuff:
        return 0
    return float(_as_dict(_card_center(state, card).get("config")).get("h_x_mult", 0) or 0)


def _p_dollars(state: RunState, card: PlayingCard) -> int:
    if card.debuff:
        return 0
    center = _card_center(state, card)
    ret = 3 if card.seal == "Gold" else 0
    p_dollars = int(_as_dict(center.get("config")).get("p_dollars", 0) or 0)
    if p_dollars > 0:
        if _card_effect(state, card) == "Lucky Card":
            if float(state.pseudorandom.pseudorandom("lucky_money")) < state.probabilities["normal"] / 15:
                ret += p_dollars
        else:
            ret += p_dollars
    return ret


def _get_x_same(state: RunState, num: int, hand: list[PlayingCard]) -> list[list[PlayingCard]]:
    # Group cards by card_id in one pass (O(n) instead of O(n²))
    groups: dict[int, list[PlayingCard]] = {}
    for card in reversed(hand):
        cid = _card_id(state, card)
        groups.setdefault(cid, []).append(card)
    return [cards for cid in sorted(groups, reverse=True) if len(cards := groups[cid]) == num]


@cython.locals(required=cython.int)
def _get_flush(state: RunState, hand: list[PlayingCard]) -> list[list[PlayingCard]]:
    ret: list[list[PlayingCard]] = []
    four_fingers = state.has_joker("Four Fingers")
    required = 5 - (1 if four_fingers else 0)
    if len(hand) > 5 or len(hand) < required:
        return ret
    for suit in ("Spades", "Hearts", "Clubs", "Diamonds"):
        cards = [card for card in hand if _is_suit(state, card, suit, flush_calc=True)]
        if len(cards) >= required:
            ret.append(cards)
            return ret
    return []


@cython.locals(
    required=cython.int,
    card_id=cython.Py_ssize_t,
    j=cython.int,
    actual=cython.int,
    straight_length=cython.int,
)
def _get_straight(state: RunState, hand: list[PlayingCard]) -> list[list[PlayingCard]]:
    ret: list[list[PlayingCard]] = []
    four_fingers = state.has_joker("Four Fingers")
    required = 5 - (1 if four_fingers else 0)
    if len(hand) > 5 or len(hand) < required:
        return ret

    ids: dict[int, list[PlayingCard]] = {}
    for card in hand:
        card_id = _card_id(state, card)
        if 1 < card_id < 15:
            ids.setdefault(card_id, []).append(card)

    # Scan ranks 1..14 for a consecutive run. j==1 maps to rank 14 first so the
    # Ace can play low (A-2-3-4-5); it's also rank 14 at the top end (10-J-Q-K-A).
    # Shortcut lets the run skip a single missing rank (can_skip / skipped_rank).
    straight_cards: list[PlayingCard] = []
    straight_length = 0
    straight = False
    can_skip = state.has_joker("Shortcut")
    skipped_rank = False
    for j in range(1, 15):
        actual = 14 if j == 1 else j
        if actual in ids:
            straight_length += 1
            skipped_rank = False
            straight_cards.extend(ids[actual])
        elif can_skip and not skipped_rank and j != 14:
            skipped_rank = True
        else:
            straight_length = 0
            skipped_rank = False
            if not straight:
                straight_cards = []
            if straight:
                break
        if straight_length >= required:
            straight = True
    if not straight:
        return ret
    ret.append(straight_cards)
    return ret


def _get_highest(state: RunState, hand: list[PlayingCard]) -> list[list[PlayingCard]]:
    if not hand:
        return []
    highest = max(hand, key=lambda card: _card_nominal(state, card))
    return [[highest]]


def evaluate_poker_hand(state: RunState, hand: list[PlayingCard]) -> dict[str, list[list[PlayingCard]]]:
    results = {
        "Flush Five": [],
        "Flush House": [],
        "Five of a Kind": [],
        "Straight Flush": [],
        "Four of a Kind": [],
        "Full House": [],
        "Flush": [],
        "Straight": [],
        "Three of a Kind": [],
        "Two Pair": [],
        "Pair": [],
        "High Card": [],
    }
    parts = {
        "_5": _get_x_same(state, 5, hand),
        "_4": _get_x_same(state, 4, hand),
        "_3": _get_x_same(state, 3, hand),
        "_2": _get_x_same(state, 2, hand),
        "_flush": _get_flush(state, hand),
        "_straight": _get_straight(state, hand),
        "_highest": _get_highest(state, hand),
    }

    if parts["_5"] and parts["_flush"]:
        results["Flush Five"] = parts["_5"]
    if parts["_3"] and parts["_2"] and parts["_flush"]:
        results["Flush House"] = [parts["_3"][0] + parts["_2"][0]]
    if parts["_5"]:
        results["Five of a Kind"] = parts["_5"]
    if parts["_flush"] and parts["_straight"]:
        flush_cards = list(parts["_flush"][0])
        straight_cards = list(parts["_straight"][0])
        ret = list(flush_cards)
        for card in straight_cards:
            if card not in flush_cards:
                ret.append(card)
        results["Straight Flush"] = [ret]
    if parts["_4"]:
        results["Four of a Kind"] = parts["_4"]
    if parts["_3"] and parts["_2"]:
        results["Full House"] = [parts["_3"][0] + parts["_2"][0]]
    if parts["_flush"]:
        results["Flush"] = parts["_flush"]
    if parts["_straight"]:
        results["Straight"] = parts["_straight"]
    if parts["_3"]:
        results["Three of a Kind"] = parts["_3"]
    if len(parts["_2"]) == 2 or (len(parts["_3"]) == 1 and len(parts["_2"]) == 1):
        second_pair = parts["_2"][1] if len(parts["_2"]) > 1 else parts["_3"][0]
        results["Two Pair"] = [parts["_2"][0] + second_pair]
    if parts["_2"]:
        results["Pair"] = parts["_2"]
    if parts["_highest"]:
        results["High Card"] = parts["_highest"]
    if results["Five of a Kind"]:
        results["Four of a Kind"] = [results["Five of a Kind"][0][:4]]
    if results["Four of a Kind"]:
        results["Three of a Kind"] = [results["Four of a Kind"][0][:3]]
    if results["Three of a Kind"]:
        results["Pair"] = [results["Three of a Kind"][0][:2]]
    return results


def get_poker_hand_info(
    state: RunState,
    cards: list[PlayingCard],
) -> tuple[str, str, dict[str, list[list[PlayingCard]]], list[PlayingCard]]:
    poker_hands = evaluate_poker_hand(state, cards)
    text = "High Card"
    scoring_hand = poker_hands["High Card"][0] if poker_hands["High Card"] else []
    for hand_name in POKER_HAND_ORDER:
        if poker_hands[hand_name]:
            text = hand_name
            scoring_hand = poker_hands[hand_name][0]
            break

    display_name = text
    if text == "Straight Flush":
        minimum = min((_card_id(state, card) for card in scoring_hand), default=0)
        if minimum >= 10:
            display_name = "Royal Flush"
    return text, display_name, poker_hands, scoring_hand


def _flush_main_editions(
    chips: float,
    mult: float,
    edition: dict[str, bool] | None,
) -> tuple[float, float]:
    chips += _edition_chip_mod(edition)
    mult += _edition_mult_mod(edition)
    mult *= _edition_x_mult_mod(edition)
    return chips, mult


def _add_money(state: RunState, amount: int) -> None:
    state.dollars += amount
    state.dollar_buffer += amount


def _type_matches(poker_hands: dict[str, list[list[PlayingCard]]], type_name: str) -> bool:
    return bool(type_name) and bool(poker_hands[type_name])


def _remaining_deck_count(state: RunState) -> int:
    if state.draw_pile or state.hand_cards or state.discard_pile or state.play_cards:
        return len(state.draw_pile)
    return len(state.deck_cards)


def _blueprint_target(state: RunState, joker: JokerInstance, index: int) -> JokerInstance | None:
    name = _joker_center(state, joker)["name"]
    target = None
    if name == "Brainstorm":
        target = state.jokers[0] if state.jokers else None
    elif name == "Blueprint" and index + 1 < len(state.jokers):
        target = state.jokers[index + 1]
    if target is joker:
        return None
    if target and _joker_center(state, target).get("blueprint_compat"):
        return target
    return None


def _evaluate_joker(
    state: RunState,
    joker: JokerInstance,
    *,
    index: int,
    phase: str,
    full_hand: list[PlayingCard],
    scoring_hand: list[PlayingCard],
    held_hand: list[PlayingCard],
    scoring_name: str,
    poker_hands: dict[str, list[list[PlayingCard]]],
    other_card: PlayingCard | None = None,
    other_joker: JokerInstance | None = None,
    card_effects: list[dict[str, float]] | None = None,
    lucky_triggered: bool = False,
    blueprint_depth: int = 0,
) -> dict[str, float] | None:
    center = _joker_center(state, joker)
    name = center["name"]
    if joker.debuff:
        return None
    if name in {"Blueprint", "Brainstorm"}:
        if blueprint_depth > len(state.jokers) + 1:
            return None
        target = _blueprint_target(state, joker, index)
        if target is None:
            return None
        target_index = next(i for i, owned in enumerate(state.jokers) if owned is target)
        return _evaluate_joker(
            state,
            target,
            index=target_index,
            phase=phase,
            full_hand=full_hand,
            scoring_hand=scoring_hand,
            held_hand=held_hand,
            scoring_name=scoring_name,
            poker_hands=poker_hands,
            other_card=other_card,
            other_joker=other_joker,
            card_effects=card_effects,
            lucky_triggered=lucky_triggered,
            blueprint_depth=blueprint_depth + 1,
        )

    # TODO: replace this name-dispatch if/elif chain with a per-joker handler
    # table keyed by center_key so each phase is a dict lookup.

    if phase == "before":
        if name == "Spare Trousers" and (poker_hands["Two Pair"] or poker_hands["Full House"]) and isinstance(joker.extra, int):
            joker.mult += joker.extra
        elif (name == "Square Joker" and len(full_hand) == 4 and isinstance(joker.extra, dict)) or (name == "Runner" and poker_hands["Straight"] and isinstance(joker.extra, dict)):
            joker.extra["chips"] += int(joker.extra.get("chip_mod", 0) or 0)
        elif name == "Ride the Bus" and isinstance(joker.extra, int):
            if any(_is_face(state, card) for card in scoring_hand):
                joker.mult = 0
            else:
                joker.mult += joker.extra
        elif name == "Space Joker" and isinstance(joker.extra, int):
            if float(state.pseudorandom.pseudorandom("space")) < state.probabilities["normal"] / float(joker.extra):
                _level_up_hand(state, scoring_name)
        elif name == "Green Joker" and isinstance(joker.extra, dict):
            joker.mult += int(joker.extra.get("hand_add", 0) or 0)
        elif name == "Midas Mask":
            for card in scoring_hand:
                if _is_face(state, card):
                    card.center_key = "m_gold"
        elif name == "Vampire" and isinstance(joker.extra, (int, float)):
            enhanced = [card for card in scoring_hand if card.center_key != "c_base" and not card.debuff]
            if enhanced:
                joker.x_mult += len(enhanced) * float(joker.extra)
                for card in enhanced:
                    card.center_key = "c_base"
        elif name == "To Do List" and isinstance(joker.extra, dict) and scoring_name == joker.to_do_poker_hand:
            _add_money(state, int(joker.extra.get("dollars", 0) or 0))
        elif name == "DNA" and state.current_round.hands_played == 0 and len(full_hand) == 1 and full_hand[0] is not None:
            duplicate = copy_playing_card(full_hand[0])
            add_playing_cards(state, [duplicate], area="hand")
        elif name == "Obelisk" and isinstance(joker.extra, (int, float)):
            play_more_than = int(state.hands[scoring_name]["played"] or 0)
            reset = True
            for hand_name, hand in state.hands.items():
                if hand_name != scoring_name and hand["visible"] and int(hand["played"]) > play_more_than:
                    reset = False
                    break
            if reset:
                joker.x_mult = 1
            else:
                joker.x_mult += float(joker.extra)
        return None

    if phase == "repetition_play" and other_card is not None:
        if name == "Sock and Buskin" and _is_face(state, other_card):
            return {"repetitions": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Hanging Chad" and scoring_hand and other_card is scoring_hand[0]:
            return {"repetitions": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Dusk" and state.current_round.hands_left == 0:
            return {"repetitions": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Seltzer":
            return {"repetitions": 1.0}
        if name == "Hack" and _card_id(state, other_card) in {2, 3, 4, 5}:
            return {"repetitions": float(joker.extra if isinstance(joker.extra, int) else 0)}
        return None

    if phase == "repetition_hand" and card_effects is not None:
        # Joker-provided held abilities (e.g. Baron) also retrigger, even when
        # the card itself has no enhancement effect.
        if name == "Mime":
            return {"repetitions": float(joker.extra if isinstance(joker.extra, int) else 0)}
        return None

    if phase == "individual_play" and other_card is not None:
        if name == "Hiker" and isinstance(joker.extra, int):
            other_card.perma_bonus += joker.extra
        if name == "Photograph":
            first_face = next((card for card in scoring_hand if _is_face(state, card)), None)
            if first_face is other_card and isinstance(joker.extra, (int, float)):
                return {"x_mult": float(joker.extra)}
        if name == "8 Ball" and _card_id(state, other_card) == 8 and isinstance(joker.extra, (int, float)):
            if float(state.pseudorandom.pseudorandom("8ball")) < state.probabilities["normal"] / float(joker.extra):
                add_generated_consumable(state, "Tarot", append="8ba")
        if name == "Lucky Cat" and lucky_triggered and isinstance(joker.extra, (int, float)):
            joker.x_mult += float(joker.extra)
        if name == "The Idol":
            # idol_card (rank id + suit) is re-rolled each round by
            # flow._reset_round_cards; playing that exact card grants the X mult.
            idol = state.current_round.idol_card
            if _card_id(state, other_card) == idol.get("id") and _is_suit(state, other_card, idol.get("suit", "")):
                return {"x_mult": float(joker.extra if isinstance(joker.extra, (int, float)) else 1)}
        if name == "Scary Face" and _is_face(state, other_card):
            return {"chips": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Smiley Face" and _is_face(state, other_card):
            return {"mult": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Golden Ticket" and _card_name(state, other_card) == "Gold Card":
            amount = int(joker.extra if isinstance(joker.extra, int) else 0)
            return {"dollars": float(amount)}
        if name == "Scholar" and _card_id(state, other_card) == 14 and isinstance(joker.extra, dict):
            return {
                "chips": float(joker.extra.get("chips", 0) or 0),
                "mult": float(joker.extra.get("mult", 0) or 0),
            }
        if name == "Walkie Talkie" and _card_id(state, other_card) in {4, 10} and isinstance(joker.extra, dict):
            return {
                "chips": float(joker.extra.get("chips", 0) or 0),
                "mult": float(joker.extra.get("mult", 0) or 0),
            }
        if name == "Business Card" and _is_face(state, other_card):
            if float(state.pseudorandom.pseudorandom("business")) < state.probabilities["normal"] / float(joker.extra or 1):
                return {"dollars": 2.0}
        if name == "Fibonacci" and _card_id(state, other_card) in {2, 3, 5, 8, 14}:
            return {"mult": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Even Steven" and 0 <= _card_id(state, other_card) <= 10 and _card_id(state, other_card) % 2 == 0:
            return {"mult": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Odd Todd":
            card_id = _card_id(state, other_card)
            if (0 <= card_id <= 10 and card_id % 2 == 1) or card_id == 14:
                return {"chips": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if center.get("effect") == "Suit Mult" and isinstance(joker.extra, dict):
            if _is_suit(state, other_card, str(joker.extra.get("suit", ""))):
                return {"mult": float(joker.extra.get("s_mult", 0) or 0)}
        if name == "Rough Gem" and _is_suit(state, other_card, "Diamonds"):
            return {"dollars": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Onyx Agate" and _is_suit(state, other_card, "Clubs"):
            return {"mult": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Arrowhead" and _is_suit(state, other_card, "Spades"):
            return {"chips": float(joker.extra if isinstance(joker.extra, int) else 0)}
        if name == "Bloodstone" and isinstance(joker.extra, dict):
            if _is_suit(state, other_card, "Hearts") and float(state.pseudorandom.pseudorandom("bloodstone")) < (
                state.probabilities["normal"] / float(joker.extra.get("odds", 1) or 1)
            ):
                return {"x_mult": float(joker.extra.get("Xmult", 1) or 1)}
        if name == "Ancient Joker" and _is_suit(state, other_card, state.current_round.ancient_card["suit"]):
            return {"x_mult": float(joker.extra if isinstance(joker.extra, (int, float)) else 1)}
        if name == "Triboulet" and _card_id(state, other_card) in {12, 13}:
            return {"x_mult": float(joker.extra if isinstance(joker.extra, (int, float)) else 1)}
        if name == "Wee Joker" and _card_id(state, other_card) == 2 and isinstance(joker.extra, dict):
            joker.extra["chips"] = int(joker.extra.get("chips", 0) or 0) + int(joker.extra.get("chip_mod", 0) or 0)
        return None

    if phase == "individual_hand" and other_card is not None:
        if name == "Shoot the Moon" and _card_id(state, other_card) == 12 and not other_card.debuff:
            return {"h_mult": 13.0}
        if name == "Baron" and _card_id(state, other_card) == 13 and not other_card.debuff:
            return {"x_mult": float(joker.extra if isinstance(joker.extra, (int, float)) else 1)}
        if name == "Reserved Parking" and _is_face(state, other_card) and not other_card.debuff and isinstance(joker.extra, dict):
            if float(state.pseudorandom.pseudorandom("parking")) < (
                state.probabilities["normal"] / float(joker.extra.get("odds", 1) or 1)
            ):
                return {"dollars": float(joker.extra.get("dollars", 0) or 0)}
        if name == "Raised Fist":
            valid_cards = [card for card in held_hand if _card_effect(state, card) != "Stone Card"]
            if not valid_cards:
                return None
            # Balatro scans left to right with >=, selecting the last tied
            # lowest rank. Object addresses cannot reproduce that decision.
            raised_card = min(reversed(valid_cards), key=lambda card: _card_id(state, card))
            if raised_card is other_card and not other_card.debuff:
                return {"h_mult": float(2 * RANK_TO_NOMINAL[other_card.rank])}
        return None

    if phase == "other_joker" and other_joker is not None:
        if name == "Baseball Card" and other_joker is not joker and _joker_center(state, other_joker).get("rarity") == 2:
            return {"x_mult": float(joker.extra if isinstance(joker.extra, (int, float)) else 1)}
        return None

    if phase != "main":
        return None

    if name == "Loyalty Card" and isinstance(joker.extra, dict):
        every = int(joker.extra.get("every", 0) or 0)
        if every > 0:
            joker.loyalty_remaining = (every - 1 - (state.hands_played - joker.hands_played_at_create)) % (every + 1)
            if joker.loyalty_remaining == every:
                return {"x_mult": float(joker.extra.get("Xmult", 1) or 1)}
    if name != "Seeing Double" and joker.x_mult > 1 and (not joker.type or _type_matches(poker_hands, joker.type)):
        return {"x_mult": joker.x_mult}
    if joker.t_mult > 0 and _type_matches(poker_hands, joker.type):
        return {"mult": float(joker.t_mult)}
    if joker.t_chips > 0 and _type_matches(poker_hands, joker.type):
        return {"chips": float(joker.t_chips)}
    if name == "Half Joker" and isinstance(joker.extra, dict) and len(full_hand) <= int(joker.extra.get("size", 0) or 0):
        return {"mult": float(joker.extra.get("mult", 0) or 0)}
    if name == "Abstract Joker" and isinstance(joker.extra, int):
        return {"mult": float(sum(1 for _ in state.jokers) * joker.extra)}
    if name == "Acrobat" and state.current_round.hands_left == 0 and isinstance(joker.extra, (int, float)):
        return {"x_mult": float(joker.extra)}
    if name == "Mystic Summit" and isinstance(joker.extra, dict):
        if state.current_round.discards_left == int(joker.extra.get("d_remaining", 0) or 0):
            return {"mult": float(joker.extra.get("mult", 0) or 0)}
    if name == "Misprint" and isinstance(joker.extra, dict):
        amount = state.pseudorandom.pseudorandom("misprint", int(joker.extra.get("min", 0)), int(joker.extra.get("max", 0)))
        return {"mult": float(amount)}
    if name == "Banner" and state.current_round.discards_left > 0 and isinstance(joker.extra, int):
        return {"chips": float(state.current_round.discards_left * joker.extra)}
    if name == "Stuntman" and isinstance(joker.extra, dict):
        return {"chips": float(joker.extra.get("chip_mod", 0) or 0)}
    if name == "Supernova":
        return {"mult": float(state.hands[scoring_name]["played"])}
    if name == "Ceremonial Dagger" and joker.mult > 0:
        return {"mult": float(joker.mult)}
    if name == "Vagabond" and isinstance(joker.extra, int) and state.dollars <= joker.extra:
        add_generated_consumable(state, "Tarot", append="vag")
    if name == "Superposition":
        aces = sum(1 for card in scoring_hand if _card_id(state, card) == 14)
        if aces >= 1 and poker_hands["Straight"]:
            add_generated_consumable(state, "Tarot", append="sup")
    if name == "Séance" and isinstance(joker.extra, dict):
        if poker_hands[str(joker.extra.get("poker_hand", ""))]:
            add_generated_consumable(state, "Spectral", append="sea")
    if name == "Flower Pot" and isinstance(joker.extra, (int, float)):
        suits = {"Hearts": 0, "Diamonds": 0, "Spades": 0, "Clubs": 0}
        for card in scoring_hand:
            if _card_name(state, card) == "Wild Card":
                continue
            for suit in suits:
                if suits[suit] == 0 and _is_suit(state, card, suit, bypass_debuff=True):
                    suits[suit] = 1
                    break
        for card in scoring_hand:
            if _card_name(state, card) != "Wild Card":
                continue
            for suit in ("Hearts", "Diamonds", "Spades", "Clubs"):
                if suits[suit] == 0 and _is_suit(state, card, suit):
                    suits[suit] = 1
                    break
        if all(value > 0 for value in suits.values()):
            return {"x_mult": float(joker.extra)}
    if name == "Seeing Double" and isinstance(joker.extra, (int, float)):
        suits = {"Hearts": 0, "Diamonds": 0, "Spades": 0, "Clubs": 0}
        for card in scoring_hand:
            if _card_name(state, card) == "Wild Card":
                continue
            for suit in suits:
                if _is_suit(state, card, suit):
                    suits[suit] += 1
        for card in scoring_hand:
            if _card_name(state, card) != "Wild Card":
                continue
            for suit in ("Clubs", "Diamonds", "Spades", "Hearts"):
                if suits[suit] == 0 and _is_suit(state, card, suit):
                    suits[suit] += 1
                    break
        if (suits["Hearts"] > 0 or suits["Diamonds"] > 0 or suits["Spades"] > 0) and suits["Clubs"] > 0:
            return {"x_mult": float(joker.extra)}
    if name == "Wee Joker" and isinstance(joker.extra, dict):
        return {"chips": float(joker.extra.get("chips", 0) or 0)}
    if name == "Castle" and isinstance(joker.extra, dict):
        return {"chips": float(joker.extra.get("chips", 0) or 0)}
    if name == "Blue Joker" and isinstance(joker.extra, int):
        remaining = _remaining_deck_count(state)
        if remaining > 0:
            return {"chips": float(joker.extra * remaining)}
    if name == "Erosion" and isinstance(joker.extra, int):
        removed = state.starting_deck_size - len(state.deck_cards)
        if removed > 0:
            return {"mult": float(joker.extra * removed)}
    if name == "Square Joker" and isinstance(joker.extra, dict):
        return {"chips": float(joker.extra.get("chips", 0) or 0)}
    if name == "Runner" and isinstance(joker.extra, dict):
        return {"chips": float(joker.extra.get("chips", 0) or 0)}
    if name == "Ice Cream" and isinstance(joker.extra, dict):
        return {"chips": float(joker.extra.get("chips", 0) or 0)}
    if name == "Stone Joker" and joker.stone_tally > 0 and isinstance(joker.extra, (int, float)):
        return {"chips": float(joker.extra * joker.stone_tally)}
    if name == "Steel Joker" and joker.steel_tally > 0 and isinstance(joker.extra, (int, float)):
        return {"x_mult": float(1 + joker.extra * joker.steel_tally)}
    if name == "Bull" and isinstance(joker.extra, (int, float)) and (state.dollars + state.dollar_buffer) > 0:
        return {"chips": float(joker.extra * max(0, state.dollars + state.dollar_buffer))}
    if name == "Driver's License" and joker.driver_tally >= 16 and isinstance(joker.extra, (int, float)):
        return {"x_mult": float(joker.extra)}
    if name == "Blackboard" and held_hand and isinstance(joker.extra, (int, float)):
        black_suits = sum(
            1 for card in held_hand if _is_suit(state, card, "Clubs", flush_calc=True) or _is_suit(state, card, "Spades", flush_calc=True)
        )
        if black_suits == len(held_hand):
            return {"x_mult": float(joker.extra)}
    if name == "Joker Stencil" and (state.starting_params.joker_slots - len(state.jokers)) > 0:
        return {"x_mult": float(joker.x_mult)}
    if name == "Swashbuckler" and joker.mult > 0:
        return {"mult": float(joker.mult)}
    if name == "Joker":
        return {"mult": float(joker.mult)}
    if name in {"Spare Trousers", "Ride the Bus", "Flash Card", "Popcorn", "Green Joker", "Red Card"} and joker.mult > 0:
        return {"mult": float(joker.mult)}
    if name == "Fortune Teller" and state.consumeable_usage_total["tarot"] > 0:
        return {"mult": float(state.consumeable_usage_total["tarot"])}
    if name == "Gros Michel" and isinstance(joker.extra, dict):
        return {"mult": float(joker.extra.get("mult", 0) or 0)}
    if name == "Cavendish" and isinstance(joker.extra, dict):
        return {"x_mult": float(joker.extra.get("Xmult", 1) or 1)}
    if name == "Card Sharp" and state.hands[scoring_name]["played_this_round"] > 1 and isinstance(joker.extra, dict):
        return {"x_mult": float(joker.extra.get("Xmult", 1) or 1)}
    if name == "Bootstraps" and isinstance(joker.extra, dict):
        dollars = int(joker.extra.get("dollars", 1) or 1)
        mult = int(joker.extra.get("mult", 0) or 0)
        if dollars > 0:
            factor = floor((state.dollars + state.dollar_buffer) / dollars)
            if factor >= 1:
                return {"mult": float(mult * factor)}
    if name == "Canio" and joker.caino_xmult > 1:
        return {"x_mult": float(joker.caino_xmult)}
    if name == "Matador" and state.blind_triggered and isinstance(joker.extra, int):
        return {"dollars": float(joker.extra)}
    return None


@cython.locals(index=cython.Py_ssize_t, repetitions=cython.long, card_mult=cython.double, x_mult=cython.double)
def score_hand(
    state: RunState,
    full_hand: list[PlayingCard],
    *,
    held_hand: list[PlayingCard] | None = None,
    hand_debuffed: bool = False,
    precomputed_hand_name: str | None = None,
    precomputed_poker_hands: dict | None = None,
) -> ScoreResult:
    held_cards = list(held_hand or [])
    state.dollar_buffer = 0
    sync_all_jokers(state)

    if precomputed_hand_name is not None and precomputed_poker_hands is not None:
        scoring_name = precomputed_hand_name
        poker_hands = precomputed_poker_hands
        display_name = scoring_name
        scoring_hand = []
        for ht in POKER_HAND_ORDER:
            if poker_hands.get(ht) and any(poker_hands[ht]):
                scoring_name = ht
                display_name = ht
                scoring_hand = poker_hands[ht][0]
                break
    else:
        scoring_name, display_name, poker_hands, scoring_hand = get_poker_hand_info(state, full_hand)
    state.hands[scoring_name]["played"] += 1
    state.hands[scoring_name]["played_this_round"] += 1
    state.hands[scoring_name]["visible"] = True
    state.last_hand_played = scoring_name

    if hand_debuffed:
        return ScoreResult(
            hand_name=scoring_name,
            display_name=display_name,
            poker_hands=poker_hands,
            scoring_cards=list(full_hand),
            held_cards=held_cards,
            chips=0,
            mult=0,
            total=0,
            dollars=0,
        )

    if state.has_joker("Splash"):
        scoring_cards = list(full_hand)
    else:
        # Hand detection groups ranks; scoring must preserve played order.
        scoring_ids = {id(card) for card in scoring_hand}
        scoring_cards = [
            card for card in full_hand
            if id(card) in scoring_ids or _card_effect(state, card) == "Stone Card"
        ]

    state.current_round.free_rerolls = sum(
        1 for owned in state.jokers if _joker_center(state, owned)["name"] == "Chaos the Clown"
    )

    for index, joker in enumerate(state.jokers):
        _evaluate_joker(
            state,
            joker,
            index=index,
            phase="before",
            full_hand=full_hand,
            scoring_hand=scoring_cards,
            held_hand=held_cards,
            scoring_name=scoring_name,
            poker_hands=poker_hands,
        )
    sync_all_jokers(state)

    hand_chips = _mod_chips(state, float(state.hands[scoring_name]["chips"]))
    mult = _mod_mult(float(state.hands[scoring_name]["mult"]))

    # Apply modify_hand (The Flint)
    mult, hand_chips = _modify_hand(state, full_hand, poker_hands, mult, hand_chips)

    for card in scoring_cards:
        if _card_effect(state, card) != "Stone Card":
            rank_name = {
                "A": "Ace",
                "2": "2",
                "3": "3",
                "4": "4",
                "5": "5",
                "6": "6",
                "7": "7",
                "8": "8",
                "9": "9",
                "T": "10",
                "J": "Jack",
                "Q": "Queen",
                "K": "King",
            }[card.rank]
            state.cards_played[rank_name]["total"] += 1
            state.cards_played[rank_name]["suits"][card.suit] = True

        if card.debuff:
            state.blind_triggered = True
            continue

        repetitions = 1 + (1 if card.seal == "Red" else 0)
        for index, joker in enumerate(state.jokers):
            rep = _evaluate_joker(
                state,
                joker,
                index=index,
                phase="repetition_play",
                full_hand=full_hand,
                scoring_hand=scoring_cards,
                held_hand=held_cards,
                scoring_name=scoring_name,
                poker_hands=poker_hands,
                other_card=card,
            )
            if rep:
                repetitions += int(rep["repetitions"])

        for _ in range(repetitions):
            lucky_triggered = False
            hand_chips = _mod_chips(state, hand_chips + _chip_bonus(state, card))
            card_mult = _chip_mult(state, card)
            mult = _mod_mult(mult + card_mult)
            if _card_effect(state, card) == "Lucky Card" and card_mult > 0:
                lucky_triggered = True
            x_mult = _chip_x_mult(state, card)
            if x_mult:
                mult = _mod_mult(mult * x_mult)
            dollars = _p_dollars(state, card)
            if dollars:
                _add_money(state, dollars)
                if _card_effect(state, card) == "Lucky Card":
                    lucky_triggered = True

            hand_chips, mult = _flush_main_editions(hand_chips, mult, _edition_dict_from_key(card.edition_key))

            for index, joker in enumerate(state.jokers):
                effect = _evaluate_joker(
                    state,
                    joker,
                    index=index,
                    phase="individual_play",
                    full_hand=full_hand,
                    scoring_hand=scoring_cards,
                    held_hand=held_cards,
                    scoring_name=scoring_name,
                    poker_hands=poker_hands,
                    other_card=card,
                    lucky_triggered=lucky_triggered,
                )
                if not effect:
                    continue
                if "chips" in effect:
                    hand_chips = _mod_chips(state, hand_chips + effect["chips"])
                if "mult" in effect:
                    mult = _mod_mult(mult + effect["mult"])
                if "x_mult" in effect:
                    mult = _mod_mult(mult * effect["x_mult"])
                if "dollars" in effect:
                    _add_money(state, int(effect["dollars"]))

    for card in held_cards:
        if card.debuff:
            continue
        base_effects: list[dict[str, float]] = []
        if h_mult := _chip_h_mult(state, card):
            base_effects.append({"h_mult": h_mult})
        if x_mult := _chip_h_x_mult(state, card):
            base_effects.append({"x_mult": x_mult})

        repetitions = 1 + (1 if card.seal == "Red" else 0)
        for index, joker in enumerate(state.jokers):
            rep = _evaluate_joker(
                state,
                joker,
                index=index,
                phase="repetition_hand",
                full_hand=full_hand,
                scoring_hand=scoring_cards,
                held_hand=held_cards,
                scoring_name=scoring_name,
                poker_hands=poker_hands,
                other_card=card,
                card_effects=base_effects,
            )
            if rep:
                repetitions += int(rep["repetitions"])

        for _ in range(repetitions):
            effects = list(base_effects)
            for index, joker in enumerate(state.jokers):
                effect = _evaluate_joker(
                    state,
                    joker,
                    index=index,
                    phase="individual_hand",
                    full_hand=full_hand,
                    scoring_hand=scoring_cards,
                    held_hand=held_cards,
                    scoring_name=scoring_name,
                    poker_hands=poker_hands,
                    other_card=card,
                )
                if effect:
                    effects.append(effect)
            for effect in effects:
                if "h_mult" in effect:
                    mult = _mod_mult(mult + effect["h_mult"])
                if "x_mult" in effect:
                    mult = _mod_mult(mult * effect["x_mult"])
                if "dollars" in effect:
                    _add_money(state, int(effect["dollars"]))

    for index, joker in enumerate(state.jokers):
        if not joker.debuff:
            hand_chips += _edition_chip_mod(joker.edition)
            mult += _edition_mult_mod(joker.edition)
        effect = _evaluate_joker(
            state,
            joker,
            index=index,
            phase="main",
            full_hand=full_hand,
            scoring_hand=scoring_cards,
            held_hand=held_cards,
            scoring_name=scoring_name,
            poker_hands=poker_hands,
        )
        if effect:
            if "chips" in effect:
                hand_chips = _mod_chips(state, hand_chips + effect["chips"])
            if "mult" in effect:
                mult = _mod_mult(mult + effect["mult"])
            if "x_mult" in effect:
                mult = _mod_mult(mult * effect["x_mult"])
            if "dollars" in effect:
                _add_money(state, int(effect["dollars"]))
        # Every joker gets an "other_joker" pass after it is scored, whether or
        # not it produced an effect itself (Baseball Card triggers on uncommon
        # jokers even when they contribute nothing this hand).
        for other_index, other in enumerate(state.jokers):
            on_joker = _evaluate_joker(
                state,
                other,
                index=other_index,
                phase="other_joker",
                full_hand=full_hand,
                scoring_hand=scoring_cards,
                held_hand=held_cards,
                scoring_name=scoring_name,
                poker_hands=poker_hands,
                other_joker=joker,
            )
            if on_joker:
                if "chips" in on_joker:
                    hand_chips = _mod_chips(state, hand_chips + on_joker["chips"])
                if "mult" in on_joker:
                    mult = _mod_mult(mult + on_joker["mult"])
                if "x_mult" in on_joker:
                    mult = _mod_mult(mult * on_joker["x_mult"])

        if not joker.debuff:
            mult *= _edition_x_mult_mod(joker.edition)

    for consumable in state.consumables:
        effect = _evaluate_planet_consumable(state, consumable, scoring_name=scoring_name)
        if effect and "x_mult" in effect:
            mult = _mod_mult(mult * effect["x_mult"])

    if state.deck_key == "b_plasma":
        total = hand_chips + mult
        hand_chips = floor(total / 2)
        mult = floor(total / 2)

    total_score = floor(hand_chips * mult)
    dollars = state.dollar_buffer
    state.dollar_buffer = 0
    return ScoreResult(
        hand_name=scoring_name,
        display_name=display_name,
        poker_hands=poker_hands,
        scoring_cards=scoring_cards,
        held_cards=held_cards,
        chips=hand_chips,
        mult=mult,
        total=total_score,
        dollars=dollars,
    )


def resolve_after_hand(state: RunState) -> None:
    to_remove: list[JokerInstance] = []
    for joker in list(state.jokers):
        name = _joker_center(state, joker)["name"]
        if name == "Ice Cream" and isinstance(joker.extra, dict):
            chips = int(joker.extra.get("chips", 0) or 0)
            mod = int(joker.extra.get("chip_mod", 0) or 0)
            if chips - mod <= 0:
                to_remove.append(joker)
            else:
                joker.extra["chips"] = chips - mod
        elif name == "Seltzer":
            if isinstance(joker.extra, int):
                if joker.extra - 1 <= 0:
                    to_remove.append(joker)
                else:
                    joker.extra -= 1
    for joker in to_remove:
        remove_joker(state, joker)
    sync_all_jokers(state)


def _evaluate_planet_consumable(
    state: RunState,
    consumable: ConsumableInstance,
    *,
    scoring_name: str,
) -> dict[str, float] | None:
    center = state.data.centers[consumable.center_key]
    if center.get("set") != "Planet":
        return None
    if state.used_vouchers.get("v_observatory") and _as_dict(center.get("config")).get("hand_type") == scoring_name:
        return {"x_mult": float(state.data.centers["v_observatory"]["config"]["extra"])}
    return None


def _modify_hand(
    state: RunState,
    cards: list[PlayingCard],
    poker_hands: dict,
    mult: float,
    hand_chips: float,
) -> tuple[float, float]:
    """Apply blind modify_hand effects (e.g. The Flint halving)."""
    if state.blind_disabled:
        return mult, hand_chips
    blind = state.round_resets.blind or {}
    blind_name = str(blind.get("name", ""))
    if blind_name == "The Flint":
        state.blind_triggered = True
        mult = max(floor(mult * 0.5 + 0.5), 1)
        hand_chips = max(floor(hand_chips * 0.5 + 0.5), 0)
    return mult, hand_chips


def _level_up_hand(state: RunState, hand_name: str, amount: int = 1) -> None:
    hand = state.hands[hand_name]
    hand["level"] = max(0, int(hand["level"]) + amount)
    hand["mult"] = max(int(hand["s_mult"]) + int(hand["l_mult"]) * (int(hand["level"]) - 1), 1)
    hand["chips"] = max(int(hand["s_chips"]) + int(hand["l_chips"]) * (int(hand["level"]) - 1), 0)
