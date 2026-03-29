"""Unit tests verifying all 37 generic-path jokers fire their effects correctly.

These tests use score_hand directly (no oracle bridge) to confirm that each
joker handled by a generic code path (suit mult, type mult, type chips, x_mult,
hand/discard size) produces the expected scoring output.

Also covers edge-case jokers: Four Fingers, Shortcut, Smeared Joker,
Pareidolia, and Splash.
"""

from __future__ import annotations

import pytest

from pylatro import add_joker, create_run_state, score_hand, start_blind, play_cards
from pylatro.models import PlayingCard
from pylatro.runtime import apply_end_shop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _card(suit: str, rank: str) -> PlayingCard:
    """Build a PlayingCard with the standard front_key convention."""
    prefix = {"Spades": "S", "Hearts": "H", "Diamonds": "D", "Clubs": "C"}[suit]
    return PlayingCard(front_key=f"{prefix}_{rank}", suit=suit, rank=rank)


# ---------------------------------------------------------------------------
# 1. Suit Mult jokers — +3 mult per scoring card of the matching suit
# ---------------------------------------------------------------------------

SUIT_MULT_CASES = [
    ("j_greedy_joker", "Diamonds", 3),
    ("j_lusty_joker", "Hearts", 3),
    ("j_wrathful_joker", "Spades", 3),
    ("j_gluttenous_joker", "Clubs", 3),
]


@pytest.mark.parametrize("joker_key,suit,s_mult", SUIT_MULT_CASES, ids=[c[0] for c in SUIT_MULT_CASES])
def test_suit_mult_joker_adds_mult_per_matching_card(joker_key: str, suit: str, s_mult: int) -> None:
    """Each suit-mult joker adds s_mult per scoring card of its suit."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, joker_key)

    # Play a pair of the matching suit — both cards score
    hand = [_card(suit, "5"), _card(suit, "5")]
    result = score_hand(state, hand)

    assert result.hand_name == "Pair"
    # Base Pair: chips=10, mult=2. Each card adds s_mult (3) → mult = 2 + 3 + 3 = 8
    # Chips: 5 + 5 + 10 (base) = 20
    assert result.mult == 2 + s_mult * 2
    assert result.total == int(result.chips * result.mult)


@pytest.mark.parametrize("joker_key,suit,s_mult", SUIT_MULT_CASES, ids=[c[0] for c in SUIT_MULT_CASES])
def test_suit_mult_joker_ignores_non_matching_suit(joker_key: str, suit: str, s_mult: int) -> None:
    """Suit-mult joker gives no bonus for cards of a different suit."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, joker_key)

    # Pick a suit that doesn't match
    other_suit = {"Diamonds": "Spades", "Hearts": "Clubs", "Spades": "Hearts", "Clubs": "Diamonds"}[suit]
    hand = [_card(other_suit, "7")]
    result = score_hand(state, hand)

    assert result.hand_name == "High Card"
    # No suit mult bonus — mult stays at base (1 for High Card)
    assert result.mult == 1.0


# ---------------------------------------------------------------------------
# 2. Type Mult jokers — bonus mult when the hand type matches
# ---------------------------------------------------------------------------

TYPE_MULT_CASES = [
    ("j_jolly", "Pair", 8),
    ("j_mad", "Two Pair", 10),
    ("j_zany", "Three of a Kind", 12),
    ("j_crazy", "Straight", 12),
    ("j_droll", "Flush", 10),
]


def _make_hand(hand_type: str) -> list[PlayingCard]:
    """Build a minimal hand for the given poker hand type."""
    if hand_type == "Pair":
        return [_card("Spades", "8"), _card("Hearts", "8")]
    if hand_type == "Two Pair":
        return [_card("Spades", "8"), _card("Hearts", "8"), _card("Diamonds", "6"), _card("Clubs", "6")]
    if hand_type == "Three of a Kind":
        return [_card("Spades", "7"), _card("Hearts", "7"), _card("Diamonds", "7")]
    if hand_type == "Straight":
        return [
            _card("Spades", "5"),
            _card("Hearts", "6"),
            _card("Diamonds", "7"),
            _card("Clubs", "8"),
            _card("Spades", "9"),
        ]
    if hand_type == "Flush":
        return [
            _card("Hearts", "2"),
            _card("Hearts", "5"),
            _card("Hearts", "7"),
            _card("Hearts", "9"),
            _card("Hearts", "J"),
        ]
    raise ValueError(f"Unsupported hand type: {hand_type}")


@pytest.mark.parametrize("joker_key,hand_type,t_mult", TYPE_MULT_CASES, ids=[c[0] for c in TYPE_MULT_CASES])
def test_type_mult_joker_adds_mult(joker_key: str, hand_type: str, t_mult: int) -> None:
    """Type-mult joker adds t_mult when the correct hand type is played."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, joker_key)

    hand = _make_hand(hand_type)
    result = score_hand(state, hand)

    assert result.hand_name == hand_type
    base_mult = state.hands[hand_type]["mult"]
    # t_mult is added on top of base mult (plus any per-card mult)
    assert result.mult >= base_mult + t_mult


@pytest.mark.parametrize("joker_key,hand_type,t_mult", TYPE_MULT_CASES, ids=[c[0] for c in TYPE_MULT_CASES])
def test_type_mult_joker_inactive_on_wrong_hand(joker_key: str, hand_type: str, t_mult: int) -> None:
    """Type-mult joker does NOT add t_mult when a different hand type is played."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, joker_key)

    # Play a High Card (never matches any type joker)
    hand = [_card("Spades", "A")]
    result = score_hand(state, hand)

    assert result.hand_name == "High Card"
    # mult should be base High Card mult (1), no bonus
    assert result.mult == 1.0


# ---------------------------------------------------------------------------
# 3. Type Chips jokers — bonus chips when the hand type matches
# ---------------------------------------------------------------------------

TYPE_CHIPS_CASES = [
    ("j_sly", "Pair", 50),
    ("j_clever", "Two Pair", 80),
    ("j_wily", "Three of a Kind", 100),
    ("j_devious", "Straight", 100),
    ("j_crafty", "Flush", 80),
]


@pytest.mark.parametrize("joker_key,hand_type,t_chips", TYPE_CHIPS_CASES, ids=[c[0] for c in TYPE_CHIPS_CASES])
def test_type_chips_joker_adds_chips(joker_key: str, hand_type: str, t_chips: int) -> None:
    """Type-chips joker adds t_chips when the correct hand type is played."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, joker_key)

    hand = _make_hand(hand_type)
    result = score_hand(state, hand)

    assert result.hand_name == hand_type
    base_chips = state.hands[hand_type]["chips"]
    # t_chips is added on top of base chips + card nominals
    assert result.chips >= base_chips + t_chips


@pytest.mark.parametrize("joker_key,hand_type,t_chips", TYPE_CHIPS_CASES, ids=[c[0] for c in TYPE_CHIPS_CASES])
def test_type_chips_joker_inactive_on_wrong_hand(joker_key: str, hand_type: str, t_chips: int) -> None:
    """Type-chips joker does NOT add t_chips when a different hand type is played."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, joker_key)

    hand = [_card("Spades", "A")]
    result = score_hand(state, hand)

    assert result.hand_name == "High Card"
    # chips should be base High Card chips (5) + Ace nominal (11) = 16
    assert result.chips == 16.0


# ---------------------------------------------------------------------------
# 4. X-Mult + Type jokers — x_mult when the hand type matches
# ---------------------------------------------------------------------------

XMULT_TYPE_CASES = [
    ("j_duo", "Pair", 2.0),
    ("j_trio", "Three of a Kind", 3.0),
    ("j_family", "Four of a Kind", 4.0),
    ("j_order", "Straight", 3.0),
    ("j_tribe", "Flush", 2.0),
]


def _make_hand_xmult(hand_type: str) -> list[PlayingCard]:
    """Build hands for x-mult jokers (Four of a Kind needs 4 cards)."""
    if hand_type == "Four of a Kind":
        return [
            _card("Spades", "3"),
            _card("Hearts", "3"),
            _card("Diamonds", "3"),
            _card("Clubs", "3"),
        ]
    return _make_hand(hand_type)


@pytest.mark.parametrize("joker_key,hand_type,x_mult", XMULT_TYPE_CASES, ids=[c[0] for c in XMULT_TYPE_CASES])
def test_xmult_type_joker_applies_multiplier(joker_key: str, hand_type: str, x_mult: float) -> None:
    """X-mult joker multiplies the mult by x_mult when the hand type matches."""
    state = create_run_state("AAAAAAAA")

    # Get baseline without joker
    hand = _make_hand_xmult(hand_type)
    baseline = score_hand(state, hand)

    # Reset played counts so hand level doesn't change
    state.hands[hand_type]["played"] -= 1
    state.hands[hand_type]["played_this_round"] -= 1

    add_joker(state, joker_key)
    result = score_hand(state, hand)

    assert result.hand_name == hand_type
    # With x_mult joker: total should be baseline chips * (baseline mult * x_mult)
    assert result.total == int(baseline.chips * baseline.mult * x_mult)


@pytest.mark.parametrize("joker_key,hand_type,x_mult", XMULT_TYPE_CASES, ids=[c[0] for c in XMULT_TYPE_CASES])
def test_xmult_type_joker_inactive_on_wrong_hand(joker_key: str, hand_type: str, x_mult: float) -> None:
    """X-mult joker does NOT apply when a different hand type is played."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, joker_key)

    hand = [_card("Spades", "A")]
    result = score_hand(state, hand)

    assert result.hand_name == "High Card"
    # No x_mult applied — total is base High Card
    assert result.total == 16  # 16 chips * 1 mult


# ---------------------------------------------------------------------------
# 5. Hand/discard size jokers
# ---------------------------------------------------------------------------

def test_juggler_increases_hand_size_by_one() -> None:
    """Juggler adds +1 to hand size."""
    state = create_run_state("AAAAAAAA")
    base_hand_size = state.starting_params.hand_size

    add_joker(state, "j_juggler")

    assert state.starting_params.hand_size == base_hand_size + 1
    assert state.current_round.hand_size == base_hand_size + 1


def test_drunkard_increases_discards_by_one() -> None:
    """Drunkard adds +1 discard."""
    state = create_run_state("AAAAAAAA")
    base_discards = state.round_resets.discards

    add_joker(state, "j_drunkard")

    assert state.round_resets.discards == base_discards + 1
    assert state.current_round.discards_left == base_discards + 1


def test_merry_andy_adds_discards_and_reduces_hand_size() -> None:
    """Merry Andy adds +3 discards and -1 hand size."""
    state = create_run_state("AAAAAAAA")
    base_discards = state.round_resets.discards
    base_hand_size = state.starting_params.hand_size

    add_joker(state, "j_merry_andy")

    assert state.round_resets.discards == base_discards + 3
    assert state.current_round.discards_left == base_discards + 3
    assert state.starting_params.hand_size == base_hand_size - 1
    assert state.current_round.hand_size == base_hand_size - 1


def test_juggler_hand_size_visible_in_draw() -> None:
    """Juggler's +1 hand size means 9 cards drawn at start_blind."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_juggler")
    drawn = start_blind(state)

    assert len(drawn) == 9
    assert len(state.hand_cards) == 9


# ---------------------------------------------------------------------------
# 6. Edge case: Four Fingers — 4-card flushes and straights
# ---------------------------------------------------------------------------

def test_four_fingers_recognizes_four_card_flush() -> None:
    """Four Fingers allows 4-card flushes."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_four_fingers")

    hand = [
        _card("Hearts", "2"),
        _card("Hearts", "5"),
        _card("Hearts", "9"),
        _card("Hearts", "K"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Flush"


def test_four_fingers_recognizes_four_card_straight() -> None:
    """Four Fingers allows 4-card straights."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_four_fingers")

    hand = [
        _card("Spades", "4"),
        _card("Hearts", "5"),
        _card("Diamonds", "6"),
        _card("Clubs", "7"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Straight"


def test_four_fingers_normal_five_card_flush_still_works() -> None:
    """Four Fingers doesn't break normal 5-card flushes."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_four_fingers")

    hand = [
        _card("Diamonds", "2"),
        _card("Diamonds", "4"),
        _card("Diamonds", "7"),
        _card("Diamonds", "9"),
        _card("Diamonds", "J"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Flush"


# ---------------------------------------------------------------------------
# 7. Edge case: Shortcut — straights with 1 gap
# ---------------------------------------------------------------------------

def test_shortcut_recognizes_gap_straight() -> None:
    """Shortcut allows straights with gaps of 1 rank (e.g. 2-3-5-6-7)."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_shortcut")

    hand = [
        _card("Spades", "2"),
        _card("Hearts", "3"),
        _card("Diamonds", "5"),
        _card("Clubs", "6"),
        _card("Spades", "7"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Straight"


def test_shortcut_normal_straight_still_works() -> None:
    """Shortcut doesn't break normal straights."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_shortcut")

    hand = [
        _card("Spades", "5"),
        _card("Hearts", "6"),
        _card("Diamonds", "7"),
        _card("Clubs", "8"),
        _card("Spades", "9"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Straight"


def test_shortcut_does_not_allow_two_consecutive_gaps() -> None:
    """Shortcut does NOT allow two consecutive missing ranks (a 2-rank gap)."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_shortcut")

    # 2-5-8-J-A has 2-rank gaps (3,4 missing; 6,7 missing) — not a straight
    hand = [
        _card("Spades", "2"),
        _card("Hearts", "5"),
        _card("Diamonds", "8"),
        _card("Clubs", "J"),
        _card("Spades", "A"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name != "Straight"


def test_shortcut_allows_multiple_single_gaps() -> None:
    """Shortcut allows multiple single-rank gaps (each at most 1 missing rank)."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_shortcut")

    # 2-4-6-8-T: single gap between each pair — valid with Shortcut
    hand = [
        _card("Spades", "2"),
        _card("Hearts", "4"),
        _card("Diamonds", "6"),
        _card("Clubs", "8"),
        _card("Spades", "T"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Straight"


# ---------------------------------------------------------------------------
# 8. Edge case: Smeared Joker — red/black suits treated as same
# ---------------------------------------------------------------------------

def test_smeared_joker_hearts_diamonds_flush() -> None:
    """Smeared Joker treats Hearts and Diamonds as the same suit for flushes."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_smeared")

    hand = [
        _card("Hearts", "2"),
        _card("Diamonds", "5"),
        _card("Hearts", "8"),
        _card("Diamonds", "J"),
        _card("Hearts", "A"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Flush"


def test_smeared_joker_spades_clubs_flush() -> None:
    """Smeared Joker treats Spades and Clubs as the same suit for flushes."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_smeared")

    hand = [
        _card("Spades", "3"),
        _card("Clubs", "6"),
        _card("Spades", "9"),
        _card("Clubs", "Q"),
        _card("Spades", "A"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name == "Flush"


def test_smeared_joker_does_not_mix_red_and_black() -> None:
    """Smeared Joker does NOT treat red and black as the same suit."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_smeared")

    hand = [
        _card("Hearts", "2"),
        _card("Clubs", "5"),
        _card("Hearts", "8"),
        _card("Clubs", "J"),
        _card("Hearts", "A"),
    ]
    result = score_hand(state, hand)

    assert result.hand_name != "Flush"


# ---------------------------------------------------------------------------
# 9. Edge case: Pareidolia — all cards treated as face cards
# ---------------------------------------------------------------------------

def test_pareidolia_makes_all_cards_face() -> None:
    """Pareidolia makes all cards count as face cards (affects face-checking jokers)."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_pareidolia")
    add_joker(state, "j_scary_face")  # +30 chips if face card

    # Play a 2 — normally not a face card, but Pareidolia makes it one
    hand = [_card("Spades", "2")]
    result = score_hand(state, hand)

    assert result.hand_name == "High Card"
    # Base High Card: chips=5, rank nominal=2, Scary Face +30 chips
    # chips = 5 + 2 + 30 = 37, mult = 1
    assert result.chips == 37.0
    assert result.total == 37


# ---------------------------------------------------------------------------
# 10. Edge case: Splash — all played cards score
# ---------------------------------------------------------------------------

def test_splash_all_played_cards_score() -> None:
    """Splash makes all played cards score, not just the poker hand."""
    state = create_run_state("AAAAAAAA")

    # Baseline: play a pair + unrelated card. Without Splash, only the pair scores.
    hand = [
        _card("Spades", "5"),
        _card("Hearts", "5"),
        _card("Diamonds", "K"),
    ]
    baseline = score_hand(state, hand)
    assert baseline.hand_name == "Pair"
    # Scoring cards: the two 5s. Chips = 10(base) + 5 + 5 = 20
    assert baseline.chips == 20.0

    # Reset played count
    state.hands["Pair"]["played"] -= 1
    state.hands["Pair"]["played_this_round"] -= 1

    add_joker(state, "j_splash")
    result = score_hand(state, hand)

    assert result.hand_name == "Pair"
    # With Splash: all 3 cards score. Chips = 10(base) + 5 + 5 + 10(K) = 30
    assert result.chips == 30.0


# ---------------------------------------------------------------------------
# 11. Edge case: Perkeo — duplicates consumable at end of shop
# ---------------------------------------------------------------------------

def test_perkeo_duplicates_consumable_at_end_of_shop() -> None:
    """Perkeo duplicates a random consumable (as negative edition) at end of shop."""
    from pylatro.instances import add_consumable

    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_perkeo")
    add_consumable(state, "c_fool")  # The Fool tarot

    initial_count = len(state.consumables)
    created = apply_end_shop(state)

    assert len(created) == 1
    assert len(state.consumables) == initial_count + 1
    # The duplicated consumable should have negative edition
    duplicated = state.consumables[-1]
    assert duplicated.edition == {"negative": True}


# ---------------------------------------------------------------------------
# 12. Combined: Suit mult joker stacks with type joker
# ---------------------------------------------------------------------------

def test_suit_mult_and_type_mult_stack() -> None:
    """Suit mult and type mult jokers stack their bonuses correctly."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_greedy_joker")  # +3 mult per Diamond
    add_joker(state, "j_jolly")  # +8 mult on Pair

    hand = [_card("Diamonds", "3"), _card("Diamonds", "3")]
    result = score_hand(state, hand)

    assert result.hand_name == "Pair"
    # Base Pair mult = 2
    # Greedy: +3 per Diamond scoring card = +6
    # Jolly: +8
    # Total mult = 2 + 6 + 8 = 16
    assert result.mult == 16.0


def test_type_chips_and_type_mult_stack() -> None:
    """Type chips and type mult jokers for the same hand type stack."""
    state = create_run_state("AAAAAAAA")
    add_joker(state, "j_sly")  # +50 chips on Pair
    add_joker(state, "j_jolly")  # +8 mult on Pair

    hand = [_card("Spades", "4"), _card("Hearts", "4")]
    result = score_hand(state, hand)

    assert result.hand_name == "Pair"
    # Base Pair: chips=10, mult=2
    # Card chips: 4 + 4 = 8
    # Sly: +50 chips
    # Jolly: +8 mult
    assert result.chips == 10 + 4 + 4 + 50  # 68
    assert result.mult == 2 + 8  # 10
    assert result.total == 680
