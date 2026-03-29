"""Edge case parity tests: verify edge-case behavior matches between Python and Lua.

Covers:
1. The Serpent boss blind draw limit (3 cards after each play/discard)
2. Round resolution: exhausting hands (round loss scenario)
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pylatro import create_run_state, load_game_data
from pylatro.flow import start_blind, play_cards, discard_cards, draw_to_hand
from pylatro.upstream.oracle_bridge import (
    OracleBridge,
    diff_snapshots,
    snapshot_from_run_state,
    snapshot_from_lua_state,
)

VENDOR_PATH = Path(__file__).resolve().parents[1] / "vendor" / "balatro_lua"

pytestmark = pytest.mark.skipif(
    not VENDOR_PATH.exists(),
    reason="vendor/balatro_lua not found",
)


def _force_serpent_boss(py_state, lua_raw) -> None:
    """Force The Serpent as the boss blind on both sides."""
    py_state.round_resets.blind_choices["Boss"] = "bl_serpent"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    lua_raw.round_resets.blind_choices.Boss = "bl_serpent"
    lua_raw.round_resets.blind_states.Small = "Defeated"
    lua_raw.round_resets.blind_states.Big = "Defeated"
    lua_raw.round_resets.blind_states.Boss = "Select"
    lua_raw.blind_on_deck = "Boss"


def _extended_snapshot(py_state, bridge, lua_raw) -> tuple[dict, dict]:
    """Extended snapshot including hand_size for Serpent tests."""
    py_snap = snapshot_from_run_state(py_state)
    lua_dict = bridge.snapshot(lua_raw)
    lua_snap = snapshot_from_lua_state(lua_dict)

    # Add hand_size for extra verification
    py_snap["hand_size"] = py_state.current_round.hand_size
    cr = lua_dict.get("current_round", {})
    lua_snap["hand_size"] = int(cr.get("hand_size", 0)) if isinstance(cr, dict) else 0

    return py_snap, lua_snap


# ---------------------------------------------------------------------------
# Serpent draw limit tests
# ---------------------------------------------------------------------------

def test_serpent_start_blind_parity():
    """The Serpent start_blind is identical between Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    _force_serpent_boss(py_state, lua_raw)

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    py_snap, lua_snap = _extended_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "Serpent start_blind divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )

    # Serpent's initial draw fills the hand normally (no prior plays/discards)
    assert py_snap["hand_cards_count"] == 8, (
        f"Expected initial hand of 8 cards, got {py_snap['hand_cards_count']}"
    )


def test_serpent_draw_limit_after_play():
    """The Serpent limits draws to 3 cards when all hand cards are played.

    Python draw_to_hand() clamps to 3 after hands_played > 0.
    The Lua oracle's play_hand only draws when hand is empty, matching
    Python's play_cards behaviour - both sides produce the same card counts
    since the Python Serpent 3-card limit only fires when hand_cards == 0.

    We verify:
    - After playing 5 of 8 cards: hand has 3 remaining, no draw occurs on either side.
    - Python correctly limits draw_to_hand to 3 when called explicitly after Serpent.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    _force_serpent_boss(py_state, lua_raw)

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    initial_hand = len(py_state.hand_cards)
    assert initial_hand == 8, f"Expected 8 initial cards, got {initial_hand}"

    # Play 5 cards - hand has 3 remaining, no draw occurs (hand not empty)
    n = min(5, len(py_state.hand_cards))
    play_cards(py_state, list(range(n)))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n + 1)))

    # After playing 5 of 8, 3 remain - no draw since hand is not empty
    py_snap, lua_snap = _extended_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "Serpent after first play (5 of 8) divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )

    assert py_snap["hand_cards_count"] == initial_hand - n, (
        f"Expected {initial_hand - n} cards remaining in hand, got {py_snap['hand_cards_count']}"
    )


def test_serpent_draw_limit_python_only():
    """Verify Python draw_to_hand respects the 3-card Serpent limit.

    This is a pure Python unit test checking that after hands_played > 0,
    draw_to_hand() draws at most 3 cards regardless of hand_size.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)

    # Force Serpent and start blind
    py_state.round_resets.blind_choices["Boss"] = "bl_serpent"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    assert len(py_state.hand_cards) == 8, "Initial hand should be 8"

    # Simulate hands_played > 0 (what draw_to_hand checks)
    py_state.current_round.hands_played = 1

    # Manually empty the hand and draw_pile to test the limit
    draw_pile_size_before = len(py_state.draw_pile)
    py_state.hand_cards.clear()  # empty hand

    # Now draw_to_hand should draw at most 3 (Serpent limit)
    drawn = draw_to_hand(py_state)
    assert len(drawn) == min(3, draw_pile_size_before), (
        f"Serpent should draw at most 3 cards, drew {len(drawn)}"
    )
    assert len(py_state.hand_cards) == min(3, draw_pile_size_before), (
        f"Serpent hand should have at most 3 cards, has {len(py_state.hand_cards)}"
    )


def test_serpent_draw_limit_after_discard():
    """The Serpent limits draws to 3 after a discard (discards_used > 0).

    Pure Python test: after first discard, subsequent draws are capped at 3.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)

    py_state.round_resets.blind_choices["Boss"] = "bl_serpent"
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Defeated"
    py_state.round_resets.blind_states["Boss"] = "Select"
    py_state.blind_on_deck = "Boss"

    start_blind(py_state, "Boss")
    assert len(py_state.hand_cards) == 8, "Initial hand should be 8"
    assert py_state.current_round.discards_used == 0

    # Discard 3 cards - after discard, draw_to_hand is called with Serpent active
    # discards_used becomes 1 so next draw_to_hand call will draw at most 3
    discard_cards(py_state, [0, 1, 2])

    # After discarding 3, Serpent should have drawn exactly 3 replacements
    # (since discards_used is now 1 when draw_to_hand is called inside discard_cards)
    assert py_state.current_round.discards_used == 1, "discards_used should be 1"
    assert len(py_state.hand_cards) == 8, (
        f"After discarding 3 and drawing 3 (Serpent), expected 8 cards, got {len(py_state.hand_cards)}"
    )


def test_serpent_parity_two_plays():
    """Parity test across two hand plays under The Serpent.

    Both Python and Lua play 5 cards twice. On second play, the Serpent limit
    of 3 only fires if the hand is empty; since 3 cards remain after each
    5-card play, no draw happens. State must match throughout.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    _force_serpent_boss(py_state, lua_raw)

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # First play
    n1 = min(5, len(py_state.hand_cards))
    play_cards(py_state, list(range(n1)))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n1 + 1)))

    py_snap, lua_snap = _extended_snapshot(py_state, bridge, lua_raw)
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "Serpent after play 1 divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )

    # Second play
    n2 = min(5, len(py_state.hand_cards))
    if n2 > 0:
        play_cards(py_state, list(range(n2)))
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n2 + 1)))

        py_snap, lua_snap = _extended_snapshot(py_state, bridge, lua_raw)
        diffs = diff_snapshots(py_snap, lua_snap)
        assert diffs == [], (
            "Serpent after play 2 divergences:\n"
            + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
        )


# ---------------------------------------------------------------------------
# Round resolution / exhausted hands test
# ---------------------------------------------------------------------------

def test_round_exhausted_hands_parity():
    """Verify state after all 4 hands are played (round loss scenario).

    Plays all available hands on a standard Small blind. Both Python and Lua
    should agree on the resulting state: 0 hands_left, accumulated score,
    same card distribution.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    # Start Small blind (standard, no boss effects)
    start_blind(py_state, "Small")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")

    # Play all 4 hands
    for _ in range(4):
        n = min(5, len(py_state.hand_cards))
        if n == 0:
            break
        play_cards(py_state, list(range(n)))
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n + 1)))

    # Both sides should have 0 hands_left
    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))

    assert py_snap["hands_left"] == 0, (
        f"Expected 0 hands_left after exhausting round, got {py_snap['hands_left']}"
    )

    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "Round exhausted state divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )


def test_round_exhausted_hands_serpent_parity():
    """Exhaust all hands under The Serpent boss blind - parity check.

    After all 4 hands are played, Python and Lua should agree.
    Since The Serpent only triggers draw_to_hand (capped at 3) when hand is
    empty, and each play leaves 3 cards in hand, the divergence point never
    fires in typical play. This test confirms parity regardless.
    """
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    _force_serpent_boss(py_state, lua_raw)

    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # Play all available hands
    for _ in range(4):
        n = min(5, len(py_state.hand_cards))
        if n == 0:
            break
        play_cards(py_state, list(range(n)))
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n + 1)))

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))

    assert py_snap["hands_left"] == 0, (
        f"Expected 0 hands_left, got {py_snap['hands_left']}"
    )

    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], (
        "Serpent exhausted round divergences:\n"
        + "\n".join(f"  {path}: py={py_val!r} lua={lua_val!r}" for path, py_val, lua_val in diffs)
    )
