"""Targeted parity tests for 22 generic-path jokers.

Tests suit-mult, hand-type mult/chips, x_mult, and stat-modifier jokers
via the Python ↔ Lua oracle bridge.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from pylatro import create_run_state, load_game_data
from pylatro.flow import start_blind, play_cards
from pylatro.instances import add_joker
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

# ---------------------------------------------------------------------------
# Joker groups
# ---------------------------------------------------------------------------

SUIT_MULT_JOKERS = [
    ("j_greedy_joker", "Greedy Joker"),
    ("j_lusty_joker", "Lusty Joker"),
    ("j_wrathful_joker", "Wrathful Joker"),
    ("j_gluttenous_joker", "Gluttonous Joker"),
]

HAND_TYPE_JOKERS = [
    ("j_jolly", "Jolly Joker"),
    ("j_zany", "Zany Joker"),
    ("j_mad", "Mad Joker"),
    ("j_crazy", "Crazy Joker"),
    ("j_droll", "Droll Joker"),
    ("j_sly", "Sly Joker"),
    ("j_wily", "Wily Joker"),
    ("j_clever", "Clever Joker"),
    ("j_devious", "Devious Joker"),
    ("j_crafty", "Crafty Joker"),
]

XMULT_JOKERS = [
    ("j_duo", "The Duo"),
    ("j_trio", "The Trio"),
    ("j_family", "The Family"),
    ("j_order", "The Order"),
    ("j_tribe", "The Tribe"),
]

STAT_MOD_JOKERS = [
    ("j_juggler", "Juggler"),
    ("j_drunkard", "Drunkard"),
    ("j_merry_andy", "Merry Andy"),
]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _run_joker_parity(joker_key: str) -> None:
    """Core parity test: add joker, play first 5 cards, compare snapshots."""
    data = load_game_data()

    # Python side
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    add_joker(py_state, joker_key)
    play_cards(py_state, [0, 1, 2, 3, 4])

    # Lua side
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "add_joker", center_key=joker_key)
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences for {joker_key}: {diffs}"


# ---------------------------------------------------------------------------
# 1. Suit-mult jokers (4 tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("joker_key,joker_name", SUIT_MULT_JOKERS, ids=[j[0] for j in SUIT_MULT_JOKERS])
def test_suit_mult_joker_parity(joker_key: str, joker_name: str) -> None:
    """Suit-mult jokers trigger correctly on matching suited cards."""
    _run_joker_parity(joker_key)


# ---------------------------------------------------------------------------
# 2. Hand-type mult/chips jokers (10 tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("joker_key,joker_name", HAND_TYPE_JOKERS, ids=[j[0] for j in HAND_TYPE_JOKERS])
def test_hand_type_joker_parity(joker_key: str, joker_name: str) -> None:
    """Hand-type mult/chips jokers produce identical scoring in Python and Lua."""
    _run_joker_parity(joker_key)


# ---------------------------------------------------------------------------
# 3. X_mult jokers (5 tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("joker_key,joker_name", XMULT_JOKERS, ids=[j[0] for j in XMULT_JOKERS])
def test_xmult_joker_parity(joker_key: str, joker_name: str) -> None:
    """X_mult jokers produce identical multiplier in Python and Lua."""
    _run_joker_parity(joker_key)


# ---------------------------------------------------------------------------
# 4. Stat modifier jokers (3 tests)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("joker_key,joker_name", STAT_MOD_JOKERS, ids=[j[0] for j in STAT_MOD_JOKERS])
def test_stat_mod_joker_parity(joker_key: str, joker_name: str) -> None:
    """Stat modifier jokers (h_size/d_size) are applied correctly after add_joker."""
    data = load_game_data()

    # Python side
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    add_joker(py_state, joker_key)
    # Play up to available hand size (stat modifiers change hand size)
    n = min(5, len(py_state.hand_cards))
    play_cards(py_state, list(range(n)))

    # Lua side
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "add_joker", center_key=joker_key)
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=list(range(1, n + 1)))

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences for {joker_key}: {diffs}"
