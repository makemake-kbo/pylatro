"""Replay parity tests: verify Python engine matches Lua oracle."""

from __future__ import annotations

import pytest
from pathlib import Path

from pylatro import create_run_state, load_game_data

VENDOR_PATH = Path(__file__).resolve().parents[1] / "vendor" / "balatro_lua"

pytestmark = pytest.mark.skipif(
    not VENDOR_PATH.exists(),
    reason="vendor/balatro_lua not found",
)


def _bridge():
    from pylatro.upstream.oracle_bridge import OracleBridge
    return OracleBridge()


# --- Task 2: Oracle data loading ---

def test_oracle_loads_data():
    bridge = _bridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_state = bridge.snapshot(lua_raw)
    assert lua_state["seed"] == "AAAAAAAA"
    assert lua_state["stake"] == 1
    assert lua_state["dollars"] == 4


def test_oracle_creates_52_card_deck():
    bridge = _bridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_state = bridge.snapshot(lua_raw)
    assert len(lua_state["deck_cards"]) == 52


def test_oracle_loads_hand_data():
    bridge = _bridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_state = bridge.snapshot(lua_raw)
    hands = lua_state.get("hands", {})
    assert "High Card" in hands
    assert "Flush" in hands
    assert hands["High Card"]["s_chips"] == 5
    assert hands["High Card"]["s_mult"] == 1


# --- Task 3: Oracle bridge ---

def test_oracle_bridge_create_run():
    """Oracle and Python produce same initial state structure."""
    from pylatro.upstream.oracle_bridge import OracleBridge

    data = load_game_data()
    bridge = OracleBridge()

    py_state = create_run_state("AAAAAAAA", data=data)
    lua_raw = bridge.create_run("AAAAAAAA", stake=1, deck_key="b_red")
    lua_state = bridge.snapshot(lua_raw)

    assert lua_state["seed"] == "AAAAAAAA"
    assert lua_state["dollars"] == py_state.dollars
    assert lua_state["stake"] == py_state.stake
    assert len(lua_state["deck_cards"]) == len(py_state.deck_cards)


# --- Task 4: Snapshot comparison ---

def test_snapshot_diff_identical():
    from pylatro.upstream.oracle_bridge import diff_snapshots
    snap = {"dollars": 4, "ante": 1, "jokers": [{"center_key": "j_joker"}]}
    assert diff_snapshots(snap, snap) == []


def test_snapshot_diff_scalar():
    from pylatro.upstream.oracle_bridge import diff_snapshots
    a = {"dollars": 4, "ante": 1}
    b = {"dollars": 5, "ante": 1}
    diffs = diff_snapshots(a, b)
    assert len(diffs) == 1
    assert diffs[0] == ("dollars", 4, 5)


def test_snapshot_diff_list():
    from pylatro.upstream.oracle_bridge import diff_snapshots
    a = {"keys": ["a", "b", "c"]}
    b = {"keys": ["a", "x", "c"]}
    diffs = diff_snapshots(a, b)
    assert len(diffs) == 1
    assert diffs[0] == ("keys[1]", "b", "x")


def test_snapshot_diff_list_length():
    from pylatro.upstream.oracle_bridge import diff_snapshots
    a = {"keys": ["a", "b"]}
    b = {"keys": ["a", "b", "c"]}
    diffs = diff_snapshots(a, b)
    assert any("length" in d[0] for d in diffs)


def test_snapshot_from_run_state_basic():
    from pylatro.upstream.oracle_bridge import snapshot_from_run_state
    data = load_game_data()
    state = create_run_state("AAAAAAAA", data=data)
    snap = snapshot_from_run_state(state)
    assert snap["dollars"] == 4
    assert snap["deck_cards_count"] == 52
    assert snap["ante"] == 1
