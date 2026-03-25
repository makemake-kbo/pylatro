"""Replay parity tests: verify Python engine matches Lua oracle."""

from __future__ import annotations

import pytest
from pathlib import Path

from pylatro import create_run_state, load_game_data
from pylatro.upstream.oracle_bridge import (
    OracleBridge, diff_snapshots, snapshot_from_run_state, snapshot_from_lua_state,
)

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


# --- Task 5: Substep parity — start_blind ---

def test_substep_parity_start_blind():
    """After start_blind, Python and Lua have identical state."""
    from pylatro.upstream.oracle_bridge import (
        OracleBridge, diff_snapshots, snapshot_from_run_state, snapshot_from_lua_state,
    )
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)

    from pylatro.flow import start_blind
    start_blind(py_state, "Small")

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


# --- Task 6: Substep parity — play_hand ---

def test_substep_parity_play_hand():
    """Scoring a hand produces same result in Python and Lua."""
    from pylatro.upstream.oracle_bridge import (
        OracleBridge, diff_snapshots, snapshot_from_run_state, snapshot_from_lua_state,
    )
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)

    from pylatro.flow import start_blind, play_cards
    start_blind(py_state, "Small")
    play_cards(py_state, [0, 1, 2, 3, 4])

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


# --- Task 7: Substep parity — discard ---

def test_substep_parity_discard():
    """After discard, Python and Lua have identical state."""
    from pylatro.upstream.oracle_bridge import (
        OracleBridge, diff_snapshots, snapshot_from_run_state, snapshot_from_lua_state,
    )
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)

    from pylatro.flow import start_blind, discard_cards
    start_blind(py_state, "Small")
    discard_cards(py_state, [0, 1, 2])

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "discard", card_indices=[1, 2, 3])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


# --- Task 8: Bot strategy and ante parity ---

def _next_blind_to_start(state) -> str | None:
    """Return the next blind type that needs starting, or None if none pending."""
    states = state.round_resets.blind_states
    for bt in ("Small", "Big", "Boss"):
        if states.get(bt) in ("Select", "Upcoming"):
            return bt
    return None


def _round_active(state) -> bool:
    """True if we're currently in a blind round (a blind is 'Current')."""
    return any(v == "Current" for v in state.round_resets.blind_states.values())


def auto_action(state) -> tuple[str, dict]:
    """Deterministic bot: play first 5 cards, discard once, cash out, skip shop."""
    # Check if we need to start a blind
    next_blind = _next_blind_to_start(state)
    if next_blind and not _round_active(state):
        return ("start_blind", {"blind_type": next_blind})

    # In a round: discard once then play hands
    if state.current_round.hands_left > 0 and state.hand_cards:
        if state.current_round.discards_left > 0 and state.current_round.discards_used == 0:
            return ("discard", {"cards": list(range(min(2, len(state.hand_cards))))})
        return ("play_hand", {"cards": list(range(min(5, len(state.hand_cards))))})

    # Round complete: defeat blind and cash out
    if _round_active(state):
        return ("defeat_blind", {})

    return ("finish_shop", {})


def _defeat_current_blind(state):
    """Mark the current blind as Defeated (simplified - assumes blind is always beaten)."""
    for bt in ("Small", "Big", "Boss"):
        if state.round_resets.blind_states.get(bt) == "Current":
            state.round_resets.blind_states[bt] = "Defeated"
            if bt == "Small":
                state.round_resets.blind_states["Big"] = "Select"
                state.blind_on_deck = "Big"
            elif bt == "Big":
                state.round_resets.blind_states["Boss"] = "Select"
                state.blind_on_deck = "Boss"
            break


def _execute_python_action(state, action: str, kwargs: dict):
    """Dispatch an action to the Python engine."""
    from pylatro.flow import start_blind, play_cards, discard_cards
    from pylatro.blind import cash_out
    from pylatro.shop import finish_shop

    if action == "start_blind":
        start_blind(state, kwargs.get("blind_type"))
    elif action == "play_hand":
        play_cards(state, kwargs["cards"])
    elif action == "discard":
        discard_cards(state, kwargs["cards"])
    elif action == "defeat_blind":
        _defeat_current_blind(state)
        cash_out(state)
    elif action == "finish_shop":
        finish_shop(state)
    else:
        raise ValueError(f"Unknown action: {action}")


def _convert_kwargs_for_lua(action: str, kwargs: dict) -> dict:
    """Convert Python kwargs to Lua-compatible (0-indexed -> 1-indexed)."""
    if action in ("play_hand", "discard") and "cards" in kwargs:
        return {"card_indices": [i + 1 for i in kwargs["cards"]]}
    return kwargs


def _run_bot_until_ante(py_state, bridge, lua_raw, target_ante: int, max_steps=500):
    """Run bot on both engines until target ante is reached, comparing at each step."""
    for step_num in range(max_steps):
        if py_state.round_resets.ante > target_ante:
            return lua_raw

        action, kwargs = auto_action(py_state)
        _execute_python_action(py_state, action, kwargs)

        lua_kwargs = _convert_kwargs_for_lua(action, kwargs)
        lua_raw = bridge.step(lua_raw, action, **lua_kwargs)

        py_snap = snapshot_from_run_state(py_state)
        lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
        diffs = diff_snapshots(py_snap, lua_snap)
        assert diffs == [], f"Step {step_num} ({action} {kwargs}): {diffs}"

    pytest.fail(f"Did not reach ante {target_ante + 1} within {max_steps} steps")


@pytest.mark.parametrize("target_ante", range(1, 9))
def test_ante_parity(target_ante):
    """Full ante cycle produces identical state in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante)
