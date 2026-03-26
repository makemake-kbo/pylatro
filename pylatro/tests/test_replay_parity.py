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


# --- Task 1: Substep parity — populate_shop ---

def test_substep_parity_populate_shop():
    """After populate_shop, Python and Lua have identical shop state."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)

    from pylatro.flow import start_blind, play_cards
    from pylatro.blind import cash_out
    from pylatro.shop import populate_shop

    # Play through Small blind
    start_blind(py_state, "Small")
    for _ in range(4):
        play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    # Defeat and cash out
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Select"
    py_state.blind_on_deck = "Big"
    cash_out(py_state)
    populate_shop(py_state)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    for _ in range(4):
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "defeat_blind")
    lua_raw = bridge.step(lua_raw, "populate_shop")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


def test_substep_parity_buy_card():
    """Buying a shop card produces identical state in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    from pylatro.flow import start_blind, play_cards
    from pylatro.blind import cash_out
    from pylatro.shop import populate_shop, buy_shop_card

    start_blind(py_state, "Small")
    for _ in range(4):
        play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Select"
    py_state.blind_on_deck = "Big"
    cash_out(py_state)
    populate_shop(py_state)
    # Find the first affordable card (may not be index 0)
    py_buy_index = None
    for i, c in enumerate(py_state.shop.cards):
        if py_state.dollars >= c.cost:
            py_buy_index = i
            break
    if py_buy_index is not None:
        buy_shop_card(py_state, py_buy_index)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    for _ in range(4):
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "defeat_blind")
    lua_raw = bridge.step(lua_raw, "populate_shop")
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    # Find same affordable card in Lua (1-indexed)
    lua_buy_index = None
    for i, sc in enumerate(lua_snap.get("shop_cards", [])):
        if lua_snap["dollars"] >= sc.get("cost", 999):
            lua_buy_index = i + 1  # 1-indexed
            break
    if lua_buy_index is not None:
        lua_raw = bridge.step(lua_raw, "buy_card", index=lua_buy_index)

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"
    # Verify the buy actually happened (not a vacuous pass)
    assert py_snap["dollars"] < 4 or len(py_snap.get("jokers", [])) > 0 or len(py_snap.get("consumable_keys", [])) > 0, \
        "Buy did not execute — test is vacuous. Try a different seed or give the bot more money."


# --- Task 5: Substep parity — finish_shop ---

def test_substep_parity_finish_shop():
    """finish_shop produces identical state in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    from pylatro.flow import start_blind, play_cards
    from pylatro.blind import cash_out
    from pylatro.shop import populate_shop, finish_shop

    start_blind(py_state, "Small")
    for _ in range(4):
        play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Select"
    py_state.blind_on_deck = "Big"
    cash_out(py_state)
    populate_shop(py_state)
    finish_shop(py_state)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    for _ in range(4):
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "defeat_blind")
    lua_raw = bridge.step(lua_raw, "populate_shop")
    lua_raw = bridge.step(lua_raw, "finish_shop")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


# --- Task 4: Substep parity — use_consumable ---

def test_substep_parity_use_consumable():
    """Using a consumable produces identical state in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    from pylatro.flow import start_blind
    from pylatro.consumables import use_consumable
    from pylatro.instances import add_consumable

    start_blind(py_state, "Small")
    add_consumable(py_state, "c_mercury")  # Mercury = levels up Pair
    use_consumable(py_state, 0, hand_targets=[])

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "add_consumable", center_key="c_mercury")
    lua_raw = bridge.step(lua_raw, "use_consumable", index=1, targets={})

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


def _shop_populated(state) -> bool:
    """True if the shop has been populated this phase."""
    return bool(state.shop.cards or state.shop.boosters or state.shop.vouchers)


def auto_action(state) -> tuple[str, dict]:
    """Deterministic bot: play hands, use shop, buy cards, open packs, use consumables."""
    from pylatro.consumables import can_use_consumable

    # 1. If a blind is ready to start -> start_blind
    next_blind = _next_blind_to_start(state)
    if next_blind and not _round_active(state) and _shop_populated(state) is False and state.pack is None:
        return ("start_blind", {"blind_type": next_blind})

    # Also start blind if shop is done (next blind available, not in round, shop populated already finished)
    if next_blind and not _round_active(state) and state.pack is None and not _shop_populated(state):
        return ("start_blind", {"blind_type": next_blind})

    # 2. In a round and hands remaining -> discard once, then play hands
    if _round_active(state) and state.current_round.hands_left > 0 and state.hand_cards:
        if state.current_round.discards_left > 0 and state.current_round.discards_used == 0:
            return ("discard", {"cards": list(range(min(2, len(state.hand_cards))))})
        return ("play_hand", {"cards": list(range(min(5, len(state.hand_cards))))})

    # 3. Round active but no hands -> defeat_blind
    if _round_active(state):
        return ("defeat_blind", {})

    # 4. Shop not populated -> populate_shop
    if not _shop_populated(state) and state.pack is None:
        return ("populate_shop", {})

    # 5. Pack is open with cards -> claim_card (index 0)
    if state.pack is not None and state.pack.cards:
        return ("claim_card", {"index": 0})

    # 6. Pack is open but empty -> close_pack
    if state.pack is not None and not state.pack.cards:
        return ("close_pack", {})

    # 7. Consumables exist and first is usable -> use_consumable
    if state.consumables:
        targets = list(range(min(3, len(state.hand_cards))))
        if can_use_consumable(state, 0, hand_targets=targets):
            return ("use_consumable", {"index": 0, "targets": targets})

    # 8. Shop has affordable cards -> buy_card (index 0)
    if state.shop.cards:
        for i, card in enumerate(state.shop.cards):
            if state.dollars >= card.cost:
                return ("buy_card", {"index": i})

    # 9. Shop has affordable boosters -> open_pack (index 0)
    if state.shop.boosters:
        for i, booster in enumerate(state.shop.boosters):
            if state.dollars >= booster.cost:
                return ("open_pack", {"index": i})

    # 10. Otherwise -> finish_shop
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
    from pylatro.shop import (
        populate_shop, buy_shop_card, open_booster_pack,
        claim_pack_card, close_pack, finish_shop,
    )
    from pylatro.consumables import use_consumable

    if action == "start_blind":
        start_blind(state, kwargs.get("blind_type"))
    elif action == "play_hand":
        play_cards(state, kwargs["cards"])
    elif action == "discard":
        discard_cards(state, kwargs["cards"])
    elif action == "defeat_blind":
        _defeat_current_blind(state)
        cash_out(state)
    elif action == "populate_shop":
        populate_shop(state)
    elif action == "buy_card":
        buy_shop_card(state, kwargs["index"])
    elif action == "open_pack":
        open_booster_pack(state, kwargs["index"])
    elif action == "claim_card":
        claim_pack_card(state, kwargs["index"])
    elif action == "close_pack":
        close_pack(state)
    elif action == "use_consumable":
        use_consumable(state, kwargs["index"], hand_targets=kwargs.get("targets", []))
    elif action == "finish_shop":
        finish_shop(state)
    else:
        raise ValueError(f"Unknown action: {action}")


def _convert_kwargs_for_lua(action: str, kwargs: dict) -> dict:
    """Convert Python kwargs to Lua-compatible (0-indexed -> 1-indexed)."""
    if action in ("play_hand", "discard") and "cards" in kwargs:
        return {"card_indices": [i + 1 for i in kwargs["cards"]]}
    if action in ("buy_card", "open_pack", "claim_card"):
        return {"index": kwargs["index"] + 1}  # 0-indexed -> 1-indexed
    if action == "use_consumable":
        result = {"index": kwargs["index"] + 1}
        if "targets" in kwargs:
            result["targets"] = [i + 1 for i in kwargs["targets"]]
        return result
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


# --- Task 3: Substep parity — open_pack ---

def test_substep_parity_open_pack():
    """Opening a booster pack and claiming a card produces identical state."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    from pylatro.flow import start_blind, play_cards
    from pylatro.blind import cash_out
    from pylatro.shop import populate_shop, open_booster_pack, claim_pack_card, close_pack

    start_blind(py_state, "Small")
    for _ in range(4):
        play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    py_state.round_resets.blind_states["Small"] = "Defeated"
    py_state.round_resets.blind_states["Big"] = "Select"
    py_state.blind_on_deck = "Big"
    cash_out(py_state)
    populate_shop(py_state)

    if py_state.shop.boosters and py_state.dollars >= py_state.shop.boosters[0].cost:
        open_booster_pack(py_state, 0)
        if py_state.pack and py_state.pack.cards:
            claim_pack_card(py_state, 0)
        if py_state.pack:
            close_pack(py_state)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    for _ in range(4):
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "defeat_blind")
    lua_raw = bridge.step(lua_raw, "populate_shop")
    lua_snap_pre = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    if lua_snap_pre.get("shop_boosters"):
        lua_raw = bridge.step(lua_raw, "open_pack", index=1)  # 1-indexed
        lua_raw = bridge.step(lua_raw, "claim_card", index=1)
        lua_raw = bridge.step(lua_raw, "close_pack")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


# --- Task 9: Full run parity ---

@pytest.mark.slow
@pytest.mark.parametrize("seed", ["AAAAAAAA", "BBBBBBBB", "12345678"])
def test_full_run_parity(seed):
    """Full run from ante 1 through ante 10 with parity checking."""
    data = load_game_data()
    py_state = create_run_state(seed, data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run(seed)

    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante=10, max_steps=1000)
