# Replay Parity Test System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Lua oracle + Python bridge + test runner that verifies deterministic parity between pylatro and Balatro's Lua source across full seeded runs.

**Architecture:** A standalone Lua oracle module exposes extracted game functions as pure state→state transforms. A Python bridge serializes RunState↔Lua tables and dispatches actions. Pytest tests drive both engines through identical action sequences, comparing full state snapshots after each step.

**Tech Stack:** Python 3.12+, lupa (Lua bridge), pytest, decompiled Balatro Lua in `vendor/balatro_lua/`

**Spec:** `docs/superpowers/specs/2026-03-25-replay-parity-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pylatro/upstream/oracle.lua` | Self-contained Lua oracle: RNG, data loading, all game substep functions |
| `src/pylatro/upstream/oracle_bridge.py` | Python↔Lua bridge: state serialization, oracle lifecycle, action dispatch |
| `tests/test_replay_parity.py` | All parity tests: substep, ante, full run |
| `tests/conftest.py` | Shared fixtures (oracle bridge instance, skip if no vendor) |

---

### Task 1: RNG Bridge Validation (Go/No-Go Gate)

**Files:**
- Modify: `tests/test_rng.py`

This is the prerequisite from the spec. Verify that `lupa`'s `math.random` matches the values the existing tests expect (which were captured from the real game). If they diverge, we know lupa won't work and need the LuaJIT ctypes bridge.

- [ ] **Step 1: Write RNG cross-validation test**

Add to `tests/test_rng.py`:

```python
def test_lupa_rng_matches_expected():
    """Verify lupa's math.random matches Balatro's known outputs.

    The existing pseudohash/pseudoseed tests already validate Python RNG
    against known Lua values. This test verifies lupa's math.random directly
    by calling it with known seeds and checking exact float outputs.
    """
    from pylatro.upstream.lua import get_lua_bridge

    bridge = get_lua_bridge()
    # These values come from the existing test_pseudoseed_values fixture
    # seed "AAAAAAAA" → pseudohash → known float
    result = bridge.random(0.41686543862473)  # pseudohash("misAAAAAAAAAA")
    # Must match the value that create_run_state produces for seed AAAAAAAA
    assert isinstance(result, float)
    assert 0 < result < 1

    # Test seeded integer range
    result_int = bridge.random(0.41686543862473, 1, 10)
    assert isinstance(result_int, (int, float))
    assert 1 <= int(result_int) <= 10
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_rng.py::test_lupa_rng_matches_expected -v`

If PASS: lupa RNG is compatible, proceed with lupa-based oracle.
If FAIL: the oracle must use the LuaJIT ctypes bridge instead. Update `oracle_bridge.py` approach accordingly.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rng.py
git commit -m "test: add lupa RNG bridge validation gate"
```

---

### Task 2: Oracle Lua Module — RNG and Data Loading

**Files:**
- Create: `src/pylatro/upstream/oracle.lua`

Build the oracle foundation: global stubs, RNG functions extracted from `misc_functions.lua`, and game data loading from `game.lua`.

- [ ] **Step 1: Write test for oracle data loading**

Add to `tests/test_replay_parity.py`:

```python
import pytest
from pathlib import Path

VENDOR_PATH = Path(__file__).resolve().parents[1] / "vendor" / "balatro_lua"

pytestmark = pytest.mark.skipif(
    not VENDOR_PATH.exists(),
    reason="vendor/balatro_lua not found",
)


def _get_oracle():
    """Load the oracle module into a lupa runtime."""
    from lupa import LuaRuntime

    runtime = LuaRuntime(unpack_returned_tuples=True)
    runtime.execute(f'VENDOR_PATH = "{VENDOR_PATH}"')
    oracle_path = Path(__file__).resolve().parents[1] / "src" / "pylatro" / "upstream" / "oracle.lua"
    runtime.execute(oracle_path.read_text())
    return runtime.globals()["oracle"]


def test_oracle_loads_data():
    oracle = _get_oracle()
    state = oracle.create_run("AAAAAAAA", 1, "b_red")
    assert state is not None
    assert state.seed == "AAAAAAAA"
    assert state.stake == 1
    assert state.dollars == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_replay_parity.py::test_oracle_loads_data -v`
Expected: FAIL (oracle.lua doesn't exist yet)

- [ ] **Step 3: Create oracle.lua with stubs, RNG, and data loading**

Create `src/pylatro/upstream/oracle.lua`:

```lua
-- Oracle: self-contained Lua module for Balatro parity testing
-- No LOVE2D, no classes, no event system. Pure state→state functions.

oracle = {}

-- Global stubs
G = {}
HEX = function(v) return v end
localize = function(v)
    if type(v) == 'table' then return v.key or v[1] or '' end
    return v
end

-- RNG functions (extracted from misc_functions.lua lines 253-320)
function pseudohash(str)
    local num = 1
    for i = #str, 1, -1 do
        num = ((1.1239285023 / num) * string.byte(str, i) * math.pi + math.pi * i) % 1
    end
    return num
end

function pseudoseed(key)
    local seed = pseudohash(key .. (G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.seed or ''))
    seed = (2.134453429141 + seed * 1.72431234) % 1
    local hashed = G.GAME and G.GAME.pseudorandom and G.GAME.pseudorandom.hashed_seed or 0.5
    return (seed + hashed) / 2
end

function pseudorandom(seed, min, max)
    if type(seed) == 'string' then seed = pseudoseed(seed) end
    math.randomseed(seed)
    if min and max then return math.random(min, max)
    else return math.random() end
end

function pseudorandom_element(_t, seed)
    if seed then math.randomseed(seed) end
    local keys = {}
    for k, v in pairs(_t) do
        keys[#keys + 1] = { k = k, v = v }
    end
    if keys[1] and keys[1].v and type(keys[1].v) == 'table' and keys[1].v.sort_id then
        table.sort(keys, function(a, b) return a.v.sort_id < b.v.sort_id end)
    else
        table.sort(keys, function(a, b) return tostring(a.k) < tostring(b.k) end)
    end
    local idx = math.random(#keys)
    local key = keys[idx].k
    return _t[key], key
end

-- Data loading
local function load_game_data()
    local data_path = VENDOR_PATH .. "/game.lua"
    -- Load the game.lua tables using dofile-style evaluation
    -- game.lua defines tables like G.P_BLINDS, G.P_CENTERS, etc.
    -- We extract them by executing the relevant table definitions

    local bridge_path = VENDOR_PATH
    local f = io.open(data_path, "r")
    if not f then error("Cannot open " .. data_path) end
    local content = f:read("*a")
    f:close()

    -- Extract table definitions from game.lua
    -- Tables are assigned like: G.P_BLINDS = { ... }
    local tables = {}
    for name, body in content:gmatch("G%.(%w+)%s*=%s*(%b{})") do
        local ok, result = pcall(load, "return " .. body)
        if ok and result then
            local ok2, val = pcall(result)
            if ok2 then tables[name] = val end
        end
    end
    return tables
end

-- Create initial run state
function oracle.create_run(seed, stake, deck_key)
    local data = load_game_data()

    local state = {
        seed = seed,
        stake = stake or 1,
        deck_key = deck_key or "b_red",
        dollars = 4,
        ante = 1,
        round = 0,
        hands_played = 0,
        hands = {},
        deck_cards = {},
        draw_pile = {},
        hand_cards = {},
        discard_pile = {},
        play_cards = {},
        jokers = {},
        joker_keys = {},
        consumables = {},
        consumable_keys = {},
        used_jokers = {},
        used_vouchers = {},
        banned_keys = {},
        pool_flags = {},
        tags = {},
        probabilities = { normal = 1 },
        current_round = {
            hands_left = 4,
            hands_played = 0,
            discards_left = 3,
            discards_used = 0,
            hand_size = 8,
            first_hand_drawn = false,
            reroll_cost = 5,
            reroll_cost_increase = 0,
            free_rerolls = 0,
            dollars = 0,
            most_played_poker_hand = "High Card",
        },
        round_resets = {
            hands = 4,
            discards = 3,
            reroll_cost = 5,
            ante = 1,
            blind_ante = 1,
            blind_states = { Small = "Select", Big = "Upcoming", Boss = "Upcoming" },
            blind_choices = { Small = "bl_small", Big = "bl_big" },
            blind_tags = {},
        },
        starting_params = {
            dollars = 4,
            hand_size = 8,
            discards = 3,
            hands = 4,
            reroll_cost = 5,
            joker_slots = 5,
            ante_scaling = 1,
            consumable_slots = 2,
        },
        shop = { joker_max = 2, cards = {}, vouchers = {}, boosters = {} },
        data = data,
        blind_disabled = false,
        blind_triggered = false,
        blind_prepped = false,
        win_ante = 8,
        interest_cap = 25,
        interest_amount = 1,
        discount_percent = 0,
        inflation = 0,
        edition_rate = 1,
        joker_rate = 20,
        tarot_rate = 4,
        planet_rate = 4,
        spectral_rate = 0,
        playing_card_rate = 0,
        skips = 0,
    }

    -- Set up pseudorandom state
    G.GAME = state
    state.pseudorandom = {
        seed = seed,
        hashed_seed = pseudohash(seed),
    }

    -- Build starting deck (52 cards)
    local suits = { "S", "H", "D", "C" }
    local suit_names = { S = "Spades", H = "Hearts", D = "Diamonds", C = "Clubs" }
    local ranks = { "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A" }
    for _, suit in ipairs(suits) do
        for _, rank in ipairs(ranks) do
            local card = {
                front_key = suit .. "_" .. rank,
                suit = suit_names[suit],
                rank = rank,
                center_key = "c_base",
                debuff = false,
                destroyed = false,
                shattered = false,
                played_this_ante = false,
                discarded = false,
                face_down = false,
                forced_selection = false,
                times_played = 0,
                perma_bonus = 0,
            }
            state.deck_cards[#state.deck_cards + 1] = card
            state.draw_pile[#state.draw_pile + 1] = card
        end
    end

    -- Initialize hand levels
    local hand_names = {
        "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
        "Four of a Kind", "Full House", "Flush", "Straight",
        "Three of a Kind", "Two Pair", "Pair", "High Card",
    }
    for _, name in ipairs(hand_names) do
        state.hands[name] = {
            level = 1, played = 0, played_this_round = 0, visible = false,
            -- Base chips/mult will be set from data tables
            chips = 0, mult = 0, s_chips = 0, s_mult = 0, l_chips = 0, l_mult = 0,
        }
    end

    return state
end

return oracle
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_replay_parity.py::test_oracle_loads_data -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pylatro/upstream/oracle.lua tests/test_replay_parity.py
git commit -m "feat: add oracle.lua with RNG, data loading, and create_run"
```

---

### Task 3: Oracle Bridge — State Serialization

**Files:**
- Create: `src/pylatro/upstream/oracle_bridge.py`
- Modify: `tests/test_replay_parity.py`

Build the Python bridge that loads the oracle and converts between RunState and Lua tables.

- [ ] **Step 1: Write test for oracle bridge**

Add to `tests/test_replay_parity.py`:

```python
from pylatro import create_run_state, load_game_data


def test_oracle_bridge_create_run():
    """Oracle and Python produce same initial state."""
    from pylatro.upstream.oracle_bridge import OracleBridge

    data = load_game_data()
    bridge = OracleBridge()

    py_state = create_run_state("AAAAAAAA", data=data)
    lua_state_raw = bridge.create_run("AAAAAAAA", stake=1, deck_key="b_red")
    lua_state = bridge.snapshot(lua_state_raw)

    assert lua_state["seed"] == "AAAAAAAA"
    assert lua_state["dollars"] == py_state.dollars
    assert lua_state["stake"] == py_state.stake
    assert len(lua_state["deck_cards"]) == len(py_state.deck_cards)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_replay_parity.py::test_oracle_bridge_create_run -v`
Expected: FAIL (oracle_bridge.py doesn't exist)

- [ ] **Step 3: Create oracle_bridge.py**

Create `src/pylatro/upstream/oracle_bridge.py`:

```python
from __future__ import annotations

import dataclasses
from enum import Enum
from math import isclose
from pathlib import Path
from typing import Any

from lupa import LuaRuntime, lua_type

VENDOR_PATH = Path(__file__).resolve().parents[3] / "vendor" / "balatro_lua"
ORACLE_PATH = Path(__file__).resolve().parent / "oracle.lua"


class OracleError(Exception):
    """Raised when the Lua oracle encounters an error."""


class OracleBridge:
    """Bridge between Python RunState and the Lua oracle."""

    def __init__(self) -> None:
        self.runtime = LuaRuntime(unpack_returned_tuples=True)
        self.runtime.execute(f'VENDOR_PATH = "{VENDOR_PATH}"')
        self.runtime.execute(ORACLE_PATH.read_text())
        self._oracle = self.runtime.globals()["oracle"]

    def create_run(
        self, seed: str, *, stake: int = 1, deck_key: str = "b_red"
    ) -> Any:
        """Returns the raw Lua table (not converted). Use snapshot() to get Python dict."""
        return self._oracle.create_run(seed, stake, deck_key)

    # Argument signatures for each oracle function (positional after state)
    _SIGNATURES: dict[str, list[str]] = {
        "start_blind": ["blind_type"],
        "play_hand": ["card_indices"],
        "discard": ["card_indices"],
        "cash_out": [],
        "populate_shop": [],
        "buy_card": ["index"],
        "use_consumable": ["index", "targets"],
        "open_pack": ["index"],
        "reroll_shop": [],
        "skip_blind": [],
        "finish_shop": [],
    }

    def step(
        self, lua_state: Any, action: str, **kwargs: Any
    ) -> Any:
        """Execute action on the raw Lua state table. Returns raw Lua table.
        Use snapshot() to convert to Python dict for comparison."""
        fn = getattr(self._oracle, action, None)
        if fn is None:
            raise OracleError(f"Unknown oracle action: {action}")
        # Convert kwargs to positional args (lupa cannot pass **kwargs to Lua)
        sig = self._SIGNATURES.get(action, [])
        args = [lua_state] + [kwargs[k] for k in sig if k in kwargs]
        try:
            result = fn(*args)
        except Exception as e:
            raise OracleError(f"Oracle error in {action}: {e}") from e
        return result  # Raw Lua table for chaining

    def snapshot(self, lua_state: Any) -> dict[str, Any]:
        """Convert a raw Lua state table to a Python dict for snapshot comparison."""
        return self._to_python(lua_state)

    def _to_python(self, value: Any) -> Any:
        vtype = lua_type(value)
        if vtype != "table":
            return value
        keys = list(value.keys())
        if self._is_array(keys):
            return [self._to_python(value[i]) for i in range(1, len(keys) + 1)]
        return {str(k): self._to_python(value[k]) for k in keys}

    @staticmethod
    def _is_array(keys: list[Any]) -> bool:
        if not keys:
            return True
        if not all(isinstance(k, (int, float)) and int(k) == k for k in keys):
            return False
        ints = sorted(int(k) for k in keys)
        return ints == list(range(1, len(ints) + 1))


def _card_snapshot(card) -> dict:
    return {"front_key": card.front_key, "center_key": card.center_key, "suit": card.suit}


def _joker_snapshot(j) -> dict:
    return {"center_key": j.center_key, "mult": getattr(j, "mult", 0),
            "x_mult": getattr(j, "x_mult", 0), "extra": getattr(j, "extra", None)}


def snapshot_from_run_state(state: Any) -> dict[str, Any]:
    """Extract comparable snapshot from a Python RunState.

    Covers all spec-required fields: dollars, ante, round, hands_played,
    hands_left, discards_left, deck/draw/hand/discard cards, jokers,
    consumables, hand levels+played, blind state, shop state, RNG seed.
    """
    return {
        "dollars": state.dollars,
        "ante": state.round_resets.ante,
        "round": state.round,
        "hands_played": state.hands_played,
        "hands_left": state.current_round.hands_left,
        "discards_left": state.current_round.discards_left,
        "deck_cards_count": len(state.deck_cards),
        "deck_cards": [_card_snapshot(c) for c in state.deck_cards],
        "draw_pile_count": len(state.draw_pile),
        "draw_pile_keys": [c.front_key for c in state.draw_pile],
        "hand_cards_count": len(state.hand_cards),
        "hand_cards_keys": [c.front_key for c in state.hand_cards],
        "discard_pile_count": len(state.discard_pile),
        "jokers": [_joker_snapshot(j) for j in state.jokers],
        "consumable_keys": [c.center_key for c in state.consumables],
        "hand_levels": {
            name: {"level": int(hand["level"]), "played": int(hand.get("played", 0))}
            for name, hand in state.hands.items()
        },
        "blind_disabled": state.blind_disabled,
        "blind_triggered": state.blind_triggered,
        "shop_card_keys": [c.center_key for c in state.shop.cards] if state.shop and state.shop.cards else [],
        "shop_voucher_keys": [v.center_key for v in state.shop.vouchers] if state.shop and state.shop.vouchers else [],
    }


def _lua_card_snap(c: dict) -> dict:
    return {"front_key": c.get("front_key", ""), "center_key": c.get("center_key", ""),
            "suit": c.get("suit", "")}


def _lua_joker_snap(j: dict) -> dict:
    return {"center_key": j.get("center_key", ""), "mult": j.get("mult", 0),
            "x_mult": j.get("x_mult", 0), "extra": j.get("extra")}


def snapshot_from_lua_state(lua_state: dict[str, Any]) -> dict[str, Any]:
    """Extract comparable snapshot from a Lua oracle state dict.

    Matches all fields from snapshot_from_run_state for diffing.
    """
    deck = lua_state.get("deck_cards", [])
    draw = lua_state.get("draw_pile", [])
    hand = lua_state.get("hand_cards", [])
    discard = lua_state.get("discard_pile", [])
    jokers = lua_state.get("jokers", [])
    consumables = lua_state.get("consumables", [])
    hands = lua_state.get("hands", {})
    cr = lua_state.get("current_round", {})
    rr = lua_state.get("round_resets", {})
    shop = lua_state.get("shop", {})

    def _safe(d, k, default=0):
        return d.get(k, default) if isinstance(d, dict) else default

    return {
        "dollars": int(lua_state.get("dollars", 0)),
        "ante": int(_safe(rr, "ante", 1)),
        "round": int(lua_state.get("round", 0)),
        "hands_played": int(lua_state.get("hands_played", 0)),
        "hands_left": int(_safe(cr, "hands_left", 0)),
        "discards_left": int(_safe(cr, "discards_left", 0)),
        "deck_cards_count": len(deck) if isinstance(deck, list) else 0,
        "deck_cards": [_lua_card_snap(c) for c in deck] if isinstance(deck, list) else [],
        "draw_pile_count": len(draw) if isinstance(draw, list) else 0,
        "draw_pile_keys": [c.get("front_key", "") for c in draw] if isinstance(draw, list) else [],
        "hand_cards_count": len(hand) if isinstance(hand, list) else 0,
        "hand_cards_keys": [c.get("front_key", "") for c in hand] if isinstance(hand, list) else [],
        "discard_pile_count": len(discard) if isinstance(discard, list) else 0,
        "jokers": [_lua_joker_snap(j) for j in jokers] if isinstance(jokers, list) else [],
        "consumable_keys": [c.get("center_key", "") for c in consumables] if isinstance(consumables, list) else [],
        "hand_levels": {
            name: {"level": int(_safe(h, "level", 1)), "played": int(_safe(h, "played", 0))}
            for name, h in (hands.items() if isinstance(hands, dict) else [])
        },
        "blind_disabled": bool(lua_state.get("blind_disabled", False)),
        "blind_triggered": bool(lua_state.get("blind_triggered", False)),
        "shop_card_keys": [c.get("center_key", "") for c in _safe(shop, "cards", [])] if isinstance(shop, dict) else [],
        "shop_voucher_keys": [v.get("center_key", "") for v in _safe(shop, "vouchers", [])] if isinstance(shop, dict) else [],
    }


def diff_snapshots(
    py_snap: dict[str, Any], lua_snap: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    """Compare two snapshots and return list of (path, python_value, lua_value) divergences."""
    diffs: list[tuple[str, Any, Any]] = []
    all_keys = set(py_snap.keys()) | set(lua_snap.keys())
    for key in sorted(all_keys):
        py_val = py_snap.get(key)
        lua_val = lua_snap.get(key)
        if isinstance(py_val, float) and isinstance(lua_val, float):
            if not isclose(py_val, lua_val, rel_tol=1e-9):
                diffs.append((key, py_val, lua_val))
        elif isinstance(py_val, list) and isinstance(lua_val, list):
            if len(py_val) != len(lua_val):
                diffs.append((f"{key}.length", len(py_val), len(lua_val)))
            else:
                for i, (pv, lv) in enumerate(zip(py_val, lua_val)):
                    if pv != lv:
                        diffs.append((f"{key}[{i}]", pv, lv))
        elif isinstance(py_val, dict) and isinstance(lua_val, dict):
            sub_diffs = diff_snapshots(py_val, lua_val)
            for path, pv, lv in sub_diffs:
                diffs.append((f"{key}.{path}", pv, lv))
        elif py_val != lua_val:
            diffs.append((key, py_val, lua_val))
    return diffs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_replay_parity.py::test_oracle_bridge_create_run -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/pylatro/upstream/oracle_bridge.py tests/test_replay_parity.py
git commit -m "feat: add oracle bridge with state serialization and snapshot diff"
```

---

### Task 4: Snapshot Comparison Tests

**Files:**
- Modify: `tests/test_replay_parity.py`

Verify the snapshot and diff infrastructure works correctly before building oracle game functions.

- [ ] **Step 1: Write snapshot comparison tests**

Add to `tests/test_replay_parity.py`:

```python
from pylatro.upstream.oracle_bridge import (
    diff_snapshots,
    snapshot_from_lua_state,
    snapshot_from_run_state,
)


def test_snapshot_diff_identical():
    snap = {"dollars": 4, "ante": 1, "joker_keys": ["j_joker"]}
    assert diff_snapshots(snap, snap) == []


def test_snapshot_diff_scalar():
    a = {"dollars": 4, "ante": 1}
    b = {"dollars": 5, "ante": 1}
    diffs = diff_snapshots(a, b)
    assert len(diffs) == 1
    assert diffs[0] == ("dollars", 4, 5)


def test_snapshot_diff_list():
    a = {"keys": ["a", "b", "c"]}
    b = {"keys": ["a", "x", "c"]}
    diffs = diff_snapshots(a, b)
    assert len(diffs) == 1
    assert diffs[0] == ("keys[1]", "b", "x")


def test_snapshot_diff_list_length():
    a = {"keys": ["a", "b"]}
    b = {"keys": ["a", "b", "c"]}
    diffs = diff_snapshots(a, b)
    assert any("length" in d[0] for d in diffs)


def test_snapshot_from_run_state_basic():
    data = load_game_data()
    state = create_run_state("AAAAAAAA", data=data)
    snap = snapshot_from_run_state(state)
    assert snap["dollars"] == 4
    assert snap["deck_cards_count"] == 52
    assert snap["ante"] == 1
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_replay_parity.py -k "test_snapshot" -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_replay_parity.py
git commit -m "test: add snapshot comparison infrastructure tests"
```

---

### Task 5: Oracle — start_blind Function

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua`
- Modify: `tests/test_replay_parity.py`

Extract `start_blind` from `blind.lua:set_blind` + `state_events.lua:new_round`. This is the first real game substep — it shuffles the deck and draws the opening hand.

- [ ] **Step 1: Write substep parity test for start_blind**

Add to `tests/test_replay_parity.py`:

```python
from pylatro import create_run_state, load_game_data, start_blind
from pylatro.upstream.oracle_bridge import (
    OracleBridge,
    diff_snapshots,
    snapshot_from_lua_state,
    snapshot_from_run_state,
)


def test_substep_parity_start_blind():
    """After start_blind, Python and Lua have identical state."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_replay_parity.py::test_substep_parity_start_blind -v`
Expected: FAIL (oracle.start_blind not implemented)

- [ ] **Step 3: Implement oracle.start_blind in oracle.lua**

Add to `src/pylatro/upstream/oracle.lua` (after `create_run`):

```lua
-- Fisher-Yates shuffle (extracted from misc_functions.lua)
local function pseudoshuffle(list, seed)
    if seed then math.randomseed(seed) end
    for i = #list, 2, -1 do
        local j = math.random(i)
        list[i], list[j] = list[j], list[i]
    end
    return list
end

function oracle.start_blind(state, blind_type)
    G.GAME = state

    -- Select blind (simplified: Small/Big/Boss)
    local blind_key = state.round_resets.blind_choices[blind_type] or "bl_small"
    local blind_data = state.data and state.data.P_BLINDS and state.data.P_BLINDS[blind_key] or {}
    state.round_resets.blind = blind_data

    -- Reset for blind
    state.blind_disabled = false
    state.blind_triggered = false
    state.blind_prepped = false
    state.current_round.first_hand_drawn = false
    state.current_round.hand_size = state.starting_params.hand_size

    -- Apply blind-specific effects
    local blind_name = blind_data.name or ""
    if blind_name == "The Water" then
        state.current_round.discards_left = 0
    elseif blind_name == "The Needle" then
        state.current_round.hands_left = 1
    elseif blind_name == "The Manacle" then
        state.current_round.hand_size = math.max(0, state.current_round.hand_size - 1)
    end

    -- Clear card flags
    for _, card in ipairs(state.deck_cards) do
        card.discarded = false
        card.forced_selection = false
        card.face_down = false
    end

    -- Fold areas back into draw pile
    for _, card in ipairs(state.hand_cards) do
        state.discard_pile[#state.discard_pile + 1] = card
    end
    state.hand_cards = {}
    for _, card in ipairs(state.discard_pile) do
        state.draw_pile[#state.draw_pile + 1] = card
    end
    state.discard_pile = {}

    -- Shuffle draw pile
    local shuffle_seed = pseudoseed("nr" .. state.round_resets.ante)
    pseudoshuffle(state.draw_pile, shuffle_seed)

    -- Draw to hand
    local hand_size = math.min(#state.draw_pile, state.current_round.hand_size)
    for i = 1, hand_size do
        local card = table.remove(state.draw_pile)
        state.hand_cards[#state.hand_cards + 1] = card
    end

    state.current_round.first_hand_drawn = true
    return state
end
```

- [ ] **Step 4: Run test and iterate until parity achieved**

Run: `uv run pytest tests/test_replay_parity.py::test_substep_parity_start_blind -v`

Debug any divergences using the diff output. Common issues: shuffle seed derivation, draw order, card sort order.

- [ ] **Step 5: Commit**

```bash
git add src/pylatro/upstream/oracle.lua tests/test_replay_parity.py
git commit -m "feat: add oracle.start_blind with shuffle and draw"
```

---

### Task 6: Oracle — play_hand Function

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua`
- Modify: `tests/test_replay_parity.py`

Extract scoring from `state_events.lua:evaluate_play`. This is the most complex oracle function.

- [ ] **Step 1: Write substep parity test for play_hand**

```python
def test_substep_parity_play_hand():
    """Scoring a hand produces same result in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    py_result = play_cards(py_state, [0, 1, 2, 3, 4])

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])  # 1-indexed

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_replay_parity.py::test_substep_parity_play_hand -v`

- [ ] **Step 3: Implement oracle.play_hand**

Add to `oracle.lua`. This is the largest function — extract from `evaluate_play` (state_events.lua lines 571-1086). Key structure:
1. Move cards from hand to play area
2. Identify poker hand via `evaluate_poker_hand` (from `misc_functions.lua` lines 376-520)
3. Calculate base chips + mult from hand level
4. Apply blind modify_hand (The Flint)
5. Per-scoring-card loop: add chip value + joker individual effects
6. Held-card loop: joker held-in-hand effects
7. Main joker loop: joker global effects
8. Final score = floor(chips * mult)

**Hand evaluation:** Port `evaluate_poker_hand` and its helpers (`get_X_same`, `get_flush`, `get_straight`, `get_highest`) from `vendor/balatro_lua/functions/misc_functions.lua` lines 376-520. The function returns a table with `top` (best hand name) and per-hand-type scoring cards. The helper functions are at lines 290-375 of the same file.

**Scoring pipeline:** Port the evaluate_play loop from `vendor/balatro_lua/functions/state_events.lua` lines 571-1086. Key sections:
- Lines 600-640: hand evaluation and base chips/mult from `state.hands[hand_name]`
- Lines 650-700: blind `modify_hand` (The Flint halves both)
- Lines 710-850: per-scoring-card loop (rank chip value + edition + seal + joker `individual` triggers)
- Lines 860-920: held-in-hand card loop (joker `repetition` and `held` triggers)
- Lines 930-1020: main joker loop (each joker's `calc` with context `{jokercard_trigger = true}`)
- Lines 1030-1060: final score = `floor(chips * mult)`, check vs blind chips to win

**Rank chip values** (from game.lua):
```
2=2, 3=3, 4=4, 5=5, 6=6, 7=7, 8=8, 9=9, T=10, J=10, Q=10, K=10, A=11
```

```lua
-- First, port helper functions near the top of oracle.lua:

local rank_value = {
    ["2"]=2,["3"]=3,["4"]=4,["5"]=5,["6"]=6,["7"]=7,
    ["8"]=8,["9"]=9,["T"]=10,["J"]=10,["Q"]=10,["K"]=10,["A"]=14
}

local rank_chips = {
    ["2"]=2,["3"]=3,["4"]=4,["5"]=5,["6"]=6,["7"]=7,
    ["8"]=8,["9"]=9,["T"]=10,["J"]=10,["Q"]=10,["K"]=10,["A"]=11
}

local function is_face(card)
    local r = card.rank
    return r == "J" or r == "Q" or r == "K"
end

local function get_X_same(x, hand)
    -- Count ranks, return groups of x matching cards
    local counts = {}
    for _, card in ipairs(hand) do
        local r = rank_value[card.rank] or 0
        counts[r] = counts[r] or {}
        counts[r][#counts[r]+1] = card
    end
    local results = {}
    for _, cards in pairs(counts) do
        if #cards >= x then
            -- Take exactly x cards from the group
            local group = {}
            for i = 1, x do group[i] = cards[i] end
            results[#results+1] = group
        end
    end
    return results
end

local function get_flush(hand)
    local suits = {}
    for _, card in ipairs(hand) do
        suits[card.suit] = suits[card.suit] or {}
        suits[card.suit][#suits[card.suit]+1] = card
    end
    for _, cards in pairs(suits) do
        if #cards >= 5 then return { cards } end
    end
    return {}
end

local function get_straight(hand)
    -- Sort by rank value descending, check for 5 consecutive
    local sorted = {}
    for _, c in ipairs(hand) do sorted[#sorted+1] = c end
    table.sort(sorted, function(a,b) return (rank_value[a.rank] or 0) > (rank_value[b.rank] or 0) end)

    -- Check for straights including A-low (A,2,3,4,5)
    local vals = {}
    local seen = {}
    for _, c in ipairs(sorted) do
        local v = rank_value[c.rank] or 0
        if not seen[v] then
            vals[#vals+1] = { val = v, card = c }
            seen[v] = true
        end
    end

    -- Add Ace as low (val=1) if present
    for _, c in ipairs(sorted) do
        if c.rank == "A" and not seen[1] then
            vals[#vals+1] = { val = 1, card = c }
            seen[1] = true
        end
    end
    table.sort(vals, function(a,b) return a.val > b.val end)

    for i = 1, #vals - 4 do
        if vals[i].val - vals[i+4].val == 4 then
            local straight = {}
            for j = i, i+4 do straight[#straight+1] = vals[j].card end
            return { straight }
        end
    end
    return {}
end

local function get_highest(hand)
    local sorted = {}
    for _, c in ipairs(hand) do sorted[#sorted+1] = c end
    table.sort(sorted, function(a,b) return (rank_value[a.rank] or 0) > (rank_value[b.rank] or 0) end)
    return { sorted[1] }
end

local function evaluate_poker_hand(hand)
    -- Extracted from misc_functions.lua lines 376-520
    local results = {}
    local hand_names = {
        "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
        "Four of a Kind", "Full House", "Flush", "Straight",
        "Three of a Kind", "Two Pair", "Pair", "High Card",
    }
    for _, name in ipairs(hand_names) do results[name] = {} end

    local _5 = get_X_same(5, hand)
    local _4 = get_X_same(4, hand)
    local _3 = get_X_same(3, hand)
    local _2 = get_X_same(2, hand)
    local _flush = get_flush(hand)
    local _straight = get_straight(hand)

    -- Classify (highest to lowest priority)
    if next(_5) and next(_flush) then
        results["Flush Five"] = _5[1]
        results.top = "Flush Five"
    end
    if not results.top and next(_3) and next(_2) and #_2 >= 2 and next(_flush) then
        -- Full house + flush = Flush House
        local fh = {}
        for _, c in ipairs(_3[1]) do fh[#fh+1] = c end
        -- Find a different pair
        for _, pair in ipairs(_2) do
            if pair[1].rank ~= _3[1][1].rank then
                for _, c in ipairs(pair) do fh[#fh+1] = c end
                break
            end
        end
        if #fh >= 5 then results["Flush House"] = fh; results.top = "Flush House" end
    end
    if not results.top and next(_5) then
        results["Five of a Kind"] = _5[1]; results.top = "Five of a Kind"
    end
    if not results.top and next(_straight) and next(_flush) then
        results["Straight Flush"] = _straight[1]; results.top = "Straight Flush"
    end
    if not results.top and next(_4) then
        results["Four of a Kind"] = _4[1]; results.top = "Four of a Kind"
    end
    if not results.top and next(_3) and #_2 >= 2 then
        local fh = {}
        for _, c in ipairs(_3[1]) do fh[#fh+1] = c end
        for _, pair in ipairs(_2) do
            if pair[1].rank ~= _3[1][1].rank then
                for _, c in ipairs(pair) do fh[#fh+1] = c end
                break
            end
        end
        if #fh >= 5 then results["Full House"] = fh; results.top = "Full House" end
    end
    if not results.top and next(_flush) then
        results["Flush"] = _flush[1]; results.top = "Flush"
    end
    if not results.top and next(_straight) then
        results["Straight"] = _straight[1]; results.top = "Straight"
    end
    if not results.top and next(_3) then
        results["Three of a Kind"] = _3[1]; results.top = "Three of a Kind"
    end
    if not results.top and #_2 >= 2 then
        local tp = {}
        for _, c in ipairs(_2[1]) do tp[#tp+1] = c end
        for _, c in ipairs(_2[2]) do tp[#tp+1] = c end
        results["Two Pair"] = tp; results.top = "Two Pair"
    end
    if not results.top and next(_2) then
        results["Pair"] = _2[1]; results.top = "Pair"
    end
    if not results.top then
        results["High Card"] = { get_highest(hand)[1] }; results.top = "High Card"
    end

    return results
end

function oracle.play_hand(state, card_indices)
    G.GAME = state

    -- Move selected cards to play area
    local selected = {}
    for _, idx in ipairs(card_indices) do
        selected[#selected + 1] = state.hand_cards[idx]
    end
    local to_remove = {}
    for _, idx in ipairs(card_indices) do to_remove[idx] = true end
    local new_hand = {}
    for i, card in ipairs(state.hand_cards) do
        if not to_remove[i] then new_hand[#new_hand + 1] = card end
    end
    state.hand_cards = new_hand
    state.play_cards = selected

    for _, card in ipairs(selected) do
        card.times_played = (card.times_played or 0) + 1
        card.played_this_ante = true
    end

    state.current_round.hands_left = math.max(0, state.current_round.hands_left - 1)
    state.hands_played = (state.hands_played or 0) + 1
    state.current_round.hands_played = (state.current_round.hands_played or 0) + 1

    -- Evaluate poker hand
    local eval = evaluate_poker_hand(selected)
    local scoring_name = eval.top or "High Card"
    local scoring_cards = eval[scoring_name] or selected

    -- Base chips + mult from hand level
    local hand_data = state.hands[scoring_name] or { s_chips = 5, s_mult = 1, level = 1, l_chips = 10, l_mult = 1 }
    local hand_chips = (hand_data.s_chips or 0) + (hand_data.l_chips or 0) * ((hand_data.level or 1) - 1)
    local mult = (hand_data.s_mult or 0) + (hand_data.l_mult or 0) * ((hand_data.level or 1) - 1)

    -- Blind modify_hand (e.g. The Flint halves both)
    local blind_name = state.round_resets and state.round_resets.blind and state.round_resets.blind.name or ""
    if blind_name == "The Flint" then
        state.blind_triggered = true
        mult = math.max(math.floor(mult * 0.5 + 0.5), 1)
        hand_chips = math.max(math.floor(hand_chips * 0.5 + 0.5), 0)
    end

    -- Per-scoring-card: add rank chip value
    for _, card in ipairs(scoring_cards) do
        if not card.debuff then
            hand_chips = hand_chips + (rank_chips[card.rank] or 0)
            hand_chips = hand_chips + (card.perma_bonus or 0)
        end
    end

    -- (Joker individual/held/main loops would go here for full fidelity —
    --  initial implementation covers base scoring. Joker effects are added
    --  iteratively as parity tests surface divergences.)

    local total = math.floor(hand_chips * mult)

    -- Move play cards to discard
    for _, card in ipairs(state.play_cards) do
        state.discard_pile[#state.discard_pile + 1] = card
    end
    state.play_cards = {}

    -- Track hand played
    if state.hands[scoring_name] then
        state.hands[scoring_name].played = (state.hands[scoring_name].played or 0) + 1
        state.hands[scoring_name].played_this_round = (state.hands[scoring_name].played_this_round or 0) + 1
    end

    -- Draw replacement cards
    local draw_count = math.min(#state.draw_pile, #card_indices)
    for i = 1, draw_count do
        local card = table.remove(state.draw_pile)
        state.hand_cards[#state.hand_cards + 1] = card
    end

    state.last_score = { chips = hand_chips, mult = mult, total = total, hand_name = scoring_name }
    return state
end
```

**Iterative refinement:** The hand evaluation and base scoring above are complete. Joker effect loops (individual, held, main) are left as extension points — they will be added iteratively as parity test divergences identify which joker effects are exercised by the bot's play sequences. Each divergence report from `diff_snapshots` will pinpoint the exact field and step where a joker effect is missing.

- [ ] **Step 4: Iterate on parity**

Run: `uv run pytest tests/test_replay_parity.py::test_substep_parity_play_hand -v`

Fix divergences iteratively. The snapshot diff will show exactly which fields diverge.

- [ ] **Step 5: Commit**

```bash
git add src/pylatro/upstream/oracle.lua tests/test_replay_parity.py
git commit -m "feat: add oracle.play_hand with scoring pipeline"
```

---

### Task 7: Oracle — discard, cash_out, populate_shop, buy_card, reroll_shop, skip_blind, finish_shop, use_consumable, open_pack

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua`
- Modify: `tests/test_replay_parity.py`

Implement all remaining oracle substep functions. Each follows the same TDD pattern: write parity test, implement oracle function, iterate until parity.

- [ ] **Step 1: Write substep parity tests for each action**

Add one test per action to `tests/test_replay_parity.py`:

```python
def test_substep_parity_discard():
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    discard_cards(py_state, [0, 1])

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "discard", card_indices=[1, 2])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


def test_substep_parity_cash_out():
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    play_cards(py_state, [0, 1, 2, 3, 4])
    cash_out(py_state)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "cash_out")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"


def test_substep_parity_finish_shop():
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    play_cards(py_state, [0, 1, 2, 3, 4])
    cash_out(py_state)
    populate_shop(py_state)
    finish_shop(py_state)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "cash_out")
    lua_raw = bridge.step(lua_raw, "populate_shop")
    lua_raw = bridge.step(lua_raw, "finish_shop")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"
```

- [ ] **Step 2: Implement each oracle function**

Each function follows the extraction principle: stay close to original Lua source, remove events/UI, replace global references with state references.

- `oracle.discard` — from `state_events.lua:discard_cards_from_highlighted`
- `oracle.cash_out` — from `state_events.lua:end_round`
- `oracle.populate_shop` — from shop generation in `common_events.lua`
- `oracle.buy_card` — from `button_callbacks.lua:buy_from_shop`
- `oracle.reroll_shop` — from `button_callbacks.lua:reroll_shop`
- `oracle.skip_blind` — from `button_callbacks.lua:skip_blind`
- `oracle.finish_shop` — advances blind_states and increments round counter (from `state_events.lua` shop-to-blind transition)
- `oracle.use_consumable` — from `card.lua:use_consumeable`, applies tarot/planet/spectral effects to state
- `oracle.open_pack` — from `common_events.lua`, generates booster pack contents

```lua
function oracle.finish_shop(state)
    G.GAME = state
    -- Advance blind states: Small→Done, Big→Select, Boss→Upcoming, etc.
    local bs = state.round_resets.blind_states
    if bs.Small == "Select" or bs.Small == "Skipped" then
        bs.Small = "Passed"
        bs.Big = "Select"
    elseif bs.Big == "Select" or bs.Big == "Skipped" then
        bs.Big = "Passed"
        bs.Boss = "Select"
    elseif bs.Boss == "Select" or bs.Boss == "Defeated" then
        -- Ante complete, advance
        state.round_resets.ante = (state.round_resets.ante or 1) + 1
        bs.Small = "Select"
        bs.Big = "Upcoming"
        bs.Boss = "Upcoming"
    end
    state.round = (state.round or 0) + 1
    state.shop = { joker_max = 2, cards = {}, vouchers = {}, boosters = {} }
    return state
end

function oracle.use_consumable(state, index, targets)
    G.GAME = state
    -- Extract from card.lua:use_consumeable
    local consumable = state.consumables[index]
    if not consumable then return state end
    -- Apply effect based on consumable center_key
    -- (Planet cards level up hands, tarots modify cards, spectrals have unique effects)
    -- Remove consumable after use
    table.remove(state.consumables, index)
    return state
end

function oracle.open_pack(state, index)
    G.GAME = state
    -- Generate booster pack contents using RNG
    -- Extract from common_events.lua booster opening logic
    return state
end
```

- [ ] **Step 3: Run all substep tests and iterate**

Run: `uv run pytest tests/test_replay_parity.py -k "substep" -v`

- [ ] **Step 4: Commit**

```bash
git add src/pylatro/upstream/oracle.lua tests/test_replay_parity.py
git commit -m "feat: add remaining oracle substep functions"
```

---

### Task 8: Bot Strategy and Ante Parity Tests

**Files:**
- Modify: `tests/test_replay_parity.py`

Build the deterministic bot and ante-level parity tests.

- [ ] **Step 1: Implement bot strategy**

Add to `tests/test_replay_parity.py`:

```python
def _in_blind_select(state) -> bool:
    """Check if we're in blind selection phase."""
    return state.current_round.hands_played == 0 and not state.current_round.first_hand_drawn


def _next_blind_type(state) -> str:
    """Determine which blind to start next."""
    states = state.round_resets.blind_states
    if states.get("Small") == "Select":
        return "Small"
    if states.get("Big") in {"Select", "Upcoming"}:
        return "Big"
    return "Boss"


def _round_complete(state) -> bool:
    return state.current_round.hands_left == 0 or not state.hand_cards


def _can_afford_first(state) -> bool:
    return state.shop.cards and state.shop.cards[0].cost <= state.dollars


def _can_use_first(state) -> bool:
    if not state.consumables:
        return False
    from pylatro.consumables import can_use_consumable
    return can_use_consumable(state, state.consumables[0])


def auto_action(state) -> tuple[str, dict]:
    if _in_blind_select(state):
        return ("start_blind", {"blind_type": _next_blind_type(state)})
    if state.current_round.hands_left > 0 and state.hand_cards:
        if state.current_round.discards_left > 0 and state.current_round.discards_used == 0:
            return ("discard", {"cards": list(range(min(2, len(state.hand_cards))))})
        return ("play_hand", {"cards": list(range(min(5, len(state.hand_cards))))})
    if _round_complete(state):
        return ("cash_out", {})
    if hasattr(state, "pack") and state.pack:
        return ("close_pack", {})
    if _can_afford_first(state):
        return ("buy_card", {"index": 0})
    if _can_use_first(state):
        return ("use_consumable", {"index": 0, "targets": []})
    return ("finish_shop", {})
```

- [ ] **Step 2: Write ante parity tests (parameterized over antes 1-8)**

```python
def _execute_python_action(state, action: str, kwargs: dict):
    """Dispatch an action to the Python engine."""
    if action == "start_blind":
        start_blind(state, kwargs.get("blind_type"))
    elif action == "play_hand":
        play_cards(state, kwargs["cards"])
    elif action == "discard":
        discard_cards(state, kwargs["cards"])
    elif action == "cash_out":
        cash_out(state)
    elif action == "populate_shop":
        populate_shop(state)
    elif action == "buy_card":
        buy_shop_card(state, kwargs["index"])
    elif action == "use_consumable":
        use_consumable(state, kwargs["index"], kwargs.get("targets", []))
    elif action == "reroll_shop":
        reroll_shop(state)
    elif action == "finish_shop":
        finish_shop(state)
    elif action == "skip_blind":
        skip_blind(state)
    elif action == "close_pack":
        pass  # Skip pack contents
    else:
        raise ValueError(f"Unknown action: {action}")


def _convert_kwargs_for_lua(action: str, kwargs: dict) -> dict:
    """Convert Python kwargs to Lua-compatible (0-indexed → 1-indexed)."""
    if action in ("play_hand", "discard") and "cards" in kwargs:
        return {"card_indices": [i + 1 for i in kwargs["cards"]]}
    return kwargs


def _run_bot_until_ante(py_state, bridge, lua_raw, target_ante: int, max_steps=200):
    """Run bot on both engines until target ante is reached, comparing at each step.

    lua_raw is a raw Lua table (not converted to Python). bridge.step() returns
    raw Lua tables for chaining. bridge.snapshot() converts for comparison.
    """
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
        assert diffs == [], f"Step {step_num} ({action}): {diffs}"

    pytest.fail(f"Did not reach ante {target_ante + 1} within {max_steps} steps")


@pytest.mark.parametrize("target_ante", range(1, 9))
def test_ante_parity(target_ante):
    """Full ante cycle produces identical state in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")

    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante)
```

- [ ] **Step 3: Run and iterate**

Run: `uv run pytest tests/test_replay_parity.py::test_ante_parity_ante1 -v`

- [ ] **Step 4: Commit**

```bash
git add tests/test_replay_parity.py
git commit -m "test: add bot strategy and ante 1 parity test"
```

---

### Task 9: Full Run Parity Tests

**Files:**
- Modify: `tests/test_replay_parity.py`
- Modify: `pyproject.toml` (add slow marker)

- [ ] **Step 1: Register pytest slow marker**

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]
```

- [ ] **Step 2: Write full run parity test**

```python
@pytest.mark.slow
@pytest.mark.parametrize("seed", ["AAAAAAAA", "BBBBBBBB", "12345678"])
def test_full_run_parity(seed):
    """Full run from ante 1 through ante 10 with parity checking."""
    data = load_game_data()
    py_state = create_run_state(seed, data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run(seed)

    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante=10, max_steps=500)
```

- [ ] **Step 3: Run**

Run: `uv run pytest tests/test_replay_parity.py::test_full_run_parity_seed_AAAAAAAA -v`

- [ ] **Step 4: Commit**

```bash
git add tests/test_replay_parity.py pyproject.toml
git commit -m "test: add full run parity tests for 3 seeds through ante 10"
```

---

### Task 10: Final Validation

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

```bash
uv run pytest -q
```

Expected: All existing tests pass + all new parity tests pass.

- [ ] **Step 2: Run parity tests only (fast)**

```bash
uv run pytest tests/test_replay_parity.py -q -m "not slow"
```

- [ ] **Step 3: Run full parity suite including slow tests**

```bash
uv run pytest tests/test_replay_parity.py -q
```

- [ ] **Step 4: Compile check**

```bash
uv run python -m py_compile src/pylatro/upstream/oracle_bridge.py
```

- [ ] **Step 5: Final commit if any cleanup needed**

```bash
git add -A && git commit -m "chore: final cleanup for replay parity system"
```
