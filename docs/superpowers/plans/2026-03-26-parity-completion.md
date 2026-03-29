# Parity Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Achieve full Balatro replay parity by implementing oracle.lua shop/pack/consumable actions, enhancing the test bot to exercise all game paths, and running comprehensive parity tests across multiple seeds.

**Architecture:** The Lua oracle (`oracle.lua`) currently stubs shop/pack/consumable actions — they return state unchanged. Python has full implementations. We must implement real Lua versions of these actions, then enhance the test bot to exercise them, and finally run parity tests to surface and fix divergences.

**Tech Stack:** Python 3.12+, lupa 2.1 (Lua 5.4 + LuaJIT RNG), pytest

**Spec:** `docs/superpowers/specs/2026-03-25-joker-parity-completion-design.md`

---

### Task 1: Implement oracle.lua `populate_shop` action

The Lua oracle's `populate_shop` is a stub that just sets `state.shop.populated = true`. It must mirror the Python `populate_shop()` in `src/pylatro/shop.py:65-88`: generate shop cards via seeded RNG, vouchers, and booster packs.

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua:1107-1113`
- Reference: `src/pylatro/shop.py:65-88` (Python populate_shop)
- Reference: `src/pylatro/shop.py:1-63` (create_shop_card, create_card_spec helpers)
- Reference: `src/pylatro/pool.py` (get_current_pool, poll_edition, get_pack)
- Test: `tests/test_replay_parity.py`

- [ ] **Step 1: Write a parity test for populate_shop**

Add a test in `tests/test_replay_parity.py` that calls `populate_shop` on both Python and Lua after `defeat_blind` on ante 1 Small blind, then compares shop state snapshots.

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_populate_shop -v`
Expected: FAIL — Lua oracle stub doesn't populate shop cards, so snapshot will differ.

- [ ] **Step 3: Add snapshot fields for shop state**

Both `snapshot_from_run_state` and `snapshot_from_lua_state` in `oracle_bridge.py` must include shop data. Add to both snapshot functions:

In `snapshot_from_run_state` (after line 221 in `oracle_bridge.py`):
```python
"shop_cards": [{"center_key": c.center_key, "cost": c.cost} for c in state.shop.cards],
"shop_vouchers": [c.center_key for c in state.shop.vouchers],
"shop_boosters": [c.center_key for c in state.shop.boosters],
```

In `snapshot_from_lua_state` (after line 273 in `oracle_bridge.py`):
```python
shop = lua_state.get("shop", {})
shop_cards = shop.get("cards", []) if isinstance(shop, dict) else []
shop_vouchers = shop.get("vouchers", []) if isinstance(shop, dict) else []
shop_boosters = shop.get("boosters", []) if isinstance(shop, dict) else []
...
"shop_cards": [{"center_key": c.get("center_key", ""), "cost": int(c.get("cost", 0))} for c in shop_cards] if isinstance(shop_cards, list) else [],
"shop_vouchers": [c.get("center_key", "") for c in shop_vouchers] if isinstance(shop_vouchers, list) else [],
"shop_boosters": [c.get("center_key", "") for c in shop_boosters] if isinstance(shop_boosters, list) else [],
```

- [ ] **Step 4: Implement `oracle.populate_shop` in oracle.lua**

Replace the stub at `oracle.lua:1107-1113` with real shop population logic. This must:

1. Call `refresh_shop(state)` to generate `state.shop.joker_max` shop cards using seeded RNG
2. Add the current voucher to `state.shop.vouchers`
3. Generate 2 booster packs using `get_pack()` with `"shop_pack"` seed

Reference `src/pylatro/shop.py:65-88` and `src/pylatro/shop.py:1-63` for the exact logic. The Lua implementation must call the same `pseudoseed()` keys in the same order to maintain RNG parity.

Key RNG calls to mirror (from Python `create_shop_card`):
- `pseudoseed("shopjoker"..N)` for each shop slot's joker/card type
- `pseudoseed("shop_pack")` for booster pack selection
- `poll_edition()` calls with their specific seed keys

This is the most complex Lua function to implement. Study `create_shop_card()` in `shop.py:1-63` and `get_current_pool()` in `pool.py` carefully.

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_populate_shop -v`
Expected: PASS

- [ ] **Step 6: Run all existing tests for regressions**

Run: `pytest tests/ -v`
Expected: All 31+ tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/pylatro/upstream/oracle.lua src/pylatro/upstream/oracle_bridge.py tests/test_replay_parity.py
git commit -m "feat: implement oracle.lua populate_shop with parity test"
```

---

### Task 2: Implement oracle.lua `buy_card` action

The Lua oracle's `buy_card` is a stub. It must mirror `buy_shop_card()` in `src/pylatro/shop.py:107-135`: deduct cost, add card to appropriate collection (deck, consumables, or jokers).

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua:1123-1127`
- Reference: `src/pylatro/shop.py:107-135` (Python buy_shop_card)
- Reference: `src/pylatro/instances.py` (add_joker, add_consumable patterns)
- Test: `tests/test_replay_parity.py`

- [ ] **Step 1: Write a parity test for buy_card**

Add a test that populates shop on both sides, then buys card at index 0, compares state.

```python
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
    if py_state.shop.cards and py_state.dollars >= py_state.shop.cards[0].cost:
        buy_shop_card(py_state, 0)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    for _ in range(4):
        lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])
    lua_raw = bridge.step(lua_raw, "defeat_blind")
    lua_raw = bridge.step(lua_raw, "populate_shop")
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    if lua_snap.get("shop_cards") and lua_snap["dollars"] >= lua_snap["shop_cards"][0].get("cost", 999):
        lua_raw = bridge.step(lua_raw, "buy_card", index=1)  # 1-indexed

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"
    # Verify the buy actually happened (not a vacuous pass)
    assert py_snap["dollars"] < 4 or len(py_snap.get("jokers", [])) > 0 or len(py_snap.get("consumable_keys", [])) > 0, \
        "Buy did not execute — test is vacuous. Try a different seed or give the bot more money."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_buy_card -v`
Expected: FAIL

- [ ] **Step 3: Implement `oracle.buy_card` in oracle.lua**

Replace stub at `oracle.lua:1123-1127`. Must:
1. Pop card from `state.shop.cards` at given index
2. Deduct `card.cost` from `state.dollars`
3. Based on card type:
   - Joker (`set == "Joker"`): add to `state.jokers` with proper fields (center_key, mult, x_mult, extra, etc.)
   - Consumable (`consumeable == true`): add to `state.consumables`
   - Playing card (`set == "Default"/"Enhanced"`): add to `state.deck_cards`
4. Apply stat modifiers (h_size, d_size) for jokers that have them

Reference `buy_shop_card()` in `shop.py:107-135` and `add_joker()` in `instances.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_buy_card -v`
Expected: PASS

- [ ] **Step 5: Run all tests for regressions**

Run: `pytest tests/ -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/pylatro/upstream/oracle.lua tests/test_replay_parity.py
git commit -m "feat: implement oracle.lua buy_card with parity test"
```

---

### Task 3: Implement oracle.lua `open_pack` and `claim_card` actions

The Lua oracle's `open_pack` is a stub. It must mirror `open_booster_pack()` in `shop.py:174-268` (generate pack cards) and support `claim_card` for selecting a card from the opened pack.

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua:1141-1145`
- Modify: `src/pylatro/upstream/oracle_bridge.py` (add `claim_card` and `close_pack` signatures)
- Reference: `src/pylatro/shop.py:174-268` (open_booster_pack)
- Reference: `src/pylatro/shop.py:138-171` (claim_pack_card)
- Reference: `src/pylatro/shop.py:269-275` (close_pack)
- Test: `tests/test_replay_parity.py`

- [ ] **Step 1: Add `claim_card` and `close_pack` to oracle bridge signatures**

In `oracle_bridge.py`, add to `_SIGNATURES` dict (line 78-91):
```python
"claim_card": ["index"],
"close_pack": [],
```

- [ ] **Step 2: Write a parity test for open_pack + claim_card**

```python
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
    # Mirror same pack actions on Lua side
    lua_snap_pre = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    if lua_snap_pre.get("shop_boosters"):
        lua_raw = bridge.step(lua_raw, "open_pack", index=1)  # 1-indexed
        lua_raw = bridge.step(lua_raw, "claim_card", index=1)
        lua_raw = bridge.step(lua_raw, "close_pack")

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Divergences: {diffs}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_open_pack -v`
Expected: FAIL

- [ ] **Step 4: Implement `oracle.open_pack`, `oracle.claim_card`, `oracle.close_pack` in oracle.lua**

`open_pack(state, index)` must:
1. Pop booster from `state.shop.boosters` at index
2. Deduct cost from `state.dollars`
3. Mark `used_packs` slot as "USED"
4. Generate pack cards based on booster type (Arcana→tarots, Celestial→planets, Spectral→spectrals, Standard→playing cards, Buffoon→jokers)
5. Store in `state.pack = {cards=..., choices_remaining=..., booster_key=...}`

Reference `open_booster_pack()` at `shop.py:174-268` — particularly the RNG seed keys used per pack type (e.g., `"Arcana"`, `"Celestial"`, etc.).

`claim_card(state, index)` must:
1. Pop card from `state.pack.cards` at index
2. Add to appropriate collection based on type
3. Decrement `state.pack.choices_remaining`
4. Clear `state.pack` if no choices remain

`close_pack(state)` must:
1. Clear `state.pack`

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_open_pack -v`
Expected: PASS

- [ ] **Step 6: Run all tests for regressions**

Run: `pytest tests/ -v`

- [ ] **Step 7: Commit**

```bash
git add src/pylatro/upstream/oracle.lua src/pylatro/upstream/oracle_bridge.py tests/test_replay_parity.py
git commit -m "feat: implement oracle.lua open_pack, claim_card, close_pack with parity test"
```

---

### Task 4: Implement oracle.lua `use_consumable` action

The Lua oracle's `use_consumable` is a stub. It must mirror `use_consumable()` in `consumables.py:75+`.

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua:1135-1139`
- Reference: `src/pylatro/consumables.py:75+` (Python use_consumable)
- Reference: `src/pylatro/consumables.py:34-73` (can_use_consumable)
- Test: `tests/test_replay_parity.py`

- [ ] **Step 1: Add `oracle.add_consumable` and `oracle.add_joker` test helpers to oracle.lua**

These test helpers are needed FIRST — both for this task's use_consumable test and for Task 9's joker verification tests. Add them before `return oracle` at the end of oracle.lua.

Add Lua functions for test setup:
```lua
function oracle.add_consumable(state, center_key)
    local center = state.data.centers[center_key]
    table.insert(state.consumables, {
        center_key = center_key,
        set = center.set,
        name = center.name,
        cost = center.cost or 0,
        sell_cost = math.max(1, math.floor((center.cost or 0) / 2)),
    })
    return state
end

function oracle.add_joker(state, center_key)
    local center = state.data.centers[center_key]
    local config = center.config or {}
    table.insert(state.jokers, {
        center_key = center_key,
        name = center.name,
        mult = config.mult or 0,
        x_mult = config.Xmult or 1,
        t_mult = config.t_mult or 0,
        t_chips = config.t_chips or 0,
        extra = config.extra,
        type = config.type,
        h_size = config.h_size or 0,
        d_size = config.d_size or 0,
        effect = config.effect,
        sell_cost = math.max(1, math.floor((center.cost or 0) / 2)),
    })
    -- Apply stat modifiers
    if (config.h_size or 0) ~= 0 then
        state.starting_params.hand_size = state.starting_params.hand_size + config.h_size
        state.current_round.hand_size = state.current_round.hand_size + config.h_size
    end
    if (config.d_size or 0) > 0 then
        state.round_resets.discards = state.round_resets.discards + config.d_size
        state.current_round.discards_left = state.current_round.discards_left + config.d_size
    end
    return state
end
```

Also add `add_consumable` and `add_joker` to `_SIGNATURES` in `oracle_bridge.py`:
```python
"add_consumable": ["center_key"],
"add_joker": ["center_key"],
```

- [ ] **Step 2: Write a parity test for use_consumable**

Test with a planet card (simplest consumable — just levels up a hand). Note: must add the consumable on BOTH sides before using it.

```python
def test_substep_parity_use_consumable():
    """Using a consumable produces identical state in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    from pylatro.flow import start_blind
    from pylatro import add_consumable, use_consumable

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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_use_consumable -v`
Expected: FAIL — Lua oracle stub doesn't actually use the consumable.

- [ ] **Step 4: Implement `oracle.use_consumable` in oracle.lua**

Start with planet cards (level up a poker hand):
```lua
function oracle.use_consumable(state, index, targets)
    G.GAME = state
    local cons = table.remove(state.consumables, index)
    if not cons then return state end

    local center = state.data.centers[cons.center_key]
    if not center then return state end

    -- Planet cards: level up the associated hand
    if center.set == "Planet" then
        local hand_name = center.config and center.config.hand_type
        if hand_name and state.hands[hand_name] then
            state.hands[hand_name].level = state.hands[hand_name].level + 1
        end
    end

    -- Track usage
    state.consumeable_usage_total = (state.consumeable_usage_total or 0) + 1
    return state
end
```

This covers the simplest case. Tarot and Spectral cards require more complex logic (card modification, creation, destruction) and should be implemented incrementally as parity tests reveal what's needed.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_use_consumable -v`
Expected: PASS

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`

- [ ] **Step 7: Commit**

```bash
git add src/pylatro/upstream/oracle.lua src/pylatro/upstream/oracle_bridge.py tests/test_replay_parity.py
git commit -m "feat: implement oracle.lua use_consumable (planets) with test helpers"
```

---

### Task 5: Implement oracle.lua `finish_shop` with end-of-shop effects

The current `finish_shop` is a no-op stub. It must match `finish_shop()` in `shop.py:277+` and `apply_end_shop()` in `runtime.py` (Perkeo consumable duplication, etc.).

**Files:**
- Modify: `src/pylatro/upstream/oracle.lua:1129-1133`
- Reference: `src/pylatro/shop.py:277+`
- Reference: `src/pylatro/runtime.py` (apply_end_shop)
- Test: `tests/test_replay_parity.py`

- [ ] **Step 1: Write a parity test for finish_shop**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_replay_parity.py::test_substep_parity_finish_shop -v`
Expected: FAIL (or PASS if finish_shop has no side effects without jokers — investigate)

- [ ] **Step 3: Implement `oracle.finish_shop` in oracle.lua**

Must mirror `finish_shop()` in `shop.py:277+`:
1. Clear shop state
2. Apply end-of-shop joker hooks (Perkeo: duplicate consumable with negative edition)
3. Reset reroll cost

- [ ] **Step 4: Run tests**

Run: `pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git add src/pylatro/upstream/oracle.lua tests/test_replay_parity.py
git commit -m "feat: implement oracle.lua finish_shop with parity test"
```

---

### Task 6: Enhance `auto_action` bot to exercise shop/pack/consumable paths

Currently the bot calls `finish_shop` immediately after defeating a blind. Enhance it to: populate shop, buy first affordable card, open first affordable pack, claim first card, use consumables, then finish shop.

**Files:**
- Modify: `tests/test_replay_parity.py:207-224` (auto_action)
- Modify: `tests/test_replay_parity.py:241-259` (_execute_python_action)
- Modify: `tests/test_replay_parity.py:262-266` (_convert_kwargs_for_lua)
- Reference: `src/pylatro/shop.py` (all shop APIs)
- Reference: `src/pylatro/consumables.py:34-73` (can_use_consumable)

- [ ] **Step 1: Plan the enhanced bot state machine**

The bot needs a shop phase after defeating a blind. The action sequence should be:

1. After `defeat_blind` → `populate_shop`
2. If consumables and first is usable → `use_consumable` (index 0, targets = first N hand cards by index)
3. If shop has affordable cards → `buy_card` (index 0)
4. If shop has affordable boosters → `open_pack` (index 0)
5. If pack is open and has cards → `claim_card` (index 0)
6. If pack still open → `close_pack`
7. `finish_shop`

- [ ] **Step 2: Add shop-phase detection helpers**

```python
def _in_shop_phase(state) -> bool:
    """True if a blind was just defeated and we haven't finished shop yet."""
    return any(v == "Defeated" for v in state.round_resets.blind_states.values()) and not _next_blind_to_start(state)

def _shop_populated(state) -> bool:
    """True if the shop has been populated this phase."""
    return bool(state.shop.cards or state.shop.boosters or state.shop.vouchers)
```

- [ ] **Step 3: Rewrite `auto_action` with shop phase**

```python
def auto_action(state) -> tuple[str, dict]:
    """Deterministic bot: play, discard, shop, pack, consumables."""
    # Start next blind if available
    next_blind = _next_blind_to_start(state)
    if next_blind and not _round_active(state):
        return ("start_blind", {"blind_type": next_blind})

    # In a round: discard once then play hands
    if _round_active(state) and state.current_round.hands_left > 0 and state.hand_cards:
        if state.current_round.discards_left > 0 and state.current_round.discards_used == 0:
            return ("discard", {"cards": list(range(min(2, len(state.hand_cards))))})
        return ("play_hand", {"cards": list(range(min(5, len(state.hand_cards))))})

    # Round complete: defeat blind
    if _round_active(state):
        return ("defeat_blind", {})

    # Shop phase
    if not _shop_populated(state):
        return ("populate_shop", {})

    # Handle open pack first
    if state.pack and state.pack.cards:
        return ("claim_card", {"index": 0})
    if state.pack:
        return ("close_pack", {})

    # Use first usable consumable
    if state.consumables:
        from pylatro.consumables import can_use_consumable
        cons = state.consumables[0]
        targets = list(range(min(3, len(state.hand_cards)))) if state.hand_cards else []
        if can_use_consumable(state, cons, hand_targets=targets):
            return ("use_consumable", {"index": 0, "targets": targets})

    # Buy first affordable card
    if state.shop.cards and state.dollars >= state.shop.cards[0].cost:
        return ("buy_card", {"index": 0})

    # Open first affordable booster
    if state.shop.boosters and state.dollars >= state.shop.boosters[0].cost:
        return ("open_pack", {"index": 0})

    return ("finish_shop", {})
```

- [ ] **Step 4: Update `_execute_python_action` with new actions**

```python
def _execute_python_action(state, action: str, kwargs: dict):
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
```

- [ ] **Step 5: Update `_convert_kwargs_for_lua` for new actions**

```python
def _convert_kwargs_for_lua(action: str, kwargs: dict) -> dict:
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
```

- [ ] **Step 6: Run ante 1 parity test with enhanced bot**

Run: `pytest tests/test_replay_parity.py::test_ante_parity[1] -v`
Expected: May FAIL if oracle.lua shop implementations have divergences. Debug and fix.

- [ ] **Step 7: Iteratively fix divergences**

For each divergence found:
1. Identify the step and action where snapshots diverge
2. Compare Python vs Lua logic for that action
3. Fix the Lua implementation
4. Re-run test

- [ ] **Step 8: Run full ante 1-8 parity**

Run: `pytest tests/test_replay_parity.py -k "test_ante_parity" -v`
Expected: All 8 ante tests pass.

- [ ] **Step 9: Commit**

```bash
git add tests/test_replay_parity.py
git commit -m "feat: enhance auto_action bot with shop/pack/consumable paths"
```

---

### Task 7: Expand parity tests to all 3 seeds and deck variants

Add parametrized tests for seeds BBBBBBBB and 12345678, plus a non-default deck variant.

**Files:**
- Modify: `tests/test_replay_parity.py`

- [ ] **Step 1: Parametrize ante parity tests with all seeds**

```python
@pytest.mark.parametrize("seed", ["AAAAAAAA", "BBBBBBBB", "12345678"])
@pytest.mark.parametrize("target_ante", range(1, 9))
def test_ante_parity(seed, target_ante):
    """Full ante cycle produces identical state in Python and Lua."""
    data = load_game_data()
    py_state = create_run_state(seed, data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run(seed)
    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante)
```

- [ ] **Step 2: Add deck variant parity test**

```python
@pytest.mark.slow
@pytest.mark.parametrize("deck_key", ["b_red", "b_blue", "b_yellow", "b_green"])
def test_deck_variant_parity(deck_key):
    """Non-default deck variant through ante 4."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data, deck_key=deck_key)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA", deck_key=deck_key)
    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante=4)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_replay_parity.py -k "test_ante_parity or test_deck_variant" -v --timeout=120`
Expected: All pass (fix divergences as found).

- [ ] **Step 4: Commit**

```bash
git add tests/test_replay_parity.py
git commit -m "feat: expand parity tests to 3 seeds and deck variants"
```

---

### Task 8: Add endless mode parity test (ante 9-12+)

Verify boss pool recycling, scaling, and showdown mechanics for endless mode.

**Files:**
- Modify: `tests/test_replay_parity.py`

- [ ] **Step 1: Add endless mode parity test**

```python
@pytest.mark.slow
def test_endless_mode_parity():
    """Endless mode (ante 9-12) produces identical state."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante=12, max_steps=2000)
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_replay_parity.py::test_endless_mode_parity -v --timeout=300`
Expected: PASS (or reveals endless-mode divergences to fix).

- [ ] **Step 3: Fix any endless-specific divergences**

Common issues:
- Boss pool recycling logic mismatch
- Ante scaling formula differences for antes 9+
- Showdown blind selection

- [ ] **Step 4: Commit**

```bash
git add tests/test_replay_parity.py src/pylatro/upstream/oracle.lua
git commit -m "feat: add endless mode parity test (ante 9-12)"
```

---

### Task 9: Joker verification — targeted tests for generic code paths

**Depends on:** Task 4 (for `oracle.add_joker` and `oracle.add_consumable` bridge actions).

Verify that data-driven joker code paths produce correct results for the 22 jokers that rely on generic handling.

**Files:**
- Create: `tests/test_joker_generic_parity.py`
- Reference: `src/pylatro/scoring.py:581-583, 636-641` (generic scoring paths)
- Reference: `src/pylatro/instances.py:133-138` (stat modifier paths)

- [ ] **Step 1: Write targeted scoring tests for suit-mult jokers**

```python
"""Targeted parity tests for jokers handled by generic data-driven code."""
from pylatro import create_run_state, load_game_data, add_joker, start_blind, play_cards
from pylatro.upstream.oracle_bridge import (
    OracleBridge, diff_snapshots, snapshot_from_run_state, snapshot_from_lua_state,
)
import pytest

SUIT_MULT_JOKERS = [
    ("j_greedy_joker", "Greedy Joker"),
    ("j_lusty_joker", "Lusty Joker"),
    ("j_wrathful_joker", "Wrathful Joker"),
    ("j_gluttenous_joker", "Gluttonous Joker"),
]

@pytest.mark.parametrize("joker_key,joker_name", SUIT_MULT_JOKERS)
def test_suit_mult_joker_parity(joker_key, joker_name):
    """Suit mult joker scoring matches Lua oracle."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    add_joker(py_state, joker_key)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "add_joker", center_key=joker_key)

    # Play first 5 cards
    play_cards(py_state, [0, 1, 2, 3, 4])
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"{joker_name} divergence: {diffs}"
```

- [ ] **Step 2: Write tests for hand-type mult/chips jokers**

```python
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

@pytest.mark.parametrize("joker_key,joker_name", HAND_TYPE_JOKERS)
def test_hand_type_joker_parity(joker_key, joker_name):
    """Hand type joker scoring matches Lua oracle."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    add_joker(py_state, joker_key)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "add_joker", center_key=joker_key)

    play_cards(py_state, [0, 1, 2, 3, 4])
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"{joker_name} divergence: {diffs}"
```

- [ ] **Step 3: Write tests for x_mult jokers**

```python
XMULT_JOKERS = [
    ("j_duo", "The Duo"),
    ("j_trio", "The Trio"),
    ("j_family", "The Family"),
    ("j_order", "The Order"),
    ("j_tribe", "The Tribe"),
]

@pytest.mark.parametrize("joker_key,joker_name", XMULT_JOKERS)
def test_xmult_joker_parity(joker_key, joker_name):
    """X_mult joker scoring matches Lua oracle."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")
    add_joker(py_state, joker_key)

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")
    lua_raw = bridge.step(lua_raw, "add_joker", center_key=joker_key)

    play_cards(py_state, [0, 1, 2, 3, 4])
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"{joker_name} divergence: {diffs}"
```

- [ ] **Step 4: Write tests for stat modifier jokers**

```python
STAT_MOD_JOKERS = [
    ("j_juggler", "Juggler", {"h_size_delta": 1}),
    ("j_drunkard", "Drunkard", {"d_size_delta": 1}),
    ("j_merry_andy", "Merry Andy", {"d_size_delta": 3, "h_size_delta": -1}),
]

@pytest.mark.parametrize("joker_key,joker_name,expected", STAT_MOD_JOKERS)
def test_stat_modifier_joker_parity(joker_key, joker_name, expected):
    """Stat modifier jokers apply correctly in both engines."""
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    start_blind(py_state, "Small")

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Small")

    # Add joker on both sides
    add_joker(py_state, joker_key)
    lua_raw = bridge.step(lua_raw, "add_joker", center_key=joker_key)

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"{joker_name} divergence: {diffs}"
```

- [ ] **Step 5: Add snapshot fields for hand_size and discards_left**

If not already present in snapshots, add `hand_size` and `discards_left` to both snapshot functions to verify stat modifier effects.

- [ ] **Step 6: Run all joker tests**

Run: `pytest tests/test_joker_generic_parity.py -v`
Expected: All pass. Fix any divergences found.

- [ ] **Step 7: Commit**

```bash
git add tests/test_joker_generic_parity.py
git commit -m "feat: add targeted parity tests for 22 generic-path jokers"
```

---

### Task 10: Boss blind verification

**Depends on:** Task 6 (enhanced bot that exercises all game paths).

Verify all 28 boss blind effects produce identical state in Python and Lua. The implementations exist but have not been oracle-verified.

**Files:**
- Create: `tests/test_boss_blind_parity.py`
- Reference: `src/pylatro/blind.py`
- Reference: `src/pylatro/flow.py` (blind effects during play/draw)
- Reference: `src/pylatro/upstream/oracle.lua` (start_blind boss effects)

- [ ] **Step 1: Identify seeds that encounter each boss**

Run a research pass: for each of the 3 test seeds, log which bosses appear at each ante. This tells us which bosses are naturally covered and which need additional seeds.

- [ ] **Step 2: Write parametrized boss blind parity tests**

```python
"""Boss blind parity tests: verify all 28 boss effects match Lua oracle."""
import pytest
from pylatro import create_run_state, load_game_data, start_blind, play_cards
from pylatro.upstream.oracle_bridge import (
    OracleBridge, diff_snapshots, snapshot_from_run_state, snapshot_from_lua_state,
)

# Map boss keys to seeds and antes where they appear
# (populate this from Step 1 research)
BOSS_ENCOUNTERS = [
    # ("bl_hook", "AAAAAAAA", 1),
    # ("bl_tooth", "BBBBBBBB", 2),
    # ... etc
]

@pytest.mark.parametrize("boss_key,seed,ante", BOSS_ENCOUNTERS)
def test_boss_blind_parity(boss_key, seed, ante):
    """Boss blind effect produces identical state after start_blind + play."""
    data = load_game_data()
    py_state = create_run_state(seed, data=data)
    bridge = OracleBridge()
    lua_raw = bridge.create_run(seed)

    # Drive both engines to the target ante's Boss blind
    _run_bot_until_ante(py_state, bridge, lua_raw, target_ante=ante)

    # Now at boss blind — start it
    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # Compare state after blind start (captures debuffs, hand size changes, etc.)
    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Boss {boss_key} divergence after start: {diffs}"

    # Play one hand to test scoring effects
    play_cards(py_state, list(range(min(5, len(py_state.hand_cards)))))
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Boss {boss_key} divergence after play: {diffs}"
```

- [ ] **Step 3: Run tests and fix divergences**

Run: `pytest tests/test_boss_blind_parity.py -v`

For each failing boss:
1. Compare Python vs Lua blind effect logic
2. Fix the divergent implementation
3. Re-run

- [ ] **Step 4: Ensure all 28 bosses are covered**

If some bosses are not naturally encountered by the 3 seeds, either:
- Add more test seeds that encounter them, OR
- Write direct tests that set `round_resets.blind_choices.Boss` to force a specific boss

- [ ] **Step 5: Commit**

```bash
git add tests/test_boss_blind_parity.py src/pylatro/blind.py src/pylatro/flow.py
git commit -m "feat: boss blind parity verification for all 28 bosses"
```

---

### Task 11: The Serpent and round-resolution verification

Verify The Serpent's 3-card draw limit and round-resolution edge cases (loss detection, saved-from-loss, bankruptcy).

**Files:**
- Create: `tests/test_edge_case_parity.py`
- Reference: `src/pylatro/flow.py:51-54` (Serpent draw logic)
- Reference: `src/pylatro/runtime.py` (apply_end_of_round, Mr. Bones)

- [ ] **Step 1: Write The Serpent parity test**

```python
"""Edge case parity tests: Serpent draw, round resolution."""
import pytest
from pylatro import create_run_state, load_game_data, start_blind, play_cards, discard_cards
from pylatro.upstream.oracle_bridge import (
    OracleBridge, diff_snapshots, snapshot_from_run_state, snapshot_from_lua_state,
)

def test_serpent_draw_limit():
    """The Serpent limits draws to 3 cards after first action."""
    # Find a seed/ante where The Serpent is the boss, or force it
    data = load_game_data()
    py_state = create_run_state("AAAAAAAA", data=data)
    # Force The Serpent as boss blind
    py_state.round_resets.blind_choices["Boss"] = "bl_serpent"

    bridge = OracleBridge()
    lua_raw = bridge.create_run("AAAAAAAA")
    # Force on Lua side too
    lua_snap = bridge.snapshot(lua_raw)
    # Set boss choice directly on Lua state
    lua_raw.round_resets.blind_choices.Boss = "bl_serpent"

    # Drive to boss blind
    # ... (play through Small and Big first)
    # Then start Boss
    start_blind(py_state, "Boss")
    lua_raw = bridge.step(lua_raw, "start_blind", blind_type="Boss")

    # After first play, hand should only refill by 3
    play_cards(py_state, [0, 1, 2, 3, 4])
    lua_raw = bridge.step(lua_raw, "play_hand", card_indices=[1, 2, 3, 4, 5])

    py_snap = snapshot_from_run_state(py_state)
    lua_snap = snapshot_from_lua_state(bridge.snapshot(lua_raw))
    # Verify hand size is <= 3 (Serpent effect)
    assert py_snap["hand_cards_count"] <= 3, f"Serpent not limiting draws: {py_snap['hand_cards_count']}"
    diffs = diff_snapshots(py_snap, lua_snap)
    assert diffs == [], f"Serpent divergence: {diffs}"
```

- [ ] **Step 2: Write round-resolution test (loss detection)**

Test that when a player fails to beat a blind (runs out of hands), both engines handle it identically. This may require extending the snapshot to include a `game_over` or `lost` field.

- [ ] **Step 3: Run tests and fix divergences**

Run: `pytest tests/test_edge_case_parity.py -v`

- [ ] **Step 4: Commit**

```bash
git add tests/test_edge_case_parity.py src/pylatro/flow.py src/pylatro/runtime.py
git commit -m "feat: Serpent and round-resolution parity verification"
```

---

### Task 12: Consumable branch audit

Enumerate all upstream consumable branches and cross-reference against Python implementation.

**Files:**
- Reference: `vendor/balatro_lua/card.lua` (Card:use_consumeable, Card:can_use_consumeable)
- Reference: `src/pylatro/consumables.py`
- Create: `docs/superpowers/audits/consumable-audit.md` (tracking document)

- [ ] **Step 1: Extract all consumable branches from upstream Lua**

Search `vendor/balatro_lua/card.lua` for all `Card:use_consumeable` branches. List every consumable key and what it does.

Run (research only — read files, don't write code):
- Read `vendor/balatro_lua/card.lua` and find all consumable handling branches
- Produce a list of every consumable key with its effect

- [ ] **Step 2: Cross-reference against `consumables.py`**

For each consumable in the upstream list, verify it has a matching branch in `consumables.py`. Mark as:
- MATCHED: Python branch exists and logic appears correct
- MISSING: No Python branch
- DIVERGENT: Python branch exists but logic differs

- [ ] **Step 3: Write audit document**

Save to `docs/superpowers/audits/consumable-audit.md` with the full comparison table.

- [ ] **Step 4: Fix any MISSING or DIVERGENT branches**

Implement missing consumable branches in `consumables.py`, fix divergent ones.

- [ ] **Step 5: Add parity tests for any fixed consumables**

For each consumable that was fixed, add a targeted parity test similar to Task 9's pattern.

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -v`

- [ ] **Step 7: Commit**

```bash
git add src/pylatro/consumables.py docs/superpowers/audits/consumable-audit.md tests/
git commit -m "audit: consumable branch parity verification and fixes"
```

---

### Task 13: Voucher parity audit

Enumerate all upstream voucher effects and cross-reference against Python implementation.

**Files:**
- Reference: `vendor/balatro_lua/` (voucher handling code)
- Reference: `src/pylatro/_helpers.py`, `src/pylatro/shop.py`
- Create: `docs/superpowers/audits/voucher-audit.md`

- [ ] **Step 1: Extract all voucher effects from upstream Lua**

Search vendor Lua for all voucher-related logic. List every voucher key and its effect.

- [ ] **Step 2: Cross-reference against Python implementation**

Check `_helpers.py` and `shop.py` for matching voucher effects. Mark MATCHED/MISSING/DIVERGENT.

- [ ] **Step 3: Write audit document**

Save to `docs/superpowers/audits/voucher-audit.md`.

- [ ] **Step 4: Fix any gaps**

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -v`

- [ ] **Step 6: Commit**

```bash
git add src/pylatro/_helpers.py src/pylatro/shop.py docs/superpowers/audits/voucher-audit.md
git commit -m "audit: voucher parity verification and fixes"
```

---

### Task 14: Final full-run parity validation

Run comprehensive end-to-end parity tests with the enhanced bot across all seeds.

**Files:**
- Modify: `tests/test_replay_parity.py`

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --timeout=300`
Expected: All tests pass.

- [ ] **Step 2: Run slow parity tests**

Run: `pytest tests/ -v -m slow --timeout=600`
Expected: All 3-seed full-run tests pass through ante 10+.

- [ ] **Step 3: Verify test count**

Confirm test count has increased from 31 to 50+ with new parity tests.

- [ ] **Step 4: Fix any remaining divergences**

Follow the divergence fix protocol from the spec for each failure.

- [ ] **Step 5: Final commit**

```bash
git add tests/ src/pylatro/upstream/oracle.lua src/pylatro/upstream/oracle_bridge.py
git commit -m "feat: complete parity verification — all seeds, all phases"
```
