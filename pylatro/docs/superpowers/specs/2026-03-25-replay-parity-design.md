# Replay Parity Test System Design

## Goal

Verify deterministic parity between pylatro (Python) and Balatro 1.0.1o (Lua) by driving both engines through identical seeded action sequences and comparing full state snapshots after every action. Coverage: ante 1-8 and endless mode (antes 9-10).

## Architecture

Three components:

```
seed + action_sequence
    |
    ├─→ Python engine (pylatro API) → Python state snapshots
    |
    ├─→ Lua oracle (via lupa bridge) → Lua state snapshots
    |
    └─→ Compare snapshots at each step → Pass/Fail
```

### 1. Lua Oracle (`src/pylatro/upstream/oracle.lua`)

A standalone Lua module that implements the key game substeps as pure functions. No LOVE2D, no classes, no event system. Each function takes a state table, mutates it, and returns it.

**Data loading:** The oracle loads game data tables (blinds, centers, cards, hands) from the decompiled Lua source files. The vendor path is passed from Python as a global `VENDOR_PATH` before loading the oracle. The oracle sets up minimal global stubs (`G = {}`, no-op `localize`, `HEX` identity function) before loading data tables.

**RNG:** Uses the original `pseudohash`/`pseudoseed`/`math.random` chain from the decompiled source.

**Prerequisite — RNG bridge validation:** Before building the oracle, verify that `lupa`'s `math.random`/`math.randomseed` output matches the LuaJIT bridge for test seeds. The existing `test_rng.py` already validates Python RNG against Lua. If `lupa`'s math.random diverges from Balatro's LuaJIT, the oracle must use the ctypes LuaJIT bridge instead. This is a go/no-go gate for the entire approach.

**Event queue ordering risk:** The original Lua code uses `G.E_MANAGER:add_event(...)` extensively (~435 occurrences). These encode sequencing dependencies that affect when mutations happen relative to each other, which affects RNG call order. The oracle replaces these with synchronous inline execution. To mitigate ordering bugs:
- Phase 1 (substep parity) validates individual actions in isolation, where event ordering is simpler.
- Phase 2 (ante/run parity) validates full sequences. Ordering bugs surface as RNG divergence in the snapshot diff.
- When ordering issues are found, the oracle function is audited against the original event sequence and corrected.

**Functions:**

| Function | Extracted From | Purpose |
|---|---|---|
| `oracle.create_run(seed, stake, deck_key)` | `game.lua` init, `common_events.lua` | Build initial run state |
| `oracle.start_blind(state, blind_type)` | `blind.lua:set_blind`, `state_events.lua:new_round` | Set blind, apply effects, shuffle deck, draw opening hand |
| `oracle.play_hand(state, card_indices)` | `state_events.lua:evaluate_play` | Move cards, score hand, resolve all joker/card/blind effects |
| `oracle.discard(state, card_indices)` | `state_events.lua:discard_cards_from_highlighted` | Discard with joker hooks, draw replacement cards |
| `oracle.cash_out(state)` | `state_events.lua:end_round` | Rewards, interest, ante progression, round reset |
| `oracle.populate_shop(state)` | `common_events.lua` pool functions | Generate shop cards, vouchers, boosters |
| `oracle.buy_card(state, index)` | `button_callbacks.lua` | Purchase card from shop, add to inventory |
| `oracle.use_consumable(state, index, targets)` | `card.lua:use_consumeable` | Apply tarot/planet/spectral effect |
| `oracle.open_pack(state, index)` | `common_events.lua` | Generate booster pack contents |
| `oracle.reroll_shop(state)` | `button_callbacks.lua` | Reroll shop with cost |
| `oracle.skip_blind(state)` | `button_callbacks.lua` | Skip blind, apply tag reward |

**Extraction principle:** Each function stays structurally close to the original Lua source. The only changes are:
- Remove `G.E_MANAGER:add_event(...)` wrappers (execute logic inline, preserving the order of state mutations within each event)
- Replace `ease_dollars(n)` / `ease_chips(n)` / etc. with direct state mutations
- Replace `G.GAME.xxx` references with `state.xxx`
- Remove UI/sound/sprite calls

### 2. Oracle Bridge (`src/pylatro/upstream/oracle_bridge.py`)

Python class that mediates between `RunState` and the Lua oracle.

**Responsibilities:**

- **State serialization:** Convert `RunState` → Lua table for oracle input. Convert Lua table → Python dict for snapshot comparison.
- **Oracle lifecycle:** Load `oracle.lua` once into a `lupa.LuaRuntime` (or LuaJIT bridge if RNG validation requires it), cache oracle function references.
- **Action dispatch:** A `step(state_table, action_name, **kwargs)` method that calls the corresponding oracle function.

**State serialization strategy:**

Recursive conversion of Python dataclasses to Lua tables:
- Dataclass fields → Lua table keys (recursively applied to nested dataclasses like `CurrentRound`, `RoundResets`, `ShopState`, `StartingParams`)
- Enums → `.value` strings
- Python lists → 1-indexed Lua arrays
- Python dicts → Lua tables
- `None` → `nil`
- `bool` → Lua boolean

Special cases:
- `PlayingCard` → `{front_key, suit, rank, center_key, edition_key, seal, perma_bonus, debuff, destroyed, shattered, played_this_ante, discarded, face_down, forced_selection, times_played}`
- `JokerInstance` → all 19 fields including `extra` (which may be int, float, dict, or list)
- `ConsumableInstance` → `{center_key, edition, extra_value, sell_cost}`
- `PseudorandomState` → `{seed}` (oracle maintains its own RNG from the seed)

**RNG independence:** The oracle and Python engine each maintain their own RNG state from the same seed. Parity is verified by comparing outputs. If they diverge, the snapshot diff pinpoints where.

### 3. Replay Test Runner (`tests/test_replay_parity.py`)

Pytest tests that drive both engines through identical action sequences.

**Canonical action names** (used consistently in traces, bot, and oracle):

| Action | Python API | Oracle Function |
|---|---|---|
| `start_blind` | `start_blind()` | `oracle.start_blind()` |
| `play_hand` | `play_cards()` | `oracle.play_hand()` |
| `discard` | `discard_cards()` | `oracle.discard()` |
| `cash_out` | `cash_out()` | `oracle.cash_out()` |
| `populate_shop` | `populate_shop()` | `oracle.populate_shop()` |
| `buy_card` | `buy_shop_card()` | `oracle.buy_card()` |
| `use_consumable` | `use_consumable()` | `oracle.use_consumable()` |
| `open_pack` | `open_booster_pack()` | `oracle.open_pack()` |
| `reroll_shop` | `reroll_shop()` | `oracle.reroll_shop()` |
| `skip_blind` | `skip_blind()` | `oracle.skip_blind()` |
| `finish_shop` | `finish_shop()` | `oracle.finish_shop()` |

**Action traces** are lists of tuples:

```python
trace = [
    ("start_blind", {"blind_type": "Small"}),
    ("play_hand", {"cards": [0, 1, 2, 3, 4]}),
    ("cash_out", {}),
    ("populate_shop", {}),
    ("finish_shop", {}),
    ("start_blind", {"blind_type": "Big"}),
    # ...
]
```

**Three test tiers:**

1. **Substep parity** (`test_substep_parity_*`) -- Individual actions in isolation: score one hand, do one discard, one shop reroll. Fast, pinpoints bugs. ~20 tests covering each action type.

2. **Ante parity** (`test_ante_parity_*`) -- Full ante cycles (blind select → play/discard loop → cash out → shop → next blind). One test per ante for seed `"AAAAAAAA"`. 8 tests.

3. **Full run parity** (`test_full_run_parity_*`) -- Complete runs from ante 1 through 8 and into endless (antes 9-10). Multiple seeds: `"AAAAAAAA"`, `"BBBBBBBB"`, `"12345678"`. Marked `@pytest.mark.slow`.

**Bot strategy** for deterministic action generation:

```python
def auto_action(state) -> tuple[str, dict]:
    """Pick the next action deterministically based on game phase."""
    # Blind select phase
    if _in_blind_select(state):
        return ("start_blind", {"blind_type": _next_blind_type(state)})

    # Playing phase
    if state.current_round.hands_left > 0 and state.hand_cards:
        # Discard once if we have discards and hand looks weak
        if state.current_round.discards_left > 0 and state.current_round.discards_used == 0:
            return ("discard", {"cards": list(range(min(2, len(state.hand_cards))))})
        return ("play_hand", {"cards": list(range(min(5, len(state.hand_cards))))})

    # Round over — cash out
    if _round_complete(state):
        return ("cash_out", {})

    # Shop phase
    if state.pack:
        return ("close_pack", {})  # skip pack contents
    if state.shop.cards and _can_afford_first(state):
        return ("buy_card", {"index": 0})
    if state.consumables and _can_use_first(state):
        return ("use_consumable", {"index": 0, "targets": []})
    return ("finish_shop", {})
```

The same bot logic drives both Python and Lua. The bot covers all game phases: blind selection, play/discard loop, cash out, and shop. It uses a simple strategy (play first 5, discard first 2 once, buy first affordable card, skip packs) to ensure runs progress through ante 8+ without getting stuck. Game-over states are detected and end the trace.

## Snapshot Comparison

**Fields compared after each action:**

- `dollars`, `ante`, `round`, `hands_played`
- `hands_left`, `discards_left`
- `deck_cards` (count + ordered front_keys + center_keys + suits)
- `draw_pile` (count + ordered front_keys)
- `hand_cards` (count + ordered front_keys)
- `discard_pile` (count)
- `jokers` (ordered center_keys + mult/x_mult/extra state)
- `consumables` (ordered center_keys)
- `hands` (all 12 hand levels + played counts)
- Blind state (name, chips, disabled, triggered)
- Shop state (card keys, voucher keys)
- RNG verification (next pseudorandom draw matches)

**Comparison function:** `diff_snapshots(python_snap, lua_snap)` returns a list of `(path, python_value, lua_value)` divergences. Card lists are compared positionally (order matters for RNG and scoring). Floats use `math.isclose(rel_tol=1e-9)`.

**Divergence output format:**

```
DIVERGENCE at step 5 (play_hand):
  jokers[1].x_mult: python=2.5, lua=3.0
  dollars: python=12, lua=11
```

## Failure Modes

- **Lua runtime error in oracle:** Surfaces with full Lua traceback. Test fails with `OracleError` containing the traceback and the step that caused it.
- **First divergence aborts:** When a snapshot diff finds divergences, the trace stops at that step. The test reports the step number, action name, and all diverged fields. Continuing past a divergence produces cascading noise.
- **Vendor file loading failure:** Caught at fixture setup (`pytest.skip` if vendor files are missing).
- **RNG bridge validation failure:** Caught as a prerequisite test. If `lupa` RNG doesn't match LuaJIT, all oracle tests are skipped with a clear message.

## Endless Mode Scope

Endless mode testing covers antes 9 and 10 only, verifying that blind scaling, boss selection, and state transitions continue to match after ante 8. This is sufficient to prove the endless loop works without unbounded test runtime.

## Performance

- Substep tests: <1s each, ~20 tests.
- Ante parity tests: ~2-5s each, 8 tests.
- Full run tests: ~10-30s each, 3 seeds. Marked `@pytest.mark.slow`.
- Total expected runtime: <2 minutes for full suite.

## Test Seeds

| Seed | Purpose |
|---|---|
| `"AAAAAAAA"` | Canonical test seed (used by all existing tests) |
| `"BBBBBBBB"` | Alternate seed for coverage diversity |
| `"12345678"` | Numeric seed variant |

## File Layout

```
src/pylatro/upstream/
    oracle.lua              # Self-contained Lua oracle
    oracle_bridge.py        # Python ↔ Lua bridge
tests/
    test_replay_parity.py   # All replay parity tests
```

## Running

```bash
# Full suite
uv run pytest tests/test_replay_parity.py -q

# Skip slow full-run tests
uv run pytest tests/test_replay_parity.py -q -m "not slow"
```
