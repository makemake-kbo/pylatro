# Parity Completion Design

**Date:** 2026-03-25
**Status:** Draft (Revised after two coverage audits)

## Overview

Complete pylatro's Balatro parity by verifying all existing systems against the Lua oracle and fixing divergences. A coverage audit confirmed that all 135 jokers are already functionally implemented via generic data-driven code paths — the remaining work is verification, not porting.

## Current State (Verified by Audit)

- **Jokers:** 135/135 functionally implemented. Scoring engine uses config-driven fields (`t_mult`, `t_chips`, `Xmult`, `s_mult`, `h_size`, `d_size`, `effect`, `type`) loaded from Lua center definitions. No explicit name-based branches needed for the 22 jokers previously thought missing — they are handled by generic code in `scoring.py` (lines 581-583 for suit mult, 636-641 for type mult/chips/x_mult) and `instances.py` (lines 133-138 for stat modifiers).
- **Boss blinds:** 28/28 implemented. Need oracle verification.
- **Shop/Consumables:** Fully implemented. Need bot to exercise paths for parity testing.
- **Consumable branches:** Not yet audited against every upstream Tarot/Planet/Spectral branch (per AGENT_HANDOFF.md).
- **Voucher interactions:** Not yet verified for parity.
- **Round resolution:** Loss detection partially exists (`Mr. Bones` save-from-loss in `runtime.py` line 374, `game_over` parameter in `apply_end_of_round`). Game-loop integration for loss/bankruptcy may be incomplete.
- **Endless mode:** Scaling implemented. Showdown mechanics and full parity unverified.
- **The Serpent:** 3-card draw implemented in flow.py:51-54. Needs oracle verification.

## Work Structure

With joker porting eliminated, all work falls into a single verification-focused stream with four phases.

### Phase 1: Bot Enhancement + Core Parity Tests

**Goal:** Wire the bot to exercise all game paths, then run full parity tests to surface divergences.

#### Bot Enhancement

Enhance `auto_action()` in `test_replay_parity.py` to exercise all game paths:

```
Current bot:
- play first 5, discard first 2 once, buy first card, skip packs, finish shop

Enhanced bot:
- play first 5, discard first 2 once
- open booster packs and claim first card (instead of skipping)
- use consumables when eligible (target: first N hand cards by index, deterministic)
- buy first affordable card (joker or consumable)
- finish shop
```

**Bot decision rules must be exact:**
- Consumable targeting: select hand cards by index (first 1-5 depending on consumable requirement)
- Pack claiming: always claim card at index 0
- No selling logic (keeps it simple and deterministic)
- All decisions based on sorted/indexed state, no heuristics

The bot must make identical decisions in both Python and Lua.

#### Oracle Parity Test Expansion

**Current:** Substep and ante parity for seed "AAAAAAAA" through ante 8.

**Target:**
- All 3 seeds (AAAAAAAA, BBBBBBBB, 12345678) through ante 8
- At least 1 seed through ante 12+ (endless mode verification)
- At least 1 seed with non-default deck variant (per AGENT_HANDOFF.md recommendation)
- Snapshot comparison at every substep: start_blind, play_hand, discard, cash_out, shop actions
- New substep types: open_pack, claim_card, use_consumable

### Phase 2: Generic Joker Verification

**Goal:** Confirm that the data-driven joker code paths produce correct results for all 135 jokers.

The scoring engine handles jokers generically via config fields. Verify that:
- **Suit Mult jokers** (Greedy, Lusty, Wrathful, Gluttonous): `effect == "Suit Mult"` path in `scoring.py` fires correctly per matching card
- **Hand-type mult/chips jokers** (Jolly, Zany, Mad, Crazy, Droll, Sly, Wily, Clever, Devious, Crafty): `t_mult`/`t_chips` path fires when hand contains target type
- **Hand-type x_mult jokers** (The Duo, The Trio, The Family, The Order, The Tribe): `x_mult` path fires correctly
- **Stat modifier jokers** (Juggler, Drunkard, Merry Andy): `h_size`/`d_size` applied on add and removed on sell/destroy

**Method:** Run seeds that generate these jokers in the shop, have the bot buy them, and compare scoring snapshots against Lua oracle. If no natural seed encounters all of them, add targeted unit tests that add specific jokers and verify scoring output.

### Phase 3: System-Specific Audits

#### Consumable Parity Audit

Per AGENT_HANDOFF.md, the consumable implementation "is not yet audited against every Tarot/Planet/Spectral branch."
- Enumerate all branches in upstream `Card:use_consumeable` and `Card:can_use_consumeable`
- Cross-reference against `consumables.py` implementation
- Fix any missing or divergent branches found

#### Voucher Parity Verification

Per AGENT_HANDOFF.md, voucher interactions need verification:
- Enumerate all voucher effects in upstream code
- Cross-reference against `_helpers.py` and `shop.py` implementations
- Fix any divergences

#### Boss Blind Verification

All 28 bosses have implementations. Verification approach:
- Run enough seeds to encounter each boss at least once
- Compare Python vs Lua snapshots after blind start and after each hand
- Focus on: card debuffs, hand debuffs, draw restrictions, scoring modifiers

### Phase 4: Edge Cases + Endless Mode

#### Endless Mode Verification

Verify against oracle:
- Boss pool recycling (after all bosses used, re-randomize)
- Ante scaling for antes 9-20+
- Showdown blind selection and effects (antes that are multiples of 8 after ante 8)
- Chip requirement overflow behavior (NaN at very high antes)

#### The Serpent Verification

Already implemented. Verify:
- 3-card draw limit applies after first play AND first discard
- Interacts correctly with hand size modifiers
- Disabled when blind is disabled (Chicot, etc.)

#### Round Resolution Gaps

Partially implemented (Mr. Bones save-from-loss exists). Verify and complete:
- Loss detection at game-loop level (player fails to beat blind)
- Integration of `apply_end_of_round(game_over=True)` flow
- Bankruptcy logic
- Address as divergences are found during parity testing

### Divergence Fix Protocol

Applied throughout all phases when parity tests reveal mismatches:

1. Identify exact substep where snapshots diverge
2. Narrow to specific field(s) that differ
3. Read upstream Lua code for that code path
4. Fix Python implementation
5. Re-run parity test
6. Add regression test if the fix was non-obvious

## Architecture Notes

### File Boundaries

| Module | Expected Changes |
|--------|-----------------|
| scoring.py | Fix any joker scoring divergences found |
| runtime.py | Fix any runtime hook divergences |
| instances.py | Fix any joker init divergences |
| flow.py | Fix any Serpent/draw/round-resolution divergences |
| shop.py | Fix any shop/voucher divergences |
| consumables.py | Fix any consumable divergences |
| blind.py | Fix any boss blind divergences |
| test_replay_parity.py | Enhanced bot + new parity tests |

### Non-Goals

- UI/rendering code
- Network/multiplayer
- Save/load state
- Smart joker purchase AI (bot buys first affordable, that's it)
- Performance optimization (correctness first)

## Success Criteria

1. All 135 jokers verified to produce correct scoring output (via oracle parity or targeted tests)
2. All 3 test seeds pass full ante 1-8 parity against Lua oracle with enhanced bot
3. At least 1 seed passes ante 9-12+ endless parity
4. Bot exercises shop, consumable, and pack paths in parity tests
5. Consumable branches audited and verified against upstream
6. Voucher interactions verified against upstream
7. No regressions on existing 31 tests
8. All boss blinds verified against oracle (at least 1 encounter each)
