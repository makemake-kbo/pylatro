# Joker Porting & Parity Completion Design

**Date:** 2026-03-25
**Status:** Draft (Revised after coverage audit)

## Overview

Complete pylatro's Balatro parity by porting the 22 remaining joker effects and verifying existing systems (boss blinds, endless mode, shop/consumables/vouchers) against the Lua oracle. Work is split into two parallel streams.

## Current State (Verified by Audit)

- **Jokers:** 113/135 implemented (83.7%). Missing 22 jokers — all are simple scoring or stat-modifier jokers.
- **Boss blinds:** 28/28 implemented. Need oracle verification.
- **Shop/Consumables:** Fully implemented. Need bot to exercise paths for parity testing.
- **Consumable branches:** Not yet audited against every upstream Tarot/Planet/Spectral branch (per AGENT_HANDOFF.md).
- **Voucher interactions:** Not yet verified for parity.
- **Round resolution:** Missing loss detection, saved-from-loss, bankruptcy logic.
- **Endless mode:** Scaling implemented. Showdown mechanics and full parity unverified.
- **The Serpent:** 3-card draw implemented in flow.py:51-54. Needs oracle verification.

## Stream 1: Port 22 Missing Jokers

### Missing Jokers (Complete List)

All 22 missing jokers fall into two categories: suit-based scoring, hand-type scoring, hand-type x_mult, and stat modifiers.

#### Group A — Suit-Based Scoring (4 jokers)

Per-card scoring hooks in the `individual` card context of `_evaluate_joker` in `scoring.py`. These fire per matching card scored.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| Greedy Joker | +3 mult per Diamond scored | scoring.py (individual_play) |
| Lusty Joker | +3 mult per Heart scored | scoring.py (individual_play) |
| Wrathful Joker | +3 mult per Spade scored | scoring.py (individual_play) |
| Gluttonous Joker | +3 mult per Club scored | scoring.py (individual_play) |

#### Group B — Hand-Type Chips/Mult (9 jokers)

Flat bonuses when the played hand contains a specific poker hand type. Joker-main context in scoring.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| Jolly Joker | +8 mult if hand contains Pair | scoring.py |
| Zany Joker | +12 mult if hand contains Three of a Kind | scoring.py |
| Mad Joker | +10 mult if hand contains Two Pair | scoring.py |
| Crazy Joker | +12 mult if hand contains Straight | scoring.py |
| Droll Joker | +10 mult if hand contains Flush | scoring.py |
| Sly Joker | +50 chips if hand contains Pair | scoring.py |
| Wily Joker | +100 chips if hand contains Three of a Kind | scoring.py |
| Clever Joker | +80 chips if hand contains Two Pair | scoring.py |
| Devious Joker | +100 chips if hand contains Straight | scoring.py |
| Crafty Joker | +80 chips if hand contains Flush | scoring.py |

#### Group C — Hand-Type X_Mult (5 jokers)

Multiplicative bonuses when hand contains a specific type.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| The Duo | x2 mult if hand contains Pair | scoring.py |
| The Trio | x3 mult if hand contains Three of a Kind | scoring.py |
| The Family | x4 mult if hand contains Four of a Kind | scoring.py |
| The Order | x3 mult if hand contains Straight | scoring.py |
| The Tribe | x2 mult if hand contains Flush | scoring.py |

#### Group D — Stat Modifiers (3 jokers)

Modify hand size or discards when added to the joker roster.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| Juggler | +1 hand size | instances.py (on add) |
| Drunkard | +1 discard per round | instances.py (on add) |
| Merry Andy | +3 discards, -1 hand size | instances.py (on add) |

### Porting Protocol

1. Read the Lua branch in `Card:calculate_joker` (vendor/balatro_lua/card.lua)
2. Identify which hook the joker uses (individual_play, joker_main, instance init)
3. Port to the appropriate Python function, matching exact values and conditions
4. For stat modifiers: update `add_joker()` in `instances.py` to apply effects on addition
5. Run existing tests to ensure no regressions
6. Run oracle parity tests with seeds that trigger these jokers

### Implementation Notes

- **Suit-based jokers (Group A)** fire in the `individual` card scoring context, not the joker-main context. They must be added to the per-card evaluation loop in `scoring.py`, matching the upstream `Card:calculate_joker` `individual` context.
- **Hand-type jokers (Groups B & C)** check whether the evaluated hand *contains* the target type (e.g., a Full House contains both Pair and Three of a Kind). Use `next(poker_hands["Pair"])` style checks matching upstream.
- **Stat modifier jokers (Group D)** apply their effects when added and remove them when sold/destroyed. Mirror existing patterns (e.g., Troubadour, Stuntman) in `instances.py`.

## Stream 2: Parity Verification + Bot Wiring

### Bot Enhancement

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

### Oracle Parity Test Expansion

**Current:** Substep and ante parity for seed "AAAAAAAA" through ante 8.

**Target:**
- All 3 seeds (AAAAAAAA, BBBBBBBB, 12345678) through ante 8
- At least 1 seed through ante 12+ (endless mode verification)
- At least 1 seed with non-default deck variant (per AGENT_HANDOFF.md recommendation)
- Snapshot comparison at every substep: start_blind, play_hand, discard, cash_out, shop actions
- New substep types: open_pack, claim_card, use_consumable

### Consumable Parity Audit

Per AGENT_HANDOFF.md, the consumable implementation "is not yet audited against every Tarot/Planet/Spectral branch." Stream 2 should:
- Enumerate all branches in upstream `Card:use_consumeable` and `Card:can_use_consumeable`
- Cross-reference against `consumables.py` implementation
- Fix any missing or divergent branches found

### Voucher Parity Verification

Per AGENT_HANDOFF.md, voucher interactions need verification:
- Enumerate all voucher effects in upstream code
- Cross-reference against `_helpers.py` and `shop.py` implementations
- Fix any divergences

### Boss Blind Verification

All 28 bosses have implementations. Verification approach:
- Run enough seeds to encounter each boss at least once
- Compare Python vs Lua snapshots after blind start and after each hand
- Focus on: card debuffs, hand debuffs, draw restrictions, scoring modifiers

### Endless Mode Verification

Verify against oracle:
- Boss pool recycling (after all bosses used, re-randomize)
- Ante scaling for antes 9-20+
- Showdown blind selection and effects (antes that are multiples of 8 after ante 8)
- Chip requirement overflow behavior (NaN at very high antes)

### The Serpent Verification

Already implemented. Verify:
- 3-card draw limit applies after first play AND first discard
- Interacts correctly with hand size modifiers
- Disabled when blind is disabled (Chicot, etc.)

### Round Resolution Gaps

Per AGENT_HANDOFF.md, these are missing:
- Loss detection (player fails to beat blind)
- Saved-from-loss mechanics (Mr. Bones, etc.)
- Bankruptcy logic
- These should be addressed as divergences are found during parity testing

### Divergence Fix Protocol

1. Identify exact substep where snapshots diverge
2. Narrow to specific field(s) that differ
3. Read upstream Lua code for that code path
4. Fix Python implementation
5. Re-run parity test
6. Add regression test if the fix was non-obvious

## Architecture Notes

### File Boundaries

| Module | Stream 1 Changes | Stream 2 Changes |
|--------|------------------|-------------------|
| scoring.py | Add 18 joker scoring hooks (Groups A, B, C) | None |
| instances.py | Add 3 joker stat modifiers (Group D) | None |
| runtime.py | None | Fix any divergences found |
| flow.py | None | Fix any Serpent/draw/round-resolution divergences |
| shop.py | None | Fix any shop/voucher divergences |
| consumables.py | None | Fix any consumable divergences |
| blind.py | None | Fix any boss blind divergences |
| test_replay_parity.py | None | Enhanced bot + new parity tests |

### Non-Goals

- UI/rendering code
- Network/multiplayer
- Save/load state
- Smart joker purchase AI (bot buys first affordable, that's it)
- Performance optimization (correctness first)

## Success Criteria

1. All 135 jokers have effect implementations (scoring, runtime, instance init, or combination as appropriate)
2. All 3 test seeds pass full ante 1-8 parity against Lua oracle
3. At least 1 seed passes ante 9-12+ endless parity
4. Bot exercises shop, consumable, and pack paths in parity tests
5. Consumable branches audited and verified against upstream
6. Voucher interactions verified against upstream
7. No regressions on existing 31 tests
8. All boss blinds verified against oracle (at least 1 encounter each)
