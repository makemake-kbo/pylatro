# Joker Porting & Parity Completion Design

**Date:** 2026-03-25
**Status:** Draft

## Overview

Complete pylatro's Balatro parity by porting all 76 remaining joker effects and verifying existing systems (boss blinds, endless mode, shop/consumables) against the Lua oracle. Work is split into two parallel streams to maximize throughput.

## Current State

- **Jokers:** 76/152 implemented (50%). All simple scoring jokers and many stateful jokers done. Missing: suit-based scoring (Greedy/Lusty/Wrathful/Gluttonous), hand-type bonuses (Jolly/Mad/etc.), stat modifiers (Four Fingers, Shortcut, Smeared), scaling jokers (Yorick, Ramen), and card creation jokers (Certificate, DNA, Sixth Sense).
- **Boss blinds:** 28/28 implemented. Need oracle verification.
- **Shop/Consumables:** Fully implemented. Need bot to exercise paths for parity testing.
- **Endless mode:** Scaling implemented. Showdown mechanics and full parity unverified.
- **The Serpent:** 3-card draw implemented in flow.py:51-54. Needs oracle verification.

## Stream 1: Systematic Joker Porting

### Approach

Port all 76 missing jokers in 5 batches of ~15. Each batch is oracle-validated before the next begins. Jokers are grouped by complexity to catch integration issues early.

### Batch 1 — Simple Scoring Jokers (16 jokers)

Flat chips/mult based on suit, hand type, or static conditions. Ports from `Card:calculate_joker` to `scoring.py`.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| Greedy Joker | +3 mult per Diamond scored | scoring.py |
| Lusty Joker | +3 mult per Heart scored | scoring.py |
| Wrathful Joker | +3 mult per Spade scored | scoring.py |
| Gluttonous Joker | +3 mult per Club scored | scoring.py |
| Jolly Joker | +8 mult if hand contains Pair | scoring.py |
| Mad Joker | +10 mult if hand contains Two Pair | scoring.py |
| Crazy Joker | +12 mult if hand contains Straight | scoring.py |
| Droll Joker | +10 mult if hand contains Flush | scoring.py |
| Sly Joker | +50 chips if hand contains Pair | scoring.py |
| Wily Joker | +100 chips if hand contains Three of a Kind | scoring.py |
| Clever Joker | +80 chips if hand contains Two Pair | scoring.py |
| Devious Joker | +100 chips if hand contains Straight | scoring.py |
| Crafty Joker | +80 chips if hand contains Flush | scoring.py |
| Duo | x2 mult if hand contains Pair | scoring.py |
| Trio | x3 mult if hand contains Three of a Kind | scoring.py |
| Family | x4 mult if hand contains Four of a Kind | scoring.py |
| Order | x3 mult if hand contains Straight | scoring.py |
| Tribe | x2 mult if hand contains Flush | scoring.py |

### Batch 2 — Stat Modifier / Passive Jokers (15 jokers)

Jokers that modify hand size, discards, evaluation rules, or have passive money effects. These touch `scoring.py`, `runtime.py`, `flow.py`, and `instances.py`.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| Juggler | +1 hand size | instances.py (on add) |
| Drunkard | +1 discard per round | instances.py (on add) |
| Troubadour | +2 hand size, -1 hand per round | instances.py (on add) |
| Merry Andy | +3 discards, -1 hand size | instances.py (on add) |
| Four Fingers | Flushes/Straights need only 4 cards | scoring.py (eval change) |
| Shortcut | Straights allow gaps of 1 | scoring.py (eval change) |
| Smeared | Hearts=Diamonds, Clubs=Spades for suit checks | scoring.py (eval change) |
| Splash | Every played card scores | scoring.py (scoring change) |
| Pareidolia | All cards count as face cards | scoring.py (eval change) |
| Oops | All probabilities doubled | runtime.py (probability hook) |
| Credit Card | Go up to -$20 in debt | runtime.py (money check) |
| To the Moon | +$1 interest per $5 held (end of round) | runtime.py |
| Faceless Joker | Earn $5 if 3+ face cards discarded | runtime.py (discard hook) |
| Mail-In Rebate | Earn $5 when target rank discarded | runtime.py (discard hook) |
| Ring Master | No limit on edition types | instances.py |

### Batch 3 — Scaling / Stateful Jokers (15 jokers)

Jokers that track state across hands/rounds and change over time. Touch `scoring.py`, `runtime.py`.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| Hit the Road | x_mult per Jack discarded this round | scoring.py + runtime.py |
| Yorick | x_mult, activates after N discards | scoring.py + runtime.py |
| Glass Joker | x0.75 mult per glass card destroyed | scoring.py |
| Flash Card | +2 mult per shop reroll | scoring.py (already has runtime hook) |
| Red Card | +3 mult per booster skipped | scoring.py (already has runtime hook) |
| Hologram | x0.25 mult per card added to deck | scoring.py (already has runtime hook) |
| Ramen | x2 mult, -0.01 per card discarded, self-destructs at 1 | scoring.py + runtime.py |
| Popcorn | +20 mult, -4 per round, self-destructs at 0 | scoring.py (already has runtime hook) |
| Rocket | +$8 end of round, +$4 per boss defeated | runtime.py (already implemented) |
| Mr. Bones | Prevents death if $5+ held | runtime.py (already has end-of-round hook) |
| Throwback | x_mult per blind skipped this run | scoring.py |
| Ticket (Golden Ticket) | Gold cards earn $4 when scored | scoring.py |
| Burnt Joker | Upgrades discarded poker hand level | runtime.py (discard hook) |
| Satellite | $1 per unique planet used (end of round) | runtime.py (already implemented) |
| Delayed Gratification | $2 per discard remaining (if unused) | runtime.py (already implemented) |

### Batch 4 — Card Creation / Mutation Jokers (15 jokers)

Jokers that create cards, modify the deck, or interact with consumables. Touch `runtime.py`, `flow.py`, `consumables.py`.

| Joker | Effect | Target Module |
|-------|--------|---------------|
| Certificate | Random playing card with seal on blind start | runtime.py (blind start hook) |
| Sixth Sense | Destroy first 6 played, create spectral if only card | runtime.py + flow.py |
| Hallucination | Chance to create tarot when opening booster | runtime.py (pack open hook) |
| Astronomer | All planet cards cost $0 | shop.py (cost modifier) |
| Trading Card | Discard earns $3, destroys random card | runtime.py (discard hook) |
| DNA | First hand of round: copy first played card to hand | flow.py (play hook) |
| Chaos the Clown | Free reroll each shop visit | shop.py (already implemented, needs scoring stub) |
| Perkeo | End of shop: create negative copy of consumable | runtime.py (already implemented) |
| Campfire | x_mult, resets on boss blind | scoring.py + runtime.py (already has reset) |
| Turtle Bean | +5 hand size, -1 per round, self-destructs | instances.py + runtime.py (already has hook) |
| Invisible Joker | After 2 rounds, sell to duplicate joker | runtime.py (already has hooks) |
| Egg | +$3 sell value per round | runtime.py (already implemented) |
| Gift Card | +$1 sell value to all jokers/consumables per round | runtime.py (already implemented) |
| Golden Joker | $4 at end of round | runtime.py (already implemented) |
| Cloud 9 | $1 per 9 in full deck at end of round | runtime.py (already implemented) |

### Batch 5 — Remaining Jokers

Any jokers not covered in batches 1-4. Final cleanup batch.

### Porting Protocol Per Joker

1. Read the Lua branch in `Card:calculate_joker` (vendor/balatro_lua/card.lua)
2. Identify which hook(s) the joker uses (scoring, end_of_round, discard, blind_start, etc.)
3. Port to the appropriate Python function, matching exact values/conditions/RNG calls
4. For jokers needing instance state: update `JokerInstance` fields in `instances.py` if needed
5. Run existing tests to ensure no regressions
6. At end of batch: run oracle parity tests with seeds that trigger these jokers

### Evaluation Rule Changes (Batch 2 — Special Handling)

Four Fingers, Shortcut, Smeared, Splash, and Pareidolia modify how poker hands are *evaluated*, not just scored. These require changes to `evaluate_poker_hand()` and/or the scoring card selection logic in `scoring.py`. Implementation approach:

- Add joker-aware flags to the evaluation context (e.g., `four_fingers=True`)
- Pass these flags from `score_hand()` based on active jokers
- Modify evaluation functions to respect flags
- This matches how upstream Lua checks `next(find_joker("Four Fingers"))` inline

## Stream 2: Parity Verification + Bot Wiring

### Bot Enhancement

Enhance `auto_action()` in `test_replay_parity.py` to exercise all game paths:

```
Current bot:
- play first 5, discard first 2 once, buy first card, skip packs, finish shop

Enhanced bot:
- play first 5, discard first 2 once
- open booster packs and claim first card (instead of skipping)
- use consumables when eligible (select first N hand cards as targets)
- buy first affordable card (joker or consumable)
- sell oldest joker when at capacity and shop has better option
- finish shop
```

The bot must make identical decisions in both Python and Lua. Keep it deterministic and simple — no heuristics, just fixed rules applied to sorted game state.

### Oracle Parity Test Expansion

**Current:** Substep and ante parity for seed "AAAAAAAA" through ante 8.

**Target:**
- All 3 seeds (AAAAAAAA, BBBBBBBB, 12345678) through ante 8
- At least 1 seed through ante 12+ (endless mode verification)
- Snapshot comparison at every substep: start_blind, play_hand, discard, cash_out, shop actions
- New substep types: open_pack, claim_card, use_consumable, sell_joker

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
| scoring.py | Add ~50 joker scoring hooks, modify eval for Four Fingers/etc. | None |
| runtime.py | Add ~20 joker runtime hooks | Fix any divergences found |
| instances.py | Add joker init state for new jokers | None |
| flow.py | Add joker play/discard hooks (DNA, Sixth Sense) | Fix any Serpent/draw divergences |
| shop.py | Astronomer cost modifier | Fix any shop divergences |
| blind.py | None | Fix any boss blind divergences |
| test_replay_parity.py | None | Enhanced bot + new parity tests |

### Merge Conflict Mitigation

- Stream 1 adds new `elif name == "..."` branches in existing match structures
- Stream 2 only modifies existing logic when divergences are found
- Conflict risk is low since they touch different sections of the same files

### Non-Goals

- UI/rendering code
- Network/multiplayer
- Save/load state
- Joker purchase AI (bot buys first affordable, that's it)
- Performance optimization (correctness first)

## Success Criteria

1. All 152 jokers have effect implementations (scoring, runtime, or both as appropriate)
2. All 3 test seeds pass full ante 1-8 parity against Lua oracle
3. At least 1 seed passes ante 9-12+ endless parity
4. Bot exercises shop, consumable, and pack paths in parity tests
5. No regressions on existing 31 tests
6. All boss blinds verified against oracle (at least 1 encounter each)
