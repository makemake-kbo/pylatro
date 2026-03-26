# Voucher Parity Audit

**Date:** 2026-03-26
**Source:** `vendor/balatro_lua/card.lua` (`Card:apply_to_run`) + `game.lua` (definitions)
**Target:** `src/pylatro/_helpers.py` (`_apply_voucher_to_run`)

## Methodology

1. Extracted all 32 voucher definitions from `game.lua` (lines 592–624).
2. Extracted all `apply_to_run` effects from `card.lua` (lines 1880–1971).
3. Cross-referenced each voucher against `_apply_voucher_to_run` in `_helpers.py`.
4. Noted runtime effects (checked via `used_vouchers` elsewhere) as separate from apply-to-run effects.

## Results Summary

| Status | Count |
|--------|-------|
| MATCHED | 30 |
| MISSING | 0 |
| DIVERGENT | 0 |

## Voucher-by-Voucher Analysis

### Tier 1 (base vouchers, order 1–31 odd)

| Voucher | Lua `apply_to_run` Effect | Python Branch | Status |
|---------|--------------------------|---------------|--------|
| Overstock | `change_shop_size(1)` → `shop.joker_max += 1` | `state.shop.joker_max += 1` | MATCHED |
| Clearance Sale | `GAME.discount_percent = 25` | `state.discount_percent = extra` (extra=25) | MATCHED |
| Hone | `GAME.edition_rate = 2` | `state.edition_rate = extra` (extra=2) | MATCHED |
| Reroll Surplus | `round_resets.reroll_cost -= 2` | `state.starting_params.reroll_cost -= extra` then synced in `redeem_voucher` | MATCHED |
| Crystal Ball | `consumeables.config.card_limit += 1` | `state.starting_params.consumable_slots += 1` | MATCHED |
| Telescope | *(empty block — no apply_to_run effect)* | *(no branch needed)* — runtime via `used_vouchers.v_telescope` in `shop.py` | MATCHED |
| Grabber | `round_resets.hands += 1` | `state.starting_params.hands += extra` then synced | MATCHED |
| Wasteful | `round_resets.discards += 1` | `state.starting_params.discards += extra` then synced | MATCHED |
| Tarot Merchant | `GAME.tarot_rate = 4*extra` | `state.tarot_rate = 4 * extra` | MATCHED |
| Planet Merchant | `GAME.planet_rate = 4*extra` | `state.planet_rate = 4 * extra` | MATCHED |
| Seed Money | `GAME.interest_cap = 50` | `state.interest_cap = extra` (extra=50) | MATCHED |
| Blank | `check_for_unlock(...)` (achievement only, no game state) | *(no branch needed)* — no game-state effect | MATCHED |
| Magic Trick | `GAME.playing_card_rate = 4` | `state.playing_card_rate = extra` (extra=4) | MATCHED |
| Hieroglyph | `ante -= 1`, `blind_ante -= 1`, `hands -= 1` | `state.round_resets.ante -= extra`, `blind_ante -= extra`, `starting_params.hands -= extra` | MATCHED |
| Director's Cut | *(no apply_to_run effect)* — runtime check via `used_vouchers.v_directors_cut` in `button_callbacks.lua` | *(no branch needed)* — `reroll_boss()` + `boss_rerolled` flag handle this | MATCHED |
| Paint Brush | `G.hand:change_size(1)` → `starting_params.hand_size += 1` | `state.starting_params.hand_size += 1` | MATCHED |

### Tier 2 (upgraded vouchers, order 2–32 even)

| Voucher | Lua `apply_to_run` Effect | Python Branch | Status |
|---------|--------------------------|---------------|--------|
| Overstock Plus | same as Overstock | shared branch `{"Overstock", "Overstock Plus"}` | MATCHED |
| Liquidation | `GAME.discount_percent = 50` | shared branch `{"Clearance Sale", "Liquidation"}`, extra=50 | MATCHED |
| Glow Up | `GAME.edition_rate = 4` | shared branch `{"Hone", "Glow Up"}`, extra=4 | MATCHED |
| Reroll Glut | same as Reroll Surplus | shared branch `{"Reroll Surplus", "Reroll Glut"}` | MATCHED |
| Omen Globe | *(no apply_to_run effect)* — runtime check in Arcana pack generation | *(no branch needed)* — handled via `used_vouchers.v_omen_globe` in `shop.py` | MATCHED |
| Observatory | *(no apply_to_run effect)* — runtime x_mult during scoring | *(no branch needed)* — handled via `used_vouchers.v_observatory` in `scoring.py` | MATCHED |
| Nacho Tong | same as Grabber | shared branch `{"Grabber", "Nacho Tong"}` | MATCHED |
| Recyclomancy | same as Wasteful | shared branch `{"Wasteful", "Recyclomancy"}` | MATCHED |
| Tarot Tycoon | same as Tarot Merchant | shared branch `{"Tarot Merchant", "Tarot Tycoon"}` | MATCHED |
| Planet Tycoon | same as Planet Merchant | shared branch `{"Planet Merchant", "Planet Tycoon"}` | MATCHED |
| Money Tree | `GAME.interest_cap = 100` | shared branch `{"Seed Money", "Money Tree"}`, extra=100 | MATCHED |
| Antimatter | `jokers.config.card_limit += 1` | `state.starting_params.joker_slots += 1` | MATCHED |
| Illusion | `GAME.playing_card_rate = 4` (same as Magic Trick) | shared branch `{"Magic Trick", "Illusion"}`, extra=4; additional runtime effect handled via `used_vouchers.v_illusion` in `shop.py` | MATCHED |
| Petroglyph | `ante -= 1`, `blind_ante -= 1`, `discards -= 1` | shared `{"Hieroglyph", "Petroglyph"}` branch with discard path | MATCHED |
| Retcon | *(no apply_to_run effect)* — unlimited boss rerolls, runtime check | *(no branch needed)* — `reroll_boss()` function handles this | MATCHED |
| Palette | same as Paint Brush | shared branch `{"Paint Brush", "Palette"}` | MATCHED |

## Implementation Notes

### Sync pattern for cumulative params
`redeem_voucher` in `shop.py` (lines 295–299) re-syncs `round_resets.hands`, `round_resets.discards`, and `round_resets.reroll_cost` from `starting_params` after calling `_apply_voucher_to_run`. This correctly emulates the Lua pattern where `apply_to_run` mutates `round_resets` directly, because Python accumulates in `starting_params` first.

### Runtime-only vouchers
Four vouchers have no `apply_to_run` effect in Lua either — their effects are purely checked at runtime via `G.GAME.used_vouchers`:
- **Telescope** — first card in Celestial packs is forced to the most-played hand's planet
- **Observatory** — planet cards in consumable area give x_mult during scoring
- **Omen Globe** — Spectral cards can appear in Arcana packs (20% chance)
- **Director's Cut / Retcon** — boss blind reroll eligibility

All five runtime effects are correctly implemented in Python (`shop.py`, `scoring.py`, `blind.py`).

### Blank voucher
The Lua `apply_to_run` for Blank only calls `check_for_unlock({type = 'blank_redeems'})`, which is the achievement/progression system — not game state relevant to simulation. Python correctly has no branch for Blank.

## Verdict

All 32 vouchers are correctly implemented. No fixes required.
