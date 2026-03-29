# Consumable Branch Audit

**Date:** 2026-03-26
**Source:** `vendor/balatro_lua/card.lua` — `Card:use_consumeable`
**Target:** `src/pylatro/consumables.py` — `use_consumable`

## Summary

| Status    | Count |
|-----------|-------|
| MATCHED   | 52    |
| MISSING   | 0     |
| DIVERGENT | 0     |

All consumable branches are implemented and logic matches the upstream Lua.

---

## Tarot Cards (22)

| # | Name | Key | Logic | Status |
|---|------|-----|-------|--------|
| 1 | The Fool | c_fool | Re-creates last used tarot/planet (up to consumable limit) | MATCHED |
| 2 | The Magician | c_magician | mod_conv → m_lucky on up to 2 highlighted cards | MATCHED |
| 3 | The High Priestess | c_high_priestess | Adds 2 random Planet cards (up to consumable limit) | MATCHED |
| 4 | The Empress | c_empress | mod_conv → m_mult on up to 2 highlighted cards | MATCHED |
| 5 | The Emperor | c_emperor | Adds 2 random Tarot cards (up to consumable limit) | MATCHED |
| 6 | The Hierophant | c_heirophant | mod_conv → m_bonus on up to 2 highlighted cards | MATCHED |
| 7 | The Lovers | c_lovers | mod_conv → m_wild on 1 highlighted card | MATCHED |
| 8 | The Chariot | c_chariot | mod_conv → m_steel on 1 highlighted card | MATCHED |
| 9 | Justice | c_justice | mod_conv → m_glass on 1 highlighted card | MATCHED |
| 10 | The Hermit | c_hermit | Doubles money (capped at config.extra=20) | MATCHED |
| 11 | The Wheel of Fortune | c_wheel_of_fortune | 1-in-4 chance to give a random edition to an editionless joker | MATCHED |
| 12 | Strength | c_strength | Raises rank of up to 2 highlighted cards by 1 (A wraps to 2) | MATCHED |
| 13 | The Hanged Man | c_hanged_man | Destroys up to 2 highlighted cards | MATCHED |
| 14 | Death | c_death | Copies rightmost of 2 highlighted cards into the other | MATCHED |
| 15 | Temperance | c_temperance | Gives dollars equal to sum of joker sell costs (capped at 50) | MATCHED |
| 16 | The Devil | c_devil | mod_conv → m_gold on 1 highlighted card | MATCHED |
| 17 | The Tower | c_tower | mod_conv → m_stone on 1 highlighted card | MATCHED |
| 18 | The Star | c_star | suit_conv → Diamonds on up to 3 highlighted cards | MATCHED |
| 19 | The Moon | c_moon | suit_conv → Clubs on up to 3 highlighted cards | MATCHED |
| 20 | The Sun | c_sun | suit_conv → Hearts on up to 3 highlighted cards | MATCHED |
| 21 | Judgement | c_judgement | Creates a random joker | MATCHED |
| 22 | The World | c_world | suit_conv → Spades on up to 3 highlighted cards | MATCHED |

---

## Planet Cards (12)

All planets use the generic `center.get("set") == "Planet"` branch which calls `_level_up_hand(state, center["config"]["hand_type"])`.

| # | Name | Key | Hand Type | Status |
|---|------|-----|-----------|--------|
| 1 | Mercury | c_mercury | Pair | MATCHED |
| 2 | Venus | c_venus | Three of a Kind | MATCHED |
| 3 | Earth | c_earth | Full House | MATCHED |
| 4 | Mars | c_mars | Four of a Kind | MATCHED |
| 5 | Jupiter | c_jupiter | Flush | MATCHED |
| 6 | Saturn | c_saturn | Straight | MATCHED |
| 7 | Uranus | c_uranus | Two Pair | MATCHED |
| 8 | Neptune | c_neptune | Straight Flush | MATCHED |
| 9 | Pluto | c_pluto | High Card | MATCHED |
| 10 | Planet X | c_planet_x | Five of a Kind | MATCHED |
| 11 | Ceres | c_ceres | Flush House | MATCHED |
| 12 | Eris | c_eris | Flush Five | MATCHED |

---

## Spectral Cards (18)

| # | Name | Key | Logic | Status |
|---|------|-----|-------|--------|
| 1 | Familiar | c_familiar | Destroys 1 random hand card; adds 3 face-card-rank enhanced cards | MATCHED |
| 2 | Grim | c_grim | Destroys 1 random hand card; adds 2 Ace-rank enhanced cards | MATCHED |
| 3 | Incantation | c_incantation | Destroys 1 random hand card; adds 4 numbered-rank enhanced cards | MATCHED |
| 4 | Talisman | c_talisman | Applies Gold seal to 1 selected card | MATCHED |
| 5 | Aura | c_aura | Applies a random non-negative edition to 1 selected card | MATCHED |
| 6 | Wraith | c_wraith | Creates a rare joker; sets money to 0 | MATCHED |
| 7 | Sigil | c_sigil | Changes all hand cards to a random suit | MATCHED |
| 8 | Ouija | c_ouija | Changes all hand cards to a random rank; reduces hand size by 1 | MATCHED |
| 9 | Ectoplasm | c_ectoplasm | Gives negative edition to an editionless joker; reduces hand size by ecto_minus; increments ecto_minus | MATCHED |
| 10 | Immolate | c_immolate | Destroys 5 random hand cards; gives $20 | MATCHED |
| 11 | Ankh | c_ankh | Destroys all non-eternal jokers except one randomly chosen; duplicates the chosen joker (stripping negative edition if present) | MATCHED |
| 12 | Deja Vu | c_deja_vu | Applies Red seal to 1 selected card | MATCHED |
| 13 | Hex | c_hex | Gives polychrome edition to an editionless joker; destroys all other non-eternal jokers | MATCHED |
| 14 | Trance | c_trance | Applies Blue seal to 1 selected card | MATCHED |
| 15 | Medium | c_medium | Applies Purple seal to 1 selected card | MATCHED |
| 16 | Cryptid | c_cryptid | Creates 2 copies of 1 selected card into hand | MATCHED |
| 17 | The Soul | c_soul | Creates a legendary joker | MATCHED |
| 18 | Black Hole | c_black_hole | Levels up all poker hands by 1 | MATCHED |

---

## Notes on Implementation Correctness

### Wheel of Fortune / Ectoplasm / Hex — eligible joker pool
The Lua assigns `eligible_strength_jokers` for Wheel of Fortune and `eligible_editionless_jokers` for Ectoplasm/Hex, but both pools use the identical filter (`ability.set == 'Joker' and not v.edition`). Python correctly uses `_eligible_editionless_jokers` for all three.

### Temperance — joker filter
The Lua pre-computes `ability.money` by summing sell costs only for jokers with `ability.set == 'Joker'`. In pylatro `state.jokers` contains only JokerInstances (all of which have set `Joker`), so the sum over `state.jokers` is equivalent.

### Ankh — edition stripping
Lua calls `copy_card(..., chosen_joker.edition and chosen_joker.edition.negative)` which strips the edition when negative, then removes any residual negative edition post-copy. Python passes `edition=None` when the chosen joker has a negative edition, producing the same result.

### Ectoplasm — ecto_minus initial value
Lua initializes `G.GAME.ecto_minus` to `1` on first use via `G.GAME.ecto_minus or 1`. Python initializes `RunState.ecto_minus = 1` at model construction, which is equivalent.

### Planets — Black Hole
`Black Hole` loops over all hand types and calls `level_up_hand` on each. Python iterates `state.hands` and delegates to `_level_up_hand`. Correct.
