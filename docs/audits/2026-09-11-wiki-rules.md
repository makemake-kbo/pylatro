# Balatro wiki rules audit — 2026-09-11

The engine is **not fully faithful to Balatro yet**. This audit fixes verified
scoring, boss-debuff and payout errors, adds independent rule fixtures, and makes
three confirmed remaining discrepancies executable as strict expected failures.
The target remains the README's `1.0.1o-FULL` mechanics; this is not certification
of every Joker combination, challenge, or seeded run against the original game.

## References

Pages were downloaded directly from the user-requested Balatro wiki. The browser
fetcher was blocked by robots.txt. Revision links preserve the reference used:

- [Poker hands, revision 27261](https://balatrowiki.org/w/Poker_hands?oldid=27261): hand hierarchy, values, Planet increments, Ace boundaries and Four Fingers exceptions.
- [Activation sequence, revision 27309](https://balatrowiki.org/w/Guide:_Activation_Sequence?oldid=27309): card order, editions, held abilities and retriggers. This page identifies itself as a community guide.
- [Card modifiers, revision 27627](https://balatrowiki.org/w/Card_modifiers?oldid=27627): enhancements, seals, editions and stickers.
- [Negative effects, revision 26540](https://balatrowiki.org/w/Negative_Effects?oldid=26540): debuffed cards retain hand classification but lose scoring effects; Wild and Stone exceptions.
- [Mime, revision 26715](https://balatrowiki.org/w/Mime?oldid=26715): held Joker abilities, Gold payouts and Blue Seal generation.
- [Decks, revision 28489](https://balatrowiki.org/w/Decks?oldid=28489) and [Stakes](https://balatrowiki.org/w/Stakes): starting parameters and cumulative difficulty. Stakes were available through the search index.
- [Tarot cards, revision 27917](https://balatrowiki.org/w/Tarot_cards?oldid=27917): target limits, conversions and money caps.
- [Blinds and Antes, revision 28650](https://balatrowiki.org/w/Blinds_and_Antes?oldid=28650): boss restrictions and effects.
- [Booster Packs, revision 27300](https://balatrowiki.org/w/Booster_Packs?oldid=27300): consumables must be used immediately and Arcana/Spectral packs provide a hand for targeting.

## Corrected behavior

1. Scoring follows played order, including Stone cards. Rank grouping previously
   reversed pairs and rearranged other hands, changing enhancement and Joker results.
2. Playing-card editions activate before card-triggered Jokers. A holographic
   King with Photograph now produces 22 Mult instead of 12.
3. Polychrome Joker editions activate after the Joker's ability and other-Joker
   effects. A Polychrome basic Joker now gives 7.5 Mult from a High Card base of 1.
4. Debuffed cards cannot trigger scoring editions or per-card Joker effects.
   Debuffed Jokers cannot score their editions or copy another Joker.
5. Held Red Seals and Mime retrigger held abilities, including plain cards whose
   ability comes from Baron or Shoot the Moon. Retriggers add together.
6. Gold-card payouts and Blue Seal generation honor Mime, Blueprint/Brainstorm
   copies and capacity; Gold settlement remains idempotent.
7. Identical Blueprint instances follow their actual positions. Dataclass value
   equality previously resolved a target to the wrong position in a copy chain.
8. Green Deck now pays for remaining discards as well as hands.
9. Suit bosses recognize Wild cards and exclude Stone cards. The Plant no longer
   treats a Stone card's printed face rank as an active rank.
10. The agent's held-card value estimate now accounts for Mime and Red Seal
    retriggers. An old test explicitly asserting the incorrect Mime behavior was
    replaced with an independent numerical expectation.

## Coverage

The new `tests/test_wiki_rules.py` uses literal reference values and manually
calculated scores; it does not construct expected values from `game_data.json`.

| Area | New checks | Existing complementary coverage |
| --- | --- | --- |
| Hands and Planets | All 12 types at base level and after 1/3 upgrades; Royal Flush; scoring-card identities/order | Blind scaling into endless mode |
| Hand boundaries | Ace high/low/no wrap, Four Fingers, Shortcut, Smeared Joker, Wild and Stone | Generic hand-dependent Jokers |
| Scoring interactions | Played order, both edition stages, debuffs, held retriggers, identical copy chains | Individual Joker formulas, Observatory, Plasma |
| Economy and seals | Green Deck, Hermit cap, Gold settlement, Blue Seal capacity and Mime copies | Interest tiers/vouchers, Purple and Blue Seals, investment tags |
| Consumables | All enhancement and suit Tarots, Strength, Death, Black Hole including hidden hands | Creation slots, removal hooks, immediate Planet claims |
| Decks and stakes | Starting resources for every deck and cumulative flags for every stake | Deck composition and initialization |
| Bosses | Four suit bosses, Wild/Stone interactions, Plant, Water, Needle, Manacle, Psychic and Flint | Eye, Mouth, Hook, finishers, boss disabling and rerolls |

This is behavioral coverage, not a measured line/branch coverage claim. In
particular, initialization tests for Orange/Gold Stake do not establish that
those stakes' entire lifecycle is implemented.

## Confirmed outstanding discrepancies

These are tracked with `pytest.mark.xfail(strict=True)`. They must remain visible
in test summaries; an unexpected pass fails the suite and signals that the marker
should be removed after verifying the fix.

| Regression | Missing behavior | Implementation involved |
| --- | --- | --- |
| `test_pack_tarots_cannot_be_banked_for_later` | Pack claims can bank consumables when immediate use is unavailable. Pack targeting is an agent-selected continuation, and opening Arcana/Spectral packs does not draw the required hand. | `shop.py`, controller/UI and agent pack actions |
| `test_perishable_expires_after_five_completed_rounds` | Perishable tally is initialized but never decremented; expired debuffs also need to survive blind resets and affect passive bonuses. | `runtime.py`, `flow.py`, `instances.py` |
| `test_rental_charges_three_dollars_even_when_debuffed` | Rental price/sticker exist, but the round-end $3 charge is absent. | `runtime.py` |

Other areas still need systematic reference fixtures: all 150 Jokers across their
activation phases and copy compatibility, the full Spectral lifecycle, all
vouchers/tags, sticker/passive-effect combinations, and remaining boss sequences.
The wiki describes rules but is not a sufficient oracle for bit-exact seeded RNG;
that needs comparison against original-game traces.

## Verification

The initial engine subset passed 146 tests despite the newly identified errors.
The first 89 independent wiki cases exposed 49 failures before the scoring fixes;
subsequent round-end and boss fixtures caught additional discrepancies.

Run the regression file with:

```sh
.venv/bin/pytest tests/test_wiki_rules.py -rx
```

Compiled `.so` modules take precedence over `.py` files in this checkout. Rebuild
changed extensions before testing compiled behavior. For a complete rebuild use
`.venv/bin/python scripts/compile_cython.py`; for a source-only run, remove the
generated extensions with that script's `--clean` option first. This audit also
checked Python source directly with a temporary import loader, then rebuilt the
modified scoring, runtime and flow extensions and tested the normal import path.

Full-suite command (thread limits avoid excessive CPU oversubscription):

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/pytest -m 'not slow'
```

Final focused results: **321 passed, 3 expected failures** on both Python source
and rebuilt native imports. The standalone wiki regression file contributes
**156 passing cases and 3 strict expected failures**. Ruff passes for the new
regression file and the changed agent value-estimation files; `git diff --check`
is clean. Engine modules have existing lint findings outside the changed lines.

The full compiled suite completed with **1,321 passed, 5 skipped, 16 warnings**
in 361 seconds, with no failures. Its collection preceded the addition of the
three expected-failure cases; the final focused run above includes those cases
and the final rebuilt scoring module. No tests were deselected by the slow marker.
