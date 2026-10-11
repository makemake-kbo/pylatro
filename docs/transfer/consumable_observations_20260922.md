# Consumable history changes action effects without changing policy inputs

Two deterministic fixtures confirm that the current policy inputs omit The
Fool's target and accumulated consumable resale value. Within each pair, **all
14 observation arrays are identical**, including legal actions, scalars and
played-hand history. The same legal action nevertheless has different effects.

These are concrete information omissions. Neither fixture measures their
frequency, effects on wins, different optimal actions, or responsibility for the
training plateau. A policy receiving complete past action history could
distinguish the Fool cases; the current observation arrays cannot.

| Fixture pair | Identical action | Different effect |
| --- | --- | --- |
| Use Hermit then Mercury / Mercury then Hermit | Use the remaining Fool | Creates Mercury / Hermit |
| Acquire Mercury before / after eight Gift Card payouts | Sell Mercury | Receives $9 / $1 |

## What was checked

The [reproducer](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/diagnose_consumable_observations.py)
uses constructed shop fixtures. The Fool cases start with Crystal Ball capacity
and Hermit, Mercury and Fool in inventory. Both orders use legal, mask-checked
environment actions. After the two uses, cash is $20 and Fool is the only remaining
consumable. The complete observations agree, including the encoded hand levels.
The internal `last_tarot_planet` differs, and using Fool creates that exact card.
The [September 6 audit](../audits/2026-09-06-model-training-audit.md) had already
identified the missing target as a concern; this fixture now reproduces the
complete observation collision and resulting action effect.

Both resale fixtures invoke the same eight Gift Card engine payout callbacks,
keeping the Joker's own accumulated state matched. Mercury is acquired before
those callbacks in one fixture and afterward in the other. The engine produces
sale values of $9 and $1; the test does not manually assign those prices. The
same legal environment sale action pays the corresponding amount.

The [results and hashes](data/consumable_observation_collisions_20260922.json)
preserve each action prefix, effect, and complete observation archive. An
[independent read-only review](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/consumable_observation_collisions/independent_review.json)
found no blocking issue. The diagnostic completed in approximately 0.83 seconds
without policy inference, training or full-game evaluation. Frozen source hashes
were checked before and after it ran.

The Gift Card callbacks are not eight played rounds. Neither test reconstructs a
complete reachable prefix from a fresh game. They establish the input limitation
for the constructed states and legal actions, rather than a gameplay improvement.

## Correction in preparation

Along with the verified [pack-enhancement omission](pack_enhancement_observation_20260922.md),
these cases motivate exposing three specific fields: pack enhancement, the
remembered Tarot/Planet target, and consumable sale proceeds. An implementation
is being prepared in a separate regular-file source copy, leaving the active
study unchanged.

The correction needs explicit tokenizer/model versions and checkpoint migration.
Old-checkpoint inference and exact optimizer resume have different compatibility
requirements. A new representation must start an explicit training branch;
it cannot silently add parameters to an old optimizer resume. Fresh matched
training and unseen-game evaluation remain necessary to test whether the added
information improves long-horizon decisions and wins.
