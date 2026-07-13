# Seed search

`pylatro seed-search` finds run seeds whose vouchers, shops, blinds, and skips
match a JSON spec. Only what the spec mentions is constrained — everything else
is generated but ignored.

```bash
pylatro seed-search spec.json                 # search random seeds
pylatro seed-search spec.json --matches 5     # collect 5 matching seeds
pylatro seed-search spec.json --max-seeds 3000000
pylatro seed-search spec.json --check ABCD1234  # verify one specific seed
pylatro seed-search spec.json --rng-seed 7    # reproducible search order
pylatro seed-search spec.json --workers 4     # parallel workers (default: all CPUs)
pylatro seed-search spec.json --json          # machine-readable output
```

Seeds are checked across all CPU cores by default; the parent process draws
the seed stream, so `--rng-seed` tries the same seeds at any `--workers` count
(though which match is found first can vary with worker timing).

Exit codes: `0` match found, `1` no match, `2` invalid spec.

## Spec format

Spec files are parsed as jsonc: full-line or trailing `#` / `//` comments and
trailing commas are all allowed. Names are fuzzy: `"overstock"`,
`"v_overstock_norm"`, and `"Overstock"` all work; `"hermit"` means The Hermit;
joker nicknames like "photo" for photograph, "chad" for the hanging chad, and
"useless" for loyalty card are all accepted.

```jsonc
{
    "deck": "red",        // optional, default red deck
    "stake": 1,           // optional, 1-8, default 1
    "unlocked": true,     // optional: search as a fully-unlocked profile (default)

    "ante1": {
        "voucher": "overstock",        // the ante's shop voucher
        "boss": "the wall",            // the ante's boss blind
        "big": {                       // small / big / boss (also smallblind, ...)
            "skip": {
                "tag": "charm",                    // tag granted by skipping
                "pack": "mega arcana",             // pack that tag opens
                "contains": ["hermit", "perkeo"]   // cards inside that pack
            }
        }
    },
    "ante2": {
        "voucher": "overstock plus",
        "small": {
            "shop": {
                "contains": ["..."],          // alias for within_1
                "within_2": ["blueprint"],    // seen within 2 shop rolls
                "packs": [                    // booster slots in this shop
                    "spectral",
                    {"pack": "jumbo celestial", "contains": ["pluto"]}
                ]
            }
        }
    }
}
```

### Semantics

- **`voucher`**: the voucher offered during that ante. The searcher "buys" it
  at the first shop that offers it (a voucher for ante N first appears in the
  shop right after ante N−1's boss), so its effects (extra shop slot, rates,
  higher-tier unlock) apply from then on.
- **`skip`**: the blind is skipped. `tag` constrains the tag it grants. `pack`
  + `contains` constrain the free pack that tag opens (Charm → Mega Arcana,
  Meteor → Mega Celestial, Standard → Mega Standard, Buffoon → Mega Buffoon,
  Ethereal → Spectral). A skipped blind has no shop.
- **`shop.within_N`**: every listed item appears within the first N shop rolls
  (roll 1 is the shop as first seen; each further roll is a reroll). The
  searcher stops rerolling as soon as everything is found.
- **Legendaries in packs**: naming a legendary joker (e.g. `"perkeo"`) in an
  Arcana/Spectral pack means: the pack contains The Soul *and* using it creates
  that joker. Use `"soul"` to accept any legendary.
- **Canonical playthrough**: blinds not marked `skip` are beaten, every shop is
  visited, and nothing is bought except spec'd vouchers (and opened spec'd
  packs). Deviating from that in a real run (extra rerolls, buying jokers,
  using consumables that roll RNG) can change later shops.

The spec is rejected up front when it breaks game rules, e.g.:
- higher-tier vouchers without their tier-1 requested in an *earlier* ante
  (`antimatter` needs `blank` before it; `overstock plus` needs `overstock`);
- the same voucher twice, or an ante-1 voucher with both ante-1 blinds skipped
  (no shop would ever offer it);
- skipping the boss blind;
- items that cannot spawn where requested (a joker inside an Arcana pack, a
  legendary in the shop, a jumbo pack from a skip tag);
- unknown or ambiguous names (with "did you mean" suggestions).

## Examples

### Photochad in the first shop, Credit Card in its Buffoon pack

The first shop (after beating ante 1's small blind) sells both Photograph and
Hanging Chad, and the Buffoon pack it always stocks contains a Credit Card.
Found seed: `5W7A7UGZ`.

```jsonc
{
    "ante1": {
        "small": {
            "shop": {
                "contains": ["photo", "chad"],          // both in the initial shop
                "packs": [
                    {"pack": "buffoon", "contains": ["credit card"]}
                ]
            }
        }
    }
}
```

```
pylatro seed-search spec.json --check 5W7A7UGZ
ante1.small: buffoon pack contains [j_trousers, j_credit_card]
ante1.small: j_photograph in shop on roll 1/1
ante1.small: j_hanging_chad in shop on roll 1/1
```

## Python API

```python
from pylatro.seedsearch import parse_spec, check_seed, search_seeds

spec = parse_spec({"ante1": {"voucher": "overstock"}})
matches = search_seeds(spec, max_seeds=100_000, matches=3)
for m in matches:
    print(m.seed, m.notes)
```

Throughput is roughly 2–5k seeds/s depending on how deep into the run the spec
reaches (constraints are checked in ante order and fail fast). Heavily stacked
specs multiply their odds: the example above (voucher chain + skip pack
contents + shop joker) is a ~1-in-a-few-million seed — raise `--max-seeds`
accordingly. Use cython
