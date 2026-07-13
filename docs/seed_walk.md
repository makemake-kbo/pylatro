# Seed walk

Seed walk is the interactive companion to [seed search](seed_search.md). Instead
of describing what you want and searching for a seed, you enter a seed and step
through its shops and blind/ante selection by hand — **without playing any
hands** — with **free rerolls**, a joker-depth lookup, a per-ante voucher
preview, and a report you can export back into seed-search notation.

It runs inside the terminal UI:

```bash
pylatro                 # launch the TUI, then choose "Seed Walk"
```

## How it works

Like the searcher, blinds are "beaten" without playing a hand. Hand RNG lives on
separate pseudoseed keys, so the shop / voucher / tag / pack streams are
identical to a real run that beats every blind. Buying or selling goes through
the real engine, so an acquired joker is marked used and **stops reappearing in
later rolls** (except with Showman), exactly as in a real run.

Money never gates a walk: rerolls and buys are always free.

> Skip-tag *effects* on the next shop (e.g. an Uncommon tag forcing the shop's
> joker rarity) are not modeled — the underlying engine has no tag-consumption
> logic — so a skipped blind records its tag but does not alter the following
> shop. This matches the searcher's fidelity.

## Blind selection

| key | action |
| --- | --- |
| `Enter` | beat the on-deck blind (phantom) and open its shop |
| `s` | skip the on-deck blind (Small/Big); pack-granting tags open their free pack |
| `x` | reroll the boss |
| `v` | **voucher schedule** — the voucher offered per ante from here forward, assuming no further purchases (simulated on a copy). `Enter` pins the focused ante's voucher. |
| `p` | pin the current ante's boss |
| `r` | open the report |

## Shop

| key | action |
| --- | --- |
| arrows / `hjkl` | move the item cursor |
| `Enter` | buy the focused card / voucher, or open the focused booster |
| `r` | reroll (free) |
| `p` | pin the focused item (records the roll depth it appeared on) |
| `f` | focus the **find** box: type a joker/card name and press `Enter` to see how many rerolls until it appears (runs on a copy, so the live shop is untouched) |
| `Tab` | focus the joker / consumable bar, then `s` to sell |
| `n` | leave the shop and go to the next blind |
| `R` | open the report |

## Report

Pinned findings accumulate into a [seed-search spec](seed_search.md). Open it
with `r` / `R`:

- `d` deletes the focused pin;
- `e` exports the report to `seedwalk_<SEED>.json` in the current directory.

The exported file is valid jsonc — its header comments are stripped by the spec
loader — so you can verify it straight away:

```bash
pylatro seed-search seedwalk_5W7A7UGZ.json --check 5W7A7UGZ
```

A pinned shop card becomes a `within_N` entry (N = the roll it was seen on),
a pinned voucher/boss becomes the ante's `voucher` / `boss`, and a pinned pack
records its `contains`. See [`docs/seed_search.md`](seed_search.md) for the full
notation.

## Python API

```python
from pylatro.seedwalk import SeedWalk

walk = SeedWalk("5W7A7UGZ")
walk.beat_blind()                       # phantom-beat the small blind -> shop
print(walk.rolls_until("blueprint"))    # ("j_blueprint", 81)  -> 81 rerolls away
print(walk.voucher_schedule(horizon=4)) # [(1, "v_..."), (2, "v_..."), ...]
walk.pin_shop_card("j_photograph")      # record a finding
print(walk.report.to_text())            # seed-search notation
```

The seed walk drives a real `RunState`, so `walk.state` is the full engine state
at any point.
