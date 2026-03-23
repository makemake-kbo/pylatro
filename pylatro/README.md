# pylatro

`pylatro` is a headless Python porting workspace for Balatro `1.0.1o-FULL`.

The upstream shipped Lua sources are unpacked into [`vendor/balatro_lua`](/Users/makemake/Documents/code/python/pylatro/vendor/balatro_lua) and treated as the reference implementation. The Python package in [`src/pylatro`](/Users/makemake/Documents/code/python/pylatro/src/pylatro) focuses on deterministic run generation, state transitions, and action handling so the project can later back a Gym-style environment for research.

The current implementation includes:

- upstream Lua table extraction for blinds, decks, cards, centers, stakes, tags, and seals
- Balatro-compatible pseudohash / pseudoseed state handling
- headless run initialization with stake modifiers and deck effects
- blind, voucher, tag, boss, shop, reroll, and pack generation logic
- differential and behavior tests over the deterministic core
