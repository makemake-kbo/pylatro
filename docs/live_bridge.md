# Live Balatro bridge

The live bridge lets the existing Pylatro heuristic or an existing checkpoint
control a manually started, vanilla Balatro run. It targets Balatro
`1.0.1o-FULL` under Proton. Choose the profile, deck, stake, and seed yourself;
automation attaches at the first blind-selection decision.

The Lua mod is deliberately a thin sensor/actuator. It takes one ordered
snapshot at a stable decision, sends it over loopback HTTP, validates the
matching response against the still-live state, and invokes Balatro's enabled
UI callbacks. Python remains the sole strategic authority.

## Prerequisites

Follow Steamodded's official
[Linux installation guide](https://github.com/Steamodded/smods/wiki/Installing-Steamodded-linux):

1. Install the Windows Lovely build's `version.dll` next to `Balatro.exe`.
2. Set Balatro's Steam launch option exactly to:

   ```text
   WINEDLLOVERRIDES="version=n,b" %command%
   ```

3. Install Steamodded under the Proton save directory's `Balatro/Mods`
   directory. Its inner directory should be `Mods/smods`, not a doubly nested
   directory.

The bridge uses Steamodded's documented
[`SMODS.https`](https://github.com/Steamodded/smods/wiki/SMODS.https)
asynchronous API and bundled `json` module. It does not ship native networking
code.

Install Pylatro's agent dependencies once:

```bash
uv sync --extra agent
```

## Install

From this repository:

```bash
uv run python scripts/install_live_bridge.py
```

The helper checks the usual native Steam, Steam Deck, Snap, and Flatpak Proton
prefixes, verifies Lovely and Steamodded, and symlinks only
`mods/pylatro_bridge` into `AppData/Roaming/Balatro/Mods`. Use `--copy` if a
symlink is unsuitable, or `--prefix /path/to/pfx` for a nonstandard library.
It never searches or writes the repository's `Application Data BACKUP`
directory.

## Run

Start Python before entering a run:

```bash
uv run --extra agent python play.py --live --heuristic
```

Or use an unchanged checkpoint:

```bash
uv run --extra agent python play.py --live --checkpoint checkpoints/ppo/ppo_update100.pt
```

`--device`, `--sample`, and `--temperature` work as they do in simulated play.
The server listens only on `127.0.0.1`; its default port is `43137`. `--host`
accepts only `127.0.0.1` or `localhost`, and `--port` may select another
loopback port (the Lua endpoint must be changed to match).

Then launch Balatro, select a profile, and manually start a vanilla run. The
server attaches to one session, logs actions, model values, and latency, and
exits after a win or game over. The bridge will retry if Python starts late or
temporarily disappears. Balatro remains interactive and does not pause.

## Supported decisions

- select, skip, or reroll a blind
- play or discard live card IDs
- use an owned consumable during a hand
- buy, reroll, sell, or leave the shop
- claim or skip every vanilla booster type
- choose hand-card or joker targets for Tarot/Spectral pack cards

Arbitrary joker reordering and using owned consumables in the shop are not
available in protocol v1. Gameplay objects from content mods are reported as
unsupported instead of being treated as vanilla. Steamodded and this bridge
are allowed.

## Troubleshooting

- No Lovely console: confirm `version.dll` is beside `Balatro.exe` and the
  Steam launch option is set.
- Lovely works but there is no Mods button: check Steamodded's directory
  nesting under the Proton prefix.
- Bridge logs connection retries: start `play.py --live`, confirm port `43137`
  is free, and ensure Lua/Python port settings match.
- Python reports unknown content: disable gameplay/content mods and start a
  fresh vanilla run.
- An action is rejected: Balatro changed while inference was running or its UI
  callback was disabled. The bridge reports the rejection in the next
  snapshot and requests a fresh decision.
- Continuing an old or modded save crashes: follow Steamodded's advice and
  start a new run.

## Uninstall

Remove only the bridge:

```bash
uv run python scripts/install_live_bridge.py --uninstall
```

This leaves Lovely and Steamodded intact. To disable all mod loading, remove
`version.dll` from the Balatro game directory and clear the launch option, as
described by Steamodded's Linux guide.

## Protocol v1

`POST /v1/decision` requests contain:

```text
protocol_version, session_id, decision_id, state_fingerprint, phase,
versions, state, legal, previous_action?
```

Responses echo the four identity fields and phase, then contain exactly one of
`action`, `wait`, or `error`. Semantic action objects use stable live object
IDs; the large internal Pylatro action index never crosses the wire. Exact
duplicate requests return the cached response. Conflicting, stale, mismatched,
or unsupported requests fail closed.
