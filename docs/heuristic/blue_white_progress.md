# Blue Deck / White Stake heuristic improvement

Goal: at least 90% full-game wins through the Ante 8 boss on fresh Blue Deck,
White Stake runs, with a usable heuristic for pretraining generation. **Not achieved.**
Use a fresh held-out seed panel after tuning; never treat reaching Ante 8 or
filtering winning seeds as the win rate. Seeds 10000–10999 are reserved for
final validation and have not been evaluated or used for tuning.

## Current validation status

The latest engine excludes Stone Cards from Odd Todd/Even Steven and charges
shop vouchers their displayed price. Earlier rates, including the 39/50 v39
panel, are provisional because they predate one or both corrections.

- V68 completed **82/100** on tuning seeds 0–99, then **68/100** on fresh
  development seeds 100–199. Combined: **150/200 (75%)** in
  `level_scalers_v68_combined200.jsonl`. The 82% tuning result does not
  generalize to the fresh panel. This does not use the reserved final seeds.
- V60: **74/100**, split 41/50 and 33/50. V66 also completed **74/100**.
  Both parent source/binary manifests match exactly; combined artifact:
  `stable_hand_ties_v60_combined100.jsonl`.
- Corrected V48 baseline: **72/100**, including 38/50 on seeds 0–49.
- V49: 38/50 on seeds 0–49. Its extension stopped after it accumulated more
  than ten losses and therefore could no longer reach 90/100.
- V53 stopped at 31 wins in 43 completed attempts; V54 stopped at 54 wins in
  75 completed attempts. Both had too many losses to reach 90/100. These are
  incomplete panels, not comparable full-panel rates.
- Matched v39 policy with both engine corrections: **35/50**,
  `economy_v39_paid_vouchers.jsonl`.
- V57 restores hand-plan preferences while protecting explicit held-card and
  Bus choices; requested hand types now match exactly. Its 152 focused checks
  passed; its completed panel is 35/50. Same-score hand substitutions were traced.
- V48 generation/evaluation replay parity on seed 11: 231 matching decisions,
  both won. V53: 173 matching decisions, both lost. V54 seed 13: 108 matching
  decisions, both lost.
- V58 stopped after 30 completions, 17 wins: it could no longer beat 38/50.
- V59 stopped at 18 wins in 30 attempts after it could no longer improve on
  38/50. V60 finished at **41/50** and V61 at **37/50**. V60 seeds 50–99 finished at 33/50, totaling 74/100. V61 removes continuation pacing and
  makes boss rerolls conditional on an actual scoring penalty.
- Defaults remain legacy until a candidate passes validation.

Completed parity checks: v42 seed 3, 227 identical decisions; v45 seed 11,
215 identical decisions. V48 engine/shop/boss/consumable checks: 105 pass.
The chronological notes below preserve experiments and superseded findings;
they should not be read as current win-rate validation.

## Worktree and execution

The worktree already contained substantial unrelated training/model changes.
Those changes were preserved. Relevant work for this goal is in the engine pool
lifecycle, `heuristic.py`, `heuristic_shop_search.py`, `fast_runner.py`,
`fast_generate.py`, and their new/updated regression tests.

Python initially imported compiled extensions over the heuristic and runner
sources. The heuristic, runner, `_helpers`, `instances`, `shop`, and `blind`
extensions were moved to `/tmp/pylatro-heuristic-original/` (engine extensions
under `engine/`) so evaluation uses edited sources. Other engine extensions are
still active. Rebuild and verify relevant Cython modules before final delivery.
Do not restore the old extensions over changed sources.

## Engine defect found from failure traces

Generated cards were added to `used_jokers` permanently. Even unbought shop
cards and unused pack cards were blocked for the remainder of the run. Almost
all initial games stopped leveling their main planet at level 2.

Verified against the user's installed vanilla Balatro executable (ZIP archive):
`card.lua`, `Card:remove`, line 4745 clears `G.GAME.used_jokers` on removal when
no owned copy remains; `functions/common_events.lua` line 1987 checks this map
for pool availability. It is occupancy, not generation history.

The fix releases removed shop/pack/inventory cards, preserves live duplicates,
and marks directly added inventory cards. Regression tests cover planet reuse,
last-copy joker removal, rerolls, pack claims/skips and shop exits.

An additional heuristic defect held debuffed suit cards through every discard
without a scoring engine. Cycling these cards restored the existing Red Deck
seed-64 early-survival regression (now reaches Ante 4).

## Experiments

Results below are tuning measurements, not proof of the 90% goal.

| Variant | Seeds | Wins |
| --- | --- | --- |
| Initial heuristic/engine, Blue/White | 0–99 | 10/100 |
| Pool lifecycle fix, original policy | 0–99 | 19/100 |
| Same fixed engine, original policy subset | 0–49 | 14/50 |
| Shop search v1, four samples | 0–49 | 18/50 |
| Shop search v2, interest reserve and more planet packs | 0–49 | 22/50 |
| Shop search v3, eight samples and independent simulation RNG | 0–99 | 44/100 |

Per-game records are under `eval/heuristic_blue/`. The benchmark counts every
attempt, requires `FastRunner.won`, fails on invalid actions/engine exceptions,
and writes confidence intervals and a source/binary manifest. The early v1/v2
summary heuristic hashes were captured at completion and can reflect subsequent
source edits; do not use them as exact provenance. Versions of the search
module were preserved under `/tmp/pylatro-heuristic-original/shop_search_v*.py`.
The benchmark now records its manifest before starting workers.

Shop search is currently opt-in (`HeuristicAgent(shop_policy="search")`). It
compares candidate purchases/replacements using full scoring on sampled public
deck hands; preserves interest; and buys useful planets. The legacy policy
remains the default until broader validation. Search v3 samples its own RNG,
independent of the run RNG and hidden draw order.

## Generation

`generate_training_data` now accepts `deck_key`, `stake`, and `shop_policy`.
Both observation-free screening and recorded replay receive identical settings;
records carry deck and stake metadata. Chunked generation excludes previously
retained seeds instead of silently duplicating episodes.

```python
from pylatro_agent.training.fast_generate import generate_training_data
records = generate_training_data(
    100, num_workers=8, deck_key="b_blue", stake=1, shop_policy="search",
)
```

A one-game generation smoke test completed: Blue/White seed 0 won and produced
241 records with legal actions and correct terminal metadata. It wrote
`eval/heuristic_blue/generation_smoke.json` and `.pkl` (not a win-rate sample).

## Verification and continuation

- 133 engine/shop/pool/seed-search tests passed.
- Existing early-seed, repaired-Ante-1, Baron and Mouth regressions passed
  after the debuffed-card fix.
- Focused runner/pool tests passed (20 tests).
- New shop-search tests verify state isolation, independence from draw order,
  purchase selection, generation configuration and chunk seed uniqueness.
- 90% is not achieved. Finish current evaluations, inspect losses, improve and
  compare candidates, then validate the selected default on fresh games.

Current process handles at time of writing (poll before assuming status):
100-game v3 benchmark: exec session `52661`, log `/tmp/heuristic-search-v3.log`.
Generation smoke: exec session `70158`.
Generation-related supervised tests: exec session `21502`, log
`/tmp/heuristic-generation-tests.log`.
New search/generation tests: exec session `21612`.


## Reproducibility follow-up

The generation smoke and benchmark seed 0 both won but had different action
counts (241 vs 243). Investigation found `flow._card_nominal` and
`scoring._card_nominal` used Python memory addresses as sorting tie-breakers.
These differ across deepcopies/processes/observation allocation and violate
reproducible scoring lookahead. The installed game uses persistent creation
order (`Card.unique_val`, card.lua lines 40 and 954). Replaced address sorting
with the existing stable `PlayingCard.reward_uid`. Stone rank sentinels also
use this identity. Raised Fist now follows the original last-lowest-held-card
scan, instead of breaking tied ranks by object addresses.

Two new tests cover deepcopy-stable ordering and Raised Fist's debuffed last
low-rank tie. They pass. 93 selected engine/heuristic tests also passed before
the final Raised Fist adjustment. Eight changed Cython modules are rebuilding
in exec session `44081`. Re-run tests and benchmark/replay parity on the compiled
result; v3's 44/100 predates this ordering fix and is not evidence of the final
current source behavior. The 95% Wilson interval on that tuning run was
34.7%–53.8%.

Further likely shop-search improvement: its synthetic hand should be removed
from a synthetic draw pile (currently the copied real draw pile remains), so
Blue Joker sees a representative remaining deck size. Include the full public
round targets and hand/discard counts in cache keys. Do not infer completion
from these plans; implement, evaluate and compare. After reproducibility,
continue improving hand/discard decisions, scaling and shopping toward 90%.

Latest continuation checkpoint:

- Cython rebuild finished successfully for the eight changed modules (the tool
  listed fourteen total installed extensions). Python now imports updated
  compiled engine, heuristic and runner modules.
- Previous generation-related supervised tests completed successfully.
- Compiled regression suite passed: 102 tests, log
  `/tmp/heuristic-compiled-tests.log`.
- New 100-seed benchmark with stable ordering is running: exec session `42336`,
  log `/tmp/heuristic-stable-v4.log`, output
  `eval/heuristic_blue/stable_order_v4.jsonl`.
- Exact action-by-action parity between no-observation and recorded generation
  for Blue/White seed 0 is running: exec session `60939`, script
  `/tmp/pylatro-replay-parity.py`, log `/tmp/heuristic-parity-v4.log`, result
  `eval/heuristic_blue/replay_parity_v4.json`. This is stronger than matching wins
  alone. Investigate any first divergence; do not weaken the assertion.
- The prior goal turn made concrete progress (engine fixes, new shop policy,
  completed 44/100 benchmark, generation fixes and tests), not a no-progress
  turn. Goal remains active, far below 90%.

## Subsequent shop, scaling and cache experiments

All results below are tuning games on Blue Deck, White Stake, seeds starting at
0. Only GAME_WON after the Ante 8 boss counts. Held-out seeds 10000–10999 remain
unused. The 90% objective is not achieved.

- Stable-order v4 finished **44/100**. Exact generation parity passed on seed 0:
  227 decisions in each path, identical actions/states and outcome.
- Realistic sampled draw piles and Card Sharp opening (v5): **24/50**.
- Stricter survival/reserve policy (v6): **24/50**.
- Safe permanent scaler growth (v7): **25/50**.
- Boss-aware safety and counterfactual planet purchases (v9): **48/100**.
- Stop lone Crafty from forcing Flush; avoid needless Summit discards when a
  Green Joker build already wins (v10): **51/100**, Wilson 95% 41.3–60.6%.
  Artifact: `eval/heuristic_blue/plan_scaling_v10.jsonl` and summary.

`models.py` now supplies an isolated, alias-preserving dataclass deepcopy that
shares immutable scalars. Sample pickle parity and explicit alias/isolation
regressions pass; microbenchmarks improved copying roughly 1.8–2.7x. This is a
performance change, not a scoring simplification.

Search samples use their own RNG, a public deck sorted independently of draw
order, and a synthetic remaining draw pile. Boss forecasts apply card debuffs,
Manacle, Needle and Water effects. Planet purchases simulate actual use,
including Constellation. Growth uses surplus visible cards while preserving a
visible finisher. Both generation paths receive deck/stake/search/growth
configuration and record it in metadata. Legacy remains the default pending a
sufficiently strong validated policy.

The next income experiment (v11) removes unconditional economy bonuses from
scoring utility. Income receives a limited-horizon value only when survival is
sufficient, including after a proposed replacement. Focused tests check buying
Golden Joker with an established scoring engine and rejecting it when weak.
This benchmark is running in session `3628`, log `/tmp/heuristic-income-v11.log`,
output `eval/heuristic_blue/income_v11.jsonl` (100 games).

V12 adds Mail-In Rebate discards that preserve an already-winning hand, and
counterfactual Buffoon/Celestial pack choices including Red Card skip value.
An actual-runner test confirms the discard earns $5, spends no hand, and still
wins. Pack selection has a state-isolation/portfolio regression. This run is
in session `78217`, log `/tmp/heuristic-income-packs-v12.log`, output
`eval/heuristic_blue/income_packs_v12.jsonl` (100 games). Wait for complete
results; partial completion rates are biased toward quick losses.

New correctness issue: score-cache keys omitted joker editions. Reproduction:
Foil changed the true score 608 -> 1008, while a warm cache incorrectly retained
608. V13 adds editions and scoring context to keys, including hand card
identity/editions/seals/bonuses, draw/deck lengths, boss identity, Eye/Mouth
constraints, round targets, probabilities and consumable usage. Regressions for
Foil and Blue Joker draw-pile changes pass. The selected engine/shop/heuristic
suite passed 96 tests (`/tmp/heuristic-v13-tests.log`). Rebuilding the heuristic
Cython extension is running in session `89937`, log
`/tmp/heuristic-compile-v13.log`; verify its completion before fresh runs.

Since v10, each benchmark saves source snapshots (`.sources.tar.gz`) and
source/binary hashes before workers launch. Since v12, traces copy mutable
joker extras and final score is captured after the last action, with the
previous score explicitly named `score_before_final_action`. Older `final.score`
values were before the last action and must not be read as final loss totals.

Unrelated pre-existing model/training changes remain untouched. Next: compare
v11/v12 paired outcomes; run compiled v13 tests and generation parity; inspect
remaining early draw decisions and late scaling failures. No final holdout or
90% claim is warranted yet.

## Correctness and rollout continuation (versions 14–20)

The unrelated Westminster-seat request was explicitly withdrawn by the user.
They said to ignore it and resume the heuristic goal. Continue the original
90% objective. `get_goal` currently reports paused (the API cannot resume that
status); work is continuing under the user's explicit instruction.

Completed older-rule experiments: v11 income **46/100**; v12 pack search
**43/100**; v13 cache-corrected compiled policy **43/100**. Neither improved on
v10's 51/100. These runs also inherited the invalid boss-reroll availability
below, so none is final validation under corrected rules.

V14 allows an early weak suit-joker build to preserve four-card flush draws
instead of forcing an unleveled Pair. Seed 37 changed from Ante-1 loss to an
Ante-8 win (one diagnostic, not a win-rate estimate). Its complete trace is
`early_draw_seed37_v14.jsonl`.

Found and corrected two engine/action-rule problems against the installed
Balatro Lua source (Balatro.exe ZIP, read only):

1. Hook used to discard selected playing cards before scoring. Upstream moves
   cards to G.play before `Blind:press_play` (`functions/state_events.lua` 483–488;
   `blind.lua` 464–483). `flow.play_cards` now moves cards first; Hook discards
   only held cards through `discard_cards(..., hook=True)`, firing Mail, Purple,
   Green etc. The regression previously turned a selected Flush into High Card;
   it now keeps all five scoring cards and earns Mail cash from the two held 2s
   without consuming a discard.
2. Paid boss rerolls require Director's Cut (once per ante) or Retcon (repeatable),
   with available cash/credit. The engine and both action masks now enforce the
   same helper. Boss-tag rerolls remain free. Source:
   `functions/button_callbacks.lua` 2784–2805. Updated old reroll fixtures to own
   the voucher; added no-voucher/repeated/credit/tag cases. Heuristic shop reserves
   now require an actually available reroll, and search can buy Director's Cut.

A third missing action is fixed: sell-joker actions are legal during hand play
in both masks (existing action IDs/execution paths are reused). The heuristic
can sell the least costly scoring loss under Verdant Leaf. Actual-runner test
sells Golden, preserves Blue/Half, clears all card debuffs, spends no hand, and
checks full/fast mask parity. No change to action-space dimensions.

`heuristic_simulation.copy_for_scoring` shares only untouched, read-only deck
cards for a single play probe, while copying hand/play cards, possible refill
cards, mutable lists and other state. Both score estimation and joker-layout
probes use it. Fifteen scenarios compare complete resulting state with a full
copy (including DNA, Hiker, Vampire, Hook/discard effects) and assert live-state
isolation. Copy microbenchmark ~1.74x faster than the already optimized generic
copy. Do not use this helper for blind resets, discards or consumable actions.

V16 short forecasts now sample first/middle/final hands, value Burglar's extra
hands (including Needle), remove discards with Burglar, restore Riff-Raff's
next-blind scoring value, and retain the established Celestial hand plan.
It finished **29/50**, still with old reroll eligibility. V17 adds Eye/Mouth
sequence constraints and Chicot-aware boss forecasts. Focused tests cover
Acrobat, Burglar, Needle and Eye/Card Sharp interactions.

Latest Square fix: its unconditional 35%-score growth allowance could replace
an immediately winning final hand with a losing four-card play. A regression
reproduced it. Four-card growth now requires sufficient scoring pace and cannot
replace a stronger final-hand win. 69 growth/shop tests pass after this change.

Fresh corrected-rule baseline v18 is running from an isolated snapshot:
`/tmp/pylatro-benchmark-v18`, session `71941`, log `/tmp/heuristic-legal-v18.log`,
output `eval/heuristic_blue/legal_boss_v18.jsonl` (100 games). This includes legal
rerolls, Hook, Leaf selling, and Eye/Mouth forecasts, but predates the Square fix.
The copied source/binaries prevent workspace edits/builds from changing workers.

New experimental `shop_policy="rollout"` compares portfolios using three public,
independently seeded complete-blind simulations, including draws, discards,
start-of-blind triggers and scoring changes over the round. It extends the
short evaluator and keeps its future option values. The simulation controller
uses a very high termination target to measure full-round capacity; the policy
must still plan its discards against the real target. Initial v19 incorrectly
used the high target for decisions too, which burned Banner/Green discards;
this flaw is identified and corrected in v20, with a Banner regression.

- v19 diagnostic 20-game run: session `63223`, log `/tmp/heuristic-rollout-v19.log`,
  output `round_rollout_v19.jsonl`. Do not promote that flawed variant.
- v20 corrected 20-game run: session `58993`, log `/tmp/heuristic-rollout-v20.log`,
  output `real_target_rollout_v20.jsonl`. Tests check live-state isolation,
  independence from both deck/draw order and useful Banner value. Both passed.
- Search remains the existing short forecast; rollout is opt-in. Legacy remains
  the public default. Final promotion awaits the actual requested win rate.

Compilation status: v17 build completed, updating flow and blind. Its heuristic
and fast-runner extensions predated the subsequent Leaf/action changes, so those
stale binaries were moved to `/tmp/pylatro-heuristic-original/*v17_before_leaf*`.
Current workspace imports heuristic and fast_runner from .py; engine flow/blind
from updated .so. Other engine extensions remain installed. Rebuild modified
modules again only after settling the next candidate and verify source/binary
parity. Source v13 generation replay parity passed (71 decisions), but a fresh
parity test is required after the engine/action changes.

Other checks: 169 engine/mask/shop/heuristic tests passed after reroll rules;
65 mask/layout/growth tests passed after Leaf selling. Latest focused Ruff and
`git diff --check` passed. The 90% goal remains unfulfilled; reserved final seeds
10000–10999 are still untouched.

### Continued experiments v21–v29

V18 completed **46/100** (25/50 and 11/20 on the first panels). V20 whole-round
forecast completed **12/20**, versus 11/20 for v18; this small difference does
not establish improvement and costs about 28 minutes for 20 games. V19's
incorrect decision target remains rejected.

Economy Tag now pays its immediate doubling (capped at $40), matching installed
Lua; previously skipping appended a tag that never executed. The policy now
requires at least $25 and both scoring roles to skip for it. Other tag effects
remain incomplete. The compiled blind extension was removed before this change.

V22 eight-sample public-card discard search completed **21/50**. V23's scoring
pace gate recovered **23/50**, still below v18's 25/50. Seed 33 improved from an
Ante-1 loss to Ante 7, but that diagnostic did not generalize. V28 removes the
call from the policy; the experimental helper and its isolation/order tests
remain available. This removal is being compared directly against v27.

V24 opens Buffoon packs with a full roster and can sell a weaker joker inside
the pack before claiming the better replacement. Both masks expose existing
sell actions during packs; action dimensions are unchanged. Actual-runner
regression verifies replacing Joker with Cavendish. V24 completed **26/50**.
The benchmark now preloads lazy policy modules before forking and traces shop
offers, pack choices and the forecast behind decisions. Its workers retain
their loaded sources while subsequent variants are edited.

V25 removes Egg's fictitious spendable income and raises early shop safety
from 0.8 to 1.1 times the target with one hand reserved. A trace showed it selling
Scholar for Egg while Ice Cream melted, then losing Water with cash unspent.
Panel: `survival_margin_v25.jsonl`, session 55471, `/tmp/heuristic-margin-v25.log`.

V26 replaces constant scaler bonuses with a bounded eight-round projection of
current owned jokers, rescored as a complete portfolio. Only valuation uses
projected stats; immediate survival uses unchanged current stats. Existing
progress reduces marginal growth value; the projection vanishes at Ante 8.
It lowers the purchase/interest penalties and prevents projected growth from
justifying an immediate scoring downgrade below survival needs. Constellation
now claims off-plan planets from an already opened pack. Focused suite: 92 pass.
Panel: `projected_growth_v26.jsonl`, session 94217, `/tmp/heuristic-growth-v26.log`.

V27 fixes boss disabling against installed `blind.lua` `Blind:disable`:
Water/Needle restore the resources removed on entry; Manacle restores its slot
and draws; card/joker debuffs, facedown flags and forced selections clear;
Wall/Violet Vessel targets halve/third. Disabling is idempotent. Chicot and
Luchador use the same helper. Selling Luchador can settle an already beaten
reduced target immediately, consistently in FastRunner and the environment.
The heuristic sells Luchador in an active boss and values its boss relief in
shops. Before-boss purchases now compare boss-specific portfolio values, not
just a boss-specific baseline against ordinary-blind purchase values.
Fourteen new engine regressions cover counters, resources and target settlement.
Source-focused suite: 104 pass. Runtime/flow/blind were rebuilt successfully.
Panel: `boss_counters_v27.jsonl`, session 14314, `/tmp/heuristic-boss-v27.log`.

V28 removes the failed sampled discard experiment, retaining v27 otherwise.
Panel: `without_sampled_draw_v28.jsonl`, session 74638,
`/tmp/heuristic-no-draw-v28.log`.

V29 lets a funded ($25+) build use a spare penultimate hand for growth through
Ante 5 when it still holds a visible 1.35x finisher. Baron no longer sacrifices
an immediately winning play to preserve Kings for later. Both have focused
regressions; the growth/shop suite passes 28 tests.
Panel: `extra_safe_growth_v29.jsonl`, session 61214,
`/tmp/heuristic-extra-growth-v29.log`.

Broader compiled checks: 122 passed, one failed legacy Red Deck seed test
(`test_heuristic_repaired_ante_one_seeds_reach_ante_two`, seed 292). Comparing
current vs isolated v18 before deciding whether the new changes caused it.
Logs: `/tmp/heuristic-red292-v29.log`, `/tmp/heuristic-red292-v18.log`.
Fresh training/evaluation decision replay parity is running in session 68354,
`/tmp/heuristic-parity-v29.log`, output `replay_parity_v29.json`.

Current execution: heuristic/fast_runner are Python source; runtime/flow/blind
have current compiled extensions. Other engine extensions remain installed.
Source snapshots live under `/tmp/pylatro-heuristic-original/`; every panel has
a source archive and manifest. No experimental policy is promoted to the
default, no final holdout seed has run, and the 90% goal is still unmet.

### Continuation v30–v36

V25 completed **26/50**, equal in aggregate to v24. V26's first panel remains
in progress; its source and every extension were restored by manifest hash to
`/tmp/pylatro-benchmark-v26` so seeds 50–99 can run without later edits changing
the candidate. Second panel: session 47184, `/tmp/heuristic-growth-v26-second50.log`,
`projected_growth_v26_second50.jsonl`. These are development seeds, not the
reserved 10000–10999 holdout.

The legacy Red Deck seed 292 regression was caused by the stricter Square
growth rule, not the new boss disabling. It cleared Big one hand sooner but
entered Pillar with no Square growth and lost. V30 restores the 35% relative
score allowance only inside the new safe-growth branch (Ante <=6, >=3 hands,
one spare hand's pace covers 1.25x the remaining target). It still cannot replace
a winning final hand. Seed 292 reaches Ante 4 again; all 24 growth/generation
regression tests pass. Diagnostic: `red292_square_v30.jsonl`.

V29 training/evaluation replay parity **passed**, 215 identical decisions,
both won. Saved `replay_parity_v29.json`.

V31 exposes legal shop consumables in both action masks. Verified against
installed `card.lua` `Card:can_use_consumeable`: planets and the listed
no-hand-target Tarot/Spectral effects can be used in a shop; leftover hand
cards cannot be targeted there. The heuristic uses planets and safe utility/
money effects before choosing its next purchase. Actual-runner tests check
Hermit cash is spendable in the same shop and Mercury updates both Pair and
Constellation immediately. Shop/env/grammar suite: 153 pass. Panel (20 seeds):
session 8390, `/tmp/heuristic-shop-use-v31.log`, `shop_consumables_v31.jsonl`.

V32 uses 12 public hand samples with stable per-card hash priorities. Adding a
card no longer reshuffles every old sample. It also sorts sampled hands using
the live engine's ordering and models Crimson Heart's joker debuff after the
opening hand. Samples are independent of future draw order. Focused suite:
27 pass. Panel (20): session 10196, `/tmp/heuristic-stable-samples-v32.log`,
`stable_samples_v32.jsonl`.

V33 prevents future-only investments while immediate capacity is below target
(Riff-Raff remains exempt because it supplies next-blind jokers). It restores
the pace-gated sampled discard call: removing it was performing worse with the
new growth valuation, so the earlier v22/v23 conclusion was not transferable to
this portfolio policy. Panel (20): session 4860,
`/tmp/heuristic-growth-guard-v33.log`, `immediate_growth_guard_v33.jsonl`.

V34 values future growth on the same already-selected sample hands instead of
searching every hypothetical future hand again. This is a conservative
valuation approximation; current survival scoring is unchanged. On a five-
joker Green/Blue/Hologram/Half/Blueprint microbenchmark, value matched and time
fell from 0.471s to 0.226s. Added physical card IDs to the cache key so reused
agents cannot reuse a previous run's sample valuation. Focused suite: 27 pass.
No standalone v34 panel.

V35 funds the growth implied by the valuation: Standard packs for Hologram and
Celestial packs for Constellation can be bought with a $5 remaining buffer;
Red Card can buy affordable packs for its skip gain. Previously these cards
were bought for growth but the pack budget often prevented any growth. Two
new purchase regressions pass; shop suite: 27 pass. Panel (20): session 46577,
`/tmp/heuristic-growth-packs-v35.log`, `growth_pack_support_v35.jsonl`.

V36 (staged, no panel yet) aggregates each opening/middle/closing group into
round capacity with the correct hand counts, then takes the geometric mean of
round capacities. It avoids counting illegal repeated middle hands under Eye.
Card Sharp's opening probes now consistently have zero prior plays; the old
`sample > 0` condition incorrectly activated later opening probes, even under
Needle. The single-hand Needle/Card Sharp regression checks this explicitly.

Boss target reductions are now also shared by tokenizer observations, hand
candidate metadata, shop evaluation and the fast target helper. A regression
checks Chicot/Vessel gives 100000 everywhere. These source changes require the
latest extension rebuild: session 95653, `/tmp/heuristic-agent-build-v36.log`
(heuristic, fast_runner, hand_candidates, tokenizer). Old extensions are safely
saved under `/tmp/pylatro-heuristic-original/`. Engine runtime/flow/blind already
have current extensions. Other check sessions: 63610 consistency suite log
`/tmp/heuristic-consistency-v36-tests.log`; 20289 capacity tests log
`/tmp/heuristic-capacity-v36-tests-final.log`; 26587 replay parity log
`/tmp/heuristic-parity-v36.log`, output `replay_parity_v36.json`.

All changes remain experimental until full panels and independent evaluation
support promotion. The goal has not reached 90%; final holdout remains unused.

### Continuation v38–v42

Completed development panels: v26 scored **34/50** on seeds 0–49 and **32/50**
on the independently restored seeds 50–99, for **66/100** total. V27 scored
29/50, v28 31/50, v29 30/50, and v31 10/20. None meets the goal.

**Invalidate v32/v33/v35 win-rate comparisons.** Replay parity v36 failed
(234 versus 230 decisions, first difference 26). The public sample hash used
absolute process-global card IDs, which changed across fresh runs and worker
scheduling. These experiments were stopped. V38 hashes front key plus relative
occurrence ordinal instead; IDs establish only creation order among duplicate
fronts. Fresh-run, reversed-deck and add-one-card tests pass. Replay parity v38
passes: **227 identical decisions**, both runs win seed 3. Target/tokenizer/
candidate consistency suite: 124 pass; compiled core suite: 86 pass.

V38 combines reproducible samples, whole-round capacity, and purchase valuation
blended 75% ordinary / 25% boss before the final boss. Survival still uses the
actual upcoming boss. Cash-independent forecasts are reused after cash changes.
Panel: `repeatable_capacity_v38.jsonl`, `/tmp/heuristic-repeatable-v38.log`.

V39 fixes consumable generation capacity: consume the owned item by identity
before generating its replacements, so Fool can copy at full inventory and
High Priestess can create two planets when used alone. Shop policy buys and
immediately uses profitable Fool copies of Hermit/Temperance; Fool copying a
planet is evaluated with the generated planet applied. Vacancy suite: 92 pass;
Fool/shop/pool suite: 42 pass. Panel: `consumable_economy_v39.jsonl`,
`/tmp/heuristic-economy-v39.log`.

V40 avoids wasting paid boss rerolls with Chicot/Luchador and adds a capacity
check for dangerous bosses such as Eye. Final-Ante shop rerolls can spend the
last cash buffer. Reroll suite: 97 pass. Panel: `emergency_rerolls_v40.jsonl`,
`/tmp/heuristic-emergency-v40.log`.

V41 delegates modified hand classification to the engine for Four Fingers,
Shortcut, Smeared, Wild and Stone cards; normal fast classification now reaches
Flush Five and Flush House. Hand-rule suite: 78 pass. No separate panel.

V42 adds standalone shop sales for Stencil/Campfire when the counterfactual
improves value without reducing immediate score. Mature Stencil builds preserve
empty slots instead of automatically using Judgement; they sell it in a shop.
This addresses seed 8 generating a Faceless Joker mid-blind and halving Stencil.
Shop/draw suite: 108 pass in source mode. Extension rebuild and panel pending.

All changes remain experimental. The reserved holdout seeds 10000–10999 remain
unused, defaults have not been promoted, and 90% has not been achieved.

### Stone-card reproducibility correction and v45

V38 completed 38/50 and v39 39/50 on seeds 0–49. These results are now
**provisional under the old scoring engine**, not validation of the corrected
engine. V42 replay parity passed 227 decisions on seed 3, but that seed did not
cover the newly found Stone-card bug.

Comparing v39/v40 seed 11 exposed a decision difference in Ante 4 before either
changed reroll rule applied. Stone Cards received a negative stable ID, and
Odd Todd/Even Steven erroneously tested parity without checking the lower
bound. Their output therefore depended on global ID parity. Verified installed
`card.lua`: both require `get_id() >= 0`. V45 adds that missing bound. Eight
regressions cover both jokers and even/odd IDs; a forecast regression offsets
all physical card IDs by 101. Scoring/simulation/reproducibility suite: 33 pass.

Stopped unfinished old-engine panels: v38 second panel 4/7, v40 31/43,
v42 11/18, v43 compiled 3/3. Preserve their traces only for diagnosis. The first
v43 source run was also stopped before restarting in an isolated compiled
checkout; its archive is valid but its partial counts are not a completed panel.

V43 preserves the planned Mouth hand while exhausting Summit discards, falls
back to Pair/High Card when the intended opener is absent, and spends one card
per Summit discard with live Ramen. Shop/draw/growth suite: 117 pass.

V44 adds sampled discard counterfactuals for Banner/Green/Ramen at all antes.
The simulation applies actual discard penalties, prefers preserving scoring
when no sampled improvement exists, and still takes a final chance when the
only remaining play would lose. It never reads future draw order. Twelve draw
tests pass, including state isolation and final-chance behavior.

V45 combines these changes with corrected Stone scoring and limits the zero-
cash reroll buffer to an unsafe final boss. The unrestricted Ante 8 rule caused
v40 to diverge from winning v39 runs at Small/Big shops for seeds 7, 10 and 32.
New panel: `corrected_stone_v45.jsonl`, 100 seeds 0–99, eight workers, session
63012, `/tmp/heuristic-corrected-v45.log`. Seed 11 replay parity: session 18543,
`/tmp/heuristic-parity-v45.log`. Corrected compiled suite: session 32225,
`/tmp/heuristic-corrected-v45-tests.log`.

An isolated v39 policy with only the same Stone scoring correction is being
rerun for a fair baseline: `economy_v39_corrected_stone.jsonl`, four workers,
`/tmp/heuristic-economy-v39-corrected.log`. All source/binary hashes are archived.
The goal remains unmet; final holdout remains untouched.

V45 compiled corrected checks: **87 pass**. The formerly divergent seed 11
passes replay parity: **215 identical decisions**, both paths lose consistently
in Ante 7. This is reproducibility evidence, not a win-rate success. V45's first
18 completions were 14 wins; the complete 100-game panel is still required.

V46 (staged) searches legal additions to a selected scoring hand, preferring
more played cards only when score does not decrease. This enables Half Joker
to redraw a third card and avoids breaking a hand through one greedy padding
choice. Focused tests protect held Steel Kings. Applying it to legacy policy
caused Red seed 17 to die in Ante 2; restrict this experiment to search/rollout
until its aggregate behavior is evaluated. Full initial suite had 102 passes
and that one failure. Follow-up legacy regression and eight growth tests:
session 92765, `/tmp/heuristic-padding-v46-legacy-check.log`; final extension
build session 37590, `/tmp/heuristic-agent-build-v46-final.log`. No v46 panel yet.

### Paid vouchers, boss preparation, and v49

V46's legacy gate passes all nine follow-up checks; compiled V46 is installed.
V47 prepares a known dangerous boss while Director's Cut is still affordable
in an earlier shop, then reserves $10 for the reroll. Seed 25's trace motivated
this: it repeatedly passed an affordable Director's Cut, spent the cash, and
lost to the already-visible Needle.

The actual-runner voucher regression uncovered another engine bug: redemption
applied effects without paying. V48 deducts the offered voucher's exact displayed
price; direct grants without a shop offer remain free. Both training/evaluation
already use this shared path. Source voucher/controller/pool suite: 64 pass,
one skipped. Broader compiled scoring/shop/boss/consumable/pool checks: 105 pass.

Stop the pre-payment panels rather than treating them as validation. V45,
corrected v39, v46 and v47 artifacts remain diagnostic only. V48 starts fresh:
`paid_vouchers_v48.jsonl`, seeds 0–99, eight workers, session 75196,
`/tmp/heuristic-paid-vouchers-v48.log`. The isolated v39 policy now uses both
Stone and voucher corrections: `economy_v39_paid_vouchers.jsonl`, 50 seeds,
session 52783, `/tmp/heuristic-economy-v39-paid.log`. V48 seed-11 parity runs in
session 17933, `/tmp/heuristic-parity-v48.log`.

V49 reduces Blue Joker's forecast draw pile by three cards per preceding hand
and decays Ice Cream by its real per-hand chip loss. Middle probes represent
the middle of the round instead of always the second hand. These are bounded
approximations, not survival guarantees. Previously every sampled late hand
received the opening Blue Joker bonus. Regressions use identical public hands
to isolate resource depletion and verify the live run remains unchanged.
Panel: `depleting_chips_v49.jsonl`, seeds 0–99, eight workers,
`/tmp/heuristic-depleting-v49.log`; tests `/tmp/heuristic-depletion-v49-tests.log`.
No candidate has achieved 90%, no default promotion, no holdout use.

### V50–v53 continuation

V48 replay parity passed: **231 identical decisions**, both won seed 11.
The broader generation/heuristic/RNG regression suite passed **111 tests**.
The initial high partial win rate did not persist; the complete panels remain
necessary. At 48 completions V48 had 36 wins; V49 had 29/39.

V50 makes the valid-action fallback deterministic and independent of NumPy's
process-global RNG. Its 15 reproducibility tests pass. Formatting cleanup
leaves the touched heuristic modules Ruff-clean.

V51 lets a Stencil sale prioritize survival when current boss capacity is
below 1.15x target and the sale brings it above that threshold, even when the
joker would regain value after the boss. It preserves target reductions and
extra-hand effects when comparing the sale. The matched seed 8 diagnostic now
**wins** through Ante 8; `stencil_survival_v51_seed8.jsonl` is a targeted
regression run, not a win-rate panel. Shop suite: 43 pass.

V52 accounts for one owned Madness activating before a normal blind: increase
its multiplier and average the complete portfolio outcomes over every eligible
victim. The boss forecast does not apply that destruction. Alone, Madness gets
its free growth. Actual copied-state removal preserves passive effects and
normalizes capacity if Burglar is destroyed. Shop/rollout tests: 47 pass.
This remains an approximation for unusual multiple-Madness builds.

V53 removes legacy main-hand score discounts from search/rollout decisions.
Those rules were overriding exact engine scoring and the earlier Bus/Baron/
Blue-seal decisions. Seed 13 selected a 911-point Queen Pair that reset Bus
from 39 to zero, although an Ace High Card scored 996 and preserved it.
Legacy behavior remains available. Source suite initially had one incorrectly
constructed regression (the fixture activated Card Sharp for Pair, unlike the
failure trace); corrected fixture and final tests are running in session
42304/its follow-up log `/tmp/heuristic-oracle-v53-tests-final.log`.
Extension rebuild: session 23605, `/tmp/heuristic-agent-build-v53.log`.
No V53 panel has started yet. Goal remains unmet; holdout unused.

V53 final source suite: **131 pass**; compiled extension installed before its
100-game panel. Panel session 51601, `oracle_priority_v53.jsonl`,
`/tmp/heuristic-oracle-v53.log`. Its seed-11 replay matches all 173 decisions,
both lose. V49 extension stopped in session 4661 after 70 completions (49 wins),
while all seeds 0–49 were complete. Unfiltered first-panel extracts and parent
provenance are `paid_vouchers_v48_first50.jsonl` and
`depleting_chips_v49_first50.jsonl`; both are 38/50. No failures were excluded.

V54 aligns live hand estimates with the runner's automatic joker ordering.
A recorded Acorn opening was estimated at 10075 before ordering but scores
30690 after the legal order the runner will apply. The old underestimation
caused unnecessary discards that erased Banner. Search/rollout now call the
same pure order planner; discard simulations use the same estimator. Live
state remains unchanged. Ordering/draw/growth/reproducibility tests: 48 pass.
Panel: `order_aware_v54.jsonl`, 100 games, session 82091,
`/tmp/heuristic-order-aware-v54.log`. New reusable validation command:
`scripts/check_heuristic_replay.py --seeds 13 --output <result.json>` compares
complete evaluation and training decision streams, visible state and outcomes.
V54 parity session 84885, `/tmp/heuristic-parity-v54.log`.

V55 (no full panel yet) evaluates Verdant Leaf after the best legal joker sale,
instead of treating its debuffs as permanent. Chicot avoids unnecessary sales;
Luchador counterfactuals use the real sale effect, including Campfire. Leaf/
shop/rollout tests: 48 pass before the final sale-effect refinement; final shop
log `/tmp/heuristic-leaf-v55-tests-final.log`. Targeted seed 16 diagnostic:
session 71633, `/tmp/heuristic-leaf-v55-seed16.log`.
The goal has not been achieved; the reserved holdout is still untouched.

V57 restores main-hand preferences while retaining explicit Bus/Baron/Blue-seal
choices, and makes `_find_type_hand` return the requested type exactly. Its
152 focused checks passed. The 50-game panel `hand_plan_v57.jsonl` regressed;
several first divergences replace an existing hand by a same-type, same-score
hand with different kickers. This does not establish a better playing policy.

V58 adds a six-sample next-hand forecast after the selected play, including
Card Sharp activation, actual played-card removal and redraw. It randomizes
only the public remaining card pool with independent deterministic RNG, and
does not consult hidden draw order. For Ante 3+ play/discard pacing it uses
current score plus expected follow-up score for the remaining hands. Four
focused continuation regressions cover activation, rare-hand depletion,
state preservation, and invariance to absolute card IDs and future draw order.
Broader continuation/draw/growth/reproducibility/generation checks passed.
`continuation_v58_source_aborted.*` preserves an accidentally launched source
run; the compiled 50-game panel is `continuation_v58.jsonl`.

V59 prioritizes sampled chances of clearing the target when the final hand
cannot currently win, in every ante. It may spend Blue seals in that emergency
and retains forced cards. The seed-6 Needle diagnostic improves from a 648/800
loss to a 1020-point available play after two discards; the complete targeted
seed-6 game now wins through Ante 8 (`final_chance_v59_seed6.jsonl`). This is a
regression check, not a win-rate claim. Focused continuation/draw/growth/
reproducibility suite: 46 pass; touched modules are Ruff-clean.


V60 retains the existing selected hand when alternative-hand planning ends
with the same poker type and the same score. A regression demonstrates the
unnecessary change from Kings to Jacks with identical 480-point scores.
Focused suite: 47 pass. Panel: `stable_hand_ties_v60.jsonl`.

V61 removes V58 continuation pacing after its stopped development panel could
no longer match the established 38/50 baseline. It retains final-chance draws
and stable same-type ties. Boss reroll decisions now require both insufficient
capacity and a meaningful loss relative to ordinary boss capacity. Rerolling
a weak build against an effectively neutral Serpent previously replaced it
with Eye. Replaying that recorded seed-71 position with the new decision
clears Serpent at 73992/70000 and enters Ante 8, rather than losing to Eye.
Shop budgeting uses the same risk decision. Comfortable Vessel forecasts do
not waste a reroll. Focused shop/rollout/draw/growth/reproducibility suite:
95 pass. V61 completed 37/50; its seed-6 replay parity passed.

A separate diagnostic checked buying Luchador early in seed 53's Ante-8 shop.
Removing an existing joker would leave the best forecast below 86000 against
100000. No new purchase rule was added for that case; the apparent counter
would not fix the underlying scoring shortage.


V61 recorded/evaluation replay parity on seed 6: all 196 decisions match and
both games win. The compiled decision regressions pass.

V62 reduces redundant shop forecasts by retaining lifetime hand counts only
for Supernova/Obelisk/Ox, while keying the computed main hand separately. Every
probe resets played_this_round, so that field is omitted. Rotating Ancient/
Idol targets are keyed only when their joker is owned. Loyalty retains its
global hand clock and creation offset. Cache equivalence and sensitive-history
regressions pass in the 56-test shop suite. A controlled 12-forecast sequence
returns identical old/new values, with 12 versus 1 cache entries and elapsed
0.962 versus 0.088 seconds. This is a forecast microbenchmark, not a claimed
full-game speedup. No benchmark policy has been promoted to default.


V60 first development panel finished **41/50**, but its second panel is weaker;
this is not a 90% validation. V61 (without continuation) finished **37/50**.
The latter's worse result, after V60 resolved equal-score hand substitutions,
means the earlier continuation ablation was confounded. V63 restores V60
continuation while retaining the boss-risk check and V62 forecast cache.
Source suite: 105 pass. Panel: `combined_v63.jsonl`.

V64 preserves the normal interest reserve before buying uncertain Celestial
packs when the build already meets the survival margin. Unsafe builds can
still invest, and Constellation/Red Card scaling exceptions remain. The prior
policy bought packs down to $10 even when comfortably safe. Three budget
regressions plus the shop/rollout suite: 61 pass. Panel:
`interest_packs_v64.jsonl`, 50 games, four workers.

V65 makes scoreless plays under a Mouth Straight lock preserve useful straight
draws. It compares bounded public-pool continuations for possible redraws.
The recorded seed-38 failure at 8960/10000 now clears at 19950 and enters Ante 5.
This is a position replay, not a fresh-game win-rate result. Targeted full seed
38 run: `mouth_redraw_v65_seed38.jsonl`. Draw/continuation tests: 23 pass.

Exhaustive final-hand diagnostics for V60 failures 11, 14, 25, 28, 32, 35 and
38 found no better legal immediate play than the selected one. No broad
exhaustive-play policy was added based on those diagnostics.

V66 prevents enhancement Tarots from overwriting an already enhanced card,
prioritizes the actual scoring cards over kickers, and gives held-card effects
(Steel/Gold) to cards outside the planned play. Stone prefers low non-scoring
cards. It waits when no base card is available. Engine classification runs on
an isolated probe to avoid mutating even live derived caches. Eight Tarot
variants plus scoring-pair targeting and waiting regressions pass. A seed-25
position replay preserves Glass but still loses that Eye, so no survival gain
is claimed from that diagnostic. Broader tests and extension build are running.


V63 finished **40/50**, so it does not beat V60's first panel. V65's complete
seed-38 diagnostic now wins through Ante 8. V66 source checks: **114 pass**;
compiled target/continuation checks: 15 pass. Its fresh 100-game development
panel is `enhancement_targets_v66.jsonl`, eight workers. V64's 50-game pack
budget panel is still running. Latest complete 100-game result remains 74%,
and the 10000–10999 holdout remains untouched.

V67 tests spending against the next blind's target instead of the largest
boss target still ahead in the ante. There is another payout and shop after
each ordinary blind; the earlier forecast could trigger emergency rerolls
before a safe Small blind because Wall/Vessel was waiting later. Boss counter
planning still uses the actual boss target. Focused budgeting tests and the
shop/rollout suite are being checked before launching its panel.


V64 completed **36/50** against V63's **40/50**. The tighter Celestial-pack
reserve was reverted in V68; it left some successful control builds without
needed hand levels. V66 completed **74/100**, matching V60 on the full panel
but with different outcomes. V67's next-blind budget experiment stopped once
its losses prevented improving the best first-panel result; it still included
the later-reverted V64 pack rule. Its initial zero-completion attempt is
archived as `next_blind_budget_v67_stale_flag_aborted.*`; the restarted version
ignores a previous blind's disabled flag when forecasting the next target.

V68 adds bounded level growth from Space Joker (two plays per projected round,
scaled by actual proc probability) and Burnt Joker (one level per round for
High Card, 0.75 for less certain types). Burglar/no-discards disables Burnt's
projection. Future type selection can change after level growth, but the
immediate survival score remains unchanged. Burnt deliberately spends its
first discard on the established hand type, protecting Blue/Steel cards and
avoiding final-hand/final-boss growth detours. Focused suite: 128 pass, with two
additional immediate-capacity regressions and the integrated Burnt decision
check passing. The compiled 100-game panel `level_scalers_v68.jsonl` is running.

V69 updates the existing opt-in complete-blind rollout to use the current
search hand policy, independent trial seeds, and the appropriate real target
for discard planning. State-isolation/draw-order/Banner tests pass. Its bounded
10-game, two-worker runtime/quality experiment is
`current_policy_rollout_v69.jsonl`. It is not the promoted policy.

V68 completed **82/100**, with all 18 losses at bosses. Recorded/training
replay parity for seed 32 matches all 194 decisions (both lose). The separate
100-game development panel is `level_scalers_v68_fresh100.jsonl`, seeds 100–199,
eight workers. Its archived search source matches its manifest. The unused
rollout module differs from the original V68 manifest because V69 updated it;
the search policy and its dependencies remain the V68 candidate.

V70 allows affordable Tarot upgrades and Arcana packs before an unsafe boss,
while preserving money for an available useful boss reroll. Shop/rollout tests:
76 pass. The recorded seed-17 shop replay buys Steel from a Jumbo Arcana pack,
improving the sampled forecast slightly, but still loses at 68934/70000. This
diagnostic is not a survival improvement. Its 50-game panel is
`emergency_tarot_v70.jsonl`.

V71 additionally compares actual Tarot-pack claim outcomes on isolated states
before an unsafe boss. Random effects use independent forecast seeds. Money
and stored consumables have bounded secondary value; Red Card skip is evaluated
through the actual engine transition. Focused tests are running.

V71 shop/rollout tests finished: **78 pass**. Its bounded 50-game panel is
`tarot_pack_capacity_v71.jsonl`, two workers. V69's full-blind rollout experiment
was stopped for runtime after only two completed games (one win); this is not
a comparable win-rate estimate.

V72 isolates held-Steel discard valuation on top of V68, without the still
unproven V70/V71 shop changes. Discard sampling includes alternatives that keep
live Steel cards and runs when such cards are held, even without Banner/Green/
Ramen. It still compares discarding Steel, allowing a stronger alternative to
win rather than protecting it absolutely. The recorded seed-74 boss position
now clears at 24424/22000 and enters Ante 6; baseline lost at 17020. This is a
position replay, not a complete fresh-game win. Tests and compilation are running.

V72 compiled draw/continuation/growth checks: **35 pass**. Its 50-game panel is
`held_steel_v72.jsonl`, two workers.

V73 experimented with evaluating the future-hand cost of Banner/Green/Ramen
discards. All 35 focused tests pass, but replays of 14 recorded V68 losing boss
positions saved none and worsened several scores. It was reverted to V72's
discard implementation without running a broad panel. Source saved locally as
`/tmp/pylatro-heuristic-original/heuristic_draw_v73.py`.

V74 targets Death at a valuable rightmost source instead of choosing two weak
cards and duplicating one. It compares the strongest structural improvements
using actual consumable transitions on isolated states, preserving direction,
and waits when no useful legal duplication exists. In seed 74's earlier boss
position it duplicates Red Seal Steel and clears 22215/22000; baseline lost,
while V72's Steel-retention change alone also saved that position. This is not
an additional saved full game. Tests and compilation are running.

V74 source and compiled consumable/growth/draw/continuation checks: **47 pass**.
The current working policy keeps V68 shop logic while V70/V71 are measured;
their associated experimental tests are saved with their source backups until
promotion or rejection. Their already-running benchmark processes use their
frozen imports and archives.

V75 is a separate, not-yet-installed shop-budget prototype at
`/tmp/pylatro-heuristic-original/shop_search_v75.py`. During Antes 3–6 it lowers
the safe-build reserve to $10 when bounded projected scoring still falls short
of the ordinary boss two antes ahead. It retains the normal reserve for a build
whose existing growth is sufficient. This tests continued investment rather
than treating immediate survival as evidence that the whole run is secure.

V70 stopped after **32 completed games, 25 wins**: seven losses mean it can
at best tie V68's 43/50 and cannot improve it. This is an incomplete panel,
not a full-panel win rate. The combined Tarot-pack V71 panel remains running.
V74's 50-game panel is `death_targets_v74.jsonl`, four workers. Current shop,
rollout, reproducibility and two budget-prototype checks: **89 pass**.

V75 is now installed over V74's hand policy. Shop/rollout checks pass after
updating the earlier next-blind budget regression to require that a later Wall
does not change the Small-blind decision; general investment for later antes
can now independently justify rerolling. Its panel is `future_budget_v75.jsonl`,
50 games, four workers.

V76 is a separate, uninstalled prototype at
`/tmp/pylatro-heuristic-original/shop_search_v76.py`. It compares free alternative
planet claims when a Celestial pack lacks the main planet, and allows an
affordable Celestial pack before an unsafe boss. The seed-115 position replay
improves 416/600 to 502/600 but still loses. No survival gain or broad win-rate
result is claimed for this prototype.

V68's fresh development panel completed **68/100** in 1570 seconds (95% Wilson
58.3–76.3%). Combined with the original 82/100: **150/200 (75%)**. Every seed
0–199 appears exactly once in `level_scalers_v68_combined200.jsonl`. Runtime
manifests differ only in the unused opt-in rollout module. V71, V72, V74 and
V75 are still being measured. Reserved validation seeds remain untouched.

V76 is now installed and its shop/rollout suite passes **75 checks**. The
new regression confirms claiming a useful Two Pair planet when Mercury is
absent, without mutating the live state; another covers an affordable
Celestial pack before a failing first boss. The full 100-game development panel
is `planet_pack_v76.jsonl`, four workers, on seeds 0–99. Its parents are V75's
future budget and V74's hand/Death policy. It has no completed win-rate result.

Current executable source: V74 compiled heuristic (Steel retention and Death),
V72 discard sampler (V73 reverted), V74 growth module, and V76 shop search.
Running panels retain their original imports. Next: finish these measurements,
compare candidates on both halves of the development set, and extend promising
candidates to the second 100 seeds before touching the reserved holdout. No
default policy promotion or final training-data generation is justified yet.

Pool audit: the existing `get_current_pool` respects each center's `unlocked`
flag. Burnt Joker, Blueprint, Brainstorm and Hanging Chad are locked in the
bundled data and are absent from these benchmark pools. No V68 development
game among seeds 0–199 owned Burnt; the V68 improvement therefore cannot be
attributed to its Burnt behavior. Space appeared in 27 games (23 wins), but that
is a descriptive association, not a causal estimate. User clarification about
current versus fully unlocked pool was requested; no unlock flags or game
rules have been changed, and all running panels keep the current pool.

V71 stopped at **28 completions, 21 wins** (seven losses); V75 stopped at
**22 completions, 15 wins** (seven losses). Neither can exceed the existing
43/50 first-panel result. V76 remains active despite sharing V75's budget
component, because its additional planet behavior changes outcomes; it won
seed 0 through Ante 8, where V68/V75 lost, but also has other losses. It is
not yet a measured improvement over the baseline.

V74 stopped at **35 completions, 27 wins** (eight losses), so its first-panel
ceiling is 42/50. Death targeting remains in later candidates only while those
combined candidates are being measured; it has not been promoted as a measured
standalone improvement.

Two bounded shop refinements follow the recorded seed-71 Ante-3 shop, where
the old policy left an affordable Negative Hologram unbought. V76 instead
rerolls it away: its future-shortage reserve permits spending, while the old
purchase/interest penalties still reject the offered growth. V77 extends the
Hologram/Constellation growth horizon from eight to at most twelve remaining
rounds (other projections remain unchanged). It buys the offered Hologram;
76 shop/rollout checks pass. Panel: `xmult_horizon_v77.jsonl`, 50 games, four
workers. A continuation of the recorded seed-71 shop is also running, separate
from the full-game panel.

V78 is a separate prototype over V76 that retains the eight-round growth
horizon and instead aligns purchase penalties with the future-shortage spending
decision. Both prototypes buy that offered Hologram; survival is not inferred
from the purchase alone.

V78 is now installed. Its 76 shop/rollout checks pass, including the available
Negative Hologram regression. Panel: `purchase_budget_v78.jsonl`, 50 games,
four workers. Root executable uses V74 compiled hand policy, V72 discard
sampler, V74 growth/Death helper and V78 shop logic. V77 remains a separate
frozen experiment rather than being combined with V78.

V72 is being extended on seeds 50–99 in `held_steel_v72_second50.jsonl`, two
workers, because its known saved seed-74 position lies outside the first panel.
The two V72 runtime manifests match exactly, including binary hashes. Root
files were restored to V78 after the second panel archived and imported V72.

The V77 seed-71 continuation from the recorded Ante-3 shop completed with a
win through Ante 8 (102771 points; Hologram reached x3.25). Baseline lost at
90962. This is a continuation from an old-policy position, not a fresh full-game
result. A bounded runtime trace confirmed normal state advancement, typically
roughly 1–3 seconds per shop decision with six jokers, and was stopped after the
original diagnostic completed. The long runtime was not evidence of a hang.

Teacher-information audit: `_cached_best_hand` and `_estimate_hand_score` read
rank/suit/front identity even for face-down hand cards; the model tokenizer
hides those identities. Current benchmark rates therefore describe the existing
full-state heuristic, not a demonstrated observation-limited policy. Clarification
was requested about this target boundary. No information masking or scoring
rules were changed while the running comparisons complete. This distinction
must be settled before claiming final teacher/model suitability.

Benchmark provenance now freezes `load_game_data()` before workers fork,
hashes/archives `game_data.json` and the benchmark script, and writes a startup
`.profile.json` describing deck/stake, seed range, full-state teacher access
and excluded locked jokers. Older artifacts have not been retroactively labeled
as containing this metadata. Syntax/Ruff checks and a metadata-only smoke check
verified the archived hashes and pool profile; that smoke used synthetic task
completion under `/tmp` and evaluated no games or win-rate evidence. Existing
live panels continue with the script and imports captured when they launched.

V72 first panel completed **43/50**, matching V68's first-panel outcomes. Its
second half continues; this is not evidence of 86% generalization.

V79 isolates a weaker-outcome shop forecast over V72's hand policy and V68's
shop rules, without V74 Death or the V75–V78 budget/planet experiments. It uses
the geometric mean of the weaker half of sampled round totals. In the recorded
seed-8 Water loss, the old 95727 forecast came from totals ranging 42576–354360;
the weaker-half forecast is 51340, against the actual loss at 56766/70000.
Other errors remain (for example Banner draw management), so this diagnostic
does not establish improved gameplay. All 71 baseline shop/rollout checks pass.
Its panel is `weak_draws_v79.jsonl`, seeds 100–149, two workers; V68's matched
control is **33/50**. New startup profile and source/game-data archive verified.
Root files were restored to V78 after the V79 workers froze their imports.

V77 stopped after **42 completions, 32 wins** (ten losses): the saved seed-71
continuation did not translate into a better first-panel policy. V78's closer
comparison is being allowed to finish despite seven losses; V76's full
100-game panel and V72's second half continue. Paired outcomes, not comparisons
between differently completed seed subsets, are being used for early diagnosis.

V78 completed **42/50**, below V68's **43/50** on the same seeds. It has not
been promoted. A replay audit of regressed seeds 1, 4, 16 and 22 found virtually
no late paid off-plan planet purchases; that proposed explanation does not
justify a new spending restriction. Seed 1 correctly opened the Mouth with
Pair, so its loss also does not support another Mouth hand-lock adjustment.

A separate remaining-blind rollout pilot compares up to four legal plays or
discards using three paired independent draw-pool permutations. Each simulation
finishes the blind with the current teacher. It preserves the live state and
does not reuse the actual future draw order or RNG. From V68's recorded boss
positions, its first-action change saved seed 17 (72588/70000 versus historical
68934); seed 55 remained a loss and winning control 14 remained a win. These
are position continuations, not fresh full-game win-rate evidence. A larger
panel now includes all V68 final-boss losses and early winning controls, with
both baseline and experimental continuations run from the identical position
under the same current policy to avoid attributing unrelated policy changes
to the rollout. The prototype remains isolated under `/tmp`.

The 30-position panel completed: **13 baseline clears versus 15 rollout
clears**, saving seeds 17 and 65 with no lost baseline clears. The sample
deliberately overrepresents historical losses and is not a game win rate.
Artifacts are in `eval/heuristic_blue/diagnostics/blind_rollout_v80_*`.

V80 adds opt-in `heuristic_blind_rollout.py` and benchmark `--blind-rollout`.
It searches the first play/discard of each boss after opening consumable/sale
actions. Three paired public draw/RNG samples compare up to four actions;
perfect sampled baseline clears skip alternative simulations because they can
only tie the objective. Each simulation has a 50-action continuation cap.
The remaining hand policy is V72, shop policy V68, excluding the later Death
and spending experiments. All **28 draw/continuation/rollout checks pass**,
including actual last-hand clear, strict live-state isolation, independence
from future draw order/RNG/absolute card IDs, and once-per-boss search.
Full-game panel: `boss_opening_v80.jsonl`, seeds 0–99, six workers. This
experimental switch is not a promoted training-generator default.

V76 stopped at **69/91** (22 losses), unable to reach the baseline's 82/100
even if all nine remaining games won. V72's second panel and V79's seeds
100–149 comparison continue independently.

Follow-up analysis confirms 32 of the 50 V68 losses across seeds 0–199 occur
in Antes 7–8. V72's saved seed-74 Window position does not become a full-game
win: that run subsequently loses Ante-6 Water at 25462 points. This is why
position rescues are kept separate from full-game outcomes.

Adaptive rollout recalculates after each real play/discard instead of searching
only the boss opening. The current paired 30-position replay panel has six
rescued losses among its first 25 completions (8, 17, 19, 65, 76, 71), with no
lost baseline clears yet; the remaining positions are still running. The
opening-only V80 full-game panel has independently finished seed 17 as a win.

V81 uses `--blind-rollout adaptive` over the same V72 hand/V68 shop combination
as V80. All **29 draw/continuation/rollout checks pass**, including a real play
and draw followed by recalculation on the final hand. Simulated continuation
agents inherit the actual teacher's shop/growth settings. Panel:
`boss_adaptive_v81.jsonl`, seeds 0–99, four workers. Neither this incomplete
comparison nor the selected replay positions establishes a 90% win rate.

V72's second panel completed **39/50**, totaling **82/100**, with the same
win/loss seeds as V68. Its two parent runtime manifests match exactly;
`held_steel_v72_combined100.jsonl` contains each seed 0–99 once. It improves
some individual positions but has not improved this panel's game win rate.

The adaptive replay panel finished **19 clears versus 13 baseline clears**,
saving exactly seeds 8, 17, 19, 65, 76 and 71, with no lost baseline clears.
Artifacts: `eval/heuristic_blue/diagnostics/blind_rollout_v81_*`. This remains
a selected position panel under the V78 current teacher, while the full-game
V80/V81 candidates deliberately use the V72/V68 baseline combination. Seed
53's replay took 506 seconds and still lost (61523 versus historical 60393);
the full-round lookahead's cost is material for future training throughput.

A separate `/tmp/wide_blind_candidates.py` pilot expands the four root choices
to at most ten, adding hand-preserving discards, single-card discards, held
Steel retention and rank/suit group draws. It replays six still-losing positions
(0, 11, 28, 32, 55, 88) and two winning controls (3, 14), four workers. No wider
candidate policy has been installed or treated as a full-game improvement.

The user asked whether random games are also sampled. Current A/B comparisons
reuse seeds 0–199; each seeded game still has randomized shuffles/offers, but
those repeated development seeds are not fresh generalization evidence. A new
100-seed panel was sampled without replacement using `secrets.SystemRandom`
from integers 1,000,000–1,999,999 and recorded before evaluation in
`random_validation_seeds_v1.json`. It is disjoint from development and the
reserved final seeds. These new seeds have not yet been played. The strongest
lookahead candidate will be frozen before testing this independent panel.

The benchmark now accepts `--seed-file`, records the exact seed list in its
profile, rejects duplicates/non-integers and mismatched explicit game counts,
and keeps ordinary contiguous-seed usage unchanged. Syntax/Ruff/help checks
pass; a mismatched-count check verified rejection before any game/output was
created. Actual seed coverage will be checked when the random panel completes.

A V79 seed-125 loss audit ruled out a simple last-hand Stencil sale rescue:
every legal single sale scores less than the existing final play. Selling
Joker earlier briefly improves the current hand, but the overall continuation
and permanent cost have not been established. No sale rule was added from
that unproven hypothesis.

V79 reached **36 wins in 44 completed games**, already exceeding V68's
**33/50** on the same seed range even before its final six games completed.
This supports testing the conservative shop forecast with adaptive blind
search. V82 combines those existing changes over V72's hand policy; all
**98 focused shop/draw/continuation/rollout checks pass**. Panel:
`weak_adaptive_v82.jsonl`, seeds 100–199, four workers. Its source archive,
adaptive flag and exact 100-seed profile were verified before restoring root
files to V78. Workers retain their frozen V79 policy. V80/V81/V79 continue
independently; no random-validation seed has yet been played.

The wider-candidate pilot has additionally rescued seed 32 (76914/70000).
The opening change plays five non-face cards instead of two, improving future
draws while preserving Ride the Bus. It was an alternative play excluded by
the four-candidate cap, rather than a new scoring or shop rule. The other
completed still-losing positions remain losses; the pilot has not yet finished.

V79 completed **42/50 (84%)** on seeds 100–149 versus V68's **33/50**. It
saves 121, 131, 133, 138, 139, 140, 141, 142, 145 and 149, with one regression
at 125. Its Wilson 95% interval is [0.715, 0.917]; this panel does not prove
90% generalization. Every requested seed appears exactly once.

With that completed improvement, the frozen V79 policy has now started the
new random 100-seed panel, two workers: `weak_draws_v79_random100.jsonl`.
Exact seeds, runtime policy hashes and source archive were verified at startup.
This is a randomized generalization check of V79 without lookahead. V82 was
already frozen before this random panel began and its separate development
comparison continues. The reserved final-validation seeds 10000–10999 remain
untouched. This supersedes the earlier note that the new random panel had not
yet been played.

The training generator now accepts experimental `blind_rollout="opening"`
or `"adaptive"` in both recorded and no-observation paths, passing the option
through worker/chunk configuration and recording it on each training example.
The default remains off. All **22 generator/rollout regression checks pass**.
`check_heuristic_replay.py` now compares actual post-search actions and records
runtime provenance. A full V81 adaptive replay comparison on seed 17 is running
between fast evaluation and recorded training generation; its startup policy
hashes match V81. No final training dataset has been promoted or generated.

The wider ten-candidate replay pilot completed **3 clears versus 2 baseline
clears** across its eight selected positions; only seed 32 was rescued.
Seed 55 took 868 seconds and still lost, so the extra computation has a
substantial cost. Artifacts: `diagnostics/blind_rollout_wide_*`. A smaller
follow-up tests whether raising the original cap from four to six, without
the additional discard branches, retains the seed-32 rescue and the two
winning controls. It remains a `/tmp` prototype, one worker.

The six-candidate follow-up completed all three positions: seed 32 still
clears at 76914, and controls 3/14 still clear. Seed 32 took 93 seconds versus
138 for the ten-candidate version. The six-candidate shortlist is now installed
for subsequent experiments; its four existing rollout checks pass. Running
V80/V81/V82 evaluations retain their original four-candidate imports.

V81 fast evaluation and recorded training generation on seed 17 completed
with **262 matching decisions**, both full-game wins, no first difference,
and matching recorded rollout metadata. Artifact: `replay_parity_v81_seed17.json`
with startup runtime manifest. This proves parity for that game/policy, not a
90% policy win rate or full observation-limited teacher suitability.

V81 rescued seed 19's Ante-2 boss but later lost Ante-8 Big Blind, where
boss-only rollout never ran. A four-position ordinary-blind pilot with the
four-candidate adaptive search rescued **three of four** losses: V79's seed
125 (30272/30000), V68's 114 (12647/11000) and 172 (31304/30000). V81's seed
19 still lost. These are position continuations, not fresh-game wins.

The next experimental mode, `--blind-rollout all`, enables adaptive search
on Small/Big blinds too and is wired through generation and the parity checker.
The current-hand score oracle now preserves a baseline play that already
clears, avoiding unnecessary randomized future probes; no simulated outcomes
are fabricated for that shortcut. This uses the existing full-state teacher's
score information. A follow-up is checking the six-candidate/clear-shortcut
combination on the same four ordinary losses before any full-game launch.

V80's opening-only full-game panel stopped after **88 completed games, 72
wins and 16 losses**: it could no longer reach 90/100. This is an incomplete
panel, not a full-panel win-rate estimate. Adaptive V81, combined V82 and the
random-seed V79 test continue. The released workers are being used for the
all-blinds follow-up. Five focused rollout tests pass, including an actual
Small Blind final-hand rescue with the new scope.

The ordinary-blind follow-up retained all three rescues with six candidates
and the current-hand clear shortcut: 125 at 30272, 172 at 31304 and 114 at
14340. Seed 19 still lost. Position artifacts are `diagnostics/ordinary_four*`
and `diagnostics/ordinary_six*`. These checks justify a full-game comparison;
they do not establish additional full-game wins.

V84 combines V79's conservative shop forecast, the six-candidate shortlist,
the guaranteed-current-clear shortcut and adaptive search in every blind.
All **101 focused checks pass**. An archived generator configuration assertion
needed the newly added rollout field; the maintained chunk test now explicitly
checks both default-off and all-blinds modes while preserving unique seeds.
Panel: `all_blinds_v84.jsonl`, seeds 100–199, four workers. The identical seed
range permits comparison with V82, whose boss-only four-candidate search
continues independently. Generation's default remains unchanged.

V81's early regressions include 37 (Ante-2 Window, 1475/1600) and 35 (a
different build after changed Ante-1 decisions, later losing Ante-5 Big).
An independent confirmation pilot compared proposed switches against the
baseline on six additional paired draw/RNG samples. Holding the underlying
six-candidate policy fixed, it finished **4 clears versus 5 unconfirmed clears**
over six positions: it saved 37 but lost the rescues at 8 and 17; 14, 65 and
76 still cleared. Global confirmation was therefore not installed. Artifacts:
`diagnostics/confirmation_*`. An initial diagnostic was stopped and replaced
with this properly matched comparison; only the matched result is reported.

A follow-up tests confirmation on six early-blind positions only: V81's 37,
V80's 67 and historical losses 115, 164, 187 and 19. This will determine
whether it is useful before mature builds, without changing late-game search.
The prototype remains under `/tmp`, two workers; it is not a full-game result.

An additional shop audit found passive/consumable-generating jokers are often
offered but rarely owned under V68: Delayed Gratification offered in 110/200
games, owned in one; Reserved Parking 113/200, owned in one; Hallucination
114/200, owned in two; Vagabond 75/200, owned in one. Current income valuation
omits their benefits. Engine code confirms Delayed Gratification requires zero
discard actions, and Reserved Parking pays probabilistically for held faces.
No new income bonus was added; their conditional value and survival tradeoffs
still need testing. This audit did not change any game rule or unlock flag.

Early-only confirmation finished **4 clears versus 3 unconfirmed clears**
on its six selected early positions. It saved 37 and 67, retained clears at
115/164, lost the rescue at 19, and still lost 187. Artifacts:
`diagnostics/early_confirmation_*`. This is a tradeoff requiring full-game
evaluation, not an established win-rate gain. The confirmation option now
exists as `--confirm-early` with an active rollout mode, and as
`blind_confirm_early=True` in generation. It only verifies proposed changes in
Antes 1–2, records both proposal and confirmation outcomes, and uses six paired
draws with an independent seed. Late-game search is unchanged when the option
is enabled. Ten focused rollout/chunk checks pass, including actual legal
switches, live-state preservation and propagation of the option to workers.

V81 stopped at **59 wins in 70 completions** after its eleventh loss made
90/100 impossible. The all-blinds and conservative-shop candidates continue.

A separate passive-income prototype over V79 now estimates Delayed
Gratification in half of future safe rounds, assigning zero when Summit or
Burglar consumes its discard opportunity. Reserved Parking uses deck face
density, probability, held-card count and Mime retriggers. All seven prototype
checks pass, including safe/unsafe purchases, zero-value conditions and actual
Delayed Gratification payout differences. An initial payout test incorrectly
included ordinary hand/interest income and used Red Deck's discard count;
the corrected test compares identical Blue Deck states with/without the joker.
Parking remains too weak to justify the tested purchase at $20, but is bought
with a $30 cushion; its income was not inflated to force an earlier purchase.
This prototype remains isolated under `/tmp` and is not in any running panel.

V85 adds early-only confirmation to V84's all-blinds policy, keeping the
conservative V79 shop forecast and six candidates. All **104 focused
shop/draw/continuation/rollout/chunk checks pass**. Panel:
`early_confirmed_v85.jsonl`, seeds 100–199, four workers. The common seed range
supports a direct comparison against V84 without confirmation. No final
validation seed has been used, and no evaluated result yet establishes 90%.

V82 stopped at **54 wins in 69 completions**, with 15 losses; even winning
all remaining games would only reach 85/100. This is an incomplete panel.

The original V68 control is now running the same 100 randomly selected seeds
as V79 (`level_scalers_v68_random100.jsonl`). Its startup source archive and
all relevant policy source/binary hashes match the original V68. The seed
list was fixed before testing; no outcomes were used to select it. Both
random panels remain incomplete, and their different completion counts must
not be treated as a matched comparison.

V86 extracts the passive-income estimates into `heuristic_income.py` and
adds them to V79 shop valuation, with V72 hand/draw/growth behavior and no
blind rollout. **81 focused checks passed** (69 existing shop checks plus
seven prototype checks and five maintained income checks). Full-game panel:
`passive_income_v86.jsonl`, seeds 100–149, two workers. Its matched comparator
is V79's 42/50. Startup archive, exact policy binary/source hashes, income
helper hash and seed range were verified before restoring the prior root
experimental files. The helper remains unused by that restored shop policy;
no default policy or training dataset has been promoted.

A pack-selection audit identified a distinct missed upgrade: every one of
17 Black Hole offers in V68's 200-game trace was passed over. The legacy
Spectral pack score is 3, below the claim threshold, while search delegates
mixed Spectral/Planet packs to that policy. This included five losing games
(0, 58, 76, 102, 190), but does not establish that taking Black Hole wins them.

The new isolated `heuristic_pack.select_black_hole_pack` helper compares
actual Black Hole/Planet claim effects through the existing shop forecast,
including Constellation's Planet-only trigger and Red Card's skip reward.
It does not change ordinary packs. Four maintained checks pass: actual
all-hand level gains and live-state preservation, a Constellation case where
Mercury wins, a Red Card case where skipping wins, and a masked-claim case.
`shop_search_v87.py` remains a candidate in the experiment backup directory;
root defaults have not been promoted.

Eight paired full-game continuations are running from recorded Black Hole
pack positions (0, 22, 29, 48, 58, 76, 102, 190), with the V79 shop/V72 hand
policy as control and only the rare-pack helper added to the experiment.
Both branches start from the same recorded state. Artifacts:
`diagnostics/black_hole_positions.jsonl`, `black_hole_replay.py`,
`black_hole_replay.jsonl`, plus verified startup manifest/source archive.
This selected-position diagnostic is not a fresh-game evaluation. First
result: seed 0 loses in both branches at Ante 5 Big, so improved pack value
alone did not rescue that continuation.

The random comparison remains incomplete and mixed: at the 23-seed matched
intersection, V79 has 16 wins and V68 has 18. Continue both full 100-seed
panels; do not substitute the favorable development-panel gain for evidence
of improved random-seed performance.

A final-shop calibration audit found only **3 of V68's 50 losses** followed
a shop forecast marked safe. The partial V79 random panel had 2 such cases
among 15 losses at its 60-game snapshot. Most shortages were already detected;
further blanket pessimism is unlikely to address the dominant failure mode.
Structured snapshot: `diagnostics/final_shop_calibration_snapshot.json`.

A late-voucher audit found Seed Money bought for $10 before Ante 8 Small
on losing seed 55 with only $22. Under the current ordinary interest multiplier,
there are only two payouts that can fund another shop; even saturated extra
interest merely recovers that price. `interest_voucher_can_repay` now checks
optimistic extra capped interest over usable remaining payouts. It accounts
for discounts and current To the Moon multipliers, and explicitly does not
predict buying To the Moon later. Eight income/helper checks pass, including
actual engine payout comparisons for full-price/discounted Seed Money.

V88 remains an isolated V79 shop prototype with this eligibility check. A
paired seed-55 continuation from the recorded late-shop position is running.
Both branches use root V74 hand/growth and V72 draw, with dynamic V79/V88 shop
classes; this distinction is recorded in its manifest. Artifacts:
`diagnostics/late_interest_seed55.*`. No full-game policy has been promoted.

Black Hole paired continuations have shown one additional win so far: seed22
changes from loss to win under the matched continuation policy. Seeds29/48
remain wins; 0/58 remain losses; 76 progresses one more ante but still loses.
These are selected-position continuations, not rates from new complete games.

The seed-55 late-interest paired continuation finished: V79 control lost at
**281292/300000**, while V88 won at **304387/300000**. The eligibility check
preserved $10, enabling additional packs/rerolls before the final boss. This
is a recorded-position rescue under the matched V74-hand continuation setup,
not a fresh seed-55 game result or a measured overall win-rate gain.

V89 is prepared in the experiment backup directory: V79 shop with both the
Black Hole pack helper and the interest-voucher payback check. It excludes the
separate V86 passive-income experiment and blind rollout. Full-game evaluation
has not started yet; wait for replay capacity and verify focused checks first.

The Black Hole replay panel completed all eight distinct requested positions:
**3 changed-policy wins versus 2 control wins**, saving22 with no lost control
wins. Seeds102/190 still lose, despite advancing further or improving final
scores. Summary: `diagnostics/black_hole_replay.summary.json`.

V89 passed **101 focused shop/draw/income/pack checks** and now runs 50 complete
games on seeds100–149 with two workers, no rollout:
`rare_pack_payback_v89.jsonl`. Comparator V79 is42/50 on the same exact seeds.
Startup archive, V72 hand binary/source, V72 draw/growth, V89 shop, both helper
hashes and exact seed range were verified before restoring prior root files.
No selected-position result is being counted as a fresh complete-game win.

V85 early confirmation stopped at **27 wins in 34 completions**, versus28
V84 wins on those same completed seeds: no extra wins and one regression105.
This was an allocation decision based on the matched result, not a claim
that90/100 was mathematically impossible. The panel remains incomplete.

Random-seed regressions all first diverged in shop actions in the six cases
examined. V79 uses its weaker-half forecast both for survival and purchase
utility. V90 keeps that survival forecast but restores the geometric mean
across all four sampled round totals for purchase utility and growth value.
Six controlled forecast checks confirm cautious survival, original valuation,
and unchanged live state across ordinary/Water/Vessel cases with/without
scalers. Reconstructing the first shop divergence reproduces both historical
V68 andV79 actions in all six cases. V90 restores V68's action in three:
1274065,1746427,1998370; this is decision evidence, not win-rate evidence.
Artifacts: `diagnostics/split_forecast*`.

V90 passed **100 focused checks** and now runs seeds100–199, four workers,
no blind rollout or income/rare-pack changes: `split_forecast_v90.jsonl`.
Its complete archive and exact source/binary hashes were verified before
restoring prior root files. V86's first15 completed outcomes matched V79
exactly (10wins each); partial completion order explains the poor raw count,
so it must not be described as a regression without a matched comparison.

The first random100 panel now informs development through failure analysis.
Its results remain valid for the fixed V68/V79 candidates, but cannot serve
as untouched final validation for subsequently tuned candidates. A separate
1000-seed final holdout is fixed in `random_final_holdout_seeds_v1.json`, with
creation metadata and hash. It samples uniformly without replacement from
1000000–1999999, excluding the development random100. It is disjoint from all
recorded benchmark profiles and has not been evaluated. The older reserved
10000–10999 range also remains untouched. Do not use either final set to tune.

V84 stopped at **52 wins in63 completions**, after11 losses made90/100
impossible. Its fully completed, predefined first50 subset is **43/50**,
versus V79's42/50. This gain is too small to establish the target and incurs
considerable rollout cost. `all_blinds_v84.stopped.summary.json` records scope.

V86 passive income stopped at **22 wins in29 completions**, exactly matching
V79 on all29 seeds, with no saved or regressed games. Seven losses cap its
possible full-panel result at43/50. Resources moved to other candidates.

An early-Luchador audit found offers before the final boss in losing seeds53
and71. V91 adds a half-weighted known-boss protection value to a held sellable
Luchador while retaining ordinary-blind survival estimates; V92 tests the
same rule with original mean forecasts. Five targeted checks pass. However,
both prototypes leave the complete recorded shop sequences unchanged in
both seeds. Neither was advanced to a full-game panel on this evidence.
Audit/test sources and results are under `diagnostics/early_counter*` and
`diagnostics/audit*counter*`; candidates remain isolated backups.

V93 keeps the **exact original V68 policy** and adds the current six-candidate
adaptive boss rollout only from Ante5 onward. New `BlindRollout(min_ante=...)`
and benchmark `--rollout-min-ante` preserve all prior defaults. Ten rollout
checks pass under both rootV74 and originalV68 hand policies, including early
state preservation and an actual clear after the threshold. Invalid CLI
combinations are rejected before creating game output. The ante threshold is
currently exposed through the benchmark/helper only; generator/parity plumbing
must be added if this mode is selected for training.

Panel: `late_boss_v93_random100.jsonl`, four workers, same development random100
list as the exact V68 control. Complete source archive, all six baseline policy
source/binary hashes, game data hash and threshold/profile were verified before
restoring prior root files. Final holdouts remain untouched.

V90 split valuation stopped at **32 wins in47 completions**, versus V79's39
on the same47 seeds, with no saves and seven regressions (121,128,130,133,
136,141,145). Its ceiling is85/100. Restoring selected original shop choices
did not establish a stronger complete-game policy.

V93's first22 completed matched games reproduce **2229 pre–Ante5 events**
exactly against the V68 control, including actions, scores, cards and shops.
Snapshot: `diagnostics/late_boss_v93_prefix_parity.json`. This validates the
scope of the late-search intervention, not its win-rate benefit.

A combined shop/pack audit found Trading Card offered in60/200 baseline games
and owned only in winning seed85. The policy lacked a targeted one-card first
discard to earn its payout. DNA was offered in79 games and never owned, but no
DNA policy was implemented; it needs action support as well as valuation.

V94 adds `heuristic_trading.py`: when retained visible cards can finish the
blind, discard one unenhanced, unsealed, unedited expendable card on the first
discard. A cloned engine discard applies Banner and other discard effects;
its refill is suppressed, and the scoring probe uses only retained known
cards, with the draw-pile count adjusted for the real refill. It avoids
permanent Green/Ramen costs and builds sensitive to unknown held cards, keeps
finisher/upgraded cards, and stops thinning below a20-card deck. The shop
estimates Trading income in half of future rounds, only with a compatible
build and an available discard. No random draw identity is used to choose it.

The initial state-preservation test included a baseline candidate search that
populated derived card caches. Moving that test's baseline preparation to a
clone isolated the new helper correctly; the helper preserves the live state.
All **77 focused checks pass**, including actual $3 income, permanent thinning,
a retained real clear, future-order/RNG independence, protected cards, unsafe
purchase rejection, and the installed agent's one-card discard. V94 runs on
exact V68 hand/draw policy plus the new income-discard/growth hook and shop
valuation: `trading_income_v94_random100.jsonl`, four workers, same development
random100 list. Full startup archive, base binary/source, candidate files,
data hash and seed list verified before restoring prior root files.

The first random100 comparison completed and was checked against the exact
100 distinct requested seeds: **V68=80/100, V79=69/100**. V79 saved2 games
and regressed13. Its favorable42/50 development result did not generalize.
`random_v68_v79_comparison.json` records paired outcomes and Wilson intervals.
V68 remains the stronger random-panel base. Voucher-hand-limit synchronization
was inspected and is already correct; no voucher forecast fix was made.

V89 completed **42/50**, with exactly the same win/loss outcomes as V79 on
seeds100–149. Selected-position pack/payback rescues did not improve this
complete-game panel. It has not been promoted.

V95 is a coordinated Vagabond prototype. The shop values generated Tarots,
uses a zero cash reserve while carrying Vagabond, avoids interest-cap vouchers,
and can buy an affordable Tarot/Planet/pack to enter the generation threshold.
`VagabondAgent` defers money Tarots that would disable generation while more
than two hands remain (with a full-money-inventory escape). It uses spare
ordinary-blind hands through Ante6 to produce Tarots only while retaining a
visible finisher with a35% scoring margin and avoiding fragile held-card,
Loyalty, Obelisk and Ice Cream dependencies. No-future-card safety is based on
retaining visible scoring cards; this remains an approximate game policy.

All **85 focused checks pass**, including actual Tarot generation followed by
a real blind clear, legal consumable handling, cash threshold transitions,
mask/state preservation, safe/unsafe purchases and existing shop/rollout
checks. The benchmark has an explicit `--vagabond-policy` opt-in and profile
field. Rollout continuations now instantiate the supplied agent's class so
their hand policy matches the acting policy; ordinary HeuristicAgent behavior
is unchanged. The generator/parity tools do not yet select this actor variant;
that integration is required if V95 is selected for training.

V95 runs100 development random seeds, four workers, no blind rollout:
`vagabond_v95_random100.jsonl`. It uses exact V68 base hand/draw/growth with
V95 shop and the new actor subclass. Complete archive, all base policy
source/binary hashes, candidate helper/shop hashes, data hash, exact seed list
and actor profile flag were verified before restoring prior root files.
Final holdouts remain untouched.

The random development comparisons continue: V93 is **70/86** and V94
**71/87**, with exactly the same winning/losing seeds as V68 on their respective
completed subsets. V95 is **30/37**, with no saved games and one regression
(seed1163710) against V68. These partial results do not establish an improvement.

An exact action replay of the three completed V95 games carrying Vagabond
found an inventory bottleneck. On seed1929467, Vagabond generated4 Tarots
across6 active plays, with2 blocked by full inventory. That game advanced from
the baseline's Ante6 loss to Ante8 Crimson Heart, but still lost90229/100000.
Seed1698312 generated10 across28 active plays;18 were blocked, though the game
still won. Seed1163710 generated3 across22 active plays;19 were blocked, and
the game regressed to an Ante6 loss. Unused suit Tarots occupied both slots for
multiple blinds. `diagnostics/audit_vagabond_generation.py` reproduces every
recorded action, verifies cash, score, portfolio, legality and terminal outcome,
and records actual consumable generation/use in `vagabond_generation_v95.json`.

V96 addresses only this concrete bottleneck. After the baseline declines to
use a consumable and selects a play/discard, an active Vagabond with a full
inventory can consume an unused suit Tarot on a visible card already of that
suit. This frees capacity without changing that card's suit or the deck. It
does not force a different suit or spend a held Tarot when capacity is free.
Existing cash deferral, farming and V95 shop choices are unchanged. Benchmark
traces now also record consumable inventory and total Tarots used.

All **88 focused checks pass**, including unchanged deck/state checks, actual
slot clearing followed by real generation, inactive/full-inventory guards,
shop checks and rollout tests. A fresh complete-game diagnostic on the three
affected seeds is running with3 workers: `vagabond_slots_v96_owner3.jsonl`.
Its full source archive, V68 base source/binary, V95 shop, V96 helper, bundled
data hash, actor flag and exact diagnostic seed list were verified before
restoring prior root policy files. These selected seeds are diagnostic evidence;
they are not a random validation panel. Final holdouts remain untouched and no
candidate has been promoted to training.

V96's three complete diagnostic games finished **1/3**, exactly the same
winning/losing seeds as V95. Actual generation increased from4/3/10 to19/33/24
Tarots on seeds1929467/1163710/1698312 respectively. Full-inventory blocks fell
to2/0/0. Seed1163710 advanced toAnte7 but still lost, so the regression against
V68 remains. Seed1929467 lost94060/100000 at Ante8. The exact replay audit of
all three V96 traces passed, including the new inventory/use-count fields.
This establishes functioning generation, not a complete-game win-rate gain.

That final shop exposed a separate acquisition issue: V68/V95 evaluate adding
Joker Stencil before considering the sales needed to benefit from its empty
slots. V97 adds `heuristic_stencil.acquisition_plan`, evaluating the affordable
final portfolios obtained by selling subsets of non-eternal jokers and buying
the offered Stencil. It uses actual sale effects, the existing public scoring
forecast, survival constraints, income valuation and price/interest penalties.
The first sale competes with ordinary purchases on the same value scale; the
shop re-evaluates after each action. The game/deck/seed rules are unchanged.

All **72 focused checks pass**: the complete legal acquisition sequence reaches
a stronger portfolio, unaffordable/eternal/strong-portfolio cases are rejected,
and existing shop checks pass. V97 runs on exact V68 hand/draw/growth and V68
shop plus this acquisition helper, with no Vagabond actor or blind rollout:
`stencil_plan_v97_random100.jsonl`, four workers, the same100 development random
seeds. Full source archive, source/binary/helper hashes, bundled data hash,
profile and seed list verified before restoring prior root policy files. The
benchmark explicitly preloads the new helper before forking.

Two paired recorded-final-shop continuations completed in
`diagnostics/stencil_final_shop_replay.jsonl`. For the V96 position on
seed1929467, the old shop reproduced its94060 loss; V97 sold Blue Joker, bought
Stencil and won118042/100000. For the V68 position on seed133, both policies
made the same shop decisions and lost79154. These are continuations from
recorded positions, not fresh-game wins. The new random panel already has an
early regression on1277135: it sold Shortcut to afford Stencil inAnte1, then
later lostAnte4 with a Stencil/Rocket/Baseball/Walkie Talkie portfolio. The
last shop forecast8869 against5000, but the actual blind produced2402. No
candidate is promoted on the strength of the recorded-position rescue.

At the latest snapshot, V93=79/96 and V94=79/98, still exactly matching V68 on
completed seeds. V95=40/49, with no saves and regressions1163710 and1348519.
V97=2/5, with no saves and regression1277135. Partial completion order is not a
random representative subset; matched outcomes, not these early raw rates,
guide the comparison. The goal and final holdouts remain unchanged.

V93 and V94 both completed the exact100-seed random development panel at
**80/100**, with precisely the same winning/losing seeds as V68. Seed coverage,
uniqueness and terminal outcome consistency were verified; paired comparison
artifacts are `late_boss_v93_random100.comparison.json` and
`trading_income_v94_random100.comparison.json`. Neither improves this panel.

V95 was stopped with **51/66**, versus V68's54/66 on the matched seeds, no
saves and regressions1163710/1348519/1037309. Its15 losses cap the planned panel
at85 wins. V97 was stopped with **15/26**, versus V68's20/26, no saves and
regressions1277135/1964236/1998370/1521522/1536538. Its11 losses cap the panel
at89 wins. Both process handles confirmed terminal after interruption; final
stopped summaries record exact counts and result hashes. Selected-position
Stencil rescue and improved Vagabond generation do not justify promotion.

A replay audit of V97's seed1277135 loss found every played-hand estimate
matched the actual score exactly (`diagnostics/stencil_scoring_miss.json`).
The shop's12 public hand probes instead included several unusually strong
poker hands. Increasing sample count changed its mean capacity forecast:

- V97 position1277135, Ante4 Small, target5000:12 samples8869;48 samples6741;
  192 samples6057.
- V68 position1830904, Ante5 Boss, target22000:12 samples42408;48 samples24069;
  192 samples18744.

These are diagnostics on selected losses, not proof that reducing a forecast
improves play. `shop_sample_convergence.py/.json` records the comparison and
runtime. The48/192 source variants are archived beside the diagnostic.

V98 tests the original V68 mean-capacity policy using48 public hand samples
instead of12, retaining common samples across candidate portfolios and the
original future-growth, income and action policy. All **69 shop checks pass**.
The exact100-seed random development comparison runs with8 workers:
`public_samples_v98_random100.jsonl`. Complete source archive, V68 source/binary,
48-sample shop hash, bundled data, profile and seed list were verified before
restoring prior root policy files. No Vagabond, Stencil acquisition helper or
blind rollout is enabled in this candidate. Final holdouts remain untouched.

A second diagnostic simulated12 complete next blinds per recorded shop with
the V68 actor, canonicalized public deck contents and independently replaced
future RNG. Both positions and the actor were preloaded before root restoration.
`shop_full_blind_calibration.jsonl` and its summary show **12/12 clears** for
1277135 and **11/12 clears** for1830904. Terminal scores stop at actual clears,
so their averages are not uncapped scoring capacity. This small selected sample
shows that the recorded losses alone do not establish inadequate portfolios;
it also shows why independent-hand capacity and complete-blind survival are
different quantities. V98 still requires its full-game outcome comparison.

An exact replay of all70 V68 losses across the200 consecutive and100 random
development games located16 losses with consumables available on the last
hand (`diagnostics/held_consumables_losses.json`). Fifteen had an unused
consumable at a final PLAY after exhausting discards; the other position used
its Empress before that point. All70 terminal outcomes and scores reproduced.

A standalone `heuristic_consumable_rescue.winning_consumable` prototype searches
legal deterministic consumable targets and then the visible best hand. It
excludes generated/random consumables and does not sample future draws. Two
checks pass, including a constructed suit-Tarot action followed by an actual
final-hand clear, input-state preservation, and rejecting nonwinning/illegal
actions. The first diagnostic run failed on Judgement's empty-list config;
the helper now normalizes empty config values, and the failed manifest/log
are retained under `consumable_rescue_pilot.failed_config.*`.

The corrected15-position pilot preloaded exact V68 policy modules and recorded
a source/binary manifest before root restoration. It completed with **zero
single-consumable rescues**. `diagnostics/consumable_rescue_pilot.jsonl` records
all15 positions and null proposals. The helper remains unused by the acting
policy and has not been added to a full-game variant. This rules out this
specific late rescue opportunity in the audited losses; it does not establish
that earlier consumable choices or multi-action sequences are optimal.

The five-position sample-decision audit reproduced the12- and48-sample
policies' first differing actions exactly on the then-completed saves and
regressions. A192-sample evaluator agreed with48 in three positions, with12 in
one, and chose a third action in one. For seed1962558,192 returned to12's
Splash purchase even though the48-sample run's Saturn purchase was followed by
a full-game win. For saved1044889,192 agreed with48's replacement of ordinary
Joker rather than Even Steven. These results do not turn larger sample counts
into ground-truth action labels. `shop_sample_decision_audit.jsonl` records
all choices, forecasts and runtimes; its prefix inputs and manifest are saved.

V99 introduces a bounded complete-blind comparison for risky shop choices,
using exact V68 as its base. Before an unsafe boss fromAnte3 onward, the
ordinary short evaluator still enumerates and values legal candidates. If its
chosen action is a Joker acquisition or a Stencil/Campfire sale with at least
one distinct alternative, the new helper compares up to three portfolios.
Joker portfolios use the actual planned sale and purchase transitions; a
negative-joker sale that would leave no room is excluded from this shortlist.
Planet choices and ordinary shop fallback are unchanged.

Each candidate runs three paired complete next blinds with the current actor,
canonical public deck contents and independently replaced future blind RNG.
Trials stop at a real clear or loss, avoiding V69's expensive uncapped full-round
capacity objective. The original purchase is retained on equal clear counts;
three original clears skip alternatives. An alternative is chosen only when
it clears more trials. This remains a noisy three-trial estimate, requiring
full-game evaluation. Purchase trial summaries are included in shop trace
forecasts, and the helper is explicitly preloaded before benchmark forking.

All **72 focused checks pass**, including an actual Needle comparison (original
portfolio1/3 clears, stronger portfolio3/3), preserving a perfect original
choice, live-state isolation, future draw-order/RNG independence and the69
existing shop checks. The first test initially demanded zero original clears;
a naturally occurring strong hand validly cleared one trial, so the assertion
now checks the intended strict improvement rather than that unsupported count.

A10-game runtime/quality pilot is running with4 workers on the first10 seeds of
the existing random development panel: `shop_choice_v99_pilot10.jsonl`.
Its full source archive, exact V68 source/binary, V99 shop/helper hashes,
bundled data hash, profile and exact seed subset were verified before restoring
prior root files. This is a pilot, not final validation. V98's100-game comparison
continues independently. No policy has been promoted and holdouts are untouched.

V98 stopped at **36/47**, versus V68's39/47 on matched seeds. It saved
1044889/1016893/1962558 but regressed1277135/1998370/1218995/1746427/1369945/
1348519. Eleven losses cap the planned panel at89 wins. The terminal
interruption and exact final result hash/counts are recorded in its stopped
summary. Increased shop sample count has not improved aggregate outcomes.

V99's10-game pilot completed **7/10**, exactly matching V68, with **every
recorded action identical**. The comparison verifies the exact10-seed set,
outcomes and complete action sequences. Its shortlisted original purchases
always cleared all three trials. A further10-position audit replaced three
trials with12 (`shop_choice_sample_audit.jsonl`, source variantV100); every
original portfolio cleared12/12, so no action changed. V100 is diagnostic only
and was not advanced to a full-game run. Its configurable sample-count helper
retains three as the default and passes the existing actual-clear/isolation
checks. These results motivate a capacity comparison rather than more binary
survival trials at these already-survivable purchase positions.

V101 keeps the same V99 shortlist/gating but tests complete-round scoring
capacity. It reuses the established `_CapacityAgent`: the controller continues
until hands are exhausted under an artificially high completion target, while
the actor's discard/play decisions still use the real blind requirement,
including Wall/Vessel target relief. The trial records whether accumulated
score reached the real target and its final scoring total. These are scoring
counterfactuals, not additional complete-game wins.

The candidate's existing utility is corrected by the log ratio of simulated
capacity to its short boss-capacity forecast, weighted0.25 beforeAnte8 and1.0
atAnte8 to preserve the existing ordinary/future utility and spending penalties.
More sampled clears take priority; for equal clears, a corrected-utility gain
above0.03 can change the purchase. Up to three candidate portfolios and three
paired public samples are evaluated. This differs from V69's costly evaluation
of every shop candidate: only the existing shortlist is simulated. The initial
purchase/sale transitions remain actual engine transitions.

All **76 focused checks pass**, including actual survival preference on Needle,
continuing capacity beyond the first clear while preserving live state, the
existing real-target rollout tests, and69 shop checks. V101's four-worker
runtime/quality pilot uses the same first10 random development seeds:
`shop_capacity_v101_pilot10.jsonl`. Complete source archive, V68 source/binary,
V101 shop and both helper hashes, bundled data and exact seed list were verified
before root restoration. This is the only active benchmark after V98 stopped
and V99 finished. No policy has been promoted; the90% goal and final holdouts
remain unchanged.

### V101 result, Investment tag correctness, and V102

V101 completed its ten-seed pilot at **7/10**, with every recorded action and
outcome identical to V68. It was not promoted.

The installed vanilla Lua sources established that Investment pays $25 per tag
on a successful boss cash-out, and Double copies the next non-Double tag.
Reference hashes and line anchors are in
`eval/heuristic_blue/diagnostics/investment_tag_reference.json`. Corrected
`blind.py` and `runtime.py` implement those effects, including repeated Economy
copies, consumption of paid Investment tags, and interest calculated before the
award. This does not implement every other tag. The focused compiled engine
suite passed **85 tests, with one existing skip**. All **300 V68 baseline action
replays (68,915 decisions, 230 wins)** retained exact cash, scores, joker keys,
legality and terminal outcomes after the fix. None of these baseline games
skipped blinds. The corrected engine remains installed.

V102 uses the exact V68 policy plus an opt-in Investment skip: cash below $25,
Ante below 8, an initial boss capacity gate, and six independently sampled boss
clears are required. Its policy/tag/shop checks passed **78 tests**. The full
100-seed random development benchmark is `investment_v102_random100.jsonl`;
startup source archive, source/binary hashes, data and seeds were verified.
It remains a candidate; neither preliminary results nor recorded-position
rescues establish a new win rate.

### Initial pretraining data readiness

The user suggested that wins across sufficiently diverse games may already
justify pretraining data below the 90% goal. We are separating that readiness
question from further win-rate improvement. V68 remains the strongest validated
base: **150/200 sequential development seeds and 80/100 random development
seeds**, or 230/300 combined. These games are development data, not untouched
validation. The reserved 1,000 random seeds and seeds 10000–10999 remain unused.

An initial final-state audit found **229 different final joker sets among 230
wins**, covering all five Ante 8 bosses (Bell 56, Acorn 56, Heart 46, Leaf 39,
Vessel 33). Pair was the most levelled hand in **188/230** wins, so combinatorial
joker diversity alone should not be presented as broad hand-strategy coverage.

Installed policy source/binary and helpers have now been returned to exact V68
for this data work (search shops, scaler growth enabled explicitly; legacy
remains the constructor default). The canonical tag engine fix stays installed.
The running V102 benchmark already preloaded its candidate implementation.

`scripts/export_heuristic_traces.py` rebuilds existing complete trajectories
through the current single-pass training generator, retaining all wins and
losses. It verifies replay state and terminal parity, legal labels, finite
observations and returns, then saves/reloads shards with the existing tokenizer
and reward metadata and exercises the supervised collator. It also measures
actual played hands, late hands, final joker sets and bosses. Export destination:
`data/pretraining/v68_development300`. These are regenerated training records
from already evaluated trajectories, **not newly sampled games**. Independent
live-policy versus training-generator parity is being checked in
`eval/heuristic_blue/pretraining_v68_parity.json`.

V102 was stopped after **46/57 wins**, versus **47/57** for V68 on the same
completed seeds: zero saves and one regression (1205470). Eleven losses capped
the batch at 89/100. The partial result and stop reason are preserved in
`investment_v102_random100.stopped.summary.json`. This does not affect the
canonical tag correctness fix or the selected V68 teacher.

The complete actual-hand replay audit is
`eval/heuristic_blue/pretraining_v68_diversity.json`, with per-game records in
`pretraining_v68_diversity_games.json`. All 300 trajectories again matched their
terminal outcomes. Among the 230 wins, 78 distinct joker keys appear in final
builds and 11 poker hand types are played. For Antes 6–8, the most frequent hand
per winning game is Pair in **195**, Two Pair in **25**, High Card in **7**,
Straight in **2**, and Flush in **1**. Pair accounts for **3,396/4,595 (73.9%)**
late played hands. On the random subset, **69/80** wins are Pair-dominant late.
These are useful bootstrap demonstrations across varied seeds and builds,
with limited strategic hand-family diversity. They are all Blue Deck/White
Stake under the bundled unlock pool, not multi-deck or multi-stake evidence.

A standalone export smoke check produced 125 records for losing seed 0, then
passed exact replay, observation/action checks, persistence reload, and the
actual supervised collator. The same check also passed in a forked worker.
Full 300-game export and live-policy recording parity remain in progress.

The first export attempt exposed an exporter assertion error: the benchmark's
`final.ante` is captured before the last action, whereas a winning cash-out can
advance `FastRunner.max_ante` to 9. The terminal win labels agreed. The check now
compares the two completed runners' maximum Ante; no engine or training outcome
semantics were changed. The failed attempt/log is retained under
`data/pretraining/v68_development300.failed_ante_assert` and
`/tmp/v68-pretraining-export.log`. The corrected export restarted with 12 workers
(`/tmp/v68-pretraining-export-v2.log`), adding per-game progress messages.
Live-policy versus training-generator seed 1 parity passed all **257 decisions**
and both paths won. Remaining parity seeds are still running.

Continuation checkpoint: corrected export process is tool session **54048**,
log `/tmp/v68-pretraining-export-v2.log`, output
`data/pretraining/v68_development300`. At the latest check, 34 complete game
exports passed per-record validation; shards are emitted after ten games each.
The export is not complete until `summary.json` exists and the process succeeds.
Live V68 parity process is session **34853**: seed 1 passed (257 decisions,
win), seed 32 passed (194 decisions, loss), seed 9 remains pending. V102 session
81648 and first failed export session 65690 are closed. Exact V68 remains
installed with the corrected canonical tag engine. Next: finish both jobs,
check final dataset counts/hashes and all shard reload/collation checks, then
report the completed initial dataset. The 90% goal remains active and unmet.

### Exhaustive final-play audit and V103 ordinary-blind pilot

The previous goal turn made concrete progress: measured complete-game hand and
build diversity, corrected and restarted the export, and obtained replay-parity
evidence. The resumed turn revalidated both live process handles before acting.

`pretraining_v68_parity.json` is now complete: seeds 1 and 9 won, seed 32 lost,
with **689 identical decisions** across evaluation and recorded generation.
All three checks passed. The corrected export has passed persistence reload and
actual supervised collation for its first 12 shards (120 games, 27,886 records).
The remainder is still running; no completed-dataset claim is made yet.

`diagnostics/exhaustive_v68_final_plays.py` exhaustively scored every legal final
play in all **70 V68 losses**. It found **zero immediate rescues** and **zero
chosen-score estimate/engine mismatches**. Three positions had a higher-scoring
legal alternative (17: 14040→14087 needing15106; 158: 2256→2976 needing4966;
1230698: 19536→19684 needing33568), none sufficient. Results, full source/binary
manifest and summary are saved beside the diagnostic. No exhaustive acting
fallback was added without evidence of a rescue.

V93's late-boss rollout had made 61 action changes in 100 games despite retaining
exactly the V68 winning/losing seed set. Some losses moved to later blinds; that
is progress within a run, not additional complete-game wins. Six random V68
losses occurred on ordinary blinds, all from Ante5 onward: 1044889, 1770728,
1016893, 1091225, 1368270, 1251315. V103's focused pilot tests the current
six-candidate adaptive rollout on those recorded death-blind openings, with
exact V68 as the base and `all_blinds=True, min_ante=5`. It samples future draws
and RNG independently, and retains the existing exact-current-clear shortcut.
No source policy modification is required for this configuration.

The pilot is process session **80780**, log `/tmp/v103-ordinary-pilot.log`,
artifacts `diagnostics/v103_ordinary_pilot.*`. Three workers run alongside the
12-worker dataset export (**54048**). These selected position continuations are
not fresh full-game results. A complete random-panel test is conditional on
useful pilot evidence. The 90% goal remains unmet and holdouts remain untouched.

### V104 Economy Tag pilot

An exact replay audit of the 100 random V68 development games found 94 Economy
Tag offers, including **37 with at least $25 held**. Seven such positions were
on six losing seeds. Data: `diagnostics/v68_economy_tag_positions.json`. The base
policy considers this skip only in Ante1 and took none in the 300-game corpus.
This identifies an available decision, not proof that skipping is beneficial.

`heuristic_economy.py` adds an opt-in `EconomyAgent`. Before Ante8, with at least
$25 and a legal Economy skip, it applies the actual skip to a clone (including
Double Tag copies), then requires a boss forecast with a spare-hand margin and
six sampled boss clears. Cash gained is immediate; the foregone round and shop
can still make a skip strategically worse. It inherits all other exact V68
behavior. The default actor and recorded-data teacher are unchanged.

All **9 focused Economy-policy and canonical tag tests pass**: actual copied
cash 25→90, six real sampled boss clears on a constructed strong build, live-state
preservation, weak-build rejection, and cash/tag/Ante boundaries. Benchmark
`--economy-policy` selects this actor; Economy, Investment and Vagabond actor
flags are mutually exclusive. Forecasts and actor choice are recorded, and the
new helper is preloaded before workers fork.

The first 20 fixed random development seeds now run as
`economy_v104_pilot20.jsonl`, two workers, process **49680**, log
`/tmp/economy-v104-pilot20.log`. Exact V68 source/binary, current engine, Economy
helper, data hash, precise seed list and full source archive were verified after
startup. This is a fresh full-game policy comparison on development seeds, not
an untouched holdout. It is not combined with V103 or any blind rollout.

At this checkpoint the data export has completed 21/30 shards: **210 games,
47,810 verified records**. V103's ordinary-blind position pilot has completed
five of six positions with one rescue (1770728). V93 had already moved that seed
past this ordinary blind before it lost to Ante8 Vessel, so that local rescue
alone does not justify a claimed new complete-game improvement. Four other
positions remained losses; the final pilot position is still running.

V103's six-position pilot completed with **2/6 local clears**: 1770728 at
30700/30000 and 1251315 at30530/30000. The other four remained losses despite
higher terminal scores. Both rescued seeds now run from the beginning with
exact V68 plus adaptive all-blind rollout from Ante5:
`late_all_v103_diagnostic2.jsonl`, two workers, session **14613**, log
`/tmp/late-all-v103-diagnostic2.log`. Source archive, exact base binary/helpers,
engine, actor flags and two-seed list were verified. These selected diagnostic
seeds cannot establish a win rate; a broader test depends on full-game benefit.
V103's position pilot session80780 is closed. The Economy pilot remains live.

The bootstrap export has reached **29/30 shards, 290 games, 66,512 records**.
Next completion check must inspect the final process result, `summary.json`,
all 300 unique seeds against source traces/reserved seeds, the expected 230 wins
and70 losses, per-shard byte sizes and SHA256s, and total68,915 decisions.

### Completed bootstrap dataset and commit review

The corrected export process54048 completed successfully: **300 games, 230 wins,
70 losses, 68,915 records, 30 shards, 3,643,055,077 bytes**. A separate final audit
reloaded every shard, verified its SHA256/size, and compared every action and
outcome label with the original traces. All seed counts matched, no seed was
duplicated, and reserved-seed intersections were empty. Local result:
`data/pretraining/v68_development300/verification.json`. A compact committed
report, shard hashes and loading instructions are in `docs/heuristic/pretraining_v68*`.
The local artifacts use the workspace's separate tokenizer-v13 changes; the
heuristic commit does not include the unrelated observation/model/PPO work.
The export script captures whichever schema is active when records are built.

The user requested a commit. Relevant heuristic/engine/generation changes and
their tests are being reviewed in an isolated HEAD checkout plus the proposed
changes; unrelated worktree changes remain outside the commit. The 90% goal is
still active and unmet. Experimental actors remain opt-in.

The isolated commit review found seven failures tied to tests left behind by
unselected variants, rather than missing V68 dependencies. Removed five obsolete
prototype test functions (six cases): automatic valuable-rightmost Death
selection, future-shortage budgeting (with/without Burnt), fallback non-main
planet claims, emergency Celestial-pack spending, and future-shortage Hologram
purchases. Those behaviors are absent from the selected V68 policy and their
experimental results remain documented above. The Stencil prototype test now
executes its standalone acquisition planner directly; it no longer assumes the
rejected V97 planner is wired into V68's shop. The selected acting policy was
not changed to satisfy these experimental expectations.

Commit verification finished with **371 passing checks and one existing skip**
across the isolated source suite and targeted reruns after the obsolete variant
expectations were removed/corrected. The standalone export also passed in that
isolated checkout: seed0 produced125 legal, finite records, then reloaded and
collated successfully with its own tokenizer-v12 metadata. This confirms the
exporter does not depend on the separate, uncommitted tokenizer-v13/PPO refactor.
The local 300-game v13 artifact remains separately verified as documented above.
