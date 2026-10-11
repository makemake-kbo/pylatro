# Consequence targets: source inspection before the predictor

No predictor has been fitted, and no policy integration is authorized by a positive prediction metric alone. The [duration experiment](continuation_duration_results_20260927.md) is complete and failed its advancement screen. The native label audit now passes; the [RNG input audit and target definition](consequence_stochastic_targets_20260927.md) follow before feature benchmarking and fitting. This note records preparation for the [queued scorer proposal](research_translation_20260927.md).

## Reusable features and comparison

The current subset head pools projected contextual card features and combines them with state/count features. The policy already constructs a global masked mean, explicit scalar features and contextual hand slots. Reusing these frozen features for a small consequence MLP is feasible without changing the base policy. A relational comparison must use the same frozen encoder, examples, targets, split and comparable trainable capacity; selected/retained membership and meaningful card/Joker positions must remain available. It should not run the encoder separately for each proposed subset.

Relevant local implementations are [subset_residual.py](../../src/pylatro_agent/subset_residual.py), [action_grammar.py](../../src/pylatro_agent/action_grammar.py) and [agent.py](../../src/pylatro_agent/agent.py). Actual extraction must bind to the verified checkpoint's frozen runtime rather than silently mixing working-tree and archived source.

## Labels must include the full action transition

In the frozen engine, `flow.play_cards` removes selected cards, decrements the hand count and calls `_press_play` **before** calculating the score and held-hand effects. The Hook can randomly discard from the remaining held cards. A direct call to the scorer on the original hand would therefore omit relevant transition effects and can give the wrong consequence target. Reuse the verified full-action branching infrastructure, with unchanged factual controls, cloned state and card-counter handling.

Source inspection also finds random calls for Lucky-card mult/money, Space Joker, 8 Ball, Business Card, Bloodstone, Reserved Parking and Misprint. This is not an exhaustive classification of visible-state predictability: downstream money effects, copied Jokers, other transition code and observation omissions still need checking. Seeing identical outcomes on a few draws cannot certify determinism.

The relevant frozen sources are [flow.py](../../eval/pretraining_transfer_20260923/candidate/src/pylatro/flow.py), [scoring.py](../../eval/pretraining_transfer_20260923/candidate/src/pylatro/scoring.py) and [rng.py](../../eval/pretraining_transfer_20260923/candidate/src/pylatro/rng.py). The engine uses named random channels and tracks continuation state; replacing only a global random seed would not establish independent label draws.

The environment also calls `apply_best_joker_order` before the committed play. It scores candidate Joker orders on `copy_for_scoring` states, which deep-copy RNG state, then selects a score- or money-oriented order. Thus there can be multiple scorer invocations for one action, and the final order can depend on the saved random realization. Label generation must preserve this existing harness behavior and the relationship between its copied probe streams and the committed stream. Independently resampling every probe call would define a different transition. This is a source-level finding, not a measured contribution to the pretraining gap. See [joker_layout.py](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/joker_layout.py) and [heuristic_simulation.py](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/heuristic_simulation.py).

The [label helpers](../../eval/consequence_prediction_20260927/label_tools_v1.py) trace committed and copied scoring states separately, restore hooks on failure, and propose legal alternatives beyond teacher/base using visible observation identity. Eight focused stdlib tests initially passed, including cloned-RNG independence and slot-order preservation; their [test receipt](../../eval/consequence_prediction_20260927/label_tools_tests_terminal_v2.json) covered stand-ins only. The subsequent native equivalence audit below now validates the pilot's simulator behavior.

## Proposed label audit, before freezing a fit

Use training games for a small technical pilot, with all branches grouped by source game. Check actual teacher/base transitions against archived replay first. Then include represented and additional legal alternatives, recording selected/retained slots and actual outcomes. Do not choose the pilot by exposed evaluation failures.

Separate three concepts explicitly:

- A historical replay score is an exact reproduction of that saved simulator realization.
- A deterministic consequence target requires showing that relevant transition randomness cannot change it; it still need not be exactly recoverable from a lossy observation.
- A stochastic predictor should target a defined conditional quantity, such as **E[log(1 + immediate score)]** and **P(immediate blind clear)**. The former is not log(1 + expected score). One realization is a noisy training observation, not an exactly knowable target or an exact expected-action ordering.

Audit random calls through the complete pre-score transition and scoring path before choosing resampling rules. A pilot should verify that any resampled labels leave model-visible input bytes identical, preserve the untouched factual branch and define the random-stream distribution precisely. Report draw counts and uncertainty; do not infer deterministic behavior from empirical equality alone. No RNG state, game seed, future draw order, branch outcome or target-derived feature enters the predictor.

Before scientific fitting, freeze the action proposal rule, game-level splits, target transforms/losses, stochastic sampling budget, matched model sizes and prediction/retention criteria. Benchmark one small batch to estimate feature-cache, label and fitting cost. Keep base policy tensors fixed and audit their identity. Improved immediate predictions alone do not establish good resource preservation, growth or eventual wins.

The primary [DouZero paper](https://proceedings.mlr.press/v139/zha21a.html), [relational RL paper](https://arxiv.org/abs/1806.01830) and [UNREAL paper](https://arxiv.org/abs/1611.05397) were checked during this preparation. They motivate action encoding, entity interactions and auxiliary learning respectively; none establishes that this particular intervention will improve Balatro.

The [native label audit](consequence_label_audit_20260927.md) is complete: all 769 plays / 3,222 decisions in 16 fixed training games reproduce their cached observations and historical outcomes. Across 4,020 proposals, all 8,040 plain/instrumented full-action branches have identical outcomes, full state fingerprints and card counters. The compiled scoring hook observes exactly one committed score per proposal and 1,862 additional copied-state scores. Before-score or scoring random APIs occur on 1,077 proposals; these are overlapping API-use diagnostics, not counts of inherently unpredictable labels. The [independent verifier](../../eval/consequence_prediction_20260927/label_verification_v1.json) passes and [both workers and owner actually exit zero](../../eval/consequence_prediction_20260927/label_owner_terminal_v1.json). Runtime is about 179 seconds. This validates instrumentation on this pilot, not observation sufficiency or a learned benefit.

For initial scientific fitting, use recorded-realization outcomes as noisy targets, with log-score MSE and immediate-clear BCE. The separately declared RNG intervention is an input-separation check; its fresh-channel outcomes will not become training labels or be presented as conditional environment probabilities. That distinction avoids silently changing the learning target while comparing the pooled and relational heads.

The [completed preparation report](consequence_preflight_results_20260927.md) now records the passed RNG-input audit, exact frozen-feature extraction, pooled-head benchmark and matched relational capacity/masking tests. No scientific model has been fitted yet.
