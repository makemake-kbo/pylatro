# Locate remaining phase headroom in the current policy

The recent terminal feature adaptation fits all 622 training comparisons but
fails transfer. Before another hand-scoring fit, measure whether that narrow
action scope addresses the remaining policy gap.

The original epoch-7 intervention found sampled wins of 6.67% for the student,
22.67% with teacher hand decisions, 23.33% with teacher shops and 67.33% with
both. The later selected PPO diagnostic found 14.67%, 49.33%, 44% and 68% on
25 different diagnostic seeds. Both indicate phase complementarity in those
policies. Neither uses the current corrected-head, adapted-encoder checkpoint.
The two terminal-GAE policies' restricted late-hand and early-shop studies
were inconclusive: +4 points [0, 10] and +6 [-8, 20]. These are different
policies, phase boundaries and sampling modes, so they cannot rank current
priorities. Sources: [original diagnosis](learning_bottleneck.md),
[head repair](action_refactor_results.md), [PPO diagnosis](outcome_ppo_results.md),
[restricted phase studies](long_horizon_generalization.md), and
[latest terminal fit](consequence_terminal_feature_adaptation_results_20260929.md).

Use the immutable stage-2 unfrozen-encoder update-4,000 checkpoint, SHA256
`a43e73ab71d3cd6ee78b6b6510d082772b5b362b976e1454a3613eb8f939706f`,
and its frozen candidate runtime. This already has the entity-sale and subset
residual heads; fixing those heads again is not a new intervention.

Select 64 seeds from the exposed development range 850000000–850000127 by
ascending SHA256 of ASCII `current-policy-phase-v1:<seed>`, breaking ties by
seed, then execute selected seeds in numeric order. Selection uses no outcomes.
Run one sampled trajectory per seed and arm, 384 games total:

| Arm | Teacher replacement |
| --- | --- |
| student | None |
| agreed_play | In CHOOSE_ACTION only, when both student and teacher propose PLAY_SUBSET |
| hand | Every CHOOSE_ACTION decision |
| shop | Every SHOP decision; packs remain student-controlled |
| hand_shop | Both CHOOSE_ACTION and SHOP |
| teacher_all | Every phase; no student proposal is needed |

In every hybrid, sample the student before considering replacement. In the
agreed-play arm, query the teacher on every CHOOSE_ACTION decision, then apply
the stated joint condition. This is a privileged diagnostic of subset choice
when both policies already choose to play. It is not an implementable gate,
does not evaluate all teacher-play states, and does not fix play/discard timing.
The hand arm also covers discards, consumables and sales legal in that phase.
Record proposed/executed action families and phase boundaries explicitly.

Use FastRunner with Blue deck, stake 1, Ante 8, 2,000 total steps and errors
raised; record any censoring rather than treating it as a loss. Use CPU FP32,
one inference thread, batch one, temperature 1 and mixture 0.5. Each game/arm
starts its card UID counter at 1. Per-game Torch, NumPy and Python seed equals
the first uint32 from NumPy SeedSequence([game_seed, 0, 1729]). This is a
matched diagnostic sampling convention, not the production global batch-32
trajectory stream. Common initial seeds do not align draws after divergence.

The V68 teacher uses shop search and scaler growth, with no blind rollout.
Query a private state copy, preserve the live runner and RNG, and restore the
global UID counter after speculative work. Assert all observation bytes,
live-state fingerprint and policy RNG unchanged by each query. The teacher
can retain its own deterministic caches within a game. Hidden state, seeds,
teacher proposals and outcomes never enter the learned policy's inputs.

Run four fixed spawned shards under one supervised measurement job; do not
replace failed workers. Then run four fixed audit shards under a separate,
serial supervised job. This is a diagnostic and its replay audit, with zero
training fits. Reset per-game state independently so scheduling cannot affect
outcomes. Save complete traces, full-state/input/RNG hashes, counters, action
families and terminal accounting. The auditor independently expresses arm
selection, forwards every student proposal, recomputes teacher proposals,
replays every action, verifies all trace fields and reconstructs summaries.
Also verify each hybrid equals its student control through the first changed
action. Shared simulator code is not a second simulator implementation.

The historical selected-PPO diagnostic's 325 per-game durations provide a
rough cost bound. Treat agreed-play cost as hand cost, sum the six arm means,
multiply by 64/4, then allow three times that estimate plus 120 seconds per
phase. Freeze the rounded allowance before launch; refuse preparation if it
exceeds 7,200 seconds. This historical estimate is approximate because the
checkpoint, runner and tracing differ. No timing pilot, automatic extension,
replacement games or retries. Preserve failures. Admit scientific results
only after the entire owner actually exits successfully. Check the running
job about ten minutes apart; no heartbeat messages to the user.

Report wins, terminal antes, censoring, replacements and family transitions.
Use paired seed-level 10,000-resample bootstrap intervals, seed 1729, for each
hybrid minus student; hand minus agreed-play; shop minus hand; hand_shop minus
hand and shop; teacher_all minus hand_shop; and the factorial interaction
hand_shop - hand - shop + student. These are exploratory marginal intervals,
not simultaneous inference. Any censoring prevents win-rate prioritization.

A positive hand-minus-agreed-play interval motivates investigating decisions
outside their common played-subset condition; it does not identify which
family caused the difference. A positive shop-minus-student interval motivates
matched-state shop investigation. If both single-phase gains are unresolved
but the combined arm improves, inspect their interaction rather than naming
one bottleneck. Do not rank interventions from unresolved point differences
or automatically select the largest estimate. If no contrast resolves, this
panel does not justify another hand-only fit. All outcomes remain exploratory;
teacher privilege and changed visited states prevent attribution to a missing
observation or a learned architecture. No teacher-assisted result qualifies
the generalization goal or changes the original fresh-policy, early-attribution
and final independent-fit requirements. World models remain deferred.
