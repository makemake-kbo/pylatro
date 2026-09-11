# Model, reward, and training audit — 2026-09-06

This is the historical frozen-run audit. The subsequently approved local
[fixes, SSH benchmark, and migration notes](2026-09-06-audit-fixes.md) are recorded
separately; the captured remote run remains unchanged.

## Bottom line

The evidence does **not** point first to insufficient transformer capacity or
another GAE sweep. There are reproducible observation and objective defects,
an incomplete link between archive exploration and fresh-run competence, and
evaluation metrics that conceal where the learner is failing.

The highest-priority finding is a PT/PPO observation mismatch: on the same
ordinary shop state, pretraining encodes **99.77% clear probability** while
PPO encodes **4.17%**. A model cannot reliably transfer a feature whose meaning
changes between stages. Fix this before interpreting another architecture
or learning-rate experiment.

Keep the typed entity representation, structured legal-action policy, fixed
Ante-8 objective, and archive idea. Repair their interfaces and credit
assignment before scaling the network. The proposed changes below have **not**
been implemented, and the remote training job was not changed by this audit.

## Evidence and scope

[Local snapshot and reproduction instructions](../../runs/audits/2026-09-06-pt-ppo.TMMeAw/README.md)
include 121 downloaded files / 27,410,695 bytes: run logs, raw TensorBoard
events, 75,212 risk forecasts, manifests, monitor reviews, and frozen Python
source. All 106 Python files under `code/src`, plus `train.py`, matched the working tree.
The audit therefore examines the actual run's Python source, not an assumed
newer implementation. Compiled binaries, dependencies, and model checkpoints
were not downloaded; this is not a checkpoint backup.

The live-file capture covers PT epochs 1–5, PPO updates 1–180 / 737,280 steps,
and 17 completed evaluations through update 170. Evaluation 180 was still
running. Different files can have slightly different capture times.

The recipe is a newly pretrained 23.33M-parameter actor, not a continuation
of the older model that reportedly won Ante 5 semi-consistently. PT uses
1,000 games / 99,374 records / five epochs. PPO resets the critic and Adam,
uses LR `3e-6`, 16 environments, 256-step rollouts, four epochs, logical
batch 320 / microbatch 160, and 50% archive-return probability per reset.

### What the logs actually establish

The windowed numbers below average logged per-update metrics, not independent
episode-level estimates. Training metrics mix fresh and archive origins unless
explicitly labeled otherwise.

| Measure | Updates 1–20 | Updates 161–180 |
| --- | ---: | ---: |
| Recent-100 archive-continuation win rate | 0.45% | 9.45% |
| Recent-100 sampled fresh-run win rate | 0% | 0.10% |
| Conditional Ante-6 survival, mixed origins | 4.68% | 25.45% |
| Return explained variance, mixed origins | -0.532 | 0.715 |
| Approximate PPO KL | 0.000534 | 0.000814 |
| PPO clipping fraction | 2.38% | 3.61% |
| Mean cash at a loss, mixed origins | $15.68 | $33.02 |
| Losses retaining at least $10, mixed origins | 58.54% | 78.59% |
| Last-play value / best legal candidate proxy | 0.862 | 0.917 |

At capture, 8,185 fresh episodes and 8,235 archive continuations had completed.
Terminal-reward totals reconstruct 432 wins across both origins. Occasional
fresh sampled wins are real: the fresh recent-100 rate reached 2% briefly.
The logs do not persist enough per-origin episode outcomes to recover an
exact cumulative fresh win count from these overlapping windows.

All 17 greedy evaluation checkpoints scored 0/100 on the **same** seed set.
That is not 1,700 independent fresh seeds, and it does not establish a zero
underlying win probability. Conversely, the improved archive metrics are not
evidence of a good fresh-run policy. No logged scalar in this snapshot is
nonfinite; all PPO minibatches ran and there were no KL rollbacks or stalls.

Source data: [derived summary](../../runs/audits/2026-09-06-pt-ppo.TMMeAw/summary.json),
[all scalar series](../../runs/audits/2026-09-06-pt-ppo.TMMeAw/scalars.json).

## Current architecture

| Component | Actual design | Parameters |
| --- | --- | ---: |
| Observation/embedding | 160 typed tokens, 47 integer fields/token, 25 scalars, three rounds of compressed play history | 1,472,640 |
| Shared backbone | 12 pre-LN self-attention blocks, width 384, eight heads, FF width 1,536 | 21,294,336 |
| Structured actor | Mean-pooled global context, 128-wide state bottleneck, macro choices and entity/candidate scoring heads | 267,858 |
| Critic | Mean pool; eight conditional survival outputs; terminal utility plus scalar residual | 299,145 |

The trunk contains about 91% of parameters. Hand/discard actions mix candidate
probabilities (90% in PPO) with a full-support ordered-card component (10%).
PT instead uses a 50/50 mixture. The so-called AR component reuses a fixed
per-card logit vector; the chosen prefix changes legality masks, not the
neural representation used to score the next card.

BF16 is used for transformer computation. Parameters, gradients, policy/value
heads, losses, and Adam moments remain FP32. This audit did not rerun GPU
benchmarks or change precision. Prior native-driver checks identified the
machine as two RTX 3090s, with only GPU 0 selected, despite fabricated 4090
reporting. Consequently none of the available GPU measurements should be
labeled RTX 4090 results.

Source: [agent](../../src/pylatro_agent/agent.py),
[embeddings](../../src/pylatro_agent/embeddings.py),
[backbone](../../src/pylatro_agent/backbone.py),
[grammar head](../../src/pylatro_agent/action_grammar.py),
[value head](../../src/pylatro_agent/value_head.py).

## Prioritized findings

### 1. Critical: pretraining and PPO disagree on the meaning of risk inputs

**Confirmed defect, reproduced on an ordinary trajectory.**

`fast_generate._build_obs` calls the tokenizer without precomputed risk.
The tokenizer falls back to `capture_state_risk(state, round_score)`, which
omits `sub_phase` and `in_shop`. `_risk_info` therefore subtracts the previous
blind's retained score even in the shop. PPO's `BalatroEnv._capture_state_info`
supplies phase context and correctly clears that stale progress before
calculating risk.

Normal teacher trajectory, seed 42, action step 6, shop, preceding score 368:

| Identical simulator state | PT fallback | PPO environment |
| --- | ---: | ---: |
| Encoded clear probability | 0.997706 | 0.041682 |
| Encoded immediate-death probability | 0.002294 | 0.958318 |

These features feed both the transformer and the direct danger-policy
adapter. Dependent strategy scalars can change too. This is a semantic
distribution shift, not an expected difference between teacher and student
actions. It is a plausible contributor to poor transfer, but its effect on
win rate still needs a controlled rerun.

**Recommended fix:** one phase-aware observation/context builder shared by
teacher generation, Gym rollout, evaluation, and live inference. Make the
tokenizer fallback require the same phase semantics. Version the observation
change and regenerate or explicitly migrate affected training data; do not
silently reinterpret an existing checkpoint or cached teacher dataset.

**Acceptance test:** replay the same action sequence through FastRunner and
BalatroEnv and compare every observation field at hands, shops, packs, and
blind selection, including risk and strategy inputs.

Locations: [fast_generate.py:123](../../src/pylatro_agent/training/fast_generate.py),
[tokenizer.py:267](../../src/pylatro_agent/tokenizer.py),
[risk.py:126 and 268](../../src/pylatro_agent/risk.py),
[env.py:836](../../src/pylatro_agent/env.py).

### 2. High: the policy cannot observe some decision-critical state

**Confirmed observation aliasing and dropped embedding fields.**

A shop with $100 and reroll cost $5 produces exactly the same complete
observation as that shop with reroll cost $17, provided both rerolls remain
legal. The cost is in diagnostic `info`, but not the policy observation.
The action mask only answers whether rerolling is affordable, not whether
it is worthwhile. More transformer layers cannot recover this distinction.

The tokenizer writes card debuff, face-down, permanent-bonus, and forced-card
flags in columns 6, 7, 8, and 10. `DeckCardEmbedding.forward` reads columns
0–5 and 11 only. Toggling each omitted field gives exactly zero embedding
change. Face-down cards also lose rank/suit, and legality/candidate/risk
features can indirectly expose some effects; this is not a claim that the
whole model is completely blind to every debuff. The direct card-state path
is nevertheless missing, especially for permanent growth and targeted
consumables. Permanent bonus is additionally quantized/capped by the tokenizer.

**Recommended fix:** expose reroll cost/free rerolls, meaningful inventory
capacity, and pack-choice context explicitly; audit other action-relevant
state such as The Fool's prior target. Add separate card-state projections,
using bounded/log-scaled continuous features where appropriate. Test fields
individually, not just tensor shapes.

Also audit the fixed caps: only 62 deck cards and four vouchers are encoded
directly. Expanded decks/voucher-heavy runs lose identities. For eventual
live play, inspect hidden-information policy: the tokenizer enumerates
`draw_pile` in internal order and positional embeddings expose that order.
That is a potential simulator-only information channel, not an established
cause of this training failure.

Locations: [tokenizer.py:253–330](../../src/pylatro_agent/tokenizer.py),
[embeddings.py:156](../../src/pylatro_agent/embeddings.py),
[constants.py](../../src/pylatro_agent/constants.py).

### 3. High: archive exploration does not yet teach reliable return paths

**Confirmed design limitation; causal importance is a strong hypothesis.**

The archive stores simulator snapshots, not action trajectories or their
ancestry. A restored Ante-6 win trains the new Ante-6+ suffix. It does not
directly assign credit to the Ante-1–5 purchases/discards that created that
build. PPO's exclusion of stale archived actions is correct; treating them
as newly on-policy data would introduce a different error.

The archive is bounded and stratified, which is good. However, it starts at
Ante 4, is private to each worker, and buckets only by Ante/phase. Repeated
seed/boundary identities replace the previous state with the latest variant,
not necessarily a stronger or more diverse build. Saved future RNG is reused.
At update 180 it contains 1,208 states, including 184 Ante-8 entries. Gains
can therefore reflect increasing familiarity with reachable late-game
suffixes rather than stronger fresh-run construction.

**Recommended structural change:** retain snapshot lineage and successful
prefix/suffix trajectories; add a separate return-path learning stage or
backward-start curriculum, moving starts earlier as later continuations become
competent. Preserve a measured fresh-transition quota, not merely a 50% reset
coin. Track original seed/build family, source policy age, start Ante, and
success by origin. Keep diverse reachable builds, not just winner-only states.

Demonstration/SIL-style actor losses must remain explicitly separate from
the on-policy PPO surrogate, with their own weighting and movement checks.
Simply enabling the existing SIL option would replay suffixes; it would not
invent the missing prefixes. Original Go-Explore also distinguishes archive
exploration from learning a policy that can execute successful trajectories;
that distinction motivates this recommendation, not a claim that our current
implementation is equivalent to the paper. [Go-Explore paper](https://adrien.ecoffet.com/files/go-explore-nature.pdf).

Locations: [archive.py:69](../../src/pylatro_agent/archive.py),
[env.py:250 and 304](../../src/pylatro_agent/env.py),
[ppo_rollout.py:134](../../src/pylatro_agent/training/ppo_rollout.py).

### 4. High: the entropy objective is not the entropy of the policy

**Confirmed mathematical defect; it affects the optimized loss.**

The AR entropy approximation is `H(count) + E[count] * H(card logits over
the union of valid cards)`. Actual sampling is ordered and without
replacement, with a different legal-prefix mask at each step. Those are
different distributions. Consequently the candidate/AR calculation is not
generally the claimed lower bound on policy entropy.

A counterexample using the actual PPO mixture `eps=0.1` has just two legal
actions, each with probability 0.5. Exact entropy is 0.693147; reported
entropy is 0.831777, giving normalized entropy **1.2**. Changing card logits
can induce an entropy gradient of 0.020999 while leaving both actual action
probabilities fixed at 0.5. This is not just misleading TensorBoard labeling:
PPO subtracts this normalized quantity in the actor objective, and PT uses
the raw quantity too.

The logged rollout-mean entropy never exceeded 1 in this run; averages do
not reveal the per-state counterexample. The magnitude of its effect on
learning is unmeasured.

**Recommended fix:** use a verified entropy calculation or an explicitly
defined conditional-entropy regularizer. If using Monte Carlo entropy,
implement and test its gradient estimator correctly, not just the scalar
`-log_prob(sample)`. Enumerate small supports in tests; assert exactness,
bounds, and zero gradients for deterministic conditional decisions.

Locations: [action_grammar.py:896–921](../../src/pylatro_agent/action_grammar.py),
[ppo_optimization.py:550 and 1121](../../src/pylatro_agent/training/ppo_optimization.py).

### 5. High: the replay holdout is not an independent critic holdout

**Confirmed measurement defect.**

Replay insertion marks episodes as held out, and the replay/SIL samplers
respect that flag. But rollout collection labels every completed non-stalled
episode in the PPO buffer regardless of the flag. PPO then trains the
terminal outcome head and shared trunk on those same labels. Thus comments
claiming no optimizer has seen the holdout episodes are false.

An actual `collect_rollout` counterexample forces two local test episodes
into the replay holdout: both still appear as terminal-labeled PPO rows.
Even removing their outcome NLL alone would not make them completely unseen
if their actor/return gradients still update the shared representation.

**Recommended fix:** rename the current metric as a replay-only holdout and
create genuinely separate, never-trained evaluation episodes/seeds for
critic calibration. Persist origin, Ante, phase, predicted outcomes, and
observed outcomes. Any true train/validation split must cover all training
paths and avoid splitting repeated archive descendants across partitions.

Locations: [sil.py:230](../../src/pylatro_agent/training/sil.py),
[ppo_rollout.py:549–567](../../src/pylatro_agent/training/ppo_rollout.py),
[ppo_optimization.py:520–547 and 793](../../src/pylatro_agent/training/ppo_optimization.py),
[ppo_config.py:151](../../src/pylatro_agent/training/ppo_config.py).

### 6. High: expensive evaluation discards the most useful failure data

**Confirmed measurement/design limitation.**

`run_seed_evaluation` produces per-seed final Ante, score, and stall status.
`evaluate_model` reduces it to one win fraction and discards the rest.
There is no PT-to-PPO baseline evaluation at update zero. PT reports only
training-set action accuracy; 64.45% is not held-out competence, and many
records are easy or nearly forced actions. Rare skills are poorly covered:
the 99,374 PT records contain only 406 hand-targeted consumable labels and
10 joker-targeted consumable labels.

The same 100 greedy seeds are evaluated repeatedly while rollout is sampled.
Both are meaningful policies, but they answer different questions. At zero
wins the best-checkpoint rule retains update 10 and the regression-only
monitor cannot flag a long plateau. Neither proves the model should be
stopped, but neither provides a learning-quality approval either.

**Recommended evaluation contract:**

- Baseline heuristic, PT actor before transfer, and transferred actor at
  update zero, including the 0.5→0.1 hand-mixture change.
- Persist per-seed outcomes, reached/cleared Antes, cause of death, cash,
  joker roster/build summary, action-family frequencies, and stalled status.
- Report fresh greedy and fresh sampled results separately; archive
  evaluations get a third, explicitly conditional label.
- Fixed paired development seeds plus disjoint final test seeds; uncertainty
  intervals and a larger win-rate panel only when needed.
- Seed-level PT validation and per-phase/Ante imitation loss, decision regret,
  and legal-alternative accuracy. Add teacher labels on states the student
  actually visits once observation semantics are fixed. This is the
  distribution-shift problem addressed by [DAgger](https://proceedings.mlr.press/v15/ross11a.html).

Use an explicit experiment review budget and fresh-progression criteria;
do not stop sparse-reward runs solely because a small panel has zero wins.

Locations: [ppo_evaluation.py:48 and 168](../../src/pylatro_agent/training/ppo_evaluation.py),
[ppo.py:758 and 803](../../src/pylatro_agent/training/ppo.py),
[supervised.py:258 and 367](../../src/pylatro_agent/training/supervised.py),
[run_monitor.py:86](../../src/pylatro_agent/training/run_monitor.py).

### 7. Medium/high: reward scale is sane, but early investment remains weakly supervised

**Design tradeoff, not a proven reward-accounting bug.**

The active objective pays 10 for an Ante-8 win and at most one total unit
across bosses 1–7 before discounting/decay. Death and stalls pay zero;
old cash/tempo/build/consumable bonuses are disabled. Snapshot bookkeeping
preserves the set of already-paid milestones, preventing past credit from
being reissued. This is substantially easier to reason about than the old
mixture of strategic bonuses.

Nevertheless, milestones are a temporary different objective, not
policy-invariant potential shaping. A rare +10 does not automatically
dominate frequent small rewards in expectation, and decay reaches zero by
update 800 regardless of whether fresh wins have become learnable. The
multiplier at update 180 is already 0.77625. Scheduling by compute rather
than demonstrated competence can remove the bridge too early.

At gamma 0.997 a terminal reward retains 74% of its weight over 100 actions
and 41% over 300. The direct GAE residual trace at lambda 0.97 retains only
3.52% over 100 actions and 0.124% over 200. This is **not** the complete
effective learning horizon: a good bootstrap critic can propagate value
beyond that trace. It explains why improving critic/trajectory coverage
matters more than repeatedly sweeping lambda alone.

The cash and final-play diagnostics suggest investigating underinvestment
and build selection before assuming hand scoring is the sole problem. Those
statistics mix origins and use a heuristic score proxy, so they do not prove
that spending more money indiscriminately would help.

**Recommended experiments:** preserve a fixed-reward control; add earlier
archive practice and complete-trajectory supervision; gate any new milestone
decay rule on independent fresh progression. Consider next-blind/next-Ante
auxiliary targets and short-horizon shop counterfactuals. If returning to
potential shaping, use a documented potential difference with correct
terminal treatment; bounded bonuses alone do not guarantee invariance.
[Reward-shaping theory](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

Locations: [reward.py:738](../../src/pylatro_agent/reward.py),
[rollout_buffer.py:163](../../src/pylatro_agent/training/rollout_buffer.py),
[ppo_config.py](../../src/pylatro_agent/training/ppo_config.py).

### 8. Medium/high: the critic has competing supervision paths and a restricted repair path

**Confirmed design constraints; benefit of alternatives requires ablation.**

The return value is `terminal_value + return_residual`. Return regression
detaches terminal value, so biased hazard predictions can be compensated
by the residual without improving win calibration. A good return explained
variance therefore does not establish a good win predictor. The observed
rise to 0.715 is real evidence of fitting mixed rollout returns, not a
fresh-run success forecast.

Outcome NLL in PPO only sees episodes completed within the collection
window. About 84.5% of rows are labeled in the latest 20 updates; that missing
15.5% is duration-dependent. Complete-episode replay recovers prefixes across
rollouts but updates only `outcome_proj` and `ante_survival`, not the shared
trunk or return residual. This cannot fully repair a representation learned
from completion-biased labels. Replay has only 256 recent episodes and mixes
archive/fresh trajectories. Calling all replay supervision unbiased also
ignores distribution drift, stale policies, and archive-selection effects.

Both a transferred actor and a freshly random critic use the same LR `3e-6`.
The actor KL is small and all minibatches run: chronic KL stopping is not
the current bottleneck. That does not justify blindly increasing actor LR.

**Recommended ablations:** critic-only warmup on current-policy complete
episodes; separate actor/critic parameter groups and rates; a private critic
adapter or shallow critic encoder; next-blind and next-Ante targets; and
completion/censoring-aware outcome supervision. Measure actor/critic gradient
norms and interference, fresh-versus-archive value errors, and calibrated
outcome quality on the independent evaluation panel from finding 5.

Locations: [value_head.py:34 and 133](../../src/pylatro_agent/value_head.py),
[ppo_optimization.py:689 and 776](../../src/pylatro_agent/training/ppo_optimization.py),
[ppo.py:234 and 443](../../src/pylatro_agent/training/ppo.py).

### 9. Medium/high: analytic danger is poorly calibrated in this regime

**Confirmed disagreement with recorded outcome labels; not a calibrated causal oracle.**

Across 75,212 resolved shop-leave forecasts:

| Shop Ante | Forecast rows | Mean predicted death | Recorded next-blind death | Brier loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 14,778 | 54.64% | 8.40% | 0.4160 |
| 2 | 18,687 | 40.31% | 5.65% | 0.3016 |
| 5 | 6,546 | 56.12% | 41.90% | 0.3200 |
| 6 | 3,733 | 47.70% | 47.52% | 0.2674 |
| 8 | 1,862 | 24.42% | 35.88% | 0.2342 |

The mapping is badly pessimistic early and optimistic at Ante 8. Close
agreement of averages at Ante 6 does not imply good individual predictions.
The current fixed Platt constants were fitted on a different policy regime.

Caveats: forecasts are correlated within runs and archive lineages, only
resolved non-stalled episodes appear, and labels infer progression from
terminal Ante/blind; skipping a blind is effectively counted as progression.
These are not 75,212 independent actual-play trials. Fix label semantics and
deduplicate/group splits before fitting anything new. This risk feature is
also distinct from the neural outcome critic.

**Recommended fix:** record origin/seed/lineage, exact forecast target and
actual played/skipped resolution, plus risk-model confidence. Refit on a
separate calibration partition with Ante/phase diagnostics. Initially avoid
increasing hard-coded danger priors; they would amplify an unreliable input.
The current launcher already leaves the fixed shop-logit penalty at zero.

Locations: [risk.py:60](../../src/pylatro_agent/risk.py),
[ppo_metrics.py:35](../../src/pylatro_agent/training/ppo_metrics.py),
[ppo_rollout.py:487](../../src/pylatro_agent/training/ppo_rollout.py).

### 10. High for cost: evaluation is consuming roughly half the PPO wall time

**Measured timing plus source-level bottleneck candidates.**

The 17 completed evaluation intervals total about 355 minutes: 48.8% of the
measured interval between PPO updates 1 and 180. Recent passes take 21–23
minutes, versus about 2.2 minutes per ordinary update. The timing interval
can include small surrounding diagnostics/checkpoint overhead; it is not a
kernel-level attribution.

Three specific paths deserve profiling before increasing batch/model size:

1. Evaluation batches network forwards but steps all environments serially
   in the trainer process. Training workers wait idle.
2. Exact `mode()` loops over batch rows, enumerates legal subsets in chunks,
   and branches on CUDA values. Batching the backbone does not vectorize
   this decoder. It computes AR scores even in candidate-only cases.
3. Evaluation constructs the default **shaped-reward** environment rather
   than the current milestone configuration. It consequently runs additional
   build/reward diagnostics even though evaluation throws rewards away.
   Passing the current config alone will not eliminate all expensive
   diagnostics: `_capture_state_info` still recomputes build/risk features.

**Recommended order:** phase timing and decoder profiling on the SSH GPU;
preserve exact greedy semantics while vectorizing/pruning mode search;
separate observation-essential work from optional diagnostics; pass explicit
evaluation configuration; then parallelize simulator stepping or run a
frozen-checkpoint evaluator on the second GPU with an explicit CPU budget.
Do not silently replace exact mode with macro-first greedy and compare it as
the same policy.

For optimization, `get_batches` materializes all shuffled minibatches on the
device every epoch. Stream/prefetch batches or keep one compact rollout
representation. Head-only outcome replay still constructs the full model
forward graph; a no-grad shared-feature pass plus trainable outcome tower
can avoid unnecessary graph storage. Add critic-only/policy-only forward
entry points rather than computing unused heads everywhere.

The previous live native reading was about 19/24 GiB on GPU 0, not an empty
card. Driver-accounted usage includes allocator caches and is not the same
as live tensor memory. Filling remaining VRAM is not itself a speed target.
Microbatch 320 at unchanged logical batch 320 is a reasonable **benchmark**,
not a promised speedup or a change made to this run.

Locations: [ppo_evaluation.py:136–166](../../src/pylatro_agent/training/ppo_evaluation.py),
[action_grammar.py:661](../../src/pylatro_agent/action_grammar.py),
[env.py:502 and 836](../../src/pylatro_agent/env.py),
[rollout_buffer.py:244](../../src/pylatro_agent/training/rollout_buffer.py),
[ppo_optimization.py:833](../../src/pylatro_agent/training/ppo_optimization.py).

## Architecture direction after correctness fixes

The first candidate is a **more task-structured model**, not a larger generic
encoder. These are proposals requiring matched experiments:

- Replace uniform whole-sequence pooling with learned decision queries or
  per-entity-type pooling. Deck tokens numerically dominate the mean, while
  a few shop/joker features may determine the important decision.
- Add a phase-specific shop/build decision module that scores offer/replacement
  choices against economy, reroll cost, current build, and future blind scale.
  Prefer explicit state/action features over relying on a single global
  128-wide bottleneck to discover every interaction.
- Keep candidate pointers but diversify proposals beyond immediate structural
  score. For the full-support head, condition neural logits on the selected
  prefix/count, not just a legality mask. First fix its entropy accounting.
- Add a critic-specific adapter and local milestone/next-blind predictions.
  Validate separate learning rates before duplicating the entire encoder.
- Compare a 4–6-layer trunk with the current 12-layer trunk under equal wall
  time and equal environment steps. A smaller model might allow more useful
  experience; current evidence does not establish either a capacity shortage
  or that a smaller model will retain competence.
- Add longer strategic memory only after the observable-state audit. The
  current three-round play history lacks a full shop-decision trajectory;
  recurrent memory may help remaining partial observability, but should not
  substitute for exposing directly available state.

For sparse wins, archive ancestry plus return-path learning and better shop
supervision is my preferred structural experiment. An entirely new algorithm
or world model would make it harder to isolate the confirmed defects first.

## Smaller correctness and operational issues

- **PT warmup runs after `optimizer.step()`.** The first update uses the full
  `3e-4` LR; TensorBoard then logs `3e-7`, which is the next update's LR.
  Cosine `T_max` includes warmup even though the scheduler only steps after
  warmup. The last logged LR is still approximately `7.14e-5`. Fix/test the
  actual per-step sequence; this is not established as the main failure.
  [supervised.py:354](../../src/pylatro_agent/training/supervised.py).
- **Evaluation omits `stake=config.stake` at the PPO call site.** The evaluator
  defaults to stake 1. This does not affect the current stake-1 run but would
  silently invalidate higher-stake comparisons.
  [ppo.py:774](../../src/pylatro_agent/training/ppo.py).
- **Watchdog timeouts are phase-blind.** Its no-progress threshold is 30
  minutes while an observed eval took over 26 minutes. A stop request gets
  only ten minutes of grace, shorter than a typical evaluation, and is
  normally checked before the next evaluation/update. Add per-game/phase
  heartbeats and cooperative evaluation cancellation; a slow evaluation
  should not be mistaken for a dead learner.
  [run_monitor.py](../../src/pylatro_agent/training/run_monitor.py).
- **Fresh/archive denominators are incomplete.** Per-origin rolling win
  rates exist, but there are no persisted per-origin cumulative wins,
  transition counts, or full fresh-only survival curves. Add them before
  interpreting reset probability as a balanced learning dataset.

## What appears sound and should be preserved

- Explicit fresh seeded evaluation excludes archive resets.
- Archive restore preserves game state/RNG and milestone bookkeeping.
- Newly collected archive suffixes use the current policy and correct PPO
  old/new probability comparisons; old prefixes are not passed off as
  on-policy samples. PPO's clipped surrogate is designed around that
  sampling/update distinction. [PPO paper](https://arxiv.org/abs/1707.06347).
- GAE separates termination from truncation and prevents cross-reset leakage.
- Logical minibatch weighting and whole-rollout advantage normalization are
  deliberate; there is no evidence here of a minibatch-denominator regression.
- Resetting a critic whose reward targets changed is reasonable; the missing
  piece is validating/warming the replacement, not blindly retaining old values.
- FP32 weights/moments with BF16 transformer compute remains a reasonable
  baseline. Nothing in this audit implicates precision as the main failure.

## Proposed experiment sequence

| Order | Change | Evidence required before proceeding |
| --- | --- | --- |
| 1 | Observation parity, missing fields, entropy tests, genuine holdout, per-seed eval persistence | Same-state observations agree; probability/entropy counterexamples resolved; no validation leakage |
| 2 | Re-establish PT and transfer baselines | Held-out per-phase competence and fresh progression, measured before any PPO update |
| 3 | Same actor: fresh-only vs current archive vs archive with earlier starts/return-path learning | Improvement in fresh reach/clear Antes 5–8 at equal steps and equal wall time, not only archive wins |
| 4 | Critic warmup/adapters/rates and shop-focused supervision | Better independent fresh critic errors and build decisions without actor regression |
| 5 | Profile evaluator/decoder/rollout, then smaller trunk or larger microbatch | End-to-end samples/evaluations per second on the SSH hardware with unchanged policy semantics |

Use a short pilot review window with explicit compute limits before committing
to another multi-day leg. The current run is useful diagnostic evidence and
has learned some continuation competence, but I would not spend its entire
remaining budget unchanged without first addressing the PT/PPO feature mismatch
and obtaining fresh-run progression diagnostics. No stop/restart or fix was
performed as part of this audit.

## Reproduction and limitations

[CPU probes](../../runs/audits/2026-09-06-pt-ppo.TMMeAw/audit_probes.py) and
[their results](../../runs/audits/2026-09-06-pt-ppo.TMMeAw/probe_results.json)
reproduce the risk mismatch on a normal teacher trajectory, reroll-cost
aliasing, dropped card fields, entropy overcount/spurious gradients, and
PPO exposure of replay-held-out episodes. They confirm defects in the
current implementation; they are not tests of proposed fixes or GPU speed.

All **93 existing tests** in the action grammar, archive return, terminal replay,
rollout buffer, run monitor, and supervised-training suites passed locally.
The supervised multiprocess fixture emitted two Python `fork()` deprecation
warnings. Passing these suites does not invalidate the new counterexamples;
they expose missing behavioral coverage. Both audit scripts pass Ruff, and
`git diff --check` is clean. No production source or existing test was edited
as part of this audit.

No new learning ablation, independent full-game checkpoint evaluation, or
GPU benchmark was launched. Thus the report distinguishes concrete defects
from plausible explanations of the Ante-5→6/fresh-run gap. Fixing any one
finding is not a promise of Ante-8 wins. Matched tests remain necessary.
