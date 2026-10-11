# Early-shop action consequences, 2026-09-22

The architectural hypothesis remains open. The completed
[pooling probe](representation_probe_20260922.md) did not improve held-out
prediction. This follow-up asks whether the policy's preferred early-shop action
leads to more terminal wins under the same continuation policy. It diagnoses
action ranking; it does not train a model or demonstrate generalization.

All 256 primary continuations and four duplicate controls are complete. The
[independent audit](data/shop_counterfactual_verification_20260922.json) replayed
all 260 trajectories exactly. The primary ranking result is **unresolved**.
The runner started after the main study completed confirmation and entered its
early-policy intervention. The
[admission record](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/early_shop_counterfactual/continuations/admission.json)
captures that transition.

## Verified results

The [completed result](data/shop_counterfactual_results_20260922.json) uses the
same 32 selected states, with four paired repeats per action and both fixed
training replicas. The intervals average repeats within each state, resample
states within replicas, and weight the two replicas equally.

| Comparison | Difference in win rate, percentage points | Pointwise 95% interval |
| --- | ---: | ---: |
| Policy-preferred action minus the other action | +4.6875 | [−7.03125, +17.1875] |
| Buy minus leave | +4.6875 | [−7.03125, +16.40625] |
| Critic-preferred action minus the other action | +6.25 | [−5.46875, +17.96875] |

The primary point estimates are +1.5625 points in replica 0 and +7.8125 in
replica 1. All three pooled intervals include zero. The fixed primary rule
therefore establishes neither poor ranking nor reliable alignment on these
pairs. The critic's larger point estimate does not establish that it ranks
choices better than the actor.

These are **32 reused training states**, not 256 independent states or unseen
full-game starts. The estimates remain compatible with meaningful positive and
negative effects. This bounded diagnostic does not identify an architectural
failure, justify a policy change, or demonstrate learned generalization. The
separate reserved early-policy intervention is still running.

The [per-state results](data/shop_counterfactual_states_20260922.json) and
[primary outcomes](data/shop_counterfactual_outcomes_20260922.jsonl) are exported
with the result and verification. All four exports match their source artifacts
byte for byte. State selection, preferences, sample counts, and decision rules
were unchanged.

## Fixed design

The [intent](data/shop_counterfactual_intent_20260922.json) was saved before state
replay. The first 64-game training cohort of each MC replica supplies candidates,
scanned in a fixed outcome-independent permutation. A game contributes its first
current-Ante-2/3 shop before reaching Ante 4 that has an affordable Joker, a legal
leave action, and a global greedy policy choice equal to the highest-probability
Joker purchase or leave. The first 16 eligible games per replica are retained.
No future outcomes determine eligibility or action selection.

Preparation scanned 34 games and selected 32 distinct states, all in Ante 2.
Each replica prefers buying in 13 states and leaving in three. The critic's
post-action win forecasts disagree with the actor in four states in replica 0
and five in replica 1. Disagreement alone establishes neither model as correct.

For each state, the intervention forces one purchase or leaving, then follows
the same fixed MC +20 replica with ordinary sampled actions, temperature 1,
CPU batch 1. Later actions may include selling the purchased Joker. Four paired
sampling repeats produce 256 primary continuations. A deterministic hash of
replica, original seed, snapshot step, and repeat assigns an independent CPU
sampling stream to each pair; both actions start from the same simulator RNG
snapshot and use the same pair seed. Different branches may consume different
numbers of random draws. The experiment does not require their later draws or
trajectory prefixes to stay aligned.

The primary contrast is the policy-preferred action minus the other action in
terminal win rate. Preferences are fixed before continuation outcomes. Four
repeats are averaged within each state, then 10,000 bootstrap samples resample
states within each replica and weight the two replicas equally. The bootstrap
seed is 1729. Buy-minus-leave and critic-preferred-minus-other are secondary,
pointwise contrasts. No action is selected from its measured outcomes to report
an apparent policy improvement.

A primary interval wholly below zero would indicate poor ranking of these pairs
on this exposed panel; an interval wholly above zero would indicate alignment.
Neither result would demonstrate a learned policy improvement on unseen games.
The 32 states reuse training games and intentionally cover a restricted choice
at early shops. They do not represent all strategic decisions.

## Replay verification and execution

The [prepared manifest](data/shop_counterfactual_preparation_20260922.json)
pins 32 snapshots, observations, source trajectories, both fixed models, and
their policy/critic preferences. Every selected game's original prefix and full
suffix were replayed, including independent snapshot restoration. All saved
observations, rewards, terminal flags, final observations, and outcomes matched
exactly. Preparation took 137.48 seconds with two CPU threads and generated no
new sampled continuations.

The [execution record](data/shop_counterfactual_execution_20260922.json)
pins the runner before any sampled continuation. It verifies main-study source,
model, snapshot, and preparation hashes. The main finisher used up to three
confirmation evaluation slots; this runner waited until confirmation finished.
Its own continuation budget is 5,400 monotonic seconds with two CPU threads,
no accelerator, and at most 4,096 decisions per continuation. Waiting does not
consume that continuation budget.

Two identical sampled buy continuations on the first state of each replica add
four control games. Their complete observation/action/reward/termination trace
hashes and outcomes must match each other and the corresponding primary run.
Five [regression checks](data/shop_counterfactual_test_verification_20260922.json)
cover CPU RNG isolation, stage admission, fixed-preference aggregation, real
snapshot/forced-action restoration, and complete/partial trace handling.

The runner saved completed outcomes and per-decision trace hashes incrementally.
STOP, changed inputs, time/decision caps, truncation, stalls, or failed duplicate
controls preserve partial files and fail without automatic retry. Its completed state is
in [continuations/status.json](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/early_shop_counterfactual/continuations/status.json).
The main study's selection rule and acceptance criteria remain unchanged.

The independent auditor completed with a pinned
[execution record](data/shop_counterfactual_audit_execution_20260922.json).
It restored every snapshot and replayed all 260 recorded continuations, including
the four controls and **37,030 decisions**. Every recorded action was legal, and
observations, rewards, terminal flags, and final outcomes matched exactly. Both
duplicate control pairs matched each other and their corresponding primary run.
The audit added no sampled games. It independently recomputed all three contrasts
from the 256 primary outcomes, including the original preferences, state pairing,
repeat averaging, bootstrap intervals, and unresolved classification. The
continuation stage took 2,095.96 seconds; the audit took 457.96 seconds. Both
finished within their original budgets and used no accelerator.

Three [audit regression checks](data/shop_counterfactual_audit_tests_20260922.json)
passed. They reject altered terminal labels, broken observation chains, incorrect
forced actions, and early termination even when the file hash is recomputed;
verify the statistics against a hand-computed example; and replay a real saved
game suffix before rejecting a deliberately incorrect outcome. The auditor has
a separate 1,800-second recorded-action replay cap after admission, two CPU
threads, and no accelerator. Its current state is in
[independent_audit/status.json](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/early_shop_counterfactual/independent_audit/status.json).
