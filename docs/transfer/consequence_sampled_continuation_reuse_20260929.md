# Sampled continuation targets: reuse before collection

Repeated policy sampling has a defined target with the original simulator snapshot
and RNG preserved, but our existing four-repeat evidence does not establish
reliable per-state action preferences. This review launches no collection or fit.
It follows the [RNG stress result](consequence_terminal_rng_stress_results_20260929.md),
whose artificial RNG replacements are excluded from training.

The [completed early-shop experiment](shop_counterfactual_20260922.md) already
sampled four continuations per action from 32 fixed snapshots, buying or leaving,
under two older MC policies. The two actions shared the initial simulator RNG
and the policy-sampling seed for each repeat. Its independent audit reproduced
all 260 trajectories, including four duplicate controls. Its original aggregate
policy-preference contrast was unresolved: +4.6875 percentage points,
95% interval [−7.03125, +17.1875]. That result remains unchanged.

The new [saved-output analysis](../../eval/consequence_prediction_20260927/sampled_continuation_reuse_v1/result_v1.json)
checks the pinned outcomes, manifest, execution and audit records, reconstructs
the paired sampling seeds, reproduces every saved state win count and the
primary point estimate, and recounts variation directly from the raw rows.
The [whole analysis process exited successfully](../../eval/consequence_prediction_20260927/sampled_continuation_reuse_owner_terminal_v1.json).
It uses only the Python standard library and generates no gameplay.

| Descriptive measurement | Count |
| --- | ---: |
| State–action proposals with both wins and losses across four repeats | 33/64 |
| States with tied four-repeat action averages | 15/32 |
| Same strict preference in repeats 0–1 and 2–3 | 6/32 |
| Opposite strict preferences in those halves | 3/32 |
| One half tied | 11/32 |
| Both halves tied | 12/32 |

This split is post hoc and descriptive. Ties do not establish equal expected
values, reversals do not identify which half is wrong, and constant outcomes
over four repeats do not establish zero variance. The 256 outcomes are clustered
within 32 selected training states, all early shops. These are different models,
source versions and actions from the current play-state consequence panel;
their labels cannot be merged into it.

For a frozen sampled policy, repeated continuations estimate
`P(win | full simulator snapshot, forced action, fixed policy)` over policy
randomness. They do not integrate over hidden game states conditional on the
model's observation. Switching the current greedy continuations to sampled
continuations also changes the controller being evaluated; any improvement
would not isolate the benefit of averaging. These distinctions were already
identified in the [critic research note](research_critic_sources_20260916.md).

The current frozen [evaluation implementation](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/training/ppo_evaluation.py)
samples from a global Torch stream in batches. A branch-local CPU stream would
define a reproducible sampled controller but would not reproduce its exact
batch-32 trajectories. The [grammar sampler](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/action_grammar.py)
also samples multiple action components, including unused branches. Equal
starting policy seeds do not guarantee aligned later random draws or lower
variance. Existing sampler tests and the historical duplicate controls already
address basic correctness; another generic sampling pilot is unnecessary.

**Decision:** do not launch a four-repeat collection and turn the winning average
into a hard preference. A future sampled-target experiment must separate target
estimation from label evaluation, preserve uncertainty rather than promote an
empirical winner to an oracle, and justify the allocation between repeats and
distinct games. It must distinguish controller changes from averaging. The old
256-continuation run took about 35 minutes plus eight minutes of replay audit;
this is historical cost, not a throughput estimate for the current model.
No next experiment is selected here. The full held-out prediction, retention,
sampled/greedy gameplay, early-decision attribution and final independent-fit
requirements remain unchanged; generalization and a win-rate gain remain unproven.
