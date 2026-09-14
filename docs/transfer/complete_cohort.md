# Complete-game outcome supervision for PPO

This experiment tests whether removing completion-window selection from the main
outcome loss improves critic predictions and subsequent gameplay. The deliverable
is an opt-in implementation, bounded CPU verification, and a prepared experiment
harness. Preparation assigns **zero accelerator hours** and starts no training.
The earlier outcome-PPO study and its artifacts remain unchanged.

The existing GAE implementation already bootstraps artificial rollout boundaries.
The confirmed problem is different: the main categorical outcome loss labels only
the portion of a game inside its completion window. Winning games tend to last
longer, so their early states receive fewer labels. Earlier complete-episode replay
does include prefixes, but trains only the private outcome tower. The controlled
frozen-representation study establishes a pessimistic forecast effect of label
selection. The previous combined critic treatment improved calibration without a
gameplay gain. See [the completed evidence and limitations](outcome_ppo_results.md).

## Problems, prior art, and implementation obligations

| Problem | Primary source and relevant result | Application here | Test or artifact; interpretation limit |
| --- | --- | --- | --- |
| Outcome labels are selectively unavailable | [Rubin (1976), *Inference and Missing Data*, summary](https://academic.oup.com/biomet/article-abstract/63/3/581/270932) identifies conditions under which a missingness mechanism can be ignored. | Describe the label-selection mechanism explicitly and compare complete, completion-selected, and randomly missing labels on identical data. | `test_completion_selected_early_wins_are_missing`; `diagnose` artifacts. The application to our collector is our analysis, supported by our local experiment, not a result about Balatro proved by Rubin. |
| Terminal outcomes should supervise earlier positions | [Silver et al. (2017), AlphaZero, Equation 1](https://arxiv.org/html/1712.01815) trains value predictions against the final game result. | Attach each completed game's outcome to every preceding decision. | Chunk-invariance and artifact round-trip tests. This borrows final-outcome supervision, not search targets, heuristic guidance, or a promise of AlphaZero-like performance. |
| Different policy versions generate different continuation outcomes | [Schulman et al. (2017), PPO, Section 5 and Algorithm 1](https://arxiv.org/pdf/1707.06347) collects using the old policy before optimizing; [Espeholt et al. (2018), IMPALA, Section 4](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) corrects behavior/learner mismatch with V-trace. | Freeze the entire collection model until every declared game completes. Exclude older-policy replay from both arms. | Policy hashes, recomputed log-probability tests, and exact resume. Targets evaluate the collection policy, not an exactly evaluated post-update policy. Clipped V-trace is not an unbiased categorical-label replacement and is not implemented here. |
| Artificial cuts must preserve future value | [Pardo et al. (2018), *Time Limits in Reinforcement Learning*, partial-episode bootstrapping](https://proceedings.mlr.press/v80/pardo18a.html). | Retain the existing bootstrap at artificial cuts; zero bootstrap only for true termination. | Numerical GAE and terminal/reset regressions. This protects already-correct behavior; it does not diagnose a generic PPO truncation bug. |
| Critic error can affect long-horizon advantages | [Schulman et al., GAE, Section 3, Equations 16-18](https://arxiv.org/html/1506.02438v6) describes the bias-variance tradeoff and dependence on value accuracy. | Hold gamma, lambda, fragment boundaries, and the return-residual loss fixed while changing the outcome mask. | Exact target and microbatch comparisons. Corrected outcome supervision does not remove all approximation, bootstrap, or delayed-credit error. |
| Probability estimates need appropriate losses and metrics | [Gneiting and Raftery (2007), Sections 1 and 3](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) develops proper scoring rules, including logarithmic and quadratic scores. | Retain categorical NLL; measure NLL, Brier scores, initial calibration, and return errors on fresh games. | Forecast denominator and finite-output tests; evaluation reports. Proper scoring rules target the supplied outcome distribution and do not repair biased data selection. |
| Measured learning must be distinguished from fitting | [Jordan et al. (2020), *Evaluating the Performance of Reinforcement Learning Algorithms*](https://proceedings.mlr.press/v119/jordan20a.html) motivates reliable, reproducible RL performance measurement. | Freeze panels and endpoints, use two training replicas, and report paired gameplay differences with uncertainty. | Comparison-seed validation, clustered-statistics tests, frozen selection. Our paired bootstrap and numerical thresholds are study choices, not prescriptions from this paper. |

Each reference has a concrete role. The missing-label mechanism is distinct from
policy lag, correct fragment bootstrapping, and general long-horizon credit
assignment. DAgger remains relevant to the earlier imitation investigation; this
experiment supplies no teacher labels or replacement actions.

### Target and selection definitions

For the fixed collection policy `pi_old`, terminal-only rewards and gamma one give
`G_t = 10 * I(win)`. With complete outcomes from that policy,
`E[G_t | s_t] = V^{pi_old}(s_t)`. Our categorical loss estimates terminal outcome
probabilities conditional on the model's observation/history representation.
State aliasing and limited model capacity remain possible sources of error.

Let `X` be the model input, `Z` the terminal class, and `M` label availability.
Ignoring selective availability can fit `P(Z | X, M=1)` rather than `P(Z | X)`.
Whether selection is ignorable depends on what is observed and conditioned on;
we do not infer a universal missing-data classification from episode length alone.
The code reconstruction and frozen-data control establish the problem here.

For a logical slot counter `c_t` and fragment length `L`, the control mask is
`M_t = I(floor(c_t / L) == floor(c_last / L))`. The treatment mask is one for
every decision in every normally completed game. Slot counters continue across
cohorts and games; GAE traces always stop at game termination.

The actor still uses truncated GAE. Complete outcomes supervise the categorical
head only; the return residual, actor objective, and shared gradient routing are
unchanged. The resulting neural critic need not be perfectly calibrated, and its
output after optimization is not an exact current-policy value function.

## Implementation and compatibility

`PPOConfig.collection_mode` defaults to `fixed_steps`. The new mode is
`complete_cohort`, with `cohort_games=64`, `cohort_training_rows=8192`,
`cohort_max_episode_steps=2048`, `cohort_workers=4`, and an explicit, distinct
`cohort_seeds` list. `outcome_label_mode` is `completion_window` or
`complete_episode`. Legacy checkpoints lacking these fields retain legacy behavior.

The public `train_ppo` entry point dispatches the new collector through a dedicated
orchestrator that calls the existing production `_run_ppo_update`. Architecture,
reward schema, and v13 observation semantics are unchanged. The engine-worker
transport adds `step_training`; its evaluation `step` interface stays compatible.

Assign two games to each of 32 logical slots before collection. A finished slot
starts its next assigned game. Every assigned game must complete; selecting only
the fastest finishers would introduce another duration-dependent selection.
All model parameters and buffers are hash-checked across collection, dropout is
off, and gradients are disabled, including for the shared encoder.

A stalled game, artificial episode limit, worker error, illegal action, or
non-finite output fails the whole cohort before training. No replacement seeds
or invented loss labels are allowed. A terminal result exactly at the step limit
is valid. Failures retain diagnostics and leave the preceding checkpoint intact.

CPU storage copies complete observations immediately, preserves all history and
scalar fields, and bit-packs only binary legality masks. Compact trajectories,
final observations, actions, behavior log probabilities, value predictions,
rewards, policy hashes, slot counters, outcomes, and draw indices are saved with
checksums and an explicit tokenizer version and semantics identifier. Only each
physical minibatch is materialized on the training device.

After collecting the full cohort, compute and normalize GAE across its valid rows.
Draw 8,192 transitions uniformly with replacement and make two PPO passes. This
keeps physical optimizer exposure fixed across arms; it is a declared study
adaptation, not an assertion that PPO requires resampling with replacement. Log
actual collected transitions, unique sampled transitions, duplicate fraction,
label exposure, and optimizer examples separately. Logical mask denominators are
computed on CPU and include partial microbatches correctly.

Terminal replay, SIL, return-path imitation, archives, scoring interventions, and
changing reward/entropy schedules are excluded from this first mode. Both arms
retain the reference return-residual loss and shared encoder gradients. A KL stop
that prevents the prescribed optimizer exposure fails the pilot update and must
be reported; the harness does not silently retune or increase its training budget.

Checkpoints commit only at completed update boundaries. They retain model and Adam
state, CPU/MPS RNG, the full recipe, seed cursor, logical slot counters, and actual
transition/example counts. Strict resume rejects changed collection or learning
settings. The initial branch is explicit and records its parent hash and every
override. An interrupted attempt restarts the same seeds/RNG from the preceding
checkpoint and preserves its prior trajectory artifact under an attempt suffix.
Old missing MPS generator states cannot be recovered: such initializations use
the replica's configured MPS seed in both arms and record that limitation.

## Experiment protocol and commands

The harness uses the two frozen `gae097` update-42 parents named in the completed
study's selection manifest. Each replica produces a completion-window control and
a complete-label treatment. Their weights and Adam state are preserved at the
branch. Policies and trajectories will diverge during online training; only the
frozen-data diagnostic has identical training records throughout.

The pilot has 20 additional cohort updates, with measurements at initialization,
10 and 20. Settings are FP32, Adam LR `1e-5`, logical batch 640, physical batch 320,
gamma `1`, lambda `.97`, entropy `.001`, mixture `.5`, temperature `1`, Blue Deck,
White Stake, Ante 8, and the inherited terminal-only rewards and KL guards.
Inherited no-progress limit, gradient clipping, and other unchanged settings are
recorded in each complete recipe rather than inferred from current defaults.

Training lists contain 1,280 games per replica starting at `7000000` and `7100000`.
Development reuses `190000-190199`; new confirmation reserves `210000-210399`.
Preparation checks inherited source manifests, explicit training panels, actual
PPO episode logs, and available historical evaluation manifests, recording their
hashes. Any conflict or changed required source manifest fails preparation.
Historical panels are also excluded from new training. Evaluation uses all three
sampling seeds `0,1,2`, plus greedy play, on CPU with batch 32 and four workers.

Prepare and verify only:

```sh
.venv/bin/python tools/study_complete_cohort.py prepare
.venv/bin/python tools/study_complete_cohort.py verify-cpu
```

The default artifact root is `eval/complete_cohort_v1`. `--root PATH` precedes the
subcommand and selects a new isolated directory. Preparation neither allocates an
accelerator budget nor copies parents into an active training run. Verification
retains its command, source hashes, JUnit results, log, and tiny CPU smoke artifacts.

After a new accelerator allowance has been assigned, the operator can use
`budget --hours HOURS` once, followed by explicit `train` commands. For example,
the command shape for one already-budgeted arm is:

```sh
.venv/bin/python tools/study_complete_cohort.py train \
  --arm complete_episode --replica 0 --endpoint 10 \
  --device mps --estimate-seconds 3600
```

Run both arms and replicas at each endpoint. The runtime estimate is admission
input, not an allocation; the ledger enforces the assigned cumulative cap and
charges failed jobs too. No training command is executed as part of implementation.
No automatic longer continuation or default-mode promotion is provided.

Use `evaluate --endpoint 0`, `evaluate --endpoint 10`, and
`evaluate --endpoint 20` for the full development panels; `report` recomputes
their summaries. `freeze` requires both trained endpoints and chooses the largest
positive paired sampled gain with both replicas positive and greedy point change
above -5 points. Greedy gain then the later endpoint break ties. Frozen hashes
include the matched control at the same endpoint. Confirmation requires this
selection and an assigned study budget; use `evaluate --endpoint N --stage
confirmation`. No eligible development candidate means no confirmation launch.

The `diagnose --cohort PATH --checkpoint PATH --output NEW_PATH` command verifies
that the checkpoint exactly matches the collection-policy hash, splits complete
games deterministically into fitting and validation halves, and fits complete,
completion-selected, and count-matched random masks. Defaults are two ordering
replicas, 100 AdamW updates, LR `3e-4`, weight decay `.01`, batch 320, and frozen
representations. Only the private outcome tower changes in this diagnostic.
These are held-out games for that fixed-data fit, not a new gameplay improvement
test or independent held-out PPO dataset. The online main-loss experiment retains
shared-encoder gradients.

## Verification and interpretation

Bounded tests cover label selection, observation/history preservation and chunk
invariance; artificial and terminal boundaries; frozen-policy probabilities and
legality; incomplete-cohort rejection; CPU versus worker transport for both decks;
lazy materialization, mask denominators and gradient accumulation; complete
trajectory round trips; exact interrupted/resumed CPU training; initialization
provenance, reservations, and incompatible configuration rejection; forecast
denominators and paired-statistics aggregation. No teacher actions are permitted
in the CPU PPO smoke test.

Reports distinguish game-weighted and decision-weighted forecast errors. Initial
forecasts and observed sampled wins measure calibration under each evaluated
policy. Greedy forecasts are diagnostic because the value head predicts the
sampled behavior policy, not the greedy continuation. Repeats and training replicas
are averaged within game seed before bootstrapping. Intervals condition on these
two trained replicas and do not estimate variance over new training seeds.

A later confirmation result is validated only with at least +5 percentage points
sampled improvement, a paired 95% interval excluding zero, positive differences
in both replicas, a greedy interval ruling out worse than -5 points, and no new
legality, non-finite, or infrastructure failures. Report absolute win rates and
physical game counts prominently. Passing this threshold validates the label
intervention against its matched cohort control; it does not establish that this
collector outperforms ordinary PPO or that five points is the final objective.

The report must distinguish established local causes, source-supported principles,
and unresolved hypotheses. Lower critic loss and passing unit tests are never
substitutes for stronger autonomous gameplay.
