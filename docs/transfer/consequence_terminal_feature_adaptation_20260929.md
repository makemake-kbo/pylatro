# Test terminal supervision through a cloned consequence feature pathway

**Question.** The [all-state relative-mean readout](consequence_terminal_relative_mean_results_20260929.md)
fits training action differences but fails validation. Can terminal supervision
learn more useful action features, or does the gap persist when those features
can adapt? Limited contexts and noisy realized outcomes remain competing
explanations. This is a feature-learning test, not another loss or coefficient
variant.

Copy the trained wide consequence scorer and initialize its terminal linear
readout from the completed relative-mean fit. Unfreeze the copy's query,
cross-attention and hidden projection, and optimize these together with the
terminal readout. Keep the original policy, transformer, scorer checkpoint and
three immediate-output rows unchanged. Those immediate-output rows are unused
and frozen inside the copy too. The copy still consumes frozen transformer
features; this does **not** test an independently trained raw-observation encoder.

Use the exact existing 1,600 training decisions from 800 games, 200 validation
decisions, three proposals per decision and original-RNG greedy-base labels.
Keep the preceding centered squared-error objective, per-action RMS normalization,
zero intercept and 0.001 ridge on the terminal readout. Do not add an auxiliary
loss, feature penalty, preservation weight or new labels. All five approved
observation-feature arrays retain their existing order. Seeds, teacher identity
and outcomes remain metadata/targets, never model inputs.

The copy has 542,400 trainable feature parameters plus 384 terminal weights;
the 1,155 unused immediate-output parameters stay frozen. Initialize exactly
at the prior fitted predictor. This increases trainable capacity without
changing the feature architecture. The nonlinear optimizer differs necessarily
from the preceding closed-form readout solver; a failed fit need not establish
an architectural limit.

**Preparation before fitting.** Reproduce all 1,800 initial three-action hidden
representations exactly in the existing sorted full-support, single-state,
512-action inference layout. Reproduce prior relative scores within the existing
float64 arithmetic tolerance. Verify action membership, retained cards, all
source identities, the game split and frozen-input provenance. Check finite,
nonzero gradients in each trainable feature group, no gradients in the unused
heads, and the analytic centered-loss derivative. Benchmark three fixed training
batches without an optimizer step. This preparation is not another fit.

**One fixed training recipe.** Inherit the consequence branch's CPU FP32,
one-thread, AdamW learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, zero
weight decay, batch size 128, gradient clip 1 and 20 epochs. RMS normalization,
terminal weights and relative loss use float64 as in the control. Each epoch
orders the 1,600 training rows by SHA256 of
`terminal-feature-order-v1:0:<epoch>:<seed>:<step>`, epochs 1–20. There are 260
updates and 32,000 state presentations. Save epochs 0, 5, 10 and 20; the endpoint
is fixed at 20, with validation measured only at that endpoint. No duration or
learning-rate search follows a failure. Training uses three proposals per state;
all endpoint comparisons use canonical full-support inference.

Proceed to a separately frozen execution protocol only after preparation passes
and the measured projected fit fits within 600 seconds. The fit and independent
audit each have a 600-second cap, plus supervisor shutdown grace. The audit must
verify exposure, first and final saved optimizer transitions, feature changes,
frozen weights, initial-control reproduction, all final outputs and statistics.
It must not run another training trajectory. No simulation or encoder inference
is required. Report the benchmark and its limitations rather than promising the
old scorer's throughput.

**Prediction and spending rule.** Training relative MSE must improve on the
optimal frozen-feature readout. Validation relative MSE must beat zero, RMS and
the frozen-feature control, each with paired 95% upper difference below zero.
Validation pair ordering must strictly beat base and retain RMS; early ordering
must retain base; same-clear ordering must exceed 50%; restricted-choice wins
must retain base overall and early. These ten checks only permit consideration
of a separately specified source-game holdout. Use the existing 10,000 whole-game
bootstrap, seed 1729, and preserve early/later and equal/mixed-outcome accounting.
Training gains without validation gains weaken this feature-adaptation recipe;
failure to improve training fit leaves optimization unresolved.

All source games contributed to the immediate-feature fit and validation has
been reused. No result on this panel proves unseen-game generalization. Teacher
proposals and restricted realized suffix wins are not a deployment policy or an
autonomous win rate. Full held-out consequence prediction and retention, every
selected alternative assessed, fresh sampled/greedy games, early-policy
attribution and final independent-fit confirmation remain required. Original
failed candidates stay rejected; world models remain deferred.
