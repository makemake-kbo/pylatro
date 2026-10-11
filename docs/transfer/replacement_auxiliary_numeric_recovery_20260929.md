# Correct an unsupported recovery repeatability requirement

The pause-aware replacement owner49900 exited1 (`bd15cf`) at its first
scheduled check. The worker completed update119's optimizer step, then
failed the newly introduced `prefix_numeric_tolerance=1e-5` assertion before
logging that update. It saved118 complete update records and no endpoint.
No pause occurred during this replacement execution. Its active duration
was263.4677seconds. Preserve this failed v2 execution and its exact criterion.

Saved-state comparison (`0cdb60`) establishes identical initial named
parameters, full model state, initialization seed, trainability and initial
policy controls. All118 saved update identities, learning rates and eligible
row counts match v1. The first objective matches exactly. Maximum difference
among saved loss components is6.433576e-6; update118's objective differs by
6.234190e-6. The exact update119 loss was not saved, so its discrepancy cannot
be quantified beyond the failed assertion. Do not claim exact trajectories
or diagnose a particular nondeterministic kernel from these observations.

The recovery-only loss-trajectory requirement was unsupported. The original
training recipe did not enable deterministic algorithms, and identical seeds
alone do not eliminate every source of nondeterminism; see
[PyTorch2.11 reproducibility guidance](https://docs.pytorch.org/docs/2.11/notes/randomness.html).
The ordinary control used that same original runtime. Changing algorithm
settings would change the matched recipe. We have not established the exact
cause of these small floating-point trajectory differences.

Version the infrastructure recovery again. Retain exact initialization,
recipe, batches, exposure and all original gradient/optimizer/critic checks.
Record and independently recount the differences against both failed
prefixes without imposing a new loss-trajectory stopping threshold. This
does not relax a held-out performance criterion or admit either failed
attempt. No endpoint has been evaluated or selected. Re-execute the same
fixed recipe to obtain its first complete exploratory endpoint; there is
still no confirmation fit or changed learning configuration. Report the
123updates from v1 and119executed updates from v2 as additional spent work.

Add an atomic latest-recovery checkpoint every100updates, containing model,
optimizer, RNG, row-stream state, exposure and gradient-probe records. This
is operational recovery data, never an alternative endpoint for selection.
Keep the final endpoint fixed at1000updates. Do not automatically retry any
further failure or inspect recovery prefixes for performance selection.
Use the already-tested pause-aware owner, unchanged fixed active-time
budgets, and the same nine held-out spending checks and final goal standard.
