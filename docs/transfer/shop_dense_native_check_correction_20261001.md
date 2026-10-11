# Native auxiliary check: match gradient tracking modes

The first native check exited 1 before any optimizer steps. Its nonzero auxiliary
fixture compared `relation.distribution` inside `torch.no_grad()` with an
auxiliary-enabled forward outside that context. Exact `macro_logits` equality
failed. The saved failure, source, protocol, worker receipt, and whole-owner exit
are preserved under their original v1 names.

The proposed correction changes the test harness only. Version 2 compares the
original and auxiliary-enabled policies with gradient tracking enabled on both,
and separately under `no_grad()` on both. Both comparisons still require exact
equality of every policy output field, selected actions, and teacher-action log
probabilities. No equality tolerance is widened. It also records the original
policy's difference between the two gradient tracking modes to test whether that
explains the earlier failure. A mode-dependent numerical difference is a
hypothesis until measured.

All original loss/gradient checks, zero-initialization and RNG checks, archived
controls, shared-feature gradient checks, critic checks, hook cleanup checks,
strict checkpoint round trip, data, model implementation, and the 180-second
active budget plus 30-second owner grace remain unchanged. The only extra policy
calls compare the nonzero auxiliary fixture under matching inference conditions.
No model fit, simulator game, or optimizer step is part of this correction.

Training remains gated on actual successful completion of the corrected check.
The learning recipe and all scientific qualification criteria remain unchanged.
