# Probe conditional learning in the existing endpoint

The admitted saved-score diagnostic finds roughly 0.000356 RMS variation in
the new residual across training sale options, versus 2.519 in the original
sale logits. No teacher-sale conditional winner changes. All six new tensors
move, and epsilon domination is absent except for 1.25% of state-projection
coordinates. The conditional output-space loss derivative remains nonzero.
This motivates a parameter-gradient diagnostic before selecting another fit.

Use the existing coupled endpoint and the first thirty logical batches of its
original shuffled row stream: exactly 9,600 training presentations, seed0,
namespace and 320-row batches unchanged. Reconstruct them from the frozen
training helper and compare the first320 to the admitted first-update record.
Do not select rows by errors, residual size or development results. This is a
post-hoc training diagnostic, not a fresh evaluation panel.

One MPS FP32 encoder pass per 32-row microbatch. Freeze existing parameters and
enable derivatives only for the six new readout tensors. This preserves their
partial derivatives in the existing network: the upstream features do not
depend on those six parameters. Keep all modules in evaluation mode, original
temperature1 and mixture0.5. Do not update any parameter or optimizer state.

For each original logical batch, record new-parameter gradients separately for
full action NLL, macro-family NLL and teacher-sale conditional NLL, each divided
by the original320 denominator. Other conditional heads do not depend on the
new readout. Full gradients must equal macro plus conditional gradients within
absolute2e-6 per coordinate. Compare native full-action modes and NLL against
the already admitted endpoint records (exact mode; NLL1e-5); record the maximum
error introduced by the different batch ordering.

Also differentiate full action NLL with respect to the applied sale residual.
On active legal slots, compare it to P'(sale_i)-1[teacher=sale_i], after undoing
the320 denominator, within2e-6. Mask inactive slots before saving this control:
the downstream tensor can have derivatives there even though its producer's
gate prevents those derivatives reaching the new parameters.

Save all thirty gradient vectors and their identities. Report norms and cosine
between macro and conditional components, their full-batch aggregate, and the
conditional loss directional derivative under the negative total gradient.
Report the corresponding derivative under the nominal endpoint Adam direction
from saved moments. Include every batch and all six tensor groups. Neither an
instantaneous gradient nor the historical-moment direction proves convergence,
a useful training step, or generalization. A sign conflict can motivate a
separately specified intervention; it cannot prove that intervention works.

An independent saved-array audit must reconstruct decomposition identities,
norms, dot products, parameter-name/coordinate mapping and residual-gradient
arithmetic from the admitted raw logits. The original checkpoint, all model
tensors and optimizer payload must remain unchanged. Pin helpers, plan, corpus,
row indices, endpoint, gradient arrays and result. Require successful actual
whole-owner completion before admitting results.

Serialize this with all native work. Allow180 active seconds for measurement
and60 for the saved-array audit, plus30 seconds owner cleanup each. Use the
tested pause-aware lifecycle; no automatic retry or budget extension. This is
zero optimizer steps and no new exploratory training run. No learning rate,
loss coefficient, duration, architecture or gameplay panel is selected here.
The existing endpoint remains rejected and the original final goal remains
unproven.
