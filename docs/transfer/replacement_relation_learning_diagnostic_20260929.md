# Why did the new readout learn little conditional preference?

This is a post-hoc diagnostic of the rejected single coupled fit, motivated by
its admitted component analysis: conditional sale NLL worsens slightly even on
training states. It does not select a new policy, change the thirteen checks,
or admit the failed endpoint for gameplay. No new fit, backbone forward,
simulator query, label or confirmation data is used.

Inspect the frozen implementation, training loop and decoder. Verify inclusion
of every added parameter in the optimizer and the loss path from corrected
macro/conditional logits to action NLL. The initial output is zero: initially
the projection layers receive zero gradient through that output, while the
output layer can learn immediately. This is an initialization property, not
evidence that the projections remain disconnected. The existing first-step
bias-gradient check does not check every conditional contrast gradient.

Use both training and development saved records, all active SHOP rows and
teacher-sale/non-sale subsets. On legal owned-Joker and present offered-Joker
slots, report the following without selecting subgroups by their outcomes:

* Applied score mean, centered RMS and range; original sale-logit RMS/range;
  conditional-probability total variation; raw conditional winner changes,
  including changes between two wrong choices; and flat action changes.
* Uniform two-way decomposition of the pair-score matrix into grand mean,
  owned effect, offer effect and interaction. Report aggregate squared energy
  and reconstruction/orthogonality checks. These are properties at fixed
  contextual features, not proof that the encoder lacks offer information.
* Difference between the recorded purchase-weighted residual and a uniform
  average over present offers, after removing their common owned-slot mean.
  This describes sensitivity of explicit Joker contrasts to proposal weights;
  it is not a counterfactual environment or a policy comparison.
* Reconstruct the downstream action-NLL derivative with respect to each sale
  residual, g_i = P'(sale_i) - 1[teacher=sale_i]. Split it into family and
  conditional components, (P'(sale)-y_sale)*P'(i|sale) and
  y_sale*(P'(i|sale)-1[i=teacher]). Propagate through the saved pair weighting
  to pair scores and purchase logits. Report their magnitudes and the
  conditional component's projection on the current residual. These are
  output-space derivatives; they cannot by themselves establish parameter
  gradient quality, a beneficial optimizer step or long-horizon value.

Use float64 arithmetic on saved uncorrected logits and scores for analytic
derivatives, and test them against central differences on a small synthetic
masked categorical example. Use actual FP32 additions when counting native
logit-argmax transitions. Reproduce all prior conditional correctness counts
and mean component NLL changes. Check every paired record identity and input
hash. Report counts and fixed descriptive quantiles (min,25%,median,75%,95%,
max); no coefficient, threshold, training-duration or panel sweep.

Separately inspect CPU checkpoint tensors and optimizer moments, without any
model forward/backward or optimizer step. Recreate only the six-parameter-set
readout initializer with its frozen seed. For each tensor report initial and
final RMS, displacement RMS/max/fraction changed, first/second-moment RMS and
the fraction of coordinates whose bias-corrected Adam denominator is dominated
by epsilon. Report the last nominal Adam update computed from the saved moments
as an estimate, not as a reconstruction of the complete trajectory. Require
all six tensors to map uniquely to saved trainable names and 1,000-step states.
Include unchanged recipe and existing initial-bias-gradient evidence. Re-pin
checkpoints after inspection.

One process, 120-second maximum, no automatic retry or extension. Preserve any
failure. Pin script, plan, sources, inputs and result; require an actual
successful whole-process exit before rendering its report. All results are
descriptive and exposed; they can motivate a separately specified intervention
but cannot qualify one. The full fresh-win, retention, early-attribution and
final independent-fit bar remains unchanged.
