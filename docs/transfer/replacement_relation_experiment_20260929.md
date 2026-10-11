# Test a direct connection between purchase preference and sale decisions

**Question.** The [auxiliary experiment](replacement_auxiliary_results_20260929.md)
learned future purchases without improving sales. The
[connection census](replacement_auxiliary_connection_results_20260929.md)
finds sale errors in 325 of 441 held-out states with a correct prospective
purchase. Does an explicit owned-Joker/offer readout transfer that available
purchase preference into useful enabling actions? A competing explanation is
that the representation or sale labels still do not support the relevant
comparisons; extra readout capacity could improve fitting without transfer.

**One change.** Add a 57,473-parameter, initially zero residual. For each
ordered owned-Joker slot and offered-Joker slot, a shared hidden-64 MLP scores
their contextual transformer features together with the existing 128-vector
state. Average each owned Joker's pair scores using the existing purchase
head's softmax over all present Joker offers, including currently unbuyable
offers. This proposal comes entirely from the current observation. Apply the
residual only in SHOP when Joker capacity is exhausted, at least one Joker
offer is present and at least one Joker sale is legal. All such states,
including teacher non-sale choices, participate through ordinary action NLL.
No pair labels, new observations, simulator targets or hidden state are added.

Use the residual as an additive energy for each legal Joker-sale action.
Adjust both the conditional sale logits and the sale-family logit so the
result equals a normalized tilt of the original joint action distribution.
This prevents a common positive score from cancelling inside the conditional
softmax: the same score can influence whether to sell and which Joker to sell.
Keep meaningful token/slot order; do not sort or collapse owned Jokers.
The exact algebra and implementation checks are in the companion
[readout specification](replacement_relation_readout_20260929.md).

**Controlled fit.** Start from the same immutable base, not the failed auxiliary
endpoint. Keep its corpus, complete row stream, training seed, trainability,
optimizer, MPS FP32, batch320/micro32 and fixed 1,000-update endpoint. Keep the
same prospective-purchase loss, coefficient one and original-batch denominator.
The only change from the completed auxiliary run is the added coupling. Train
one exploratory model. No coefficient, width, duration or eligibility sweep.
The critic stays fixed. Atomic recovery checkpoints are operational only.

**Prediction and decision.** Reuse base, ordinary1k and auxiliary measurements.
The new endpoint must pass the original nine held-out spending comparisons
against base/ordinary1k and additionally improve sale NLL and exact agreement
over the auxiliary endpoint. Measure the learned endpoint with its new residual
disabled, using the same encoder output; require improved sale NLL and exact
agreement with the residual enabled. These are 13 point-estimate checks;
report paired source-game intervals and every failure. The ablation measures
the residual's contribution within the trained model, not an independent
capacity-matched training control. Future-purchase performance against the
auxiliary endpoint is reported; it is not an additional tuned threshold.

If any check fails, stop this candidate before gameplay or confirmation and
identify whether fitting, transfer, retention or effective use of the new
readout failed. If all pass, freeze a separate fresh-game SHOP-only evaluation
with base behavior elsewhere, matched controls, sampled wins, greedy retention
and learned early-decision attribution. Final qualification still includes the
original ordinary control, a single independent fit at the end and untouched
confirmation. Teacher-state agreement is not a win-rate result.

**Budget.** The completed auxiliary fit took 2,153 seconds. Allow the same
4,517 active seconds for 1,000 updates and 600 seconds for the endpoint audit,
plus 30 seconds owner cleanup per phase. Expect roughly 36–45 minutes before
evaluation. Before fitting, run one mathematical/native integration check with
zero optimizer steps, capped at 180 seconds. Reuse tested pause-aware ownership;
check long jobs about ten minutes apart. Freeze inference costs separately
before that stage. No automatic extension, training pilot or retry. World
models and additional data changes remain deferred.
