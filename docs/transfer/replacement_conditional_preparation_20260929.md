# Conditional-only replacement scorer: verified preparation

The [experiment](replacement_conditional_experiment_20260929.md) removes the
sale-family correction from the existing relation scorer. The single matched
1,000-update fit has started; there is no admitted learned endpoint or held-out
improvement yet. This document records zero-update implementation checks only.

The complete technical-check owner exited successfully on September29 at
15:22:13UTC. Its [result](../../eval/consequence_prediction_20260927/replacement_conditional_check_v1/result_v1.json)
and [owner receipt](../../eval/consequence_prediction_20260927/replacement_conditional_check_owner_terminal_v1.json)
show:

* CPU float64 and MPS float32 checks pass six masked, empty, singleton,
  inactive and non-sale cases each. Maximum probability errors are1.11e-16 and
  4.06e-8; residual-gradient errors are2.78e-17 and6.71e-9.
* All320 initial native decisions and140 output-field comparisons reproduce
  base logits, modes and NLL exactly, with58 active readout states.
* Under nonzero corrections, macro probabilities and94 legal non-sale NLL
  comparisons remain exactly equal to the disabled readout. All244 legal
  action comparisons reproduce conditional normalization within1.34e-6.
  A common score shift changes NLL by at most9.54e-7 from finite precision,
  within the existing2e-5 tolerance.
* On the original first320 shuffled training rows, including16 active teacher
  sales, independently recomputed CPU float64 projections, GELU activations and
  proposal weights reproduce the output-weight gradient within4.98e-9. The
  output-bias gradient is1.17e-9, consistent with its theoretical cancellation.
  Initially zero output weights correctly give zero projection gradients.
* Strict checkpoint reload, rejection by the old coupled loader, hook cleanup
  on success and exception, and preservation of original tensors all pass.

The check took17.27seconds and made zero optimizer steps. The fit uses the same
57,473 new parameters, initialization, base, data, row stream, labels, auxiliary
loss, AdamW schedule and endpoint as the coupled experiment. Critic parameters
remain frozen. Existing encoder and macro parameters still train; removing
the new branch's macro correction does not freeze the base policy.

The [fit protocol](../../eval/consequence_prediction_20260927/replacement_conditional_fit_protocol_v1.json)
fixes4,537 active seconds for training and600 for its audit, with pause-aware
timing and operational recovery checkpoints. There is one exploratory fit and
no automatic retry or extension. Actual successful whole-owner completion is
required before the [six-policy comparison](replacement_conditional_scan_20260929.md).
The thirteen original spending checks and full fresh-game success bar remain
unchanged; these technical checks establish no learning or win-rate gain.
