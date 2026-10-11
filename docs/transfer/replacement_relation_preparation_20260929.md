# Direct sale–purchase coupling: preparation complete

The single matched 1,000-update fit is running. The learning and generalization
hypothesis remains untested; the completed checks establish that the new
readout is connected to the policy as intended.

The [experiment](replacement_relation_experiment_20260929.md) keeps the failed
auxiliary run's data, future-purchase loss and training recipe. It adds an
initially neutral 57,473-parameter readout of owned-Joker/offer pairs, weighted
by current purchase preference. Its [joint-action correction](replacement_relation_readout_20260929.md)
can change both whether to sell and which Joker to sell. Non-sale SHOP states
provide negative examples through the unchanged full-action loss.

The admitted technical check took 16.79 seconds with zero optimizer steps:

- CPU float64 joint probabilities agree with independent flat-action
  enumeration within 1.12e-16; gradients agree within 2.78e-17.
- MPS float32 errors are at most 4.07e-8 for probabilities and 2.36e-8 for
  gradients. Empty supports, inactive gates and illegal slots stay masked.
- Neutral initialization reproduces all 320 original-layout native modes,
  action NLL values and 140 output-field comparisons exactly. These include
  58 active replacement-gate states; archived base NLL also reproduces exactly.
- A nonzero readout agrees with the intended joint-action tilt on 244 legal
  native comparisons within 1.70e-6, below the fixed 2e-5 tolerance.
- Strict checkpoint reload preserves the complete model. Original parameters
  remain unchanged, and the transient integration hook is removed after both
  success and a deliberately injected exception.

The training and audit code preserve the original optimizer, auxiliary loss,
backward, step and zero-gradient calls. Six new trainable tensors are appended
to the actor. The first actual update additionally records a direct check of
the new output-bias gradient, independently reconstructed from sale probability
and teacher sale/non-sale labels. Exposure, optimizer state, critic retention,
neutral controls and endpoint loading remain mandatory.

After successful fit admission, the fixed held-out screen must pass all nine
original comparisons and improve sale NLL and exact agreement over both the
previous auxiliary endpoint and the trained model with its new readout disabled
(13 point checks total). An improvement over a disabled readout is an inference
ablation within one trained model, not a separate capacity-matched training
experiment. No extra replica or final confirmation is commissioned now.

The fit has 4,517 active seconds and its audit 600, with tested user-pause
accounting and 30 seconds cleanup grace per phase. The only admitted training
endpoint is update 1,000. Recovery files are operational. Long-running process
checks remain roughly ten minutes apart.

Artifacts: [technical result](../../eval/consequence_prediction_20260927/replacement_relation_check_v1/result_v1.json),
[whole-process receipt](../../eval/consequence_prediction_20260927/replacement_relation_check_owner_terminal_v1.json),
[frozen fit protocol](../../eval/consequence_prediction_20260927/replacement_relation_fit_protocol_v1.json),
[implementation](../../eval/consequence_prediction_20260927/replacement_relation_v1.py).

Fresh-game sampled wins, greedy retention, learned early-decision attribution
and final independent-fit verification remain necessary. This preparation is
not evidence of improved win rate or long-horizon generalization.
