# Terminal feature adaptation fits training outcomes but does not transfer

Allowing the copied consequence feature pathway to learn produces **622/622
correct training comparisons**, up from 490/622 with frozen features. Validation
ordering is **41/84 (48.81%)**, versus base **50/84 (59.52%)**, and validation
relative prediction error worsens. **Nine of ten prospective spending checks
fail.** The candidate is rejected; no held-out collection, policy integration
or independent confirmation fit follows from this result.

The [fixed experiment](consequence_terminal_feature_adaptation_20260929.md)
keeps the preceding relative-outcome objective, labels, split, RMS normalization
and terminal-readout penalty. It starts exactly from the fitted frozen-feature
predictor, then trains a copy's query, attention and hidden projection together
with its terminal weights: **542,784 trainable parameters**. The transformer
inputs remain frozen. This tests terminal feature adaptation within the
consequence branch, not an independent raw-observation encoder. The existing
policy and immediate-consequence predictor are not replaced.

One fixed 20-epoch fit processes 1,600 decisions from 800 source games, producing
260 updates and 32,000 state presentations. Validation is measured at the fixed
final endpoint. No earlier checkpoint is selected from validation results.

| Measurement | Frozen-feature control | Adapted features | Base |
| --- | ---: | ---: | ---: |
| Training centered outcome MSE | 0.037695 | **0.001362** | — |
| Validation centered outcome MSE | **0.053152** | 0.063722 | — |
| Training terminal ordering | 490/622 | **622/622** | 311/622 |
| Validation terminal ordering | 40/84 | **41/84** | 50/84 |
| Early validation ordering | 28/52 | 21/52 | 29/52 |
| Later validation ordering | 12/32 | 20/32 | 21/32 |
| Restricted-choice validation wins | 108/200 | **108/200** | 116/200 |
| Early restricted-choice wins | 29/74 | 27/74 | 32/74 |

The validation MSE increase against frozen features is **+0.010570
[+0.004389, +0.017315]**. Against zero relative prediction, whose error is
0.046667, the excess is **+0.017056 [+0.010566, +0.023933]**. The previous RMS
readout scores 0.064502; the new difference of **−0.000780
[−0.010595, +0.009488]** is unresolved. Intervals are paired, 10,000 whole-game
bootstrap resamples on the reused development panel.

Error increases on both outcome partitions. On 158 equal-outcome validation
states, centered MSE rises from **0.004905 to 0.012017**, difference **+0.007112
[+0.004908, +0.009646]**. On 42 mixed states it rises from **0.234655 to 0.258233**,
with an unresolved difference **+0.023578 [−0.004773, +0.054609]**. The new pathway
can fit training outcomes closely while assigning unsupported differences to
validation actions.

Overall ordering against base differs by **−10.71 percentage points
[−27.38, +5.95]**, unresolved. Restricted choices differ by **−4.0 points
[−8.5, +0.5]** against base. Their equal aggregate win count against the frozen
control does not imply identical choices. Same-clear state-weighted ordering
remains **44.83%**, versus base's 53.45%.

Later ordering improves by **+25 points [+6.25, +43.75]** against the frozen
control, but involves only 16 mixed states and remains below base in point
estimate. Early ordering declines, and early relative MSE increases by
**+0.025146 [+0.011005, +0.041225]**. These exploratory partitions do not qualify
a later-only policy or overturn the failed whole-candidate decision.

Preparation reproduces all **2,073,600 initial hidden coordinates exactly** and
the prior scores within **4.44×10⁻¹⁶**. It verifies usable gradients without any
optimizer step. The fit then changes all three feature groups and the terminal
weights while preserving the 1,155 unused immediate-output parameters. It takes
**7.96 seconds**; the independent audit takes **5.71 seconds**. The whole owner
exits successfully before admission.

The audit checks the game split, action membership, every prescribed exposure,
all saved checkpoints, and the first and final optimizer transitions using an
independently expressed pair-difference loss. It replays both initial and final
hidden outputs through the original branch interface, reproducing **4,147,200
coordinates exactly**, and verifies all final scores, metrics, intervals and
gates. It does not repeat the training trajectory. No simulator or encoder
inference is needed.

This result shows that the terminal training comparisons are learnable by the
available branch, but making that branch trainable is insufficient for transfer
under this recipe. It does not establish that no representation could generalize,
or separate sparse context coverage from unpredictable realized return variation.
Further same-panel loss, learning-rate or duration searches are not justified
by this result. The next decision must address those remaining data/target
questions rather than equating additional training fit with progress in policy
quality.

All 1,000 source games contributed to immediate-feature training; terminal-label
validation is adaptively reused. Restricted teacher-containing proposals and
realized greedy-base suffix outcomes are not autonomous win rates or known
expected action values. The full source-game holdout, retention, fresh sampled
and greedy games, early-policy attribution and final independent-fit standard
remain unmet. World models remain deferred.

Artifacts: [preparation owner](../../eval/consequence_prediction_20260927/terminal_feature_preflight_owner_terminal_v1.json),
[fit owner](../../eval/consequence_prediction_20260927/terminal_feature_fit_owner_terminal_v1.json),
[frozen fit protocol](../../eval/consequence_prediction_20260927/terminal_feature_fit_protocol_v1.json),
[results](../../eval/consequence_prediction_20260927/terminal_feature_fit_v1/result_v1.json),
and [independent audit](../../eval/consequence_prediction_20260927/terminal_feature_fit_v1/verification_v1.json).
