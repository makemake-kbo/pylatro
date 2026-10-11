# Including equal-outcome states does not repair terminal transfer

The [single relative-mean fit](consequence_terminal_relative_mean_20260929.md)
reduces validation prediction error against the previous RMS readout but remains
worse than predicting zero action difference. Validation ordering is **40/84
(47.62%)**, versus base **50/84 (59.52%)**. **Seven of nine spending checks fail.**
The candidate is rejected; no held-out collection, integration or independent
confirmation fit follows from this result.

The preceding conditional-density fit used only 311 of 1,600 training states.
This experiment includes the remaining 1,289 equal-outcome states in squared
relative-outcome regression, while retaining the same labels, frozen features,
384 parameters, RMS normalization, ridge coefficient and source-game split.
Those states supply **78.79% of the data-curvature trace**, so they materially
constrain the new fit. A realized zero outcome difference remains a noisy sample,
not a claim that the actions have equal expected values. The objective and its
curvature change together; this does not isolate exclusion alone.

| Centered outcome MSE | Relative-mean readout | Prior RMS readout | Zero difference |
| --- | ---: | ---: | ---: |
| Training, 1,600 states | 0.037695 | 0.054418 | 0.043194 |
| Validation, 200 states | **0.053152** | 0.064502 | **0.046667** |

Validation error falls by **−0.011350 [−0.019091, −0.003464]** against RMS, but
exceeds zero difference by **+0.006486 [+0.002664, +0.010872]**. These are paired
95% whole-game bootstrap intervals on exposed development data. The fixed RMS
control was originally fitted on the first 800 training states; the preceding
conditional-density control used the same full 1,600-state panel as the new fit.

The improvement against RMS comes from the 158 equal-outcome validation states:
their centered MSE falls from **0.020193 to 0.004905**. On the 42 mixed-outcome
states it moves from **0.231186 to 0.234655**, an unresolved difference of
**+0.003468 [−0.030871, +0.036594]**. Reducing unsupported score variation helps
prediction error without establishing useful action discrimination.

| Ordering / restricted-choice diagnostic | Relative mean | Conditional density | Base | RMS |
| --- | ---: | ---: | ---: | ---: |
| Training terminal pairs | 490/622 | 553/622 | 311/622 | 374/622 |
| Validation terminal pairs | **40/84** | 40/84 | **50/84** | 51/84 |
| Early validation terminal pairs | 28/52 | 28/52 | 29/52 | 36/52 |
| Later validation terminal pairs | 12/32 | 12/32 | 21/32 | 15/32 |
| Validation restricted-choice wins | **108/200** | 109/200 | **116/200** | 117/200 |
| Early restricted-choice wins | 29/74 | 31/74 | 32/74 | 36/74 |

Identical aggregate pair counts do not mean identical per-pair predictions.
The overall ordering difference against base is **−11.90 percentage points
[−28.57, +4.76]**, unresolved. Restricted-choice outcomes differ by **−4.0 points
[−8.5, +0.5]** against base and **−4.5 [−9.0, −0.5]** against RMS. The same-clear
state-weighted ordering is **44.83%**, versus 53.45% for base. The later ordering
deficit against base has an exploratory interval below zero, but involves only
16 mixed states; no early-only candidate is selected.

Training ordering reaches 78.78%, down from the conditional fit's 88.91% as
equal-outcome examples constrain relative scores. Its improved training loss
still does not transfer. This weakens the proposed remedy of including those
states while keeping the feature pathway fixed; it does not establish that all
relative-value learning, more diverse data or trainable features would fail.
Further coefficient, density and loss variants are not justified by this result.
The next decision should examine whether terminal supervision can learn useful
action features, with representation and data sufficiency still competing
explanations. This report selects no new architecture or data collection.

The fit takes **0.345 seconds** and the independent audit **0.385 seconds**. The
normal-equation residual is **8.67×10⁻¹⁸**. The audit derives the equivalent
pair-difference objective, verifies positive curvature and stationarity without
another optimization, checks the initial gradient against the previous
conditional objective, and reproduces all 5,400 scores, fixed controls, splits,
metrics, intervals and gates. The complete owner actually exits zero before
admission. No simulator or encoder inference runs.

All source games contributed to the original feature fit, and the validation
split is adaptively reused. The three-action proposals include teacher choices;
restricted suffix wins are not autonomous win rates. Original-RNG labels remain
realized outcomes under the fixed greedy continuation. The full held-out
prediction/retention, fresh sampled/greedy games, early-policy attribution and
final independent-fit requirements remain unmet. World models remain deferred.

Artifacts: [owner receipt](../../eval/consequence_prediction_20260927/terminal_relative_mean_owner_terminal_v1.json),
[frozen protocol](../../eval/consequence_prediction_20260927/terminal_relative_mean_protocol_v1.json),
[complete results](../../eval/consequence_prediction_20260927/terminal_relative_mean_v1/result_v1.json),
[independent audit](../../eval/consequence_prediction_20260927/terminal_relative_mean_v1/verification_v1.json),
and [per-state metrics](../../eval/consequence_prediction_20260927/terminal_relative_mean_v1/state_metrics.jsonl).
