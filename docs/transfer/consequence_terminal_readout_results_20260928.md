# Terminal readout fits training labels but fails validation

The [single terminal-readout fit](consequence_terminal_readout_20260928.md) converges and improves training action ordering, but fails its fixed spending rule. Validation terminal prediction is worse than both analytic baselines, and restricted choices win **115/200** continuations versus base's **116/200**. No separate held-out collection, policy integration or confirmation fit qualifies for this endpoint. The full long-horizon goal remains unproven.

The 385-parameter readout uses only the frozen wider scorer's action features. An outcome-independent seed hash assigns 800 games and their 2,400 branches to fitting, and 200 games/600 branches to terminal-label validation. All these source games were already used to train the frozen features. Validation is therefore an exploratory terminal-label transfer check, not unseen-source-game generalization.

| Metric | Training | Terminal-label validation |
|---|---:|---:|
| Terminal BCE | 0.446656 | 1.197680 |
| Training-prevalence baseline BCE | 0.692994 | 0.691375 |
| Ante-bucket baseline BCE | 0.663972 | 0.650190 |
| Terminal Brier error | 0.146361 | 0.274366 |
| Ante-bucket Brier error | 0.235751 | 0.229332 |
| Terminal pair ordering | 68.98% | 60.71% |
| Base pair ordering | 50.60% | 59.52% |
| Existing preference ordering | 48.80% | 61.90% |
| Same-clear terminal ordering, state weighted | 65.41% | 51.72% |
| Same-clear base ordering, state weighted | 48.87% | 53.45% |
| Restricted-choice terminal wins | 437/800 | 115/200 |
| Base terminal wins | 401/800 | 116/200 |

The validation BCE deficit against the ante baseline is **+0.54749 [0.10083, 1.35485]**; Brier error is worse by **+0.04503 [0.01449, 0.07748]**. Both intervals exclude zero. Action-ordering improvement over base is only **+1.19 percentage points [−16.67, +19.05]**, using 42 mixed-outcome games. The same-clear comparison is **−1.72 points [−24.14, +20.69]**, using 29 games. Restricted-choice wins differ by **−0.50 points [−5.00, +4.00]**. These wide exploratory intervals do not establish a policy benefit.

Early validation decisions have a favorable but unresolved point estimate: pair ordering **65.38% versus 55.77%**, with restricted choices **34/74 versus 32/74** terminal wins. Later ordering is **53.13% versus 65.63%**, and choices **81/126 versus 84/126**. We do not select an early-only policy from these outcomes. All restricted proposals contain the teacher action, and every branch uses the same base continuation after the first action.

Six full-batch Newton updates meet the fixed stationarity criterion, maximum absolute gradient **6.37×10⁻¹¹**, below 10⁻⁸. The independent audit checks the outcome-independent split, training-only normalization, objective derivatives, positive Hessians, every logged update, endpoint predictions, metrics and all bootstrap intervals. Fit and audit take **0.229 and 0.329 seconds**. This is one exploratory model and a numerical audit, not two independent fits. There is no evidence that failure to converge caused the validation result.

A separate [saved-error audit](consequence_terminal_readout_error_audit_20260928.md) identifies a specific extrapolation problem introduced by this linear probe's featurewise standardization. State `4000487:112` contributes **66.10% of net excess validation BCE** versus the ante baseline. All three continuations lose, while predicted logits are **79.40, 63.35 and 76.62**. One hidden coordinate has training standard deviation **0.00764**, but a validation activation of **10.73** becomes **1,403.54** standardized units; its largest training value was **34.28** standardized units. That coordinate alone contributes **+91.34** to one predicted logit. This is exact feature-contribution accounting, not evidence that the original transformer had the same failure.

The extreme state is not the whole problem. The other 199 states still contribute mean excess BCE **+0.18654**. All observations remain in the failed result; none are removed or clipped. Validation median state BCE is 0.64188 versus training 0.36489. The maximum absolute logit is 79.40 in validation versus 12.23 in training. Frozen features support substantially better training ordering, but the present sample/readout/normalization recipe does not establish useful transfer.

The next justified controlled change is to prevent rare-coordinate scaling from amplifying unseen activations: test per-proposal RMS normalization with the same labels, split, objective, optimizer and spending criteria. This is motivated by measured input extrapolation, not a coefficient or checkpoint sweep. It may improve calibration while leaving action transfer weak; that competing explanation remains. The failed endpoint stays rejected, and the reused validation data remains exploratory.

Artifacts: [fit owner](../../eval/consequence_prediction_20260927/terminal_readout_owner_terminal_v1.json), [result](../../eval/consequence_prediction_20260927/terminal_readout_v1/result_v1.json), [audit](../../eval/consequence_prediction_20260927/terminal_readout_v1/verification_v1.json), [error-audit owner](../../eval/consequence_prediction_20260927/terminal_readout_error_audit_owner_terminal_v1.json), and [error accounting](../../eval/consequence_prediction_20260927/terminal_readout_error_audit_v1.json).
