# Conditional output-weight optimization

The complete-panel gradient diagnostic is admitted: all 5,380 active training sales and 703 development sales reproduce, with independently checked gradients and curvature. Training and development gradients align (cosine 0.535). The training-only quadratic direction has held-out loss derivative −0.000728 per original row, with descriptive paired-game interval [−0.001296, −0.000203]. These are local derivatives, not achieved loss reductions. The saved Adam direction is slightly adverse on the complete training panel.

The existing hidden basis therefore contains a locally useful transferable correction. This supports testing optimization of the readout before changing its architecture. It does not establish feature completeness, explain the entire training trajectory, or demonstrate policy generalization.

## One controlled fit

Start from the admitted conditional-only 1,000-update endpoint. Freeze its encoder, old heads, all new projections, proposal probabilities, and output bias. Fit only the 64 output weights against the original teacher's conditional sale targets on all 5,380 active training sale states. All other rows have zero derivative into this head; normalize by the original 202,141 training rows. The auxiliary objective and non-sale likelihoods are constants in this fit.

Use deterministic float64 full-batch Newton optimization of this convex softmax objective, with no regularization. Initialize from the saved output weights. Compute the Newton direction with the existing machine-precision numerical-rank cutoff; reject a material unresolved gradient in the null space. Use training-only Armijo backtracking (coefficient 0.0001, halve from one, at most 32 attempts). Stop when half the squared Newton decrement is at most 1e-12 per original row and gradient infinity norm is at most 1e-8. At most 50 accepted updates; nonconvergence fails the fit. This is one optimizer trajectory, without development-selected steps, duration, coefficients, or regularization.

Cast the terminal weights to the checkpoint's float32 format. Save their float64 training trajectory and a new checkpoint without stale Adam state. Verify that only the declared output-weight tensor changed, independently reconstruct initial/final objective, gradient and Hessian, verify convergence after quantization, and reproduce the training predictions through native inference. Never overwrite the original checkpoint or any executed diagnostic source.

Budget: 300 active seconds for the fit and 300 for its audit, with 30 seconds of owner grace per phase. The feature computation is reused from the admitted diagnostic; native audit covers all 5,380 training sale states. Explicit user pauses suspend these clocks. The subsequent scan retains its existing measurement/audit budgets of 1,554/596 active seconds.

## Evaluation and decision

After the whole fit and audit succeed, run the unchanged full training/development scan and the original thirteen checks against base, ordinary1k, auxiliary, and disabled-readout controls. Preserve the coupled model as an additional descriptive control. Also compare against the actual starting conditional endpoint to isolate this intervention's effect. The exposed development panel is exploratory. No failed gate is relaxed to accommodate the frozen backbone.

Only a qualifying endpoint may advance to fresh-game evaluation. The final bar remains at least +5 percentage points sampled wins versus base and original ordinary4k, paired lower bounds above zero, greedy lower bounds strictly above −5 points versus both, positive early-policy attribution, and an independent final fit and untouched confirmation. Current-Ante subgroup diagnostics do not replace first-passage attribution. No learned win-rate gain has yet been established.

Sources: [admitted diagnostic](../../eval/consequence_prediction_20260927/conditional_full_gradient_owner_terminal_v2.json), [independent gradient and curvature audit](../../eval/consequence_prediction_20260927/conditional_full_gradient_v2/verification_v1.json), [interpretation](../../eval/consequence_prediction_20260927/conditional_full_gradient_findings_v1.json), [original thirteen-check implementation](../../eval/consequence_prediction_20260927/replacement_conditional_metrics_v1.py).
