# Broader supervision for the same fixed readout

The 1,000-game output-only fit reached its numerical optimum but failed seven of thirteen development checks. Its training conditional matches improved by 66, while held-out conditional matches fell by two. At that endpoint, the full-readout training gradient has negative cosine with the development gradient; the paired local-derivative interval spans zero. These results provide no affirmative reason to train the existing feature projections further. They do not prove that features cannot improve.

The next controlled question is whether broader independent game coverage improves the same 64-weight predictor. An earlier whole-policy data-scale experiment improved imitation but lost existing clears. Here every other parameter remains fixed to the admitted conditional-only 1,000-update parent. That parent and its representations are used unchanged; the weights from the earlier whole-policy data-scale experiment are not used.

## Fixed population and fit

Reuse the admitted original 1,000 games (202,141 rows) and the 4,000 additional current-rule games already collected in September (811,414 rows, seeds 4001000–4004999). The new cache matches the exact frozen runtime, observation schema, and dtypes; all 22 native arrays have been rehashed. It adds 21,402 gate-active teacher sale states to the original 5,380, for **26,782** selected states. Keep all games and both outcomes. The 128 development games remain separate. Do not collect, filter, replace, or relabel games.

Extract frozen-parent features on the additional selected states and audit their identities, masks, proposals, and readout reconstruction before fitting. This stage performs no optimization. Reuse the original selected features unchanged.

Train exactly one new model: the same 64 output weights, initialized from the same conditional parent as the 1,000-game solver experiment. Freeze the encoder, old heads, new feature projections, proposal pathway, and output bias. Minimize the same unregularized conditional teacher NLL, now summed over the combined population and divided by its original 1,013,555 rows. All omitted rows have zero derivative into this conditional-only head. Auxiliary and non-sale losses remain constant. This is a larger-population convergence comparison, not a matched-presentation comparison.

Reuse the full-batch float64 Newton solver unchanged: numerical-rank cutoff at machine epsilon × dimension × largest eigenvalue; training-only Armijo coefficient 0.0001, halving from one, at most 32 backtracks; at most 50 accepted updates; stop only when half the squared Newton decrement is at most 1e-12 and gradient infinity norm at most 1e-8. No regularization, validation-selected step, epoch, coefficient, or architecture sweep. Cast the final weights to float32 and verify that exactly one tensor changed.

## Prediction and decision

Predict lower development sale NLL and better sale matches than the completed 1,000-game output solver. The original thirteen checks against base, ordinary1k, auxiliary, and the disabled readout remain mandatory. Add the 1,000-game output solver as a descriptive control; it does not replace either original policy control. Record paired source-game intervals and current-Ante subgroups without treating these exposed data as confirmation.

Only a qualifying endpoint may advance to fresh-game evaluation. The sampled +5-point gains versus base and original ordinary4k, positive paired bounds, greedy guardrails, early-policy attribution, and one final independent fit/untouched confirmation remain required. There is no learned long-horizon or win-rate improvement yet.

## Bounded preparation and execution

Feature extraction and its independent audit are each capped at 600 active seconds, plus 30 seconds of owner grace per phase. The subsequent fit and endpoint audit are each capped at 600 active seconds; the full comparison retains the existing 1,554/596-second measurement/audit budgets. Explicit user pauses suspend timeout clocks. Model/native jobs remain serialized and long jobs are checked about ten minutes apart. No new fit starts before feature admission.

[1,000-game solver results](conditional_output_solver_results_20260930.md) · [Remaining-gradient diagnostic](../../eval/consequence_prediction_20260927/conditional_output_gradient_owner_terminal_v2.json) · [Additional-data census](../../eval/consequence_prediction_20260927/conditional_diversity_census_results_v1.json) · [Original dataset admission](../../eval/rule_aligned_scale_20260927/dataset_admission_v1.json)
