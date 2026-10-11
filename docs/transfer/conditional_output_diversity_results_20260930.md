# More game contexts for the same 64-weight readout

Fails 7 of the original thirteen development checks. This endpoint does not qualify for gameplay or confirmation.
No learned long-horizon or win-rate gain is established by this teacher-state evaluation.

This experiment changes only the population used to fit the existing output weights: 1,000 → 5,000 current-rule teacher games, with 5,380 → 26,782 eligible sale states. It reuses already collected games, including wins and losses. Architecture, parent initialization, hidden features, objective, and numerical stopping criteria are unchanged. Both solvers run to the same convergence criterion; presentations are not matched.

The single broader-data fit accepted 3 Newton updates. Exactly one tensor (64 output weights) changed; 325 other state tensors are identical to the parent. No new feature learning or encoder update occurred. Training-only line search and stopping did not consult development data.

## Training conditional choice

| Population | Selected sales | Initial NLL | Fitted NLL | Initial matches | Fitted matches |
|---|---:|---:|---:|---:|---:|
| Original 1,000 games | 5380 | 0.539399 | 0.541931 | 4364 | 4396 |
| Additional 4,000 games | 21402 | 1.280320 | 1.241604 | 11533 | 11628 |

These NLLs and matches describe which Joker is selected given a sale. They are distinct from full joint-action matches. The original and additional games were not filtered by outcome. Fitted matches are from native inference.

## Fixed development comparison

| Policy | Sale matches | Sale NLL | Future purchase matches | Purchase matches |
|---|---:|---:|---:|---:|---:|
| Base | 167/716 | 2.311783 | 428/469 | 4990/5323 |
| Ordinary 1k | 170/716 | 2.394442 | 426/469 | 4979/5323 |
| Purchase auxiliary | 158/716 | 2.406345 | 441/469 | 4987/5323 |
| Coupled scorer | 159/716 | 2.436645 | 437/469 | 4976/5323 |
| Output solver: 1,000 games | 161/716 | 2.425616 | 438/469 | 4983/5323 |
| Disabled readout | 165/716 | 2.433811 | 438/469 | 4983/5323 |
| Output solver: 5,000 games | 158/716 | 2.401581 | 438/469 | 4982/5323 |

Relative to the 1,000-game output solver, sale NLL changes by -0.024034 [-0.047357, -0.001750]; sale exact-match rate changes by -0.004190 [-0.018284, +0.008783]. These are descriptive paired source-game 95% bootstrap intervals on the exposed development panel, without multiple-comparison adjustment.

## Unchanged checks

| Check | Result |
|---|---|
| future_nll_vs_base | PASS |
| future_nll_vs_ordinary1k | PASS |
| non_sale_nll_vs_ordinary1k | PASS |
| purchase_exact_vs_base | FAIL |
| purchase_exact_vs_ordinary1k | PASS |
| sale_exact_vs_ablation | FAIL |
| sale_exact_vs_auxiliary | FAIL |
| sale_exact_vs_base | FAIL |
| sale_exact_vs_ordinary1k | FAIL |
| sale_nll_vs_ablation | PASS |
| sale_nll_vs_auxiliary | PASS |
| sale_nll_vs_base | FAIL |
| sale_nll_vs_ordinary1k | FAIL |

## Verification and limits

All 21,402 additional feature rows passed a separate native replay. The endpoint audit checked all 26,782 selected training rows; maximum native logit error is 1.01e-05, NLL error 5.38e-06, and independent objective/gradient/Hessian error 7.98e-17. It recorded 0 float64/native near-boundary mode disagreements under the prospective rounding rule. Native predictions determine every evaluation result.

The full scan and its independent native audit completed before admission. Sale-family probabilities and all other heads remain fixed at a given state. Better teacher-state likelihood alone would not establish improved games; the original fresh-game comparisons, greedy guardrails, early-policy attribution, and final independent confirmation are still necessary.

[Experiment](conditional_output_diversity_20260930.md) · [Fit owner](../../eval/consequence_prediction_20260927/conditional_diversity_fit_owner_terminal_v1.json) · [Fit audit](../../eval/consequence_prediction_20260927/conditional_diversity_fit_v1/verification_v1.json) · [Scan owner](../../eval/consequence_prediction_20260927/conditional_diversity_scan_owner_terminal_v1.json) · [Scan audit](../../eval/consequence_prediction_20260927/conditional_diversity_scan_v1/verification_v1.json) · [Original versus additional context diagnostic](../../eval/consequence_prediction_20260927/conditional_diversity_support_results_v1.json)
