# Full-batch optimization of the conditional output weights

Fails 7 of thirteen original checks. This endpoint does not qualify for fresh-game evaluation or confirmation.
No learned win-rate or long-horizon generalization gain is established.

One training-only fit changed exactly one tensor: the existing 64 output weights. All 325 other state tensors remain identical. The solver accepted 3 Newton updates and met its fixed stationarity checks; development data chose neither steps nor stopping. This is an additional output-only optimization stage from the prior 1,000-update endpoint, not a matched-exposure rerun of that fit.

Training conditional loss per original corpus row changed from 0.014356164 to 0.013936879. Output-weight norm changed from 0.004174 to 7.557160. Independent arithmetic and all 5,380 selected native training decisions pass; maximum native logit error is 3.51e-06.

| Policy | Held-out sale matches | Sale NLL | Future purchase matches | Purchase matches |
|---|---:|---:|---:|---:|
| Base | 167/716 | 2.311783 | 428/469 | 4990/5323 |
| Ordinary 1k | 170/716 | 2.394442 | 426/469 | 4979/5323 |
| Purchase auxiliary | 158/716 | 2.406345 | 441/469 | 4987/5323 |
| Coupled scorer | 159/716 | 2.436645 | 437/469 | 4976/5323 |
| Starting conditional scorer | 165/716 | 2.433829 | 438/469 | 4983/5323 |
| Disabled readout | 165/716 | 2.433811 | 438/469 | 4983/5323 |
| Output solver | 161/716 | 2.425616 | 438/469 | 4983/5323 |

Against the starting endpoint, sale NLL changes by -0.008214 [-0.028552, +0.010986]; sale exact-match rate changes by -0.005587 [-0.018132, +0.006757]. Intervals are paired source-game 95% bootstrap intervals on the exposed development panel, without multiple-comparison correction.

## Original checks

| Check | Result |
|---|---|
| future_nll_vs_base | PASS |
| future_nll_vs_ordinary1k | PASS |
| non_sale_nll_vs_ordinary1k | PASS |
| purchase_exact_vs_base | FAIL |
| purchase_exact_vs_ordinary1k | PASS |
| sale_exact_vs_ablation | FAIL |
| sale_exact_vs_auxiliary | PASS |
| sale_exact_vs_base | FAIL |
| sale_exact_vs_ordinary1k | FAIL |
| sale_nll_vs_ablation | PASS |
| sale_nll_vs_auxiliary | FAIL |
| sale_nll_vs_base | FAIL |
| sale_nll_vs_ordinary1k | FAIL |

## Limits and provenance

The output solver is a controlled test of optimization in the existing hidden basis. A development imitation change does not prove a win-rate increase. The sale-family probabilities, future-purchase logits, and hidden features are frozen. Full action modes can nevertheless change when probability is redistributed among sale actions. Current-Ante subgroup counts are descriptive and do not replace the final early-policy intervention.

The full scan and independent native audit succeeded before these results were admitted. The original thirteen gates and final fresh-game, greedy-retention, early-attribution, and independent-confirmation requirements remain unchanged.

The first fit owner failed because its audit could not serialize a NumPy integer in the final receipt. The completed checkpoint and failed attempt are preserved. The corrected audit repeated the same checks on that checkpoint, with a separate successful owner; no second fit occurred. The unused fit time was reassigned to the repeated audit within the original combined 600-second budget.

[Experiment](conditional_output_solver_20260930.md) · [Fit owner](../../eval/consequence_prediction_20260927/conditional_output_fit_owner_terminal_v1.json) · [Fit audit](../../eval/consequence_prediction_20260927/conditional_output_audit_recovery_v1/verification_v1.json) · [Audit repair](../../eval/consequence_prediction_20260927/conditional_output_audit_recovery_protocol_v1.json) · [Scan owner](../../eval/consequence_prediction_20260927/conditional_output_scan_owner_terminal_v1.json) · [Scan audit](../../eval/consequence_prediction_20260927/conditional_output_scan_v1/verification_v1.json)
