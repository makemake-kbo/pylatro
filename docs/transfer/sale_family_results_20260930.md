# Independent sale-family result

The single 193-parameter family fit fails 3 of the 13 original development checks. This endpoint stops before gameplay and confirmation. No learned win-rate improvement or long-horizon generalization is established by this scan.

All existing model tensors remain fixed. Only the newly added weight and bias learn from all 286,044 active training states, including 259,262 non-sales, using the fixed 192-dimensional observation features. The training-only Newton fit converged after 5 accepted updates. The native audit replayed all 286,044 training states and verified that the conditional action logits and other family logits remain unchanged.

| Development policy | Sale matches | Sale NLL | Purchase matches | Future-purchase matches |
| --- | ---: | ---: | ---: | ---: |
| Base | 167/716 | 2.31178298 | 4990/5323 | 428/469 |
| Ordinary 1k | 170/716 | 2.39444228 | 4979/5323 | 426/469 |
| Auxiliary | 158/716 | 2.40634531 | 4987/5323 | 441/469 |
| Coupled | 159/716 | 2.43664531 | 4976/5323 | 437/469 |
| Conditional parent | 160/716 | 2.23037282 | 4990/5323 | 438/469 |
| Original all-readout ablation | 165/716 | 2.43381118 | 4983/5323 | 438/469 |
| Independent family head | 163/716 | 2.19205868 | 4991/5323 | 438/469 |

Against the conditional parent, sale NLL changes by -0.03831414, with marginal paired-game 95% interval [-0.05333057, -0.02374910]. Sale exact-match rate changes by 0.00418994, with interval [-0.00681245, 0.01617707]. These are descriptive intervals from an already exposed development panel. The conditional parent is an added comparison; it does not replace an original gate or the original all-readout ablation.

| Training partition | Active states | Teacher sales | Initial binary NLL | Deployed native binary NLL |
| --- | ---: | ---: | ---: | ---: |
| Original | 57,257 | 5,380 | 0.14298369 | 0.14227298 |
| Additional | 228,787 | 21,402 | 0.15881156 | 0.15608299 |

| Original check | Result |
| --- | --- |
| `future_nll_vs_base` | PASS |
| `future_nll_vs_ordinary1k` | PASS |
| `non_sale_nll_vs_ordinary1k` | PASS |
| `purchase_exact_vs_base` | PASS |
| `purchase_exact_vs_ordinary1k` | PASS |
| `sale_exact_vs_ablation` | FAIL |
| `sale_exact_vs_auxiliary` | PASS |
| `sale_exact_vs_base` | FAIL |
| `sale_exact_vs_ordinary1k` | FAIL |
| `sale_nll_vs_ablation` | PASS |
| `sale_nll_vs_auxiliary` | PASS |
| `sale_nll_vs_base` | PASS |
| `sale_nll_vs_ordinary1k` | PASS |

The scan audited 456,372 new policy records and 52,730 independent native encoder rows, reconstructed the family and conditional changes in full action NLL, and preserved the original 13 gate definitions. The whole fit owner and whole scan owner both exited successfully before this report was rendered. No development label selected a fitting step, endpoint or coefficient.

The final proof bar remains fresh-game sampled improvement versus the base and original ordinary-4k control, greedy retention, positive learned early-policy attribution, and one final independent fit with untouched confirmation. Ordinary-1k remains an additional diagnostic control. No confirmation outcome was inspected here.

[Fit plan](sale_family_fit_20260930.md) · [Scan plan](sale_family_scan_20260930.md) · [Fit owner](../../eval/consequence_prediction_20260927/sale_family_fit_owner_terminal_v1.json) · [Scan owner](../../eval/consequence_prediction_20260927/sale_family_scan_owner_terminal_v1.json) · [Native scan audit](../../eval/consequence_prediction_20260927/sale_family_scan_v1/verification_v1.json).
