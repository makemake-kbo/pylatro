# Sale-family and Joker-choice contributions

This saved-prediction decomposition compares the coupled readout with the same trained model when its readout is disabled. Negative NLL differences favor the coupled readout. It adds no qualification gate and measures imitation on exposed teacher states, not action value or win rate.

## Development

| Group | States / active | Joint NLL change | Macro component | Conditional Joker component |
|---|---:|---:|---:|---:|
| all_shop | 11596 / 7512 | -0.000002 | -0.000004 | +0.000002 |
| early_shop_sale | 162 / 160 | -0.001366 | -0.001396 | +0.000030 |
| later_shop_sale | 554 / 543 | -0.001934 | -0.001968 | +0.000033 |
| shop_non_sale | 10880 / 6809 | +0.000117 | +0.000117 | +0.000000 |
| shop_sale | 716 / 703 | -0.001806 | -0.001838 | +0.000033 |

| Group / decision | Assessed | Both correct | Gained | Lost | Both wrong |
|---|---:|---:|---:|---:|---:|
| all_shop / flat | 11596 | 10412 | 0 | 0 | 1184 |
| all_shop / macro | 7512 | 6895 | 0 | 1 | 616 |
| all_shop / conditional | 703 | 360 | 0 | 0 | 343 |
| early_shop_sale / flat | 162 | 50 | 0 | 0 | 112 |
| early_shop_sale / macro | 160 | 111 | 0 | 0 | 49 |
| early_shop_sale / conditional | 160 | 80 | 0 | 0 | 80 |
| later_shop_sale / flat | 554 | 109 | 0 | 0 | 445 |
| later_shop_sale / macro | 543 | 274 | 0 | 0 | 269 |
| later_shop_sale / conditional | 543 | 280 | 0 | 0 | 263 |
| shop_non_sale / flat | 10880 | 10253 | 0 | 0 | 627 |
| shop_non_sale / macro | 6809 | 6510 | 0 | 1 | 298 |
| shop_non_sale / conditional | 0 | 0 | 0 | 0 | 0 |
| shop_sale / flat | 716 | 159 | 0 | 0 | 557 |
| shop_sale / macro | 703 | 385 | 0 | 0 | 318 |
| shop_sale / conditional | 703 | 360 | 0 | 0 | 343 |

Macro and conditional counts cover active rows only; flat counts cover every row in the group. Macro and conditional winners are masked logit argmaxes. A flat-mode family mismatch is not necessarily a macro-head error. Inactive rows contribute zero change to NLL averages.

## Training

| Group | States / active | Joint NLL change | Macro component | Conditional Joker component |
|---|---:|---:|---:|---:|
| all_shop | 90355 / 57257 | -0.000007 | -0.000008 | +0.000001 |
| early_shop_sale | 1242 / 1207 | -0.001225 | -0.001244 | +0.000019 |
| later_shop_sale | 4229 / 4173 | -0.001858 | -0.001874 | +0.000017 |
| shop_non_sale | 84884 / 51877 | +0.000103 | +0.000103 | +0.000000 |
| shop_sale | 5471 / 5380 | -0.001714 | -0.001731 | +0.000017 |

| Group / decision | Assessed | Both correct | Gained | Lost | Both wrong |
|---|---:|---:|---:|---:|---:|
| all_shop / flat | 90355 | 83043 | 1 | 5 | 7306 |
| all_shop / macro | 57257 | 53089 | 4 | 2 | 4162 |
| all_shop / conditional | 5380 | 4366 | 0 | 0 | 1014 |
| early_shop_sale / flat | 1242 | 620 | 1 | 0 | 621 |
| early_shop_sale / macro | 1207 | 870 | 0 | 0 | 337 |
| early_shop_sale / conditional | 1207 | 976 | 0 | 0 | 231 |
| later_shop_sale / flat | 4229 | 1464 | 0 | 0 | 2765 |
| later_shop_sale / macro | 4173 | 2106 | 4 | 0 | 2063 |
| later_shop_sale / conditional | 4173 | 3390 | 0 | 0 | 783 |
| shop_non_sale / flat | 84884 | 80959 | 0 | 5 | 3920 |
| shop_non_sale / macro | 51877 | 50113 | 0 | 2 | 1762 |
| shop_non_sale / conditional | 0 | 0 | 0 | 0 | 0 |
| shop_sale / flat | 5471 | 2084 | 1 | 0 | 3386 |
| shop_sale / macro | 5380 | 2976 | 4 | 0 | 2400 |
| shop_sale / conditional | 5380 | 4366 | 0 | 0 | 1014 |

Macro and conditional counts cover active rows only; flat counts cover every row in the group. Macro and conditional winners are masked logit argmaxes. A flat-mode family mismatch is not necessarily a macro-head error. Inactive rows contribute zero change to NLL averages.

## Development paired source-game intervals

| Group / component | Difference [95% interval] |
|---|---:|
| all_shop / conditional_delta | +0.000002 [+0.000001, +0.000003] |
| all_shop / macro_delta | -0.000004 [-0.000019, +0.000012] |
| all_shop / nll_delta | -0.000002 [-0.000017, +0.000014] |
| early_shop_sale / conditional_delta | +0.000030 [-0.000013, +0.000072] |
| early_shop_sale / macro_delta | -0.001396 [-0.001704, -0.001114] |
| early_shop_sale / nll_delta | -0.001366 [-0.001677, -0.001081] |
| later_shop_sale / conditional_delta | +0.000033 [+0.000008, +0.000060] |
| later_shop_sale / macro_delta | -0.001968 [-0.002172, -0.001771] |
| later_shop_sale / nll_delta | -0.001934 [-0.002140, -0.001735] |
| shop_non_sale / conditional_delta | +0.000000 [+0.000000, +0.000000] |
| shop_non_sale / macro_delta | +0.000117 [+0.000107, +0.000128] |
| shop_non_sale / nll_delta | +0.000117 [+0.000107, +0.000128] |
| shop_sale / conditional_delta | +0.000033 [+0.000010, +0.000056] |
| shop_sale / macro_delta | -0.001838 [-0.001997, -0.001686] |
| shop_sale / nll_delta | -0.001806 [-0.001965, -0.001654] |

Intervals use 10,000 paired resamples of all 128 source games, seed1729, with state-count denominators. They are marginal descriptive intervals. Early means current Ante<4 and does not replace permanent first-passage early-policy attribution. A conditional improvement supports a useful within-model contribution; it cannot separate explicit relations from added capacity or establish autonomous generalization.

All 456,372 paired record identities, 20 aggregate controls and 30 independent categorical recounts pass. The maximum rowwise component-identity error is 1.71322659e-06, within 2e-5. Both primary coupled-versus-ablation paired NLL intervals reproduce within 1e-12. Runtime: 4.05s. No model, training, simulator or teacher call was used.

Multiple-legal-sale counts on active teacher-sale states: development 703, training 5380.

Artifacts: [replacement_relation_components_20260929.md](replacement_relation_components_20260929.md), [result_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_components_v1/result_v1.json), [replacement_relation_components_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_components_owner_terminal_v1.json).
