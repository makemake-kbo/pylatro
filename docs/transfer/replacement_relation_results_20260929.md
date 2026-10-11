# Coupled sale–purchase readout at 1,000 updates

The fixed held-out spending screen fails. This endpoint does not qualify for fresh-game integration or an independent confirmation fit.

The change adds a 57,473-parameter relation between contextual owned-Joker and offered-Joker features, weighted by current purchase preference. Its action-energy correction affects both sale-family probability and conditional Joker choice. The base, corpus, row stream, optimizer, future-purchase labels, auxiliary coefficient and endpoint match the earlier auxiliary experiment. Teacher non-sales provide negative supervision through ordinary action NLL.

All development rows are from the existing exposed 128-game teacher-state panel. Agreement and likelihood do not measure autonomous win rate or expected action value. The ablation disables the added readout within the same trained model; it measures that readout's contribution at inference, not a separate capacity-matched training intervention.

## Training states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Readout disabled | Coupled readout |
|---|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 3494 | 0.236988 / 3198 | 0.233550 / 3203 | 0.176879 / 3252 | 0.175041 / 3251 | 0.175041 / 3251 |
| Future purchase, all eligible | 5183 | 0.159760 / 4887 | 0.157442 / 4892 | 0.119239 / 4941 | 0.118000 / 4940 | 0.118000 / 4940 |
| SHOP sale | 5471 | 1.610484 / 1963 | 1.513097 / 2103 | 1.526982 / 2063 | 1.505214 / 2084 | 1.503500 / 2085 |
| SHOP purchase | 41729 | 0.158364 / 39509 | 0.156281 / 39466 | 0.155313 / 39505 | 0.154554 / 39502 | 0.154692 / 39498 |
| Non-sale SHOP | 84884 | 0.147912 / 80821 | 0.144693 / 80823 | 0.141171 / 80948 | 0.140083 / 80964 | 0.140186 / 80959 |
| Early SHOP sale | 1242 | 1.279034 / 592 | 1.173554 / 629 | 1.192736 / 621 | 1.170155 / 620 | 1.168930 / 621 |
| Later SHOP sale | 4229 | 1.707826 / 1371 | 1.612816 / 1474 | 1.625145 / 1442 | 1.603616 / 1464 | 1.601758 / 1464 |
| Early discard | 3705 | 2.846316 / 1355 | 2.849884 / 1317 | 2.844830 / 1331 | 2.843886 / 1332 | 2.843886 / 1332 |
| All decisions | 202141 | 0.496954 / 173497 | 0.467944 / 174709 | 0.466587 / 174859 | 0.465563 / 174858 | 0.465560 / 174854 |

## Development states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Readout disabled | Coupled readout |
|---|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 469 | 0.217848 / 428 | 0.221066 / 426 | 0.171388 / 441 | 0.170408 / 437 | 0.170408 / 437 |
| Future purchase, all eligible | 674 | 0.151589 / 633 | 0.153828 / 631 | 0.119260 / 646 | 0.118578 / 642 | 0.118578 / 642 |
| SHOP sale | 716 | 2.311783 / 167 | 2.394442 / 170 | 2.406345 / 158 | 2.438451 / 159 | 2.436645 / 159 |
| SHOP purchase | 5323 | 0.185486 / 4990 | 0.190388 / 4979 | 0.186792 / 4987 | 0.189009 / 4976 | 0.189154 / 4976 |
| Non-sale SHOP | 10880 | 0.176525 / 10250 | 0.179500 / 10219 | 0.174552 / 10252 | 0.174209 / 10253 | 0.174326 / 10253 |
| Early SHOP sale | 162 | 1.849605 / 55 | 2.025562 / 56 | 2.001099 / 49 | 2.045111 / 50 | 2.043745 / 50 |
| Later SHOP sale | 554 | 2.446933 / 112 | 2.502310 / 114 | 2.524847 / 109 | 2.553471 / 109 | 2.551537 / 109 |
| Early discard | 514 | 3.487149 / 168 | 3.677664 / 157 | 3.664179 / 160 | 3.671002 / 160 | 3.671002 / 160 |
| All decisions | 26045 | 0.639106 / 21544 | 0.648459 / 21539 | 0.646685 / 21543 | 0.648141 / 21547 | 0.648141 / 21547 |

Future-purchase predictions are exactly identical for the coupled model and its disabled-readout ablation. Singleton supports are automatic choices; the primary prospective comparison uses 3,494 training and 469 development states with multiple offers. Early means current Ante<4, including Ante0 and re-entry. This descriptive split does not redefine the final learned early-policy attribution requirement.

## All fixed spending checks

| Comparison | Result |
|---|---|
| `future_nll_vs_base` | PASS |
| `future_nll_vs_ordinary1k` | PASS |
| `non_sale_nll_vs_ordinary1k` | PASS |
| `purchase_exact_vs_base` | FAIL |
| `purchase_exact_vs_ordinary1k` | FAIL |
| `sale_exact_vs_ablation` | FAIL |
| `sale_exact_vs_auxiliary` | PASS |
| `sale_exact_vs_base` | FAIL |
| `sale_exact_vs_ordinary1k` | FAIL |
| `sale_nll_vs_ablation` | PASS |
| `sale_nll_vs_auxiliary` | FAIL |
| `sale_nll_vs_base` | FAIL |
| `sale_nll_vs_ordinary1k` | FAIL |

These thirteen deterministic point comparisons were fixed before outcomes. They are a spending screen, not significance tests or final qualification. No threshold, training duration, coefficient or eligibility gate was tuned using these results.

## Paired source-game uncertainty

| Group and control | NLL difference [95% interval] | Exact-agreement difference, points [95% interval] |
|---|---:|---:|
| `future_multiple_vs_ablation` | +0.000000 [+0.000000, +0.000000] | +0.000 [+0.000, +0.000] |
| `future_multiple_vs_auxiliary` | -0.000980 [-0.003809, +0.001757] | -0.853 [-1.972, +0.000] |
| `future_multiple_vs_base` | -0.047440 [-0.062247, -0.031150] | +1.919 [+0.645, +3.297] |
| `future_multiple_vs_ordinary1k` | -0.050658 [-0.065043, -0.034778] | +2.345 [+0.982, +3.822] |
| `shop_non_sale_vs_ablation` | +0.000117 [+0.000107, +0.000128] | +0.000 [+0.000, +0.000] |
| `shop_non_sale_vs_auxiliary` | -0.000226 [-0.001601, +0.001119] | +0.009 [-0.141, +0.163] |
| `shop_non_sale_vs_base` | -0.002199 [-0.004107, -0.000230] | +0.028 [-0.171, +0.232] |
| `shop_non_sale_vs_ordinary1k` | -0.005174 [-0.008074, -0.002774] | +0.312 [+0.146, +0.480] |
| `shop_purchase_vs_ablation` | +0.000145 [+0.000131, +0.000159] | +0.000 [+0.000, +0.000] |
| `shop_purchase_vs_auxiliary` | +0.002362 [+0.000555, +0.004352] | -0.207 [-0.432, +0.019] |
| `shop_purchase_vs_base` | +0.003668 [+0.001152, +0.006377] | -0.263 [-0.560, +0.038] |
| `shop_purchase_vs_ordinary1k` | -0.001233 [-0.006527, +0.002848] | -0.056 [-0.295, +0.182] |
| `shop_sale_vs_ablation` | -0.001806 [-0.001965, -0.001654] | +0.000 [+0.000, +0.000] |
| `shop_sale_vs_auxiliary` | +0.030300 [-0.006976, +0.068647] | +0.140 [-1.752, +2.016] |
| `shop_sale_vs_base` | +0.124862 [+0.071258, +0.176746] | -1.117 [-3.552, +1.288] |
| `shop_sale_vs_ordinary1k` | +0.042203 [+0.002933, +0.082888] | -1.536 [-3.735, +0.664] |

Differences are coupled minus control. Intervals use 10,000 paired resamples of all 128 source games, seed1729, retaining decision-weighted denominators. These are marginal descriptive 95% intervals, without multiple-comparison correction.

## Execution and verification

The single completed fit applied 1,000 updates and 320,000 presentations, matching ordinary1k row order and exposure. It verifies 316 optimizer states, leaves critic parameters unchanged, and strictly reloads both endpoint and operational recovery checkpoint. The first-update purchase-gradient error is 2.41260147e-10; the independently reconstructed new sale-bias gradient error is 1.74529551e-10.

Measurement uses 228,186 new encoder rows for 456,372 enabled/disabled records. It reuses 684,558 previously admitted control records. The audit checks every new record and reconstructs 78,135 control development rows; existing control training summaries come from the pinned prior audit.

An additional 52,730 native encoder rows reproduce both policies on every development state and the first 320 training states. The disabled policy is audited through a separate native call without the coupling hook. Input construction is independent, but the underlying action decoder is shared. All model hashes remain unchanged.

Float64 arithmetic reconstructs 11,714 future distributions and 64,769 active relation records, including the joint-action NLL adjustment. Outside the gate, modes and NLL are exactly unchanged by disabling the relation. Maximum errors: {"future_arithmetic": 2.4092259387487047e-07, "joint_nll_arithmetic": 1.5687742571657282e-06, "macro_arithmetic": 1.3438409780519578e-06, "native_logits": 0.0, "native_nll": 0.0, "pair_arithmetic": 1.3411448440692197e-07}.

Active times: fit 2208.16s; fit audit 6.37s; measurement 716.97s; scan audit 147.96s. This is one exploratory training run, not final independent confirmation.

The full goal remains unproven. Fresh sampled wins over the base and original ordinary control, the shorter matched control, greedy retention, positive learned early-decision attribution and final independent-fit verification are still required. No teacher-state screen or implementation check replaces those outcomes.

## Artifacts

- [result_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_fit_v1/result_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_fit_v1/verification_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_scan_v1/verification_v1.json)
- [replacement_relation_fit_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_fit_owner_terminal_v1.json)
- [replacement_relation_scan_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_scan_owner_terminal_v1.json)
- [replacement_relation_experiment_20260929.md](replacement_relation_experiment_20260929.md)
- [replacement_relation_scan_20260929.md](replacement_relation_scan_20260929.md)
- [replacement_relation_v1.py](../../eval/consequence_prediction_20260927/replacement_relation_v1.py)
- [replacement_relation_preparation_20260929.md](replacement_relation_preparation_20260929.md)
- [replacement_auxiliary_results_20260929.md](replacement_auxiliary_results_20260929.md)
