# Conditional-only replacement scorer at 1,000 updates

The fixed held-out spending screen fails. This endpoint does not qualify for fresh-game integration or an independent confirmation fit.

The change uses the same 57,473-parameter relation between contextual owned-Joker and offered-Joker features, weighted by purchase preference. It corrects only conditional sale logits and leaves macro logits unchanged. The coupled endpoint is a reporting control; the base, corpus, row stream, optimizer, future-purchase labels, auxiliary coefficient and endpoint remain fixed. This changes both forward factorization and gradient routing. Existing macro parameters still learn full action NLL; teacher non-sales provide no direct gradient to the new scorer.

All development rows are from the existing exposed 128-game teacher-state panel. Agreement and likelihood do not measure autonomous win rate or expected action value. The ablation disables the added readout within the same trained model; it measures that readout's contribution at inference, not a separate capacity-matched training intervention.

## Training states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Coupled endpoint | Readout disabled | Conditional scorer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 3494 | 0.236988 / 3198 | 0.233550 / 3203 | 0.176879 / 3252 | 0.175041 / 3251 | 0.175166 / 3254 | 0.175166 / 3254 |
| Future purchase, all eligible | 5183 | 0.159760 / 4887 | 0.157442 / 4892 | 0.119239 / 4941 | 0.118000 / 4940 | 0.118084 / 4943 | 0.118084 / 4943 |
| SHOP sale | 5471 | 1.610484 / 1963 | 1.513097 / 2103 | 1.526982 / 2063 | 1.503500 / 2085 | 1.512002 / 2068 | 1.512015 / 2069 |
| SHOP purchase | 41729 | 0.158364 / 39509 | 0.156281 / 39466 | 0.155313 / 39505 | 0.154692 / 39498 | 0.152075 / 39567 | 0.152075 / 39567 |
| Non-sale SHOP | 84884 | 0.147912 / 80821 | 0.144693 / 80823 | 0.141171 / 80948 | 0.140186 / 80959 | 0.139229 / 81026 | 0.139229 / 81026 |
| Early SHOP sale | 1242 | 1.279034 / 592 | 1.173554 / 629 | 1.192736 / 621 | 1.168930 / 621 | 1.156999 / 627 | 1.157015 / 628 |
| Later SHOP sale | 4229 | 1.707826 / 1371 | 1.612816 / 1474 | 1.625145 / 1442 | 1.601758 / 1464 | 1.616261 / 1441 | 1.616274 / 1441 |
| Early discard | 3705 | 2.846316 / 1355 | 2.849884 / 1317 | 2.844830 / 1331 | 2.843886 / 1332 | 2.857831 / 1322 | 2.857831 / 1322 |
| All decisions | 202141 | 0.496954 / 173497 | 0.467944 / 174709 | 0.466587 / 174859 | 0.465560 / 174854 | 0.465374 / 174958 | 0.465374 / 174959 |

## Development states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Coupled endpoint | Readout disabled | Conditional scorer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 469 | 0.217848 / 428 | 0.221066 / 426 | 0.171388 / 441 | 0.170408 / 437 | 0.169783 / 438 | 0.169783 / 438 |
| Future purchase, all eligible | 674 | 0.151589 / 633 | 0.153828 / 631 | 0.119260 / 646 | 0.118578 / 642 | 0.118143 / 643 | 0.118143 / 643 |
| SHOP sale | 716 | 2.311783 / 167 | 2.394442 / 170 | 2.406345 / 158 | 2.436645 / 159 | 2.433811 / 165 | 2.433829 / 165 |
| SHOP purchase | 5323 | 0.185486 / 4990 | 0.190388 / 4979 | 0.186792 / 4987 | 0.189154 / 4976 | 0.185550 / 4983 | 0.185550 / 4983 |
| Non-sale SHOP | 10880 | 0.176525 / 10250 | 0.179500 / 10219 | 0.174552 / 10252 | 0.174326 / 10253 | 0.173204 / 10253 | 0.173204 / 10253 |
| Early SHOP sale | 162 | 1.849605 / 55 | 2.025562 / 56 | 2.001099 / 49 | 2.043745 / 50 | 2.035805 / 53 | 2.035848 / 53 |
| Later SHOP sale | 554 | 2.446933 / 112 | 2.502310 / 114 | 2.524847 / 109 | 2.551537 / 109 | 2.550196 / 112 | 2.550207 / 112 |
| Early discard | 514 | 3.487149 / 168 | 3.677664 / 157 | 3.664179 / 160 | 3.671002 / 160 | 3.683573 / 158 | 3.683573 / 158 |
| All decisions | 26045 | 0.639106 / 21544 | 0.648459 / 21539 | 0.646685 / 21543 | 0.648141 / 21547 | 0.646646 / 21559 | 0.646646 / 21559 |

Future-purchase predictions, macro probabilities and non-sale teacher NLL are exactly identical for the conditional model and its disabled-readout ablation. Active non-sale flat modes can still change when a sale alternative becomes the joint winner. Singleton supports are automatic choices; the primary prospective comparison uses 3,494 training and 469 development states with multiple offers. Early means current Ante<4, including Ante0 and re-entry. This descriptive split does not redefine the final learned early-policy attribution requirement.

## All fixed spending checks

| Comparison | Result |
|---|---|
| `future_nll_vs_base` | PASS |
| `future_nll_vs_ordinary1k` | PASS |
| `non_sale_nll_vs_ordinary1k` | PASS |
| `purchase_exact_vs_base` | FAIL |
| `purchase_exact_vs_ordinary1k` | PASS |
| `sale_exact_vs_ablation` | FAIL |
| `sale_exact_vs_auxiliary` | PASS |
| `sale_exact_vs_base` | FAIL |
| `sale_exact_vs_ordinary1k` | FAIL |
| `sale_nll_vs_ablation` | FAIL |
| `sale_nll_vs_auxiliary` | FAIL |
| `sale_nll_vs_base` | FAIL |
| `sale_nll_vs_ordinary1k` | FAIL |

These thirteen deterministic point comparisons were fixed before outcomes. They are a spending screen, not significance tests or final qualification. No threshold, training duration, coefficient or eligibility gate was tuned using these results.

## Paired source-game uncertainty

| Group and control | NLL difference [95% interval] | Exact-agreement difference, points [95% interval] |
|---|---:|---:|
| `future_multiple_vs_ablation` | +0.000000 [+0.000000, +0.000000] | +0.000 [+0.000, +0.000] |
| `future_multiple_vs_auxiliary` | -0.001605 [-0.004804, +0.001714] | -0.640 [-1.681, +0.000] |
| `future_multiple_vs_base` | -0.048065 [-0.062821, -0.032035] | +2.132 [+0.828, +3.564] |
| `future_multiple_vs_coupled` | -0.000625 [-0.002957, +0.001795] | +0.213 [+0.000, +0.683] |
| `future_multiple_vs_ordinary1k` | -0.051283 [-0.065369, -0.035764] | +2.559 [+1.106, +4.049] |
| `shop_non_sale_vs_ablation` | +0.000000 [+0.000000, +0.000000] | +0.000 [+0.000, +0.000] |
| `shop_non_sale_vs_auxiliary` | -0.001348 [-0.002599, -0.000168] | +0.009 [-0.140, +0.166] |
| `shop_non_sale_vs_base` | -0.003321 [-0.005150, -0.001397] | +0.028 [-0.189, +0.243] |
| `shop_non_sale_vs_coupled` | -0.001122 [-0.001915, -0.000342] | +0.000 [-0.129, +0.127] |
| `shop_non_sale_vs_ordinary1k` | -0.006296 [-0.008965, -0.004071] | +0.312 [+0.157, +0.468] |
| `shop_purchase_vs_ablation` | +0.000000 [+0.000000, +0.000000] | +0.000 [+0.000, +0.000] |
| `shop_purchase_vs_auxiliary` | -0.001242 [-0.002806, +0.000249] | -0.075 [-0.274, +0.131] |
| `shop_purchase_vs_base` | +0.000064 [-0.002600, +0.002794] | -0.132 [-0.454, +0.185] |
| `shop_purchase_vs_coupled` | -0.003604 [-0.005047, -0.002255] | +0.132 [-0.038, +0.306] |
| `shop_purchase_vs_ordinary1k` | -0.004837 [-0.009832, -0.001106] | +0.075 [-0.153, +0.298] |
| `shop_sale_vs_ablation` | +0.000018 [-0.000010, +0.000048] | +0.000 [+0.000, +0.000] |
| `shop_sale_vs_auxiliary` | +0.027484 [-0.009421, +0.063594] | +0.978 [-0.810, +2.798] |
| `shop_sale_vs_base` | +0.122046 [+0.076398, +0.166236] | -0.279 [-2.603, +2.155] |
| `shop_sale_vs_coupled` | -0.002816 [-0.031324, +0.024973] | +0.838 [-0.726, +2.418] |
| `shop_sale_vs_ordinary1k` | +0.039387 [+0.004345, +0.073477] | -0.698 [-2.486, +1.128] |

Differences are conditional scorer minus control. Intervals use 10,000 paired resamples of all 128 source games, seed1729, retaining decision-weighted denominators. These are marginal descriptive 95% intervals, without multiple-comparison correction.

## Execution and verification

The original supervisor disappeared during a user-requested pause; its exit status is unknown. The original stopped worker survived with 440 updates and resumed in place under replacement supervision. Model, optimizer, RNG and row-stream state remained in memory. A tested Darwin kernel exit watch recorded the original worker exiting successfully; the unchanged accounting auditor and the replacement whole owner also exited successfully. The original pause interval is excluded from active-time budgets. This repair did not start another fit or change the training recipe.

The single completed fit applied 1,000 updates and 320,000 presentations, matching ordinary1k row order and exposure. It verifies 316 optimizer states, leaves critic parameters unchanged, and strictly reloads both endpoint and operational recovery checkpoint. The first-update purchase-gradient error is 2.41260147e-10; the independently reconstructed conditional output-weight gradient error is 5.83967193e-09.

Measurement uses 228,186 new encoder rows for 456,372 enabled/disabled records. It reuses 912,744 previously admitted control records. The audit checks every new record and reconstructs 104,180 control development rows; existing control training summaries come from the pinned prior audit.

An additional 52,730 native encoder rows reproduce both policies on every development state and the first 320 training states. The disabled policy is audited through a separate native call without the coupling hook. Input construction is independent, but the underlying action decoder is shared. All model hashes remain unchanged.

Float64 arithmetic reconstructs 11,714 future distributions and 64,769 active conditional-readout records, including sale-conditional normalization and the zero applied macro correction. Outside the gate, modes and NLL are exactly unchanged by disabling the relation. Maximum errors: {"future_arithmetic": 2.502027554029951e-07, "joint_nll_arithmetic": 1.2229904911009726e-06, "macro_arithmetic": 0.0, "native_logits": 0.0, "native_nll": 0.0, "pair_arithmetic": 1.3610714477696462e-07}.

Active times: fit 2295.74s; fit audit 7.02s; measurement 740.69s; scan audit 156.59s. This is one exploratory training run, not final independent confirmation.

The full goal remains unproven. Fresh sampled wins over the base and original ordinary control, the shorter matched control, greedy retention, positive learned early-decision attribution and final independent-fit verification are still required. No teacher-state screen or implementation check replaces those outcomes.

## Artifacts

- [result_v1.json](../../eval/consequence_prediction_20260927/replacement_conditional_fit_v1/result_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/replacement_conditional_fit_v1/verification_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/replacement_conditional_scan_v1/verification_v1.json)
- [replacement_conditional_fit_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_conditional_fit_owner_terminal_v1.json)
- [replacement_conditional_scan_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_conditional_scan_owner_terminal_v1.json)
- [replacement_conditional_experiment_20260929.md](replacement_conditional_experiment_20260929.md)
- [replacement_conditional_scan_20260929.md](replacement_conditional_scan_20260929.md)
- [replacement_conditional_v1.py](../../eval/consequence_prediction_20260927/replacement_conditional_v1.py)
- [replacement_conditional_preparation_20260929.md](replacement_conditional_preparation_20260929.md)
- [replacement_auxiliary_results_20260929.md](replacement_auxiliary_results_20260929.md)
- [replacement_relation_results_20260929.md](replacement_relation_results_20260929.md)
- [replacement_conditional_adoption_protocol_v1.json](../../eval/consequence_prediction_20260927/replacement_conditional_adoption_protocol_v1.json)
