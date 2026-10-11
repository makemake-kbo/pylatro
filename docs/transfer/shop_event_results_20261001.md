# Complete beneficial-comparison event supervision

The fixed policy screen fails. This endpoint does not qualify for fresh gameplay or an independent confirmation fit.

This is one 1,000-update exploratory fit. The third auxiliary target changes from a beneficial indicator conditional on a native final comparison to the fully observed event that a potential pair reached a final comparison and exceeded gain 0.03. All observable potential pairs receive the event target; skipped or rejected comparisons receive false event labels, while their counterfactual gains remain unknown. First two targets, stage classes, original action/future-purchase losses, policy architecture and pooling, exact initialization, original 1,000 games, row stream and optimizer match the completed boundary fit. Auxiliary forecasts remain training-only outputs and never directly set policy action scores.

The boundary endpoint is the matched supervision control. Continuous-gain regression is a separate completed comparison; neither control is retrained. Development consists of the existing exposed 128-game teacher-state panel. Teacher agreement and prediction errors are not autonomous wins or action values.

## Training policy agreement

Cells show mean NLL / exact matches.

| Group | States | Base | Ordinary1k | Prior auxiliary | Gain regression | Boundary control | Correction disabled | Event candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SHOP sales | 5471 | 1.610484 / 1963 | 1.513097 / 2103 | 1.526982 / 2063 | 1.523017 / 2069 | 1.510670 / 2086 | 1.587446 / 1972 | 1.530784 / 2040 |
| SHOP purchases | 41729 | 0.158364 / 39509 | 0.156281 / 39466 | 0.155313 / 39505 | 0.153529 / 39525 | 0.155213 / 39489 | 0.150935 / 39529 | 0.154169 / 39505 |
| Non-sale SHOP | 84884 | 0.147912 / 80821 | 0.144693 / 80823 | 0.141171 / 80948 | 0.140666 / 80958 | 0.141881 / 80941 | 0.136727 / 81004 | 0.139908 / 80964 |
| Early SHOP sales | 1242 | 1.279034 / 592 | 1.173554 / 629 | 1.192736 / 621 | 1.174174 / 624 | 1.153604 / 633 | 1.198935 / 607 | 1.162356 / 618 |
| Future purchase, multiple offers | 3494 | 0.236988 / 3198 | 0.233550 / 3203 | 0.176879 / 3252 | 0.172182 / 3260 | 0.172985 / 3251 | 0.172753 / 3254 | 0.172753 / 3254 |
| All decisions | 202141 | 0.496954 / 173497 | 0.467944 / 174709 | 0.466587 / 174859 | 0.466487 / 174803 | 0.466960 / 174815 | 0.466553 / 174806 | 0.466355 / 174834 |

## Development policy agreement

Cells show mean NLL / exact matches.

| Group | States | Base | Ordinary1k | Prior auxiliary | Gain regression | Boundary control | Correction disabled | Event candidate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SHOP sales | 716 | 2.311783 / 167 | 2.394442 / 170 | 2.406345 / 158 | 2.414839 / 165 | 2.397772 / 158 | 2.489194 / 165 | 2.431962 / 167 |
| SHOP purchases | 5323 | 0.185486 / 4990 | 0.190388 / 4979 | 0.186792 / 4987 | 0.187599 / 4978 | 0.187880 / 4979 | 0.183743 / 4988 | 0.187061 / 4985 |
| Non-sale SHOP | 10880 | 0.176525 / 10250 | 0.179500 / 10219 | 0.174552 / 10252 | 0.174619 / 10256 | 0.175211 / 10243 | 0.170041 / 10266 | 0.173419 / 10256 |
| Early SHOP sales | 162 | 1.849605 / 55 | 2.025562 / 56 | 2.001099 / 49 | 2.028187 / 51 | 2.046520 / 51 | 2.017238 / 52 | 1.975197 / 54 |
| Future purchase, multiple offers | 469 | 0.217848 / 428 | 0.221066 / 426 | 0.171388 / 441 | 0.175638 / 434 | 0.169949 / 441 | 0.170015 / 437 | 0.170015 / 437 |
| All decisions | 26045 | 0.639106 / 21544 | 0.648459 / 21539 | 0.646685 / 21543 | 0.648310 / 21559 | 0.647666 / 21542 | 0.647983 / 21555 | 0.647821 / 21547 |

Early means the descriptive current-Ante<4 group, not the final permanent-first-passage early-policy intervention. The disabled correction is an inference ablation within the same trained model; it does not isolate the training effect of auxiliary supervision.

## Original thirteen policy checks

| Check | Result |
|---|---|
| `future_nll_vs_base` | PASS |
| `future_nll_vs_ordinary1k` | PASS |
| `non_sale_nll_vs_ordinary1k` | PASS |
| `purchase_exact_vs_base` | FAIL |
| `purchase_exact_vs_ordinary1k` | PASS |
| `sale_exact_vs_ablation` | PASS |
| `sale_exact_vs_auxiliary` | PASS |
| `sale_exact_vs_base` | FAIL |
| `sale_exact_vs_ordinary1k` | FAIL |
| `sale_nll_vs_ablation` | PASS |
| `sale_nll_vs_auxiliary` | FAIL |
| `sale_nll_vs_base` | FAIL |
| `sale_nll_vs_ordinary1k` | FAIL |

These predeclared point comparisons control further spending; they are not significance tests and do not replace final qualification.

## Paired development policy comparisons

| Comparison | NLL difference [95% interval] | Exact-rate difference [95% interval] |
|---|---:|---:|
| `future_multiple_vs_ablation` | 0 [0, 0] | 0 [0, 0] |
| `future_multiple_vs_auxiliary` | -0.00137269 [-0.00707512, 0.00479991] | -0.00852878 [-0.0169858, -0.0020284] |
| `future_multiple_vs_base` | -0.0478326 [-0.0663182, -0.0271866] | 0.0191898 [0.00647913, 0.0329909] |
| `future_multiple_vs_ordinary1k` | -0.0510511 [-0.0685932, -0.0312279] | 0.0234542 [0.00970874, 0.0381356] |
| `shop_non_sale_vs_ablation` | 0.00337793 [0.00310661, 0.00365285] | -0.000919118 [-0.00151719, -0.00037868] |
| `shop_non_sale_vs_auxiliary` | -0.00113331 [-0.00275187, 0.000477617] | 0.000367647 [-0.00130952, 0.00203315] |
| `shop_non_sale_vs_base` | -0.00310668 [-0.00539009, -0.000740998] | 0.000551471 [-0.0017346, 0.00284557] |
| `shop_non_sale_vs_ordinary1k` | -0.00608142 [-0.00910789, -0.00342574] | 0.00340074 [0.00133511, 0.00553568] |
| `shop_purchase_vs_ablation` | 0.00331814 [0.00301546, 0.00362452] | -0.000563592 [-0.00131063, 0] |
| `shop_purchase_vs_auxiliary` | 0.00026913 [-0.00203669, 0.00261267] | -0.000375728 [-0.00239425, 0.00153141] |
| `shop_purchase_vs_base` | 0.00157489 [-0.00151855, 0.0048966] | -0.00093932 [-0.00406276, 0.00207944] |
| `shop_purchase_vs_ordinary1k` | -0.00332641 [-0.00883033, 0.000914961] | 0.00112718 [-0.00155915, 0.00386034] |
| `shop_sale_vs_ablation` | -0.0572318 [-0.0622588, -0.0524472] | 0.0027933 [-0.00273607, 0.00842726] |
| `shop_sale_vs_auxiliary` | 0.0256169 [-0.0147789, 0.0686866] | 0.0125698 [-0.0056899, 0.0314286] |
| `shop_sale_vs_base` | 0.120179 [0.0698644, 0.169896] | 0 [-0.0250009, 0.0254246] |
| `shop_sale_vs_ordinary1k` | 0.0375199 [-0.00273911, 0.0791901] | -0.00418994 [-0.0261248, 0.0174033] |
| `future_multiple_vs_matched` | 6.60552e-05 [-0.00309678, 0.00311617] | -0.00852878 [-0.0170949, -0.00205761] |
| `shop_non_sale_vs_matched` | -0.00179196 [-0.00282099, -0.000738263] | 0.00119485 [-0.000265872, 0.00260248] |
| `shop_purchase_vs_matched` | -0.000818612 [-0.00244085, 0.000778024] | 0.00112718 [-0.00091946, 0.0030837] |
| `shop_sale_vs_matched` | 0.0341902 [0.00569891, 0.0627278] | 0.0125698 [-0.00688753, 0.0322129] |
| `future_multiple_vs_regression` | -0.00562252 [-0.0103535, -0.00122641] | 0.00639659 [0, 0.0145236] |
| `shop_non_sale_vs_regression` | -0.00120006 [-0.00214028, -0.000252508] | 0 [-0.00146269, 0.00145376] |
| `shop_purchase_vs_regression` | -0.000537718 [-0.00204902, 0.000867162] | 0.00131505 [-0.000750902, 0.00342922] |
| `shop_sale_vs_regression` | 0.017123 [-0.0136606, 0.0470606] | 0.0027933 [-0.0165295, 0.0225228] |

Differences are event candidate minus control, with 10,000 paired resamples of all 128 complete development games, seed1729, preserving decision-weighted denominators. Intervals are marginal descriptive 95% intervals without multiplicity correction.

## Training predictions on the same full-event targets

57,257 active decisions, 445,329 potential pairs, 1000 complete games. Numeric tasks retain their own labelled-decision denominators. Each labelled decision weights its observed pairs equally. The old boundary forecasts use their original inverse normalization, then common event-target units; the training constant uses only the event training statistics.

| Metric | Decisions | Event candidate | Boundary control | Training constant | Event minus boundary [95% interval] |
|---|---:|---:|---:|---:|---:|
| log_score_ratio/normalized_mse | 53447 | 0.764725 | 0.768522 | 1 | -0.00379746 |
| log_score_ratio/normalized_mae | 53447 | 0.653533 | 0.656293 | 0.754446 | -0.0027599 |
| log_score_ratio/raw_mse | 53447 | 0.136014 | 0.136689 | 0.177859 | -0.000675414 |
| log_score_ratio/raw_mae | 53447 | 0.275617 | 0.276781 | 0.318176 | -0.00116394 |
| forecast_value_difference/normalized_mse | 53447 | 0.734956 | 0.738636 | 1 | -0.00367986 |
| forecast_value_difference/normalized_mae | 53447 | 0.647993 | 0.650697 | 0.75957 | -0.00270395 |
| forecast_value_difference/raw_mse | 53447 | 0.119869 | 0.120469 | 0.163096 | -0.000600172 |
| forecast_value_difference/raw_mae | 53447 | 0.261693 | 0.262785 | 0.306754 | -0.001092 |
| reached_beneficial_comparison/normalized_mse | 57257 | 0.811467 | 0.970616 | 1 | -0.15915 |
| reached_beneficial_comparison/normalized_mae | 57257 | 0.362498 | 0.517588 | 0.326006 | -0.15509 |
| reached_beneficial_comparison/raw_mse | 57257 | 0.0215607 | 0.0257893 | 0.02657 | -0.00422862 |
| reached_beneficial_comparison/raw_mae | 57257 | 0.0590882 | 0.0843685 | 0.0531401 | -0.0252802 |
| stage/cross_entropy | 57257 | 0.621995 | 0.613817 | 0.884576 | 0.00817867 |
| stage/accuracy | 57257 | 0.792174 | 0.794685 | 0.737633 | -0.00251129 |

Lower error/cross-entropy and higher stage accuracy are favorable. Raw event MSE is Brier-form error on unmodified, potentially unbounded forecasts; none of these targets is an expected win value.

### Fixed event threshold >0.5

| Model | TP | FN | FP | TN | Forecast range |
|---|---:|---:|---:|---:|---:|
| boundary | 0 | 5382 | 0 | 51875 | [-0.139355, 0.474936] |
| candidate | 0 | 5382 | 0 | 51875 | [-0.0951898, 0.367741] |

The event is whether any observable pair reached a beneficial native comparison. This is distinct from teacher SALE and from a policy decision. No threshold fitting, clipping or calibration occurred.

### Teacher sale discrimination on observable support

| Model | ROC AUC [95% interval] | Average precision [95% interval] |
|---|---:|---:|
| boundary | 0.892567 | 0.455351 |
| candidate | 0.930537 | 0.684507 |
| candidate_minus_boundary | 0.0379696 | 0.229156 |

Teacher sales: 5,380. The maximum forecast across observable potential pairs is a ranking score, not a calibrated state probability. Support uses legal sale slots and visible offered Jokers; it does not assume every joint plan is affordable.

| Decision metric | Decisions | Event candidate | Boundary control | Difference [95% interval] |
|---|---:|---:|---:|---:|
| event_recall | 5382 | 0 | 0 | 0 |
| event_false_positive_rate | 51875 | 0 | 0 | 0 |
| event_accuracy | 57257 | 0.906003 | 0.906003 | 0 |
| teacher_slot_hit | 5380 | 0.326952 | 0.305019 | 0.0219331 |
| uniform_teacher_slot_hit | 5380 | 0.199331 | 0.199331 | 0 |
| teacher_slot_hit_minus_uniform | 5380 | 0.127621 | 0.105688 | 0.0219331 |
| out_of_range_fraction | 57257 | 0.409052 | 0.36825 | 0.0408019 |

Slot agreement is conditional on teacher SALE, pooling the maximum over offers separately for each owned slot and choosing uniformly among tied maximizing owned slots. It does not decide to sell. Decision metrics keep their own denominators and resample complete games.

## Development predictions on the same full-event targets

7,512 active decisions, 59,236 potential pairs, 128 complete games. Numeric tasks retain their own labelled-decision denominators. Each labelled decision weights its observed pairs equally. The old boundary forecasts use their original inverse normalization, then common event-target units; the training constant uses only the event training statistics.

| Metric | Decisions | Event candidate | Boundary control | Training constant | Event minus boundary [95% interval] |
|---|---:|---:|---:|---:|---:|
| log_score_ratio/normalized_mse | 7081 | 0.739599 | 0.74297 | 0.988598 | -0.00337001 [-0.0047344, -0.00211205] |
| log_score_ratio/normalized_mae | 7081 | 0.644996 | 0.647539 | 0.750867 | -0.0025426 [-0.00328777, -0.00183332] |
| log_score_ratio/raw_mse | 7081 | 0.131545 | 0.132144 | 0.175832 | -0.000599388 [-0.000842057, -0.000375647] |
| log_score_ratio/raw_mae | 7081 | 0.272017 | 0.273089 | 0.316666 | -0.0010723 [-0.00138657, -0.000773173] |
| forecast_value_difference/normalized_mse | 7081 | 0.727107 | 0.73042 | 1.00402 | -0.00331295 [-0.00485354, -0.0019146] |
| forecast_value_difference/normalized_mae | 7081 | 0.640351 | 0.643012 | 0.756598 | -0.00266097 [-0.00346333, -0.00189754] |
| forecast_value_difference/raw_mse | 7081 | 0.118589 | 0.119129 | 0.163752 | -0.000540331 [-0.000791595, -0.000312265] |
| forecast_value_difference/raw_mae | 7081 | 0.258607 | 0.259682 | 0.305554 | -0.00107464 [-0.00139867, -0.000766327] |
| reached_beneficial_comparison/normalized_mse | 7512 | 0.793458 | 0.946966 | 0.972558 | -0.153508 [-0.170771, -0.136016] |
| reached_beneficial_comparison/normalized_mae | 7512 | 0.360602 | 0.512672 | 0.321533 | -0.15207 [-0.160898, -0.143231] |
| reached_beneficial_comparison/raw_mse | 7512 | 0.0210822 | 0.0251609 | 0.0258409 | -0.00407872 [-0.00453739, -0.00361396] |
| reached_beneficial_comparison/raw_mae | 7512 | 0.0587792 | 0.0835672 | 0.0524109 | -0.0247879 [-0.0262268, -0.0233472] |
| stage/cross_entropy | 7512 | 0.635219 | 0.627708 | 0.874345 | 0.00751082 [0.00619178, 0.00884535] |
| stage/accuracy | 7512 | 0.794775 | 0.796495 | 0.746395 | -0.00172021 [-0.00300549, -0.00052259] |

Lower error/cross-entropy and higher stage accuracy are favorable. Raw event MSE is Brier-form error on unmodified, potentially unbounded forecasts; none of these targets is an expected win value.

### Fixed event threshold >0.5

| Model | TP | FN | FP | TN | Forecast range |
|---|---:|---:|---:|---:|---:|
| boundary | 0 | 700 | 0 | 6812 | [-0.123362, 0.47867] |
| candidate | 0 | 700 | 0 | 6812 | [-0.0887655, 0.351945] |

The event is whether any observable pair reached a beneficial native comparison. This is distinct from teacher SALE and from a policy decision. No threshold fitting, clipping or calibration occurred.

### Teacher sale discrimination on observable support

| Model | ROC AUC [95% interval] | Average precision [95% interval] |
|---|---:|---:|
| boundary | 0.892407 [0.877698, 0.905338] | 0.468507 [0.435293, 0.505279] |
| candidate | 0.92439 [0.910859, 0.936084] | 0.658365 [0.622108, 0.694686] |
| candidate_minus_boundary | 0.0319824 [0.0268674, 0.0370804] | 0.189858 [0.16579, 0.213068] |

Teacher sales: 703. The maximum forecast across observable potential pairs is a ranking score, not a calibrated state probability. Support uses legal sale slots and visible offered Jokers; it does not assume every joint plan is affordable.

| Decision metric | Decisions | Event candidate | Boundary control | Difference [95% interval] |
|---|---:|---:|---:|---:|
| event_recall | 700 | 0 | 0 | 0 [0, 0] |
| event_false_positive_rate | 6812 | 0 | 0 | 0 [0, 0] |
| event_accuracy | 7512 | 0.906816 | 0.906816 | 0 [0, 0] |
| teacher_slot_hit | 703 | 0.315789 | 0.305832 | 0.00995733 [-0.0218314, 0.0423373] |
| uniform_teacher_slot_hit | 703 | 0.19872 | 0.19872 | 0 [0, 0] |
| teacher_slot_hit_minus_uniform | 703 | 0.11707 | 0.107112 | 0.00995733 [-0.0218314, 0.0423373] |
| out_of_range_fraction | 7512 | 0.411442 | 0.386289 | 0.0251523 [0.0105625, 0.0403876] |

Slot agreement is conditional on teacher SALE, pooling the maximum over offers separately for each owned slot and choosing uniformly among tied maximizing owned slots. It does not decide to sell. Decision metrics keep their own denominators and resample complete games.

## Verification and full goal

The fit audit reconstructs all 1,000 updates, 320,000 presentations and 318 optimizer states, verifies unchanged critic parameters, strict endpoint/recovery loading and matched row exposure. Original first-update action and future-purchase loss differences: {"auxiliary_nll_per_original_row": 0.0, "bc_nll": 0.0, "boundary_control/auxiliary_nll_per_original_row": 0.0, "boundary_control/bc_nll": 0.0}.

The scan uses 228,186 encoder rows for 456,372 new enabled/disabled policy records, reusing 1,140,930 control records. Native reconstruction covers 52,730 encoder rows. Maximum scan errors: {"dense_native": 0.0, "future_arithmetic": 2.947635753258737e-07, "joint_nll_arithmetic": 1.045619636208528e-06, "macro_arithmetic": 1.2513886602505409e-06, "native_logits": 0.0, "native_nll": 0.0, "pair_arithmetic": 1.3642342910902272e-07}.

The separate forecast auditor reconstructs raw errors under each original normalization, all decisions, ties, complete-game metrics and ranking intervals: {"development": {"active_decisions": 7512, "maximum_metric_error": 3.3306690738754696e-16, "maximum_ranking_error": 1.6653345369377348e-16, "metrics": 72, "pairs": 59236}, "training": {"active_decisions": 57257, "maximum_metric_error": 2.220446049250313e-16, "maximum_ranking_error": 1.1102230246251565e-16, "metrics": 72, "pairs": 445329}}.

Active times: fit 2248.47s, fit audit 8.14s, scan 757.21s, scan audit 155.05s. This is one exploratory fit; no final independent confirmation has occurred.

The full goal remains unproven. Fresh sampled win gains must reach at least five percentage points over both base and original ordinary4k, with paired lower bounds above zero. Greedy paired95 lower bounds must be strictly above minus five points versus both. Learned early-decision attribution needs a positive paired lower bound and positive direction in the independent fit. One final independent fit plus untouched confirmation remains required. Ordinary1k cannot replace ordinary4k; no teacher-state, event-prediction or calibration metric replaces these outcomes.

Plan: [shop_event_supervision_experiment_20261001.md](shop_event_supervision_experiment_20261001.md). Fit: [result_v1.json](../../eval/consequence_prediction_20260927/shop_event_fit_v1/result_v1.json). Fit audit: [verification_v1.json](../../eval/consequence_prediction_20260927/shop_event_fit_v1/verification_v1.json). Scan: [result_v1.json](../../eval/consequence_prediction_20260927/shop_event_scan_v1/result_v1.json). Scan audit: [verification_v1.json](../../eval/consequence_prediction_20260927/shop_event_scan_v1/verification_v1.json). Decisions: [shop_event_decision_results_v1.json](../../eval/consequence_prediction_20260927/shop_event_decision_results_v1.json). Decision audit: [shop_event_decision_verification_v1.json](../../eval/consequence_prediction_20260927/shop_event_decision_verification_v1.json).
