# Dense shop supervision at 1,000 updates

The fixed held-out spending screen fails. This endpoint does not qualify for fresh-game integration or an independent confirmation fit.

This run adds a training-only 715-parameter readout to the existing 57,473-parameter coupled policy. It predicts native score ratios, forecast-value changes, final replacement gains and search-path classes from the existing pair features. Auxiliary predictions never enter the policy forward pass. The immutable base checkpoint, original 1,000-game corpus, seed, row stream, optimizer, ordinary action and future-purchase objectives, and 1,000-update endpoint match the earlier coupled experiment. The changed mechanism is additional supervised gradient into the shared features.

All development rows are from the existing exposed 128-game teacher-state panel. Agreement and likelihood do not measure autonomous win rate or expected action value. The ablation disables the existing sale/purchase policy correction within the same trained model. It measures that correction at inference, not the effect of auxiliary supervision during training. Removing auxiliary prediction computation itself leaves policy outputs unchanged; the earlier coupled training run is the supervision comparison.

## Training states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Prior coupled | Readout disabled | Dense-supervised policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 3494 | 0.236988 / 3198 | 0.233550 / 3203 | 0.176879 / 3252 | 0.175041 / 3251 | 0.172182 / 3260 | 0.172182 / 3260 |
| Future purchase, all eligible | 5183 | 0.159760 / 4887 | 0.157442 / 4892 | 0.119239 / 4941 | 0.118000 / 4940 | 0.116072 / 4949 | 0.116072 / 4949 |
| SHOP sale | 5471 | 1.610484 / 1963 | 1.513097 / 2103 | 1.526982 / 2063 | 1.503500 / 2085 | 1.573140 / 1997 | 1.523017 / 2069 |
| SHOP purchase | 41729 | 0.158364 / 39509 | 0.156281 / 39466 | 0.155313 / 39505 | 0.154692 / 39498 | 0.150575 / 39547 | 0.153529 / 39525 |
| Non-sale SHOP | 84884 | 0.147912 / 80821 | 0.144693 / 80823 | 0.141171 / 80948 | 0.140186 / 80959 | 0.137742 / 81001 | 0.140666 / 80958 |
| Early SHOP sale | 1242 | 1.279034 / 592 | 1.173554 / 629 | 1.192736 / 621 | 1.168930 / 621 | 1.209716 / 609 | 1.174174 / 624 |
| Later SHOP sale | 4229 | 1.707826 / 1371 | 1.612816 / 1474 | 1.625145 / 1442 | 1.601758 / 1464 | 1.679873 / 1388 | 1.625468 / 1445 |
| Early discard | 3705 | 2.846316 / 1355 | 2.849884 / 1317 | 2.844830 / 1331 | 2.843886 / 1332 | 2.846830 / 1337 | 2.846830 / 1337 |
| All decisions | 202141 | 0.496954 / 173497 | 0.467944 / 174709 | 0.466587 / 174859 | 0.465560 / 174854 | 0.466615 / 174774 | 0.466487 / 174803 |

## Development states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Prior coupled | Readout disabled | Dense-supervised policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 469 | 0.217848 / 428 | 0.221066 / 426 | 0.171388 / 441 | 0.170408 / 437 | 0.175638 / 434 | 0.175638 / 434 |
| Future purchase, all eligible | 674 | 0.151589 / 633 | 0.153828 / 631 | 0.119260 / 646 | 0.118578 / 642 | 0.122217 / 639 | 0.122217 / 639 |
| SHOP sale | 716 | 2.311783 / 167 | 2.394442 / 170 | 2.406345 / 158 | 2.436645 / 159 | 2.465024 / 160 | 2.414839 / 165 |
| SHOP purchase | 5323 | 0.185486 / 4990 | 0.190388 / 4979 | 0.186792 / 4987 | 0.189154 / 4976 | 0.184570 / 4981 | 0.187599 / 4978 |
| Non-sale SHOP | 10880 | 0.176525 / 10250 | 0.179500 / 10219 | 0.174552 / 10252 | 0.174326 / 10253 | 0.171491 / 10264 | 0.174619 / 10256 |
| Early SHOP sale | 162 | 1.849605 / 55 | 2.025562 / 56 | 2.001099 / 49 | 2.043745 / 50 | 2.068344 / 50 | 2.028187 / 51 |
| Later SHOP sale | 554 | 2.446933 / 112 | 2.502310 / 114 | 2.524847 / 109 | 2.551537 / 109 | 2.581020 / 110 | 2.527903 / 114 |
| Early discard | 514 | 3.487149 / 168 | 3.677664 / 157 | 3.664179 / 160 | 3.671002 / 160 | 3.676384 / 157 | 3.676384 / 157 |
| All decisions | 26045 | 0.639106 / 21544 | 0.648459 / 21539 | 0.646685 / 21543 | 0.648141 / 21547 | 0.648383 / 21562 | 0.648310 / 21559 |

Future-purchase predictions are exactly identical for the coupled model and its disabled-readout ablation. Singleton supports are automatic choices; the primary prospective comparison uses 3,494 training and 469 development states with multiple offers. Early means current Ante<4, including Ante0 and re-entry. This descriptive split does not redefine the final learned early-policy attribution requirement.

## All fixed spending checks

| Comparison | Result |
|---|---|
| `future_nll_vs_base` | PASS |
| `future_nll_vs_ordinary1k` | PASS |
| `non_sale_nll_vs_ordinary1k` | PASS |
| `purchase_exact_vs_base` | FAIL |
| `purchase_exact_vs_ordinary1k` | FAIL |
| `sale_exact_vs_ablation` | PASS |
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
| `future_multiple_vs_auxiliary` | +0.004250 [-0.002692, +0.012217] | -1.493 [-2.778, -0.420] |
| `future_multiple_vs_base` | -0.042210 [-0.061770, -0.020876] | +1.279 [-0.217, +2.826] |
| `future_multiple_vs_ordinary1k` | -0.045429 [-0.063588, -0.025639] | +1.706 [+0.218, +3.247] |
| `shop_non_sale_vs_ablation` | +0.003128 [+0.002867, +0.003392] | -0.074 [-0.133, -0.027] |
| `shop_non_sale_vs_auxiliary` | +0.000067 [-0.001569, +0.001683] | +0.037 [-0.128, +0.205] |
| `shop_non_sale_vs_base` | -0.001907 [-0.004185, +0.000463] | +0.055 [-0.184, +0.295] |
| `shop_non_sale_vs_ordinary1k` | -0.004881 [-0.007811, -0.002299] | +0.340 [+0.145, +0.533] |
| `shop_purchase_vs_ablation` | +0.003029 [+0.002750, +0.003315] | -0.056 [-0.130, +0.000] |
| `shop_purchase_vs_auxiliary` | +0.000807 [-0.001457, +0.003229] | -0.169 [-0.384, +0.039] |
| `shop_purchase_vs_base` | +0.002113 [-0.000813, +0.005287] | -0.225 [-0.546, +0.093] |
| `shop_purchase_vs_ordinary1k` | -0.002789 [-0.008102, +0.001491] | -0.019 [-0.278, +0.244] |
| `shop_sale_vs_ablation` | -0.050184 [-0.054714, -0.045829] | +0.698 [+0.141, +1.364] |
| `shop_sale_vs_auxiliary` | +0.008494 [-0.035549, +0.052661] | +0.978 [-1.045, +3.017] |
| `shop_sale_vs_base` | +0.103056 [+0.046835, +0.157440] | -0.279 [-2.743, +2.257] |
| `shop_sale_vs_ordinary1k` | +0.020397 [-0.016083, +0.057909] | -0.698 [-2.805, +1.411] |
| `future_multiple_vs_matched` | +0.005230 [-0.001714, +0.013176] | -0.640 [-1.639, +0.222] |
| `shop_non_sale_vs_matched` | +0.000293 [-0.001108, +0.001647] | +0.028 [-0.130, +0.188] |
| `shop_purchase_vs_matched` | -0.001555 [-0.003213, +0.000131] | +0.038 [-0.188, +0.266] |
| `shop_sale_vs_matched` | -0.021806 [-0.060417, +0.017338] | +0.838 [-1.100, +2.730] |

Differences are dense-supervised policy minus control; the prior coupled endpoint is the matched supervision comparison. Intervals use 10,000 paired resamples of all 128 source games, seed1729, retaining decision-weighted denominators. These are marginal descriptive 95% intervals, without multiple-comparison correction.

## Auxiliary target prediction

Numeric errors give each decision with that target equal weight, averaging its observed pairs first. Unassessed pairs remain missing. Constants use only training decision-balanced means and the training class prior. Each numeric task retains its own denominator; values are native teacher forecasts and gains, not expected win values. Known sampling/UID sensitivity is not an estimated irreducible-error floor.

Loss differences are candidate minus training constant; negative is favorable. Accuracy differences use the same sign; positive is favorable. Development intervals resample all 128 complete games, including games without annotated states. Zero-denominator bootstrap draws are reported and excluded from that metric interval; missing metrics remain absent.

### Training consequence targets

57,257 annotated decisions from 1000 complete games; 48 games have no annotations. Observed pair counts in task order log_score_ratio, forecast_value_difference, net_gain, stage: [415788, 415788, 330282, 445329].

| Metric | Labelled decisions | Candidate | Training constant | Difference [95% interval] | Zero-denominator draws |
|---|---:|---:|---:|---:|---:|
| forecast_value_difference/normalized_mae | 53447 | 0.641728 | 0.75957 | -0.117842 | 0 |
| forecast_value_difference/normalized_mse | 53447 | 0.721622 | 1 | -0.278378 | 0 |
| forecast_value_difference/raw_mae | 53447 | 0.259163 | 0.306754 | -0.0475909 | 0 |
| forecast_value_difference/raw_mse | 53447 | 0.117694 | 0.163096 | -0.0454024 | 0 |
| log_score_ratio/normalized_mae | 53447 | 0.649581 | 0.754446 | -0.104865 | 0 |
| log_score_ratio/normalized_mse | 53447 | 0.755589 | 1 | -0.244411 | 0 |
| log_score_ratio/raw_mae | 53447 | 0.27395 | 0.318176 | -0.0442253 | 0 |
| log_score_ratio/raw_mse | 53447 | 0.134389 | 0.177859 | -0.0434707 | 0 |
| net_gain/normalized_mae | 50055 | 0.625147 | 0.764832 | -0.139685 | 0 |
| net_gain/normalized_mse | 50055 | 0.659176 | 1 | -0.340824 | 0 |
| net_gain/raw_mae | 50055 | 0.216149 | 0.264446 | -0.0482972 | 0 |
| net_gain/raw_mse | 50055 | 0.0788034 | 0.119548 | -0.040745 | 0 |
| stage/accuracy | 57257 | 0.79454 | 0.737633 | 0.0569069 | 0 |
| stage/cross_entropy | 57257 | 0.617962 | 0.884576 | -0.266614 | 0 |

Four-task auxiliary loss per original corpus row: candidate 0.112986; training constant 0.159694. This includes zero loss for unannotated decisions and divides by 202,141 original decisions.

### Development consequence targets

7,512 annotated decisions from 128 complete games; 5 games have no annotations. Observed pair counts in task order log_score_ratio, forecast_value_difference, net_gain, stage: [55380, 55380, 44606, 59236].

| Metric | Labelled decisions | Candidate | Training constant | Difference [95% interval] | Zero-denominator draws |
|---|---:|---:|---:|---:|---:|
| forecast_value_difference/normalized_mae | 7081 | 0.634167 | 0.756598 | -0.122431 [-0.13696, -0.10855] | 0 |
| forecast_value_difference/normalized_mse | 7081 | 0.714658 | 1.00402 | -0.289359 [-0.340606, -0.243365] | 0 |
| forecast_value_difference/raw_mae | 7081 | 0.25611 | 0.305554 | -0.0494441 [-0.0553116, -0.043838] | 0 |
| forecast_value_difference/raw_mse | 7081 | 0.116558 | 0.163752 | -0.0471934 [-0.0555516, -0.039692] | 0 |
| log_score_ratio/normalized_mae | 7081 | 0.640446 | 0.750867 | -0.110421 [-0.124445, -0.0970217] | 0 |
| log_score_ratio/normalized_mse | 7081 | 0.730748 | 0.988598 | -0.25785 [-0.303725, -0.216928] | 0 |
| log_score_ratio/raw_mae | 7081 | 0.270098 | 0.316666 | -0.0465683 [-0.0524825, -0.0409173] | 0 |
| log_score_ratio/raw_mse | 7081 | 0.12997 | 0.175832 | -0.0458611 [-0.0540204, -0.0385827] | 0 |
| net_gain/normalized_mae | 6617 | 0.621372 | 0.763506 | -0.142134 [-0.161786, -0.123355] | 0 |
| net_gain/normalized_mse | 6617 | 0.673724 | 1.02918 | -0.355458 [-0.422397, -0.297699] | 0 |
| net_gain/raw_mae | 6617 | 0.214844 | 0.263988 | -0.049144 [-0.0559388, -0.0426511] | 0 |
| net_gain/raw_mse | 6617 | 0.0805427 | 0.123037 | -0.0424944 [-0.0504969, -0.0355894] | 0 |
| stage/accuracy | 7512 | 0.7968 | 0.746395 | 0.0504053 [0.0439898, 0.057226] | 0 |
| stage/cross_entropy | 7512 | 0.629905 | 0.874345 | -0.24444 [-0.262235, -0.227262] | 0 |

Four-task auxiliary loss per original corpus row: candidate 0.115937; training constant 0.163448. This includes zero loss for unannotated decisions and divides by 26,045 original decisions.


## Execution and verification

The single completed fit applied 1,000 updates and 320,000 presentations, matching ordinary1k row order and exposure. It verifies 318 optimizer states, leaves critic parameters unchanged, and strictly reloads both endpoint and operational recovery checkpoint. The first-update purchase-gradient error is 2.41260147e-10; the independently reconstructed new sale-bias gradient error is 1.74529551e-10.

Measurement uses 228,186 new encoder rows for 456,372 enabled/disabled records. It reuses 912,744 previously admitted control records. The audit checks every new record and reconstructs 104,180 control development rows; existing control training summaries come from the pinned prior audit.

An additional 52,730 native encoder rows reproduce both policies on every development state and the first 320 training states. The disabled policy is audited through a separate native call without the coupling hook. Input construction is independent, but the underlying action decoder is shared. All model hashes remain unchanged.

Float64 arithmetic reconstructs 11,714 future distributions and 64,769 active relation records, including the joint-action NLL adjustment. Outside the gate, modes and NLL are exactly unchanged by disabling the relation. Maximum errors: {"dense_native": 0.0, "future_arithmetic": 2.1022385521263232e-07, "joint_nll_arithmetic": 1.4258601694905337e-06, "macro_arithmetic": 1.322265882564011e-06, "native_logits": 0.0, "native_nll": 0.0, "pair_arithmetic": 1.2152576422952421e-07}.

Active times: fit 2249.51s; fit audit 7.87s; measurement 768.01s; scan audit 160.64s. This is one exploratory training run, not final independent confirmation.

Independent first-update dense gradient checks: {"bias_gradient": 2.5902409106492996e-09, "prediction_gradient": 4.656612877414201e-11, "weight_gradient": 9.617635951530978e-09}. Original losses at the first update match the previous coupled run within the frozen tolerance; original and coupled optimizer coordinates remain in order. The critic parameters are unchanged.

The full goal remains unproven. Fresh sampled win-rate gains must reach at least five percentage points over both the base and original ordinary4k control, with paired lower bounds above zero. Greedy retention must retain a paired 95% lower bound strictly above minus five points versus both. Learned early-decision attribution must have a paired lower bound strictly above zero, with positive direction in the independent fit. Final independent-fit/untouched-confirmation verification remains required. Ordinary1k is a shorter diagnostic control and cannot replace ordinary4k. No teacher-state or prediction metric replaces these outcomes.

## Artifacts

- [result_v1.json](../../eval/consequence_prediction_20260927/shop_dense_fit_v1/result_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/shop_dense_fit_v1/verification_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/shop_dense_scan_v1/verification_v1.json)
- [shop_dense_fit_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_dense_fit_owner_terminal_v1.json)
- [shop_dense_scan_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_dense_scan_owner_terminal_v1.json)
- [shop_dense_supervision_experiment_20261001.md](shop_dense_supervision_experiment_20261001.md)
- [shop_dense_auxiliary_v1.py](../../eval/consequence_prediction_20260927/shop_dense_auxiliary_v1.py)
- [result_v1.json](../../eval/consequence_prediction_20260927/shop_dense_pack_v1/result_v1.json)
- [shop_dense_pack_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_dense_pack_owner_terminal_v1.json)
- [replacement_relation_scan_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_scan_owner_terminal_v1.json)
- [shop_uid_alias_results_20260930.md](shop_uid_alias_results_20260930.md)
- [replacement_relation_results_20260929.md](replacement_relation_results_20260929.md)
