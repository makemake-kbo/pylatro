# Beneficial-replacement supervision at 1,000 updates

The fixed held-out spending screen fails. This endpoint does not qualify for fresh-game integration or an independent confirmation fit.

This run keeps the existing 57,473-parameter replacement readout and 715-parameter training-only readout. It changes the third target from continuous native gain to the indicator that an assessed replacement exceeds the fixed native gain threshold 0.03. The indicator uses training-only decision-balanced mean and variance with the same standardized squared loss. The first two forecasts, search-path classes, masks, loss weights, immutable base, exact initial tensors, original 1,000-game corpus, seed, row stream, optimizer and 1,000-update endpoint match the completed dense-regression experiment. Auxiliary predictions never enter the policy forward pass.

All development rows are from the existing exposed 128-game teacher-state panel. Agreement and likelihood do not measure autonomous win rate or expected action value. The ablation disables the existing sale/purchase policy correction within the same trained model. It measures that correction at inference, not the effect of auxiliary supervision during training. Removing auxiliary prediction computation itself leaves policy outputs unchanged; the earlier dense-regression training run is the supervision comparison.

## Training states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Dense regression | Readout disabled | Boundary-supervised policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 3494 | 0.236988 / 3198 | 0.233550 / 3203 | 0.176879 / 3252 | 0.172182 / 3260 | 0.172985 / 3251 | 0.172985 / 3251 |
| Future purchase, all eligible | 5183 | 0.159760 / 4887 | 0.157442 / 4892 | 0.119239 / 4941 | 0.116072 / 4949 | 0.116614 / 4940 | 0.116614 / 4940 |
| SHOP sale | 5471 | 1.610484 / 1963 | 1.513097 / 2103 | 1.526982 / 2063 | 1.523017 / 2069 | 1.591047 / 1959 | 1.510670 / 2086 |
| SHOP purchase | 41729 | 0.158364 / 39509 | 0.156281 / 39466 | 0.155313 / 39505 | 0.153529 / 39525 | 0.150388 / 39534 | 0.155213 / 39489 |
| Non-sale SHOP | 84884 | 0.147912 / 80821 | 0.144693 / 80823 | 0.141171 / 80948 | 0.140666 / 80958 | 0.137086 / 81019 | 0.141881 / 80941 |
| Early SHOP sale | 1242 | 1.279034 / 592 | 1.173554 / 629 | 1.192736 / 621 | 1.174174 / 624 | 1.208495 / 600 | 1.153604 / 633 |
| Later SHOP sale | 4229 | 1.707826 / 1371 | 1.612816 / 1474 | 1.625145 / 1442 | 1.625468 / 1445 | 1.703398 / 1359 | 1.615535 / 1453 |
| Early discard | 3705 | 2.846316 / 1355 | 2.849884 / 1317 | 2.844830 / 1331 | 2.846830 / 1337 | 2.846918 / 1330 | 2.846918 / 1330 |
| All decisions | 202141 | 0.496954 / 173497 | 0.467944 / 174709 | 0.466587 / 174859 | 0.466487 / 174803 | 0.467122 / 174766 | 0.466960 / 174815 |

## Development states

Each policy cell is mean NLL / exact matches. Lower NLL and more exact matches are favorable imitation metrics.

| Group | States | Base | Ordinary1k | Prior auxiliary | Dense regression | Readout disabled | Boundary-supervised policy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Future purchase, multiple offers | 469 | 0.217848 / 428 | 0.221066 / 426 | 0.171388 / 441 | 0.175638 / 434 | 0.169949 / 441 | 0.169949 / 441 |
| Future purchase, all eligible | 674 | 0.151589 / 633 | 0.153828 / 631 | 0.119260 / 646 | 0.122217 / 639 | 0.118258 / 646 | 0.118258 / 646 |
| SHOP sale | 716 | 2.311783 / 167 | 2.394442 / 170 | 2.406345 / 158 | 2.414839 / 165 | 2.479047 / 148 | 2.397772 / 158 |
| SHOP purchase | 5323 | 0.185486 / 4990 | 0.190388 / 4979 | 0.186792 / 4987 | 0.187599 / 4978 | 0.182921 / 4985 | 0.187880 / 4979 |
| Non-sale SHOP | 10880 | 0.176525 / 10250 | 0.179500 / 10219 | 0.174552 / 10252 | 0.174619 / 10256 | 0.170078 / 10261 | 0.175211 / 10243 |
| Early SHOP sale | 162 | 1.849605 / 55 | 2.025562 / 56 | 2.001099 / 49 | 2.028187 / 51 | 2.108716 / 49 | 2.046520 / 51 |
| Later SHOP sale | 554 | 2.446933 / 112 | 2.502310 / 114 | 2.524847 / 109 | 2.527903 / 114 | 2.587339 / 99 | 2.500485 / 107 |
| Early discard | 514 | 3.487149 / 168 | 3.677664 / 157 | 3.664179 / 160 | 3.676384 / 157 | 3.669766 / 157 | 3.669766 / 157 |
| All decisions | 26045 | 0.639106 / 21544 | 0.648459 / 21539 | 0.646685 / 21543 | 0.648310 / 21559 | 0.647756 / 21550 | 0.647666 / 21542 |

Future-purchase predictions are exactly identical for the coupled model and its disabled-readout ablation. Singleton supports are automatic choices; the primary prospective comparison uses 3,494 training and 469 development states with multiple offers. Early means current Ante<4, including Ante0 and re-entry. This descriptive split does not redefine the final learned early-policy attribution requirement.

## All fixed spending checks

| Comparison | Result |
|---|---|
| `future_nll_vs_base` | PASS |
| `future_nll_vs_ordinary1k` | PASS |
| `non_sale_nll_vs_ordinary1k` | PASS |
| `purchase_exact_vs_base` | FAIL |
| `purchase_exact_vs_ordinary1k` | PASS |
| `sale_exact_vs_ablation` | PASS |
| `sale_exact_vs_auxiliary` | FAIL |
| `sale_exact_vs_base` | FAIL |
| `sale_exact_vs_ordinary1k` | FAIL |
| `sale_nll_vs_ablation` | PASS |
| `sale_nll_vs_auxiliary` | PASS |
| `sale_nll_vs_base` | FAIL |
| `sale_nll_vs_ordinary1k` | FAIL |

These thirteen deterministic point comparisons were fixed before outcomes. They are a spending screen, not significance tests or final qualification. No threshold, training duration, coefficient or eligibility gate was tuned using these results.

## Paired source-game uncertainty

| Group and control | NLL difference [95% interval] | Exact-agreement difference, points [95% interval] |
|---|---:|---:|
| `future_multiple_vs_ablation` | +0.000000 [+0.000000, +0.000000] | +0.000 [+0.000, +0.000] |
| `future_multiple_vs_auxiliary` | -0.001439 [-0.006773, +0.004519] | +0.000 [-0.630, +0.615] |
| `future_multiple_vs_base` | -0.047899 [-0.066601, -0.027110] | +2.772 [+1.486, +4.184] |
| `future_multiple_vs_ordinary1k` | -0.051117 [-0.069444, -0.030841] | +3.198 [+1.818, +4.661] |
| `shop_non_sale_vs_ablation` | +0.005132 [+0.004712, +0.005555] | -0.165 [-0.244, -0.095] |
| `shop_non_sale_vs_auxiliary` | +0.000659 [-0.000777, +0.002085] | -0.083 [-0.240, +0.074] |
| `shop_non_sale_vs_base` | -0.001315 [-0.003391, +0.000912] | -0.064 [-0.282, +0.143] |
| `shop_non_sale_vs_ordinary1k` | -0.004289 [-0.007206, -0.001728] | +0.221 [+0.027, +0.415] |
| `shop_purchase_vs_ablation` | +0.004959 [+0.004506, +0.005419] | -0.113 [-0.207, -0.037] |
| `shop_purchase_vs_auxiliary` | +0.001088 [-0.000672, +0.002951] | -0.150 [-0.367, +0.057] |
| `shop_purchase_vs_base` | +0.002394 [-0.000065, +0.005072] | -0.207 [-0.492, +0.074] |
| `shop_purchase_vs_ordinary1k` | -0.002508 [-0.007685, +0.001570] | +0.000 [-0.287, +0.292] |
| `shop_sale_vs_ablation` | -0.081275 [-0.087754, -0.075109] | +1.397 [+0.652, +2.338] |
| `shop_sale_vs_auxiliary` | -0.008573 [-0.043212, +0.028510] | +0.000 [-1.868, +1.887] |
| `shop_sale_vs_base` | +0.085989 [+0.036113, +0.134704] | -1.257 [-3.453, +1.054] |
| `shop_sale_vs_ordinary1k` | +0.003330 [-0.036615, +0.044041] | -1.676 [-3.753, +0.295] |
| `future_multiple_vs_matched` | -0.005689 [-0.011775, -0.000209] | +1.493 [+0.464, +2.585] |
| `shop_non_sale_vs_matched` | +0.000592 [-0.000420, +0.001611] | -0.119 [-0.265, +0.018] |
| `shop_purchase_vs_matched` | +0.000281 [-0.001262, +0.001812] | +0.019 [-0.170, +0.209] |
| `shop_sale_vs_matched` | -0.017067 [-0.051117, +0.016625] | -0.978 [-3.090, +1.114] |

Differences are boundary-supervised policy minus control; the prior dense-regression endpoint is the matched supervision comparison. Intervals use 10,000 paired resamples of all 128 source games, seed1729, retaining decision-weighted denominators. These are marginal descriptive 95% intervals, without multiple-comparison correction.

## Auxiliary target prediction

Numeric errors give each decision with that target equal weight, averaging its observed pairs first. Unassessed pairs remain missing. Constants use only training decision-balanced means and the training class prior. Each numeric task retains its own denominator; the first two targets are native teacher forecasts and the third is a beneficial-replacement indicator. None is an expected win value. The indicator raw MSE is unclipped Brier-form error; forecast values may fall outside [0,1]. Known sampling/UID sensitivity is not an estimated irreducible-error floor.

Loss differences are candidate minus training constant; negative is favorable. Accuracy differences use the same sign; positive is favorable. Development intervals resample all 128 complete games, including games without annotated states. Zero-denominator bootstrap draws are reported and excluded from that metric interval; missing metrics remain absent.

### Training consequence targets

57,257 annotated decisions from 1000 complete games; 48 games have no annotations. Observed pair counts in task order log_score_ratio, forecast_value_difference, beneficial_replacement, stage: [415788, 415788, 330282, 445329].

| Metric | Labelled decisions | Candidate | Training constant | Difference [95% interval] | Zero-denominator draws |
|---|---:|---:|---:|---:|---:|
| beneficial_replacement/normalized_mae | 50055 | 0.416722 | 0.403598 | 0.0131242 | 0 |
| beneficial_replacement/normalized_mse | 50055 | 0.772709 | 1 | -0.227291 | 0 |
| beneficial_replacement/raw_mae | 50055 | 0.084094 | 0.0814456 | 0.00264846 | 0 |
| beneficial_replacement/raw_mse | 50055 | 0.0314669 | 0.0407228 | -0.00925593 | 0 |
| forecast_value_difference/normalized_mae | 53447 | 0.650697 | 0.75957 | -0.108873 | 0 |
| forecast_value_difference/normalized_mse | 53447 | 0.738636 | 1 | -0.261364 | 0 |
| forecast_value_difference/raw_mae | 53447 | 0.262785 | 0.306754 | -0.0439687 | 0 |
| forecast_value_difference/raw_mse | 53447 | 0.120469 | 0.163096 | -0.0426276 | 0 |
| log_score_ratio/normalized_mae | 53447 | 0.656293 | 0.754446 | -0.0981538 | 0 |
| log_score_ratio/normalized_mse | 53447 | 0.768522 | 1 | -0.231478 | 0 |
| log_score_ratio/raw_mae | 53447 | 0.276781 | 0.318176 | -0.0413948 | 0 |
| log_score_ratio/raw_mse | 53447 | 0.136689 | 0.177859 | -0.0411705 | 0 |
| stage/accuracy | 57257 | 0.794685 | 0.737633 | 0.0570517 | 0 |
| stage/cross_entropy | 57257 | 0.613817 | 0.884576 | -0.270759 | 0 |

Four-task auxiliary loss per original corpus row: candidate 0.117196; training constant 0.159694. This includes zero loss for unannotated decisions and divides by 202,141 original decisions.

### Development consequence targets

7,512 annotated decisions from 128 complete games; 5 games have no annotations. Observed pair counts in task order log_score_ratio, forecast_value_difference, beneficial_replacement, stage: [55380, 55380, 44606, 59236].

| Metric | Labelled decisions | Candidate | Training constant | Difference [95% interval] | Zero-denominator draws |
|---|---:|---:|---:|---:|---:|
| beneficial_replacement/normalized_mae | 6617 | 0.421824 | 0.401509 | 0.0203146 [0.0111672, 0.02959] | 0 |
| beneficial_replacement/normalized_mse | 6617 | 0.781029 | 0.989651 | -0.208621 [-0.249819, -0.171123] | 0 |
| beneficial_replacement/raw_mae | 6617 | 0.0851236 | 0.0810241 | 0.00409946 [0.00225352, 0.00597122] | 0 |
| beneficial_replacement/raw_mse | 6617 | 0.0318057 | 0.0403013 | -0.00849565 [-0.0101733, -0.00696859] | 0 |
| forecast_value_difference/normalized_mae | 7081 | 0.643012 | 0.756598 | -0.113587 [-0.126329, -0.101178] | 0 |
| forecast_value_difference/normalized_mse | 7081 | 0.73042 | 1.00402 | -0.273597 [-0.31766, -0.233126] | 0 |
| forecast_value_difference/raw_mae | 7081 | 0.259682 | 0.305554 | -0.0458722 [-0.0510183, -0.0408609] | 0 |
| forecast_value_difference/raw_mse | 7081 | 0.119129 | 0.163752 | -0.0446227 [-0.0518093, -0.0380219] | 0 |
| log_score_ratio/normalized_mae | 7081 | 0.647539 | 0.750867 | -0.103328 [-0.11598, -0.091119] | 0 |
| log_score_ratio/normalized_mse | 7081 | 0.74297 | 0.988598 | -0.245629 [-0.285627, -0.209241] | 0 |
| log_score_ratio/raw_mae | 7081 | 0.273089 | 0.316666 | -0.0435771 [-0.0489127, -0.038428] | 0 |
| log_score_ratio/raw_mse | 7081 | 0.132144 | 0.175832 | -0.0436874 [-0.0508016, -0.0372155] | 0 |
| stage/accuracy | 7512 | 0.796495 | 0.746395 | 0.0501006 [0.0439373, 0.0566238] | 0 |
| stage/cross_entropy | 7512 | 0.627708 | 0.874345 | -0.246637 [-0.26465, -0.229181] | 0 |

Four-task auxiliary loss per original corpus row: candidate 0.120137; training constant 0.162192. This includes zero loss for unannotated decisions and divides by 26,045 original decisions.


## Fixed-boundary recognition and observed-pair ranking

These diagnostics use only pairs with a recorded native gain. This oracle observation mask is unavailable as a policy rule, and unassessed alternatives may be better. The candidate uses its unclipped indicator forecast strictly above 0.5; the saved regression control uses its raw gain forecast strictly above 0.03. Neither threshold was selected from these results. Ranking uses the same observed alternatives, uniform treatment of exact predicted ties, and native-gain regret.

Intervals below are candidate minus regression, using 10,000 paired resamples of all 128 complete development games. Each metric keeps its own denominator. Recognition recall conditions on a true observed gain above 0.03; false-positive rate conditions on no such observed gain. These are descriptive metrics on the existing exposed panel.

### Training decisions

50,055 labelled decisions and 330,282 observed pairs; 7,202 annotated decisions have no observed gain and are excluded. All 1000 source games remain in the grouping, including 50 without gain labels.

| Rule | True positive | False negative | False positive | True negative |
|---|---:|---:|---:|---:|
| Indicator > 0.5 | 0 | 5382 | 0 | 44673 |
| Regression gain > 0.03 | 621 | 4761 | 29 | 44644 |

| Metric | Decisions | Candidate | Regression | Difference [95% interval] |
|---|---:|---:|---:|---:|
| Beneficial opportunity recall | 5382 | 0 | 0.115385 | -0.115385 |
| False-positive rate | 44673 | 0 | 0.000649162 | -0.000649162 |
| Decision classification accuracy | 50055 | 0.892478 | 0.904305 | -0.011827 |
| Best observed pair hit | 48224 | 0.270239 | 0.28575 | -0.0155109 |
| Pair concordance | 48224 | 0.598989 | 0.609461 | -0.0104719 |
| Native-gain regret | 48224 | 0.18658 | 0.175324 | +0.0112561 |

Indicator forecasts on observed pairs range from -0.139355 to 0.474936; 27,308 decisions have at least one forecast outside [0,1]. The equal-decision out-of-range pair fraction is 0.396325. Errors and decision rules use these original forecasts without clipping or calibration.

### Development decisions

6,617 labelled decisions and 44,606 observed pairs; 895 annotated decisions have no observed gain and are excluded. All 128 source games remain in the grouping, including 5 without gain labels.

| Rule | True positive | False negative | False positive | True negative |
|---|---:|---:|---:|---:|
| Indicator > 0.5 | 0 | 700 | 0 | 5917 |
| Regression gain > 0.03 | 96 | 604 | 5 | 5912 |

| Metric | Decisions | Candidate | Regression | Difference [95% interval] |
|---|---:|---:|---:|---:|
| Beneficial opportunity recall | 700 | 0 | 0.137143 | -0.137143 [-0.166434, -0.109038] |
| False-positive rate | 5917 | 0 | 0.000845023 | -0.000845023 [-0.00182669, -0.000162179] |
| Decision classification accuracy | 6617 | 0.894212 | 0.907964 | -0.0137525 [-0.0173022, -0.0105053] |
| Best observed pair hit | 6341 | 0.274247 | 0.291437 | -0.0171897 [-0.0327544, -0.00158416] |
| Pair concordance | 6341 | 0.603641 | 0.61921 | -0.015569 [-0.0219811, -0.00935691] |
| Native-gain regret | 6341 | 0.182356 | 0.162022 | +0.0203343 [+0.0112239, +0.030287] |

Indicator forecasts on observed pairs range from -0.123362 to 0.460514; 3,698 decisions have at least one forecast outside [0,1]. The equal-decision out-of-range pair fraction is 0.415753. Errors and decision rules use these original forecasts without clipping or calibration.


## Execution and verification

The single completed fit applied 1,000 updates and 320,000 presentations, matching ordinary1k row order and exposure. It verifies 318 optimizer states, leaves critic parameters unchanged, and strictly reloads both endpoint and operational recovery checkpoint. The first-update purchase-gradient error is 2.41260147e-10; the independently reconstructed new sale-bias gradient error is 1.74529551e-10.

Measurement uses 228,186 new encoder rows for 456,372 enabled/disabled records. It reuses 912,744 previously admitted control records. The audit checks every new record and reconstructs 104,180 control development rows; existing control training summaries come from the pinned prior audit.

An additional 52,730 native encoder rows reproduce both policies on every development state and the first 320 training states. The disabled policy is audited through a separate native call without the coupling hook. Input construction is independent, but the underlying action decoder is shared. All model hashes remain unchanged.

Float64 arithmetic reconstructs 11,714 future distributions and 64,769 active relation records, including the joint-action NLL adjustment. Outside the gate, modes and NLL are exactly unchanged by disabling the relation. Maximum errors: {"dense_native": 0.0, "future_arithmetic": 4.4341400151637345e-07, "joint_nll_arithmetic": 1.434086659202194e-06, "macro_arithmetic": 1.3545115855606582e-06, "native_logits": 0.0, "native_nll": 0.0, "pair_arithmetic": 1.3897870421164527e-07}.

Active times: fit 2145.38s; fit audit 8.01s; measurement 749.01s; scan audit 158.24s. This is one exploratory training run, not final independent confirmation.

Independent first-update dense gradient checks: {"bias_gradient": 2.5902409106492996e-09, "prediction_gradient": 4.656612877414201e-11, "weight_gradient": 9.617635951530978e-09}. Original losses at the first update match the previous coupled run within the frozen tolerance; original and coupled optimizer coordinates remain in order. The critic parameters are unchanged.

The full goal remains unproven. Fresh sampled win-rate gains must reach at least five percentage points over both the base and original ordinary4k control, with paired lower bounds above zero. Greedy retention must retain a paired 95% lower bound strictly above minus five points versus both. Learned early-decision attribution must have a paired lower bound strictly above zero, with positive direction in the independent fit. Final independent-fit/untouched-confirmation verification remains required. Ordinary1k is a shorter diagnostic control and cannot replace ordinary4k. No teacher-state or prediction metric replaces these outcomes.

## Artifacts

- [shop_boundary_decision_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_decision_owner_terminal_v1.json)
- [shop_boundary_decision_results_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_decision_results_v1.json)
- [shop_boundary_decision_verification_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_decision_verification_v1.json)
- [shop_dense_results_20261001.md](shop_dense_results_20261001.md)
- [shop_dense_decision_results_20261001.md](shop_dense_decision_results_20261001.md)
- [result_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_fit_v1/result_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_fit_v1/verification_v1.json)
- [verification_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_scan_v1/verification_v1.json)
- [shop_boundary_fit_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_fit_owner_terminal_v1.json)
- [shop_boundary_scan_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_scan_owner_terminal_v1.json)
- [shop_boundary_supervision_experiment_20261001.md](shop_boundary_supervision_experiment_20261001.md)
- [shop_boundary_auxiliary_v1.py](../../eval/consequence_prediction_20260927/shop_boundary_auxiliary_v1.py)
- [result_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_pack_v1/result_v1.json)
- [shop_boundary_pack_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_pack_owner_terminal_v1.json)
- [shop_dense_scan_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_dense_scan_owner_terminal_v1.json)
- [shop_uid_alias_results_20260930.md](shop_uid_alias_results_20260930.md)
- [replacement_relation_results_20260929.md](replacement_relation_results_20260929.md)
