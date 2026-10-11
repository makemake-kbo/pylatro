# Relative consequence supervision

The fixed policy screen fails. This endpoint does not qualify for fresh gameplay or an independent confirmation fit.

One matched 1,000-update exploratory fit changes only the third auxiliary task: its gain prediction-minus-target error is centered within the observed comparison pairs before squared error. This removes the common state-error component and preserves pairwise difference errors. The original dense targets, missingness masks, normalizers, architecture, initial tensors, original 1,000 games, action/future-purchase losses, task weights and optimizer recipe are unchanged. The completed absolute-gain DENSE endpoint is the matched control and was not refitted.

Centering occurs only in supervised loss and offline metrics. The policy forward receives observations only and keeps its original offer averaging. Auxiliary predictions remain training-only outputs. The gain output has an unconstrained common offset: absolute gain errors, event probabilities and absolute-gain thresholds are not used to qualify this objective. Development is the existing exposed 128-game teacher-state panel.

## Training policy agreement

Cells show mean NLL / exact matches.

| Group | States | Base | Ordinary1k | Prior auxiliary | Absolute-gain control | Correction disabled | Relative candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| SHOP sale | 5471 | 1.610484 / 1963 | 1.513097 / 2103 | 1.526982 / 2063 | 1.523017 / 2069 | 1.546248 / 2021 | 1.500264 / 2100 |
| SHOP purchase | 41729 | 0.158364 / 39509 | 0.156281 / 39466 | 0.155313 / 39505 | 0.153529 / 39525 | 0.153747 / 39506 | 0.156662 / 39478 |
| Non-sale SHOP | 84884 | 0.147912 / 80821 | 0.144693 / 80823 | 0.141171 / 80948 | 0.140666 / 80958 | 0.139602 / 80979 | 0.142469 / 80925 |
| Early SHOP sale | 1242 | 1.279034 / 592 | 1.173554 / 629 | 1.192736 / 621 | 1.174174 / 624 | 1.160301 / 627 | 1.128312 / 647 |
| Future purchase, multiple offers | 3494 | 0.236988 / 3198 | 0.233550 / 3203 | 0.176879 / 3252 | 0.172182 / 3260 | 0.173776 / 3253 | 0.173776 / 3253 |
| All decisions | 202141 | 0.496954 / 173497 | 0.467944 / 174709 | 0.466587 / 174859 | 0.466487 / 174803 | 0.466775 / 174800 | 0.466734 / 174825 |

## Development policy agreement

Cells show mean NLL / exact matches.

| Group | States | Base | Ordinary1k | Prior auxiliary | Absolute-gain control | Correction disabled | Relative candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| SHOP sale | 716 | 2.311783 / 167 | 2.394442 / 170 | 2.406345 / 158 | 2.414839 / 165 | 2.439178 / 155 | 2.392756 / 163 |
| SHOP purchase | 5323 | 0.185486 / 4990 | 0.190388 / 4979 | 0.186792 / 4987 | 0.187599 / 4978 | 0.187269 / 4972 | 0.190286 / 4969 |
| Non-sale SHOP | 10880 | 0.176525 / 10250 | 0.179500 / 10219 | 0.174552 / 10252 | 0.174619 / 10256 | 0.173245 / 10245 | 0.176329 / 10240 |
| Early SHOP sale | 162 | 1.849605 / 55 | 2.025562 / 56 | 2.001099 / 49 | 2.028187 / 51 | 2.040707 / 46 | 2.004441 / 46 |
| Future purchase, multiple offers | 469 | 0.217848 / 428 | 0.221066 / 426 | 0.171388 / 441 | 0.175638 / 434 | 0.170690 / 439 | 0.170690 / 439 |
| All decisions | 26045 | 0.639106 / 21544 | 0.648459 / 21539 | 0.646685 / 21543 | 0.648310 / 21559 | 0.648133 / 21538 | 0.648145 / 21541 |

Early is descriptive current Ante<4, not a permanent first-passage early-policy intervention. Correction-disabled is a within-model inference ablation, not a control for the training effect of the auxiliary loss.

## Original policy checks

| Check | Result |
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
| `sale_nll_vs_auxiliary` | PASS |
| `sale_nll_vs_base` | FAIL |
| `sale_nll_vs_ordinary1k` | PASS |

These unchanged point comparisons control further spending; they are not significance tests or final goal qualification.

## Paired development policy differences

| Comparison | NLL difference [95% interval] | Exact-rate difference [95% interval] |
|---|---:|---:|
| `future_multiple_vs_ablation` | 0 [0, 0] | 0 [0, 0] |
| `future_multiple_vs_auxiliary` | -0.000698428 [-0.00616937, 0.00483533] | -0.00426439 [-0.0105714, 0] |
| `future_multiple_vs_base` | -0.0471583 [-0.0649322, -0.0278135] | 0.0234542 [0.0111851, 0.037037] |
| `future_multiple_vs_ordinary1k` | -0.0503769 [-0.0673746, -0.0315466] | 0.0277186 [0.0145221, 0.0419591] |
| `shop_non_sale_vs_ablation` | 0.00308373 [0.00283249, 0.00333895] | -0.000459559 [-0.000898321, -9.24811e-05] |
| `shop_non_sale_vs_auxiliary` | 0.00177716 [0.000225731, 0.00332196] | -0.00110294 [-0.00269341, 0.000475334] |
| `shop_non_sale_vs_base` | -0.000196215 [-0.00242558, 0.0021873] | -0.000919118 [-0.00307026, 0.00112224] |
| `shop_non_sale_vs_ordinary1k` | -0.00317095 [-0.00608087, -0.000555701] | 0.00193015 [-8.99794e-05, 0.00392794] |
| `shop_purchase_vs_ablation` | 0.00301707 [0.00273981, 0.00329931] | -0.000563592 [-0.00130525, 0] |
| `shop_purchase_vs_auxiliary` | 0.00349417 [0.00148294, 0.00581145] | -0.00338155 [-0.00554736, -0.00130303] |
| `shop_purchase_vs_base` | 0.00479993 [0.001839, 0.00806989] | -0.00394514 [-0.00709922, -0.000788334] |
| `shop_purchase_vs_ordinary1k` | -0.000101365 [-0.00522743, 0.00402745] | -0.00187864 [-0.0047502, 0.000971114] |
| `shop_sale_vs_ablation` | -0.0464216 [-0.0501173, -0.0428958] | 0.0111732 [0.00425532, 0.0190619] |
| `shop_sale_vs_auxiliary` | -0.0135893 [-0.0515722, 0.0253204] | 0.00698324 [-0.0124658, 0.0266858] |
| `shop_sale_vs_base` | 0.080973 [0.0288877, 0.13221] | -0.00558659 [-0.0293717, 0.018814] |
| `shop_sale_vs_ordinary1k` | -0.00168628 [-0.0413465, 0.0386447] | -0.00977654 [-0.0304711, 0.0106954] |
| `future_multiple_vs_matched` | -0.00494825 [-0.00948288, -0.000834] | 0.010661 [0.00214133, 0.0206612] |
| `shop_non_sale_vs_matched` | 0.00171041 [0.000590781, 0.00281399] | -0.00147059 [-0.00295834, 0] |
| `shop_purchase_vs_matched` | 0.00268732 [0.000965949, 0.00437652] | -0.00169078 [-0.00384408, 0.000381689] |
| `shop_sale_vs_matched` | -0.0220832 [-0.0599667, 0.0160234] | -0.0027933 [-0.0217715, 0.0158526] |

All differences are candidate minus comparator. Intervals resample all 128 complete development games 10,000 times with seed1729, preserving decision denominators. They are descriptive marginal intervals without multiplicity correction.

## Training consequence prediction

57,257 active decisions in 1000 complete games. Both models use the identical original dense pack and normalization. Each metric retains its own labelled-decision denominator.

| Metric | Decisions | Relative candidate | Absolute-gain control | Training constant | Difference [95% interval] |
|---|---:|---:|---:|---:|---:|
| log_score_ratio/normalized_mse | 53447 | 0.764417 | 0.755589 | 1 | 0.00882707 |
| log_score_ratio/normalized_mae | 53447 | 0.653423 | 0.649581 | 0.754446 | 0.00384238 |
| log_score_ratio/raw_mse | 53447 | 0.135959 | 0.134389 | 0.177859 | 0.00156998 |
| log_score_ratio/raw_mae | 53447 | 0.275571 | 0.27395 | 0.318176 | 0.00162046 |
| forecast_value_difference/normalized_mse | 53447 | 0.732118 | 0.721622 | 1 | 0.0104961 |
| forecast_value_difference/normalized_mae | 53447 | 0.646489 | 0.641728 | 0.75957 | 0.00476123 |
| forecast_value_difference/raw_mse | 53447 | 0.119406 | 0.117694 | 0.163096 | 0.00171188 |
| forecast_value_difference/raw_mae | 53447 | 0.261086 | 0.259163 | 0.306754 | 0.00192283 |
| net_gain/centered_normalized_mse | 50055 | 0.357408 | 0.370333 | 0.419211 | -0.0129251 |
| net_gain/centered_normalized_mae | 50055 | 0.443225 | 0.451436 | 0.480977 | -0.00821125 |
| net_gain/centered_raw_mse | 50055 | 0.0427276 | 0.0442727 | 0.0501159 | -0.00154518 |
| net_gain/centered_raw_mae | 50055 | 0.153248 | 0.156088 | 0.166301 | -0.0028391 |
| stage/cross_entropy | 57257 | 0.613442 | 0.617962 | 0.884576 | -0.00452029 |
| stage/accuracy | 57257 | 0.794247 | 0.79454 | 0.737633 | -0.000292929 |

Gain errors are centered within the actually observed pairs for both models. The gain reference has zero pairwise differences; the first two references retain their original training means. No missing gain is assigned zero. Lower error/CE and higher stage accuracy are favorable.

| Choice metric | Decisions | Relative candidate | Absolute-gain control | Difference [95% interval] |
|---|---:|---:|---:|---:|
| observable_teacher_slot_hit | 5380 | 0.299442 | 0.299071 | 0.000371747 |
| observed_teacher_slot_hit | 5380 | 0.342751 | 0.341822 | 0.000929368 |
| uniform_teacher_slot_hit | 5380 | 0.199331 | 0.199331 | 0 |
| top_pair_unobserved | 57257 | 0.197303 | 0.194195 | 0.00310879 |
| observed_pair/concordance | 48224 | 0.619162 | 0.609461 | 0.00970098 |
| observed_pair/best_hit | 48224 | 0.295869 | 0.28575 | 0.0101194 |
| observed_pair/uniform_best_hit | 48224 | 0.188616 | 0.188616 | 0 |
| observed_pair/regret | 48224 | 0.168274 | 0.175324 | -0.00704928 |
| observed_pair/uniform_regret | 48224 | 0.27045 | 0.27045 | 0 |

Observable slot agreement uses only legal owned slots and visible offered Jokers; it is conditional on teacher SALE and does not decide whether to sell. Observed slot/pair metrics use the oracle comparison mask and cannot be treated as executable policy results. Owned ties are uniform over owned slots, pair ties over pairs. Pair ranking/regret excludes singleton and constant-gain states. Lower regret is favorable.

## Development consequence prediction

7,512 active decisions in 128 complete games. Both models use the identical original dense pack and normalization. Each metric retains its own labelled-decision denominator.

| Metric | Decisions | Relative candidate | Absolute-gain control | Training constant | Difference [95% interval] |
|---|---:|---:|---:|---:|---:|
| log_score_ratio/normalized_mse | 7081 | 0.736491 | 0.730748 | 0.988598 | 0.00574311 [0.00157889, 0.0104224] |
| log_score_ratio/normalized_mae | 7081 | 0.643712 | 0.640446 | 0.750867 | 0.00326601 [0.0012492, 0.00530351] |
| log_score_ratio/raw_mse | 7081 | 0.130992 | 0.12997 | 0.175832 | 0.00102147 [0.000280821, 0.00185372] |
| log_score_ratio/raw_mae | 7081 | 0.271475 | 0.270098 | 0.316666 | 0.00137739 [0.000526831, 0.00223667] |
| forecast_value_difference/normalized_mse | 7081 | 0.721055 | 0.714658 | 1.00402 | 0.00639693 [0.0016168, 0.0120434] |
| forecast_value_difference/normalized_mae | 7081 | 0.637777 | 0.634167 | 0.756598 | 0.00360939 [0.00119886, 0.00608238] |
| forecast_value_difference/raw_mse | 7081 | 0.117601 | 0.116558 | 0.163752 | 0.00104332 [0.000263694, 0.00196423] |
| forecast_value_difference/raw_mae | 7081 | 0.257567 | 0.25611 | 0.305554 | 0.00145766 [0.000484162, 0.00245638] |
| net_gain/centered_normalized_mse | 6617 | 0.368137 | 0.383093 | 0.441752 | -0.0149562 [-0.0181459, -0.0118783] |
| net_gain/centered_normalized_mae | 6617 | 0.441079 | 0.451565 | 0.486005 | -0.0104854 [-0.0129855, -0.00810729] |
| net_gain/centered_raw_mse | 6617 | 0.0440102 | 0.0457981 | 0.0528107 | -0.00178798 [-0.00216931, -0.00142003] |
| net_gain/centered_raw_mae | 6617 | 0.152506 | 0.156132 | 0.16804 | -0.0036254 [-0.00448985, -0.00280316] |
| stage/cross_entropy | 7512 | 0.627487 | 0.629905 | 0.874345 | -0.00241719 [-0.00431176, -0.000471059] |
| stage/accuracy | 7512 | 0.795996 | 0.7968 | 0.746395 | -0.000804639 [-0.00222505, 0.000506461] |

Gain errors are centered within the actually observed pairs for both models. The gain reference has zero pairwise differences; the first two references retain their original training means. No missing gain is assigned zero. Lower error/CE and higher stage accuracy are favorable.

| Choice metric | Decisions | Relative candidate | Absolute-gain control | Difference [95% interval] |
|---|---:|---:|---:|---:|
| observable_teacher_slot_hit | 703 | 0.297297 | 0.315789 | -0.0184922 [-0.0412375, 0.00428587] |
| observed_teacher_slot_hit | 703 | 0.347084 | 0.364154 | -0.0170697 [-0.0393349, 0.00455357] |
| uniform_teacher_slot_hit | 703 | 0.19872 | 0.19872 | 0 [0, 0] |
| top_pair_unobserved | 7512 | 0.190761 | 0.190096 | 0.000665602 [-0.00430548, 0.00599287] |
| observed_pair/concordance | 6341 | 0.627328 | 0.61921 | 0.00811866 [0.0035825, 0.0126401] |
| observed_pair/best_hit | 6341 | 0.297587 | 0.291437 | 0.00615045 [-0.00580465, 0.0184987] |
| observed_pair/uniform_best_hit | 6341 | 0.183244 | 0.183244 | 0 [0, 0] |
| observed_pair/regret | 6341 | 0.159699 | 0.162022 | -0.00232357 [-0.00788205, 0.00295738] |
| observed_pair/uniform_regret | 6341 | 0.274136 | 0.274136 | 0 [0, 0] |

Observable slot agreement uses only legal owned slots and visible offered Jokers; it is conditional on teacher SALE and does not decide whether to sell. Observed slot/pair metrics use the oracle comparison mask and cannot be treated as executable policy results. Owned ties are uniform over owned slots, pair ties over pairs. Pair ranking/regret excludes singleton and constant-gain states. Lower regret is favorable.

## Verification and original goal

The endpoint audit verifies all 1,000 updates and 320,000 presentations, matched initial state and row exposure, unchanged critic, strict checkpoint loading and independent first-update gradients. Original first-update loss differences: {"absolute_gain_control/auxiliary_nll_per_original_row": 0.0, "absolute_gain_control/bc_nll": 0.0, "auxiliary_nll_per_original_row": 0.0, "bc_nll": 0.0}. The auditor's inherited `boundary_control/` field prefix refers here to the absolute-gain checkpoint explicitly pinned as the supervision control.

Native scan reconstruction covers 52,730 encoder rows. Maximum errors: {"dense_native": 0.0, "future_arithmetic": 2.5241260814823363e-07, "joint_nll_arithmetic": 1.5974337337354427e-06, "macro_arithmetic": 1.4637486316360082e-06, "native_logits": 0.0, "native_nll": 0.0, "pair_arithmetic": 1.355830360205701e-07}. The separate consequence auditor recomputes raw-unit metrics, scalar ranks and complete-game intervals: {"development": {"active_decisions": 7512, "maximum_metric_error": 3.3306690738754696e-16, "metrics": 81}, "training": {"active_decisions": 57257, "maximum_metric_error": 2.220446049250313e-16, "metrics": 81}}.

No fresh games or final independent confirmation are implied by this report. Full success still requires fresh sampled win improvement of at least five percentage points against both base and original ordinary4k, paired lower bounds above zero; greedy lower bounds strictly above minus five points against both; positive learned early-policy attribution and independent-fit direction; and one final independent fit plus untouched confirmation. Ordinary1k is diagnostic and cannot replace ordinary4k. Teacher-state agreement and consequence errors do not establish win-rate or long-horizon improvement.

Plan: [shop_relative_supervision_experiment_20261001.md](shop_relative_supervision_experiment_20261001.md). Fit: [result_v1.json](../../eval/consequence_prediction_20260927/shop_relative_fit_v1/result_v1.json). Fit audit: [verification_v1.json](../../eval/consequence_prediction_20260927/shop_relative_fit_v1/verification_v1.json). Scan: [result_v1.json](../../eval/consequence_prediction_20260927/shop_relative_scan_v1/result_v1.json). Scan audit: [verification_v1.json](../../eval/consequence_prediction_20260927/shop_relative_scan_v1/verification_v1.json). Decisions: [shop_relative_decision_results_v1.json](../../eval/consequence_prediction_20260927/shop_relative_decision_results_v1.json). Decision audit: [shop_relative_decision_verification_v1.json](../../eval/consequence_prediction_20260927/shop_relative_decision_verification_v1.json).
