# Opportunity scores survive observable support, but coverage and slot choice remain gaps

The opportunity signal is not explained by using teacher-filtered alternatives: on the same gain-labelled development states, replacing the oracle pair mask with all observable potential pairs leaves boundary ROC AUC essentially unchanged, 0.924806 to 0.924980. However, across the complete active SHOP population, the indicator model has lower teacher-sale average precision than the matched continuous-gain model. Conditional selection of the teacher's sold Joker remains weak in both models. Neither result qualifies the rejected policies for gameplay.

## Complete observable population

Support is reconstructed directly from legal sale actions and attended offered Joker tokens, within the existing SHOP/capacity-exhausted gate. It includes combinations that are unaffordable or rejected by native search. No teacher filter ID, final-comparison flag, gain, future action or hidden RNG value selects these scores. Teacher actions are evaluation labels only. The support is verified against every packed pair position.

| Split | Active states | Teacher sales | Potential pairs | Observed gain pairs | States without gains |
|---|---:|---:|---:|---:|---:|
| development | 7,512 | 703 | 59,236 | 44,606 | 895 |
| training | 57,257 | 5,380 | 445,329 | 330,282 | 7,202 |

The 703 development sales here are the active subset of the 716 sales used by the unchanged policy spending screen. These populations are not interchangeable.

## Predicting whether the teacher sells

Each score is the maximum original third forecast over all observable potential pairs. This is a ranking score, not a calibrated sale probability or an action policy.

| Split | Model | ROC AUC [95% interval] | Average precision [95% interval] |
|---|---|---:|---:|
| development | candidate | 0.892407 [0.877698, 0.905338] | 0.468507 [0.435293, 0.505279] |
| development | candidate_minus_regression | 0.007041 [0.000816, 0.013409] | -0.054491 [-0.075832, -0.031034] |
| development | regression | 0.885366 [0.869294, 0.899504] | 0.522998 [0.485005, 0.560688] |
| training | candidate | 0.892567 | 0.455351 |
| training | candidate_minus_regression | 0.006905 | -0.084036 |
| training | regression | 0.885662 | 0.539386 |

The indicator retains a slightly higher overall AUC, but its precision among higher-ranked sale opportunities is worse. The paired development AP difference is -5.45 points, with interval [-7.58,-3.10]; this is also worse in training. Better binary-opportunity discrimination on the labelled subset therefore did not imply a better sale classifier over all shops.

## Removing the oracle mask at fixed labels and population

Here the label remains any observed gain>0.03, and the population remains the exact previously studied gain-labelled decisions. This comparison isolates support; it does not treat missing gains as zero.

| Split | Model | Support | ROC AUC [95% interval] | Average precision [95% interval] |
|---|---|---|---:|---:|
| development | candidate | observable | 0.924980 [0.911352, 0.936645] | 0.686446 [0.650431, 0.719820] |
| development | candidate | observable_minus_oracle | 0.000174 [-0.000463, 0.000916] | -0.000534 [-0.002275, 0.001118] |
| development | candidate | oracle | 0.924806 [0.911136, 0.936555] | 0.686980 [0.650921, 0.720341] |
| development | regression | observable | 0.905894 [0.890639, 0.919043] | 0.611087 [0.569350, 0.649183] |
| development | regression | observable_minus_oracle | -0.000698 [-0.001683, 0.000319] | -0.004988 [-0.009994, -0.000952] |
| development | regression | oracle | 0.906592 [0.891466, 0.919609] | 0.616075 [0.575357, 0.653814] |
| training | candidate | observable | 0.928749 | 0.716100 |
| training | candidate | observable_minus_oracle | -0.000313 | -0.001377 |
| training | candidate | oracle | 0.929061 | 0.717477 |
| training | regression | observable | 0.907541 | 0.645348 |
| training | regression | observable_minus_oracle | -0.000974 | -0.005224 |
| training | regression | oracle | 0.908516 | 0.650573 |

## Choosing the owned Joker, conditional on a teacher sale

For each legal owned slot, take the maximum forecast over visible offered Jokers. Choose uniformly among tied maximizing owned slots. This gives conditional teacher-slot agreement; it does not decide whether to sell or which offer to purchase afterward.

| Split | Metric | Decisions | Mean [95% interval] |
|---|---|---:|---:|
| development | candidate/slot_hit_minus_uniform | 703 | 0.107112 [0.073824, 0.141351] |
| development | candidate/teacher_slot_hit | 703 | 0.305832 [0.272601, 0.339921] |
| development | candidate/top_pair_unobserved | 7,512 | 0.194356 [0.170289, 0.220137] |
| development | candidate_minus_regression/teacher_slot_hit | 703 | -0.009957 [-0.040059, 0.020319] |
| development | candidate_minus_regression/top_pair_unobserved | 7,512 | 0.004260 [-0.002057, 0.010760] |
| development | regression/slot_hit_minus_uniform | 703 | 0.117070 [0.084264, 0.150414] |
| development | regression/teacher_slot_hit | 703 | 0.315789 [0.282934, 0.349138] |
| development | regression/top_pair_unobserved | 7,512 | 0.190096 [0.165965, 0.216057] |
| development | uniform_teacher_slot_hit | 703 | 0.198720 [0.197634, 0.199621] |
| training | candidate/slot_hit_minus_uniform | 5,380 | 0.105688 |
| training | candidate/teacher_slot_hit | 5,380 | 0.305019 |
| training | candidate/top_pair_unobserved | 57,257 | 0.198596 |
| training | candidate_minus_regression/teacher_slot_hit | 5,380 | 0.005948 |
| training | candidate_minus_regression/top_pair_unobserved | 57,257 | 0.004401 |
| training | regression/slot_hit_minus_uniform | 5,380 | 0.099740 |
| training | regression/teacher_slot_hit | 5,380 | 0.299071 |
| training | regression/top_pair_unobserved | 57,257 | 0.194195 |
| training | uniform_teacher_slot_hit | 5,380 | 0.199331 |

Boundary slot accuracy is 30.58% versus 31.58% for regression and 19.87% for uniform legal-slot choice. Boundary versus regression uncertainty includes zero. Both have useful but insufficient slot information; neither justifies replacing the policy's existing sale choice with these raw forecasts. About 19% of top-scoring potential pairs have no recorded gain, including all states without any final comparison. Their gains remain unknown.

## Descriptive coverage accounting

This follow-up accounting was performed after the predeclared analysis. It uses the admitted state scores and recorded teacher labels, with fixed quantiles and no selected operating threshold. All 895 development states without final gains, and all 7,202 such training states, are teacher non-sales. The boundary score median is 0.0980 on development non-sales without gains versus 0.0210 on non-sales with gains; the corresponding upper quartiles are 0.2829 and 0.0674. The same pattern occurs in training. These no-gain states include native score/safety rejections, preempted search, affordability and other recorded branches.

Crucially, this is a coverage gap in the third auxiliary target. The original action-imitation loss already includes these negative decisions, and the eight-class stage auxiliary already labels their search path. This finding does not establish that the entire training corpus lacks negative labels, nor uniquely explain the original pretraining transfer failure. It suggests that a forecast trained conditional on surviving native search should not be interpreted directly as a complete decision signal.

## Consequence for the next experiment

Test one explicit supervision change before changing pooling: replace the conditional beneficial-comparison indicator with the fully observed event “this potential pair reached a final native comparison and exceeded gain 0.03.” Its target is false when native search never reached that comparison; this labels an observed search event and does not invent a counterfactual gain. The third task can then cover every observable potential pair, with training-only normalization and the same model, loss, exposure and optimizer. This would isolate whether complete event supervision helps the policy use the opportunity information. The existing stage task is a reason the intervention may add little, not proof it will succeed. Keep within-shop choice and the original policy screen as required outcomes. A separate plan must freeze the precise recipe before another fit.

The architectural alternative remains a readout that connects consequence information to the action decision. Current policy corrections average a separate learned pair score under current purchase preferences; the auxiliary forecasts never directly enter the policy. The current measurements do not prove that changing averaging to maximum, or directly consuming the raw indicator, improves behavior.

## Verification and limits

Independent code reconstructs all 64,769 active states and 504,565 potential pairs directly from archived inputs, all forecast maxima, teacher labels and tie-aware slot choices. It separately integrates ROC/precision-recall curves and reconstructs 10,000 complete-game bootstrap intervals. Maximum ranking discrepancy is 3.34e-16 and conditional-metric discrepancy 5.56e-17. The original oracle results reproduce exactly. Both analysis and auditor returned exit zero. No model forward pass, training or simulator call occurred.

Intervals are descriptive on the exposed 128-game development panel, with all complete games retained, seed1729. No bootstrap draw lacked a class or metric denominator. Training has no intervals. Neither model advanced through the thirteen policy checks, and no learned win-rate or long-horizon improvement is claimed. The original fresh sampled +5-point gains over base and ordinary4k, paired positive lower bounds, greedy retention, learned early-attribution and final independent-fit/untouched-confirmation requirements remain unchanged.

Plan: [shop_observable_support_plan_20261001.md](shop_observable_support_plan_20261001.md). Result: [shop_observable_support_results_v1.json](../../eval/consequence_prediction_20260927/shop_observable_support_results_v1.json). Independent audit: [shop_observable_support_verification_v1.json](../../eval/consequence_prediction_20260927/shop_observable_support_verification_v1.json). Admission: [shop_observable_support_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_observable_support_owner_terminal_v1.json). Descriptive coverage: [shop_observable_support_coverage_accounting_v1.json](../../eval/consequence_prediction_20260927/shop_observable_support_coverage_accounting_v1.json).

Prior opportunity diagnosis: [shop_boundary_discrimination_results_20261001.md](shop_boundary_discrimination_results_20261001.md). Full policy result: [shop_boundary_results_20261001.md](shop_boundary_results_20261001.md).
