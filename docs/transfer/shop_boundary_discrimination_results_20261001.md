# Opportunity information survives; action selection still fails

The boundary-supervised model separates shops containing an observed beneficial replacement better than the matched continuous-gain model, despite recognizing none at its fixed 0.5 indicator threshold. The initial zero-recall result is not evidence that the representation contains no opportunity information. Scores are compressed, including on training data. This is a diagnostic refinement, not a qualified policy or win-rate improvement.

These scores pool only alternatives with a recorded native gain. That oracle mask is unavailable at inference. The label is whether any observed gain exceeds 0.03, not whether the teacher sells or the game is eventually won. A maximum pair forecast is a ranking score, not a state-event probability.

## Between-shop discrimination

| Split | Positive states / labelled states | Model | ROC AUC [95% interval] | Average precision [95% interval] |
|---|---:|---|---:|---:|
| development | 700 / 6,617 | candidate | 0.924806 [0.911136, 0.936555] | 0.686980 [0.650921, 0.720341] |
| development | 700 / 6,617 | regression | 0.906592 [0.891466, 0.919609] | 0.616075 [0.575357, 0.653814] |
| development | 700 / 6,617 | candidate_minus_regression | 0.018214 [0.012963, 0.023532] | 0.070904 [0.051456, 0.089373] |
| training | 5,382 / 50,055 | candidate | 0.929061 | 0.717477 |
| training | 5,382 / 50,055 | regression | 0.908516 | 0.650573 |
| training | 5,382 / 50,055 | candidate_minus_regression | 0.020546 | 0.066904 |

ROC AUC measures how often a randomly chosen positive shop scores above a negative shop, giving half credit to ties. Average precision summarizes precision over ranked positive retrieval. Positive development prevalence is 10.58%. Intervals use 10,000 paired resamples of all 128 complete games, seed 1729, including the five games without labelled decisions. No draw lacked a class. Training uses the original 1,000 games and has no intervals. These are descriptive comparisons on an exposed development panel, not independent confirmation.

## Fixed-bin pair calibration

Each decision receives weight one, divided among its observed pairs. Bins were fixed before outcomes; forecasts are neither clipped nor calibrated. Empty bins are omitted from this display but retained in the result artifact.

| Split | Original forecast interval | Pair count | Mean forecast | Observed positive fraction |
|---|---|---:|---:|---:|
| development | (-inf, 0.0] | 20,157 | -0.030282 | 0.000933 |
| development | (0.0, 0.1] | 16,648 | 0.037570 | 0.013526 |
| development | (0.1, 0.2] | 4,703 | 0.140676 | 0.099076 |
| development | (0.2, 0.3] | 2,046 | 0.239620 | 0.257622 |
| development | (0.3, 0.4] | 784 | 0.341334 | 0.552782 |
| development | (0.4, 0.5] | 268 | 0.424984 | 0.765714 |
| training | (-inf, 0.0] | 142,290 | -0.029356 | 0.000910 |
| training | (0.0, 0.1] | 129,853 | 0.037194 | 0.010761 |
| training | (0.1, 0.2] | 36,686 | 0.141889 | 0.097684 |
| training | (0.2, 0.3] | 14,585 | 0.241913 | 0.304052 |
| training | (0.3, 0.4] | 5,295 | 0.341620 | 0.599452 |
| training | (0.4, 0.5] | 1,573 | 0.423146 | 0.794760 |

The upper occupied development bin averages 0.425 although 76.6% of its decision-weighted pairs are beneficial; training shows the same pattern, 0.423 versus 79.5%. This indicates underconfidence in the high-scoring range. Lower ranges can overpredict instead, so a blanket additive bias is not established. This descriptive bin analysis does not fit a calibration rule.

## What improved, and what did not

The raw Brier-form improvement comes from positive pairs: their contribution falls from 0.03856850 under the training constant to 0.02530316. Negative-pair error rises from 0.00173284 to 0.00650254. All four contributions use the same denominator of 6,617 labelled decisions. Thus this improvement is not merely better prediction of the negative majority.

Better separation between shops does not imply better selection within a shop. The independently audited boundary experiment has worse development within-shop native-gain concordance (60.36% versus 61.92%) and best-pair selection (27.42% versus 29.14%). Its policy matches 158 of 716 teacher sales, versus 165 for the regression model and 167 for the base. It fails six of thirteen fixed policy checks. The auxiliary predictor is used during training only; its outputs do not directly set policy action scores. These observations narrow the problem to confidence, action ranking and transfer of learned information into choices, without establishing which intervention fixes it.

No threshold, calibrator, additional model, alternative pooling function, gameplay run or confirmation was fitted or selected. The rejected candidate remains stopped. No autonomous win-rate or long-horizon generalization improvement is demonstrated.

## Next discriminating check

Before choosing a new training objective or architecture, test whether the opportunity signal survives removal of the oracle comparison mask. The recorded support of legal owned-Joker/visible offered-Joker pairs is independently verified against observations. A separate frozen diagnostic can score that complete observable support, include states with no final native comparisons, and evaluate teacher sale labels without inventing gain labels for unassessed alternatives. This would distinguish information available to a deployable readout from information helped by diagnostic selection. No such result is claimed here.

## Verification and sources

The independent auditor reconstructs all 56,672 state scores from 374,888 observed pairs, all bins and both error partitions. Independent descending-threshold integration reproduces AUC/AP and complete-game bootstrap intervals; maximum point error is 1.12e-16 and interval error 3.27e-16. Arithmetic tests additionally compare literal weighted positive/negative pairs, tied-score average precision and 1,000 bootstrap draws including a game without states. Analysis and audit processes both returned exit zero. No model or simulator calls were made.

Plan: [shop_boundary_discrimination_plan_20261001.md](shop_boundary_discrimination_plan_20261001.md). Results: [shop_boundary_discrimination_results_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_discrimination_results_v1.json). Verification: [shop_boundary_discrimination_verification_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_discrimination_verification_v1.json). Terminal admission: [shop_boundary_discrimination_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_discrimination_owner_terminal_v1.json).

Full policy experiment: [shop_boundary_results_20261001.md](shop_boundary_results_20261001.md). Fixed recognition/ranking diagnostic: [shop_boundary_decision_results_v1.json](../../eval/consequence_prediction_20260927/shop_boundary_decision_results_v1.json).

The full goal remains fresh sampled win gains of at least five percentage points over both base and original ordinary4k, paired lower bounds above zero; greedy paired lower bounds strictly above minus five points against both; positive learned early-decision attribution with paired lower bound above zero; and one final independent fit plus untouched confirmation. Neither forecast metrics nor teacher-state agreement replaces that bar.
