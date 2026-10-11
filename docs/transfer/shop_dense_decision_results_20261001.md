# Useful comparisons are learned, but beneficial replacements are underpredicted

The single dense-supervision fit learns predictive signal on the existing development panel, but its policy fails seven of thirteen fixed checks. It matches 165 of 716 teacher sales, versus 167 for the base and 170 for ordinary1k. It does not qualify for gameplay or a confirmation fit. The full [training and policy report](shop_dense_results_20261001.md) retains all controls and the original win-rate, retention, early-decision attribution, and independent-confirmation requirements.

The saved-output diagnostic narrows the gap. Replacement-gain MSE improves by 34.54% against the training constant, but 83.50% of that reduction comes from predicting the mean gain within each shop more accurately. There is also useful discrimination among alternatives; the result is not explained entirely by state averages.

| Development diagnostic | Predictor | Comparison | Paired difference, 95% interval |
|---|---:|---:|---:|
| Raw gain MSE, 6,617 decisions | 0.080543 | Training constant 0.123037 | −0.042494 [−0.050497, −0.035589] |
| Mean-component MSE, same decisions | 0.034745 | Constant 0.070226 | −0.035482 [−0.043007, −0.029030] |
| Centered pair-component MSE, same decisions | 0.045798 | Constant 0.052811 | −0.007013 [−0.008288, −0.005768] |
| Pair concordance, 6,341 nontrivial decisions | 61.92% | Chance 50% | +11.92 points [+10.39, +13.41] |
| Best observed pair selected, same nontrivial decisions | 29.14% | Uniform choice 18.32% | +10.82 points [+8.48, +13.18] |
| Native-gain regret, same nontrivial decisions | 0.162022 | Uniform choice 0.274136 | −0.112114 [−0.128966, −0.094509] |

At the teacher's existing strict gain threshold of 0.03, the predictor misses most observed beneficial opportunities on both training and development data.

| Population with observed gains | Decisions | True maximum above 0.03 | Recognized | Missed | False positives |
|---|---:|---:|---:|---:|---:|
| Training | 50,055 | 5,382 | 621 (11.54%) | 4,761 | 29 |
| Development | 6,617 | 700 | 96 (13.71%) | 604 | 5 |

For pairs whose recorded gain exceeds 0.03, decision-balanced prediction bias is −0.374939 in training and −0.371727 in development (development 95% interval [−0.389731, −0.353872]). The mean training target is −0.485641. Only 1,646 of 44,606 observed development pairs exceed the threshold; with the experiment's equal-decision weighting their average fraction is 4.21%. The similar training and development misses show that the boundary problem is already present at the fitted training endpoint. They do not prove that model capacity, training duration, or the objective is uniquely responsible.

The evidence motivates a controlled test of supervision focused on the beneficial-replacement boundary, holding the representation and budget fixed. It does not justify shifting a threshold, increasing a coefficient, extending this rejected fit, or assuming the learned gain prediction can safely control gameplay. No next fit has been launched or recipe frozen.

All selection diagnostics use the **true observed-pair mask**. They reveal what the predictor knows among assessed pairs, not an executable policy: unassessed alternatives may be better. The 895 development and 7,202 training annotated states without recorded gains are excluded from gain metrics. Numeric targets are heuristic forecasts, not expected win values. The true observed maximum above 0.03 corresponds to 700 teacher sales and zero teacher non-sales in development, with three teacher sales below that criterion. In training it corresponds to 5,370 sales and twelve non-sales, with ten missed sales. The criterion therefore does not universally reconstruct teacher priorities.

The MSE decomposition weights decisions equally and exactly reconstructs the admitted raw MSE. Ranking requires at least two assessed pairs with unequal true gains; predicted ties share probability uniformly, while concordance excludes true ties and gives predicted ties half credit. Development intervals use 10,000 complete-game resamples of all 128 games, seed 1729, including five games with no gain labels. Each metric retains its own denominator; no draw had a zero denominator. These are marginal exploratory intervals on an already exposed panel, without multiplicity adjustment.

The independent scalar audit reconstructs all 50,055 training and 6,617 development decisions, 374,888 observed pairs, all 25 reported metrics, and development intervals using independently accumulated game multiplicities. Maximum point discrepancy is 5.56e−17 and interval discrepancy 1.67e−16. Neither analysis performs training, model forwards, or simulator queries. The first diagnostic launcher failed before loading its first array because a strict file-pin validator rejected additional array metadata; the corrected version strips metadata only for pin validation and retains shape/dtype checks. The failed source, protocol, and log remain preserved; mathematical functions and populations are unchanged.

Artifacts: [prospective diagnostic definitions](shop_dense_decision_diagnostic_20261001.md), [results](../../eval/consequence_prediction_20260927/shop_dense_decision_diagnostic_results_v2.json), [independent audit](../../eval/consequence_prediction_20260927/shop_dense_decision_diagnostic_verification_v1.json), [process admission](../../eval/consequence_prediction_20260927/shop_dense_decision_diagnostic_owner_terminal_v2.json), [preserved failed attempt](../../eval/consequence_prediction_20260927/shop_dense_decision_diagnostic_failed_owner_v1.json).
