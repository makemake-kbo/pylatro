# Binary opportunity labels lose choice information, but learning is also weak

The event target often marks several owned Jokers as beneficial to replace. Even a perfect prediction followed by the existing maximum-over-offers, uniform-owned-tie rule would reproduce only **63.29%** of development teacher card choices. Maximizing the recorded continuous gain instead gives **99.22%** under the same owned-slot tie convention. These are oracle diagnostics of recorded teacher search, not learned policies or game outcomes.

| Active teacher-sale population | Training | Development |
|---|---:|---:|
| States | 5,380 | 703 |
| No positive owned slot | 10 | 3 |
| Exactly one positive owned slot | 2,351 | 296 |
| More than one positive owned slot | 3,019 | 404 |
| Teacher is the unique positive slot | 2,346 | 296 |
| Binary-event oracle agreement | 64.25% | 63.29% |
| Observed continuous-gain oracle agreement | 99.01% | 99.22% |
| Learned event forecast agreement | 32.70% | 31.58% |

The development gain-versus-event agreement difference is +35.93 percentage points, with a paired complete-game 95% interval of [33.53, 38.32]. Binary-event versus learned agreement differs by +31.71 points [27.93, 35.51]. Fractional agreements are expected counts under uniform ties among owned slots, not rounded observed policy actions.

Crucially, target ties do not explain the whole learning shortfall. Where the teacher is the unique positive slot, the learned event predictor selects it in **98/296 development states (33.11%, interval [27.76%, 38.58%])** and **855/2,346 training states (36.45%)**. The problem is already present on clear training examples. More data, a larger network or a different target is not established as a sufficient fix.

All 703 development teacher slots have at least one observed gain. One training teacher slot does not; it remains a miss in the recorded-gain comparison. Unknown gains are never filled with zero. The mean gap between the best and second-best observed owned-slot maxima is 0.137921 on the 667 development states with at least two observed owned slots, with interval [0.126423, 0.150273]. These are heuristic gain units, not eventual win value.

The event oracle is a bound only for the specified scalar maximum and tie rule. The full binary event matrix, other consequence targets, card features and history may contain additional information. The recorded-gain oracle also uses the teacher's observed-comparison support, which is unavailable to policy inference. Neither result permits an oracle deployment or qualifies the rejected event endpoint for gameplay.

The [plan](shop_choice_label_diagnostic_plan_20261001.md) and source were frozen before the analysis. Independent vectorized and scalar enumeration agreed exactly on all 6,083 teacher-sale rows; learned agreement reproduces the previously admitted event analysis. Five synthetic cases check ties, unequal offer multiplicities, absent gains and the strict native 0.03 boundary. The existing tested aggregator supplies 10,000 paired resamples of all 128 complete development games, seed1729. The three zero-positive development states have 3,700 empty resamples, so their subgroup interval is not informative. The full process exited zero before admission. No model, simulator or training calls occurred.

Together with the earlier finding that most continuous-gain MSE improvement came from state-average prediction, this motivates one matched test of explicitly learning differences between alternatives. It does not justify another pooling-only change or a direct connection from the current event forecast to policy decisions. The original win-rate, retention, early-attribution and final confirmation requirements remain unmet.

[Results and source pins](../../eval/consequence_prediction_20260927/shop_choice_label_diagnostic_results_v1.json) · [Actual-exit admission](../../eval/consequence_prediction_20260927/shop_choice_label_diagnostic_owner_terminal_v1.json) · [Earlier continuous-gain diagnosis](shop_dense_decision_results_20261001.md) · [Policy pooling diagnosis](shop_pooling_diagnostic_results_20261001.md)
