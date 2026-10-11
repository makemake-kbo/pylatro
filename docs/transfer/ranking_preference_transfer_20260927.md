# Ranking preferences: a large training-to-held-out gap, with modest local transfer

The ranking objective's large training gains transfer only partly to comparable held-out cases. Correct pairwise ordering is **89.23% / 90.79% on training** and **43.36% / 44.54% on held-out states**. Nevertheless, ranking improves held-out ordering over matched ordinary continuation by **4.2773 percentage points [1.6447, 6.8000]**. The early-state improvement remains unresolved. These previously exposed teacher-state diagnostics do not overturn the [failed fresh-game generalization study](counterfactual_ranking_results_20260927.md).

The population was fixed before new inference: every historical teacher/base disagreement in which both choices are plays and exactly one clears the current blind. This gives **339 pairs at 339 states in 106 of 128 held-out games**, with 233 teacher-preferred and 106 alternative-preferred actions. The corresponding subset of the [previous training census](ranking_target_learning_20260927.md) contains **1,977 pairs** whose alternative was the archived base mode. The original 2,027-pair training table also included additional structural alternatives; those extra pairs are excluded from this matched comparison.

| Model | Training: preferred action outranks other | Held-out: preferred action outranks other | Held-out: preferred action is global mode |
|---|---:|---:|---:|
| Base | 582/1977 (29.44%) | 106/339 (31.27%) | 106/339 (31.27%) |
| Ordinary 0 | 877/1977 (44.36%) | 132/339 (38.94%) | 114/339 (33.63%) |
| Ordinary 1 | 853/1977 (43.15%) | 137/339 (40.41%) | 126/339 (37.17%) |
| Ranking 0 | 1764/1977 (89.23%) | 147/339 (43.36%) | 128/339 (37.76%) |
| Ranking 1 | 1795/1977 (90.79%) | 151/339 (44.54%) | 136/339 (40.12%) |

Most of the held-out errors remain in the teacher-preferred group: ranking corrects only **47 / 48 of 233** cases where the teacher clears and the base play does not. It retains **100 / 103 of 106** base-preferred rankings. This is consistent with limited transfer of the newly taught corrections, while preserving most of the already correct alternative preferences. The gap does not by itself identify an architectural cause, omitted input, or pure memorization mechanism.

| Model | Teacher preferred | Alternative preferred | Early, Ante < 4 | Known fatal alternative |
|---|---:|---:|---:|---:|
| Base | 0/233 (0.00%) | 106/106 (100.00%) | 35/117 (29.91%) | 0/7 (0.00%) |
| Ordinary 0 | 40/233 (17.17%) | 92/106 (86.79%) | 46/117 (39.32%) | 2/7 (28.57%) |
| Ordinary 1 | 42/233 (18.03%) | 95/106 (89.62%) | 50/117 (42.74%) | 2/7 (28.57%) |
| Ranking 0 | 47/233 (20.17%) | 100/106 (94.34%) | 53/117 (45.30%) | 2/7 (28.57%) |
| Ranking 1 | 48/233 (20.60%) | 103/106 (97.17%) | 50/117 (42.74%) | 2/7 (28.57%) |

Ranking chooses the preferred action overall in **128 / 136 of 339** cases and a third action in **63 / 54**. Both ranking replicas correctly order only two of the seven known fatal-alternative pairs and choose the known clearing action in one of seven. These counts do not measure the consequences of the third choices. The full held-out prediction scan below prepares that next check.

All fixed contrasts are shown below. Differences and intervals are percentage points; ordering and global-mode choice are separate metrics. A positive interval here describes local transfer on this selected diagnostic population, not a fresh-game win improvement.

| Stratum | Comparison | Pairwise ordering difference [paired 95% interval] | Preferred global-mode difference [paired 95% interval] |
|---|---|---:|---:|
| all | ranking − ordinary | +4.2773 [+1.6447, +6.8000] | +3.5398 [+0.8117, +6.2663] |
| all | ranking − base | +12.6844 [+8.6600, +16.8657] | +7.6696 [+3.5247, +12.2258] |
| all | ordinary − base | +8.4071 [+4.1925, +13.0435] | +4.1298 [-0.4121, +9.4595] |
| early | ranking − ordinary | +2.9915 [-0.8774, +6.9231] | +0.8547 [-3.5398, +5.3398] |
| early | ranking − base | +14.1026 [+7.1429, +21.6817] | +7.2650 [-0.4065, +15.4871] |
| early | ordinary − base | +11.1111 [+4.3688, +18.6957] | +6.4103 [-0.4168, +14.2857] |
| late | ranking − ordinary | +4.9550 [+1.6744, +8.3970] | +4.9550 [+1.7699, +8.5065] |
| late | ranking − base | +11.9369 [+7.6595, +16.4975] | +7.8829 [+2.9847, +13.0769] |
| late | ordinary − base | +6.9820 [+2.2000, +12.2504] | +2.9279 [-2.5232, +8.9385] |

Intervals use 10,000 bootstrap draws, seed 1729, resampling all 128 whole-game seed clusters and keeping both continuation replicas together. States receive equal weight; seeds without a relevant state retain zero numerator and denominator contributions. An independent frequency-count implementation reproduced all 18 intervals. These are unadjusted descriptive intervals on a panel already used for diagnosis. In the early stratum, ranking-minus-ordinary ordering effects are +5.9829 and 0.0000 points by replica; global-mode effects are +3.4188 and −1.7094. Early improvement is not established.

Both ranking endpoints were also measured on all **26,045 held-out teacher decisions**, including **6,113 teacher plays**, to save their actual choices for consequence replay. Teacher agreement is descriptive: different actions can have equal or better outcomes, so lower agreement is not itself proof of harm.

| Model | Teacher-action agreement, all decisions | Mean teacher NLL | Teacher-play exact matches |
|---|---:|---:|---:|
| Base | 21544/26045 (82.72%) | 0.639106 | 4156/6113 |
| Ordinary 0 | 21477/26045 (82.46%) | 0.690329 | 4127/6113 |
| Ordinary 1 | 21480/26045 (82.47%) | 0.690402 | 4137/6113 |
| Ranking 0 | 21440/26045 (82.32%) | 0.691767 | 4108/6113 |
| Ranking 1 | 21459/26045 (82.39%) | 0.685914 | 4097/6113 |

The chosen comparison population excludes 105 states where the base selected a different action family, including 56 clear-status disagreements. It also excludes 4,156 equal-action and 1,513 equal-clear-status play cases. Baseline disagreement selection makes the base's low pairwise accuracy intentional, not an unbiased policy error rate. The same selection rule on training and held-out data improves comparability but does not make the states identical in difficulty. The current positive local result can coexist with regressions outside these 339 cases, which this comparison alone cannot detect.

Next, replay the saved ranking choices on **all 6,113 held-out teacher-play states**, with exact historical teacher/base/ordinary controls. That will measure whether local ranking improvements increase clears, whether newly preferred third actions are useful, and whether regressions elsewhere offset the gain. No new fit is needed. Immediate clears still do not establish long-horizon benefit; the original goal requires independently verified fresh-game and early-decision improvement. World models remain deferred, and future exploratory changes follow the [single-fit workflow](experiment_workflow.md).

Measurement and independent audit completed with actual exit code zero, as did their serial supervisor. The audit rebuilt the entire target population from the previously verified raw consequence records, reconstructed all 1,695 pair margins and selected probabilities in float64, and checked every selected prediction against saved full-scan predictions. The 1,017 historical base/ordinary modes matched exactly; the 678 ranking modes matched their newly generated full scans exactly. Maximum probability-reconstruction error was 1.862843e−6 and maximum prediction-NLL discrepancy 4.768372e−6, below the established 1e−5 bound. No pair was near that tolerance's tie boundary. Source, cache and checkpoint hashes, plus all model state tensors, remained unchanged. There was no training or gameplay.

Artifacts: [complete structured results](data/ranking_preference_transfer_20260927.json), [frozen protocol](../../eval/ranking_preference_transfer_20260927/protocol_v1.json), [independent numerical and statistical audit](../../eval/ranking_preference_transfer_20260927/audit_result_v1.json), [completed execution](../../eval/ranking_preference_transfer_20260927/complete_v1.json), and [current handoff](../../eval/ranking_preference_transfer_20260927/RESUME.md). Full prediction pins, raw pair rows, all strata, all six individual-model contrasts and all 18 pooled intervals are retained in the structured results.
