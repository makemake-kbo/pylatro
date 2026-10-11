# Ranking and preservation: development evaluation

**Combining ranking and preservation did not qualify for confirmation.** The single exploratory fit, all 1,024 autonomous development episodes, and the independent audits completed successfully. The candidate won 34/128 sampled games, versus 39/128 for base and 40/128 for ordinary continuation. Using the candidate only for early decisions produced 37/128 wins, exactly the control count. Three of the five fixed development gates failed. No second fit or final confirmation is scheduled; the [single-run workflow](experiment_workflow.md) still applies, and long-horizon generalization remains unproven.

| Full-game policy | Greedy wins / 128 | Sampled wins / 128 |
|---|---:|---:|
| Base | 59 (46.09%) | 39 (30.47%) |
| Ordinary continuation | 57 (44.53%) | 40 (31.25%) |
| Ranking + preservation | 57 (44.53%) | 34 (26.56%) |

The early intervention uses a separate exposed 128-seed panel. It routes the candidate until the first preaction Ante >= 4, then permanently uses ordinary0. Both the hybrid and ordinary0 win 37 games (28.91%). All seven declared comparisons are retained below; differences and paired 95% intervals are in percentage points.

| Comparison | Difference | Paired 95% interval |
|---|---:|---:|
| Sampled combined − base | -3.91 | [-12.50, 4.69] |
| Sampled combined − ordinary | -4.69 | [-14.06, 4.69] |
| Greedy combined − base | -1.56 | [-9.38, 6.25] |
| Greedy combined − ordinary | 0.00 | [-8.59, 8.59] |
| Early hybrid − ordinary | 0.00 | [-8.59, 8.59] |
| Sampled ordinary − base | +0.78 | [-7.81, 9.38] |
| Greedy ordinary − base | -1.56 | [-10.94, 7.03] |

Both sampled +5-point development gates and the positive early-effect gate fail. The two provisional greedy point gates pass, but their intervals do not establish the stricter final greedy guardrails. Every interval spans zero, so the sampled point deficits do not establish harm. These are unadjusted descriptive intervals from one exploratory fit, one sampling stream and previously exposed development seeds, using 10,000 whole-seed bootstrap draws with seed 1729. They are not untouched confirmation evidence.

Independent verification checked all 1,024 episodes, 160,169 action records, permanent early routing and all seven intervals. Every panel, analyzer, verifier, outer phase and the outer runner exited successfully. The [complete autonomous export](data/ranking_preservation_autonomous_20260927.json) binds the raw outcomes, actual exits, frozen manifest, fit evidence and independent verification. The development result supports no further confirmation of this recipe. The remaining diagnostics explain what it fitted and what failed to transfer; they do not establish that a particular local error caused the sampled win difference.

The combination reduces some forgetting but gives up some corrections. On all6,113 teacher-play states, it gains40 net clears in the339 preference cases, loses35 where base already matched the teacher, and gains1 elsewhere: **only6 net clears over base**. The fixed partition shows how those effects compare with the separate objectives; every number below is a clear-count difference from the same base.

| Continuation | Preference cases (339) | Base equals teacher (4156) | Other disagreements (1618) | All states (6113) |
|---|---:|---:|---:|---:|
| Ordinary | +30 | -54 | -4 | -28 |
| Ranking | +50 | -54 | 0 | -4 |
| Preservation | +29 | -22 | -3 | +4 |
| Ranking + preservation | +40 | -35 | +1 | +6 |

Combined clears2705states, versus base2699, ordinary2671, ranking2695 and preservation2703. The combined-minus-base difference is **+0.098 percentage points, paired95% interval[-0.261,0.464]**, so improvement over the base remains unresolved. Combined minus ordinary is **+0.556 points[0.236,0.903]**. Differences from ranking (**+0.164[-0.132,0.443]**) and preservation (**+0.033[-0.251,0.340]**) are unresolved. These are immediate action consequences on exposed teacher paths, not full-game win rates.

The35net lost clears where base matched the teacher comprise3new clears and38lost immediate clears, with19losses each in early and late states. Four of the38changes are discards that leave the game active, rather than a played hand; the replay does not measure whether those discards improve a subsequent hand. Only one of the38immediately ends the game: the previously traced Pair-to-High-Card regression at850000064/138. Treating all38as lost games or as errors inside the play-scoring head would overstate this evidence.

There is no early-state clear gain over ordinary: both clear1063of2152states at preaction Ante<=3, with a paired difference interval[-0.472,0.476]points. All34 extra clears over ordinary occur in the late stratum. The completed autonomous early-policy intervention also shows no point gain in eventual wins:37/128for each policy. Its interval remains wide, as reported above.

On the339 preference cases, both combined and ranking choose the designated preferred action128times. Combined's40third choices clear18times, versus28clears from ranking's63third choices; this accounts for its10fewer target clears. Neither model's third choices create a new immediate loss in replica0. Across the entire replay, combined creates4new immediate losses relative to the teacher, versus base7, ordinary5, ranking5 and preservation4. All214,417 raw replay checks,24 bootstrap intervals and47 independent cohort comparisons passed.

The combined objective learns training preferences but retains a large transfer gap. On the matched target populations, its pairwise accuracy is **82.55% in training and 40.12% on exposed held-out states**. When the teacher action is better than the base action, those figures are **75.48% (1053/1395) and 14.59% (34/233)**. On held-out cases where the base action is better, it retains102/106 correct pair rankings. This identifies a continuing difficulty in transferring corrections; it does not establish whether missing coverage, representation, or another factor causes that difficulty.

The [saved-score decomposition](ranking_preservation_score_decomposition_20260927.md) narrows this gap without another fit. Relative to base, the combined model shifts the teacher-better pair margin upward on1394/1395training pairs but only131/233held-out pairs; both its internal reference and correction terms show much weaker held-out changes. Inside the combined endpoint, the correction repairs572training reference errors and introduces32, versus16repairs and44reversals held out. The same mismatch appears in early and late states. This is a selected-pair description, not evidence that removing the correction head improves the policy; shared encoder features affect its scores, and earlier actual component interventions ruled out head-weight changes as the sole cause of a shared fatal error.

Pairwise accuracy means assigning more probability to the better of the two compared actions. The global mode can still be a third action. The matched training population contains1,977 pairs whose comparison action is the base mode; the held-out population contains339 such pairs. All2,027 original training pairs were measured as well.

| Model | Matched training pair accuracy | Held-out pair accuracy | Held-out preferred global mode |
|---|---:|---:|---:|
| Pretrained base | 582/1977 (29.44%) | 106/339 (31.27%) | 106/339 |
| Ordinary continuation | 877/1977 (44.36%) | 132/339 (38.94%) | 114/339 |
| Ranking | 1764/1977 (89.23%) | 147/339 (43.36%) | 128/339 |
| Preservation | 791/1977 (40.01%) | 125/339 (36.87%) | 115/339 |
| Ranking + preservation | 1632/1977 (82.55%) | 136/339 (40.12%) | 128/339 |

On all2,027 training pairs, adding preservation reduces ranking accuracy from1814/2027 (89.49%) to1682/2027 (82.98%). The combined model makes18 new pairwise corrections relative to ranking0 but loses150 previously correct rankings. It preserves10 additional alternative-preferred rankings while losing142 net teacher-preferred rankings. These are training measurements, not generalization evidence.

On the held-out panel, combined minus ordinary pair accuracy is **+1.18 percentage points, paired95% interval[-2.13,4.15]**, so that difference is unresolved. Preferred global modes increase by14/339, **+4.13 points[0.32,7.73]**. Relative to ranking alone, combined pair accuracy falls **3.24 points[-6.16,-0.83]**, while the preferred-mode count remains128/339. There are40 third choices rather than ranking's63; their actual consequences are measured separately.

The early-state preference improvement over ordinary is unresolved: pair accuracy **+0.85 points[-4.90,6.31]**, preferred modes **+3.42[-2.80,9.40]**. All24 descriptive intervals were independently reconstructed using10,000 whole-seed bootstrap draws, seed1729, retaining all128 game clusters. They are unadjusted intervals on previously exposed development evidence, not final confirmation.

The complete26,045-row teacher-state scan shows less change from the base than ranking or ordinary continuation. Combined teacher NLL is0.646252 versus base0.639106, ordinary0.690329, ranking0.691767 and preservation0.643091. Combined matches4,182 of6,113 teacher plays, versus base4,156, ordinary4,127, ranking4,108 and preservation4,196. Teacher agreement does not establish that an action clears the blind or improves eventual wins; the full replay and game evaluations address those questions.

The [earlier action replay](ranking_hand_cost_20260927.md) motivated this combination: ranking gained50 clears on targeted preference cases but lost54 on states where base already matched the teacher in replica0. Preservation retained more existing behavior but learned fewer corrections. The current study tests whether combining them improves that tradeoff; separate prior effects did not guarantee a combined benefit.

The fixed objective is teacher NLL once +4 × mean-within-state preference loss +1 × full forward KL(base || current). Coefficients are reused without a sweep. The fit uses the same202,141-row/1,000-game cache, pretrained base and ordinary0 row stream;4,000 updates, logical batch320/micro32, MPSfp32, fresh AdamW3e-5 after100 warmup updates, no weight decay or gradient clipping, disabled dropout and fixed critic weights. Only the fixed4,000-update endpoint is evaluated as the candidate. The existing ordinary0, ranking0 and preservation0 controls are reused under verified provenance.

Fit verification completed successfully: actual worker, CPU auditor and supervisor exits are all0;1,280 checkpoint tensors and930 optimizer states were checked across snapshots0/1000/2000/4000, with all1,280,000 training exposures and update logs checked. The independent midpoint audit also reconstructed all640,000 exposures through three full dataset passes. The final training-cache scan has202,141 rows,88.1583% teacher agreement and NLL0.408869; these training numbers are not a win rate.

The completed greedy base and ordinary0 controls also pass a [trajectory reproducibility check](data/ranking_preservation_greedy_control_reproducibility_20260927.json). Across the same128seeds for each policy, all256game results and42,057recorded actions match the corresponding earlier ranking-study runs exactly. Checkpoint identities, runtime source and evaluation settings match. Timing fields and critic floating-point forecasts are excluded from outcome equality; complete recorded action traces are compared. This checks the unchanged controls, not the new candidate, and does not replace the pending full verification. Sampled streams are deliberately not compared this way because changing the full panel changes their global random stream.

The completed fixed development evaluation contains1,024 autonomous episodes. The replay includes the339 preference cases,4,156 base-equals-teacher cases and1,618 other disagreements, all third choices and new immediate losses. Autonomous evaluation uses base/ordinary0/combined0 in greedy and sampled modes on the first128 already exposed ranking-study seeds, plus a separate128-seed early-policy intervention that permanently switches to ordinary0 from the first preaction Ante>=4. It uses one sampling stream. The unchanged provisional readiness screen rejects this candidate. Any future final candidate still needs one independent confirmation fit and untouched verification evidence under the full generalization standard. World-model work remains deferred.

Evidence: [autonomous counts and all seven intervals](data/ranking_preservation_autonomous_20260927.json), [consequence data and all24 intervals](data/ranking_preservation_consequences_20260927.json), [preference data and all24 intervals](data/ranking_preservation_preferences_20260927.json), [fit audit](../../eval/ranking_preservation_20260927/fit_audit_v1.json), [training preference audit](../../eval/ranking_preservation_20260927/development_train_v1/audit_result_v1.json), [held-out preference audit](../../eval/ranking_preservation_20260927/development_held_v1/audit_result_v1.json), [frozen development protocol](../../eval/ranking_preservation_20260927/development_protocol_v1.json), [current process handoff](../../eval/ranking_preservation_20260927/RESUME.md).
