# A fixed base distribution recovers useful decisions, but corrections still fail retention

The single fixed-base consequence fit recovers 150 clears relative to the preceding standalone survival scorer, including 118 early clears. It still falls 12 clears and four early clears short of base. Both clear-retention gates therefore fail, while both immediate-loss gates pass. This endpoint does not advance to policy integration, autonomous games or a confirmation fit. Long-horizon generalization remains unproven.

| Policy | Clears / 6,008 | Early clears / 2,092 | Immediate losses | Losses where teacher survives |
| --- | ---: | ---: | ---: | ---: |
| Base | 2,699 | 1,066 | 40 | 7 |
| Standalone survival-ranking control | 2,537 | 944 | 39 | 6 |
| Fixed-base consequence residual | 2,687 | 1,062 | 36 | 3 |

The clear difference versus the matched scorer is +2.4967 percentage points, paired 95% interval [+1.8395, +3.1760]. Against base it is −0.1997 points [−0.6765, +0.2836], unresolved. The early difference versus control is +5.6405 points [+4.3071, +7.0566]; versus base it is −0.1912 points [−0.7551, +0.3848]. The declared gates use counts and remain unchanged despite these unresolved deficits. The overall immediate-loss difference versus base has an interval ending at zero; four fewer observed losses does not establish a strictly resolved overall risk reduction.

The [controlled intervention](consequence_fixed_base_residual_20260928.md) changes the preference parameterization to `log P_base(action | play, state) + R_theta − R_initial`. The complete base conditional is immutable, its coefficient is one, and the initial readout is a fixed reference. Trainable initialization, 132,035 parameters, frozen visible features, 292,762 labels, 3,396 survival pairs, 47,149 states, order, optimizer, four losses and fixed 20-epoch endpoint match control. No new features, duration selection, loss sweep or hidden RNG input is introduced. The reference network is not another fitted replica. This comparison includes how the changed parameterization affects learning; it is not a post-hoc addition of base logits to the old trained scorer.

The [saved decision accounting](consequence_fixed_base_choice_alignment_20260928.md) identifies where the recovery comes from. These disjoint categories cover every eligible state:

| Relationship to base choice | States | Control clears | Candidate clears | Base clears |
| --- | ---: | ---: | ---: | ---: |
| Both retain base | 4,061 | 1,860 | 1,860 | 1,860 |
| Candidate returns to base | 1,100 | 344 | 505 | 505 |
| Candidate newly leaves base | 248 | 115 | 93 | 115 |
| Both keep the same override | 335 | 136 | 136 | 127 |
| Candidate changes to another override | 264 | 82 | 93 | 92 |

Returning to base recovers a net 161 clears, while newly leaving base costs 22. Changing between overrides adds 11 versus control. The candidate preserves base at 5,161 of 6,008 states and still makes 847 overrides. Relative to base, its retained old overrides add nine clears, changed overrides add one, and new overrides lose 22: net −12. This supports the preservation hypothesis in this exploratory fit without showing that the new correction function is sufficiently selective.

Against base, all changes gain 62 clears and lose 74. Among the 110 eligible last-hand states, the candidate clears 74 versus base's 70 and control's 71: five base-fatal actions are corrected and one new fatal action appears. With more hands remaining, it gains 57 clears and loses 73 versus base. An immediate clear is not eventual value; resource preservation and growth can justify deferring a clear, but no continuation-value evidence establishes that explanation for this deficit.

Where base already matches the teacher, the candidate retains 1,977 clears versus control's 1,826 and base's 2,007. The remaining −30-clear difference versus base is −0.7218 points [−0.9830, −0.4802]. Where they disagree, the candidate has 710 clears versus control's 711 and base's 692; the clear differences remain unresolved. Teacher agreement is an analysis group, not an available deployment gate.

One new fatal override is concrete: at development seed 850000043, step 31, base and control choose Pair action 144, scoring 900 and clearing. The candidate chooses Pair action 94, scoring 600 and losing. Its score prediction incorrectly ranks the latter higher: predicted log scores 7.327515 versus 6.939985; its clear logits are also slightly higher, 2.110481 versus 2.107375. The learned preference correction overturns base's preference for the successful action. All four original survival-pair classifications at this state remain unchanged. This illustrates a remaining consequence-estimation and action-selection error; it does not establish a unique missing feature or architectural cause.

All four prediction gates against training-only constant and count-by-Ante baselines pass. Prediction quality against the matched scorer remains mixed:

| Held-out prediction comparison | Control | Candidate | Candidate minus control, paired 95% interval |
| --- | ---: | ---: | ---: |
| Original proposals: log-score MSE | 0.647855 | 0.718214 | +0.070359 [+0.044564, +0.097169] |
| Original proposals: clear BCE | 0.321475 | 0.310271 | −0.011204 [−0.017438, −0.005101] |
| Original proposals: clear Brier | 0.100780 | 0.096806 | −0.003974 [−0.005829, −0.002125] |
| Fixed control choices: log-score MSE | 0.608301 | 0.666637 | +0.058336 [+0.019102, +0.098213] |
| Fixed control choices: clear BCE | 0.412148 | 0.410797 | −0.001351 [−0.011923, +0.008811] |

The fixed-control comparison evaluates the same 6,008 actions and outcomes under both predictors. The candidate's own selections differ, so their prediction metrics are not a matched prediction comparison. It selects 427 actions outside the original proposals versus control's 1,043. Eligible teacher argmax agreement rises from 3,802 to 4,173; full-support teacher NLL falls from 1.594686 to 1.231676. Better imitation here does not establish better autonomous play.

The [full training-pair audit](consequence_fixed_base_training_evaluation_20260928.md) finds preference error falling from 10.6118% to 9.7907%, with 173 pairs improved and 135 worsened. Raw incorrect pairs fall from 382 to 344, but mean preference pair loss rises from 0.342645 to 0.488100; the scaled full-training survival-loss contribution rises from 0.0124997 to 0.0178059. Fewer sign errors do not imply better confidence margins. Score-pair error rises from 9.4839% to 11.5129%, and clear-pair error from 9.0756% to 10.3062%. Expanded-label pairs improve in preference ordering, while original-proposal pairs worsen slightly.

The [held-out pair check](consequence_fixed_base_development_alignment_20260928.md) uses the same 318 original-proposal pairs at 56 mixed last-hand states from 43 games. Preference error rises from 9.4494% to 10.7242%, difference +1.2748 points [−1.5418, +4.5274]. Eight pairs improve and 14 worsen. Early error falls from 8.3333% to 6.8750%, but all 15 head/group comparison intervals include zero. No held-out pair-ranking gain is established. The better complete-action outcome therefore cannot be attributed to an improvement in this pair metric.

Intervals use 10,000 whole-source-game bootstrap draws with seed 1729 and state-balanced aggregation. The pair check resamples contributing games; prediction and retention use the original 128-game population. These are exploratory comparisons on reused development games, without an independent training replica or multiple-comparison adjustment. Original-RNG outcomes are realizations, not exactly knowable conditional expectations. No autonomous win-rate result follows from these teacher-state interventions.

The fit audit accounts for all 7,380 updates, 942,980 state presentations, 5,855,240 consequence labels and 67,920 survival-pair presentations, with the fixed reference unchanged. Fit/audit took 2,519.276/5.055 seconds. Before fitting, a preserved strict-centering preflight failure exposed a tiny attention-dispatch difference; a separate corrected preflight established exact initial preservation and stopped reference gradients without widening tolerances. No optimizer steps ran in these preflights, and this numerical issue was not offered as an explanation for poor transfer.

Prediction measurement/audit took 162.726/18.041 seconds and checked all 1,631,530 legal development state/action entries. Retention took 312.367/2.859 seconds, reproducing all 26,045 factual observations and 128 terminals. It preserved 105 non-play choices and evaluated every eligible choice using 5,581 original outcomes, 306 identical prior outcomes and 121 new branches. Both complete owners actually exited zero before admission. The saved-data pair and decision checks took 0.846 and 0.411 seconds, reproduced the prior audits, and used no new model inference or simulation.

- [Fit audit](../../eval/consequence_prediction_20260927/fixed_base_residual_fit_v1/verification_v1.json) and [owner](../../eval/consequence_prediction_20260927/fixed_base_residual_fit_owner_terminal_v1.json).
- [Full training-pair audit](../../eval/consequence_prediction_20260927/fixed_base_training_predict_v1/verification_v1.json) and [owner](../../eval/consequence_prediction_20260927/fixed_base_training_predict_owner_terminal_v1.json).
- [Prediction audit](../../eval/consequence_prediction_20260927/prediction_fixed_base_residual_v1/relational_fixed_base_residual/verification_v1.json) and [owner](../../eval/consequence_prediction_20260927/fixed_base_residual_prediction_owner_terminal_v1.json).
- [Held-out pair result](../../eval/consequence_prediction_20260927/fixed_base_development_alignment_v1/result_v1.json) and [owner](../../eval/consequence_prediction_20260927/fixed_base_development_alignment_owner_terminal_v1.json).
- [Retention audit](../../eval/consequence_prediction_20260927/retention_fixed_base_residual_v1/relational_fixed_base_residual/verification_v1.json) and [owner](../../eval/consequence_prediction_20260927/fixed_base_residual_retention_owner_terminal_v1.json).
- [Decision-accounting result](../../eval/consequence_prediction_20260927/fixed_base_choice_alignment_v1/result_v1.json) and [owner](../../eval/consequence_prediction_20260927/fixed_base_choice_alignment_owner_terminal_v1.json).

The next diagnosis should distinguish poor consequence estimates from poor preference corrections on the remaining overrides using these saved predictions and assessed outcomes. It should not automatically sweep the base coefficient, use a teacher-agreement oracle, or assume independent features are the bottleneck. The original fresh sampled-win, greedy-retention, early-attribution and final independent-fit confirmation requirements remain unchanged and unmet. World models remain deferred.
