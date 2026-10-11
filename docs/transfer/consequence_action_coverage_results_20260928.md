# Broader action labels improve consequence prediction, but not retention

Adding 47,125 training action labels improves consequence estimates on the fixed control's previously unlabelled development alternatives. It does **not** improve selected-action outcomes: the new scorer clears 2,527 blinds versus the control's 2,528, and incurs 43 immediate losses versus 41. All four prediction gates pass; all four base-retention gates fail. This candidate does not advance to policy integration, autonomous games or confirmation.

Exactly one exploratory fit ran, following the [declared action-coverage intervention](consequence_action_coverage_20260928.md). The model, frozen features, initialization, state order, optimizer, normalization, full-legal teacher-preference objective and fixed epoch-20 endpoint match the control. The additional labels expand the consequence targets from 245,637 to 292,762 on the same 47,149 training states. Consequence losses retain equal state weighting while averaging over each expanded action set; this reallocates individual action weights. There are no additional state contexts or games, new features, RNG inputs, duration selection or confirmation replica.

The [completed label audit](consequence_action_coverage_labels_results_20260928.md) also shows that the control already loses retention on familiar training states: 20,105 clears versus the base's 20,991, and 336 immediate losses versus 304. This rules out held-out-state shift or autonomous trajectory drift as the sole cause of this scorer's failure. It does not establish a frozen-feature or architectural bottleneck.

The [predeclared fixed-action comparison](consequence_action_coverage_evaluation_20260928.md) evaluates both models on the control's same 6,008 selected development actions. The 1,058 actions outside their states' original proposal sets show clear improvement:

| Fixed control-selected actions outside original proposals | Control | Expanded labels | Difference, paired 95% interval |
| --- | ---: | ---: | ---: |
| Log-score MSE | 1.789396 | 1.322757 | −0.466638 [−0.588036, −0.357235] |
| Clear BCE | 0.436517 | 0.368532 | −0.067985 [−0.090521, −0.045982] |
| Clear Brier | 0.141626 | 0.116542 | −0.025084 [−0.033593, −0.016706] |

These are approximately 26% lower score error and 16% lower clear BCE. Across all 6,008 fixed actions, MSE improves by −0.071276 [−0.097066, −0.046834], BCE by −0.012155 [−0.020704, −0.003492], and Brier by −0.004308 [−0.007259, −0.001253]. All three inside-proposal comparisons remain unresolved. Early score MSE improves, but early clear BCE changes by +0.008196 [−0.006612, +0.023693], so better early clear prediction is not established.

On the original 32,299 development proposals, expanded-label MSE is 0.728472, BCE 0.321881 and Brier 0.101278. Relative to the control, MSE changes by +0.011482 [−0.002240, +0.025072], BCE by −0.005414 [−0.009908, −0.000983], and Brier by −0.001393 [−0.002829, +0.000052]. The four required score/BCE comparisons against the training-only constant and count-by-Ante baselines all pass.

Action outcomes do not follow the prediction improvement:

| Policy | Clears / 6,008 | Early clears / 2,092 | Immediate losses | Losses where teacher survives |
| --- | ---: | ---: | ---: | ---: |
| Base | 2,699 | 1,066 | 40 | 7 |
| Full-support control, epoch 20 | 2,528 | 954 | 41 | 8 |
| Expanded action labels, epoch 20 | 2,527 | 937 | 43 | 10 |

The new scorer's clear-rate difference from the control is −0.0166 percentage points [−0.4785, +0.4207], unresolved. Its early count falls by 17 while later clears rise by 16; neither within-period clear-rate difference excludes zero. Relative to base, the deficit is 172 clears, −2.8628 points [−3.5272, −2.2128], including 129 early clears. The scorer gains 90 clears and loses 262. Its observed loss increases fail the declared count-based gates, even though the overall loss-rate intervals versus base span zero.

Of the 172 net lost clears, 164 occur among the 4,156 states where base already agrees with the teacher: 1,843 versus 2,007 clears, with four new immediate losses versus zero. In the 1,852 disagreement states, clears are 684 versus 692, with six versus seven losses where the teacher survives. Teacher agreement is a diagnostic grouping and cannot serve as a deployment rule.

Eligible teacher argmax matches change little: 3,804 versus the control's 3,800. Full-support teacher NLL improves from 1.572541 to 1.547746, while selected actions outside original proposals increase from 1,058 to 1,072. On the new model's own outside-proposal choices, MSE is 1.467351 and BCE 0.411660; their mean predicted clear probability is 33.9546% versus 28.2649% realized. This is a different choice-conditioned set from the control's 1,058 actions; the fixed-action comparison above is the appropriate prediction contrast.

The two controlled interventions now separate failure modes. [Full-support preference training](consequence_full_support_results_20260928.md) recovered many clears without a resolved improvement in original-proposal consequence errors. Broader consequence labels now improve predictions on the fixed alternatives without improving retention. The policy still chooses the highest preference logit; predicted score and clear probability are auxiliary outputs, not direct terms in that choice. Shared features allow interaction, but the experiment shows that improving these auxiliary predictions is insufficient to repair action selection under this recipe. It neither proves that a different readout cannot help nor explains the entire original pretraining win gap.

Before another fit, inspect the saved predictions on lost-clear and fatal choices: distinguish cases where the consequence heads already favor the surviving base action from cases where they also misrank it. This requires no new inference, simulator runs or training. Do not infer a safe confidence gate or replace long-term policy value with immediate-score maximization from aggregate prediction gains.

All results use original-RNG single-action interventions on the same 128 exposed development games. Intervals use 10,000 whole-game bootstrap draws with seed 1729 and do not include independent-fit uncertainty. They are exploratory mechanism evidence, not fresh autonomous wins, conditional RNG averages or long-horizon generalization.

The technical preflight verified every original and added label, preserved frozen feature/support arrays, and matched the initial predictions and preference gradients exactly to the control. Independent first-batch output-gradient error was at most 5.92e−10. The single fit completed 20 epochs, 7,380 updates, 942,980 state presentations and 5,855,240 consequence-label presentations; its saved-artifact audit verified all exposure, checkpoints and first-update arithmetic, with Adam maximum error 4.04e−9. Fit/audit took 1,244.704/4.837 seconds. Unlike the disclosed original control's post-fit validator failure and separate artifact recovery, this new fit and its owner exited zero without a retry.

Prediction measurement/audit took 28.231/12.457 seconds and verified all 1,631,530 development legal state/action entries. Retention measurement/audit took 192.627/2.734 seconds. Replay checked all 26,045 factual decisions and 128 terminals, preserved 105 base non-play decisions, and assessed every eligible choice: 4,936 reused original labels, 736 reused identical prior outcomes, and 336 required new branches. The audit recomputed all outcome counts and gained/lost sets. Every owner actually exited zero before admission.

- [Fit audit](../../eval/consequence_prediction_20260927/action_coverage_fit_v1/verification_v1.json), SHA256 `13d10b3251bb21bb5bdf172f018141377c491b672aceb6ce73f6427385edb18c`; [owner receipt](../../eval/consequence_prediction_20260927/action_coverage_fit_owner_terminal_v1.json), actual exit `7163d8`, admission `fc4b1d`.
- [Prediction audit](../../eval/consequence_prediction_20260927/prediction_action_coverage_v1/pooled_action_coverage/verification_v1.json), SHA256 `e0dece2246109a957cb877f9e5df2db4cd841c02754774423bdd79bb612875a2`; [owner receipt](../../eval/consequence_prediction_20260927/action_coverage_prediction_owner_terminal_v1.json), actual exit `53303c`, admission `de6270`.
- [Retention audit](../../eval/consequence_prediction_20260927/retention_action_coverage_v1/pooled_action_coverage/verification_v1.json), SHA256 `eb0e96327a339d1817c6bde44fafdfdc1758469c98e04470b5b8bb015384b604`; [owner receipt](../../eval/consequence_prediction_20260927/action_coverage_retention_owner_terminal_v1.json), actual exit `607964`, admission `067562`.

The final fresh-win, early-attribution, greedy and independent-confirmation requirements remain unchanged and unmet. World models remain deferred.
