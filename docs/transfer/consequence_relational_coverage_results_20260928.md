# Relational readout improves prediction but still fails clear retention

The matched relational scorer improves held-out score prediction and early clear prediction. It also passes the two count-based immediate-loss retention gates, unlike the pooled control. It still loses 150 clears relative to the base, including 120 early clears. The candidate therefore fails the required retention screen and does not advance to policy integration, autonomous games or confirmation.

Exactly one exploratory fit followed the [declared comparison](consequence_relational_coverage_20260928.md). The relational readout has 132,035 parameters versus the pooled control's 131,971. Both use the same frozen visible features, 292,762 action labels on 47,149 training states, normalization, state order, full-legal teacher-preference loss, three unit loss coefficients, optimizer, seed and 20-epoch exposure. The relational model starts from its original architecture-specific initialization. This changes the readout only; it adds no independently learned feature pathway, state diversity, selection rule, duration selection or hidden RNG input.

The predeclared prediction comparison holds the pooled control's 6,008 chosen development actions fixed:

| Fixed actions and metric | Pooled | Relational | Difference, paired 95% interval |
| --- | ---: | ---: | ---: |
| All: log-score MSE | 0.650542 | 0.586510 | −0.064032 [−0.088422, −0.040087] |
| All: clear BCE | 0.417628 | 0.407753 | −0.009875 [−0.022124, +0.002724] |
| Early: log-score MSE | 0.870925 | 0.805133 | −0.065792 [−0.120499, −0.012798] |
| Early: clear BCE | 0.508440 | 0.471851 | −0.036590 [−0.059390, −0.014666] |
| Outside original proposals: log-score MSE | 1.467351 | 1.262117 | −0.205234 [−0.298109, −0.118521] |
| Outside original proposals: clear BCE | 0.411660 | 0.375845 | −0.035816 [−0.061342, −0.008890] |

The early group contains 2,092 fixed actions; the outside group contains 1,072. Overall clear BCE and Brier differences remain unresolved. Early and outside-group Brier differences favor the relational readout, with intervals excluding zero. These fixed groups differ from each model's own selected-action groups.

On the original development proposals, relational MSE is 0.642022, BCE 0.317653 and Brier 0.100001. Against the pooled coverage control, MSE improves by −0.086449 [−0.112378, −0.058472], while overall BCE and Brier differences remain unresolved. All four required score/BCE gates against the training-only constant and count-by-Ante baselines pass. This supports a prediction benefit from the readout change in this single matched fit; it does not include independent-fit uncertainty.

Complete chosen-action assessment gives:

| Policy | Clears / 6,008 | Early clears / 2,092 | Immediate losses | Losses where teacher survives |
| --- | ---: | ---: | ---: | ---: |
| Base | 2,699 | 1,066 | 40 | 7 |
| Pooled coverage control | 2,527 | 937 | 43 | 10 |
| Relational coverage | 2,549 | 946 | 39 | 6 |

Relative to pooled, the relational scorer gains 22 clears: +0.3662 percentage points [−0.2029, +0.9479], unresolved. It gains nine early and 13 later clears; both intervals also span zero. Four fewer immediate losses are observed, with the paired overall interval reaching zero. Passing the declared count-based loss gates does not establish a statistically resolved reduction in loss risk.

Relative to base, clear rate drops by 2.4967 points [−3.1380, −1.8707]. The early deficit is 5.7361 points [−7.0560, −4.4819]. Both clear gates fail. Among 4,156 states where base already agrees with the teacher, clears fall from 2,007 to 1,854 and two new immediate losses appear. Among 1,852 disagreement states, clears rise from 692 to 695, an unresolved difference. Thus the net clear deficit remains concentrated in overriding existing correct teacher matches. Teacher agreement is a diagnostic grouping, not a deployable gate.

Eligible teacher argmax agreement is effectively unchanged: 3,803 versus pooled's 3,804. Full-legal teacher NLL is 1.590855 versus 1.547746. The relational model selects 1,003 actions outside their original proposal sets, versus pooled's 1,072. On its own outside choices, MSE is 1.198108 and BCE 0.375111; mean predicted clear probability is 30.4722% versus 29.0130% realized. These choice-conditioned metrics cannot replace the fixed-action comparison above.

The action selector still maximizes the preference logit. Score and clear probability remain auxiliary predictions, so better forecasts need not directly change the selected action. The next saved-data diagnosis will separate remaining prediction errors from cases where the preference head contradicts its own consequence estimates. Neither this failure nor the prediction gain establishes that frozen features are the limiting factor, or that an independently trainable branch would fix retention. Immediate-score maximization remains an inadequate long-horizon policy.

All results are original-RNG single-action interventions on 128 exposed development games. Intervals use 10,000 whole-game bootstrap draws with seed 1729. These are exploratory teacher-path measurements, not autonomous wins, conditional RNG means or proof of long-horizon generalization.

The preflight and fit audits verified unchanged inputs, architecture-specific initialization, all 7,380 updates, 942,980 state presentations, 5,855,240 consequence-label presentations and 244,984,880 legal-action presentations. First-batch output-gradient error was at most 6.32e−10; independent first-update Adam error was at most 7.66e−9. Fit/audit took 1,536.505/4.902 seconds. Prediction measurement/audit took 84.819/12.390 seconds and verified all 1,631,530 development legal state/action entries. Retention measurement/audit took 243.169/2.789 seconds, reproducing 26,045 factual observations and 128 terminals. It preserved 105 base non-play decisions and assessed every eligible choice: 5,005 original outcomes, 724 identical prior outcomes and 279 new branches. Every phase and owner actually exited zero before admission; no retry or second fit ran.

- [Fit audit](../../eval/consequence_prediction_20260927/relational_coverage_fit_v1/verification_v1.json), SHA256 `b98ce9d062a6e56e9593f848cf2dfd6cf33f30043b89df084216d4f26643e175`; [owner receipt](../../eval/consequence_prediction_20260927/relational_coverage_fit_owner_terminal_v1.json), actual exit `e56e82`, admission `8bb181`.
- [Prediction audit](../../eval/consequence_prediction_20260927/prediction_relational_coverage_v1/relational_action_coverage/verification_v1.json), SHA256 `47a35a88d493db3ba53c184e33dd1e4f7f7308a9fca107df3b856e2fac774cf2`; [owner receipt](../../eval/consequence_prediction_20260927/relational_coverage_prediction_owner_terminal_v1.json), actual exit `b51e11`, admission `8d4f91`.
- [Retention audit](../../eval/consequence_prediction_20260927/retention_relational_coverage_v1/relational_action_coverage/verification_v1.json), SHA256 `d5960f3aedb5a09b65b174844ce72e53598843dfed6f1c86c056c518387f41c4`; [owner receipt](../../eval/consequence_prediction_20260927/relational_coverage_retention_owner_terminal_v1.json), actual exit `74abeb`, admission `da522b`.

The final fresh-win, early-attribution, greedy and independent-confirmation requirements remain unchanged and unmet. World models remain deferred.
