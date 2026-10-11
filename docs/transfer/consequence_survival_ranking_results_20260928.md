# Survival supervision transfers a local ranking relation but fails retention

Direct survival-pair supervision reduces training preference error and improves the same relation on held-out original proposals. It does not improve full action selection: the candidate records 2,537 clears versus the matched relational control's 2,549 and the base's 2,699, with 39 immediate losses versus control's 39. Both clear-retention gates fail. This endpoint does not advance to integration, autonomous evaluation or a confirmation fit. Long-horizon generalization remains unproven.

Exactly one exploratory fit followed the [declared objective comparison](consequence_survival_ranking_20260928.md). The 132,035-parameter relational readout, seed-0 initialization, frozen visible features, 292,762 labels on 47,149 states, normalization, state order, optimizer and 20-epoch exposure match the relational coverage control. The sole scientific change adds `4 × mean_pairs softplus(preference(fatal) − preference(survivor))` on 3,396 labelled last-hand pairs at 430 states. Pairs are averaged within states; inactive states contribute zero. Full-legal teacher CE, score MSE and clear BCE remain in the objective. No new feature pathway, data expansion, duration selection or hidden RNG input is introduced.

The [complete training comparison](consequence_survival_training_evaluation_20260928.md) covers all pairs, including 1,030 involving expanded labels:

| Training comparison | Pairs | Control preference error | Survival objective |
| --- | ---: | ---: | ---: |
| All, 430 states | 3,396 | 18.3101% | 10.6118% |
| Early, 133 states | 1,052 | 19.2502% | 11.5622% |
| Later, 297 states | 2,344 | 17.8891% | 10.1861% |
| Original proposals, 420 states | 2,366 | 13.1303% | 7.0073% |
| Pairs involving expanded labels | 1,030 | 25.8450% | 16.8527% |
| Non-teacher survivor | 2,096 | 25.3838% | 15.4988% |

Errors average pairs within each state, then give states equal weight. No ties occurred. Raw incorrect pairs fall from 625 to 382: 279 improve and 36 worsen. Mean preference pair loss falls from 0.891239 to 0.342645; its contribution scaled across the full training population falls from 0.0325125 to 0.0124997. Training score-pair error falls from 11.7016% to 9.4839%, and clear-pair error from 10.7041% to 9.0756%. These are in-sample mechanism measurements, not policy outcomes.

The separately specified [held-out alignment check](consequence_survival_development_alignment_20260928.md) finds 115 last-hand states, including 56 with both surviving and fatal original proposals. Its 318 pairs come from 43 development games:

| Held-out preference comparison | States / pairs | Control error | Survival objective | Difference, paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| All | 56 / 318 | 11.8552% | 9.4494% | −2.4058 points [−5.0926, −0.2204] |
| Early | 20 / 113 | 9.6528% | 8.3333% | −1.3194 points [−5.4825, +2.2096] |
| Later | 36 / 205 | 13.0787% | 10.0694% | −3.0093 points [−6.5315, −0.4032] |
| Non-teacher survivor | 39 / 153 | 22.8632% | 18.3761% | −4.4872 points [−8.8895, −0.4762] |

Raw incorrect preference pairs fall from 44 to 34: 13 improve and three worsen. The overall interval supports a local transfer improvement in this exploratory fit; the early result remains unresolved. This check covers original labelled proposals only and supplies no evidence about unassessed actions or independent-fit variability. The bootstrap uses 10,000 draws with seed 1729, resampling contributing source games and keeping all their qualifying states and pairs together.

The other heads do not show the same held-out pair improvement. Score-pair error rises from 7.4752% to 9.2708%, and clear-pair error from 7.7976% to 10.7589%; both overall differences are unresolved. On teacher-survivor pairs, clear-pair error rises by 6.3988 points [1.25, 12.5]. These are exploratory head/group comparisons without multiple-comparison adjustment. The pattern warrants checking interactions between the losses, but does not establish a unique interference mechanism or a frozen-feature bottleneck.

All four original prediction gates against training-only constant and count-by-Ante baselines pass. Across 32,299 original development proposals, log-score MSE is 0.647855, clear BCE 0.321475 and Brier 0.100780. Differences from the matched relational control are unresolved for all three metrics.

Holding the control's 6,008 selected actions fixed gives:

| Fixed-action prediction | Control | Survival objective | Difference, paired 95% interval |
| --- | ---: | ---: | ---: |
| All: log-score MSE | 0.579529 | 0.566373 | −0.013155 [−0.039472, +0.009610] |
| All: clear BCE | 0.402865 | 0.406136 | +0.003271 [−0.006404, +0.013171] |
| Early: clear BCE | 0.458813 | 0.472592 | +0.013779 [−0.002589, +0.030578] |
| Outside original proposals: log-score MSE | 1.198108 | 1.045806 | −0.152302 [−0.304844, −0.046321] |
| Outside original proposals: clear BCE | 0.375111 | 0.364833 | −0.010277 [−0.031866, +0.011137] |

The fixed outside group contains 1,003 actions; its score prediction improves, while its clear prediction difference remains unresolved. These prediction comparisons use the original 128-game bootstrap. They differ from the contributing-game survival-pair analysis above. The candidate's own preference argmax selects 1,043 outside actions. Eligible teacher argmax agreement is 3,802 versus control's 3,803; full-legal teacher NLL is 1.594686 versus 1.590855.

Complete-action retention gives:

| Policy | Clears / 6,008 | Early clears / 2,092 | Immediate losses | Losses where teacher survives |
| --- | ---: | ---: | ---: | ---: |
| Base | 2,699 | 1,066 | 40 | 7 |
| Relational coverage control | 2,549 | 946 | 39 | 6 |
| Survival-ranking objective | 2,537 | 944 | 39 | 6 |

The candidate-minus-control clear difference is −0.1997 percentage points [−0.6395, +0.2387], unresolved. Against base, it is −2.6964 points [−3.3673, −2.0549]. Early clears fall by 122 relative to base: −5.8317 points [−7.1259, −4.5809]. Both count-based immediate-loss gates pass, but both clear gates fail. Passing those count gates does not establish a statistically resolved reduction in loss risk.

Among 4,156 states where base agrees with the teacher, clears fall from control's 1,854 to 1,826, versus base's 2,007. That additional −28-clear difference is −0.6737 points [−1.1494, −0.2173]. Among 1,852 disagreement states, clears rise from control's 695 to 711, versus base's 692; those differences remain unresolved. The net clear deficit therefore continues to involve overriding existing teacher matches. Teacher agreement is an analysis group, not an available deployment gate.

The [saved choice-alignment diagnostic](consequence_survival_choice_alignment_20260928.md) reconciles both policies' complete retention counts and connects the improved pair rankings to their actual choices. Thirteen improved pairs occur at seven states. Five retain the same selected action; two change it. In every improved pair the candidate chooses a third action rather than either pair member. One of those changed states fixes a fatal action: at seed 850000021, step 186, the control's Full House scores 21,195 and loses, while the candidate's outside-proposal Pair scores 52,056 and clears. This is an observed local correction, not evidence that the pair alone caused it.

A different last-hand change offsets that correction. At seed 850000027, step 121, the control's outside-proposal Pair scores 14,859 and clears; the candidate returns to the base's Three of a Kind, scoring 5,382 and losing. All nine original-proposal survival-pair classifications at this state remain unchanged. Thus both harmful and helpful action changes can escape a metric restricted to the original pairs. Both alternatives are fully assessed by the retention replay; neither is an unmeasured third choice.

Across 110 eligible last-hand states, 28 choices change, with one failure cured and one introduced. Both policies retain 71 clears and 39 losses. Across 5,898 eligible states with more hands remaining, 1,226 choices change, gaining 95 clears and losing 107. Although the added loss is active only on last-hand comparisons, the jointly trained scorer changes choices elsewhere. Overall it changes 1,254 choices, gains 96 clears and loses 108 relative to control. This demonstrates the remaining decision-preservation gap without establishing its unique architectural cause.

Selection still uses full-legal preference argmax with base non-play preservation. The survival relation is an original-RNG observed outcome, not a known conditional probability. Neither predicted score nor observed one-action survival substitutes for eventual value, resource preservation or growth. The next hypothesis should address the measured gap between learning a relation and making useful, retained corrections; another architecture change requires its own controlled comparison.

The fit audit verifies all 7,380 updates, 942,980 state presentations, 5,855,240 consequence-label presentations, 244,984,880 legal-action presentations and 67,920 survival-pair presentations. Independent first-active-batch output-gradient error is at most 8.90e−10; first and first-active Adam errors are at most 7.66e−9 and 7.56e−9. Fit/audit took 1,603.354/4.965 seconds. The training comparison took 1.981 seconds plus a 0.367-second audit; its original control outputs reproduce exactly. The standard-library held-out pair diagnostic took 0.816 seconds and exactly reproduces both prior prediction audits' MSE, BCE, Brier and teacher NLL.

Prediction measurement completed successfully in 151.716 seconds and assessed all 1,631,530 legal development state/action entries. Its initial audit failed on a stale pooled-control count assertion, preserved with the original exit-1 owner. The [separate recovery](consequence_survival_prediction_recovery_20260928.md) corrects only that count assertion plus provenance/output routing and audits the same saved files in 12.240 seconds. It changes no examples, predictions, gates or statistics. No fit or neural inference was repeated. The recovery owner actually exited zero before prediction admission.

Retention measurement/audit took 311.178/2.824 seconds. It reproduced 26,045 factual observations and all 128 terminals, preserved 105 base non-play decisions, and assessed every eligible choice using 4,965 original outcomes, 806 identical prior outcomes and 237 new branches. Its owner actually exited zero before admission. The saved choice-alignment diagnostic took 0.315 seconds, used no native inference or simulation, and independently recounted both policies' previously audited outcome groups. No additional fit ran.

- [Fit audit](../../eval/consequence_prediction_20260927/survival_rank_fit_v1/verification_v1.json), SHA256 `1ee35a01285824ddaf82cb2958b8cbd020d9b94532b298d6fcc92b71f87fd5de`; [owner](../../eval/consequence_prediction_20260927/survival_rank_fit_owner_terminal_v1.json), actual exit `f79f23`, admission `b6a1b3`.
- [Full training-pair audit](../../eval/consequence_prediction_20260927/survival_training_predict_v1/verification_v1.json), SHA256 `7479af0a1e77b40bdf68e8beb7e3a97fa6fd7e5cab2250436ec001c36ad7329b`; [owner](../../eval/consequence_prediction_20260927/survival_training_predict_owner_terminal_v1.json), actual exit `e30bb6`, admission `d86665`.
- [Corrected prediction audit](../../eval/consequence_prediction_20260927/prediction_survival_rank_v1/relational_survival_rank/verification_recovery_v1.json), SHA256 `547c6b0b237fa0e8e778a15b6431ba580b084444786372e72e4732b42f294cc7`; [admitted owner](../../eval/consequence_prediction_20260927/survival_rank_prediction_owner_terminal_v1.json), actual recovery exit `3f1fd3`, admission `8b512f`; [original failed owner](../../eval/consequence_prediction_20260927/survival_rank_prediction_failed_owner_v1.json).
- [Held-out pair result](../../eval/consequence_prediction_20260927/survival_development_alignment_v1/result_v1.json), SHA256 `0a89a9cdd5e21198495deb2296ac721a773845b8fdd75450b8c7b189d54c6777`; [owner](../../eval/consequence_prediction_20260927/survival_development_alignment_owner_terminal_v1.json), actual exit `fd8cd6`, admission `ffbfe0`.
- [Retention audit](../../eval/consequence_prediction_20260927/retention_survival_rank_v1/relational_survival_rank/verification_v1.json), SHA256 `31ff2700acc9a66a3fce677a8ad7fac4cc4cd54b395649a81d42cf1cb60c2cf6`; [owner](../../eval/consequence_prediction_20260927/survival_rank_retention_owner_terminal_v1.json), actual exit `ef2591`, admission `ae29be`.
- [Choice-alignment result](../../eval/consequence_prediction_20260927/survival_choice_alignment_v1/result_v1.json), SHA256 `77befe57da5c77c8bffdeec16e554ffd95e1abc6b3cb5f80996a456a85494e94`; [owner](../../eval/consequence_prediction_20260927/survival_choice_alignment_owner_terminal_v1.json), actual exit `48f6fd`, admission `f506f0`.

The fresh sampled-win, greedy-retention, early-attribution and final independent-fit confirmation requirements remain unchanged and unmet. World models remain deferred.
