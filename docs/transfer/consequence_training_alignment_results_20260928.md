# Survival-ranking errors already occur on supervised training alternatives

Both completed coverage scorers make survival-ranking errors on actions that were explicitly supervised during training. Among 420 last-hand states with both surviving and fatal assessed alternatives, the relational preference head's state-averaged pair-ranking error is 13.13%, while its clear head's error is 9.21%. The teacher survives in all 420 states. Fresh-game trajectory drift and disagreement with a failing teacher cannot be the sole explanations for these observed errors.

This [predeclared post-hoc audit](consequence_training_alignment_20260928.md) uses the saved epoch-20 predictions on the original 245,637 supervised proposals across 47,149 training states and 1,000 games. Both models also trained on 47,125 expanded action labels, whose predictions are absent from these saved six-slot curves. Results therefore cover the original assessed proposals, not all training alternatives or the actual full-action argmax policy. No new training, inference or simulation ran.

There are 904 last-hand states across 579 games, containing 4,707 proposals. Of these, 420 states across 327 games contain both surviving and fatal actions, yielding 2,366 survivor–fatal pairs. For each pair, the correct observed ranking places the surviving action above the fatal action. The following error fractions give each mixed state equal weight; ties count as half an error, though none occurred:

| Head | Pooled error | Relational error | Pooled incorrectly ranked pairs | Relational incorrectly ranked pairs |
| --- | ---: | ---: | ---: | ---: |
| Score | 8.13% | 10.34% | 201 | 256 |
| Clear probability | 7.64% | 9.21% | 192 | 221 |
| Preference | 13.49% | 13.13% | 329 | 320 |

These descriptive results do not include independent-fit uncertainty. The relational model's lower aggregate score MSE does not imply better survival ranking in this subset. Last-hand score MSE improves from 1.416383 to 1.256455, while clear BCE changes from 0.371778 to 0.381062. Overall original-training MSE improves from 0.702892 to 0.608300 and BCE from 0.286879 to 0.282265. Aggregate prediction gains can coexist with worse rankings at survival decisions.

The teacher survives in 632 last-hand states and fails in 272. In the 272 failing-teacher states, no original assessed alternative survives. Thus none of these original proposal labels asks preference training to favor a fatal teacher action over an observed surviving alternative. This is limited to the assessed original-RNG outcomes; it does not establish optimal teacher actions, absence of stochastic selection bias, or conditional survival probabilities. Preference cross-entropy also does not explicitly rank every non-teacher survivor above every non-teacher fatal action.

Restricting the preference argmax to original proposals illustrates in-sample mistakes, but is not a deployable policy because the set includes the teacher:

| Original-proposal diagnostic, eligible last-hand states | Base | Pooled restricted | Relational restricted |
| --- | ---: | ---: | ---: |
| Clears / 870 | 566 | 569 | 563 |
| Immediate losses | 304 | 301 | 307 |
| Fatal overrides of a surviving base action | — | 10 | 17 |

The relational restricted choice gains 14 clears and loses 17 relative to base. In six of its 17 fatal overrides, its clear head already favors base; in seven, its score head favors base. The pooled counts are four and five of ten. Across all 46,427 base-play training states, restricted clears are 20,983 for pooled and 20,919 for relational, versus 20,991 for base. These counts must not be confused with complete-action policy outcomes from earlier experiments.

The evidence supports separating two mechanisms in subsequent work: learning accurate survival consequences and making preferences respect that information. It does not identify a frozen-feature bottleneck, prove insufficient context, or justify immediate-score maximization. A controlled objective experiment on already labelled last-hand survival pairs is a more direct test of the observed preference gap than assuming another feature pathway will repair it. Before specifying that experiment, compare it with the earlier ranking interventions to avoid repeating an already-tested change. Keep the architecture, labels, exposure and original scientific gates fixed if the preference objective is changed; any independently trainable feature experiment remains separate.

The audit read the saved float32 arrays with a standard-library parser checked against independently encoded signed values and truncation rejection. It reproduced both models' previously audited original-training MSE, BCE, Brier and proposal teacher NLL exactly, checked row/action identities and group/pair arithmetic, and saved all derived state rows. It took 3.681 seconds. This reconciliation is not a second independent implementation of every new statistic.

- [Result](../../eval/consequence_prediction_20260927/training_consequence_alignment_v1/result_v1.json), SHA256 `e0e1db95464e204640220f79f5a56595d55e2f9d706875d34e2887f9a2a35060`.
- [Owner receipt](../../eval/consequence_prediction_20260927/training_consequence_alignment_owner_terminal_v1.json), actual exit `42279d`, admission `f379dc`; SHA256 `9cca20e398eee6e1240efd31f5422a1dbb0b15f7329e3ce8a1b5d00777ebfbfc`.

The matched relational endpoint remains rejected by its clear-retention gates. No confirmation fit, autonomous-win improvement or long-horizon generalization claim follows from this training diagnostic.
