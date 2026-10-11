# Full-action preference search explains where readout retention fails

The [post-hoc saved-data diagnosis](consequence_support_diagnostic_20260928.md) is complete. In both failed readouts, more than 95% of lost clears and every newly introduced fatal choice occur when the full-action preference winner lies outside the original supervised proposal set. This localizes the failure to these choices; it does not establish that the proposal denominator alone caused it or that a restricted candidate policy would work autonomously.

No model was fitted or evaluated again, and no game was simulated. The analysis used the completed [readout study](consequence_readout_results_20260928.md): full legal-action values, all selected-action outcomes, shared labels, and the existing audits. It reproduced all retention counts and gained/lost sets exactly. The largest aggregate prediction-metric discrepancy from the earlier proposal-batch computation was 7.34e-9, below the predeclared 1e-5 tolerances for inference-grouping differences. The diagnostic process and its owner both actually exited zero in 4.60 seconds.

| Readout and selected-action group | States | Base clears | Selected clears | Gained / lost clears | New selected losses where teacher survives |
|---|---:|---:|---:|---:|---:|
| Pooled, inside original proposals | 2,592 | 1,182 | 1,203 | 50 / 29 | 0 |
| Pooled, outside original proposals | 3,416 | 1,517 | 962 | 58 / 613 | 16 |
| Relational, inside original proposals | 2,526 | 1,129 | 1,153 | 53 / 29 | 0 |
| Relational, outside original proposals | 3,482 | 1,570 | 975 | 62 / 657 | 20 |

The pooled split is +21 net clears inside and −555 outside, yielding the previously verified −534 overall. Relational is +24 inside and −595 outside, yielding −571 overall. The groups are defined by each learned head's selected action and contain different states across arms, so their raw group means are descriptive rather than a controlled comparison of the architectures.

Consequence predictions also degrade on these newly selected actions. At the same outside-selection states, compare errors on the chosen action with the mean error across the original proposals:

| Readout | Original-proposal mean log-score MSE | Selected-action log-score MSE | Original-proposal mean clear BCE | Selected-action clear BCE |
|---|---:|---:|---:|---:|
| Pooled | 0.779377 | 1.321777 | 0.341106 | 0.400268 |
| Relational | 0.714881 | 1.260811 | 0.317866 | 0.403951 |

These action distributions differ even after holding the source states fixed. The result limits the earlier prediction success to its evaluated proposal distribution; it does not show uniformly accurate consequences over every legal play. Among the selected outside actions, pooled predicts a mean clear probability of 31.40% versus 28.16% realized clears; relational predicts 36.43% versus 28.00%. Those means alone do not describe full probability calibration.

Every new fatal choice is on the last hand: 16/16 pooled and 20/20 relational when compared with a surviving teacher action. Relative to the base itself, the heads introduce 14 and 18 fatal choices where the base survives. In **9/14 pooled cases and 11/18 relational cases**, the chosen action also has a strictly lower predicted clear probability than the available base action. Predicted log-score is lower in 10/14 and 12/18 cases. These are disagreements between the preference output and the head's own immediate predictions. Recorded outcomes include stochastic realizations, so this does not prove the predictions are correct or establish a universally optimal clear-maximizing rule.

The ranking comparison makes the support change explicit. On the 6,008 eligible states, restricting argmax to the original proposals raises exact teacher agreement from 2,187 to 4,696 for pooled and from 2,137 to 4,678 for relational. Mean teacher rank changes from 13.17 to 1.29 and from 13.44 to 1.30. The corresponding conditional preference NLLs are 2.620 versus 0.630, and 2.739 versus 0.636. NLL necessarily depends on the available action set; the rank and measured outcomes provide the more direct diagnostic.

The restricted choices would clear 2,677 and 2,661 states, still below the base's 2,699. They would introduce four and seven new losses versus the base's seven. **This restricted set includes the teacher action by construction and is not a deployable policy.** It also fails the overall-clear advancement requirement. These counts are diagnostic and do not reopen either failed endpoint.

The next intervention should address preference generalization before adding a new feature pathway. The existing saved prefixes already show an earlier development preference peak, so their full-action retention is the cheaper first check and needs no additional fit. That requires a separately declared duration evaluation; the completed study's epoch-20 endpoint stays unchanged. If earlier stopping still fails retention, a controlled follow-up can change only the preference objective from sampled-proposal cross-entropy to the complete legal-play denominator, keeping features, consequence labels, state order and capacity fixed. The current diagnosis motivates that test but does not establish it as a remedy. Preserving the base preference pathway remains a separate mechanism to isolate, rather than combining changes immediately.

No policy integration, autonomous games, confirmation fit or world-model work followed this diagnostic. Long-horizon improvement remains unproven. Evidence: [pinned diagnostic result](../../eval/consequence_prediction_20260927/support_diagnostic_v1/result_v1.json), [actual owner exit](../../eval/consequence_prediction_20260927/support_diagnostic_owner_terminal_v1.json), and the per-state rows and fatal cases bound by that result.
