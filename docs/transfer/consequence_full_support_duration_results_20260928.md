# Earlier checkpoint of the full-support scorer

The declared full-support loss rule selects **epoch 10**, without another training run. It recovers another **45 blind clears** versus epoch 20, but still **fails all four retention gates**. Earlier stopping helps the new scorer; it does not resolve the remaining regression. No policy integration, autonomous evaluation or confirmation fit advances.

| Policy | Clears / 6,008 states | Early clears / 2,092 | Immediate losses | New losses where teacher survives |
|---|---:|---:|---:|---:|
| Base | 2,699 | 1,066 | 40 | 7 |
| Full-support epoch 20 | 2,528 | 954 | 41 | 8 |
| Selected full-support epoch 10 | 2,573 | 952 | 42 | 9 |

The selected-minus-20 clear difference is **+0.749 percentage points**, descriptive paired 95% interval **[+0.199, +1.353]**. This consists of 47 more later clears and two fewer early clears. Early performance does not improve: −0.096 points **[−1.060, +0.878]**. Immediate losses increase by one versus epoch 20, although that interval spans zero.

Against base, epoch 10 loses **126 net clears**, −2.097 points **[−2.631, −1.579]**, including **114 early clears**, −5.449 points **[−6.663, −4.323]**. It gains 93 and loses 219 individual clears. Both total and newly introduced immediate losses exceed base by two; the declared no-increase gates fail even though the overall loss-difference interval spans zero. Prediction passes all four baseline gates; retention passes none.

The fixed base-agreement groups locate the remaining preservation failure:

| State group | States | Base clears | Epoch-10 clears | Base new losses | Epoch-10 new losses |
|---|---:|---:|---:|---:|---:|
| Base action equals teacher | 4,156 | 2,007 | 1,854 | 0 | 4 |
| Base action differs from teacher | 1,852 | 692 | 719 | 7 | 5 |

The scorer gains 27 clears on base/teacher disagreements while losing 153 where base already matches the teacher. This is a diagnostic split, not a deployable gate: the policy does not have the teacher action at inference. Aggregate correction gains are outweighed by damage to existing successful choices, especially early in games.

This is the separate, post-hoc development study specified in the [duration plan](consequence_full_support_duration_20260928.md). It preserves the [fixed epoch-20 failure](consequence_full_support_results_20260928.md) and the original training failure/recovery provenance. It uses the same exposed 128 games. Selection and intervals are exploratory, not independent confirmation.

| Saved epoch | Full-action teacher NLL, all 6,113 states | Teacher is best legal play, 6,008 eligible states | Mean teacher rank, eligible states |
|---|---:|---:|---:|
| 0 | 5.491014 | 14 | 187.139 |
| 5 | 1.602365 | 3,791 | 7.187 |
| **10** | **1.556596** | **3,860** | **6.777** |
| 20 | 1.579065 | 3,800 | 6.179 |

Epoch 10 minimizes mean teacher NLL over every legal play across all 6,113 teacher-play states, with earlier epochs winning exact ties. Its difference from epoch 20 is −0.022469, descriptive paired 95% interval **[−0.051930, +0.007114]**. Later-state NLL improves by −0.058440 **[−0.101183, −0.016301]**; early-state NLL is +0.043739 worse **[−0.007025, +0.091973]**. Mean rank and likelihood measure different aspects of the distribution; neither guarantees a better chosen outcome.

The earlier original-proposal loss would have selected epoch 5. Measuring the complete action set changes the duration choice, reinforcing that proposal-only loss does not reliably assess this full-support policy.

At epoch 10, original-proposal log-score MSE is **0.735502**, clear BCE **0.307930**, and Brier **0.097085**. All four prediction gates pass against training-only constant and count-by-Ante baselines. There are **857** selected actions outside the original proposal sets, versus 1,058 at epoch 20. Only the declared epoch-10 selection advances to action replay; epochs 0 and 5 receive no retention replay.

New support export took 22.170 seconds and the independent saved-data audit took 10.899 seconds. The audit verified all four supports (6,526,120 legal state/action entries), original-proposal overlaps, identities, legal argmaxes, unchanged non-play decisions, exact reproduction of admitted epoch-20 summaries, and the declared selection. It reproduced the selected checkpoint's previous proposal metrics exactly. All checkpoint tensors remained unchanged; no optimization or second neural audit forward occurred.

Retention replay took 179.056 seconds and its independent saved-data audit took 2.607 seconds. All 26,045 factual decisions, 6,113 teacher plays and 128 original game terminals reproduced. Of 6,008 eligible selected plays, 5,151 outcomes reused shared labels, 590 reused identical previously audited actions and 267 required new branches. The 105 base non-play choices remained unchanged. Full identities, duplicate outcomes, parent/RNG isolation, provenance, counts, gained/lost sets and whole-game intervals passed. These are one-action interventions on teacher trajectories, not autonomous wins or long-horizon evidence.

The [selection audit](../../eval/consequence_prediction_20260927/full_support_duration_v1/verification_v1.json), [selection owner receipt](../../eval/consequence_prediction_20260927/full_support_duration_owner_terminal_v1.json), [retention audit](../../eval/consequence_prediction_20260927/retention_full_support_duration_v1/pooled_full_support/verification_v1.json), and [retention owner receipt](../../eval/consequence_prediction_20260927/full_support_duration_retention_owner_terminal_v1.json) bind these admitted results. Selection owner 39131, retention owner 34601 and all phase workers actually exited zero before admission. No new fit or optimizer step ran.

Duration and action-support mismatch now have measured contributions to the new scorer's failure, but neither fixes retention. Before another architectural change, inspect whether its consequence outputs recognize the damaging corrections, especially early and on last-hand states. That separates a consequence-prediction error from a preference-selection error without choosing a new policy on this exposed development set. The original pretraining gap and final long-horizon/fresh-win/independent-confirmation requirements remain unresolved.
