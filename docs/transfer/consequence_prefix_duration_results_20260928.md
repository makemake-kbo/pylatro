# Earlier checkpoints improve some choices, but neither retains the base

The saved pooled epoch-5 and relational epoch-10 checkpoints **both fail all four retention gates**, despite passing the consequence-prediction gates. Earlier stopping helps the relational head, but does not make either scorer eligible for policy integration. No new model was trained, no autonomous policy games were run, and no confirmation data were used.

This was the separate, post-hoc [duration evaluation](consequence_prefix_duration_20260928.md) declared after the [fixed-endpoint experiment](consequence_readout_results_20260928.md) and [support diagnosis](consequence_support_results_20260928.md). The epochs were selected from already audited development proposal-preference losses before measuring their full legal choices. These 128 development games were already exposed; this is not fresh confirmation.

| Policy | Clears / 6,008 eligible states | Early clears / 2,092 | Immediate losses | New losses where teacher survives |
|---|---:|---:|---:|---:|
| Base | 2,699 | 1,066 | 40 | 7 |
| Pooled, epoch 5 | 2,179 | 746 | 45 | 12 |
| Pooled, epoch 20 | 2,165 | 732 | 49 | 16 |
| Relational, epoch 10 | 2,264 | 783 | 47 | 14 |
| Relational, epoch 20 | 2,128 | 720 | 53 | 20 |

Relational epoch 10 gains **136 clears** over its epoch-20 endpoint: +2.264 percentage points, paired 95% interval **[+1.426, +3.141]**. Its early gain is 63 clears, +3.011 points **[+1.739, +4.306]**. Duration therefore contributes to the relational head's regression. Pooled gains only 14 clears, +0.233 points **[−0.528, +0.999]**, leaving that gain unresolved.

Both prefixes remain substantially below the base: pooled −8.655 clear-rate points **[−9.889, −7.506]**, relational −7.240 points **[−8.164, −6.330]**. Each has more total and newly introduced immediate losses. Their loss-rate differences have intervals spanning zero, but the declared retention gates require no observed increase; both fail. These are single-action outcomes at teacher-path states, **not game win rates**. Intervals resample whole games, not independent training replicas.

The prediction audits exactly reproduce the earlier saved-array metrics:

| Selected checkpoint | Log-score MSE | Clear BCE | Clear Brier | Proposal preference NLL |
|---|---:|---:|---:|---:|
| Pooled epoch 5 | 0.739436 | 0.322194 | 0.101780 | 0.540208 |
| Relational epoch 10 | 0.657006 | 0.295380 | 0.093179 | 0.556550 |

Both pass all four paired prediction comparisons against the training-only constant and count-by-Ante baselines. Those losses cover the original proposals; they do not establish accuracy across the entire action space. The prefixes still choose outside those proposals in 2,970 pooled and 3,024 relational states. Every such chosen action now has a verified recorded-continuation outcome.

Each arm reproduced all 26,045 factual observations/teacher decisions and 128 game terminals, scored all 1,631,530 legal plays across 6,113 teacher-play states, and preserved 105 base non-play decisions. Each required 1,531 new branch simulations; the remaining selected outcomes were reused from identical admitted state/action records. Duplicate outcomes agree exactly. Independent audits checked legal support, argmax, cached-prefix prediction overlap, identities, outcome provenance, counts, changed sets, prior means, gates and intervals. They did not perform a second neural forward or simulator replay.

Pooled measurement/audit took 195.029/6.299 seconds; relational took 197.818/6.477. All four phases and the actual owner exited zero. The [completion record](../../eval/consequence_prediction_20260927/prefix_duration_complete_v1.json) selects no candidate. The [owner receipt](../../eval/consequence_prediction_20260927/prefix_duration_owner_terminal_v1.json) binds those exits and artifacts; [pooled](../../eval/consequence_prediction_20260927/prefix_duration_v1/pooled/verification_v1.json) and [relational](../../eval/consequence_prediction_20260927/prefix_duration_v1/relational/verification_v1.json) audits contain the full statistics.

This closes the saved-prefix intervention. The next controlled change is the pooled scorer's preference-loss denominator: include every legal play during training while keeping its architecture, frozen features, consequence labels, initialization, state order and exposure fixed. Run one exploratory fit after checking the full-support loss and gradient implementation. Do not combine this with new features, base-preservation penalties or data diversity. The existing subset policy already normalizes over full support, so this proposed repair addresses the new scorer's training/evaluation mismatch; it is not an explanation of the original pretraining gap. No new win-rate or long-horizon generalization benefit has been demonstrated.
