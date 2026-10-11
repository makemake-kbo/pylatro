# Ranking policy: complete held-out action consequences

Ranking improves immediate clears over ordinary continuation, but its gains over the base are almost entirely offset by losses of previously correct behavior. Each replica gains **50 net clears** in the targeted preference cases, while losing **54 / 42 net clears** where the base already matched the teacher. Overall ranking clears **2,695 / 2,707** states versus base **2,699**, with no resolved advantage over the base.

This diagnostic replays saved modal choices on all **6,113 teacher-play states in the existing 128-game held-out panel**, including the 339 preference cases and every other teacher-play state. The same archived simulator reproduces all 26,045 factual teacher decisions. There is no new fit, neural inference, autonomous policy rollout, or untouched confirmation evidence. The [fresh-game ranking experiment](counterfactual_ranking_results_20260927.md) still failed all five generalization criteria.

The recorded teacher clears 2,882 of these states. Counts below distinguish immediate clears, missed teacher clears, additional clears, and newly introduced immediate losses. A different action can be better than the teacher; exact agreement is not the objective.


| Policy | Clears | Missed teacher clears | Extra clears | New immediate losses | Exact teacher matches | Other action family |
|---|---:|---:|---:|---:|---:|---:|
| Base | 2699 | 289 | 106 | 7 | 4156 | 105 |
| Ordinary 0 | 2671 | 303 | 92 | 5 | 4127 | 109 |
| Ordinary 1 | 2683 | 290 | 91 | 5 | 4137 | 96 |
| Ranking 0 | 2695 | 294 | 107 | 5 | 4108 | 93 |
| Ranking 1 | 2707 | 291 | 116 | 6 | 4097 | 95 |

All fixed comparisons follow. Differences and intervals are percentage points; positive new-loss differences mean more newly introduced losses. The final column gives the two replica clear-rate effects separately.

| Stratum | Comparison | Clear difference [paired 95% interval] | New-loss difference [paired 95% interval] | Clear effects, replicas 0 / 1 |
|---|---|---:|---:|---:|
| all | ordinary − base | -0.3599 [-0.6965, +0.0000] | -0.0327 [-0.0951, +0.0246] | -0.4580 / -0.2617 |
| all | ranking − base | +0.0327 [-0.4005, +0.4804] | -0.0245 [-0.0890, +0.0337] | -0.0654 / +0.1309 |
| all | ranking − ordinary | +0.3926 [+0.0557, +0.7828] | +0.0082 [+0.0000, +0.0252] | +0.3926 / +0.3926 |
| ante<=3 | ordinary − base | -0.1162 [-0.6464, +0.4241] | -0.0465 [-0.1421, +0.0000] | -0.1394 / -0.0929 |
| ante<=3 | ranking − base | +0.1626 [-0.5206, +0.9247] | -0.0465 [-0.1421, +0.0000] | -0.0929 / +0.4182 |
| ante<=3 | ranking − ordinary | +0.2788 [-0.1658, +0.8006] | +0.0000 [+0.0000, +0.0000] | +0.0465 / +0.5112 |
| ante>=4 | ordinary − base | -0.4923 [-0.9865, -0.0124] | -0.0252 [-0.1047, +0.0514] | -0.6312 / -0.3534 |
| ante>=4 | ranking − base | -0.0379 [-0.5489, +0.4864] | -0.0126 [-0.0952, +0.0724] | -0.0505 / -0.0252 |
| ante>=4 | ranking − ordinary | +0.4544 [+0.0634, +0.8859] | +0.0126 [+0.0000, +0.0395] | +0.5807 / +0.3282 |

Early states (Ante ≤ 3) number **2,152**; late states number **3,961**. Intervals use 10,000 whole-seed bootstrap resamples, seed 1729, all 128 game clusters including zero-count seeds, and both replicas inside each cluster. The estimator is the decision-weighted ratio of sums. Every one of the 18 unadjusted descriptive intervals was reproduced independently. These intervals describe already-exposed teacher states, not fresh-game success.

The predeclared partition below separates the fixed preference cases, the states where the base exactly matched the teacher, and the remaining base disagreements. Each replica's gained / lost / net entries compare ranking with its matched ordinary continuation. All partitions sum exactly to the full-panel effects.

| Cohort | States | R0 − O0: gained / lost / net clears | R1 − O1: gained / lost / net clears | Net clears versus base, R0 / R1 |
|---|---:|---:|---:|---:|
| base equals teacher | 4156 | 22 / 22 / +0 | 28 / 20 / +8 | -54 / -42 |
| other base disagreement | 1618 | 11 / 7 / +4 | 8 / 6 / +2 | +0 / +0 |
| preference pairs | 339 | 26 / 6 / +20 | 23 / 9 / +14 | +50 / +50 |

Within the 339 preference cases, we also distinguish choosing the known clearing action, the known non-clearing comparison, and a third action. These are actual replay outcomes, filling the uncertainty left by the [preference-only measurement](ranking_preference_transfer_20260927.md).

| Policy | Preferred choices | Comparison choices | Third choices | Third choices that clear | Third choices causing a new immediate loss | Third choices from another family |
|---|---:|---:|---:|---:|---:|---:|
| Base | 106 | 233 | 0 | 0 | 0 | 0 |
| Ordinary 0 | 114 | 169 | 56 | 22 | 0 | 0 |
| Ordinary 1 | 126 | 164 | 49 | 16 | 1 | 1 |
| Ranking 0 | 128 | 148 | 63 | 28 | 0 | 0 |
| Ranking 1 | 136 | 149 | 54 | 20 | 1 | 0 |

Against matched ordinary continuation, ranking gains **24 clears per replica**, or **+0.3926 points [0.0557, 0.7828]**. Against the base, the pooled effect is only **+0.0327 points [−0.4005, 0.4804]**. In the remaining 1,618 base-disagreement states, net changes against the base are zero for both replicas. This gives an exact accounting of where the targeted gains are offset on these teacher states; it does not prove the cause of the autonomous win decline.

Early ranking-minus-ordinary gains are **1 / 11 clears**, versus **23 / 13** late. The early pooled difference, **+0.2788 points [−0.1658, 0.8006]**, remains unresolved. Ranking introduces **5 / 6 immediate losses**, versus ordinary **5 / 5** and base **7**. Both ranking replicas retain the previously diagnosed Needle failure at 850000064/138. Replica 1 also makes the previously diagnosed Card Sharp error at 850000034/104: the teacher's 36,011-point Pair clears a 13,272 target, while the model's 9,291-point Three of a Kind loses. Every immediate-loss case is retained in the structured results.

Third choices sometimes help: **28 of 63** and **20 of 54** clear, so preferred-action mode counts alone understate actual clears in the selected population. The remaining **35 / 34** third choices do not clear immediately; one replica-1 third choice introduces an immediate loss. A non-clearing choice can still have future value when another hand remains.

A supplemental descriptive recount applies the same complete partition to the already completed [preservation study](preservation_hand_cost_20260926.md). Preservation loses **22 / 23 net clears** on base-equals-teacher states, compared with ranking's **54 / 42**, but gains only **29 / 26** in the preference cases, compared with ranking's **50 / 50**. This recount was performed after the ranking replay and is not a new independent experiment. All 6,113 factual states and shared controls match exactly; the [comparison artifact](../../eval/ranking_hand_cost_20260927/preservation_comparison_v1.json) retains its provenance. The two objectives address different parts of the observed gap, but their separate effects do not predict the result of combining them.

The same-state controls are exact: every factual observation, teacher transition, base transition and both ordinary transitions reproduce their historical records. State copies retain controller aliases; branches restore RNG/state and the playing-card identity allocator. The previous independently verified full base control was reused by explicit hashes and successful actual-exit receipts. The current replay still checks all 6,113 base outcomes again; it does not merely copy them.

The panel evaluates **9,328 distinct action branches**. The independent raw audit passes **196,284 checks**, followed by independent transition-predicate and bootstrap verification. A separate raw recount verifies the complete cohort partition and every third-choice outcome. Replay, scoring and state-handling routines are unchanged; the analyzer and main result verifier are byte-identical to the prior verified versions. The verifier checks controlled replay and accounting rather than providing a second simulator.


One-step outcomes on teacher trajectories do not measure the states reached by an autonomous student or eventual game value. Not clearing immediately can be a useful discard or resource-saving play; immediate loss has a stricter meaning. Only modal actions are replayed, not the full sampled policy. Both continuations share a pretrained base. These limits prevent attributing the fresh-game win differences to this diagnostic alone.

The next exploratory candidate will test the existing comparison loss together with the existing base-policy preservation loss, retaining teacher NLL and the matched training recipe. The concrete hypothesis is that useful corrections can be learned while retaining more already-correct behavior. It remains a hypothesis: neither prior objective established the required fresh-game and early-decision improvement. Prepare and check the combined objective, then fit **one exploratory model**. Evaluate development evidence before considering an independent final confirmation run.

Future changes follow [one exploratory fit per change, final independent confirmation after candidate freeze](experiment_workflow.md). World models remain deferred. The long-horizon objective remains active and unproven.

Artifacts: [structured results](data/ranking_hand_cost_20260927.json), [frozen protocol](../../eval/ranking_hand_cost_20260927/protocol_v1.json), [source comparison](../../eval/ranking_hand_cost_20260927/source_equivalence_v1.json), [raw replay](../../eval/ranking_hand_cost_20260927/panel_v1/result.json), [all comparisons](../../eval/ranking_hand_cost_20260927/analysis_v2.json), [independent verification](../../eval/ranking_hand_cost_20260927/verification_v2.json), [cohort verification](../../eval/ranking_hand_cost_20260927/cohort_verification_v1.json), [actual phase exits](../../eval/ranking_hand_cost_20260927/execution_v1/complete.json), and [handoff](../../eval/ranking_hand_cost_20260927/RESUME.md).
