# Opportunity recognition improved; the action-choice gap remains

The completed [event-supervision experiment](shop_event_results_20261001.md) improved development opportunity discrimination relative to the boundary control, but still fails six of thirteen policy checks. The new saved-trace analysis narrows the interpretation: its existing policy already ranks sale situations better than the auxiliary forecast, and reweighting the existing pair scores cannot fix most of its missed teacher sales. This is an endpoint diagnosis, not a new trained policy or a win-rate result.

## The policy already contains a strong opportunity signal

Across all 7,512 active development SHOP states (703 teacher sales), compare the actual SALE-family log odds with the auxiliary maximum event forecast:

| Score | ROC AUC | Average precision |
|---|---:|---:|
| Policy SALE-family log odds | 0.941203 | 0.706296 |
| Auxiliary event maximum | 0.924390 | 0.658365 |
| Policy minus auxiliary, paired 95% interval | +0.016813 [0.009249, 0.024111] | +0.047931 [0.028880, 0.068707] |

The training direction agrees: policy AUC/AP 0.953493/0.748581 versus auxiliary 0.930537/0.684507. These are ranking metrics, not calibrated action decisions. They weaken the simple explanation that the policy fails because the auxiliary head alone knows when a sale is appropriate. Connecting that forecast directly would still require evidence of complementary information; its standalone ranking is weaker.

## Conditional card choice still transfers poorly

| Population | Conditional card choice before correction | After correction | Correct full sale after correction |
|---|---:|---:|---:|
| Original training sales | 4,345/5,380 (80.76%) | 4,349/5,380 (80.84%) | 2,034/5,380 (37.81%) |
| Development sales | 378/703 (53.77%) | 378/703 (53.77%) | 167/703 (23.76%) |

The development conditional rate has a complete-game 95% interval of [50.28%, 57.37%]. The other thirteen development SHOP sales lie outside the fixed replacement predicate and remain wrong, giving the previously reported 167/716 full sale matches. The auxiliary consequence head's conditional card choice is weaker still: 222/703 (31.58%), versus uniform 19.87%.

This is persistence of the previously documented [conditional transfer gap](conditional_context_coverage_results_20260930.md), not its first discovery. That earlier census found weak transfer even on familiar owned/offered Joker identity pairs. The earlier [five-thousand-game feature adaptation](conditional_feature_adaptation_results_20260930.md) also failed policy qualification. The present finding is that improved event coverage did not close this gap, despite updating the encoder and actor.

Correct card ranking is also insufficient for a full sale: probability assigned to selling and competing actions still matter. These conditional and full-action counts must not be conflated.

## Bounds on changing the offer weights

The current correction averages learned pair scores over offered Jokers. Any convex average is bounded by the minimum and maximum existing score. On each teacher-sale state, we give the teacher slot its maximum possible correction and every competing owned slot its minimum. This grants more freedom than the actual shared offer proposal. If the teacher still loses to another owned slot or legal LEAVE, no convex reweighting of these fixed pair scores can select it.

Of **536 missed active development sales**:

| Obstruction with the existing scores fixed | Misses |
|---|---:|
| Teacher still loses to another owned slot under optimistic bounds | 322 |
| Teacher still loses to LEAVE at its maximum pair correction | 134 |
| Union of those obstructions | **369 (68.84%)** |

The union's paired game 95% interval is [64.79%, 72.82%]. The two component counts overlap and cannot be added. The remaining 167 misses are unresolved by these bounds; they are not demonstrated repairs. Other non-sale competitors may further restrict them.

The current teacher sale loses to LEAVE in 135 development states. Its maximum existing pair correction clears LEAVE in only **one** of them, and that does not establish a correct full action. Training shows the same limited crossing potential: 6 of 833 current losses to LEAVE. Across all development teacher-sale states, the average teacher correction is 0.087276 logit units; replacing it by its maximum would add only 0.025997 on average. The average offer-to-offer spread is 0.026893.

These results argue against a pooling-only change as a sufficient repair to this checkpoint. They do not rule out a different learning trajectory after retraining. Normalized log-sum-exp pooling is bounded by the same maximum; an unnormalized sum or extra bias changes more than the offer weights and was not evaluated.

## Verification and disposition

The [plan](shop_pooling_diagnostic_plan_20261001.md) and [protocol](../../eval/consequence_prediction_20260927/shop_pooling_diagnostic_protocol_v1.json) were frozen before reading the diagnostic outcomes. Computation used only admitted saved traces and forecasts. It made no model, simulator or training calls. Synthetic simplex and edge cases checked support, ties and the joint-action identity. Every actual teacher-sale row reproduced archived NLL and the normalized joint odds; maximum NLL discrepancy was below 9e-7. A separate scalar implementation reconstructed all twelve bound/accuracy counts on all 6,083 training/development teacher-sale rows, with maximum native NLL discrepancy below 8.1e-7. Ranking and intervals reuse previously tested helpers; the separate scalar verifier did not rerun those intervals.

Development intervals use 10,000 paired resamples of all 128 complete games, seed 1729, retaining the original decision denominators. They are descriptive marginal intervals on an exposed panel, without multiplicity correction. Both diagnostic and verifier processes exited zero before admission.

The event endpoint remains rejected for gameplay and confirmation. No pooling swap, forecast threshold, calibration, extra fit or architecture change was applied. The next investigation should distinguish reusable discrimination between owned-card alternatives from state-level opportunity prediction, using the available pair labels and the previous conditional-learning results. Neither longer context, wider capacity nor a direct forecast-to-policy connection is established as the fix. The original fresh-game, retention, early-attribution and final independent-confirmation requirements remain unmet.

[Results](../../eval/consequence_prediction_20260927/shop_pooling_diagnostic_results_v1.json) · [Independent count verification](../../eval/consequence_prediction_20260927/shop_pooling_diagnostic_verification_v1.json) · [Actual-exit admission](../../eval/consequence_prediction_20260927/shop_pooling_diagnostic_owner_terminal_v1.json)
