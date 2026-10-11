# Relational scorer still has both prediction and action-ranking errors

The [completed matched experiment](consequence_relational_coverage_results_20260928.md) improved consequence prediction but lost 150 clears relative to base. Its saved-data residual diagnosis finds 86 gained clears and 236 lost clears. In 118 of those 236 lost-clear cases, its own clear head predicts a better outcome for base; in 125, its score head favors base. In the 86 gained-clear cases, those counts are ten and 12 respectively. Thus preference/consequence disagreement remains a concrete part of the failure, alongside consequence-prediction errors.

The scorer chooses by preference logit. Auxiliary score and clear estimates affect shared training features but do not directly constrain that choice. These disagreements therefore do not require a missing-context explanation. They also do not prove every lower-scoring choice is wrong over a longer horizon: this diagnosis uses original-RNG immediate outcomes on recorded teacher states.

Most of the net clear deficit still occurs outside the states' original proposal sets:

| Relational model's own selected-action group | States | Base clears | Selected clears | Net change |
| --- | ---: | ---: | ---: | ---: |
| Inside original proposals | 5,005 | 2,275 | 2,258 | −17 |
| Outside original proposals | 1,003 | 424 | 291 | −133 |

These groups depend on the relational model's choices and differ from the pooled model's groups. Outside-proposal does not mean the action structure was never seen in training. The fixed-action prediction comparison in the main report remains the appropriate architecture comparison.

There are three selected-action fatalities where base survives. All occur on the last hand, and two are outside the original proposals:

| Seed:step | Base → selected hand | Actual score, base → selected | Predicted clear, base → selected | Consequence heads favor base? |
| --- | --- | ---: | ---: | --- |
| 850000021:186 | Pair → Full House | 52,488 → 21,195 | 48.30% → 18.99% | Both |
| 850000034:40 | Flush → Pair | 504 → 303 | 14.49% → 37.64% | Neither |
| 850000064:138 | Pair → High Card | 13,296 → 5,002 | 40.00% → 19.62% | Both |

The first and third cases expose disagreement between preference and the learned consequences even at immediate survival decisions. The second case would survive a simple veto based on either learned consequence head: both predict the failing Pair more favorably. Scores and clear labels here are single original-RNG realizations, not conditional means or proof of deterministic ranking errors.

Six selected-action fatalities occur where the teacher survives, all on the last hand. The prediction reference in this diagnostic remains base for both fatal groups. Two of those six preserve exactly the base action; another changes the action while both base and selected fail. The three base-survives cases above are the new failures attributable to an override of a surviving base choice. Aggregate loss retention can pass while individual harmful overrides remain, because other changes rescue failures.

Restricting the preference argmax to the original teacher-containing proposals would yield 2,666 clears, 40 immediate losses and seven losses where the teacher survives, versus base's 2,699/40/7. This still fails clear retention and is explicitly nondeployable: the proposal set includes the teacher action. It is a support diagnostic, not a candidate policy.

The evidence does not establish that independently trainable features are the next bottleneck. Before allocating another fit, the remaining controlled question is how much of the action-ranking gap is already present on assessed training alternatives, especially last-hand choices where survival has a direct consequence. A separate training-only audit can use the saved predictions and labels without changing this failed endpoint, selecting a development threshold, or maximizing immediate score as a general policy. Any later feature-path or preference-objective intervention must change one mechanism at a time and retain the original final success bar.

The [frozen diagnostic plan](consequence_relational_coverage_residual_20260928.md) and helper were executed once, using only saved files and the standard library. It reproduced every prior audited retention group and gained/lost identity set, and all proposal metrics within a maximum absolute difference of 6.69e−9. It took 3.024 seconds with no training, inference, simulation or gate changes. Reconciliation of existing audits is not a second independent implementation of every new statistic.

- [Diagnostic result](../../eval/consequence_prediction_20260927/relational_coverage_residual_v1/result_v1.json), SHA256 `527f532fde5d56d31f2a8a31621c7a0beda732e260df1f83d81c514124ef39e3`.
- [Owner receipt](../../eval/consequence_prediction_20260927/relational_coverage_residual_owner_terminal_v1.json), actual exit `535d38`, admission `69039d`; SHA256 `06d5ae04b487432f21d5822e457405900ba9c8ff80842bbcea1d395a3f24d0cd`.

No autonomous win or long-horizon generalization improvement has been established, and no confirmation fit is warranted for this endpoint.
