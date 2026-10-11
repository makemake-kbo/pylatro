# Removal-choice learning remains weak on the training set

The relative-consequence model ranks offers much more accurately than owned-card removals on recorded teacher-sale states. The same pattern occurs on training and development. This narrows the unresolved learning problem, but does not establish whether the limiting factor is optimization, features, target noise or their interaction.

The initial hypothesis was only partly supported: the aggregate improvement is not exclusively offer discrimination. Across all gain-labelled states, same-offer owned-card concordance improves. On the teacher-sale subset, that improvement does not clearly persist.

## Pair ordering on teacher-sale states

Concordance excludes equal true gains and gives half credit for prediction ties. Each eligible decision receives equal weight within each comparison type. The types have different eligible states, so their accuracies are descriptive, not a paired causal comparison.

| Comparison | Training control → relative | Development control → relative | Development states | Paired difference, percentage points [95% interval] |
|---|---:|---:|---:|---:|
| Different owned Jokers, same offer | 58.18% → 58.66% | 59.93% → 58.84% | 667 | -1.09 [-2.30, +0.15] |
| Different offers, same owned Joker | 90.68% → 90.72% | 90.72% → 91.10% | 433 | +0.38 [-0.80, +1.62] |
| Both different | 76.88% → 77.92% | 77.92% → 78.85% | 425 | +0.93 [+0.19, +1.69] |

Training removal comparisons reach only 58.66%, close to development 58.84%. This is a training-endpoint shortfall, beyond the previously known policy transfer gap. It is not proof of irreducible error or proof that a larger architecture is required.

## Where the error reduction comes from

For n observed pairs, summing squared difference errors over each type and dividing by n² gives that type’s exact contribution to centered MSE. All three contributions share the same labelled-state denominator and add to the previously reported error.

| Development teacher-sale comparisons | Control contribution | Relative contribution | Difference [95% interval] | Share of total MSE reduction |
|---|---:|---:|---:|---:|
| Different owned Jokers, same offer | 0.033439 | 0.032570 | -0.000869 [-0.001187, -0.000577] | 11.8% |
| Different offers, same owned Joker | 0.003806 | 0.002351 | -0.001455 [-0.001706, -0.001221] | 19.7% |
| Both different | 0.026033 | 0.020985 | -0.005048 [-0.006028, -0.004138] | 68.5% |

On these 703 development states, 88.2% of the improvement comes from comparisons that change the offer. Only 11.8% comes from comparing removals with the offer held fixed. Training shows a similar split: 12.9% comes from same-offer removal comparisons. These are point decompositions, not confidence intervals for the shares.

The relative predictor’s mean squared removal contrast is only 3.99% of the target contrast energy on development teacher sales and 4.15% on training. For offer contrasts those ratios are 39.20% and 37.62%. These ratios describe prediction amplitude; they are not explained variance or an estimate of label noise. Exact prediction ties are absent in every measured ranking group.

## The broader population gives a different picture

| All gain-labelled development states | Control concordance | Relative concordance | States | Paired difference, percentage points [95% interval] |
|---|---:|---:|---:|---:|
| Different owned Jokers, same offer | 59.21% | 60.35% | 6241 | +1.14 [+0.58, +1.70] |
| Different offers, same owned Joker | 77.20% | 75.44% | 3187 | -1.76 [-2.66, -0.84] |
| Both different | 67.18% | 67.77% | 3155 | +0.59 [+0.11, +1.09] |

In this broader population, same-offer removal comparisons account for 45.8% of the centered-error reduction, and their concordance improves by 1.14 points. Offer-only concordance falls despite lower squared error. Thus neither aggregate error reduction nor aggregate ordering is a sufficient proxy for the specific removal choice needed on teacher-sale states.

## Verification and next decision

The source and [plan](shop_pair_axes_diagnostic_plan_20261001.md) were frozen before this analysis. Independent vector and scalar enumeration agree on all 64,769 active states, with maximum discrepancy 3.34e-16. The original centered errors, overall concordance and development intervals reproduce to within 1e-12. The partition covers 1,173,327 training and 164,060 development unordered comparisons. All 101 synthetic cases passed, including missing support, singleton states, ties, independent offsets and slot permutations. The main process exited zero before admission.

The existing tested aggregator resamples all 128 complete development games 10,000 times with seed 1729; training uses all 1,000 games for point estimates. The independent scalar check validates every input row, not a second bootstrap implementation. Intervals are descriptive and marginal on an exposed panel. All comparison metrics use recorded-gain support; that oracle support must never enter policy inference.

A useful next discriminator is whether the current 64-dimensional pair features can support a substantially better relative-gain readout when its linear coefficients are solved to convergence. That would isolate an auxiliary readout optimization gap from limitations of this fixed feature basis. It differs from the earlier conditional-policy output solver: the target here is continuous consequence differences, and the policy would remain unchanged. No such probe has run yet.

The [relative endpoint remains rejected](shop_relative_results_20261001.md), with 163/716 sale matches versus base167 and five failed policy checks. No new model fit, model forward or simulator call occurred in this diagnostic. No learned win-rate or long-horizon improvement is established; all original fresh-game, retention, early-attribution and final independent-confirmation requirements remain.

[Full metrics and source pins](../../eval/consequence_prediction_20260927/shop_pair_axes_diagnostic_results_v1.json) · [Actual-exit admission](../../eval/consequence_prediction_20260927/shop_pair_axes_diagnostic_owner_terminal_v1.json)
