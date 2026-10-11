# Better survival-pair rankings still coexist with worse chosen actions

The completed standalone survival-ranking scorer orders the evaluated survivor/fatal pairs better than the immutable base policy on both training and development data. On the original held-out panel, its state-balanced pair error is **9.4494%**, versus **12.4355%** for base. Nevertheless, the [completed retention replay](consequence_survival_ranking_results_20260928.md) found **2,537 clears versus base 2,699**, and **944 early clears versus 1,066**.

This does not support the simple prediction that the base has better survival-pair ordering. It reinforces the distinction between learning a relation on assessed alternatives and choosing a useful action from the entire legal set. The [fixed-base residual experiment](consequence_fixed_base_residual_20260928.md) tests whether the full base distribution helps preserve useful choices while allowing learned corrections; this diagnostic does not establish that it will work.

| Population | Pairs / states | Base pair error | Standalone pair error | Base / standalone wrong pairs |
|---|---:|---:|---:|---:|
| Training, all | 3,396 / 430 | 16.6221% | 10.6118% | 583 / 382 |
| Development, all | 318 / 56 | 12.4355% | 9.4494% | 45 / 34 |
| Development, early | 113 / 20 | 10.2778% | 8.3333% | 13 / 10 |
| Development, nonteacher survivor | 153 / 39 | 22.8632% | 18.3761% | 33 / 27 |

Errors average each state's pair error with equal state weight; raw wrong-pair counts use unequal numbers of pairs per state. There are no ties. Base is better on four development pairs and worse on fifteen; the standalone score's relation is also imperfect. These are descriptive comparisons on already-used development data, with no new confidence interval or independent-fit claim.

The [standard-library diagnostic](../../eval/consequence_prediction_20260927/fixed_base_pair_baseline_v1.py) used admitted full base conditional log probabilities and the prior scored-pair artifacts. It checked each cache index, source seed, step and legal global action, then recounted saved rows. It introduced no model forward, labels, simulation, optimizer step or gate change. The [result](../../eval/consequence_prediction_20260927/fixed_base_pair_baseline_v1/result_v1.json) includes all early/later, original/extra and teacher/nonteacher groups. Its actual owner exited zero in tool chunk `aac8c1`; the [owner receipt](../../eval/consequence_prediction_20260927/fixed_base_pair_baseline_owner_terminal_v1.json) was admitted in `a7709f`. Runtime was 0.545 seconds.

The residual fit was already frozen and running when this diagnostic completed. Its endpoint, coefficient, labels and decision rules remain unchanged. No new candidate outcome, autonomous win-rate improvement or long-horizon generalization is established here.
