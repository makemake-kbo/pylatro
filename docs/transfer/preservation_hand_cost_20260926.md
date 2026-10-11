# Preservation: consequences across the held-out teacher-play panel

**Preservation recovers some immediate hand quality lost during ordinary continuation, but does not establish an improvement over the starting model or reliably protect fatal choices.** It clears 2,703 / 2,702 blinds versus ordinary continuation’s 2,671 / 2,683 and the base’s 2,699. Preservation − ordinary is +0.417 percentage points [0.168, 0.659]; preservation − base is +0.057 [−0.213, 0.331]. These are descriptive intervals on exposed states. New immediate losses are four / seven for preservation, five / five for ordinary continuation and seven for the base. The loss differences remain unresolved.

Of the mean 25.5 extra clears over ordinary continuation, 21.5 occur at Ante 4+. The early clear difference is +0.186 points [−0.303, 0.643], still unresolved. This adds a limited consequence benefit to the earlier imitation result, while leaving the failed fresh-win and early-attribution criteria unchanged.

This completed diagnostic compares all 6,113 teacher-play states in the existing 128-game panel. The base and both ordinary continuations reproduce their historical outcomes exactly. All six execution phases exited successfully, and independent verification reproduces all eighteen fixed comparisons. These exposed teacher-state interventions do not establish autonomous wins or long-horizon generalization.

The [completed preservation study](policy_preservation_results_20260926.md) failed four of five fresh-game criteria. Its [imitation diagnostic](policy_preservation_mechanism_20260926.md) showed less forgetting, while the [eight selected cases](preservation_selected_cases_20260926.md) retained consequential errors. The present diagnostic covers the full panel without selecting states from preservation outcomes. It uses the fixed final endpoints and saved modal actions, with no new model inference or training.

A new immediate loss means that the chosen action ends the game while the teacher action survives. The teacher clears 2,882 of these states. Every policy is measured on the same 6,113 states.

| Policy | Teacher matches | Blind clears | Missed teacher clears | Extra clears | New immediate losses | Other action family | Score ≤ half teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 4156 | 2699 | 289 | 106 | 7 | 105 | 408 |
| Ordinary 0 | 4127 | 2671 | 303 | 92 | 5 | 109 | 432 |
| Ordinary 1 | 4137 | 2683 | 290 | 91 | 5 | 96 | 435 |
| Preservation 0 | 4196 | 2703 | 280 | 101 | 4 | 107 | 416 |
| Preservation 1 | 4191 | 2702 | 281 | 101 | 7 | 102 | 417 |

All eighteen descriptive paired intervals follow. Values are percentage points; positive clear differences and negative new-loss differences favor the first policy. Replica effects are listed as 0 / 1. The intervals are unadjusted and are not new success criteria.

| States | Comparison | Blind clears, difference [95% interval] | New immediate losses, difference [95% interval] | Clear replica effects | New-loss replica effects |
|---|---|---:|---:|---:|---:|
| All plays | Ordinary − base | -0.360 [-0.697, +0.000] | -0.033 [-0.095, +0.025] | -0.458 / -0.262 | -0.033 / -0.033 |
| All plays | Preservation − base | +0.057 [-0.213, +0.331] | -0.025 [-0.089, +0.034] | +0.065 / +0.049 | -0.049 / +0.000 |
| All plays | Preservation − ordinary | +0.417 [+0.168, +0.659] | +0.008 [-0.017, +0.040] | +0.523 / +0.311 | -0.016 / +0.033 |
| Ante 1–3 | Ordinary − base | -0.116 [-0.646, +0.424] | -0.046 [-0.142, +0.000] | -0.139 / -0.093 | -0.046 / -0.046 |
| Ante 1–3 | Preservation − base | +0.070 [-0.332, +0.471] | -0.023 [-0.071, +0.000] | +0.093 / +0.046 | -0.046 / +0.000 |
| Ante 1–3 | Preservation − ordinary | +0.186 [-0.303, +0.643] | +0.023 [+0.000, +0.071] | +0.232 / +0.139 | +0.000 / +0.046 |
| Ante 4+ | Ordinary − base | -0.492 [-0.987, -0.012] | -0.025 [-0.105, +0.051] | -0.631 / -0.353 | -0.025 / -0.025 |
| Ante 4+ | Preservation − base | +0.050 [-0.317, +0.433] | -0.025 [-0.120, +0.064] | +0.050 / +0.050 | -0.050 / +0.000 |
| Ante 4+ | Preservation − ordinary | +0.543 [+0.253, +0.863] | +0.000 [-0.037, +0.038] | +0.682 / +0.404 | -0.025 / +0.025 |

The early stratum contains 2,152 states in 128 games; the late stratum contains 3,961 states in 125 games. All 128 game clusters remain in every one of the 10,000 bootstrap resamples, including games with no late states. The seed is 1729. Both continuation replicas remain inside each game cluster and share one pretrained base. The estimator is the decision-weighted ratio of sums.

The following table retains the seven fixed historical base-loss cases and every additional state where any measured policy introduces an immediate loss. The machine-readable record retains the complete raw rows.

| Seed / step | Ante | Remaining target | Teacher | Base | Ordinary 0 | Ordinary 1 | Preservation 0 | Preservation 1 |
|---|---:|---:|---|---|---|---|---|---|
| 850000015 / 28 | 2 | 800 | 1,989 clear | 795 loss | 795 loss | 795 loss | 795 loss | 795 loss |
| 850000027 / 121 | 5 | 11,000 | 14,859 clear | 5,382 loss | 5,382 loss | 5,382 loss | 5,382 loss | 5,382 loss |
| 850000034 / 104 | 5 | 13,272 | 36,011 clear | 36,011 clear | 36,011 clear | 36,011 clear | 36,011 clear | 9,291 loss |
| 850000042 / 113 | 5 | 11,000 | 14,364 clear | 8,510 loss | 14,364 clear | 14,364 clear | 14,364 clear | 14,364 clear |
| 850000064 / 138 | 5 | 11,000 | 13,296 clear | 13,296 clear | 6,313 loss | 6,313 loss | 6,313 loss | 6,313 loss |
| 850000076 / 120 | 5 | 3,217 | 7,893 clear | 2,693 loss | 2,693 loss | 7,593 clear | 7,593 clear | 7,593 clear |
| 850000077 / 77 | 4 | 5,000 | 24,308 clear | 1,280 loss | 17,952 clear | 3,840 loss | 17,952 clear | 1,280 loss |
| 850000093 / 57 | 3 | 887 | 2,760 clear | 672 loss | 2,080 clear | 1,212 clear | 2,080 clear | 672 loss |
| 850000096 / 130 | 6 | 20,000 | 102,316 clear | 9,838 loss | 9,838 loss | 9,838 loss | 9,838 loss | 9,838 loss |

Both preservation replicas retain the previously diagnosed 13,296 → 6,313 fatal regression at 850000064 / 138. The full panel also exposes a new regression at **850000034 / 104**: with one hand left and 13,272 points needed, preservation replica 1 chooses a 9,291-point Three of a Kind instead of the 36,011-point Pair selected by the teacher, base, both ordinary continuations and preservation replica 0.

In that new case, preservation replica 1 actually assigns **more probability to the correct teacher action than the base does: 16.2695% → 16.8482%**. Its teacher NLL improves from 1.815880 to 1.780926, yet another action becomes the mode and immediately loses. This is a concrete instance where better likelihood of the correct label coexists with a worse selected action. It does not show that likelihood training cannot work, or identify a unique architectural cause. It does show why average imitation improvement is insufficient evidence that critical action rankings are protected.

The next diagnosis needs to distinguish insufficient training coverage of these consequential comparisons from failure to fit comparisons already present, and then measure the effects on student-visited states. Repeating broad continuation or interpreting preservation as a sufficient remedy is not supported by these results. The full original pretraining gap remains only partly explained.

The archived simulator, observations, state-copy rules, RNG handling and card-identity allocator are unchanged. The new full base control had to pass its independent audit before endpoint replay began. Both replays reproduce every one of the 26,045 factual decisions and all 6,113 historical base outcomes. Endpoint replay additionally reproduces both ordinary controls’ full action, probability and transition records. It executes 9,018 unique action branches; the independent raw audit passes 195,970 checks.

Independent verification recomputes raw transition predicates, rates, replica effects, all eighteen intervals using a frequency bootstrap, score partitions and every fatal case. It verifies accounting and controlled replay; it does not implement a second simulator. The analyzer and verifier are byte-identical to the previously corrected versions. The adaptations only map preservation endpoints and add historical ordinary-control checks.

Individual choices on teacher paths do not measure full-game counterfactual value or errors on student-visited states. A non-clearing discard can be useful later and has no comparable immediate hand score. Only modal actions are replayed; the full sampled policy is not measured. The original long-horizon completion criteria remain unmet. World models remain deferred.

Artifacts:

- [Frozen protocol](../../eval/preservation_hand_cost_20260926/protocol_v1.json)
- [Source adaptation proof](../../eval/preservation_hand_cost_20260926/source_equivalence_v1.json)
- [Raw replay](../../eval/preservation_hand_cost_20260926/panel_v1/result.json)
- [All fixed comparisons](../../eval/preservation_hand_cost_20260926/analysis_v2.json)
- [Independent verification](../../eval/preservation_hand_cost_20260926/verification_v2.json)
- [All actual phase exits](../../eval/preservation_hand_cost_20260926/execution_v1/complete.json)
- [Machine-readable results](data/preservation_hand_cost_20260926.json)
