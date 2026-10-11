# Continuation hand consequences on fixed teacher states

The continuation gap is **mixed at the level of immediate hand outcomes**. Both ordinary continuations clear fewer blinds than the base on the same 6,113 teacher-play states, but introduce fewer immediate losses. Weighting has worse held-out imitation than ordinary fitting yet slightly better hand-consequence point estimates. Neither weighting comparison establishes a reliable advantage over the base or ordinary fitting on the two primary measures. The long-horizon generalization goal remains unproven; world-model work remains deferred.

This follows the [held-out imitation diagnosis](continuation_transfer_diagnostic_20260925.md) and the [completed fresh-game study](consequence_weighting_study_20260924.md). It uses the already exposed 128-game panel, unchanged checkpoints and saved mode actions. Each choice is executed from an identical copied teacher state. These are separate one-step interventions, not games played by those policies.

All policies see 6,113 play states. The teacher clears 2,882 of them. A new immediate loss means the chosen action ends the game while the teacher action survives; it is distinct from merely failing to clear the current blind.

| Policy | Teacher-action matches | Blind clears | Missed teacher clears | Extra clears | New immediate losses | Other action family | Score ≤ half teacher |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 4156 | 2699 | 289 | 106 | 7 | 105 | 408 |
| Ordinary 0 | 4127 | 2671 | 303 | 92 | 5 | 109 | 432 |
| Ordinary 1 | 4137 | 2683 | 290 | 91 | 5 | 96 | 435 |
| Weighted 0 | 4037 | 2687 | 301 | 106 | 3 | 92 | 456 |
| Weighted 1 | 4076 | 2692 | 296 | 106 | 5 | 81 | 446 |

Ordinary fitting averages 22 fewer immediate clears than the base: −0.360 percentage points [−0.697, 0.000]. Most of that difference is late: 19.5 fewer clears at Ante 4+, −0.492 points [−0.987, −0.012]. The latter is an unadjusted diagnostic interval among eighteen comparisons, not fresh confirmation. Both ordinary replicas have five new immediate losses versus seven for the base; that pooled difference remains unresolved.

Weighting averages 12.5 more clears than ordinary fitting, +0.204 points [−0.114, +0.563], despite matching fewer teacher actions. Its new-loss counts are three and five. The aggregate evidence therefore does not support equating its worse imitation scores with worse immediate hand outcomes. It also does not establish a consequential improvement or explain the earlier sampled-win results.

All eighteen predeclared intervals appear below. Values are percentage-point differences; positive clear differences and negative new-loss differences favor the first policy. Replica effects are listed as 0 / 1.

| States | Comparison | Blind clears, difference [95% interval] | New immediate losses, difference [95% interval] | Clear replica effects | New-loss replica effects |
|---|---|---:|---:|---:|---:|
| All plays | Ordinary − base | -0.360 [-0.697, +0.000] | -0.033 [-0.095, +0.025] | -0.458 / -0.262 | -0.033 / -0.033 |
| All plays | Weighted − base | -0.155 [-0.551, +0.256] | -0.049 [-0.119, +0.016] | -0.196 / -0.115 | -0.065 / -0.033 |
| All plays | Weighted − ordinary | +0.204 [-0.114, +0.563] | -0.016 [-0.041, +0.000] | +0.262 / +0.147 | -0.033 / +0.000 |
| Ante 1–3 | Ordinary − base | -0.116 [-0.646, +0.424] | -0.046 [-0.142, +0.000] | -0.139 / -0.093 | -0.046 / -0.046 |
| Ante 1–3 | Weighted − base | -0.093 [-0.804, +0.674] | -0.070 [-0.186, +0.000] | +0.093 / -0.279 | -0.093 / -0.046 |
| Ante 1–3 | Weighted − ordinary | +0.023 [-0.443, +0.539] | -0.023 [-0.072, +0.000] | +0.232 / -0.186 | -0.046 / +0.000 |
| Ante 4+ | Ordinary − base | -0.492 [-0.987, -0.012] | -0.025 [-0.105, +0.051] | -0.631 / -0.353 | -0.025 / -0.025 |
| Ante 4+ | Weighted − base | -0.189 [-0.651, +0.283] | -0.038 [-0.130, +0.050] | -0.353 / -0.025 | -0.050 / -0.025 |
| Ante 4+ | Weighted − ordinary | +0.303 [-0.026, +0.661] | -0.013 [-0.039, +0.000] | +0.278 / +0.328 | -0.025 / +0.000 |

The early stratum contains 2,152 states in all 128 games. The late stratum contains 3,961 states in 125 games. Every bootstrap draw samples from all 128 games, retaining zero-support games. Both continuation replicas remain inside each game cluster, with the shared base counted once. Estimates use decision-weighted ratios of sums, 10,000 resamples and fixed seed 1729. All intervals are descriptive, unadjusted percentile intervals; none is a new acceptance gate.

The seven historically identified base losses were fixed before this replay. The table reports the actual hand score and whether the blind clears. Both ordinary replicas fix three of the seven; weighted replica 0 fixes five, while weighted replica 1 fixes three.

| Seed / step | Ante | Remaining target | Teacher | Base | Ordinary 0 | Ordinary 1 | Weighted 0 | Weighted 1 |
|---|---:|---:|---:|---|---|---|---|---|
| 850000015 / 28 | 2 | 800 | 1,989 | 795 loss | 795 loss | 795 loss | 1,989 clear | 795 loss |
| 850000027 / 121 | 5 | 11,000 | 14,859 | 5,382 loss | 5,382 loss | 5,382 loss | 5,382 loss | 5,382 loss |
| 850000042 / 113 | 5 | 11,000 | 14,364 | 8,510 loss | 14,364 clear | 14,364 clear | 14,364 clear | 14,364 clear |
| 850000076 / 120 | 5 | 3,217 | 7,893 | 2,693 loss | 2,693 loss | 7,593 clear | 7,593 clear | 7,593 clear |
| 850000077 / 77 | 4 | 5,000 | 24,308 | 1,280 loss | 17,952 clear | 3,840 loss | 17,952 clear | 3,840 loss |
| 850000093 / 57 | 3 | 887 | 2,760 | 672 loss | 2,080 clear | 1,212 clear | 2,080 clear | 2,080 clear |
| 850000096 / 130 | 6 | 20,000 | 102,316 | 9,838 loss | 9,838 loss | 9,838 loss | 9,838 loss | 9,838 loss |

The previously diagnosed 795-versus-800 failure at seed 850000015 / step 28 is corrected only by weighted replica 0, which selects the 1,989-point teacher hand. The other three continuations still choose 795. This specific correction does not replicate.

All four continuations also introduce the same new immediate loss at **850000064 / step 138**. With one hand left and 11,000 points remaining at Ante 5, the base selects the teacher’s **13,296-point Pair** and clears. Every continuation selects the same **6,313-point High Card** and loses. Thus their lower total loss counts combine repaired old errors with a newly introduced shared error; they do not simply remove the old failures. The [structured data](data/continuation_hand_cost_20260925.json) retains every new-loss case and all score partitions.

The [frozen protocol](../../eval/continuation_hand_cost_20260925/protocol_v1.json) required a full base control before any new endpoint consequences were graded. Both complete replays reproduce all 26,045 factual decisions, all fourteen observation fields, full teacher transitions, recorded terminal outcomes, and all 6,113 historical base outcomes. The panel executes 9,437 unique action branches, preserving runner state aliases, RNG state and the global card-identity counter. Same-action branches share the same full outcome. Non-play choices are actually stepped, but have no comparable hand score; a discard that does not clear immediately may still be useful over later decisions.

The [independent raw audit](../../eval/continuation_hand_cost_20260925/panel_verification_v1.json) passes 178,039 checks. The [independent analysis verification](../../eval/continuation_hand_cost_20260925/verification_v2.json) reproduces transition predicates, summaries, both replica effects, all eighteen intervals through a frequency-matrix bootstrap, direct score comparisons and every fatal case. Four previously completed numerical tests validate the reused statistical implementations. All jobs exited and their owned sleep guards ended. The [terminal receipt](../../eval/continuation_hand_cost_20260925/verification_terminal_v2.json) binds the completed artifacts. This is independent accounting and controlled replay, not a second engine implementation.

The first analysis verifier rejected the late-stratum support count: building a denominator list through a defaultdict had inserted three absent games, also diluting its equal-game means. The [versioned correction](../../eval/continuation_hand_cost_20260925/analysis_preparation_v2.json) uses a nonmutating lookup and passes a zero-support regression check. The original analysis and failed verification are retained. Raw outcomes, per-decision counts, replica effects and all eighteen primary intervals are exactly unchanged; version 2 is authoritative.

The [completed checkpoint and coverage trace](continuation_regression_trace_20260925.md) finds that the learned subset correction overturns a still Pair-favoring reference by update 1,000 in all four continuations. The broader training motif fits better while its small exposed holdout regresses; the critical Red-seal Bonus refinement is absent from current training. The next causal check separates correction-head parameter changes from changes to its encoder inputs. These results do not yet explain how errors compound on student-visited states or how the full sampled action distribution affects wins. No new training, neural inference, game seeds, architectural change or world model was introduced.
