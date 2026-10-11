# Counterfactual-ranking results: the generalization criteria failed

The frozen study completed both training replicas, all 23 gameplay panels, combined analysis and independent verification. **All five scientific criteria failed.** The independent verifier checked 9,200 game episodes and 1,378,963 action records. Successful execution and verification establish the integrity of this negative result; they do not establish the long-horizon goal.

The intervention added coefficient 4 times a pairwise preference loss to teacher NLL, using 2,027 measured clear disagreements at 2,022 training states. Both continuations used the same base, training cache, row streams, 4,000 updates and fixed final endpoints as the ordinary controls. The controls' checkpoints were reused, but every reported game was newly played. See the [frozen method](counterfactual_ranking_study_20260926.md) and [objective preparation](counterfactual_ranking_preparation_20260926.md).

| Confirmation mode | Base | Ordinary continuation | Ranking continuation |
|---|---:|---:|---:|
| Sampled | 224/800 (28.0000%) | 485/1600 (30.3125%) | 504/1600 (31.5000%) |
| Greedy | 179/400 (44.7500%) | 337/800 (42.1250%) | 313/800 (39.1250%) |

The separate early-decision panel recorded 426/1600 (26.6250%) for ordinary continuation and 417/1600 (26.0625%) for the hybrid. The hybrid used the ranking policy before the first pre-action Ante ≥ 4, then permanently used its paired ordinary continuation, including after any later Ante decrease.

All differences and intervals below are percentage points. Intervals use the frozen 10,000-resample paired bootstrap, seed 1729, with 400 whole-game seed clusters and all replicas and sampling streams kept together inside each cluster.

| Comparison | Difference | Paired 95% interval | Replica 0, replica 1 | Criterion |
|---|---:|---:|---:|---|
| Sampled ranking − base | +3.5000 | [+0.3750, +6.7500] | +4.6250, +2.3750 | Fail |
| Sampled ranking − ordinary | +1.1875 | [-1.5000, +3.9375] | +3.2500, -0.8750 | Fail |
| Greedy ranking − base | -5.6250 | [-10.2500, -1.0000] | -5.7500, -5.5000 | Fail |
| Greedy ranking − ordinary | -3.0000 | [-6.5000, +0.5000] | -0.7500, -5.2500 | Fail |
| Early hybrid − ordinary | -0.5625 | [-2.9375, +1.8125] | +0.3750, -1.5000 | Fail |
| Sampled ordinary − base | +2.3125 | [-0.8750, +5.5000] | +1.3750, +3.2500 | Diagnostic |
| Greedy ordinary − base | -2.6250 | [-7.0000, +1.8750] | -5.0000, -0.2500 | Diagnostic |

Sampled ranking beats the base by 3.5 points with an interval excluding zero, but misses the frozen minimum gain of 5 points. Against matched ordinary continuation, its +1.1875-point difference is unresolved and the replica directions disagree. Both sampled criteria therefore fail.

Greedy ranking is 5.625 points below the base, with its entire interval below zero and both replicas negative. Its interval against ordinary continuation also extends below the −5-point guardrail. Both greedy criteria fail. The early hybrid's −0.5625-point estimate has an interval crossing zero and opposite replica directions, so the early-decision criterion fails too. An unresolved comparison does not prove equality or harm.

The sampled-versus-greedy divergence warrants diagnosis, but does not identify its mechanism. These results do not establish a robust benefit from the ranking objective over ordinary fitting, or a win gain attributable to improved early decisions. Immediate clear preferences are not full-game action values. The original pretraining transfer gap remains only partly explained, and world-model work remains deferred.

The evidence covers two continuation replicas sharing one pretrained base, one fixed recipe and one fixed endpoint. It does not replicate independent pretraining. Sampling uses global Torch streams rather than per-game common random numbers. All seven comparisons and all 23 panel counts are retained in the [structured results](data/counterfactual_ranking_results_20260927.json); no prior study's outcomes are pooled here. The independent audit verified raw outcomes, action routes, the permanent early-policy latch, bootstrap results and every frozen criterion. The separate fit audit verified saved model/optimizer states and exposure.

Next, measure whether the objective learned its 2,027 intended training comparisons, separately for teacher-preferred and alternative-preferred actions. Saved teacher-probability predictions alone cannot recover the alternative-action margins. This diagnostic is not another training run or evidence of generalization. Subsequent changes follow the user's [one exploratory fit, final confirmation workflow](experiment_workflow.md).

Artifacts: [analysis](../../eval/counterfactual_ranking_20260926/analysis_v1.json), [independent verification](../../eval/counterfactual_ranking_20260926/independent_results_verification_v2.json), [verification exit receipt](../../eval/counterfactual_ranking_20260926/independent_results_verification_terminal_v2.json), [completed execution](../../eval/counterfactual_ranking_20260926/execution_v1/complete.json), and [current handoff](../../eval/counterfactual_ranking_20260926/RESUME.md).
