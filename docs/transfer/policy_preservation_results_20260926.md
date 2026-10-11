# Policy preservation: completed fresh-game results

**The fixed experiment failed four of its five scientific criteria.** All 23 panels, 9,200 played game episodes and 1,396,901 action records passed independent verification. Preservation reduced teacher-state forgetting, but did not establish better fresh-game wins or better early decisions. The long-horizon goal remains unproven; world models remain deferred.

The [frozen study](policy_preservation_study_20260925.md) adds 1.0 times full forward KL from the starting policy to teacher-action NLL. Both treatment replicas trained all actor, embedding and Transformer parameters for 4,000 updates on the existing 1,000-game corpus. They share one pretrained base; the matched ordinary endpoints were reused after a full training-identity audit. Every reported gameplay outcome was newly played. No checkpoint, mode, stream or panel was selected from these results.

Confirmation used 400 environment seeds, with each policy played greedily and on two sampling streams. Each cell below is wins out of 400 games.

| Policy | Greedy | Sample 0 | Sample 1 |
|---|---:|---:|---:|
| Base | 156 | 102 | 116 |
| Ordinary replica 0 | 150 | 89 | 100 |
| Preservation replica 0 | 154 | 109 | 101 |
| Ordinary replica 1 | 135 | 112 | 116 |
| Preservation replica 1 | 151 | 104 | 119 |

Pooled sampled wins were **433/1,600 (27.0625%)** for preservation, **417/1,600 (26.0625%)** for ordinary continuation, and **218/800 (27.25%)** for the shared base. Greedy wins were 305/800 (38.125%), 285/800 (35.625%), and 156/400 (39%), respectively. The shared base is counted once, not as two independent controls.

All seven fixed contrasts are reported below. Differences and intervals are percentage points.

| Comparison | Gain | Paired 95% interval | Replica 0 gain | Replica 1 gain |
|---|---:|---:|---:|---:|
| Sampled: preservation − ordinary | +1.0000 | [-1.6266, +3.6875] | +2.6250 | -0.6250 |
| Sampled: preservation − base | -0.1875 | [-3.2500, +3.0000] | -1.0000 | +0.6250 |
| Sampled: ordinary − base | -1.1875 | [-4.3141, +2.0625] | -3.6250 | +1.2500 |
| Greedy: preservation − ordinary | +2.5000 | [-0.7500, +5.7500] | +1.0000 | +4.0000 |
| Greedy: preservation − base | -0.8750 | [-5.0000, +3.3750] | -0.5000 | -1.2500 |
| Greedy: ordinary − base | -3.3750 | [-7.7500, +1.1250] | -1.5000 | -5.2500 |
| Early hybrid − ordinary | -1.2500 | [-3.6250, +1.1250] | -2.1250 | -0.3750 |

Intervals use 10,000 whole-environment-seed bootstrap resamples with seed 1729. Both replicas and sampling streams remain inside each seed cluster. There are 400 confirmation seed clusters and a separate 400-seed early-attribution bank. Sampling uses fixed global Torch streams, not per-game common random numbers.

The early hybrid uses preservation before the first pre-action Ante ≥ 4, then permanently uses the paired ordinary policy. Each early cell is wins out of 400 games:

| Replica | Ordinary sample 0 | Ordinary sample 1 | Hybrid sample 0 | Hybrid sample 1 |
|---|---:|---:|---:|---:|
| 0 | 105 | 123 | 96 | 115 |
| 1 | 113 | 108 | 109 | 109 |

The hybrid won **429/1,600 (26.8125%)**, versus **449/1,600 (28.0625%)** for ordinary continuation. Its −1.25-point difference has interval [−3.625, +1.125]; both replica point estimates are negative. This does not establish either a benefit or a statistically resolved harmful effect.

| Fixed scientific criterion | Result | Reason |
|---|---|---|
| Sampled gain against ordinary ≥ 5 pp; lower bound > 0; both replicas positive | Fail | +1.0 pp, lower bound −1.6266; replica 1 negative |
| Sampled gain against base ≥ 5 pp; lower bound > 0; both replicas positive | Fail | −0.1875 pp, lower bound −3.25; replica 0 negative |
| Greedy lower bound against ordinary > −5 pp | Pass | Lower bound −0.75 pp |
| Greedy lower bound against base > −5 pp | Fail | Lower bound exactly −5 pp; the rule is strictly above −5 |
| Separate early-hybrid lower bound > 0; both replicas positive | Fail | Lower bound −3.625 pp; both replicas negative |

The [verified teacher-state diagnostic](policy_preservation_mechanism_20260926.md) still establishes less degradation than ordinary fitting: teacher NLL improves by 0.047675 and exact agreement by 0.390 points. That intervention did not translate into a demonstrated fresh-win or early-decision gain. The evidence supports a modifiable retention gap, but does not establish retention as a sufficient explanation or remedy for the remaining teacher–student gap. It does not prove that every preservation objective is ineffective.

These are later 1,000-game continuations, not a reconstruction of the original 5,000-game pretraining. Fitting limitations, sparse coverage, teacher information differences and errors on student-visited states remain unresolved. This result does not justify moving to world models or claiming an architectural bottleneck.

The original evaluation controller disappeared after 18 completed panels; its exit remains unknown. The [recorded recovery](../../eval/policy_preservation_20260925/evaluation_recovery_protocol_v1.md) preserved those exits, captured the nineteenth worker’s actual OS exit, and ran only the four previously unstarted panels and original analyzer under the original deadlines. A detached parent recorded the recovery coordinator’s actual exit 0. The queued frozen verifier then exited 0 after checking all 24 phase receipts, 9,200 outcomes, 1,396,901 action records, permanent early routing, seed/checkpoint/settings identity, seven comparisons and five gates. It independently recounts saved data; it does not rerun neural inference or engine execution. The separate full fit audit checks model and optimizer tensors.

Artifacts:

- [Frozen scientific manifest](../../eval/policy_preservation_20260925/execution_manifest_v1.json)
- [All raw panel counts and primary analysis](../../eval/policy_preservation_20260925/analysis_v1.json)
- [Independent raw-results verification](../../eval/policy_preservation_20260925/independent_results_verification_v2.json)
- [Actual verifier exit](../../eval/policy_preservation_20260925/independent_results_verification_terminal_v2.json)
- [Actual recovery coordinator exit](../../eval/policy_preservation_20260925/evaluation_recovery_terminal_v1.json)
- [Complete fit audit](../../eval/policy_preservation_20260925/fit_audit_v1.json)
- [Machine-readable study record](data/policy_preservation_study_20260925.json)
