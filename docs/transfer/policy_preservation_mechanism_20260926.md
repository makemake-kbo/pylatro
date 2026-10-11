# Policy preservation: verified diagnostic results

Both preservation fits completed and passed the full independent fit audit. On the existing exposed 128-game teacher-state panel, preservation reduced the degradation caused by ordinary continued fitting. **Autonomous win-rate improvement and long-horizon generalization remain unproven.** World models remain deferred.

This comparison uses 26,045 teacher decisions per policy, with two continuation replicas sharing one pretrained base. These states were excluded from fitting but had already been exposed during diagnosis. The results are descriptive mechanism evidence and cannot select training settings, checkpoints or fresh-game outcomes.

| Policy | Exact teacher matches | Agreement | Teacher NLL |
|---|---:|---:|---:|
| Base | 21,544 / 26,045 | 82.7184% | 0.639106 |
| Ordinary replica 0 | 21,477 / 26,045 | 82.4611% | 0.690329 |
| Ordinary replica 1 | 21,480 / 26,045 | 82.4726% | 0.690402 |
| Preservation replica 0 | 21,588 / 26,045 | 82.8873% | 0.643091 |
| Preservation replica 1 | 21,572 / 26,045 | 82.8259% | 0.642292 |

Pooled across replicas, preservation improves NLL by 0.047675 and teacher-action agreement by 0.390 percentage points versus ordinary fitting. Both replicas improve on both measures. NLL remains 0.003586 worse than the base; the overall agreement gain versus the base is uncertain.

All four planned uncertainty slices are shown below. Cells report the paired difference and its 95% descriptive interval. Lower NLL is better; higher agreement is better. Agreement differences are percentage points.

| Slice | Metric | Preservation − ordinary | Preservation − base | Ordinary − base |
|---|---|---:|---:|---:|
| Overall | NLL | -0.04767 [-0.05328, -0.04225] | +0.00359 [+0.00010, +0.00712] | +0.05126 [+0.04321, +0.05986] |
| Overall | Agreement (pp) | +0.390 [+0.240, +0.550] | +0.138 [-0.036, +0.313] | -0.251 [-0.480, -0.023] |
| Ante ≤ 3 | NLL | -0.04612 [-0.05438, -0.03806] | +0.00437 [-0.00119, +0.00979] | +0.05049 [+0.03802, +0.06302] |
| Ante ≤ 3 | Agreement (pp) | +0.332 [+0.052, +0.622] | -0.124 [-0.440, +0.180] | -0.455 [-0.898, -0.039] |
| Ante ≥ 4 | NLL | -0.04833 [-0.05521, -0.04186] | +0.00326 [-0.00106, +0.00758] | +0.05158 [+0.04181, +0.06190] |
| Ante ≥ 4 | Agreement (pp) | +0.414 [+0.236, +0.595] | +0.248 [+0.038, +0.453] | -0.166 [-0.426, +0.091] |
| Play decisions | NLL | -0.09055 [-0.10376, -0.07742] | -0.00707 [-0.01662, +0.00250] | +0.08348 [+0.06320, +0.10376] |
| Play decisions | Agreement (pp) | +1.006 [+0.530, +1.504] | +0.613 [+0.075, +1.129] | -0.393 [-1.045, +0.222] |

Intervals use 10,000 whole-seed bootstrap samples with seed 1729, retaining both replicas inside each cluster. All 128 seeds are included; the late slice contains 18,357 rows from 125 games and retains the three zero-row seed clusters. The early slice contains 7,688 rows. The 24 intervals are unadjusted and describe a previously exposed panel.

Across 52,090 replica-decision instances, preservation recovered 610 of the 1,107 base-correct choices lost by ordinary continuation. It also lost 416 of the 976 ordinary corrections to base errors. Overall it made 203 more teacher-matching choices than ordinary continuation. Matching a teacher action does not by itself establish better game outcomes.

This supports loss of useful behavior during continued fitting as a real, modifiable gap. It does not establish that retention alone explains the original 5,000-game pretraining transfer failure: these fits use the later 1,000-game continuation corpus. The fresh comparison must still meet the [fixed sampled, greedy and separate early-attribution requirements](policy_preservation_study_20260925.md).

The independent recount checked all 78,647 new prediction rows, 52,090 reused ordinary rows and 26,557 reproduction rows; it reproduced 80 metric groups, 32 retention groups and all 24 intervals directly from saved predictions and raw cache identities, action legality, family and Ante fields.

The first diagnostic verifier stopped before its prediction recount because the shared hash-checking helper returns a filename string and four readers called `.open()` on it. Its failed exit and original source are retained. Version 2 uses a hash-checked `Path` reader; reversing this correction and versioned filenames reproduces the original source and syntax tree exactly. Six path-integrity, retention and missing-exit tests passed before the corrected independent recount passed. No fitting, model inference, analysis, numerical threshold or scientific endpoint was rerun or changed.

The [completed fresh-game experiment](policy_preservation_results_20260926.md) now supplies the outcome test: all 9,200 episodes passed independent raw verification, but four of five scientific criteria failed. Sampled preservation wins were 27.0625%, versus 26.0625% for ordinary continuation and 27.25% for the base; its early hybrid difference was −1.25 points [−3.625, +1.125]. The diagnostic retention benefit therefore did not establish better fresh wins or early decisions. The report preserves the original controller’s unknown exit and the explicit recovery provenance.

Artifacts:

- [Frozen diagnostic protocol](../../eval/policy_preservation_20260925/mechanism_protocol_v1.json)
- [Full fit audit and actual exit](../../eval/policy_preservation_20260925/fit_audit_terminal_v1.json)
- [Raw diagnostic scan](../../eval/policy_preservation_20260925/mechanism_scan_v1/result.json)
- [All metrics, transitions and intervals](../../eval/policy_preservation_20260925/mechanism_analysis_v1.json)
- [Independent verification](../../eval/policy_preservation_20260925/mechanism_verification_v2.json)
- [Successful verification exit](../../eval/policy_preservation_20260925/mechanism_verification_terminal_v2.json)
- [Retained verifier failure](../../eval/policy_preservation_20260925/mechanism_verification_terminal_v1.json)
- [Verifier correction and source-equivalence evidence](../../eval/policy_preservation_20260925/mechanism_verifier_preparation_v2.json)
- [Six verifier regression tests](../../eval/policy_preservation_20260925/mechanism_verifier_tests_terminal_v2.json)
- [Controller-loss record](../../eval/policy_preservation_20260925/evaluation_controller_loss_v1.json)
- [Existing-worker exit observer](../../eval/policy_preservation_20260925/orphan_observer_launch_v1.json)
- [Machine-readable study record](data/policy_preservation_study_20260925.json)

- [Recovery admission and test evidence](../../eval/policy_preservation_20260925/evaluation_recovery_admission_terminal_v1.json)
- [Detached supervisor launch](../../eval/policy_preservation_20260925/evaluation_recovery_launch_v1.json)
- [Proof that original raw-results checks are unchanged](../../eval/policy_preservation_20260925/evaluation_recovery_source_equivalence_v1.json)
