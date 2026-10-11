# Encoder and scoring-head rollbacks across the full exposed panel

**The encoder-related held-out likelihood regression extends beyond the eight selected errors.** Restoring the earlier encoder improves teacher-action likelihood on the complete panel for both continuation types, in both replicas, and in both the early and late strata. Restoring the subset head also helps, with a smaller overall point difference. These interventions identify dependence on updated components at fixed remaining parameters; they do not imply that an encoder frozen during training would yield the same result.

For ordinary continuation, encoder rollback changes overall NLL by −0.034034 [−0.041240, −0.027008] relative to the endpoint; head rollback changes it by −0.010416 [−0.013525, −0.007420]. The corresponding weighted differences are −0.039667 [−0.046468, −0.033007] and −0.011930 [−0.015002, −0.008975]. Restoring both gives −0.041703 for ordinary and −0.047194 for weighted. The NLL interactions are positive: +0.002747 [0.001215, 0.004296] and +0.004403 [0.002861, 0.005994]. Keeping the interaction explicit avoids assigning the same change independently to both components.

Likelihood recovery does not translate into an equally clear recovery of exact choices. Encoder rollback changes ordinary agreement by only +0.042235 percentage points [−0.183081, +0.265525], while weighted agreement improves by +0.420426 [+0.219281, +0.624166]. Every rollback still has worse overall NLL and fewer exact teacher matches than the base. Even restoring both leaves pooled NLL higher than the base by +0.009557 [0.007674, 0.011541] for ordinary and +0.009918 [0.007816, 0.012080] for weighted. The remaining projections and other action heads are still updated in that intervention.

Rollback recovers old matches while removing many new ones. Across the two ordinary replicas, encoder rollback rescues 898 of 1,107 base-correct decisions lost by continuation, but loses 803 of 976 decisions newly matched by continuation. Across the weighted replicas, those counts are 1,026 / 1,252 rescued and 779 / 955 newly learned matches lost. These are replica–decision counts over 52,090 evaluations, not distinct states or game wins. Overall ordinary rollback recovers 1,030 endpoint errors and breaks 1,008 endpoint matches, a net gain of just 22 matches. Effects also vary by action family: encoder rollback improves NLL for played hands, discards and Joker sales, but worsens it for pack skips and shop rerolls in both continuation types.

The evidence supports a retention problem during further fitting: better training imitation coexists with worse held-out likelihood, and changing the learned representation contributes to that regression while also supporting newly learned choices. A blanket component rollback does not solve the transfer gap. The next intervention should constrain harmful changes during learning while allowing useful corrections, then face matched ordinary fitting and the base on fresh outcomes. The subsequent [preservation preflight](policy_preservation_preparation_20260925.md) implements and checks one proposed training rule. Its scientific benefit remains untested; neither study establishes immediate action-quality gains or explains the entire original pretraining failure.

The [previous eight-state intervention](correction_component_attribution_20260925.md) found that restoring the earlier encoder rescues a shared fatal choice while losing another corrected choice. This follow-up measures every one of the 26,045 teacher decisions in the existing 128-game panel, across all thirteen observed action families. It evaluates three fixed saved-parameter interventions per continuation: restore encoder and subset head together (010), encoder only (011), or subset head only (110). The shared projections and all remaining tensors stay at the continuation endpoint. The unchanged endpoint is 111. E includes the embedding and Transformer; H is the subset correction head.

These are in-memory combinations of saved parameters, with no retraining, new checkpoints or checkpoint selection. Each ordinary and weighted continuation has two replicas. The full base is a separate control; restoring E+H does not restore the base projections or the other action heads. Teacher agreement is an imitation metric: a different action can have an equally good or better outcome. This scan assigns no new game scores or wins.

All policies below use the same 26,045 decisions. NLL is lower when the teacher action receives more probability; exact agreement is higher when the full policy mode matches the teacher.

| Policy | Matches | Agreement | NLL | Mean teacher probability |
| --- | --- | --- | --- | --- |
| Base | 21544 | 82.7184% | 0.639106 | 76.4013% |
| Ordinary 0 / endpoint | 21477 | 82.4611% | 0.690329 | 77.6752% |
| Ordinary 0 / restore E+H | 21502 | 82.5571% | 0.648681 | 76.5830% |
| Ordinary 0 / restore E | 21482 | 82.4803% | 0.656502 | 77.1110% |
| Ordinary 0 / restore H | 21481 | 82.4765% | 0.679237 | 77.1722% |
| Ordinary 1 / endpoint | 21480 | 82.4726% | 0.690402 | 77.7821% |
| Ordinary 1 / restore E+H | 21511 | 82.5917% | 0.648645 | 76.6487% |
| Ordinary 1 / restore E | 21497 | 82.5379% | 0.656161 | 77.1737% |
| Ordinary 1 / restore H | 21482 | 82.4803% | 0.680663 | 77.2833% |
| Weighted 0 / endpoint | 21389 | 82.1232% | 0.699683 | 77.4422% |
| Weighted 0 / restore E+H | 21510 | 82.5878% | 0.648904 | 76.6985% |
| Weighted 0 / restore E | 21506 | 82.5725% | 0.656491 | 77.1284% |
| Weighted 0 / restore H | 21415 | 82.2231% | 0.686416 | 77.0807% |
| Weighted 1 / endpoint | 21402 | 82.1732% | 0.692752 | 77.3947% |
| Weighted 1 / restore E+H | 21515 | 82.6070% | 0.649143 | 76.7332% |
| Weighted 1 / restore E | 21504 | 82.5648% | 0.656611 | 77.1965% |
| Weighted 1 / restore H | 21411 | 82.2077% | 0.682159 | 76.9783% |

The transition table retains both sides of rollback. “Broken” is endpoint-correct → hybrid-wrong; “recovered” is endpoint-wrong → hybrid-correct. “Old behavior rescued” conditions on decisions the base matched but the endpoint lost. “Learned behavior lost” conditions on decisions the base missed but the endpoint learned. These conditional counts are descriptive, and exact matching does not establish action quality.

| Rollback | Broken | Recovered | Net matches | Old behavior rescued | Learned behavior lost |
| --- | --- | --- | --- | --- | --- |
| Ordinary 0 / restore E+H | 511 | 536 | 25 | 482 / 562 | 430 / 495 |
| Ordinary 0 / restore E | 514 | 519 | 5 | 456 / 562 | 406 / 495 |
| Ordinary 0 / restore H | 131 | 135 | 4 | 86 / 562 | 60 / 495 |
| Ordinary 1 / restore E+H | 486 | 517 | 31 | 466 / 545 | 416 / 481 |
| Ordinary 1 / restore E | 494 | 511 | 17 | 442 / 545 | 397 / 481 |
| Ordinary 1 / restore H | 120 | 122 | 2 | 70 / 545 | 53 / 481 |
| Weighted 0 / restore E+H | 474 | 595 | 121 | 541 / 629 | 409 / 474 |
| Weighted 0 / restore E | 475 | 592 | 117 | 521 / 629 | 388 / 474 |
| Weighted 0 / restore H | 125 | 151 | 26 | 109 / 629 | 58 / 474 |
| Weighted 1 / restore E+H | 476 | 589 | 113 | 539 / 623 | 413 / 481 |
| Weighted 1 / restore E | 469 | 571 | 102 | 505 / 623 | 391 / 481 |
| Weighted 1 / restore H | 136 | 145 | 9 | 94 / 623 | 69 / 481 |

Every predeclared interval appears below. Positive NLL differences are worse imitation; positive agreement differences are better. Labels 010/011/110-minus-111 compare a rollback to its matched continuation endpoint. Comparisons against the shared base appear for the overall panel. The interaction is f(111) − f(110) − f(011) + f(010), with the projections and remaining weights fixed at the endpoint. Interactions are reported explicitly, without assigning percentages of causal credit.

| Group | Arm | Contrast | NLL difference [95% interval] | Agreement difference, pp [95% interval] |
| --- | --- | --- | --- | --- |
| overall | ordinary | 010_minus_111 | -0.041703 [-0.049525, -0.034206] | +0.107506 [-0.105791, +0.319682] |
| overall | ordinary | 011_minus_111 | -0.034034 [-0.041240, -0.027008] | +0.042235 [-0.183081, +0.265525] |
| overall | ordinary | 110_minus_111 | -0.010416 [-0.013525, -0.007420] | +0.011519 [-0.083540, +0.107988] |
| overall | ordinary | EH_interaction | +0.002747 [+0.001215, +0.004296] | +0.053753 [-0.069898, +0.178761] |
| overall | ordinary | 010_minus_base | +0.009557 [+0.007674, +0.011541] | -0.143982 [-0.260883, -0.029745] |
| overall | ordinary | 011_minus_base | +0.017226 [+0.014637, +0.019945] | -0.209253 [-0.355928, -0.064176] |
| overall | ordinary | 110_minus_base | +0.040844 [+0.033919, +0.048224] | -0.239969 [-0.479718, -0.005904] |
| overall | weighted | 010_minus_111 | -0.047194 [-0.054744, -0.039979] | +0.449222 [+0.247395, +0.648602] |
| overall | weighted | 011_minus_111 | -0.039667 [-0.046468, -0.033007] | +0.420426 [+0.219281, +0.624166] |
| overall | weighted | 110_minus_111 | -0.011930 [-0.015002, -0.008975] | +0.067191 [-0.034675, +0.168955] |
| overall | weighted | EH_interaction | +0.004403 [+0.002861, +0.005994] | -0.038395 [-0.185853, +0.112229] |
| overall | weighted | 010_minus_base | +0.009918 [+0.007816, +0.012080] | -0.120945 [-0.232591, -0.013592] |
| overall | weighted | 011_minus_base | +0.017445 [+0.014791, +0.020178] | -0.149741 [-0.295617, -0.009416] |
| overall | weighted | 110_minus_base | +0.045182 [+0.038632, +0.051823] | -0.502976 [-0.720633, -0.289912] |
| ante<=3 | ordinary | 010_minus_111 | -0.042919 [-0.054536, -0.031260] | +0.344693 [-0.084478, +0.782913] |
| ante<=3 | ordinary | 011_minus_111 | -0.030965 [-0.042063, -0.019989] | +0.097555 [-0.321572, +0.530247] |
| ante<=3 | ordinary | 110_minus_111 | -0.013406 [-0.018839, -0.008366] | +0.091051 [-0.084516, +0.267561] |
| ante<=3 | ordinary | EH_interaction | +0.001453 [-0.000633, +0.003564] | +0.156087 [-0.090511, +0.399332] |
| ante<=3 | weighted | 010_minus_111 | -0.044142 [-0.055437, -0.033055] | +0.845473 [+0.429463, +1.253610] |
| ante<=3 | weighted | 011_minus_111 | -0.032797 [-0.043717, -0.022233] | +0.630853 [+0.240319, +1.015728] |
| ante<=3 | weighted | 110_minus_111 | -0.014196 [-0.019332, -0.009438] | +0.201613 [+0.019521, +0.377262] |
| ante<=3 | weighted | EH_interaction | +0.002851 [+0.000875, +0.004909] | +0.013007 [-0.263903, +0.295941] |
| ante>=4 | ordinary | 010_minus_111 | -0.041194 [-0.050706, -0.032314] | +0.008171 [-0.234452, +0.253745] |
| ante>=4 | ordinary | 011_minus_111 | -0.035320 [-0.043941, -0.027091] | +0.019066 [-0.246461, +0.279240] |
| ante>=4 | ordinary | 110_minus_111 | -0.009164 [-0.013353, -0.005256] | -0.021790 [-0.135196, +0.092928] |
| ante>=4 | ordinary | EH_interaction | +0.003290 [+0.001396, +0.005290] | +0.010895 [-0.137941, +0.161098] |
| ante>=4 | weighted | 010_minus_111 | -0.048473 [-0.057749, -0.039563] | +0.283271 [+0.034581, +0.533823] |
| ante>=4 | weighted | 011_minus_111 | -0.042544 [-0.050830, -0.034532] | +0.332298 [+0.065926, +0.599457] |
| ante>=4 | weighted | 110_minus_111 | -0.010981 [-0.015067, -0.007098] | +0.010895 [-0.114657, +0.134658] |
| ante>=4 | weighted | EH_interaction | +0.005053 [+0.003155, +0.007020] | -0.059923 [-0.231828, +0.112619] |
| play_subset | ordinary | 010_minus_111 | -0.071839 [-0.091122, -0.052040] | +0.229020 [-0.374952, +0.868247] |
| play_subset | ordinary | 011_minus_111 | -0.044118 [-0.061452, -0.026437] | -0.188124 [-0.801314, +0.434920] |
| play_subset | ordinary | 110_minus_111 | -0.035337 [-0.046727, -0.024265] | +0.114510 [-0.255701, +0.492259] |
| play_subset | ordinary | EH_interaction | +0.007616 [+0.002824, +0.012327] | +0.302634 [-0.207202, +0.812884] |
| play_subset | weighted | 010_minus_111 | -0.109713 [-0.132454, -0.087204] | +1.578603 [+0.881951, +2.282956] |
| play_subset | weighted | 011_minus_111 | -0.080738 [-0.101192, -0.060365] | +1.316866 [+0.623434, +2.006400] |
| play_subset | weighted | 110_minus_111 | -0.045094 [-0.056761, -0.033765] | +0.384427 [-0.032750, +0.789011] |
| play_subset | weighted | EH_interaction | +0.016119 [+0.010569, +0.021755] | -0.122689 [-0.733497, +0.486218] |

All 76 intervals use 10,000 paired whole-game bootstrap draws, seed 1729. The two continuation replicas stay inside each of the 128 seed clusters; the shared base is counted once. The estimator is the ratio of summed game-level decision differences to summed game-level row counts. Early means encoded Ante ≤3; late means Ante ≥4. Zero-count games stay in the resampling universe. These are unadjusted descriptive intervals on an already exposed panel. Early teacher-state agreement is not early-policy outcome attribution.

The [frozen protocol](../../eval/component_panel_20260925/protocol_v1.json) fixes the models, interventions, panel, metrics and comparisons before full hybrid inference. The complete base reproduces all 26,045 saved predictions exactly. Each of the four unchanged endpoint controls reproduces its eight original 32-row batches exactly; their complete prediction panels come from the [previous independent audit](../../eval/continuation_transfer_diagnostic_20260925/verification_v1.json). Each full hybrid also reproduces all 256 of its prior component-test predictions exactly. Maximum reproduction NLL error is zero throughout.

The [new scan](../../eval/component_panel_20260925/scan_v1/result.json) records 339,609 predictions: 26,045 base controls, 1,024 endpoint controls and 312,540 full hybrid predictions. The [independent audit](../../eval/component_panel_20260925/verification_v1.json) checks those rows plus all 104,180 reused endpoint rows directly against the raw cache. It verifies decoded action families, legal actions, probabilities, every summary and all 192 transition groups. Actual model tensor hashes match the donor assignments established by the previous independent CPU checkpoint audit. All 76 intervals are reconstructed using per-row contrasts and frequency-weighted bootstrap draws with manual percentile calculation. There is no second neural forward pass or new gameplay in this verification.

The [scan terminal](../../eval/component_panel_20260925/scan_terminal_v1.json), [analysis terminal](../../eval/component_panel_20260925/analysis_terminal_v1.json) and [verification terminal](../../eval/component_panel_20260925/verification_terminal_v1.json) record actual successful exits. All processes and their owned sleep guards have ended. No hybrid training fit is inferred from these measurements; replacing checkpoint components is not equivalent to training with them frozen.

The [complete JSON](data/component_panel_20260925.json) retains all results. CSV exports contain [all 272 policy/group summaries](data/component_panel_20260925_metrics.csv), [all 76 contrasts](data/component_panel_20260925_contrasts.csv), and [all 192 transition groups](data/component_panel_20260925_transitions.csv), including every action family and both replicas. World models remain deferred. The long-horizon goal still requires replicated fresh sampled gains over matched ordinary fitting and the base, greedy guardrails, early-policy outcome attribution and independent verification.
