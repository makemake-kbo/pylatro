# Earlier stopping improves replay clears but misses the win threshold

The broader-data run's **2,000-update checkpoint** passes the teacher-path duration screen but fails the subsequent autonomous advancement screen. It wins 42/128 sampled games versus base 39 and ordinary continuation 40, below both required +5-point gains. The complete game results appear below; this duration candidate is closed. Its earlier replay peak remains informative: 2,712 clears versus base 2,699, falling to 2,677 at 4,000 updates despite improving teacher-action likelihood.

The [duration diagnostic](continuation_duration_20260927.md) measures every previously unmeasured saved replica-0 ordinary/data-scale prefix (1,000 and 2,000 updates), reuses the common initial base and both fixed endpoints, and reuses the completed initial Stage 2 trajectory. No model was trained. The original fixed-4,000-update [data experiment](rule_aligned_data_scale_results_20260927.md) remains completed with a failed screen.

| Policy / updates | Overall NLL | Play NLL | Early NLL | Clears / 6,113 | Early clears / 2,152 | New immediate losses versus teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base / 0 | 0.639106 | 1.295090 | 0.722486 | 2,699 | 1,066 | 7 |
| Ordinary / 1,000 | 0.648459 | 1.284830 | 0.735167 | 2,700 | 1,079 | 7 |
| Ordinary / 2,000 | 0.653712 | 1.308620 | 0.735218 | 2,685 | 1,059 | 6 |
| Ordinary / 4,000 | 0.690329 | 1.389750 | 0.775147 | 2,671 | 1,063 | 5 |
| Data scale / 1,000 | 0.615139 | 1.237688 | 0.698700 | 2,691 | 1,067 | 6 |
| **Data scale / 2,000** | **0.600309** | **1.194048** | **0.686203** | **2,712** | **1,072** | **8** |
| Data scale / 4,000 | 0.583929 | 1.150939 | 0.670916 | 2,677 | 1,057 | 6 |

Early means Ante 1–3. All 26,045 inference states and 6,113 replayed plays come from the same exposed 128 development games. The table is descriptive; these are neither fresh confirmation results nor autonomous win rates. Lower NLL does not consistently imply more clears, and the selected prefix has one more new immediate loss than the base.

Against the base, data scale at 2,000 gains 67 clears and loses 54 (+13); early it gains 28 and loses 22 (+6). The fixed partitions contribute +31 on the 339 preference states, −19 on 4,156 states where base equals teacher, and +1 on the other 1,618 disagreements. Existing-clear losses therefore remain substantial.

Between updates 2,000 and 4,000, data scale loses 70 previously clearing choices and gains 35, a net loss of 35 clears; 26 of that net loss occur in the base-equals-teacher partition. Early clears decline by 15. The same training interval improves overall NLL by approximately 0.01638. This is evidence of a mismatch between late imitation gains and consequence retention on these teacher paths.

The ordinary continuation also has an earlier consequence peak: its 1,000-update checkpoint gains one net clear and 13 early clears versus base. Data scale at 2,000 has 12 more overall clears than ordinary at 1,000 but seven fewer early clears. The shorter ordinary checkpoint is therefore a meaningful competing duration control, even though its mean NLL is worse than the base. Do not compare only with the weaker 4,000-update ordinary endpoint when attributing a future gain to broader data.

Both data-scale prefixes improve NLL over base and their matched ordinary prefixes. Only update 2,000 passes all five declared checks: lower NLL than base/ordinary, more clears than ordinary, at least base clears, and at least base early clears. It also passes the analogous checks against ordinary at 4,000. Update 1,000 fails the overall-clear requirements.

The [complete immediate-loss comparison](../../eval/continuation_duration_20260927/fatal_retention_v1.json) further qualifies that candidate: its eight new losses include **all seven base failures plus one additional failure**. Every case occurs with one hand left and a teacher action that clears. The earlier Even Steven example (795 points against an 800 requirement, versus the teacher’s 1,989) remains wrong at every measured ordinary and data-scale checkpoint. In the added failure, seed 850000060 / step 84, the candidate scores 690 against a remaining 1,098; base scores 2,576 and teacher 5,824. The 4,000-update data-scale checkpoint fixes that particular case despite losing more clears overall. These are exact teacher-path set comparisons, not a causal explanation of eventual win rates or an architectural impossibility result. They do not alter the frozen spending rules after outcomes are observed.

## Verification and next decision

Four new scans produced 104,180 predictions; the independent audit checked 182,315 rows including archived controls. All checkpoints bind to their existing tensor/exposure audits, and inference leaves model tensors unchanged. Replay assesses 9,424 unique state–action outcomes, with 8,803 new branch calls and 621 reused ordinary-control outcomes. The independent raw audit passes 258,636 checks. All three archived endpoint policies reproduce their saved full outcomes exactly across all plays. These checks verify provenance, factual transitions and saved arithmetic; they do not constitute a second engine implementation.

Measurement, inference audit, replay, raw audit, analysis and both owners actually exited successfully. Evidence: [scan audit](../../eval/continuation_duration_20260927/scan_audit_v1.json), [replay audit](../../eval/continuation_duration_20260927/replay_v1/panel_verification_v1.json), [complete duration analysis](../../eval/continuation_duration_20260927/analysis_v1.json), [observed scan exit](../../eval/continuation_duration_20260927/scan_owner_terminal_v1.json), [observed replay exit](../../eval/continuation_duration_20260927/replay_owner_terminal_v1.json).

## Completed autonomous development screen

The separately frozen [game evaluation](duration_development_20260927.md) is complete and **fails both sampled-gain gates**. No new model was fitted. The 384 new candidate episodes and 640 reused control episodes yield:

| Evaluation | Data scale / 2,000 | Base | Ordinary / 4,000 |
| --- | ---: | ---: | ---: |
| Full sampled wins / 128 | 42 | 39 | 40 |
| Full greedy wins / 128 | 57 | 59 | 57 |
| Early-only sampled hybrid wins / 128 | 39 | — | 37 |

The early intervention switches permanently to ordinary at the first preaction Ante >= 4. Its seeds differ from the full-policy panel. All seeds were already exposed development seeds; this is not fresh confirmation. Sampling uses a global Torch stream, not per-game common random numbers.

| Comparison | Gain (percentage points) | Descriptive paired 95% interval |
| --- | ---: | ---: |
| Sampled candidate − base | +2.34375 | [−7.03125, +11.71875] |
| Sampled candidate − ordinary | +1.56250 | [−7.03125, +10.15625] |
| Greedy candidate − base | −1.56250 | [−10.93750, +7.81250] |
| Greedy candidate − ordinary | 0 | [−7.81250, +8.59375] |
| Early hybrid − ordinary | +1.56250 | [−9.37500, +12.50000] |
| Sampled ordinary − base | +0.78125 | [−7.81250, +9.37500] |
| Greedy ordinary − base | −1.56250 | [−10.93750, +7.03125] |

The three other provisional gates pass on point estimates, but neither sampled comparison reaches the predeclared +5-point threshold. All intervals span zero; the stricter final proof standard is also unmet. The independent audit verifies all 1,024 episode records, 159,597 action records, route transitions and seven bootstrap intervals. All three new panel processes, analysis, verification and the outer owner actually exited zero; archived controls retain their original receipts. Evidence: [independent verification](../../eval/continuation_duration_20260927/autonomous_v1/verification_v1.json), [analysis](../../eval/continuation_duration_20260927/autonomous_v1/analysis_v1.json), [observed owner exit](../../eval/continuation_duration_20260927/autonomous_owner_terminal_v1.json).

Per the frozen rule, stop this duration candidate without further ordinary-duration games or a confirmation fit. Earlier stopping improves teacher-path consequences but has not established the required autonomous benefit. Proceed to the [consequence-label audit](consequence_label_audit_20260927.md), then pooled prediction before a matched relational readout. One exploratory fit per change and final-only independent confirmation remain in force; the long-horizon goal is unproven.
