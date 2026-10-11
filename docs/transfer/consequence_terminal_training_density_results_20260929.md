# More terminal labels within the same games do not repair transfer

The [fixed training-density experiment](consequence_terminal_training_density_20260928.md) is complete and rejected. Adding a second decision from each training game increases informative training states from **166 to 311**, but the new readout orders only **40/84 validation win/loss pairs (47.62%)**, versus the preceding conditional readout's **47/84 (55.95%)** and base's **50/84 (59.52%)**. Eight of nine prospective checks fail. No held-out collection, policy integration or independent confirmation fit qualifies.

The intervention preserves the 800 training games, 200 validation decisions, original labels, frozen wide features, RMS normalization, 384-weight conditional scorer, penalty 0.001 and optimizer. It adds one outcome-independent hash-selected decision from each training game. This changes supervision density within familiar games, not source-game diversity. There is exactly one new fit. The numerical audit independently checks its recorded optimization; it does not fit a second model.

All **2,400 new continuations** finish uncensored. They provide **145 mixed-outcome states and 290 winning/losing pairs**, including 220 pairs with the same immediate-clear label. The combined training panel contains **1,600 decisions from 800 games**, with **311 informative states from 271 games** and **622 terminal comparisons**. The comparison count exceeds 384, satisfying the planned eligibility check; these are correlated comparisons, not 622 independent games or a measured matrix rank. The other 1,289 training states have zero conditional loss. All original label bytes, feature arrays, control scores and validation rows are preserved exactly.

| Validation metric | Previous conditional | Denser conditional | Joint RMS | Base |
|---|---:|---:|---:|---:|
| Conditional NLL, lower is better | 1.662005 | **1.950547** | 1.355129 | Uncalibrated |
| Terminal-pair ordering | 55.95% | **47.62%** | 60.71% | 59.52% |
| Same-clear ordering, state weighted | 58.62% | **50.00%** | 56.90% | 53.45% |
| Restricted-choice wins | 112/200 | **109/200** | 117/200 | 116/200 |
| Early pair ordering | 51.92% | **53.85%** | 69.23% | 55.77% |
| Early restricted-choice wins | 30/74 | **31/74** | 36/74 | 32/74 |
| Later pair ordering | 62.50% | **37.50%** | 46.88% | 65.63% |
| Later restricted-choice wins | 82/126 | **78/126** | 81/126 | 84/126 |

Uniform conditional NLL is 1.098612. The new model is worse by **+0.851935 [0.372675, 1.429261]** and worse than the joint RMS control by **+0.595418 [0.076846, 1.173561]**. Both exploratory paired intervals exclude zero. Its NLL change against the previous conditional readout is **+0.288542 [−0.182428, 0.781616]**, unresolved.

Validation pair ordering differs from base by **−11.90 percentage points [−28.57, +4.76]** and from the previous conditional readout by **−8.33 points [−20.24, +3.57]**. Restricted-choice wins differ from base by **−3.50 points [−7.50, +0.50]** and from the previous conditional readout by **−1.50 points [−5.00, +2.00]**. These intervals do not establish an overall regression relative to those two controls, but there is no demonstrated improvement and the fixed point-estimate criteria fail. Equality at 50% also fails the strictly-greater-than-50% same-clear criterion.

The predeclared later group accounts for much of the adverse movement: ordering falls from 20/32 to 12/32 pairs, a **−25.00-point [−40.63, −9.38]** difference against the previous conditional readout. Later restricted-choice wins decline by four, **−3.17 points [−6.35, −0.79]**. This subgroup has only 16 informative games; its exploratory result does not establish a universal late-game mechanism. Early results remain below base and do not qualify an early-only policy.

| Training population | Informative states | Previous conditional ordering | Denser conditional ordering | Base ordering |
|---|---:|---:|---:|---:|
| Original 800 decisions | 166 | 98.49% | **92.47% (307/332)** | 50.60% |
| Added 800 decisions | 145 | 49.31% | **84.83% (246/290)** | 49.31% |
| Combined 1,600 decisions | 311 | 75.56% | **88.91% (553/622)** | 50.00% |

Combined training NLL is **0.547801**, versus the fixed previous conditional model's 1.104654 and joint RMS model's 1.190379. Restricted training choices win **956/1,600**, versus base's 827. The old model's 49.31% ordering on the newly labelled decisions is an additional useful observation: even new decisions from its familiar source games were not reliably ordered before fitting them. The enlarged fit learns many of those labels, while validation remains weak. Training gains therefore remain insufficient evidence of reusable delayed-consequence knowledge.

The controlled conclusion is that **this increase in within-game label density is insufficient**. It does not show that all larger datasets would fail, nor identify architecture, label variability, context coverage or regularization as the unique cause. Terminal outcomes are single original-RNG realizations under the fixed base controller, not known expected action values. The experiment does not test broader source-game diversity. No further density level or regularization sweep is selected from this result.

Only the training-relative-fit check passes. Validation NLL versus uniform and RMS, pair ordering versus base and RMS, early pair ordering, same-clear ordering, overall restricted choices and early restricted choices all fail. Three-proposal choices include the teacher action and use the same base continuation after the first action; none of the win counts above is an autonomous win rate. All source games contributed to feature fitting, and this validation split has been reused. Intervals use 10,000 paired source-game resamples with the fixed seed 1729 and are exploratory, not final confirmation.

Collection and independent native replay take **9,485.66 and 9,310.16 seconds**. The audit verifies all 202,141 factual decisions and 206,887 branch actions, terminal outcomes and summary counts. The new-feature audit exactly reproduces 921,600 hidden coordinates, 7,200 prediction coordinates and 1,600 original-proposal overlaps. The readout converges in **five Newton updates**, maximum absolute gradient **6.16×10⁻¹²**; fitting and its independent mathematical audit take **0.232 and 0.297 seconds**. The audit verifies the subset likelihood and derivatives, accepted updates, stationarity, all 5,400 scores, both fixed controls, source-game split, summaries, intervals and all nine checks. Every whole owner actually exits successfully before its results are admitted.

The full source-game holdout, retention, fresh sampled and greedy gameplay, early-policy attribution and final independent-fit requirements remain unmet. The next investigation should distinguish the remaining representation/coverage and terminal-label reliability explanations using existing evidence before committing to another fit. World-model work remains deferred.

Artifacts: [collection owner](../../eval/consequence_prediction_20260927/terminal_density_collection_owner_terminal_v1.json), [label merge](../../eval/consequence_prediction_20260927/terminal_density_labels_v1/result_v1.json), [array merge](../../eval/consequence_prediction_20260927/terminal_density_arrays_v1/result_v1.json), [fit owner](../../eval/consequence_prediction_20260927/terminal_density_fit_owner_terminal_v1.json), [frozen fit protocol](../../eval/consequence_prediction_20260927/terminal_density_fit_protocol_v1.json), [fit result](../../eval/consequence_prediction_20260927/terminal_density_fit_v1/result_v1.json), and [independent mathematical audit](../../eval/consequence_prediction_20260927/terminal_density_fit_v1/verification_v1.json).
