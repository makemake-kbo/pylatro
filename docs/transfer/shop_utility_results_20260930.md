# Teacher comparison margins on a fixed eight-game replay

The diagnostic reproduces **76 teacher choices** and **1736 recorded decisions** with exact equality of all fourteen observation fields. Adding the comparison observer leaves every teacher choice unchanged. The whole native owner and the separate saved-record verifier exited successfully before admission. No model was trained or evaluated autonomously.

The [fixed plan](shop_utility_trace_20260930.md) selects the first four original-training and first four exposed-development games. It includes every active teacher-sale state and the first two active non-sale states per game. No prediction or margin determines inclusion. These small panels illustrate the remaining gap; they do not estimate its corpus-wide prevalence.

The table reports **median heuristic gains**, with the number of scored comparisons in parentheses. A teacher's existing gain cutoff is 0.03. Different replacement plans emitting the same immediate sale action are reduced to their maximum feasible gain. The other-action column compares against the strongest *different* scored action. The model-action column is available only when the teacher's search actually scored that model choice. Unscored actions are not assigned zero utility.

| Panel | Sale group | States | Gain above native cutoff | Gap to other scored action | Gap to model action |
| --- | --- | ---: | ---: | ---: | ---: |
| training | Correct sale | 12 | 0.213372 (12) | 0.096762 (12) | 0.000000 (12) |
| training | Correct Joker rank, wrong full action | 8 | 0.074323 (8) | 0.162524 (8) | unscored |
| training | Wrong Joker rank | 8 | 0.318135 (8) | 0.059147 (8) | 0.084279 (6) |
| development | Correct sale | 8 | 0.433820 (8) | 0.157274 (8) | 0.000000 (8) |
| development | Correct Joker rank, wrong full action | 14 | 0.148938 (14) | 0.137779 (14) | unscored |
| development | Wrong Joker rank | 10 | 0.327518 (10) | 0.055825 (9) | 0.031566 (5) |

These gains combine the teacher's public sampled build output, projected growth, income and price/interest costs after its safety checks. They are not observed score differences, calibrated win probabilities, or eventual win values. Plans rejected before a gain comparison and fallback actions remain unscored. A positive gap demonstrates the teacher's preference only.

The observer inserts callbacks before exactly three native comparisons and preserves the original AST when those calls are removed. An independent reduction reconstructs every incumbent update, action maximum and reported margin, retaining the original strict tie rule. Original and observed teacher queries use separate state copies with restored UID allocation; factual state, observations and RNG states remain unchanged. Recorded game endings are reproduction controls, not new policy-performance evidence.

The sale-family endpoint remains rejected by its unchanged development screen. This diagnosis selects no new architecture, close-call threshold, decoder, coefficient or fit. Long-horizon attribution, fresh-game win-rate improvement and final independent confirmation remain unproven.

Artifacts: [raw native comparisons](../../eval/consequence_prediction_20260927/shop_utility_diagnostic_v1/records.jsonl), [native reproduction](../../eval/consequence_prediction_20260927/shop_utility_diagnostic_v1/result_v1.json), [independent reductions and full margin summaries](../../eval/consequence_prediction_20260927/shop_utility_diagnostic_v1/verification_v1.json), [whole-owner admission](../../eval/consequence_prediction_20260927/shop_utility_diagnostic_owner_terminal_v1.json), [frozen protocol](../../eval/consequence_prediction_20260927/shop_utility_diagnostic_protocol_v1.json).
