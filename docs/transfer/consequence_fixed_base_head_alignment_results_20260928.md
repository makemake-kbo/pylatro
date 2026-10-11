# Many clear-losing overrides contradict the scorer's consequence predictions

The completed fixed-base scorer does not merely lack consequence information on every harmful override. Its clear head ranks the successful base action above the selected action in **52 of 74** observed clear-losing overrides. Both score and clear heads rank base higher in **48**. The preference argmax nevertheless selects the alternative. This is evidence of disagreement between learned outputs, not proof that a different policy would improve eventual wins.

The [declared saved-data diagnostic](consequence_fixed_base_head_alignment_20260928.md) covers every one of the 847 overrides admitted by the [fixed-base experiment](consequence_fixed_base_residual_results_20260928.md). It reads all 6,113 saved full-support rows, matches selected actions and predictions to retention records, verifies fixed-base plus learned-correction preference margins, and reproduces all previous gain/loss counts. No model forward, simulator branch, fitting or new policy evaluation ran.

| Selected action loses a clear that base achieves | Cases | Clear head correctly favors base | Score head correctly favors base | Both favor base | Neither favors base |
| --- | ---: | ---: | ---: | ---: | ---: |
| All overrides | 74 | 52 | 52 | 48 | 18 |
| Newly leaving base | 29 | 23 | 22 | 21 | 5 |
| Keeping the previous override | 26 | 17 | 18 | 16 | 7 |
| Changing to another override | 19 | 12 | 12 | 11 | 6 |

In eight cases, exactly one head ranks base correctly: four are score-only and four are clear-only. The joint categories are 48 both-correct, four score-only, four clear-only and 18 both-wrong, totaling 74. No ties occur. Each ordering compares only the two already assessed actions and uses their original-RNG realized clear outcomes.

On the 62 clear-gaining overrides, the clear head correctly favors the successful selected action in 54 cases and the score head in 50; both are correct in 46. Eight are clear-only, four score-only, and four have both heads wrong. Thus a consequence ordering can be informative in both directions, but neither head is an error-free decision rule.

Among the 248 new overrides, seven gain clears and 29 lose them. Clear predictions correctly order 23 of those 29 losses; both heads do so for 21. The new fatal Pair override described in the experiment report belongs to the both-wrong group, so consequence-estimation errors remain relevant alongside output disagreement.

The next hypothesis should distinguish how preferences use already learned consequences from what the representation cannot predict. The current score, clear and preference outputs share a learned hidden representation but have separate final output coordinates; changing a final readout or providing its own outputs as linear inputs would not automatically create new representational capacity or better supervision. The observed disagreement does not establish which change would help. It also does not justify an immediate-score or clear-probability maximization rule: 73 of these 74 lost clears occur with more hands remaining, where growth and resource preservation can matter. No continuation-value measurement shows that these particular deferrals help or hurt eventual wins.

This is post-hoc descriptive evidence from one fit on reused development games. No confidence interval, architecture selection, threshold, coefficient or gate was added. The rejected endpoint remains rejected, and the original fresh-win and final independent-fit requirements remain unmet.

The standard-library analysis completed in 2.128 seconds. Its owner actually exited zero in tool chunk `2ef108` before admission `d69083`. [Result](../../eval/consequence_prediction_20260927/fixed_base_head_alignment_v1/result_v1.json), SHA256 `63e076c1cc379b7f5ae1489d6feb46a11ae20e40672a86961fb92d1ecf38ef03`; [owner](../../eval/consequence_prediction_20260927/fixed_base_head_alignment_owner_terminal_v1.json), SHA256 `fedb0117907f8316606f2ee2b5e424b812b3c6e2f9f37fd24508191658f6a1e0`; [protocol](../../eval/consequence_prediction_20260927/fixed_base_head_alignment_protocol_v1.json), SHA256 `ade6f8cb1bdb4a8ede791cccec1cf04a746a3af209909b75510246cb5ac4cdd5`.
