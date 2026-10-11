# The teacher's comparison scores leave most competing actions unvalued

The [eight-game diagnostic](shop_utility_results_20260930.md) recovered the teacher's existing comparisons while reproducing all 76 queried actions and all 1,736 recorded observations. Its separate saved-record verifier passed with zero reduction error. This establishes that these comparison values can be recovered faithfully on the fixed panel.

It also exposes a limitation: **19 of the 24 missed development sales have no comparable teacher utility for the model's alternative action**. The teacher's search evaluates Joker replacement plans and some other purchases, then uses fallback rules for further actions. Its chosen label expresses a preference across those branches, but the saved scalar comparisons do not give a common utility difference across them.

| Unscored model alternative on a missed sale | Original training | Development |
| --- | ---: | ---: |
| Booster purchase | 6 | 8 |
| Voucher purchase | 0 | 2 |
| Reroll | 2 | 5 |
| Leave shop | 2 | 3 |
| Other Joker sale without a feasible scored plan | 0 | 1 |
| Total unscored alternatives | 10 of 16 missed sales | 19 of 24 missed sales |

All **14 development cases with the correct Joker ranking but the wrong full action** fall in this unscored group: five booster purchases, two voucher purchases, four rerolls and three exits. Their teacher replacement gains range from 0.034278 to 0.426961, with median 0.178938, against the teacher's existing 0.03 cutoff. That cutoff is a heuristic rule, not an eventual-win threshold. The median gap to the best different *scored* action is 0.137779; it does not measure the gap to these unscored model alternatives.

For the five development mistakes with comparable action values, the teacher-minus-model gap ranges from 0.016561 to 0.180279, with median 0.031566. No exact alternative-action ties occur among the 31 development sales with another scored action. One of the six comparable training mistakes is an exact tie under this heuristic. These observations do not define a threshold for harmless errors or establish simulator equivalence.

This small panel supports neither declaring all disagreements costly nor dismissing them as close calls. A purchase may precede a still-available replacement; spending, rerolling, leaving or choosing a different sale can also remove that opportunity. The next useful diagnosis is to trace whether the model's alternative actually preserves the teacher's intended replacement, including resolving any opened pack. Such a bounded simulator check would measure opportunity retention, not eventual wins, and would require its own frozen protocol and factual controls. It has not been executed.

Do not start another fit from this evidence alone. Richer supervision limited to the teacher's existing candidate gains would still leave most of these competing actions unvalued. The rejected model remains rejected, and the original fresh-game, retention, early-attribution and final independent-confirmation requirements remain unchanged.

Sources: [verified comparison summaries](../../eval/consequence_prediction_20260927/shop_utility_diagnostic_v1/verification_v1.json), [unscored action families and identities](../../eval/consequence_prediction_20260927/shop_utility_unscored_actions_v1.json), [purchase types from pinned observations](../../eval/consequence_prediction_20260927/shop_utility_unscored_purchase_types_v1.json), [actual native-owner and verifier exits](../../eval/consequence_prediction_20260927/shop_utility_diagnostic_owner_terminal_v1.json).
