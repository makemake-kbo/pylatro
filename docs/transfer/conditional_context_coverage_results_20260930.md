# The conditional transfer gap persists on familiar Joker pairs

The frozen conditional parent gets **279/529 (52.74%)** development sale choices right even when every legal-owned/offered Joker identity pair has appeared in its original training sale states. Its overall training conditional accuracy is **4,364/5,380 (81.12%)**. All 703 development states use previously represented Joker identities. Unseen card types, or unseen owned–offered identity pairs alone, therefore cannot explain the observed transfer gap.

This is a descriptive coverage census of the already admitted parent, not an evaluation of the running broader-data candidate. It makes no model, simulator or training call and reads no partial candidate scan. Conditional accuracy asks which Joker is selected given a sale; it is not full-action accuracy or game win rate.

| Population | All active sale states | All Joker identities seen | All owned–offered pairs seen |
|---|---:|---:|---:|
| Original 1,000 training games | 4,364 / 5,380 (81.12%) | 4,364 / 5,380 (81.12%) | 4,364 / 5,380 (81.12%) |
| Additional 4,000 games | 11,533 / 21,402 (53.89%) | 11,526 / 21,391 (53.88%) | 8,918 / 16,201 (55.05%) |
| Development 128 games | 372 / 703 (52.92%) | 372 / 703 (52.92%) | 279 / 529 (52.74%) |

Every reference set is built only from the original 5,380 active training sale states. Those states contain 106 distinct legal-owned/offered Joker center IDs and 7,287 directed owned–offered identity pairs. Additional or development observations never enter the reference sets. Every legal owned choice is paired with every present Joker offer; this classification does not use the teacher's chosen pair or the model's correctness.

Among states with familiar identities but at least one unseen pair, accuracy is 2,608/5,190 (50.25%) on additional games and 93/174 (53.45%) on development. The development pair-novelty groups are thus similarly weak. These are selected state groups with different difficulty and correlated decisions, not controlled causal comparisons or independent-trial significance tests.

No additional or development row repeats the original training set's exact ordered tuple of legal-owned and offered Joker identities with their slots. This tuple still omits editions, counters, unsellable owned Jokers, cards, deck, history, money and other features. Familiar pairs do not imply familiar states, equivalent consequences, or sufficient representation of a strategy. The result narrows a simple identity-coverage explanation; it does not identify feature learning, memorization or a specific architecture as the unique cause.

All 27,485 records retain their cached seed, step and teacher-action identities. Token arrays were rehashed, every parent conditional-correct count reproduces the previously admitted controls, and a synthetic check verifies that the context signature preserves ordering while pair membership does not. The successful census took 7.75 seconds and its entire process exited zero before admission. The original attempt stopped because the older archive stores corrected logits directly rather than a separate residual field; its source and failure receipt are preserved. The repaired reader uses those native corrected logits, without another forward or fit.

[Result and source pins](../../eval/consequence_prediction_20260927/conditional_context_coverage_results_v2.json) · [Actual-exit receipt](../../eval/consequence_prediction_20260927/conditional_context_coverage_owner_terminal_v2.json) · [Census source](../../eval/consequence_prediction_20260927/conditional_context_coverage_v2.py) · [Broader-data experiment](conditional_output_diversity_20260930.md)
