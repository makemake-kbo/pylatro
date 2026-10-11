# What the conditional sale branch cannot repair

The feature-adapted model improves held-out sale likelihood but still fails three of thirteen fixed checks. A saved-state oracle calculation identifies a real restriction: **318 of 703 active development sales cannot become the highest-probability action through any change to conditional Joker choice alone.** The probability assigned to selling is below the probability of the best non-sale action, even if all sale probability were assigned to the teacher's Joker.

This includes **186 of 232** states where the model already ranks the teacher's Joker first but chooses another action family. Of the 35 newly corrected Joker rankings that still fail as full actions, 20 face this restriction and 15 could theoretically be repaired by concentrating the conditional sale distribution further.

| Current outcome | A perfect conditional sale distribution can win | Requires changing the family or non-sale probabilities | Total |
|---|---:|---:|---:|
| Correct Joker and full action | 160 | 0 | 160 |
| Correct Joker, wrong full action | 46 | 186 | 232 |
| Wrong Joker ranking | 179 | 132 | 311 |
| All active teacher sales | 385 | 318 | 703 |

The other thirteen development sales are outside this branch's activation rule and remain incorrect. The resulting per-state teacher-oracle ceiling is **385/716**, versus the model's actual **160/716**. This is an upper bound for a hypothetical teacher-dependent conditional distribution, not an attainable trained policy or a game result. It is well above the original comparison scores, so the restriction does **not** prove that the failed development checks were impossible to pass or that family choice is the sole bottleneck.

For a sale action, joint probability is `P(sell | state) × P(Joker | sell, state)`. Conditional concentration cannot exceed the first factor. When the saved native mode is a sale, perfect conditional choice is already feasible. When another family wins, its best action probability can be compared with the sale-family mass. All 703 rows resolve exactly: 301 already select some sale; 216 have a purchase competitor whose probability is reconstructed from its saved logits and legal mask; 186 have a competitor from a family with one legal action. No unavailable-distribution bounds or numerical ties remain unresolved. The existing 2e-5 numerical tolerance is retained.

This calculation uses the admitted scan, its saved native modes, and the pinned legal-action masks. It makes no model forward, simulator call, optimizer step, or policy intervention. All row identities and reconstructed choice factors match the admitted analysis. A separate probability recount and 200 synthetic full-distribution cases verify the oracle comparison. The first script version failed before producing results because it looked for the action mask in the label manifest; the preserved second version reads the existing pinned observation-cache manifest. No model or numerical threshold changed.

The next diagnosis should examine whether the frozen sale-family probabilities are miscalibrated across **all** eligible shop states, including states where the teacher does not sell. This can distinguish a broad probability bias from errors that require different decisions in different contexts. A subsequent controlled experiment could train a separately supervised sale-family correction while freezing the improved conditional scorer. Simply enabling the old joint correction is not justified: conditional-only training does not identify a common additive offset to sale logits, yet that offset would affect a joint correction. No new fit has been selected here.

[Feature-adaptation result](conditional_feature_adaptation_results_20260930.md) · [Oracle calculation](../../eval/consequence_prediction_20260927/conditional_sale_ceiling_results_v2.json) · [Independent verification](../../eval/consequence_prediction_20260927/conditional_sale_ceiling_verification_v2.json) · [Successful owner](../../eval/consequence_prediction_20260927/conditional_sale_ceiling_owner_terminal_v3.json)
