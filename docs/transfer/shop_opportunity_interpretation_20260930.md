# Sale disagreement mixes purchase order, missed opportunities and wrong removals

The [completed simulator diagnostic](shop_opportunity_results_20260930.md) separates three kinds of disagreement in the fixed development sample. All 32 teacher replacement plans were executable after the actual sale; the teacher's next recorded action bought the proposed item in 31 cases. The forecast-versus-real-sale distinction did not invalidate a plan in this panel.

| Model alternative on a missed development sale | Cases | Local result |
| --- | ---: | --- |
| Buy a pack or voucher | 10 | The original sale and purchase both remain legal and execute on a disposable probe. |
| Sell another Joker | 6 | The offered replacement can be bought, but a different original Joker has already been removed. |
| Reroll | 5 | The original offered item is removed. |
| Leave shop | 3 | The current purchase opportunity is closed. |

All ten competing purchases preserve both the original intended Joker to sell and the same physical offered item. Eight open packs; the fixed teacher resolves each pack before the probe. None loses purchase affordability after the original sale. This establishes that these first actions can precede the replacement. It does **not** establish that the learned policy follows through, that the purchases are beneficial, or that the resulting build has the same value.

For the **14 cases where the model already ranks the intended Joker first**, seven competing purchases preserve the replacement, while four rerolls and three exits close this particular opportunity. Thus the observed gap between conditional ranking and full-action agreement contains both possible sequencing differences and decisions that forego the current offer. A reroll may still find something better; leaving may preserve resources. Their eventual value is unmeasured.

The original-training sample shows the same categories: six competing pack purchases preserve the replacement, six different-Joker sales leave the offer buyable while removing another Joker, two rerolls remove the offer, and two exits close the shop. All 28 teacher replacement plans execute, and 26 are followed immediately by the proposed recorded purchase. No purchase branch is blocked or unresolved in either panel. Across both panels, fourteen packs require fifteen teacher actions to resolve.

These counts come from 1,736 reproduced factual decisions, 60 teacher-sale controls and 40 model-alternative branches. The independent audit checks 100 branches, 243 transitions and 120 cached observation hashes. The 57 preserved records from the earlier recorder attempt match after removing only mask-sensitive state fingerprints; actions, observation hashes, identities, prices, UID allocation and outcomes are unchanged. The whole native run took 17.1 seconds. Completed artifacts and failed attempts are preserved.

The next useful check is **fixed-state follow-through prediction** on the sixteen post-purchase states across both panels. Reconstruct and pin those states using the admitted branch actions and fingerprints, then inspect the existing model's next prediction without executing it. Re-query the teacher on the changed state too: a pack or voucher can alter which replacement is desirable. Reproduce the model's original-state controls under its frozen runtime before interpreting new predictions. This remains a local diagnostic; do not use the rejected endpoint for autonomous games or the gated game-interface study, change the original qualification checks, or launch another fit from this small sample.

The broader lesson is limited but concrete: immediate teacher-action disagreement does not identify whether a policy misses the replacement, delays it, or removes the wrong Joker. We still need evidence about learned follow-through and eventual outcomes. Long-horizon generalization and improved win rate remain unproven.

Sources: [independent branch accounting](../../eval/consequence_prediction_20260927/shop_opportunity_v4/verification_v1.json), [raw transitions and probes](../../eval/consequence_prediction_20260927/shop_opportunity_v4/records.jsonl), [native controls and timing](../../eval/consequence_prediction_20260927/shop_opportunity_v4/result_v1.json), [whole-owner and verifier exits](../../eval/consequence_prediction_20260927/shop_opportunity_owner_terminal_v4.json).
