# Card IDs can change shop targets without changing policy inputs

The native probe found two changed teacher decisions with exactly identical values in all fourteen policy-input arrays. Both changes occurred in one training game. Restoring only the sampler's original card-ID ordering restored the original decisions and every recorded comparison value. This establishes a bounded counterfactual dependency of teacher targets on an omitted identifier. It does not establish conflicting labels among naturally reachable corpus states or explain the broader imitation gap.

The fixed panel contained all 245 active SHOP states in the first four original-training games, replaying 889 factual decisions. Selection preceded the new teacher queries. Every original observation, teacher control, and game endpoint reproduced its recorded counterpart. At each eligible state, the probe swapped the IDs of one deterministically selected pair with the same underlying rank and suit but different enhancements or other variants. Every other card field and physical pile/Joker order remained unchanged. No proposed contrast action was executed. [Frozen protocol](../../eval/consequence_prediction_20260927/shop_uid_alias_protocol_v1.json) · [Admitted execution](../../eval/consequence_prediction_20260927/shop_uid_alias_owner_terminal_v1.json).

| Original teacher label | Selected states | Eligible swaps with identical inputs | Changed native action | Changed action after sampling control |
| --- | ---: | ---: | ---: | ---: |
| Joker sale | 28 | 24 | 0 | 0 |
| Other action | 217 | 204 | 2 | 0 |
| Total | 245 | 228 | 2 | 0 |

The remaining 17 states had no eligible pair. The audit independently checked 490 observation hashes, 3,430 original cached fields, every recorded card field, and 16,416 sample-membership lists. All 946 teacher queries completed. Both the whole native owner and the independent auditor exited successfully. [Native result](../../eval/consequence_prediction_20260927/shop_uid_alias_v1/result_v1.json) · [Independent verification](../../eval/consequence_prediction_20260927/shop_uid_alias_v1/verification_v1.json).

Both changed cases were in game 4,000,003, Ante 4, steps 82 and 84. The pair was a normal Ace of Hearts and a Stone card with the same underlying front. Changing their IDs altered which physical card entered some of the teacher's fixed samples. The best scored plan was to sell Business Card and buy Seltzer; its gain crossed the teacher's existing 0.03 threshold:

| Factual step | Original plan gain | Gain after ID swap | Original action | Action after ID swap |
| --- | ---: | ---: | --- | --- |
| 82 | 0.023460 | 0.054200 | Buy shop item 3 | Sell Joker 1 |
| 84 | 0.010596 | 0.041335 | Buy shop item 3 | Sell Joker 1 |

Indices are zero-based. The original purchase was a fallback action without a scored gain in these comparisons. These numbers therefore compare the replacement plan with the teacher's threshold, not with the purchase's strategic value. The two cases are correlated observations from one game, not two independent games.

Value sensitivity was more common than action changes: 75 of the 228 input-equivalent contrasts changed at least one matched candidate gain; 426 of 1,192 matched comparisons changed. The complete recorded trace changed in 96 states. The sampling-order control restored the entire recorded scientific trace exactly in all 228 cases, including actions, forecasts, candidate lists, and gains. This points to sampling as the mechanism in this panel and matters if those continuous values become auxiliary training targets. [Trace analysis](../../eval/consequence_prediction_20260927/shop_uid_alias_analysis_results_v1.json) · [Independent recount](../../eval/consequence_prediction_20260927/shop_uid_alias_analysis_verification_v1.json).

Swapped-ID states have not been shown naturally reachable with the same history or equivalent for future gameplay; the engine also uses IDs in ordering and Stone-card identities. Consequently, 2/228 is only this perturbation's observed sensitivity, not a corpus noise rate or an irreducible-error estimate. No policy input, teacher, checkpoint, or training data was changed. This study provides no evidence of improved win rate or long-horizon generalization.

The original sale labels survived all 24 eligible swaps. The broader missed-sale problem remains open. Before another architecture fit, the next useful comparison is whether the model can recover the teacher's scored replacement advantage and threshold decision from observable features, while accounting for this newly measured target sensitivity. That requires a separately specified supervision experiment; neither changing the sampler nor adding hidden IDs to the policy is justified by this probe alone.
