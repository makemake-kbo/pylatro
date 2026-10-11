# Local retention of the teacher's intended replacement

The bounded diagnostic replayed **1736 factual decisions**, checked **60 teacher-sale controls**, and executed the **40 differing saved model actions** from the same eight-game panel. All fourteen factual observation fields and every factual sale's state, observation and UID allocation reproduced. An independent saved-record audit checked 100 branches and 243 transitions. Both whole native owner and verifier exited successfully before admission.

The [frozen plan](shop_opportunity_retention_20260930.md) tracks the actual original offered Joker and owned Joker by object identity. It never substitutes a rerolled offer with the same name. Each model branch starts at the same recorded state and RNG as its teacher control. A fixed teacher resolves any pack opened by the model action, with a 32-action limit. No learned policy acts beyond the single saved action.

Teacher controls test whether the forecasted purchase is legal and actually executes after the original sale. The final column asks whether the *recorded* next teacher action was that proposed purchase; a later teacher can reconsider its plan.

| Panel | Teacher sales | Proposed purchase succeeds after sale | Recorded next action is proposed purchase |
| --- | ---: | ---: | ---: |
| training | 28 | 28 | 26 |
| development | 32 | 32 | 31 |

For model alternatives, “reachable” means that an actual disposable simulator probe buys the original offered Joker either immediately or after selling the original intended Joker. Other original Jokers may already have been lost; this predicate does not establish portfolio equivalence. Counts below partition each group into reachable, removed offer, left shop, blocked purchase, or unresolved pack.

| Panel | Group | States | Target reachable | Original offer removed | Left shop | Offer present, buy blocked | Unresolved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| training | All mismatches | 16 | 12 | 2 | 2 | 0 | 0 |
| training | Previously unscored alternative | 10 | 6 | 2 | 2 | 0 | 0 |
| training | Correct Joker rank, wrong full action | 8 | 5 | 1 | 2 | 0 | 0 |
| development | All mismatches | 24 | 16 | 5 | 3 | 0 | 0 |
| development | Previously unscored alternative | 19 | 11 | 5 | 3 | 0 | 0 |
| development | Correct Joker rank, wrong full action | 14 | 7 | 4 | 3 | 0 | 0 |

These are local reachability results on a small, already exposed panel. They do not show that the learned policy completes a reachable replacement, that a different purchase is harmless, or that losing this offer reduces eventual wins. Packs, vouchers, rerolls and different sales can change the build and have benefits outside this measurement. Pack continuation uses the teacher, not the learned model. No error is relabeled as safe and no original qualification gate is relaxed.

Three failed attempts are preserved. Version 1 attempted to serialize a string phase as an integer. Version 2 compared a clone after its cached action-mask refresh with a factual runner before that refresh. Version 3 fixed the sale comparison but left the same timing inconsistency between pack choices. Separate native diagnoses proved exact equality after matching refresh stages, with teacher actions, UID allocation and RNG unchanged. Version 4 refreshes masks consistently across phases and retains every full-state equality check. Cases, bounds, controls, tolerances and scientific plan are unchanged; partial outcomes were not used to choose corrections. The independent audit also compares the 57 preserved version-3 records with version 4 after removing only the mask-sensitive state fingerprints.

The independent audit verifies recorded identities, action chains, actual probe results and cache observation hashes; it does not implement a second simulator. No new fit, autonomous game evaluation or confirmation was run. Learned long-horizon generalization and increased win rate remain unproven.

Artifacts: [raw branches](../../eval/consequence_prediction_20260927/shop_opportunity_v4/records.jsonl), [native controls](../../eval/consequence_prediction_20260927/shop_opportunity_v4/result_v1.json), [independent accounting and subgroup counts](../../eval/consequence_prediction_20260927/shop_opportunity_v4/verification_v1.json), [whole-owner and verifier admission](../../eval/consequence_prediction_20260927/shop_opportunity_owner_terminal_v4.json), [first failed attempt](../../eval/consequence_prediction_20260927/shop_opportunity_failed_owner_terminal_v1.json), [second failed attempt](../../eval/consequence_prediction_20260927/shop_opportunity_failed_owner_terminal_v2.json), [sale mask-stage diagnosis](../../eval/consequence_prediction_20260927/shop_opportunity_mask_stage_debug_results_v1.json), [third failed attempt](../../eval/consequence_prediction_20260927/shop_opportunity_failed_owner_terminal_v3.json), [pack mask-stage diagnosis](../../eval/consequence_prediction_20260927/shop_opportunity_pack_mask_debug_results_v1.json).
