# Shop teacher inputs and observed representation limits

The source check does not support a simple explanation that the shop teacher predicts actual hidden draws. Its scoring routine samples public deck compositions and replaces the run RNG with separate `SHOP_SAMPLE` RNGs. One narrower dependency remains: relative creation order determines which distinguishable copy of a same-rank, same-suit card enters a deterministic sample. Changing absolute identifiers or the input list order does not change the sample.

An isolated test of the exact sampling method changed one of twelve samples after swapping the relative identifiers of a base and enhanced Ace of Spades in a synthetic 53-card deck. Identical copies were invariant. This is a sampling difference, not a demonstrated teacher-action difference or an equivalence between complete model observations. [Source-method audit](../../eval/consequence_prediction_20260927/shop_sample_source_audit_v1.json).

The census covers every state where the current replacement branch is active: SHOP, exhausted Joker capacity, at least one offered Joker and at least one legal owned-Joker sale. Teacher sales and non-sales are both included.

| Observed condition | Original training (57,257) | Additional training (228,787) | Development (7,512) |
| --- | ---: | ---: | ---: |
| All 62 deck slots occupied | 2,871 (5.01%) | 12,359 (5.40%) | 362 (4.82%) |
| At least one masked rank or suit | 1,519 (2.65%) | 6,345 (2.77%) | 197 (2.62%) |
| Visible same-front copies with different represented variants | 50,241 (87.75%) | 201,762 (88.19%) | 6,606 (87.94%) |
| Permanent-bonus bucket at its cap | 0 | 0 | 0 |

A full set of 62 slots can represent exactly 62 cards, so occupancy alone does not establish truncation. Distinguishable duplicates make the sampler dependency relevant to investigate, but do not establish a dependence of teacher actions on information missing from the complete observation. Other scalar summaries and history also enter the model. Likewise, masked card tokens do not by themselves prove that all information about those cards is absent. The macro head receives economy information through the encoder's projected META token; these scalars are not simply missing inputs.

Before changing the observation schema or teacher, the useful next check is a matched-state counterexample: identical complete policy inputs, changed teacher action, and a measured occurrence rate in the corpus. None of those three has been established here. These findings do not explain the imitation gap or demonstrate a win-rate effect.

The census and its independent verification exited successfully. All used cache arrays were rehashed; group partitions were checked; an independent card-set implementation matched 512 fixed positions in each of the three panels. No model, simulator, training process or current sale-family feature output was used. [Admitted census](../../eval/consequence_prediction_20260927/shop_observation_exposure_owner_terminal_v1.json) · [Results](../../eval/consequence_prediction_20260927/shop_observation_exposure_results_v1.json) · [Verification](../../eval/consequence_prediction_20260927/shop_observation_exposure_verification_v1.json).
