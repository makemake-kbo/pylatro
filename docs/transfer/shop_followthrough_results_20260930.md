# Competing purchases do not lead to a sale on the next model decision

In this fixed diagnostic panel, neither the candidate nor the original base predicts the planned Joker sale immediately after a competing purchase. The teacher still requests that original sale in all ten development cases and five of six training cases. The remaining training case changes to a purchase, which both models predict correctly.

| Panel / model | Cases | Original sale next | Another purchase | Reroll | Leave | Fresh teacher agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Development / candidate | 10 | 0 | 7 | 3 | 0 | 0 |
| Development / base | 10 | 0 | 7 | 3 | 0 | 0 |
| Training / candidate | 6 | 0 | 4 | 0 | 2 | 1 |
| Training / base | 6 | 0 | 4 | 0 | 2 | 1 |

The candidate and base choose exactly the same post-purchase action in all sixteen cases, including the specific purchase action. This weakens the explanation that the original disagreements were simply one-step purchase-order differences. It does not establish that the seven further development purchases would prevent a later replacement, or that following the teacher would improve eventual outcomes.

The probability decomposition is more specific than action agreement alone. In all ten development states, the total probability of the Joker-sale category is below the probability of the best individual action outside that category. Concentrating all sale probability on the correct Joker would still not make it the model's preferred action. The candidate ranks the teacher's intended Joker first within the sale category in four of ten cases, compared with three for the base. All five training states where the teacher still requests the original sale have the same category-level limitation; both models rank the intended Joker first in four.

The correction remains eligible in all sixteen changed states. The candidate increases total sale probability relative to the base in seven of ten development cases and four of six training cases, but changes none of the next actions. These findings locate an immediate decision limitation in competition between action categories. They do not establish whether the underlying cause is missing features, weak supervision, training-state coverage, or inadequate readout capacity. A constant increase to sale preference is not justified by this panel.

## What was checked

The [fixed protocol](shop_followthrough_20260930.md) includes every competing SHOP_BUY alternative from the admitted eight-game opportunity study: six training and ten exposed-development cases. State reconstruction reproduced all fourteen cached observation fields across 1,736 factual decisions, then replayed the already recorded purchase and pack-resolution actions on copies. Every recorded branch fingerprint and UID counter matched. Before/after observations were saved in full; the independent audit checked 32 observation hashes and 224 original cached fields. Original and instrumented teacher queries agreed while preserving state, UID allocation and RNG.

Both existing checkpoints ran in the frozen native MPS runtime, evaluation mode, temperature 1 and hand mixture 0.5. Their original modes and NLL reproduced exactly across 416 control rows each, using the original contiguous 32-row cache batches. Each model then made sixteen fixed-state predictions. Both parameter hashes remained unchanged. A separate NumPy audit rebuilt all 288 legal-action log probabilities from the saved macro and conditional logits; its maximum error was 1.407e-6, below the unchanged 2e-5 tolerance. State collection took 27.71 native seconds; prediction and its controls took 18.08 seconds. Both whole owners and both independent auditors exited successfully.

Pack decisions were resolved by the fixed teacher before the changed states were presented to the models. The new predictions were never executed. This is a local diagnostic on a small, already exposed panel selected from candidate mistakes, not an autonomous rollout, an unbiased model comparison, a training experiment, or a win-rate estimate. The candidate remains rejected by the original qualification checks. Learned long-horizon improvement remains unproven.

## Evidence

- [State owner and audit receipt](../../eval/consequence_prediction_20260927/shop_followthrough_states_owner_terminal_v1.json)
- [Reconstructed identities and fresh teacher comparisons](../../eval/consequence_prediction_20260927/shop_followthrough_states_v1/records.jsonl)
- [Prediction owner and audit receipt](../../eval/consequence_prediction_20260927/shop_followthrough_predict_owner_terminal_v1.json)
- [Original controls and runtime measurements](../../eval/consequence_prediction_20260927/shop_followthrough_predict_v1/result_v1.json)
- [All next-action predictions and probabilities](../../eval/consequence_prediction_20260927/shop_followthrough_predict_v1/predictions.jsonl)
- [Independent reconstruction of action probabilities](../../eval/consequence_prediction_20260927/shop_followthrough_predict_v1/verification_v1.json)
- [Sale ranking and action-category decomposition](../../eval/consequence_prediction_20260927/shop_followthrough_analysis_results_v1.json)
