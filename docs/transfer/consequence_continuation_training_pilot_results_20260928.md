# The pilot produces terminal comparisons that immediate-clear labels miss

The eight-state training pilot completed all 24 continuations and passed an independent native replay and model audit. **Two states contain both winning and losing continuations. All four win/loss-separated action pairs have identical immediate-clear labels.** This establishes that the proposed collection can produce comparative terminal supervision absent from those immediate labels. It does not establish how common such comparisons are, whether a model can learn them, or whether a resulting policy improves wins.

Both informative cases are at Ante 2:

| Training state | Action and immediate score | Clears immediately? | Terminal result under the common base follow-up policy |
| --- | --- | --- | --- |
| 4,000,036 / step 19 | Teacher Two Pair, action 2519, score 1,023 | Yes | Loss at Ante 7 |
| Same state | Base Pair, action 2718, score 681 | Yes | Win |
| Same state | Additional Pair, action 4940, score 652 | Yes | Loss at Ante 6 |
| 4,000,055 / step 23 | Teacher and base Full House, action 4940, score 413 | No | Loss at Ante 4 |
| Same state | Additional Pair, action 2519, score 200 | No | Win |
| Same state | Additional Three of a Kind, action 534, score 234 | No | Loss at Ante 4 |

The second case contains a winning alternative beyond the teacher/base action, supporting its inclusion in label collection. Neither higher nor lower immediate score supplies a correct ranking of all three alternatives in these cases. All branches begin from the same original snapshot and RNG state, but their later paths and RNG consumption can differ. These are realized, controller-dependent outcomes, not conditional expected values or proof of strategic intent.

The other six states have a common terminal win/loss label across their three actions. Overall there are 11 winning and 13 losing branches, with zero censoring. These are labels for selected state/action interventions, not a policy's fresh-game win rate. No model is trained by this pilot.

| Pilot group | States | Mixed terminal-outcome states | Total rollout time, seconds |
| --- | ---: | ---: | ---: |
| Early, base differs from teacher | 2 | 1 | 26.167 |
| Early, base matches teacher | 2 | 1 | 14.819 |
| Later, base differs from teacher | 2 | 0 | 13.281 |
| Later, base matches teacher | 2 | 0 | 10.743 |

The [frozen pilot](consequence_continuation_training_pilot_20260928.md) used the previously selected targets without substitutions. Measurement took **112.000 seconds**, including 5.611 seconds of setup and 41.183 seconds of factual reconstruction. The separate audit took **117.156 seconds**. It reproduced all 11,012 original factual decisions across the first 56 training games and independently re-forwarded and replayed all **1,441 branch actions**. Every original immediate-outcome field, snapshot/observation hash, UID counter, trace state, boundary and terminal matched. Base model tensors remained unchanged with no gradients. The older outcome schema was supported without changing its saved records.

The complete owner actually exited zero in tool chunk `1b21e9` before admission `11ee42`. The preparation had separately checked 54 synthetic terminal/censoring combinations and both immediate-outcome schemas. The two 900-second phase budgets were not extended, and no restart occurred. No process remains live from this pilot.

The next collection can spread labels across the existing 1,000 training games using one prespecified decision per game and the same three-action rule. Reuse all eight pilot cases regardless of outcome; select the other games without new terminal labels. Estimate the collection budget from measured group runtimes and factual reconstruction cost before execution. A new fit, its exact objective, policy integration and evaluation protocol remain separate decisions. The existing gates, rejected endpoint, final win-rate and early-attribution requirements remain unchanged and unmet.

- [Measurement](../../eval/consequence_prediction_20260927/continuation_training_pilot_v1/result_v1.json), SHA256 `b5007f3cec8663102d0d007522a7331e51bc2cd4767394a6564faabe3e14da94`.
- [Independent audit](../../eval/consequence_prediction_20260927/continuation_training_pilot_v1/verification_v1.json), SHA256 `f9bdf48c79c7250382005c5734cad4c200e5594c7777a465deebe454f32cf33a`.
- [Owner](../../eval/consequence_prediction_20260927/continuation_training_pilot_owner_terminal_v1.json), SHA256 `94dd7bbc5f14401f340cfeda50ed434ca612cdab776717781f53974a6d541eaf`.
- [Protocol](../../eval/consequence_prediction_20260927/continuation_training_pilot_protocol_v1.json), SHA256 `f23a4a13ed1217c6ae5112891b9511dc300dc6efbfeacfd5868a91fcaf10b89f`.
