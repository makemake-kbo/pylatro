# Learning in the coupled sale readout

Post-hoc diagnosis of the rejected single fit. This does not change its failed screen or qualify it for gameplay. All rows come from exposed training/development teacher states. No additional fit, model forward/backward, optimizer step or simulator call was used.

The frozen source includes the new module in the optimizer and applies corrected logits before action NLL. Zero output initialization initially blocks gradients to its input projections; it does not block the output layer. The original first-update bias gradient is independently checked. These observations do not establish useful conditional parameter gradients throughout training.

## Saved action-score geometry

Metrics below are means over active states. Score contrast RMS removes the common shift across legal owned Jokers. Proposal contrast measures the change from purchase-weighted to uniformly weighted offers after removing that common shift.

| Split / group | Active / multiple offers | Common score | Score contrast RMS | Original sale-logit contrast RMS | Pair interaction RMS | Proposal contrast RMS | Conditional winner / flat action changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| development / all_shop | 7512 / 3773 | 0.00487803185 | 0.000429403927 | 2.35072748 | 1.94772248e-05 | 1.55839577e-05 | 0 / 1 |
| development / shop_non_sale | 6809 / 3288 | 0.00504141678 | 0.000437078991 | 2.34114442 | 1.85056098e-05 | 1.44448522e-05 | 0 / 0 |
| development / shop_sale | 703 / 485 | 0.00329554545 | 0.000355066071 | 2.44354552 | 2.88879311e-05 | 2.66169155e-05 | 0 / 1 |
| training / all_shop | 57257 / 28517 | 0.00470753838 | 0.000429780287 | 2.41363586 | 1.9130953e-05 | 1.53733588e-05 | 4 / 6 |
| training / shop_non_sale | 51877 / 24902 | 0.00486469323 | 0.000437396021 | 2.40273332 | 1.82009223e-05 | 1.43021666e-05 | 4 / 5 |
| training / shop_sale | 5380 / 3615 | 0.00319216248 | 0.000356345085 | 2.51876431 | 2.80988341e-05 | 2.5702399e-05 | 0 / 1 |

Pair interaction is the remaining score after removing uniform owned, offer and grand-mean effects. This is an algebraic property of a fixed contextual representation. It does not show that the encoder lacks offer information. Uniform proposal weighting is a score diagnostic, not an evaluated environment intervention.

## Downstream action-loss derivatives

| Split / group | Residual gradient L2 | Family component L2 | Conditional component L2 | Purchase-logit gradient L2 | Conditional purchase gradient L2 | Conditional gradient dot residual |
|---|---:|---:|---:|---:|---:|---:|
| development / all_shop | 0.111118353 | 0.0676617906 | 0.0651164375 | 1.67180594e-06 | 1.52062726e-07 | 3.11046285e-06 |
| development / shop_non_sale | 0.0373001318 | 0.0373001318 | 0 | 8.72650364e-07 | 0 | 0 |
| development / shop_sale | 0.826094553 | 0.361733675 | 0.695810354 | 9.41213355e-06 | 1.62488649e-06 | 3.32372645e-05 |
| training / all_shop | 0.0957844378 | 0.0668183732 | 0.0379025184 | 1.71399921e-06 | 8.59109045e-08 | 1.63009044e-06 |
| training / shop_non_sale | 0.0351287921 | 0.0351287921 | 0 | 9.12277834e-07 | 0 | 0 |
| training / shop_sale | 0.680660448 | 0.37238722 | 0.403380018 | 9.44464972e-06 | 9.1431239e-07 | 1.73483436e-05 |

These are float64 derivatives in output space, reconstructed from saved logits and pair scores, with no batch-denominator scaling. Their conditional component sums to zero over owned slots. A positive dot product means increasing the current conditional correction locally raises conditional NLL at this endpoint; it does not identify an effective parameter update. Shared parameter Jacobians and the training trajectory are not observed by this calculation.

## Checkpoint movement

| Added tensor | Initial RMS | Final RMS | Displacement RMS | Fraction changed | Epsilon-dominated fraction | Last nominal Adam update RMS |
|---|---:|---:|---:|---:|---:|---:|
| offer_proj.weight | 0.0293412893 | 0.0293683218 | 0.00124288316 | 1 | 0 | 7.94725792e-06 |
| output.bias | 0 | 0.00031452658 | 0.00031452658 | 1 | 0 | 4.69402769e-06 |
| output.weight | 0 | 0.000476709892 | 0.000476709892 | 1 | 0 | 5.08803657e-06 |
| owned_proj.weight | 0.0294397787 | 0.0294655958 | 0.0012325554 | 1 | 0 | 7.40750931e-06 |
| state_proj.bias | 0.0484905363 | 0.0488078498 | 0.00140382476 | 1 | 0 | 7.87714994e-06 |
| state_proj.weight | 0.0508427812 | 0.0508416178 | 0.00119687461 | 1 | 0.0124511719 | 7.75439845e-06 |

All six tensors map uniquely to saved trainable names and optimizer states at step1,000. Initializer seed0 and the frozen source reproduce their initialization. Epsilon dominance means sqrt of the bias-corrected second moment is below Adam epsilon. The last nominal update is calculated from endpoint moments; it is not a reconstruction of historical updates or proof of training convergence.

Validation: 456,372 paired record identities, 18 previous component aggregates, all prior conditional correctness counts, pair decomposition identities and 18 finite-difference derivatives pass. Maximum finite-difference error 1.57134306e-11; saved-residual reconstruction error 1.66255811e-09. Runtime 12.77s. Checkpoints and all inputs were re-pinned after inspection. Full descriptive quantiles and squared-energy totals are in the result JSON.

Artifacts: [replacement_relation_learning_diagnostic_20260929.md](replacement_relation_learning_diagnostic_20260929.md), [result_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_learning_v1/result_v1.json), [replacement_relation_learning_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_relation_learning_owner_terminal_v1.json).
