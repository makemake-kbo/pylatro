# Training fit is compatible with interpolation; context coverage remains sparse

The completed [capacity and context diagnostic](consequence_terminal_capacity_context_20260928.md) shows that the conditional readout has enough unconstrained linear capacity to assign arbitrary relative scores to its represented training states. It also identifies gaps in consequential context coverage. Neither finding establishes the cause of validation failure or demonstrates a policy improvement. No model was fitted, simulated game run, observation changed or candidate promoted.

The preceding [conditional readout](consequence_terminal_conditional_results_20260928.md) orders **327/332 training terminal pairs correctly (98.49%)**, but only **47/84 validation pairs (55.95%)**, versus base **50/84 (59.52%)**. Only **166 of 800 training states** have mixed terminal outcomes and contribute to its conditional objective.

For each state, subtract proposal 0's frozen RMS feature vector from proposals 1 and 2. The active training matrix has **332 rows, 384 columns and rank 332**. Its smallest singular value is **0.1503**, above both the float64 rank threshold and the **0.00384** FP32-resolution diagnostic. Consequently, an unregularized linear score could interpolate any assignment of these relative scores. The actual fit uses a ridge penalty; this analysis neither fits an interpolant nor establishes that the labels are random.

| Contrast population | States | Matrix shape | Numerical rank | Condition number | Dimensions for 90% / 99% energy |
|---|---:|---:|---:|---:|---:|
| All training states | 800 | 1,600 × 384 | 384 | 1,405.04 | 181 / 317 |
| Mixed training states | 166 | 332 × 384 | 332 | 558.66 | 105 / 208 |
| Mixed validation states | 42 | 84 × 384 | 84 | 26.90 | 39 / 66 |

Ranks agree under both numerical thresholds. The fitted coefficient lies within the active training row space to numerical precision: its relative residual norm is **3.17×10⁻¹⁵**. Directions containing 90% of training contrast energy contain only **30.99%** of coefficient energy; directions containing 99% contain **79.88%**. These are descriptive measurements, not a selected rank cutoff or evidence that another regularization sweep would improve transfer.

The observation audit preserves ordered Joker records, current-hand card properties, selected and retained slots, and resource scalars for all 1,000 states. Every validation Joker identity occurs somewhere in the 800 training states. However, **90/200 validation states** contain a Joker pair absent from those training states. Against the 166 states supplying conditional supervision, **27/200** contain an unseen individual Joker and **160/200** an unseen Joker pair.

Coverage among the **42 mixed validation states**, where terminal ordering can be assessed:

| Missing training support | Against all 800 training states | Against 166 mixed training states |
|---|---:|---:|
| At least one individual Joker | 0/42 | 5/42 |
| At least one Joker pair | 20/42 | 31/42 |
| At least one Joker/card-count conjunction | 14/42 | 32/42 |

A card-count conjunction means a present Joker identity, visible rank or suit category, and magnitude of the selected-card count difference between proposals. These signatures are defined without outcomes; the mixed-training reference restricts support to terminal-different pairs. They are generic coverage indicators, not assertions that each Joker mechanically interacts with each card category. Unknown card properties remain unknown. Joker-pair counts ignore order, while saved observation records retain it.

Coverage does not cleanly separate errors. Against all training states, conditional ordering is **55.00%** with an unseen Joker pair and **56.82%** without one. Against mixed training states, it is **54.69%** with an unsupported card-count conjunction and **60.00%** without one. These small descriptive partitions have no confidence intervals and do not identify a causal bottleneck. Exact repeated inventories are not required for successful compositional generalization.

The useful conclusion is narrower: near-perfect training ordering does not establish reusable consequence learning, and coarse action-shape coverage does not establish coverage of relevant interactions. A further experiment needs to distinguish insufficient consequential examples, label variability and representation limitations. No next fit, data collection or policy integration is selected by this diagnostic.

The first execution failed because a coverage dictionary overwrote a Boolean field. Its source, protocol, receipt and partial arrays remain preserved. The [documented correction](consequence_terminal_capacity_context_correction_20260928.md) separates `mixed_outcome` from the `mixed` coverage dictionary without changing the scientific definitions. The corrected execution completed successfully; spectrum reconstruction and energy identities, score reconstruction, source identities, action membership and prior validation counts passed. A separate standard-library recount checks the saved context signatures and coverage partitions before this report is pinned.

All source games were used by the frozen feature fit, and the terminal-label validation split has been adaptively reused. Original-RNG labels describe realized outcomes under the shared base continuation, not known expected action values. The full fresh-game generalization, retention and final independent-confirmation requirements remain unmet. Draw-pile ordering was already documented in the earlier information audit; it is not a new discovery or an observation-preserving invariance test.

Artifacts: [successful owner](../../eval/consequence_prediction_20260927/terminal_capacity_context_owner_terminal_v2.json), [frozen protocol](../../eval/consequence_prediction_20260927/terminal_capacity_context_protocol_v2.json), [results](../../eval/consequence_prediction_20260927/terminal_capacity_context_v2/result_v1.json), and [saved contexts](../../eval/consequence_prediction_20260927/terminal_capacity_context_v2/contexts.jsonl).
