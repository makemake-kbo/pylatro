# Which parameter changes produce the continuation regression?

**Changed encoder features are sufficient to reproduce the shared fatal choice with the old correction head still installed.** Restoring the base encoder to each of the four continuation endpoints restores the winning choice on seed 850000064 / step 138. Restoring only the correction head does not. Head updates amplify the error, but are not required for it. This is a local parameter intervention on an already exposed case, not evidence that globally freezing the encoder improves training or wins.

The [preceding trace](continuation_regression_trace_20260925.md) found that further fitting overturns an already correct decision: a 13,296-point Pair becomes a losing 6,313-point High Card against an 11,000-point target with one hand remaining. The learned reference distribution still favors the Pair, while the subset correction reverses that preference. The present experiment separates three groups of actual saved parameters in memory:

- **E:** token embedding and Transformer backbone.
- **P:** global, scalar and danger projections used to form the shared state vector.
- **H:** the subset correction head, including its card projection, state projection, count embedding and scoring layers.

Each group comes from either the base (0) or the corresponding continuation endpoint (1). Every other parameter stays at that endpoint, including the other action heads. All eight combinations are measured for each of the two ordinary and two weighted continuations. **000 is not the complete base policy:** its correction equals the base correction, but its remaining action-head weights come from the endpoint. The complete base is measured separately. E and P also affect the reference and other policy outputs; only H-only swaps leave those other outputs unchanged.

The primary table reports the final log-probability difference, **winning Pair minus fixed losing High Card**. Positive favors the Pair. The full base difference is +0.497601. The action in the last column is the full policy mode, not merely the winner of the two-action comparison.

| EPH | Component setting | Ordinary 0 | Ordinary 1 | Weighted 0 | Weighted 1 | Mode / known outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 000 | Base E/P/H; endpoint remainder | +0.464735 | +0.445799 | +0.411789 | +0.392472 | 415 / clear |
| 001 | New head only | +0.217903 | +0.190388 | +0.167207 | +0.157410 | 415 / clear |
| 010 | New projections only | +0.411913 | +0.402262 | +0.361367 | +0.350712 | 415 / clear |
| 011 | Old encoder; new projections/head | +0.152943 | +0.140547 | +0.104305 | +0.104090 | 415 / clear |
| 100 | New encoder only | -0.126735 | -0.213622 | -0.509735 | -0.957467 | 434 / loss |
| 101 | New encoder/head; old projections | -0.525077 | -0.659092 | -0.941131 | -1.469609 | 434 / loss |
| 110 | Old head; new encoder/projections | -0.191669 | -0.277743 | -0.582268 | -1.031817 | 434 / loss |
| 111 | Unchanged continuation endpoint | -0.598327 | -0.729034 | -1.029887 | -1.561088 | 434 / loss |

With the base encoder, all four P/H combinations keep the winning mode in all four continuations. With the new encoder, every P/H combination chooses the losing action. In particular, 100 versus 000 changes only E and is sufficient to cross the decision boundary; 011 versus 111 restores only E and reverses the error. The learned reference favors the Pair at all 32 primary vertices, so the correction still supplies the reversal with the old head. This rules out correction-head weight updates as a necessary condition for this particular regression under the tested endpoint remainder.

The next table decomposes the change in **correction energy difference**, rather than full action probability. It uses base-anchored finite differences: E = f(100) − f(000), EP = f(110) − f(100) − f(010) + f(000), and the corresponding terms for the other groups. All seven terms sum to f(111) − f(000). Negative values push toward the losing action. Interactions are retained; these are not percentages of uniquely attributable causal credit.

| Continuation | E | P | H | EP | EH | PH | EPH | Total energy change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ordinary 0 | -0.606444 | -0.046011 | -0.246831 | -0.010229 | -0.151510 | -0.012138 | +0.003822 | -1.069342 |
| Ordinary 1 | -0.650404 | -0.037227 | -0.255411 | -0.021681 | -0.190058 | -0.006303 | +0.000482 | -1.160603 |
| Weighted 0 | -0.732523 | -0.049211 | -0.244582 | -0.016687 | -0.186814 | -0.012480 | -0.003743 | -1.246039 |
| Weighted 1 | -0.841518 | -0.040761 | -0.235062 | -0.029131 | -0.277080 | -0.011561 | -0.005568 | -1.440681 |

The E term is the largest negative individual term at this state in every continuation. H is also negative, and the E–H interaction makes the combined change more harmful than their separate base-anchored changes. P contributes a smaller negative term here. The result supports a specific retention failure: changed representations make even the previously saved scoring head favor the wrong hand. It does not locate the responsible embedding fields, attention layers, or the card-feature versus pooled-state route within E.

All eight selected fatal cases remain in the analysis. The table gives each hybrid's mode and its previously replayed one-step outcome: C = blind cleared, L = immediate loss, S = survives without clearing, ? = that exact action has not been replayed. These are choices on fixed teacher states, not autonomous game win rates.

| Continuation | Seed / step | 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ordinary 0 | 850000015 / 28 | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L |
| Ordinary 0 | 850000027 / 121 | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L |
| Ordinary 0 | 850000042 / 113 | 534 C | 5600 L | 534 C | 5600 L | 534 C | 534 C | 534 C | 534 C |
| Ordinary 0 | 850000064 / 138 | 415 C | 415 C | 415 C | 415 C | 434 L | 434 L | 434 L | 434 L |
| Ordinary 0 | 850000076 / 120 | 2519 L | 2519 L | 4940 C | 2519 L | 2519 L | 2519 L | 4940 C | 2519 L |
| Ordinary 0 | 850000077 / 77 | 6105 L | 6105 L | 6105 L | 6105 L | 4989 C | 4989 C | 4989 C | 4989 C |
| Ordinary 0 | 850000093 / 57 | 699 L | 3884 C | 699 L | 3884 C | 3884 C | 3884 C | 3884 C | 3884 C |
| Ordinary 0 | 850000096 / 130 | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L |
| Ordinary 1 | 850000015 / 28 | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L |
| Ordinary 1 | 850000027 / 121 | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L |
| Ordinary 1 | 850000042 / 113 | 534 C | 5600 L | 534 C | 5600 L | 534 C | 534 C | 534 C | 534 C |
| Ordinary 1 | 850000064 / 138 | 415 C | 415 C | 415 C | 415 C | 434 L | 434 L | 434 L | 434 L |
| Ordinary 1 | 850000076 / 120 | 2519 L | 2519 L | 4940 C | 2519 L | 4940 C | 4940 C | 4940 C | 4940 C |
| Ordinary 1 | 850000077 / 77 | 6105 L | 6105 L | 6105 L | 6105 L | 6105 L | 6635 L | 6105 L | 6635 L |
| Ordinary 1 | 850000093 / 57 | 699 L | 3884 C | 699 L | 3884 C | 989 C | 989 C | 989 C | 989 C |
| Ordinary 1 | 850000096 / 130 | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L |
| Weighted 0 | 850000015 / 28 | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 5600 C | 5600 C | 5600 C |
| Weighted 0 | 850000027 / 121 | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L |
| Weighted 0 | 850000042 / 113 | 534 C | 5600 L | 534 C | 5600 L | 534 C | 534 C | 534 C | 534 C |
| Weighted 0 | 850000064 / 138 | 415 C | 415 C | 415 C | 415 C | 434 L | 434 L | 434 L | 434 L |
| Weighted 0 | 850000076 / 120 | 2519 L | 2519 L | 2519 L | 2519 L | 4940 C | 4940 C | 4940 C | 4940 C |
| Weighted 0 | 850000077 / 77 | 6105 L | 6105 L | 6105 L | 6105 L | 4989 C | 4989 C | 4989 C | 4989 C |
| Weighted 0 | 850000093 / 57 | 699 L | 3884 C | 699 L | 3884 C | 1166 ? | 3884 C | 1166 ? | 3884 C |
| Weighted 0 | 850000096 / 130 | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L |
| Weighted 1 | 850000015 / 28 | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L |
| Weighted 1 | 850000027 / 121 | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L | 534 L |
| Weighted 1 | 850000042 / 113 | 534 C | 5600 L | 534 C | 5600 L | 534 C | 534 C | 534 C | 534 C |
| Weighted 1 | 850000064 / 138 | 415 C | 415 C | 415 C | 415 C | 434 L | 434 L | 434 L | 434 L |
| Weighted 1 | 850000076 / 120 | 2519 L | 2519 L | 2519 L | 2519 L | 4940 C | 4940 C | 4940 C | 4940 C |
| Weighted 1 | 850000077 / 77 | 6105 L | 6105 L | 6105 L | 6105 L | 6635 L | 4989 C | 6635 L | 6635 L |
| Weighted 1 | 850000093 / 57 | 699 L | 3884 C | 699 L | 3884 C | 699 L | 3884 C | 699 L | 3884 C |
| Weighted 1 | 850000096 / 130 | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L |

The other cases show why a blanket encoder rollback is not justified. At 850000042 / 113, the old encoder with the updated projections/head (011) chooses the losing action in all four continuations; their full endpoints (111) clear. At 850000093 / 57, changing only H from 000 to 001 changes loss to clear in all four continuations. The weighted replica-0 modes at 100 and 110 on that case choose action 1166, whose outcome is unknown in this diagnostic. No outcome has been inferred for it. The full JSON retains all 32 case-level factorial decompositions and all 256 vertex records, including these mixed effects.

The [protocol](../../eval/correction_component_attribution_20260925/protocol_v1.json), [measurement helper](../../eval/correction_component_attribution_20260925/measure_v1.py), [factor algebra](../../eval/correction_component_attribution_20260925/factorial_v1.py) and [independent verifier](../../eval/correction_component_attribution_20260925/verify_v1.py) were fixed before the intervention outcomes were inspected. Five unchanged controls reproduced all 1,280 saved batch predictions exactly, with maximum teacher-NLL difference zero. The experiment then evaluated 28 actual hybrids using the same original 32-row batches. No new checkpoint was trained or selected.

The [verification](../../eval/correction_component_attribution_20260925/verification_v1.json) passes 224,123 checks over 8,448 batch predictions, 264 actual case measurements and 128,865 legal-support entries. It independently loads all five donor checkpoints on CPU, hashes every tensor, and verifies every hybrid tensor assignment against hashes collected from the actual MPS models. It checks cached identities and legality, complete-support normalization, ranks, probabilities, factorial algebra and known-outcome bindings. Feature hashes show that H-only swaps preserve every non-energy output; 000 exactly reproduces both base correction channels despite the endpoint remainder. Known-polynomial and constant-function checks validate the factor algebra. This verification does not run a second neural forward pass.

Both the [measurement terminal](../../eval/correction_component_attribution_20260925/measurement_terminal_v1.json) and [verification terminal](../../eval/correction_component_attribution_20260925/verification_terminal_v1.json) record actual exit code zero. All jobs and their owned sleep guards have ended. [Machine-readable evidence](data/correction_component_attribution_20260925.json) preserves the complete findings and source hashes.

This closes the narrow question of whether correction-head updates alone caused the shared regression: encoder changes can produce it without those updates, and encoder/head interactions worsen it. It does not explain the whole original pretraining gap. Hybrid parameters may lie off the training trajectory, the cases were selected for errors, and restoring a checkpoint component is not equivalent to retraining with that component frozen. The subsequently completed [full-panel diagnostic](component_panel_20260925.md) quantifies both losses and gains on all 26,045 teacher decisions. Encoder rollback improves likelihood broadly but removes many newly learned matches; every rollback remains below the base overall. A training intervention still requires design and matched fresh-outcome testing. World models remain deferred; replicated fresh long-horizon generalization remains unproven.
