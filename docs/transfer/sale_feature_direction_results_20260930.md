# The separate sale-feature learning direction does not transfer reliably

The predeclared full-panel diagnostic does **not** qualify the proposed independently learnable sale-feature branch for a fit. The training-derived direction has a development-loss derivative of **−9.630e−7**, with a complete-game bootstrap 95% interval of **[−1.017e−5, +8.591e−6]**. The interval crosses zero. The cosine between training and development gradients is **0.0114**, indicating little alignment in these feature coordinates at this starting point.

| Development component | Loss derivative along the fixed training direction |
| --- | ---: |
| All active states | −9.630e−7 |
| Teacher sells | −5.564e−5 |
| Teacher does not sell | +5.468e−5 |
| Current Ante < 4 | +5.578e−7 |
| Current Ante ≥ 4 | −1.521e−6 |

Negative values mean a local loss decrease. The favorable derivative on teacher-sale states is almost canceled by the adverse derivative on non-sale states. The early/later split describes current observations; it is not an early-policy intervention or evidence about eventual wins. These components do not define alternative directions or new eligibility rules.

The hypothesis was specific: the current 193-parameter family readout uses frozen replacement features originally learned for conditional Joker ranking. A separate copy of four projection tensors could learn when to sell while leaving the encoder, conditional ranking, pooling probabilities, and linear family readout fixed. The diagnostic differentiates those 57,408 coordinates without updating any parameter. Its result gives no reliable local support for training this copy. It does not prove that every separate feature architecture, representation change, or richer supervision would fail.

The linear readout remains effectively stationary: its independently reconstructed training gradient infinity norm is **3.818e−9**. Thus this result is not explained by an unfinished linear optimization. The hypothetical feature-gradient norms are 0.002282 on training and 0.037006 on development; the norms and derivative magnitudes depend on this parameterization and normalization and should not be compared directly with other architectures.

## Coverage and verification

The native run examined every previously admitted active state: **286,044 training states from 5,000 games** and **7,512 development states from the exposed 128-game panel**. Training includes 26,782 sale labels and 259,262 non-sale labels; development includes 703 and 6,809 respectively. Both use the original per-decision weighting and complete-corpus denominators. Only the combined training gradient defines the direction. The interval uses 10,000 bootstrap resamples of all 128 complete development games, including games with no active states.

The current checkpoint remained unchanged. Native features and parent macro logits exactly reproduced the previously admitted collection. Independent float64 initial-function errors were at most 4.268e−6 for features, 5.723e−6 for family corrections, and 1.133e−6 for sale probabilities, all below the unchanged 2e−5 bound. The saved-result audit reproduced identities, group projections and bootstrap endpoints. On the three retained actual contexts, analytic derivatives agreed with autograd to 1.111e−15 and finite differences to 1.985e−11. The whole run took **795.21 active seconds**; the native owner and independent auditor both exited successfully.

No branch preflight, packing Torch check, optimizer step, new policy checkpoint, autonomous game evaluation, or confirmation run followed this result. The prepared branch and checks are preserved as unexecuted artifacts, with a [stop receipt](../../eval/consequence_prediction_20260927/sale_feature_proposal_stopped_v1.json). No block selection, sign flip, step-size sweep, or alternative bootstrap threshold is used to rescue the failed direction. The original qualification checks and final generalization/win-rate requirements remain unchanged.

## Independent preparation findings

A read-only mask census found 2,246,806 legal Joker–offer pairs in the training panel, versus 22,883,520 pairs in the padded representation. Exact compact arrays would occupy **3,056,302,424 bytes**, compared with **8,077,024,428 bytes** padded. Pure NumPy packing tests preserve entity order, pair weights and partition boundaries. The Torch implementation has not been run, so these counts establish potential storage and pair-work savings, not an observed training speedup.

There are **zero states with only one legal Joker to sell** in these panels. The singleton example used during derivative testing is a mathematical illustration of the difference between ranking and deciding whether to sell; it is not an observed explanation of this corpus's failure. The full-panel result, rather than that illustration, determines the stopped fit proposal.

Sources: [frozen diagnostic protocol](sale_feature_direction_20260930.md), [whole-owner and auditor receipts](../../eval/consequence_prediction_20260927/sale_feature_direction_owner_terminal_v1.json), [native results and preservation checks](../../eval/consequence_prediction_20260927/sale_feature_direction_v1/result_v1.json), [independent audit and component derivatives](../../eval/consequence_prediction_20260927/sale_feature_direction_v1/verification_v1.json), [per-state development derivatives](../../eval/consequence_prediction_20260927/sale_feature_direction_v1/development_derivatives.jsonl), [packing census](../../eval/consequence_prediction_20260927/sale_feature_packing_census_results_v1.json).
