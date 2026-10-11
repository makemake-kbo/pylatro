# Broader-data endpoint supports investigating feature adaptation

At the converged 5,000-game output endpoint, the remaining readout training gradient points toward lower development conditional loss. The training/development cosine is **+0.3070**, and the negative training gradient has development derivative **−0.00006472** per original development row, with descriptive source-game 95% interval **[−0.00010502, −0.00002603]**. The corresponding derivative at the earlier 1,000-game endpoint was positive, with an interval spanning zero.

This supports a controlled feature-adaptation experiment. It is a local derivative at one fixed endpoint, not a finite update, trained-policy improvement, causal attribution solely to data quantity, or a win-rate result. Both the training population and fitted output endpoint differ from the earlier diagnostic. Development has already been exposed.

The direction is computed only from all 26,782 active training sale states, normalized by the original 1,013,555 corpus rows. The development calculation uses all 703 active sales, normalized by 26,045 original rows. No gradient component or step size is chosen using development. The full readout has 57,473 parameters; the 64 output weights are stationary, and the common output bias is unidentifiable for conditional choice.

| Feature parameter | Training/development gradient cosine |
|---|---:|
| Owned-Joker projection | +0.3161 |
| Offered-Joker projection | +0.2938 |
| State projection weight | +0.2483 |
| State projection bias | +0.2446 |

All four feature components align positively. Output-gradient cosine is uninterpretable near training stationarity: its norm is only 1.14e-10; the bias gradient is numerical zero. The full training gradient norm is 0.007616. The direction decreases training loss locally by 0.00005800 per training row. Current-Ante early and later development contributions are both negative, but these subgroups do not replace final first-passage early-policy attribution.

The companion saved-output diagnostic shows little benefit from weighting offers by the predicted purchase. Learned versus uniform offer weighting yields 377 versus 376 correct conditional development choices, changing six decisions (two gains, one loss). Its conditional NLL difference is +0.001882, with game interval [−0.003575, +0.007369]. The explicit pair-interaction energy is below 1% of total centered pair-score energy. These contextual features already contain offer information, so uniform weighting does not remove all offer information or isolate architectural value.

All 26,782 training losses reconstruct the independently audited native endpoint, and all 703 development rows reconstruct native pair scores and joint likelihood. Maximum native logit error is 8.43e-6; maximum training NLL error is 5.38e-6; development NLL error is 4.85e-6. The output gradient agrees with separate convex-readout arithmetic to 4.05e-19. Thirty Richardson finite-difference checks pass the unchanged 5e-9 bound (maximum error 2.73e-10). The process took 12.79 seconds, made no model forward or optimizer step, and exited zero before admission.

The broader-data candidate remains rejected by seven of thirteen development checks. No gameplay or independent confirmation is authorized by these diagnostics alone; any subsequent feature-adapted candidate must pass the original screen and final proof bar.

[Diagnostic plan](conditional_diversity_gradient_20260930.md) · [Admitted result](../../eval/consequence_prediction_20260927/conditional_diversity_gradient_v1/result_v1.json) · [Actual-exit receipt](../../eval/consequence_prediction_20260927/conditional_diversity_gradient_owner_terminal_v1.json) · [Proposal-weighting diagnostic](../../eval/consequence_prediction_20260927/conditional_diversity_interactions_results_v1.json) · [Broader-data candidate result](conditional_output_diversity_results_20260930.md)
