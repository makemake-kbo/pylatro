# Sale-family calibration across positive and negative decisions

The frozen policy is not generally biased against selling on the exposed development panel. Its probability sum predicts about 708 sales across 7,512 eligible shop states, versus 703 teacher sales. The earlier oracle restriction therefore does not justify a global increase in sale probability.

| Panel | Eligible states | Teacher sales | Expected sampled sales | Binary NLL | Brier loss | AUC |
|---|---:|---:|---:|---:|---:|---:|
| Original 1,000 training games | 57257 | 5380 | 5280.328 | 0.142984 | 0.042385 | 0.953997 |
| 128 development games | 7512 | 703 | 707.959 | 0.157807 | 0.045589 | 0.941827 |

These are all states where the conditional branch is active, including non-sales. They are probability sums on fixed teacher states, not counts from sampled trajectories. All source-game outcomes are retained. Additional 4,000-game negative-state predictions are not part of this calibration result.

The fixed negative training gradient for a single sale-logit bias would increase sale probability: direction +0.000493084. Its development loss derivative is +9.38896093e-08, with a descriptive source-game 95% interval [-6.93579147e-07, +8.69550048e-07]. The interval includes both directions; the point estimate is adverse. No bias was fitted, step taken, threshold swept, or candidate selected.

With ten fixed equal-width bins, expected calibration error is 0.002056 on training and 0.007941 on development. Positive and negative errors can cancel in aggregate. AUC and calibration also do not measure downstream game value. The lower development AUC and higher losses identify remaining contextual errors without proving which feature or architecture would repair them.

The next read-only check will evaluate a separately supervised family readout using frozen global state features and a proposal-weighted representation of the candidate Joker/offer combinations. It will use every eligible state from all 5,000 training games, including non-sales, and first test the transfer of a direction obtained from training alone. The improved conditional scorer and the entire existing model remain frozen. No training run is selected by this calibration analysis.

[Calibration results](../../eval/consequence_prediction_20260927/sale_family_calibration_results_v1.json) · [Successful owner](../../eval/consequence_prediction_20260927/sale_family_calibration_owner_terminal_v1.json) · [Independent recount](../../eval/consequence_prediction_20260927/sale_family_calibration_verification_v1.json) · [Conditional-sale ceiling](conditional_sale_ceiling_results_20260930.md)
