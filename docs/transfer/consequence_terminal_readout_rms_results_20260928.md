# Stable scaling improves prediction but does not establish terminal transfer

The [single RMS-normalized readout](consequence_terminal_readout_rms_20260928.md) reduces terminal-label validation BCE from **1.197680 to 0.784264**, but still fails both frozen prediction criteria. The training-prevalence baseline scores **0.691375**, and the ante-bucket baseline **0.650190**. The candidate does not qualify for separate held-out continuation collection, policy integration or confirmation. Long-horizon generalization and an autonomous win-rate gain remain unproven.

The change replaces featurewise training-standard-deviation scaling with per-proposal RMS normalization. It prevents the diagnosed rare-coordinate amplification and bounds each normalized coordinate by √384. All labels, game splits, analytic controls, loss, penalty, optimizer and spending criteria match the [rejected first readout](consequence_terminal_readout_results_20260928.md). Source checks confirm unchanged optimization and metric code; runtime and independent audit reproduce every analytic-control value exactly. This changes the effective regularization of raw coordinates as well as their numerical scale, so it is not a pure architecture comparison.

| Metric | First readout | RMS readout | Comparator |
|---|---:|---:|---:|
| Training BCE | 0.446656 | 0.453069 | Ante baseline 0.663972 |
| Validation BCE | 1.197680 | 0.784264 | Ante baseline 0.650190 |
| Validation Brier | 0.274366 | 0.260960 | Ante baseline 0.229332 |
| Training terminal pair ordering | 68.98% | 69.88% | Base 50.60% |
| Validation terminal pair ordering | 60.71% | 60.71% | Base 59.52% |
| Validation same-clear ordering, state weighted | 51.72% | 56.90% | Base 53.45% |
| Validation restricted-choice terminal wins | 115/200 | 117/200 | Base 116/200 |

The remaining prediction deficit is resolved against the ante baseline in this exploratory validation split: BCE difference **+0.13407 [0.04438, 0.23371]**, and Brier difference **+0.03163 [0.00250, 0.06296]**. Against constant prevalence, the BCE deficit is **+0.09289 [−0.00183, 0.19696]**, unresolved. Both point-estimate prediction spending criteria still fail.

The action-ordering difference against base is **+1.19 percentage points [−16.67, +19.05]**, from 42 mixed-outcome source games. Same-clear ordering differs by **+3.45 points [−18.97, +25.86]**, from 29 games. Restricted-choice terminal wins differ by **+0.50 points [−4.00, +5.00]**. None establishes a gain. These three proposals include the teacher action, and every continuation uses the same base controller after its first action; they are not autonomous policy episodes.

Early point estimates improve: ordering **69.23% versus base 55.77%**, restricted wins **36/74 versus 32/74**. Later ordering is **46.88% versus base 65.63%**, with wins **81/126 versus 84/126**. Both early and later action/choice intervals include zero. No early-only policy or checkpoint is selected from these results.

Six Newton updates converge at maximum absolute gradient **3.75×10⁻¹³**. Fit and independent numerical audit take **0.213 and 0.313 seconds**. The auditor verifies the normalization bound, exact data and controls, accepted updates, stationarity, all 3,000 predictions and every metric/interval. Each preprocessing recipe receives one exploratory fit; there is no independent confirmation training run. Neither fit changes the base encoder, relational feature pathway or existing policy.

The evidence distinguishes two issues. The first readout had a concrete feature-scaling failure; correcting it improves calibration substantially. The remaining training/validation gap persists with a converged, bounded-input readout, so that scaling issue was not sufficient to explain failed transfer. The features do support better training action ordering, which weakens a claim that they contain no useful terminal signal at all. It does not prove the target is learnable on unseen games, nor distinguish insufficient terminal-label coverage, stochastic label variance, regularization and readout capacity.

The next research decision should inspect coverage and within-state action information in the saved data before another coefficient, architecture or loss change. There are only **166 mixed-outcome training states** for this readout, versus **800 source states** overall; three correlated branches do not create 2,400 independent games. The same 200-game terminal-label validation split has now been adaptively reused, and all 1,000 source games contributed to feature training. Any future generalization claim still requires held-out source-game prediction/retention and the full fresh-game, greedy, early-attribution and final independent-confirmation evidence. Both current endpoints remain rejected; no third fit is selected here.

Artifacts: [owner](../../eval/consequence_prediction_20260927/terminal_readout_rms_owner_terminal_v1.json), [frozen protocol](../../eval/consequence_prediction_20260927/terminal_readout_rms_protocol_v1.json), [result](../../eval/consequence_prediction_20260927/terminal_readout_rms_v1/result_v1.json), and [audit](../../eval/consequence_prediction_20260927/terminal_readout_rms_v1/verification_v1.json).
