# A local training signal for when to sell

The fixed 193-coordinate family readout has a training direction that also lowers development loss locally. The training/development gradient cosine is **0.860827**. For the single negative combined-training gradient, the development directional derivative is **-0.0000763061**, with a marginal game-bootstrap 95% interval **[-0.0001059846, -0.0000468455]**. Negative means lower sale-vs-rest cross-entropy for a sufficiently small step in that direction.

| Fixed feature block | Gradient cosine | Contribution to development derivative |
| --- | ---: | ---: |
| Contextual state (128) | 0.856184 | -0.0000477947 |
| Weighted replacement features (64) | 0.886169 | -0.0000286012 |
| Intercept (1) | -1.000000 | 0.0000000897 |

The intercept contribution is adverse while both contextual blocks are favorable. Together with the earlier nearly matching expected and observed sale totals, this supports investigating a context-dependent sale decision. It does not support a blanket increase in selling. The feature blocks describe the one fixed full direction; they do not select a new subset or separate fit.

The analysis uses all 286,044 active training states from 5,000 games, including 259,262 non-sales, and 7,512 development states from the same exposed 128-game panel. Normalization statistics and the direction use training data only. All 128 development source games enter the 10,000 bootstrap draws, including games without an active state. Original-row denominators are 1,013,555 training and 26,045 development. Current-Ante-below-four derivatives are descriptive and do not constitute the required early-policy intervention.

The full 293,556-row feature collection and independent native audit completed successfully. The local-gradient owner also exited successfully in 2.013 active seconds. Maximum independent-gradient error was 1.19e-18; the full-training finite-difference error was 3.03e-13. The saved direction and interval were independently reconstructed at admission.

This crosses the predeclared bar for specifying **one** subsequent fit. It does not show that the finite optimum transfers, that any teacher action improves, or that win rate rises. The interval is exploratory on an already exposed development panel. The original 13 development checks and final fresh-game, retention, early-attribution and independent-confirmation requirements remain unchanged.

[Feature plan](sale_family_features_20260930.md) · [Gradient owner](../../eval/consequence_prediction_20260927/sale_family_gradient_owner_terminal_v1.json) · [Raw results](../../eval/consequence_prediction_20260927/sale_family_gradient_v1/result_v1.json) · [Single-fit plan](sale_family_fit_20260930.md).
