# Consequence prediction transfers; both readouts fail action retention

The controlled [consequence-readout experiment](consequence_predictor_experiment_20260927.md) is complete. Both readouts beat the fixed aggregate prediction baselines on the exposed development panel. The matched relational readout improves prediction further, but neither readout passes any of the four action-retention gates. **No candidate advances to policy integration, autonomous games or a confirmation fit.** Long-horizon generalization and a new win-rate gain remain unproven.

Each readout received one exploratory fit: 20 epochs, 7,380 updates and 942,980 state presentations, with the same original 1,000 training games, cached frozen features, labels, row order and optimizer schedule. The pooled head has 131,971 parameters; relational has 132,035. The base encoder was never loaded by either fit, and feature extraction verified unchanged base tensors. Epoch 20 was the fixed endpoint. No extra 4,000-game diversity data or hidden-RNG stress labels entered this comparison.

The [verified shared dataset](consequence_preflight_results_20260927.md) contains 277,936 assessed actions at 53,262 teacher-play states, including 211,825 alternatives beyond teacher/base choices. The development split is the previously exposed 128 games, not fresh confirmation. Prediction metrics weight each state equally and each valid proposal equally within its state. The stronger fixed baseline uses training-only empirical means conditioned on selected-card count and Ante bucket.

| Endpoint | Development log-score MSE | Clear BCE | Clear Brier | Teacher-preference NLL |
|---|---:|---:|---:|---:|
| Constant baseline | 4.228841 | 0.594175 | 0.202169 | — |
| Count × Ante baseline | 1.788753 | 0.585661 | 0.198872 | — |
| Pooled | 0.716282 | 0.325015 | 0.102548 | 0.627259 |
| Relational | 0.650749 | 0.299992 | 0.094187 | 0.633141 |

Both learned readouts pass all four prediction gates: log-score MSE and clear BCE improve over both baselines with strictly negative paired 95% upper bounds. Relational minus pooled is **−0.065532 MSE [−0.089613, −0.041713]** and **−0.025022 BCE [−0.032354, −0.017832]**. Its preference-NLL difference is +0.005882 [−0.017911, +0.029372], so preference improvement is unresolved. These are descriptive whole-game bootstrap intervals from 10,000 shared draws; they do not measure variation across training replicas. Preference NLL uses the fixed proposal set and must not be compared directly with earlier full-policy NLL values.

At the fixed endpoint, each head scored all **1,631,530 legal plays** across the 6,113 development teacher-play states. Its preference argmax replaced the base play choice on 6,008 eligible states; the base's 105 non-play choices stayed unchanged. This is an offline teacher-path choice diagnostic. Every selected action received an outcome, including 3,416 previously unassessed pooled choices and 3,482 relational choices, each replayed through the full native action transition.

| Choice rule | Clears / 6,008 eligible states | Early clears / 2,092 states | Total immediate losses | New losses where teacher survives |
|---|---:|---:|---:|---:|
| Base | 2,699 | 1,066 | 40 | 7 |
| Pooled preference | 2,165 | 732 | 49 | 16 |
| Relational preference | 2,128 | 720 | 53 | 20 |

Pooled gains 108 clears and loses 642 relative to the base, a net loss of 534. Relational gains 115 and loses 686, a net loss of 571. Their eligible-state clear-rate differences versus base are −8.888 percentage points [−9.966, −7.880] and −9.504 points [−10.647, −8.435]. These are **one-action clear rates, not game win rates**. Relational minus pooled is −0.616 points [−1.425, +0.211], providing no resolved retention advantage. Both readouts fail more-overall-clears, retained-early-clears, no-more-total-losses and no-more-new-losses gates. The frozen candidate rule therefore returns no candidate.

The regressions concentrate in previously correct base choices. Where base equals teacher, the pooled readout loses 473 net clears and introduces 14 new immediate losses; relational loses 501 net clears and introduces 15. On base/teacher disagreements, new losses actually fall from seven to two for pooled and five for relational, but those local improvements are outweighed by regressions elsewhere. The base encoder is frozen here: this failure concerns the newly learned readout and its replacement of the base's play preference, not changes to encoder weights.

Saved prefixes also show preference overfitting before any autonomous trajectory drift is involved:

| Readout | Epoch | Training preference NLL | Development preference NLL | Development clear BCE |
|---|---:|---:|---:|---:|
| Pooled | 5 | 0.404770 | 0.540208 | 0.322194 |
| Pooled | 10 | 0.354320 | 0.550269 | 0.313968 |
| Pooled | 20 | 0.254873 | 0.627259 | 0.325015 |
| Relational | 5 | 0.427356 | 0.577785 | 0.313997 |
| Relational | 10 | 0.372098 | 0.556550 | 0.295380 |
| Relational | 20 | 0.296432 | 0.633141 | 0.299992 |

These curves do not change the fixed endpoint or establish that an earlier checkpoint would retain better actions. Only epoch 20 received the complete legal-choice and retained-action evaluation in this study.

The evidence supports a limited architectural result: with labels and capacity controlled, direct access to ordered Joker, hand-level and history context improves these immediate-prediction metrics. It does not establish reusable causal rules for particular card/Joker combinations or long-horizon benefit. The distinction between prediction and choice is now explicit: auxiliary prediction success did not produce a safe preference policy. The diagnostic used the new head's full play argmax; it did not test a bounded correction that retains the base preference pathway.

The subsequent [saved-data support diagnosis](consequence_support_results_20260928.md) now localizes more than 95% of lost clears and all newly introduced fatal choices to selected actions outside the original proposals. Prediction errors worsen on those selected actions, and several fatal choices contradict the head's own clear predictions for the available base action. An oracle-assisted restriction to the teacher-containing proposal set removes much of the damage but still fails the overall-clear requirement and is not a deployable policy. The earlier saved checkpoints are the next cheaper retention check under a separately declared duration evaluation. A complete-legal-support preference loss is then a motivated controlled experiment if earlier stopping fails. Neither an independently trainable feature branch nor a world model is justified as the next remedy by these results alone.

All extraction, fit, prediction, retained-action and comparison owners actually exited zero; all declared audits passed. Fitting audits checked all exposure and saved checkpoints, training normalization and first-update arithmetic without repeating optimization. Prediction audits checked saved arrays, legal support and argmax selection; retention audits checked complete saved provenance, counts and gained/lost sets. The final saved-data comparison reproduced both arms' audited summaries and identical outcomes whenever they selected the same action. It did not rerun inference or simulation, and no second independent full comparison implementation is claimed.

Evidence: [pooled prediction audit](../../eval/consequence_prediction_20260927/prediction_v1/pooled/verification_v1.json), [relational prediction audit](../../eval/consequence_prediction_20260927/prediction_v1/relational/verification_v1.json), [pooled retention audit](../../eval/consequence_prediction_20260927/retention_v1/pooled/verification_v1.json), [relational retention audit](../../eval/consequence_prediction_20260927/retention_v1/relational/verification_v1.json), [matched comparison](../../eval/consequence_prediction_20260927/comparison_result_v1.json), and [actual comparison owner exit](../../eval/consequence_prediction_20260927/comparison_owner_terminal_v1.json).
