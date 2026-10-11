# Widening improves predictions but fails policy retention

The requested wider consequence scorer improves held-out score and clear-probability prediction in this exploratory fit. It still fails both clear-retention gates, so it does not advance to policy integration, autonomous games or an independent confirmation fit. Better consequence estimates have not established better long-horizon decisions or a win-rate gain.

The [matched experiment](consequence_relational_width_20260928.md) changes `query_dim=64 → 192` and `hidden=128 → 384`, keeping `d_model=384`, `state_dim=128` and four attention heads. Trainable parameters increase from **132,035 to 543,555**. Both scorers use the same frozen features and base priors, 47,149 training states, 292,762 immediate labels, 3,396 survival pairs, four losses, order and optimizer recipe. The single wider fit completes the fixed 20 epochs, 7,380 updates and 942,980 state presentations. It takes **4,974 seconds**, versus 2,519 for the smaller scorer. All training and optimizer checks pass. The new continuation labels do not enter either fit.

| Original-proposal metric | Smaller scorer | Wider scorer |
|---|---:|---:|
| Training log-score MSE | 0.634604 | 0.564238 |
| Training clear BCE | 0.261576 | 0.231710 |
| Training teacher preference NLL | 0.456377 | 0.334568 |
| Held-out log-score MSE | 0.718214 | 0.664385 |
| Held-out clear BCE | 0.310271 | 0.300293 |
| Held-out clear Brier error | 0.096806 | 0.093610 |
| Held-out teacher preference NLL | 0.921477 | 0.975469 |

The held-out population contains 32,299 proposals at 6,113 teacher-play states from the reused 128-game development panel. These games are held out from this fit but were exposed during preceding experiments; this is exploratory evidence, not untouched final confirmation. The wider-minus-smaller log-score MSE difference is **−0.053829 [−0.076879, −0.030719]**, about a 7.5% reduction. Clear BCE improves by **−0.009978 [−0.018058, −0.001851]**. Brier error also improves with its paired interval below zero. Intervals use the existing 10,000 whole-game resamples and state weighting. All four prediction gates against the original constant and cell baselines pass.

The gain also holds on the same 6,008 actions fixed by the smaller scorer: log-score MSE is **0.591073 versus 0.656231**, clear BCE **0.413972 versus 0.436490**, and Brier error **0.134065 versus 0.139915**, with all three paired intervals below zero. These matched-action comparisons separate prediction quality from changing which actions are evaluated.

Preference transfer behaves differently. Training preference NLL improves while held-out preference NLL worsens. On the complete eligible action support, teacher NLL rises from **1.231676 to 1.305855**, and teacher argmax matches fall from **4,173 to 4,152**. Selected actions outside the original proposals increase from 427 to 564; every selected action is assessed, including 139 newly replayed alternatives. Teacher agreement is diagnostic and is not used as an oracle for selecting corrections.

The training survival-pair diagnostic confirms improved fitting on all 3,396 pairs at 430 states from 332 games. State-weighted preference error falls from **9.7907% to 5.4302%**, with 174 improved and 28 worsened pairs. Mean preference pair loss falls from **0.488100 to 0.207043**. Score and clear ordering also improve in training. These are in-sample results, with exact original-proposal reconciliation and the smaller control's previous counts reproduced.

The held-out survival-pair diagnostic contains 318 pairs at 56 mixed states from 43 games. State-weighted preference error falls from **10.7242% to 10.0446%**, with 10 improved and seven worsened pairs. The difference is **−0.6796 percentage points [−3.5738, +2.1796]**, unresolved. Score and clear pair-ordering errors have worse point estimates, also unresolved. All 15 fixed group/head intervals include zero. The aligned control layout changes no pair classifications.

| Selected-action screen | Base | Smaller scorer | Wider scorer | Required |
|---|---:|---:|---:|---:|
| Clears, 6,008 eligible states | 2,699 | 2,687 | **2,676** | > 2,699 |
| Early clears, 2,092 states | 1,066 | 1,062 | **1,054** | ≥ 1,066 |
| Immediate losses | 40 | 36 | **37** | ≤ 40 |
| New losses versus teacher | 7 | 3 | **4** | ≤ 7 |

Both loss gates pass and both clear gates fail. The overall clear differences remain statistically unresolved: wider-minus-smaller is **−0.1831 points [−0.6284, +0.2718]**, and wider-minus-base is **−0.3828 points [−0.8436, +0.0510]**. A failed fixed screen is not evidence of a resolved overall decline.

The existing partition identifies a retention cost. Where base and teacher agree, the wider scorer clears **1,956** times versus **1,977** for the smaller scorer and **2,007** for base. The 21-clear deficit against the smaller scorer is resolved: **−0.5053 points [−0.8901, −0.1219]**. Where base differs from teacher, the wider scorer clears **720** times versus 710 for the smaller scorer and 692 for base; its gain over base is resolved, while its gain over the smaller scorer is not. Thus additional useful corrections coexist with lost existing decisions.

A post-hoc accounting of the saved choices reconciles the net 11-clear deficit against the smaller model. The wider scorer newly leaves base in 577 states, costing 27 net clears; returning to base in 304 states recovers 15; changing one override to another in 178 states adds one. The other 4,949 states retain the same action. Wide/base disagreements total 1,120, versus 847 for the smaller scorer.

The saved final-hand cases show two distinct remaining errors. At game `850000098`, step 60, the wider policy selects action 2024, which scores **150 and loses**, over the shared base/teacher/smaller action 144, which scores **6,650 and survives**. Its own clear head assigns the chosen action **5.57%**, versus **90.23%** for action 144: the preference head overrides a useful consequence estimate. At game `850000042`, step 113, both consequence heads instead favor the newly fatal choice, so prediction error also remains. The wider scorer introduces these two losses relative to the smaller scorer and cures its previous fatal choice at `850000043:31`. These examples diagnose realized outcomes and output disagreement; they do not establish conditional expected game values.

Immediate clear retention remains the frozen advancement screen. The earlier [continuation diagnostic](consequence_fixed_base_continuation_results_20260928.md) showed that many missed immediate clears recover later in the blind, so this screen alone cannot establish the wider model's long-horizon value. No fresh autonomous win-rate comparison or confirmation fit was run. The observed capacity benefit is in consequence prediction; reliable preference transfer and retention remain gaps.

An [evaluation consistency correction](consequence_relational_width_evaluation_layout_20260928.md) precedes admission. The original FP32 evaluation produced two proposal-overlap discrepancies from different attention batch layouts. Explicit math attention left one discrepancy and was not adopted. Proposal inference now uses the same ordered full-support layout as policy selection, retaining default FP32, the trained checkpoint and all gates. Independent recomputation reproduces **all 1,631,530 prior action outputs exactly** and requires exact proposal overlap. The smaller control uses its saved full-support outputs for the same-action comparison; the resulting overall held-out metric shifts are below 5×10⁻⁹. All failed artifacts and numerical probes are preserved. This numerical issue is not offered as an explanation for the learning result.

Artifacts: [fit owner](../../eval/consequence_prediction_20260927/relational_width_fit_owner_terminal_v1.json), [prediction owner](../../eval/consequence_prediction_20260927/relational_width_prediction_owner_terminal_v1.json), [prediction audit](../../eval/consequence_prediction_20260927/prediction_relational_width_v2/relational_wide_fixed_base_residual/verification_v1.json), [retention owner](../../eval/consequence_prediction_20260927/relational_width_retention_owner_terminal_v1.json), [retention audit](../../eval/consequence_prediction_20260927/retention_relational_width_v2/relational_wide_fixed_base_residual/verification_v1.json), [training pair audit](../../eval/consequence_prediction_20260927/relational_width_training_predict_v2/verification_v1.json), [held-out pair analysis](../../eval/consequence_prediction_20260927/relational_width_development_alignment_v2/result_v1.json), [saved choice accounting](../../eval/consequence_prediction_20260927/relational_width_choice_accounting_v1.json), and [fatal-case outputs](../../eval/consequence_prediction_20260927/relational_width_fatal_output_diagnostic_v1.json).
