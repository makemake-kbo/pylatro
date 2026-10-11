# Full-support preference training repairs much of the scorer regression

Changing the pooled head's preference loss to include every legal play recovers **363 blind clears** and removes **eight newly introduced immediate losses** relative to its original proposal-trained endpoint. The architecture, frozen features, consequence labels, initialization, state order and training exposure were held fixed. This supplies controlled exploratory evidence that the new scorer's training/action-selection support mismatch materially harmed retention.

The repaired scorer still falls below the existing base and **fails all four retention gates**. It does not advance to policy integration, autonomous games or confirmation. There is no new win-rate or long-horizon generalization claim.

| Policy | Clears / 6,008 states | Early clears / 2,092 | Immediate losses | New losses where teacher survives |
|---|---:|---:|---:|---:|
| Base | 2,699 | 1,066 | 40 | 7 |
| Original pooled epoch 20 | 2,165 | 732 | 49 | 16 |
| Full-support pooled epoch 20 | 2,528 | 954 | 41 | 8 |

Compared with the original pooled scorer, the clear-rate gain is **+6.042 percentage points**, paired 95% interval **[+5.146, +6.954]**. Early clears rise by 222, **+10.612 points [ +9.021, +12.209 ]**. The eight-loss reduction is −0.133 points **[−0.264, −0.016]**. The clear gain recovers 68.0% of the original scorer's 534-clear deficit relative to base; this describes the net count difference, not identification of 363 particular repaired states.

Against base, the remaining deficit is **171 clears**, −2.846 points **[−3.597, −2.138]**; early clears are 112 lower. The model also has one more total and newly introduced immediate loss. The overall loss-rate interval spans zero, but the declared gates require no observed increase. All four retention gates fail. The comparison gains 89 clears and loses 260 relative to base.

These are single-action interventions at identical teacher-path states in 128 already exposed development games, not autonomous wins. Intervals resample whole source games with the fixed 10,000 draws and seed 1729. They do not measure independent-fit uncertainty.

The residual deficit is concentrated in retention of already correct base behavior:

| Fixed state group | States | Base clears | Full-support clears | Base new losses | Full-support new losses |
|---|---:|---:|---:|---:|---:|
| Base action equals teacher | 4,156 | 2,007 | 1,837 | 0 | 2 |
| Base action differs from teacher | 1,852 | 692 | 691 | 7 | 6 |

Thus 170 of the remaining 171 net lost clears occur where base already matches the teacher. Teacher agreement is unavailable as a deployment gate; these groups diagnose the failure rather than define a deployable correction rule.

Full-action ranking improves even though proposal-only preference loss worsens:

| Eligible-state metric | Original pooled | Full-support pooled |
|---|---:|---:|
| Teacher is highest-ranked legal play | 2,187 / 6,008 | 3,800 / 6,008 |
| Mean teacher rank | 13.169 | 6.179 |
| Full-legal-support teacher NLL | 2.620480 | 1.572541 |
| Selected action outside original proposals | 3,416 | 1,058 |

On all 6,113 teacher-play states, the new proposal-only preference NLL is 0.963757 versus the original 0.627259. Those values normalize over the same original proposal sets, but improvement on that restricted metric was not sufficient for full-action selection. The opposite directions of proposal NLL and full-support ranking make the metric distinction consequential.

Consequence prediction stays approximately unchanged: new development log-score MSE is **0.716989**, clear BCE **0.327296**, and Brier **0.102670**. Paired differences from original pooled are MSE +0.000708 **[−0.013893, +0.015252]**, BCE +0.002281 **[−0.002372, +0.006839]**, and Brier +0.000122 **[−0.001234, +0.001444]**. The model passes all four prediction gates against the training-only constant and count-by-Ante baselines. Its outcome gain therefore accompanies substantially better action ranking, without a resolved improvement in these proposal-distribution consequence errors.

Exactly one exploratory fit ran. The [preflight](consequence_full_support_preparation_20260928.md) verified the loss, gradients and all training masks; the [recipe](consequence_full_support_20260928.md) fixed 20 epochs, 7,380 updates and 942,980 state presentations. The worker completed and saved its checkpoints, then **exited one at its final runtime-origin check** because that validator imported the working-tree schema before the frozen path was installed. The original failure remains preserved. An import-only reproduction and [separate saved-artifact recovery](consequence_full_support_recovery_20260928.md) passed with zero additional optimizer steps. That recovery independently verified all 12,249,244 unique training state/action entries, 244,984,880 action presentations, every update/checkpoint, initial weights and first-update loss/gradient/clipping/Adam arithmetic. This is a disclosed procedural recovery, not a pristine successful fit execution or a second fit.

Prediction measurement/audit took 16.719/11.249 seconds. Retention measurement/audit took 180.810/2.584 seconds. Replay reproduced all 26,045 factual decisions and 128 game terminals, retained 105 base non-play decisions, and assessed every selected play: 4,950 outcomes reused original labels, 504 reused identical previously audited actions, and 554 required new branches. All duplicate outcomes, identities, legal masks, parent/RNG isolation and terminal checks passed. All evaluation workers and owners actually exited zero before admission.

The [prediction audit](../../eval/consequence_prediction_20260927/prediction_full_support_v1/pooled_full_support/verification_v1.json), [retention audit](../../eval/consequence_prediction_20260927/retention_full_support_v1/pooled_full_support/verification_v1.json), [recovery receipt](../../eval/consequence_prediction_20260927/full_support_recovery_owner_terminal_v1.json), and [retention owner receipt](../../eval/consequence_prediction_20260927/full_support_retention_owner_terminal_v1.json) bind the measurements and provenance.

Before another fit, the remaining cheap duration check should compare saved prefixes using **full-legal-support teacher NLL**, the objective actually trained here. The current prefix curves cover only original proposals; they cannot establish an earlier full-support peak. Declare that separate post-hoc development selection before measuring new full-support prefixes, select at most one earlier checkpoint for action replay, and preserve this failed fixed epoch-20 result. Any later correction must still pass prediction and retention before fresh games, and the original long-horizon/independent-confirmation bar remains unchanged. This repair concerns the new scorer; the original subset policy already normalizes over full support, so it does not by itself explain the original pretraining win gap.
