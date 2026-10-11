# Policy-preservation training: implementation and preflight

The preservation objective is implemented and has passed mathematical and native training checks. **No scientific training run or fresh-game evaluation has started, and no learning or win-rate benefit is established.** The [full-panel diagnosis](component_panel_20260925.md) motivates the experiment: further fitting loses previously matched held-out choices while learning other choices, and saved-component rollback gives up many of those gains.

The fixed proposed objective is the mean teacher-action negative log probability plus **1.0 × KL(starting policy ‖ current policy)** on each training state. The coefficient was fixed before the native checks, with no sweep or held-out calibration. The reference is the complete immutable starting checkpoint. All actor, embedding and Transformer parameters remain trainable; the critic remains fixed. The policy architecture and inference behavior are unchanged.

The KL covers every legal flat action through the grammar's chain rule: macro divergence plus reference-macro-probability-weighted conditional divergences. This includes corrected play/discard subsets, indexed actions, and consumable slot/target choices. Reference probabilities are detached. The implementation enumerates conditional support exactly, rather than sampling actions or penalizing an unweighted collection of logits. The penalty may also preserve starting-policy mistakes; its usefulness remains an experimental question.

The [preparation protocol](../../eval/policy_preservation_20260925/preparation_protocol_v1.json) permits training-only checks. The [KL implementation](../../eval/policy_preservation_20260925/preservation_v1.py) is independently checked against exhaustive calls to the original flat-action `log_prob` API. Test states are selected by fixed rules from the training cache: the first teacher example of each observed family, plus the earliest state with each legal grammar family. There are fourteen observed teacher families in training and fifteen legal action families. The selected union contains sixteen states.

| Mathematical check | Verified result |
| --- | --- |
| Original MPS/fp32/eval batch controls | 288 predictions; maximum NLL difference 0 |
| Complete legal support | 16 training states, 15 action families, 1,856 legal leaves |
| Independently recounted flat KL versus structured KL | Maximum difference 1.1449174941446927e-16 |
| Every output-gradient coordinate versus exhaustive flat KL | 225,072 coordinates; maximum difference 1.4239722670070504e-16 |
| Identity policy | KL and current-policy gradient zero within the fixed double-precision tolerances |
| Reference gradients | None |
| Central finite differences | Exhaustive flat-loss directional differences agree within 2e-7 |
| Invalid comparisons | Different legal supports and empty supports rejected |

The [mathematical audit](../../eval/policy_preservation_20260925/kl_audit_v1.json) recounts saved probability vectors, raw cached legal support and every paired gradient array. Its [terminal](../../eval/policy_preservation_20260925/kl_audit_terminal_v1.json) records actual exit code zero. This audit does not run a second neural forward pass.

The initial native training check required byte-identical model and optimizer updates at zero penalty. It failed before the twenty-update pilot began. Restoring the original explicit unit-weight multiplication in the [loss wrapper](../../eval/policy_preservation_20260925/loss_v2.py) did not remove the discrepancy, so that arithmetic difference was not established as its cause. A separate [control-repeat diagnostic](../../eval/policy_preservation_20260925/zero_reproduction_v1/result.json) then found the same issue when repeating the identical original update:

| One-update comparison | Maximum parameter difference | Maximum gradient difference |
| --- | --- | --- |
| Original versus repeated original | 2.9802322387695312e-8 | 7.450580596923828e-9 |
| Original versus zero-penalty wrapper | 2.200249582529068e-8 | 1.4901161193847656e-8 |

All three runs started with identical tensor bytes and produced exactly the same objective; all thirty checks of the loss derivative with respect to teacher log probabilities equaled exactly −1/320. Byte equality was therefore too strict for the observed native execution, even for its own control. The failed [v1](../../eval/policy_preservation_20260925/training_preflight_terminal_v1.json) and [v2](../../eval/policy_preservation_20260925/training_preflight_terminal_v2.json) attempts remain preserved. Neither reached the paired pilot or produced scientific fits.

Before the next run, [v3 preparation](../../eval/policy_preservation_20260925/training_preflight_preparation_v3.json) replaced that technical byte-equality condition with a fixed absolute tolerance of 2e-7 for resulting parameter, gradient and optimizer-moment differences. It retained exact checks for initial tensors, loss values and leaf derivatives, optimizer structure and step counts, and row exposure. The coefficient, data, training recipe and scientific acceptance criteria did not change. A prelaunch review ensured that all parameter gradients are compared, including those whose parameter values round identically; the earlier draft is retained.

The [paired native preflight](../../eval/policy_preservation_20260925/training_preflight_v3/result.json) then passed. Both arms start from the same checkpoint with fresh AdamW, use the same row order and run twenty updates of 320 examples in physical batches of 32. The learning rate warms linearly toward 3e-5 over one hundred updates; weight decay is zero and dropout is disabled. These are short pilot checkpoints, explicitly marked as non-scientific candidates.

| Native training check | Verified result |
| --- | --- |
| Original supervised batch controls | 320 predictions; maximum NLL difference 0 |
| Zero-penalty loss derivatives | 20 exact checks |
| Zero-penalty parameter / gradient / optimizer-moment differences | 2.9802322387695312e-8 / 1.4901161193847656e-8 / 1.862645149230957e-9 |
| Original twenty-update NLL trajectory | Maximum difference 5.587935447692871e-8 |
| Matched exposure | 20 updates and 6,400 examples per arm; identical per-row exposure and ordering |
| CPU-loaded checkpoint inspection | 640 tensors across the two pilot checkpoints |
| Optimizer exposure | 620 parameter states across the two arms; every state advanced twenty steps |
| Gradient and parameter groups | Actor, embedding and backbone receive gradients and change |
| Preservation gradient after fitting | Nonzero in all three trainable groups |
| Fixed components | Critic tensors and reference model unchanged |
| Checkpoint serialization | Both pilot models reload with identical tensor bytes |

The [independent training audit](../../eval/policy_preservation_20260925/training_audit_v3.json) loads the saved models and optimizer states on CPU, reconstructs the row permutation independently, checks every exposure count, and recounts the logged objective. Its [terminal](../../eval/policy_preservation_20260925/training_audit_terminal_v3.json) records actual exit code zero. The maximum final parameter difference between the two pilot arms is 3.33636999130249e-5. No direction of a training metric was required for technical admission, and the pilot is not evidence of generalization.

A preparation-only assertion also initially assumed the training corpus had the same thirteen observed teacher families as the exposed held-out panel. The actual training inventory includes six consumable-sale examples, giving fourteen families. That attempt stopped before a protocol or native output was created; its [receipt](../../eval/policy_preservation_20260925/preparation_attempt_v1.json) is retained.

All preparation processes and their owned sleep guards have ended. The next work is the scientific execution manifest, matched-control provenance, fresh seed reservation and bounded fit/evaluation workers. The full study must retain replicated fresh sampled gains of at least five percentage points over both matched ordinary fitting and the common base, positive paired lower bounds and positive gains in both replicas, greedy guardrails, early-policy outcome attribution and independent verification. No checkpoint or coefficient is selected on the exposed holdout. World models remain deferred, and the long-horizon goal remains unproven.
