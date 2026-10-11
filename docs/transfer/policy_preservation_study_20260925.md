# Policy-preservation experiment: completed

Both fits, the full independent fit audit, all 23 evaluation panels and the independent raw-results verification are complete. **Four of five scientific criteria failed.** The [full results](policy_preservation_results_20260926.md) report all seven contrasts and both replica directions. Reduced teacher-state forgetting did not establish a fresh-win or early-decision benefit. The long-horizon goal remains unproven.
The [execution protocol](../../eval/policy_preservation_20260925/execution_protocol_v1.md)
and [manifest](../../eval/policy_preservation_20260925/execution_manifest_v1.json)
were frozen before either scientific fit. World models remain deferred.

The original evaluation controller disappeared after 18 successful panels; its exit remains unknown in the [controller-loss record](../../eval/policy_preservation_20260925/evaluation_controller_loss_v1.json). The [versioned recovery](../../eval/policy_preservation_20260925/evaluation_recovery_protocol_v1.md) captured the nineteenth worker’s actual OS exit and completed only the four originally unstarted panels and original analyzer. Its coordinator and the frozen independent verifier both exited 0 under direct parent supervision. Original raw-results checks, scientific criteria and deadlines were preserved.

The question follows the [component diagnosis](component_panel_20260925.md): can
we reduce the loss of previously correct behavior during further fitting while
retaining useful corrections? Treatment adds 1.0 times full forward KL from the
immutable starting policy to ordinary teacher-action NLL. This coefficient was
fixed before the [mathematical and native preflight](policy_preservation_preparation_20260925.md).
All actor, embedding and Transformer parameters remain trainable; the critic and
reference remain fixed. Preserving the base can also preserve its errors.

Both treatment fits start from the same original Stage 2 checkpoint, using the
existing 1,000-game / 202,141-decision corpus. They use 4,000 updates, batches of
320 examples in physical batches of 32, fresh AdamW, learning rate 3e-5 with 100-update warmup,
zero weight decay, disabled dropout and the original per-replica row orders.
The final update is the only scientific endpoint; intermediate checkpoints
cannot select a policy or change training.

The two completed ordinary controls are reused. A new
[admission audit](../../eval/policy_preservation_20260925/control_reuse_admission_v1.json)
reconstructed all 8,000 ordinary updates and eight exposure snapshots, checked
the complete recipe/source/data provenance, and matched every control checkpoint
to the completed independent CPU tensor audit. Current checkpoint bytes match
those earlier audited artifacts. The controls were fitted earlier, and both
continuation replicas share one pretrained base. These limitations remain part
of the interpretation. Every comparison uses newly played outcomes.

The actual scientific fitting worker also completed a disposable twenty-update
run for each arm. Both trajectories reproduce the previously checked pilot,
the initial model and RNG match the ordinary control, and saved checkpoints
reload exactly. The first independent worker audit stopped on a parameter-count
mismatch: the checkpoint stores one shared hand-slot embedding under three
names, while the worker counts distinct named parameters. A
[read-only diagnostic](../../eval/policy_preservation_20260925/movement_count_diagnostic_v1.json)
confirmed the two extra aliases. The corrected audit still checks every state
entry, explicitly checks equality of shared aliases, and counts each parameter
once. It passed all 1,280 checkpoint-state-entry checks and 620 optimizer-state
checks. The [failed receipt](../../eval/policy_preservation_20260925/worker_smoke_audit_terminal_v1.json)
and [successful receipt](../../eval/policy_preservation_20260925/worker_smoke_audit_terminal_v2.json)
are retained. Training and scientific acceptance thresholds did not change.

Two fresh banks of 400 environment seeds were reserved before fitting:
884260000–884260399 for confirmation and 884270000–884270399 for early attribution.
The [reservation](../../eval/policy_preservation_20260925/seed_reservation_v1.json)
checked 9,779 text files containing 5,710,214,791 bytes with two search methods.
It excluded 950 files from the failed Stage 3 study and read only that study's
seed reservation. This covers recorded/reserved seeds, not undocumented binary
or procedural use. No new outcomes were inspected to choose these banks.

Confirmation played the common base and four endpoints, each greedily and
on two sampled streams: 6,000 episodes. The separate early panel played each
ordinary control and a hybrid on both streams: 3,200 episodes. Each hybrid uses
preservation until the first pre-action Ante >= 4, then permanently uses its paired
ordinary continuation. The existing verified routing and statistical components
are unchanged. Evaluation settings, panel order and all 23 panels are frozen.

Both sampled treatment comparisons—against ordinary fitting and against the
base—must gain at least five percentage points, have paired 95% lower bounds
above zero, and improve in both replicas. Greedy lower bounds must exceed minus
five points against both controls. The separate early hybrid must improve with
a lower bound above zero and positive gains in both replicas. Bootstrap units
are whole environment seeds, keeping streams and replicas together. The shared
base is not counted as two independently played controls. Raw actions, wins, routing and statistics require independent verification.

A secondary comparison on the existing exposed 128-game teacher panel has
completed, including independent verification of raw rows, retention counts and
all 24 planned descriptive intervals. Preservation improves teacher NLL by
0.047675 and exact agreement by 0.390 percentage points versus ordinary fitting.
Overall NLL remains slightly worse than the base, and the overall agreement
gain versus the base is uncertain. The [full diagnostic results](policy_preservation_mechanism_20260926.md)
include early/late and play-action splits and the retained verifier failure.
This panel provides mechanism evidence; it cannot select training settings,
checkpoints or fresh outcomes. Better imitation does not meet the win-rate or
long-horizon criteria.

The [training owner](../../eval/policy_preservation_20260925/fit_execution_terminal_v1.json)
completed both fits serially and exited successfully within the fixed ceilings.
The full independent CPU model/optimizer/exposure audit
[passed](../../eval/policy_preservation_20260925/fit_audit_terminal_v1.json),
checking eight saved checkpoints, 2,560 state entries and 1,860 optimizer states.
Fresh evaluation was admitted under the
[original execution intent](../../eval/policy_preservation_20260925/evaluation_execution_v1/intent.json).
Its later controller interruption is recorded above.
The
[admission review](../../eval/policy_preservation_20260925/execution_admission_review_v1.json)
checked that both missing-audit and missing-training-exit conditions block the
dependent work before native imports. Failures retain their evidence without
automatic retry, endpoint substitution or budget extension.
