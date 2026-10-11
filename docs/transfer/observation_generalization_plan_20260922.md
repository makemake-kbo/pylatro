# Matched observation-learning experiment

The three reproduced observation omissions now have an isolated implementation.
The [new experiment](../../eval/observation_generalization_20260922/manifest.json)
has **completed all 32 training updates, both final control input-invariance
proofs, and the amended training audit and checkpoint lock**. All four training
branches exited successfully. The original supervisor subsequently exited at its
known, preserved original all-zero-adapter lock failure; the external amended
validator then passed the full audit and created the endpoint lock. Independent
review verified both final proofs and the lock's artifact coverage and provenance.
The original frozen coordinator has resumed into held-out evaluation: the parent,
full-input, and ablated replica-0 confirmation jobs are running. Confirmation
and early-policy outcomes are still pending; no partial comparisons are being used.
No learning gain is established. The original credit study
is complete and independently audited; its early-decision gain was +1.625 points
[−1.125, +4.4375], and all three main acceptance gates failed.

The frozen new protocol has SHA256
`0f096371a730dc484817aeb6b40a56fa239a0c6567cdc405a138c26f18c8ce9e`.
Its 236 source files have combined fingerprint
`52bb12a4912af57fd02f8b3f5871c6bb1841e1c9ab2e54f8297cacf636a6bcb0`.
Four initial checkpoints were prepared and validated before launch. Training
ran serially in the order full replica 0, ablated replica 0, ablated replica 1,
full replica 1.

The primary comparison uses the same corrected architecture in both arms. The
`full` arm receives pack enhancement, the global Fool target, and consumable
resale value. The `ablated` arm replaces only those inputs with zero inside the
shared embedding entry. Both retain all three adapters and their computation
paths. This avoids confounding information access with model size or parameter
registration. Shared observation buffers remain intact.

The [first control audit](../../eval/observation_generalization_20260922/analysis/first_ablated_update_health_1790110837588369000.json)
failed its original requirement that every ablated adapter remain zero. A
[minimal backend reproduction](../../eval/observation_generalization_20260922/analysis/padding_gradient_diagnosis_v1.json)
found that PyTorch 2.11.0 on MPS accumulates gradients in embedding row 0 despite
`padding_idx=0`, whereas CPU suppresses them. The
[checkpoint diagnosis](../../eval/observation_generalization_20260922/analysis/control_padding_diagnosis_1790111137065834000.json)
found changes only in the two categorical adapters' row-0 weights and moments;
all other categorical rows and the continuous resale projection remain zero.
The first cohorts' 2,112 saved arrays and both arms' initial training state match.

The [scientific validity review](../../eval/observation_generalization_20260922/analysis/control_validation_amendment_review.md)
supports retaining the frozen training and policy source **only after** the
trained control is shown invariant to changes in the hidden fields, separately
on CPU and MPS, for actor and all critic outputs. A learned constant embedding
can remain an information ablation. The versioned validation amendment preserves
the original failed artifact and continues checking all other adapter rows,
optimizer state, exposure and seed requirements. The first trained-control proof
passed before training resumed; the final endpoint proofs and amended audit have
now also passed. Every outcome and
early-attribution gate below remains unchanged. This supplies no claim of
legacy-model equivalence or CPU/MPS training parity.

Each replica starts from its completed terminal-GAE checkpoint at global update
142. An explicit migration preserves every actor and critic tensor and every
existing Adam state, adding exactly three zero adapters with fresh optimizer
state. Both arms clone the same migrated state. This is an explicitly migrated
v14 branch. The completed branch writer preserves RNG, controller, counter,
schedule and inherited seed-exclusion state; four initial checkpoints passed
strict write-and-reload checks using the pinned donors.

Preparation evidence:

- [Ablation implementation and checks](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/ablation_review.json):
  19 synthetic CPU checks passed, including identical parameter initialization,
  checkpoint visibility, both resume guards, input preservation and explicit
  zero adapter gradients in the ablated arm on CPU. The later MPS diagnosis
  demonstrates that this gradient assumption does not hold on that backend.
- [Historical donor parameter order](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_legacy_donor_parameter_order.json):
  reconstructed all 318 parameters using the frozen original source for each
  pinned donor. This supports the historical positional Adam-order contract.
- [Migration verification](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/donor_migration_verification.json):
  all existing model tensors and Adam states were preserved. Source, full and
  ablated models produced exactly equal raw actor and critic outputs for four
  observations per replica, including an ordinary reset observation and fixtures
  containing the new fields. Critic-only execution also matched. No optimization
  step or full game ran.
- [Branch preparation verification](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/branch_prepare_review.json):
  15 focused checks and four real-donor checkpoint roundtrips passed. The
  preflight snapshots were provisional; the launched manifest records the final
  source and matched initial states for both replicas.
- [Native execution preflight](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/native_execution_preflight.json):
  all 15 isolated native extensions compiled with the existing floating-point
  settings. Across 73 fixtures, 1,022 tokenizer arrays exactly matched the frozen
  Python output; 93 native execution and input-visibility checks passed.
- [Early-policy wrapper checks](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/observation_horizon_policy_tests.xml):
  27 checks passed, including all-phase routing, an Ante decrease from 4 to 3,
  shared action grammar, and the component actor and critic outputs.
- [Training-evidence auditor](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/training_audit_review.json):
  17 synthetic checks passed. The auditor links each saved cohort to its
  collecting policy and resulting checkpoint, checks ordered seeds and complete
  terminal labels, and verifies actual Adam-step counts and the ablated adapters.
  These checks ran no learning updates or games. Its original all-zero adapter
  condition later failed on the actual MPS control update; the completed
  amendment and control functional proofs are described below.
  A separate [production-format review](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/training_audit_production_review.json)
  found no mismatch against a completed original GAE update: all 64 trajectories,
  11,809 decisions, policy digests, recipes, counters, draw indices and existing
  Adam states matched the auditor's assumptions. That historical review did not
  verify learned v14 adapter behavior; the completed amended audit does so under
  the disclosed categorical-row exception.
- [Seed and resource plan](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_seed_resource_plan.json):
  1,024 distinct training seeds and 864 reserved evaluation/mechanism seeds passed
  separation checks against inherited data and 964 earlier manifests. This
  initial audit was provisional while the original study was running.
  A [stronger provisional scan](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/nested_seed_preflight.json)
  also includes numeric replica keys and reserved panel lists; 965 manifests
  containing 35,872 historical seeds remained disjoint. The final audit in the
  launched manifest rechecked 966 completed historical manifests and the same
  35,872 historical seeds; all proposed training and reserved panels are disjoint.

The fixed endpoint is **eight updates per arm and replica**, using the
established terminal-GAE recipe: 64 complete games per update, 8,192 sampled rows,
two PPO epochs, logical minibatches of 640 and physical batches of 160. Each of
the four branches completed 512 games, 131,072 optimizer examples and 208 logical
optimizer steps. There was no development endpoint selection. Trajectories
could diverge as policies learned; seed schedules and optimizer exposure remained
matched.

The new accelerator ledger carries 10,255.876 seconds from the prior allowance.
Before launch, recent updates averaged 256.67 seconds. The planning estimate at
260 seconds per update was 8,320 seconds, or 9,984 seconds with 20% contingency.
Completed training was charged 8,450.1513 seconds, including both diagnostic
pauses; the final endpoint proofs were charged another 96.4194 seconds. The fixed
allowance has 1,709.3054 seconds remaining before held-out evaluation. The old
allowance is not extended. An
[exclusive transfer record](../../eval/long_horizon_generalization_20260921_logical_kl/unspent_compute_transfer.json)
and the [new ledger](../../eval/observation_generalization_20260922/compute.json)
record that transfer. Eight updates may be insufficient to learn useful behavior. That uncertainty
does not relax the acceptance criteria or support a claim that the information
is useless if the experiment fails.

The confirmation panel is seeds 290000–290399, with two sampled repeats and
greedy evaluation for both replicas. Success retains the existing substantive
bar: at least five percentage points over both the frozen starting policy and
the independently trained ablated GAE control, positive paired 95% lower bounds,
positive sampled gains in each replica, and greedy lower bounds above minus
five points. Confidence intervals resample game seeds after averaging repeats
and the two fixed replicas. Complete uncensored outcomes are required.

A separate panel, seeds 300000–300399, uses the full-input policy during current
Antes 1–3 and the matched ablated policy afterward, against the ablated policy
throughout. A legal Ante decrease may reactivate the early policy. The terminal
win difference must have a positive pooled 95% lower bound and positive gains
in both replicas. This tests the contribution of decisions at those Ante levels;
it does not distinguish strategic preparation from improved immediate survival.

Feature exposure and adapter weight changes should be measured from ordinary
saved training observations and checkpoints, without additional rollouts. These
measurements can show whether a failed treatment scarcely encountered the inputs
or failed to learn from them. They cannot replace held-out win evidence.

Seeds 310000–310063 are reserved for a possible common-state mechanism study.
No outcome-dependent state selection is permitted. A concrete sampling and
intervention protocol must be fixed before using that panel; reserving seeds is
not a completed delayed-strategy test.

An [independent review of the proposed single-action diagnostic](../../eval/observation_generalization_20260922/analysis/delayed_decision_design_review.md)
recommends deferring implementation and execution. With 64 seed clusters, a
single early shop boundary may yield few differing decisions and does not
directly cover Standard-pack card choices. Its paired survival decomposition
would remain dependent on how suffix randomness is coupled and would require
additional event verification. The reserved seeds remain unused; the current
study and its acceptance criteria are unchanged.

The coordinator implements preparation, four serial training branches,
checkpoint and training-evidence validation, confirmation, early-policy transfer,
and final comparison. It validates completed work before resuming, preserves
attempt logs, and supervises up to three whole confirmation jobs concurrently.
Its latest [26 synthetic checks](../../eval/long_horizon_generalization_20260921_logical_kl/analysis/observation_ablation_candidate/.verification/observation_study_driver_tests_v3.xml)
passed, including stopping new job admission and retaining ownership of running
children after a failure. The unspent compute allowance is claimed only after
all initial checkpoint writes succeed.

The [first real v14 update audit](../../eval/observation_generalization_20260922/analysis/first_v14_update_health_1790108939060165000.json)
passed for full replica 0, update 142 to 143. All 64 complete trajectories and
10,290 terminal-labeled decision rows matched their policy hashes, seeds,
counters and recipe. All 317 active parameters advanced 26 Adam steps; the four
inactive critic parameters and their optimizer states remained unchanged. All
three new adapters acquired nonzero weights and moments.

The saved observations contained 324 rows with pack-enhancement information,
8,710 with a known Fool target, and 1,283 with positive consumable resale value.
These count raw field availability, including phases where the corresponding
action may be unavailable; they do not measure useful decisions or win impact.
Collection took 141.84 seconds and optimization 125.94 seconds, with 269.39
seconds total recorded for the update. This records real training and nonzero
adapter updates; it does not establish useful reliance on the inputs or held-out
improvement.

The historical [first completed branch summary](../../eval/observation_generalization_20260922/analysis/first_completed_branch_summary.json)
records all eight updates for full replica 0: 512 complete training games,
131,072 optimizer examples and 208 logical steps, with the prescribed seed order
and no recorded KL rollback. It used 1,929.54 seconds of accelerator time, leaving
8,326.33 seconds of the transferred allowance at that handoff. The coordinator
then started ablated replica 0. The full trajectory and checkpoint audit was
pending at that handoff and has since passed under the amendment; the metadata
summary itself does not establish a win gain.

The original study finished and passed its full evidence audit before preparation
or launch. The final seed audit, source freeze, initial checkpoint validation and
allowance transfer are complete. Both isolated implementations remain separate
from the primary source tree. The new source remains frozen after training
completion and external validation.

The [trained-control input-invariance check](../../eval/observation_generalization_20260922/analysis/control_input_invariance_1790111644465926000.json)
passed on CPU and MPS, at batch sizes 32 and 160. Across 160 saved training
observations, changing each hidden field separately and all three together left
the embeddings and every raw actor and critic output exactly unchanged. Inputs
were preserved. These were 20 read-only policy forwards, with no games, optimizer
steps, or held-out observations. This establishes the measured control behavior,
not CPU/MPS training parity or a win improvement.

The [recorded continuation decision](../../eval/observation_generalization_20260922/analysis/control_training_resume_decision_1790111755231047000.json)
resumed the same training processes after a 785.48-second pause. The pause and
diagnostics remain charged to the existing wall-time allowance. Frozen training
source, training exposure, seeds and all outcome thresholds remain unchanged.
The original audit failure is retained. The required external audit of every
training checkpoint, with the disclosed constant-row exception, and the CPU/MPS
proofs for both final control checkpoints have now passed. The original
coordinator stopped at its unchanged lock gate before the external validator
created the amended lock. Independent lock review is a separate admission step
before held-out evaluation.

The [separate critic-only supplement](../../eval/observation_generalization_20260922/analysis/control_endpoint_invariance_seed0_update143_1790112271296545000.json)
also passed on the first trained control. It checked `critic_only=True` under
the same four input perturbations, on CPU and MPS at both batch sizes, and
confirmed exact parity with the ordinary forward's critic outputs. It used 24
forwards and no games or optimizer steps; its 71.22-second training pause remains
charged to the existing allowance. The original 20-forward proof is preserved
with its narrower execution scope.

The [external amended validator](../../eval/observation_generalization_20260922/analysis/observation_training_audit_padding_v2_integration.md)
passed 43 focused checks. It reuses the original audit and outcome code with a
separate binding for the disclosed categorical-row predicate, without changing
frozen files or module globals. Its full 32-update audit and endpoint lock passed
in 34.70 seconds.
The [integration sequence](../../eval/observation_generalization_20260922/analysis/control_amendment_integration_plan.md)
was followed: the original owner finished training and stopped at the known
original audit failure, then both final control proofs and the amended audit
passed. Independent lock review precedes evaluation through the original
coordinator.

The historical [first completed matched-pair summary](../../eval/observation_generalization_20260922/analysis/first_completed_pair_summary.json)
verifies that both replica-zero branches finished with exit code 0. Each has
eight updates, 512 ordered training games, 131,072 optimizer examples and 208
logical steps, with the fixed recipe and no recorded KL rollback. Their final
checkpoint hashes are preserved. This was metadata verification; the full
trajectory and Adam audit and final functional proofs were pending at that
handoff and have since passed.

The pair used 4,705.96 seconds of the transferred allowance. The control's
2,776.41-second charge includes both diagnostic pauses, totaling 856.71 seconds.
The unchanged coordinator then started the second replica's control branch with
5,549.92 seconds remaining at that historical handoff.

The historical [final training-branch handoff](../../eval/observation_generalization_20260922/analysis/final_training_branch_handoff.json)
records the second control's successful completion and final checkpoint hash.
The first three branches used 6,581.22 seconds. The unchanged coordinator admitted
the final full-input branch with 3,674.65 seconds remaining. The final control
proofs and full training audit were scheduled after that last training job and
have now passed.

The [final completed training metadata summary](../../eval/observation_generalization_20260922/analysis/final_completed_training_summary.json)
independently verifies all four successful training attempts, their exact recipes
and ordered seed schedules, and no recorded training failure or KL rollback.
Across 32 updates, the branches executed 2,048 training games on 1,024 distinct
matched seeds, with 524,288 optimizer examples and 832 logical optimizer steps.
All four final checkpoint hashes and finalized training charges are recorded.
This metadata/hash summary remains distinct from the completed full training
audit and supplies no win evidence.

Both final learned controls passed combined ordinary and `critic_only=True`
input-invariance proofs on CPU and MPS at batch sizes 32 and 160:
[replica 0](../../eval/observation_generalization_20260922/analysis/control_endpoint_invariance_seed0_update150_1790117085192183000.json)
and [replica 1](../../eval/observation_generalization_20260922/analysis/control_endpoint_invariance_seed1_update150_1790117133814747000.json).
The proofs cover separate and combined perturbations of all three hidden fields,
exact embedding, actor and critic invariance, ordinary/critic-only parity, and
preservation of shared inputs. They establish the measured ablation behavior of
the final controls; they do not establish a gameplay gain or CPU/MPS training
parity.

The approved [validation amendment](../../eval/observation_generalization_20260922/analysis/control_validation_amendment_v2_1790117275633694000.json)
has SHA256 `5f3f382644783044cf83e06c77feefed42fd41d78046f4bace056849e28377b4`.
The [amended audit and lock receipt](../../eval/observation_generalization_20260922/analysis/amended_training_lock_run_1790117275633975000.json)
has SHA256 `4f8996dc0b9bafa0704b3fb487fa91c86a35b36854f93673dc308194c237f8ae`.
It records the passing full 32-update audit and the
[endpoint lock](../../eval/observation_generalization_20260922/locked_checkpoints.json),
SHA256 `5c7a3dabfb901ed0c2c3853f4958c4ecad2d78de64ff5eec7aa87ad21a0724ec`.
The original first-control and coordinator lock failures remain preserved; the
external amendment changes only the disclosed categorical padding-row predicate.
All other training checks and every held-out outcome threshold remain unchanged.
At this post-lock handoff, independent review was pending and held-out evaluation
had not started.


Operational evaluation diagnostic on September 23: after a source and ownership review, the original parent/control replica-0 inference processes were paused for approximately 40 seconds, with a detached identity-checking resume watchdog, then resumed unchanged. A 40-second baseline and recovery window bracketed the pause. Full-arm CPU share rose during the solo window, but completed-game counts were too small to establish a total-throughput benefit. Original coordinator scheduling was retained. This changes timing telemetry only at the application level; pause timestamps and CPU/count evidence are preserved in `eval/observation_generalization_20260922/analysis/evaluation_contention_run_1790123712352529000.json`, with limits and provenance in `evaluation_contention_interpretation_1790123712352529000.json`. No checkpoints, source, RNG streams, seeds, batch/thread settings, outcome criteria, or training resource accounting changed; no partial win comparisons were inspected.


Evaluation recovery on September 23: the original session became unavailable and native inspection found its entire process group absent; the stale running status did not establish liveness. Cause and exit code are unknown. All three partial greedy panels (703 terminal records) and stale status/attempt records were hashed and archived intact under `analysis/interrupted_evaluation_1790155113402409000`. Under the original protocol, the original coordinator now regenerates all three complete 400-seed greedy panels, while validating and reusing the six completed sampled panels whose byte hashes remain unchanged. New owner session11595, PID/group3455; original source, endpoints, seeds, settings, criteria and mandatory final `finalize_v2` remain unchanged. Recovery evidence is `analysis/evaluation_recovery_1790155113402409000.json` and `analysis/evaluation_recovered_owner_1790155113402409000.json`. The 703 interrupted partial executions are retained separately from final unique-panel totals; no partial win comparisons were inspected.


Host metadata subsequently confirmed a boot at 2026-09-23 11:05:18 local, after the original owner's last heartbeat. Those original processes could not persist across that boot. This supports host restart as an explanation for their absence, while the exact exit time or any earlier termination remains unknown. Read-only evidence: `analysis/evaluation_interruption_boot_check_1790155860.json`.
