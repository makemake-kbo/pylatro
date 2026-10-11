# Replicated pretraining-transfer confirmation

**Status: incomplete after a deadline timeout on September 24.** Supervisor
session 26101 exited with code 1 during the trainable-encoder arm's update-2,000
training measurement. Teacher collection and the frozen-encoder fit completed;
fresh imitation and both autonomous game panels never started. This study has
no generalization or win-rate result. Fresh scientific outcomes remain unopened,
and world-model work remains deferred until the existing transfer gaps are understood.

The [failure diagnostic](../../eval/pretraining_confirmation_20260923/analysis/failure_diagnostic_v1.json)
records 19 host sleep periods totaling 18,080 seconds. This approximately explains
the 18,026-second difference between wall time and recorded monotonic elapsed
time; the remaining 54 seconds are not separately attributed. The fixed wall
deadline expired during sleep, and the worker recorded its timeout immediately
after a logged wake. Recorded cumulative fitting time was about 3.18 hours,
below the eight-hour monotonic allowance. Both owned processes and process groups
were subsequently confirmed absent. The frozen contract prohibits retry, resume
or deadline extension; this attempt remains preserved and terminal.

Completed training measurements remain usable as diagnostics:

| Training snapshot | Frozen encoder F1 agreement / NLL | Trainable encoder U1 agreement / NLL |
| --- | ---: | ---: |
| Initial | 82.1629% / 0.672312 | 82.1629% / 0.672312 |
| Update 1,000 | 82.3168% / 0.659905 | 83.3685% / 0.608472 |
| Update 4,000 | 82.4583% / 0.648118 | Not reached |

The [matched update-1,000 diagnostic](../../eval/pretraining_confirmation_20260923/analysis/matched_training_1000_diagnostic_v1.json)
checks identical row order and exposure. U1's 1.0517-point training agreement
advantage closely repeats the first training order's 1.0512-point advantage at
the same update. U1 completed 2,000 updates, but that measurement is incomplete:
there is no update-2,000 metric report or checkpoint. Partial predictions are
excluded. These observations establish neither convergence nor fresh transfer.

## Frozen design of the incomplete study

The study was designed to test whether the encoder adaptation benefit in the
[completed diagnostic](pretraining_transfer_results_20260923.md) repeats with a
second training order and improves fresh sampled wins, and whether its early
decisions contribute to final wins under a fixed later policy. The design below
is preserved; its planned gameplay was not executed.

The diagnostic found a positive greedy comparison, but its sampled gain of
4.69 percentage points had a 95% interval of [−3.91, 13.28]. Training improvement
also exceeded fresh-state improvement. Those exposed results motivated this new
study; they are not its confirmation observations.

The two existing endpoints remain fixed. A new frozen/unfrozen encoder pair
starts from the same original corrected-head checkpoint, uses all 1,000 later
demonstration games and receives the same 4,000-update BC recipe. Only the new
pair's training seed changes from 0 to 1. This replicates training order from a
shared pretrained initialization and corpus, not independent pretraining.

| Panel | Fixed policies | Fresh seeds | Student games |
| --- | --- | ---: | ---: |
| Confirmation | Initial, both frozen endpoints, both unfrozen endpoints | 400 | 6,000 |
| Early intervention | Each replica's frozen control and early-unfrozen hybrid | 400 separate | 4,800 |

Every policy is evaluated greedily and with sampled streams 0 and 1. The common
initial policy is evaluated once per mode, yielding 1,200 physical games; its
use in two comparisons does not create independent observations. An additional
400 complete teacher trajectories provide shared fresh-state imitation data.

Each hybrid uses the unfrozen policy until the first pre-action observation at
Ante 4 or higher, then permanently switches to its matched frozen continuation.
Ante decreases never reactivate the early policy. All allocated games count,
including deaths before the handoff. This can identify a contribution of the
early segment, but cannot separate early survival from strategic preparation or
establish explicit planning.

The inherited acceptance bar is unchanged:

- Sampled gains against both matched frozen training and the initial policy must
  reach five percentage points, have paired 95% lower bounds above zero, and be
  positive in both training replicas.
- Greedy lower bounds must exceed minus five percentage points against both
  controls.
- The early sampled comparison must have a lower bound above zero and positive
  points in both replicas. No five-point minimum or greedy-superiority gate is
  added to this early comparison.

Intervals resample whole game seeds, keeping replicas, streams and shared
controls together. Imitation measurements remain diagnostic. The original
pretraining already trained its encoder; this later comparison does not explain
the whole original failure, teacher information advantage or residual errors.

Preparation passed 46 unit checks and independent code review. A separately
bounded check on 96 already exposed games verified exact all-frozen routing
parity for actions, non-timing outcomes and final sampler RNG state, using zero
and eight workers. It took 463.91 seconds, charged against the study's CPU and
outer limits. Both fresh seed panels passed the historical overlap scan.

The execution limits are 24 hours total, including eight hours for fitting and
imitation and sixteen hours for teacher collection and gameplay. The conservative
projection is about 20.33 hours; completion within the caps is not guaranteed.
Phases run serially, with no retries, resumed training, deadline extensions or
outcome-based checkpoint selection. Fresh outcome inspection waits until the
entire supervisor has actually exited successfully.

The [frozen protocol](../../eval/pretraining_confirmation_20260923/protocol_frozen_v1.md),
[manifest](../../eval/pretraining_confirmation_20260923/manifest_frozen_v1.json),
[admission preflight](../../eval/pretraining_confirmation_20260923/final_preflight_v1.json)
and [current execution handoff](../../eval/pretraining_confirmation_20260923/RESUME.md)
record the exact inputs and terminal state. Manifest SHA256:
`424f9cb112d86631f508b9d767af8caa9d90e4a571f194bc1e7d57fe77a93e11`.
