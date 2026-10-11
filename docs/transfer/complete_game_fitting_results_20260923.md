# Complete-game fitting: encoder training closes most of the small-set gap

Allowing the encoder to train increased exact teacher-action agreement from
**83.82% to 98.75%** on the same 32 complete training games. Both arms started
from identical corrected supervised weights and received the same 320,000 row
exposures. This identifies a substantial constraint from freezing the inherited
representation under the tested training recipe. It does not measure fresh-state
generalization or autonomous wins, and it does not explain the entire original
pretraining transfer gap.

## Fixed comparison

The data contain 6,569 decisions from 32 games selected by a fixed hash ranking,
without filtering by outcome. They come from the later 1,000-game action-head
corpus, not the original 5,000-game corpus. Both arms start from corrected
supervised seed 0, not a PPO descendant. All actor heads train in both arms;
embeddings and the transformer train only in the unfrozen arm. Critics remain
frozen. Training uses full-action behavioral cloning, a fresh AdamW optimizer,
learning rate 3e-5, identical shuffled rows, disabled dropout, and 1,000 updates
of 320 examples. Each row appears 48 or 49 times.

Both arms and all five scheduled snapshots completed. Total elapsed time,
including compatibility checks and measurements, was 3,619.77 seconds, within
the fixed 7,200-second ceiling. Initial predictions match exactly for every row.
There was no endpoint selection, retry, or extension.

## Results

Exact full-action agreement on the training trajectories:

| Update | Frozen encoder | Trainable encoder |
| --- | ---: | ---: |
| 0 | 81.90% | 81.90% |
| 100 | 82.23% | 84.12% |
| 300 | 82.80% | 89.92% |
| 600 | 83.16% | 95.46% |
| 1,000 | 83.82% | 98.75% |

At the fixed endpoint, action NLL is **0.58228 versus 0.04691** and mean
teacher-action probability is **76.14% versus 96.66%**. Weighting each game equally
gives agreement of 83.61% versus 98.75%, so the aggregate result is not driven
solely by longer trajectories.

| Teacher action family | Decisions | Frozen encoder | Trainable encoder |
| --- | ---: | ---: | ---: |
| Play a hand | 1,466 | 72.10% | 99.93% |
| Discard | 333 | 45.95% | 98.80% |
| Sell a Joker | 216 | 26.85% | 99.07% |
| Buy | 1,395 | 93.84% | 99.86% |
| Claim a pack item | 685 | 79.85% | 89.64% |

These are complete actions, including their family and arguments. Historical
corrected-head fitting metrics conditioned on the teacher family and therefore
cannot be compared directly with these percentages.

The unfrozen arm has 82 disagreements, versus 1,063 in the frozen arm. Of those
82, **71 are pack claims**; every one selects another pack slot. Other errors are
four discards, two buys, two Joker sales, one play, one pack skip, and one
hand-targeted consumable. Only two of the 32 recorded trajectories have no
greedy disagreement at all. Exact disagreement alone does not establish a bad
action, and the offline teacher-path probabilities do not measure win rates.

The saved-row [residual analysis](../../eval/pretraining_transfer_20260923/analysis/residual_pack_claim_analysis_v1.md)
localizes 70 of the 71 pack errors to 231 Standard playing-card claims. Other
pack compositions reach 453/454 correct claims. Sixty-four of those 70 Standard
errors choose slot zero. This does not establish a score tie: slots and absolute
positions have separate embeddings, most chosen cards differ in encoded rank
or suit, and full score vectors were not saved. No two complete observations
are identical in this panel. The known missing enhancement input remains a
plausible contributor, but the shards lack the original offered enhancement
identities, so these errors cannot be attributed to it from saved rows alone.
Standard-card fitting continued to improve through the fixed endpoint; no
unavoidable error floor has been demonstrated.

## What this resolves and what remains

The corrected policy can fit nearly all of the observed hand and sale decisions
in these complete training games when the representation is allowed to change.
The frozen representation was a consequential restriction under this recipe.
The experiment does not show that more frozen-head training could never work,
that a different optimizer would fail, or that the same fit is achievable over
the full corpus. It uses one training initialization and heavily repeats a small
set; memorization is a possible explanation for the improvement.

The next direct questions are whether the improvement survives training on the
larger corpus, whether it transfers to fresh teacher-generated states, and
whether the resulting policy wins when it chooses its own actions. Remaining
pack errors require separate analysis of action encoding, indistinguishable
items, unavailable information, and potentially harmless alternatives. Teacher
access to the full simulator, scoring, and copied RNG remains an unresolved
information difference. World-model work remains deferred.

## Runtime consistency

The prepared bounded replay passed after fitting completed. Two existing
trajectories, one win and one loss, contain 367 unique decisions. Saved
observations matched the current compiled FastRunner and Gym; rewards, legality,
states, and terminal outcomes matched. Ordinary evaluation with zero and eight
workers also matched. It took 23.31 seconds within its separate 300-second
ceiling, with no policy inference or teacher search. Only two workers received
games. This establishes consistency on this bounded panel, not all reachable
states. The previously documented deck-exhaustion correction remains a real
historical runtime difference; this replay does not quantify its impact.

## Evidence

Artifacts live under `eval/pretraining_transfer_20260923/`:

- `stage1_manifest_frozen_v1.json` and `stage1_launch_frozen_v1.json` record
  the fixed recipe and source/data pins.
- `arms/{frozen_encoder,unfrozen_encoder}/` contain all predictions, metrics,
  row-exposure logs, and checkpoints.
- `execution/complete.json` records both completed endpoints.
- `analysis/fit_owner_terminal_28872_17901743774921558.json` records the observed
  supervisor exit 0.
- `analysis/frozen_encoder_artifact_audit_v1.json` and
  `analysis/unfrozen_and_paired_artifact_audit_v1.json` independently verify all
  ten snapshots, paired baselines, identical row ordering/exposure, and artifact
  hashes. All 50 first-arm and 64 second-arm/paired checks passed. Reported
  parameter movement was checked, but checkpoint tensors were not independently
  decoded. The paired audit SHA256 is
  `028dee6d1cd927d3440f69303051eff9cc85d82198bbf435e72c9ca14fd1b039`.
- `analysis/runtime_parity_replay_v1/replay_report.json` records every bounded
  runtime comparison; `complete.json` records successful completion.
- `analysis/residual_pack_claim_analysis_v1.json` preserves all 71 pack-error
  rows, source/data hashes and type-specific curves (SHA256
  `7e953e7c6068f31cc65c02c7e58f585688b1c153db9b69eac6de760e493ad3d8`).

The broader long-horizon generalization objective remains unproven.
