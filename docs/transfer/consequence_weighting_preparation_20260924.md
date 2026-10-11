# Consequential imitation errors in the existing training corpus

The current corrected transformer still makes costly hand choices on its own
training states. Replaying every recorded decision in the later 1,000-game
Stage 2 corpus reproduces the cached observations, legal teacher actions and
terminal outcomes exactly. On these teacher paths, substituting the saved
trainable-encoder model's played hand misses **1,395 teacher blind clears** and
introduces **33 immediate losses in 33 games**, including 26 recorded teacher
wins. These are separate one-step interventions, not an autonomous win rate.

This is the later corpus: **202,141 decisions, 47,149 plays and 728 teacher
wins**. It is distinct from the original 5,000-game pretraining corpus, whose
audited teacher win rate was 74.56%. The base model is the fixed Stage 2
trainable-encoder checkpoint at update 4,000. No new scientific fitting or
fresh evaluation game has run in this preparation.

The [structured results](data/consequence_weighting_preparation_20260924.json)
pin the inputs, replay results, independent verification and training smoke.
The [design draft](../../eval/consequence_weighting_20260924/training_protocol_draft_v1.md)
compares ordinary additional imitation with a fixed consequence-weighted loss,
using the current model, observations, labels, row order and update budget.
Its historical preparation-status header is superseded by this report.

## What the complete replay establishes

Of 47,149 recorded teacher plays, the model selects a play on 46,427 rows:
35,430 identical actions and 10,997 different hands. The remaining **722** model
choices belong to other action families and are unassessed by this hand-score
criterion. They are not counted as zero-scoring plays.

| Measure | Ante 1–3 | Ante 4+ | All |
|---|---:|---:|---:|
| Recorded teacher plays | 16,326 | 30,823 | 47,149 |
| Missed teacher blind clears | 460 | 935 | 1,395 |
| New immediate losses | 10 | 23 | 33 |
| Decisions receiving extra weight | 1,443 | 4,407 | 5,850 |

All 33 new immediate losses occur with one hand remaining. There are 904 such
teacher plays overall; 88 receive extra weight. Both teacher and model clear
the blind in 20,409 assessed decisions, which receive no extra weight even if
the model scores fewer points.

The replay checks every cached observation array, recorded action and terminal
result in all 1,000 games. Each different eligible hand receives factual and
alternative branches from identical copied full simulator state and RNG,
with unchanged-parent and factual-post-state checks: 21,994 branches in total.
Exact action matches have identical consequences by identity. This validates
the recorded trajectories under the archived runtime; it does not rerun the
teacher's decision algorithm or establish that its labels are optimal.

## The fixed intervention

The rule was fixed before the full cost replay. Let R be the remaining blind
requirement, T the teacher score and P the model score. For eligible plays,
severity is `clip((min(T,R) - min(P,R)) / R, 0, 1)` when R is positive.
A missed teacher clear or newly introduced immediate loss gives severity one;
both branches clearing gives severity zero. Exact matches and unassessed
action families retain raw weight one. Raw weight is `1 + 4 * severity`, then
divided by its mean across **all** training decisions.

The rule increases weight on **5,850 / 202,141 rows (2.894%)**, spread across
964 games. Those rows receive **6.803% of the total weighted loss mass**.
The raw mean is 1.0419385; normalized weights range from 0.9597495 to 4.7987477.
Weights remain fixed from the base model throughout fitting. No Joker name,
card rank, exposed held-out case or fresh outcome enters the criterion.

This tests the allocation of existing supervision. It supplies no missing
training states and cannot resolve sparse coverage by itself. Full simulator
and copied-RNG information is used to construct training weights; the student
does not receive it at inference. Immediate score loss is not full-game regret.

## Verification and remaining work

Independent reconstruction passes 437,310 checks covering every play identity,
severity, raw and normalized weight, game terminal and aggregate count. It also
matches every play in the four-game smoke replay. This is independent arithmetic
and provenance verification, not a second full simulator replay.

All fifteen targeted tests pass: eight weight-rule boundaries, four paired
row-stream checks and three loss/gradient/optimizer checks. The native training
smoke reproduces the saved model's first 320 predictions with zero NLL error,
starts both arms from identical model/RNG states and rows, moves every intended
parameter group, and leaves the critic unchanged. The weighting changes the
update. Model, optimizer, exposure and row-stream checkpoint round trips pass.
These disposable one-update smoke checkpoints cannot seed scientific fitting.

Independent text scans reserve 400 unused confirmation seeds and 400 separate
early-attribution seeds, disjoint from prior reservations. The failed Stage 3
study's outcome files remain unopened; only its seed-reservation metadata was
read. The reservation cannot rule out undocumented binary or procedural use.

Scientific admission still requires the evaluation implementation, its smoke
checks, frozen analysis and a concrete execution manifest. The intended test
uses two paired continuation replicas from the common base, ordinary versus
weighted imitation, and fixed 4,000-update endpoints. Fresh outcomes stay gated
until all four endpoints are complete and verified. Replicated sampled wins,
greedy guardrails and the existing early-policy attribution gate remain required.
Preparation has established neither a learned win improvement nor long-horizon
generalization. World-model work remains deferred.
