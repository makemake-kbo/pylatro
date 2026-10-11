# Testing consequence weighting against ordinary imitation

Status on September 25: **the study is complete and independently verified;
the generalization criteria failed**. Ordinary continuation substantially
improves training fit but sampled fresh wins are 28.19%, versus 28.00% for
the common base. Consequence weighting reaches 26.81%. The early-only hybrid
has a positive but unresolved difference. The long-horizon goal remains unproven.

All four workers exited zero after 4,000 updates and a full training-corpus scan. The
[independent paired endpoint audit](../../eval/consequence_weighting_20260924/paired_replica0_endpoint_verification_v1.json)
and [second paired endpoint audit](../../eval/consequence_weighting_20260924/paired_replica1_endpoint_verification_v1.json)
verify identical row order and every exposure count within each pair across
all 1,280,000 presentations. Of the 202,141 rows, 134,987 appeared six times
and 67,154 seven times in each arm. All four prediction-metric recounts passed,
as did independent recounts of
all six consequential-error categories and the 33 previously fatal cases.
The [direct checkpoint audit](../../eval/consequence_weighting_20260924/independent_checkpoint_tensor_verification_v1.json)
passed all 16 saved snapshots: initial models equal the base byte for byte,
critics stay unchanged, paired initial RNG and empty optimizer states match,
and the intended trainable groups move. The
[independent result audit](../../eval/consequence_weighting_20260924/independent_results_verification_v1.json)
recounts all 9,200 played game episodes and 1,391,954 action records, verifies
the permanent early-policy switch, and reproduces all seven comparisons and
five gates. Four numerical tests of that independent verifier also passed.
The [terminal verification receipt](../../eval/consequence_weighting_20260924/post_execution_verification_terminal_v1.json)
binds the actual successful process exits and artifact hashes. All study and
audit workers have exited; their scoped power guards have closed.

## Verified fresh-game results

The confirmation set contains 400 distinct seeds. Each endpoint is evaluated
greedily and with two fixed sampling streams; the common base is evaluated
once per mode/stream. Counts below preserve the two continuation replicas.

| Policy | Greedy wins / 400 | Sample stream 0 wins / 400 | Sample stream 1 wins / 400 | Combined sampled win rate |
| --- | ---: | ---: | ---: | ---: |
| Common base | 166 | 111 | 113 | 28.00% |
| Ordinary 0 | 166 | 118 | 117 | 29.375% |
| Weighted 0 | 156 | 108 | 109 | 27.125% |
| Ordinary 1 | 151 | 106 | 110 | 27.00% |
| Weighted 1 | 166 | 104 | 108 | 26.50% |

The separate early-attribution set contains another 400 seeds. The hybrid uses
the weighted policy until the first pre-action Ante 4 crossing, then its matched
ordinary policy permanently.

| Policy on early-attribution set | Sample stream 0 wins / 400 | Sample stream 1 wins / 400 | Combined sampled win rate |
| --- | ---: | ---: | ---: |
| Ordinary 0 | 112 | 122 | 29.25% |
| Early hybrid 0 | 128 | 130 | 32.25% |
| Ordinary 1 | 116 | 124 | 30.00% |
| Early hybrid 1 | 136 | 113 | 31.125% |

All intervals below use the frozen 10,000-resample whole-seed bootstrap, keeping
replicas and sampling streams together. Shared base episodes are counted once.
The 9,200 played episodes are not 9,200 independent seed clusters.

| Comparison | Pooled change, pp | Paired 95% interval, pp | Replica 0 / 1 change, pp | Fixed criterion |
| --- | ---: | ---: | ---: | --- |
| Sampled weighted − ordinary | −1.375 | [−3.875, +1.063] | −2.250 / −0.500 | Fail |
| Sampled weighted − base | −1.188 | [−4.563, +2.125] | −0.875 / −1.500 | Fail |
| Sampled ordinary − base | +0.188 | [−2.938, +3.252] | +1.375 / −1.000 | Descriptive control |
| Greedy weighted − ordinary | +0.625 | [−2.878, +4.125] | −2.500 / +3.750 | Guardrail passes |
| Greedy weighted − base | −1.250 | [−5.750, +3.125] | −2.500 / 0.000 | Guardrail fails |
| Greedy ordinary − base | −1.875 | [−6.250, +2.250] | 0.000 / −3.750 | Descriptive control |
| Early hybrid − ordinary | +2.063 | [−0.563, +4.688] | +3.000 / +1.125 | Fail |

The sampled treatment is below both controls in both replicas, although the
intervals include zero and therefore do not establish a reliable harmful effect.
Neither sampled comparison approaches the required five-point gain. The greedy
guardrail against the base fails because its lower bound extends below −5 points;
this does not establish a decline of that size. The early point estimate is
positive in both replicas, but its lower bound remains below zero. Four of five
scientific gates fail. The successful audits validate these negative or unresolved
results; they are not evidence that the intervention succeeded.

The main new finding is a replicated disconnect between improved training fit
and fresh autonomous wins under this continuation recipe. Ordinary NLL falls
from 0.496954 to about 0.3717 and agreement rises by about three points, yet the
fresh sampled change is only +0.188 points with an interval spanning zero.
Weighting fits many more selected costly training errors but does not improve
fresh wins. This does not isolate whether the remaining gap is imitation
overfitting, action-family interference, teacher-label limitations, or errors
on states reached by the student. The early and full-policy contrasts use
different seed sets, so subtracting them would not identify a late-policy cause.

The subsequent [held-out teacher-state diagnostic](continuation_transfer_diagnostic_20260925.md)
finds overfitting in both continuations: ordinary training NLL falls to about
0.3717 while held-out NLL rises from 0.6391 to about 0.6904. Weighting worsens
held-out play imitation further. That exposed-state diagnostic does not amend
this completed study or supply a new confirmation set. World-model work remains
deferred.

## Training findings and experimental provenance

This experiment follows the [complete training-cost replay](consequence_weighting_preparation_20260924.md).
The corrected transformer misses 1,395 teacher blind clears and introduces
33 immediate losses through alternative plays on the existing 1,000-game
training corpus. These are one-step teacher-path interventions, not autonomous
win rates. The test asks whether giving those consequential errors more weight
improves transfer beyond ordinary additional fitting.

Both arms keep the current architecture, observations, labels and training
examples. Treatment uses the previously fixed consequence weights; control uses
ordinary imitation. Two paired continuation replicas each run 4,000 updates
from the same Stage 2 base with matching row order and compute. These replicate
continuation fitting, not independent original pretraining. Sparse coverage and
teacher-label quality remain separate possible gaps. World-model work stays
deferred.

An [independent starting-metric recount](../../eval/consequence_weighting_20260924/training_metric_baseline_v1.json)
matches all 202,141 saved row identities and the recorded game/family metrics.
Base-model NLL is 0.496954 overall and 2.992501 on the 5,850 upweighted rows.
The base agrees with none of those selected labels, as expected from selecting
consequential base-mode errors. At this fixed starting model, those rows account
for 17.43% of ordinary loss and 34.15% of the weighted loss. These are loss-value
shares, not gradient-norm shares. The fixed weighted objective starts at 0.598074;
comparing that directly with the ordinary objective would not measure learning.
The [completed control recount](../../eval/consequence_weighting_20260924/training_metric_r0_ordinary_4000_v1.json)
and [weighted recount](../../eval/consequence_weighting_20260924/training_metric_r0_weighted_4000_v1.json)
use the same rows and objectives:

| Training measure | Common base | Ordinary, replica 0 | Weighted, replica 0 | Ordinary, replica 1 | Weighted, replica 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher agreement, all rows | 85.83% | 88.88% | 88.45% | 88.91% | 88.48% |
| Ordinary NLL | 0.496954 | 0.371671 | 0.388691 | 0.371731 | 0.388709 |
| Fixed consequence-weighted NLL | 0.598074 | 0.446365 | 0.423565 | 0.445957 | 0.422723 |
| NLL on the 5,850 selected costly errors | 2.992501 | 2.164965 | 1.756754 | 2.150508 | 1.720739 |
| Exact teacher actions on those selected rows | 0 / 5,850 | 1,966 / 5,850 | 2,775 / 5,850 | 1,983 / 5,850 | 2,829 / 5,850 |

Ordinary continued training alone learns more of the existing labels, including
33.61% and 33.90% of the selected costly errors across the two replicas. This is
evidence of remaining fitting headroom in the current model. Weighting raises
selected-error agreement to 47.44% and 48.36%, improvements of 13.83 and 14.46
points over the matched controls. Overall agreement falls by 0.425 and 0.429
points, while ordinary NLL rises by 0.017020 and 0.016978. The fixed weighted
objective improves by 0.022800 and 0.023234.

Both continuation pairs therefore show the same training tradeoff: loss
allocation changes which existing labels the model learns under the same
exposure budget. These measurements concern training states; the completed
fresh-game comparisons above did not establish an autonomous gain. Initial
model/RNG states and recipe/row-stream metadata match within both pairs;
direct paired tensor inspection now verifies the saved states. No recipe or
acceptance threshold changed.

The [second control endpoint audit](../../eval/consequence_weighting_20260924/second_control_endpoint_verification_v1.json)
independently certifies all 4,000 updates and 1,280,000 presentations and binds
the completed checkpoint to the full prediction recount. It repeats the first
control's training gains: 88.91% overall agreement and 33.90% agreement on the
selected costly errors, versus 88.88% and 33.61% in replica 0. Both controls
have the same exposure histogram but different row orders. This strengthens
the evidence for remaining fitting headroom under the current architecture;
it replicates continuation fitting from one common pretrained base, not fresh
pretraining or autonomous-game improvement. The completed
[second weighted metric recount](../../eval/consequence_weighting_20260924/training_metric_r1_weighted_4000_v1.json)
now independently verifies the weighting comparison in the final table column.

The second control's [fixed-cost recount](../../eval/consequence_weighting_20260924/training_cost_strata_r1_ordinary_4000_v1.json)
finds 13 teacher actions, 16 repeats of the known fatal base action and four
unreplayed alternatives on the 33 previously fatal training states. Its new
alternatives have not been assigned gameplay consequences. Starting metadata
matches between the second control and its weighted partner, including reported
model/RNG hashes and a fresh empty optimizer; direct tensor inspection now
confirms the paired starting states.

A [raw-decision comparison of the two ordinary replicas](../../eval/consequence_weighting_20260924/ordinary_replica_agreement_v1.json)
shows what their nearly identical averages conceal. The comparison joins all
202,141 predictions by state and label identity and reconciles both overall
metrics and all six previously recounted cost strata. The selected and fatal
sets below are subsets of the all-row set.

| Training cases | Both match teacher | Only replica 0 matches | Only replica 1 matches | Neither matches |
| --- | ---: | ---: | ---: | ---: |
| All 202,141 decisions | 177,130 | 2,529 | 2,601 | 19,881 |
| Selected 5,850 costly errors | 1,542 | 424 | 441 | 3,443 |
| Previously fatal 33 cases | 9 | 1 | 4 | 19 |

On 865 selected rows, only one continuation learns the teacher label. However,
both controls repeat the same previously assessed costly base action on 2,505
selected rows, including 12 of the 33 previously fatal states. Thus there are
specific residual errors shared by both continuations as well as variation in
which labels they learn. Other non-teacher alternatives remain unreplayed, so a
label mismatch alone does not establish a harmful action. These
are descriptive training-state comparisons, without fresh-game evidence or an
isolated architectural explanation. No acceptance gate changed.

A [four-way raw-prediction comparison](../../eval/consequence_weighting_20260924/training_replica_comparison_v1.json)
now follows those shared residual errors into both weighted endpoints. It
reconciles all four models' overall and action-family metrics, all six cost
strata, the earlier ordinary-replica comparison and the first paired tradeoff.

| Cases where both controls repeat the assessed base error | Rows | Both weighted runs match teacher | Only weighted replica 0 | Only weighted replica 1 | Neither weighted run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared costly errors | 2,505 | 406 | 155 | 172 | 1,772 |
| Shared known fatal actions | 12 | 6 | 1 | 0 | 5 |

Thus both weighted runs learn the same teacher correction for some errors that
survive both ordinary continuations. Many shared errors remain. On the twelve
shared fatal states, weighted replicas 0 and 1 repeat the known fatal base
action four and five times, respectively; each chooses one unreplayed
alternative. Failing to match the teacher does not by itself classify that
alternative as fatal. These results do not isolate a representation limit or
establish autonomous-game improvement.

The same four-way comparison extends the
[first paired recount](../../eval/consequence_weighting_20260924/training_tradeoff_r0_v1.json)
to localize both replicas' tradeoffs. The three groups within each replica
partition all 202,141 decisions.
“New matches” are labels matched only by weighting; “lost matches” are labels
matched only by ordinary fitting.

| Replica | Training rows | Decisions | New matches | Lost matches | Net matches |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Selected costly plays | 5,850 | 1,008 | 199 | +809 |
| 0 | Other plays | 41,299 | 433 | 1,370 | −937 |
| 0 | Other action families | 154,992 | 613 | 1,344 | −731 |
| 0 | All rows | 202,141 | 2,054 | 2,913 | −859 |
| 1 | Selected costly plays | 5,850 | 1,059 | 213 | +846 |
| 1 | Other plays | 41,299 | 423 | 1,460 | −1,037 |
| 1 | Other action families | 154,992 | 653 | 1,330 | −677 |
| 1 | All rows | 202,141 | 2,135 | 3,003 | −868 |

Among other action families, discard agreement falls from 66.09% to 62.93%
(354 fewer matches across 11,215 decisions), and Joker-sale agreement falls
from 55.54% to 52.03% (225 fewer matches across 6,415 decisions) in replica 0.
Replica 1 repeats these regressions: discard agreement falls from 65.09% to
62.35% (307 fewer matches), and Joker-sale agreement from 53.97% to 51.08%
(185 fewer matches). Even the selected costly subset contains 199 and 213
lost matches alongside 1,008 and 1,059 new ones. The recount independently
matches every family aggregate and all four overall metrics. These are
exact-label regressions on training states; their gameplay
consequences have not been replayed. The result does not establish harmful
autonomous behavior, a representation bottleneck or a net win-rate benefit.

The descriptive breakdowns for
[ordinary fitting](../../eval/consequence_weighting_20260924/training_cost_strata_r0_ordinary_4000_v1.json)
and [weighted fitting](../../eval/consequence_weighting_20260924/training_cost_strata_r0_weighted_4000_v1.json)
show improvements on both early and late selected errors. All categories below
were identified from the base model's saved one-step replay. Categories overlap.
Each action-count cell lists **teacher action / original base action / different,
unreplayed action**, in that order.

| Base-error category | Rows | Ordinary 0 | Weighted 0 | Ordinary 1 | Weighted 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early upweighted, Ante 1–3 | 1,443 | 448 / 781 / 214 | 696 / 554 / 193 | 425 / 820 / 198 | 709 / 536 / 198 |
| Late upweighted, Ante 4+ | 4,407 | 1,518 / 2,229 / 660 | 2,079 / 1,667 / 661 | 1,558 / 2,229 / 620 | 2,120 / 1,675 / 612 |
| Missed teacher blind clear | 1,395 | 441 / 707 / 247 | 1,027 / 216 / 152 | 455 / 701 / 239 | 1,051 / 201 / 143 |
| New immediate loss | 33 | 10 / 18 / 5 | 24 / 6 / 3 | 13 / 16 / 4 | 25 / 5 / 3 |
| Last-hand upweighted | 88 | 27 / 40 / 21 | 43 / 28 / 17 | 29 / 43 / 16 | 42 / 28 / 18 |

In replica 0, on the 33 formerly fatal cases, weighting repeats the same known fatal action
six times, compared with 18 for ordinary fitting. It chooses the teacher action
24 times versus ten; its three other choices have no replayed consequence here.
All ten control teacher matches remain teacher matches. Eleven control fatal
repeats become teacher actions, two become unreplayed alternatives and five
remain fatal repeats. One unreplayed control choice becomes the known fatal
base action; the control alternative's consequence remains unmeasured.
The [second weighted cost recount](../../eval/consequence_weighting_20260924/training_cost_strata_r1_weighted_4000_v1.json)
finds 25 teacher actions versus its control's 13, five known fatal repeats versus
16, and three unreplayed alternatives versus four.

These counts concern saved mode actions on teacher states, not autonomous wins
or losses. Four synthetic tests cover classification and corrupted joins. The
paired endpoint audit separately joins by `(seed, step)` and reproduces all six
category/action/game counts, NLL means and all 33 fatal-row relations for both
arms. The earlier control
[verification record](../../eval/consequence_weighting_20260924/training_cost_strata_r0_ordinary_verification_v1.json)
distinguishes unreplayed alternatives from the two previously assessed actions.
This post hoc training diagnosis changes no acceptance gate.

The [execution protocol](../../eval/consequence_weighting_20260924/execution_protocol_v1.md)
and [frozen manifest](../../eval/consequence_weighting_20260924/execution_manifest_v1.json)
specify all fitting, evaluation and acceptance details before fresh outcomes.
The manifest SHA256 is
`8b7d1b6ffaed1b1412171a97502c0cc88124f79876d2de96fd9070041091859c`.
All four fixed endpoints must finish and pass paired-state/exposure checks before
the evaluator opens the reserved fresh panels. No checkpoint selection or
automatic retry is allowed.

The evaluation implementation passed an identity check on 34 exposed games in
both modes: the base and base/base hybrid reproduce all **5,595 greedy actions**
and **4,755 sampled actions**, outcomes and critic forecasts exactly. The hybrid
permanently switches at the first pre-action Ante>=4, even if a voucher later
reduces Ante. Five routing tests cover this latch, changing batch positions and
new games. A separate check with two distinct policies reproduces selected
likelihoods, modes and critic outputs exactly on 32 cached training rows, while
preserving the sampling RNG. These are evaluator checks, not gameplay gains.

All fifteen earlier weight/training tests and four new statistical tests pass.
The statistical checks independently reconstruct the bootstrap using seed
frequencies, verify shared-baseline accounting and test acceptance boundaries.
The complete native evaluation smoke exited zero in 714 seconds; the distinct
policy check also exited zero. Their receipts are pinned by the
[admission record](../../eval/consequence_weighting_20260924/admission_v1.json).
An [independent admission check](../../eval/consequence_weighting_20260924/execution_admission_verification_v1.json)
verifies final-plan equality, every evidence hash and all panel identities/counts.

A subsequent [independent trace recount](../../eval/consequence_weighting_20260924/independent_smoke_trace_verification_v1.json)
checks all 136 smoke games and 20,700 action records without importing the
evaluator. It reconstructs every game's first Ante4 crossing and verifies all
routes, step counts and win/Ante8-clear identities. Five tests also reject
corrupted traces and false routing summaries. No smoke game returned below
Ante4 after crossing; that boundary is covered by a synthetic test. This checks
evaluation accounting, not a learned gameplay benefit. The independent final
result verifier and its four numerical bootstrap tests have now passed on the
completed study, as recorded above.

The fixed evaluation comprises 6,000 confirmation games and 3,200 separate
early-attribution games. The shared base is evaluated once per mode/stream.
Uncertainty resamples whole seeds with both continuation replicas and sampling
streams inside each cluster. Weighted imitation must beat both ordinary fitting
and the common base by at least five sampled percentage points, with positive
paired lower bounds and positive gains in both replicas. Greedy guardrails and
the existing early-policy attribution gate must also pass. Better training fit
alone cannot meet the goal. The completed independent recount verifies every
raw count and statistical contrast; the scientific criteria fail as shown above.

The coordinator ran native workloads serially and recorded all 28 successful
subprocess exits before its own successful exit. Current process details and the latest
continuation instructions are in the [resume record](../../eval/consequence_weighting_20260924/RESUME.md).
