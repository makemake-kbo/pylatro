# Consequences of corrected-policy hand errors

Training the encoder reduces some costly hand-selection errors on the already
exposed Stage 2 held-out states. Across **6,113 recorded plays from 128 games**,
the trainable policy clears 2,699 blinds versus 2,652 for the frozen encoder:
**+0.769 percentage points**, paired 95% interval **[+0.200, +1.336]**.
It also reduces choices scoring at most half the teacher's score from 511 to 408.

**46 of the 47 additional clears occur at Ante 4 or later.** The early clear
difference is only +0.046 points [-0.762, +0.876]. Both policies introduce seven
immediate game losses where the teacher survives. The result supports a limited
improvement in immediate hand selection; it does not establish that better early
decisions account for the earlier autonomous win gain or prove long-horizon
generalization. World-model work remains deferred.

The [complete analysis](../../eval/corrected_hand_cost_20260924/analysis_v1.json)
and [compact report data](data/corrected_hand_cost_20260924.json) contain every
policy, stratum, fixed contrast, count, interval and immediate-loss state.

## Comparison and coverage

All three policies use the corrected action heads: the common starting policy,
the frozen-encoder endpoint and the trainable-encoder endpoint after 4,000 matched
updates. The primary comparison is trainable minus frozen; both comparisons with
the starting policy are secondary. This is not a direct comparison with the
original epoch-10 policy in the [original-corpus audit](original_hand_cost_20260924.md).
Those panels differ in data and runtime, so their raw error rates do not isolate
the effect of replacing the original heads.

The [protocol](../../eval/corrected_hand_cost_20260924/protocol_v1.json) fixed all
128 already-exposed games, all recorded teacher plays, three contrasts and
early/late strata before executing new action branches. All **26,045 decisions**
reproduce exactly under the Stage 2 archived runtime: all fourteen observation
arrays, saved action identities, legality and recorded terminal outcomes match.
There are 95 recorded teacher wins; this is not a new teacher evaluation.

For each play, the worker finishes the factual game before evaluating alternate
actions. It copies the full runner, controller, history and hidden RNG state,
preserves mutable aliases, and restores the factual card-identity counter.
Replaying the recorded action must reproduce its complete factual post-state.
Each unique action is then evaluated once from the same state; identical policy
actions share that outcome. There are 9,065 unique action branches in total.
No new policy forward pass, teacher query, optimization or autonomous game runs.

## Remaining errors

The teacher action clears 2,882 of the 6,113 states and immediately loses in 33.
The following counts are separate one-action interventions on those states.

| Outcome | Starting policy | Frozen encoder | Trainable encoder |
| --- | ---: | ---: | ---: |
| Exact teacher action | 3,980 | 4,004 | 4,156 |
| Different played hand | 2,011 | 2,008 | 1,852 |
| Another action family | 122 | 101 | 105 |
| Blind cleared | 2,655 | 2,652 | 2,699 |
| Teacher clears, policy does not | 336 | 342 | 289 |
| Policy clears, teacher does not | 109 | 112 | 106 |
| New immediate loss where teacher survives | 9 | 7 | 7 |
| Games containing a new immediate loss | 8 | 6 | 7 |
| Immediate teacher loss averted | 1 | 1 | 0 |
| All immediate losses | 41 | 39 | 40 |
| Played hand scores lower than teacher | 1,481 | 1,504 | 1,341 |
| Different played hand ties teacher score | 276 | 254 | 268 |
| Played hand scores higher than teacher | 254 | 250 | 243 |
| Played hand scores at most half the teacher | 496 | 511 | 408 |

Among its 1,852 different played hands, the trainable policy scores lower 72.4%
of the time, ties 14.5% and scores higher 13.1%. Disagreement is still frequently
associated with a scoring error. Lower immediate score is not necessarily worse
long-term value, and a missed clear is not a lost game: a policy can choose a
discard or consumable instead. Immediate loss means the actual engine enters
`GAME_OVER`, including its normal survival effects.

For a direct comparison, both trained policies choose a play in 5,988 states.
The trainable policy scores higher in 515, lower in 335 and equally in 5,138;
125 other-family states are excluded from this score comparison. Both policies
still have seven new immediate losses, but not on the same set of states. The
trainable policy has one more total immediate loss because it does not avert
the recorded loss that the frozen policy averts.

The first new immediate-loss state in seed/step order is shared by all three
policies: **seed 850000015, step 28, Ante 2**, with one hand left and 800 points
needed. The teacher scores **1,989**; every policy chooses the same **795-point**
hand and loses. This illustrates a residual error after both head correction
and encoder training. It does not diagnose whether the cause is input information,
representation, action ranking or optimization.

## Fixed early and late comparisons

All rates below use every recorded teacher-play state in that stratum, including
states where a policy chooses another action family. There are 2,152 early
states across 128 games and 3,961 later states across 125 games.

| Stratum | Starting clears | Frozen clears | Trainable clears | New losses: starting / frozen / trainable |
| --- | ---: | ---: | ---: | --- |
| All | 2,655 | 2,652 | 2,699 | 9 / 7 / 7 |
| Ante 1–3 | 1,062 | 1,065 | 1,066 | 2 / 2 / 2 |
| Ante 4+ | 1,593 | 1,587 | 1,633 | 7 / 5 / 5 |

Differences are percentage points, followed by paired 95% intervals. A positive
clear difference is favorable; a positive immediate-loss difference is unfavorable.

| Contrast | Stratum | Blind clear difference | All immediate-loss difference |
| --- | --- | --- | --- |
| Trainable − frozen | All | +0.769 [+0.200, +1.336] | +0.016 [−0.066, +0.100] |
| Trainable − frozen | Ante 1–3 | +0.046 [−0.762, +0.876] | 0.000 [0.000, 0.000] |
| Trainable − frozen | Ante 4+ | +1.161 [+0.439, +1.875] | +0.025 [−0.102, +0.156] |
| Trainable − starting | All | +0.720 [+0.143, +1.291] | −0.016 [−0.100, +0.066] |
| Trainable − starting | Ante 1–3 | +0.186 [−0.611, +1.000] | 0.000 [0.000, 0.000] |
| Trainable − starting | Ante 4+ | +1.010 [+0.278, +1.739] | −0.025 [−0.155, +0.103] |
| Frozen − starting | All | −0.049 [−0.283, +0.197] | −0.033 [−0.084, 0.000] |
| Frozen − starting | Ante 1–3 | +0.139 [−0.145, +0.442] | 0.000 [0.000, 0.000] |
| Frozen − starting | Ante 4+ | −0.151 [−0.491, +0.206] | −0.050 [−0.130, 0.000] |

Intervals use 10,000 paired whole-game bootstrap draws with
`numpy.default_rng(1729)`. Each draw resamples the 128 games together for all
policies and recomputes the decision-weighted ratio. Intervals are unadjusted
and the panel is already exposed. Descriptive equal-game means are also retained;
games without target states are excluded only from those descriptive means.
An observed [0, 0] interval reflects identical per-game counts in this panel,
not proof of no population effect. Separate stratum intervals do not test the
difference between early and late effects.

The trainable-minus-frozen reduction in at-most-half-score choices is 103 states,
or −1.685 points [−2.275, −1.095] over all 6,113 teacher-play states. The complete
analysis also reports agreement, lower scores, missed/extra clears, new/averted
losses, other-family choices and direct score ordering for every fixed contrast
and stratum, including null and negative results.

## Verification and interpretation

The two-game smoke actually exited zero before panel admission. The full worker
then exited zero after 251.2 seconds, within its 1,800-second limit. All 248
archived source files, 22 input cache arrays, teacher shards, saved predictions,
metrics and checkpoints retain their pinned hashes. An independent analyzer
recomputed the saved aggregates, coverage and prediction identities and passed
103,214 checks. That audit verifies saved arithmetic and provenance; it is not
a second independent execution of the simulator or neural network. Both worker
and analyzer are terminal. An earlier CLI argument typo failed before the smoke
started and is preserved in the smoke receipt.

This gives a concrete behavioral counterpart to the
[Stage 2 imitation improvement](pretraining_transfer_results_20260923.md): some
fresh-state gains reflect more productive immediate hands. Most of the net
clear-count improvement occurs later, while dangerous individual choices and
early errors persist. These results cannot assign a causal share of the previous
autonomous win gain, quantify errors on student-visited states, or prove that
early strategy improved. There is one training replica and no fresh confirmation
in this diagnostic.

The subsequent [inspection of all seven residual fatal choices](residual_hand_errors_20260924.md)
reproduces the saved neural predictions and identifies concrete scoring and
ranking mistakes, including a visible Even Steven interaction. Existing training
coverage and fitting of these interactions are the next priority, followed by
distinguishing teacher-state errors from autonomous trajectory drift. The failed
Stage 3 confirmation remains closed; its fresh-outcome gate was not opened. The
broader replicated win and early-decision requirements remain unchanged and unmet.
