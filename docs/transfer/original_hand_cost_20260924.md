# Consequences of original-model hand errors

The original model's disagreement with the demonstrations is not mostly a matter
of equal-scoring hands. On **9,336 recorded plays from 200 original training
games**, its saved prediction chooses a different played hand 4,445 times.
**3,559 of those choices (80.1%) score lower**, and 1,595 score at most half as
much. Substituting the saved prediction also causes **23 immediate game losses
across 21 games** where the recorded action survives.

This establishes consequential errors on original training states, before any
student-generated trajectory drift is needed. It does not isolate their cause,
measure the entire autonomous win-rate gap, or demonstrate a newly learned
improvement. World-model work remains deferred.

The [complete result](../../eval/original_hand_cost_20260924/panel_v3/result.json)
and [compact summary](data/original_hand_cost_20260924.json) retain the counts.
The [independent audit](../../eval/original_hand_cost_20260924/arithmetic_audit_v1.json)
passed 547 arithmetic, coverage and provenance checks.

## What was measured

The fixed panel is the original diagnostic's 200 uniformly sampled training
games, including 147 recorded wins. All **44,348 original decisions** were
replayed under generation-pinned sources. All 14 observation arrays, saved
action identities, legality and terminal outcomes match. At each recorded play,
the diagnostic preserves the full runner state, completes the factual game,
then compares the recorded action with the saved epoch-10 prediction from
identical copies of that state.

Every repeated recorded action reproduces its factual score, phase and full
post-action state. Every identical student/teacher action has an identical
outcome. The runner applies the normal Joker-ordering policy and survival
effects, including Mr. Bones; immediate losses are actual engine `GAME_OVER`
transitions rather than estimates from score thresholds. Each comparison is a
single action, without an autonomous continuation. These are saved mode-action
predictions, not newly sampled student actions.

| Among the 9,336 recorded plays | Decisions |
| --- | ---: |
| Student chooses the recorded action | 4,693 |
| Student chooses a different played hand | 4,445 |
| Student chooses another action family | 198 |

| Among the 4,445 different played hands | Decisions | Share |
| --- | ---: | ---: |
| Student scores less | 3,559 | 80.1% |
| Equal immediate score | 531 | 11.9% |
| Student scores more | 355 | 8.0% |
| Student scores at most half as much | 1,595 | 35.9% |

The last row is a subset of lower scores. Lower score is not automatically
long-term regret: **885 lower-scoring choices still clear the blind in both
branches**. Equal score likewise need not preserve the same cards, economy or
later opportunities. Higher student scores do not establish superior strategy.

Across all individual branches, the teacher action clears the current blind
4,431 times and the student action clears it 3,671 times. There are **918
teacher-clear/student-no-clear pairs**, and **158 pairs in the reverse direction**.
Some alternative actions are discards or consumable uses, so the former count
must not be called 918 lost games.

The stricter terminal comparison finds 23 new immediate losses and no averted
immediate losses. Sixteen new losses occur on fifteen of the 147 recorded winning
paths. Eleven occur at Ante 2 or 3, across ten games. Multiple counterfactual
branches from one teacher game are dependent. These counts do not estimate the
probability that an autonomous student visits or loses on those states.

## Relation to the original candidate restriction

An [additional exploratory analysis](../../eval/original_hand_cost_20260924/candidate_strata_result_v1.json)
decodes valid play candidates directly from saved token rows, attention masks
and action masks. It ignores the older diagnostic's defective coverage field.
All **9,336 coverage flags agree with the separate earlier corrected Torch
audit**, and the [verification](../../eval/original_hand_cost_20260924/candidate_strata_verification_v1.json)
reproduces all grouped counts.

| Recorded action | Play states | Exact matches | Student plays a lower-scoring hand | Teacher clears, student does not | New immediate losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inside valid candidates | 3,539 | 2,938 | 372 | 113 | 6 |
| Outside valid candidates | 5,797 | 1,755 | 3,187 | 805 | 17 |

Thus 89.5% of lower-scoring plays and 87.7% of missed teacher blind clears occur
where the teacher action is outside the candidate set. In 3,765 states the
teacher action is outside but the student's selected action is inside; these
include 3,077 lower scores, 717 missed clears and seventeen immediate losses.

The earlier [learning-bottleneck study](learning_bottleneck.md) established that
the original fixed 50/50 mixture caps an outside-candidate action's conditional
probability at 0.5 when candidates exist. Every audited play has valid candidates,
and no saved outside-candidate teacher probability exceeds that cap. The new
measurement connects that known restriction to the states with costly errors.
It is an association: candidate coverage also varies with state difficulty and
teacher choice. It does not isolate the causal effect of the mixture, establish
that all candidate choices are worse, or explain the whole transfer failure.

For a concrete example, seed **2000051, step 182, Ante 6** has one hand remaining
and needs 20,000 points. The recorded Pair scores **41,064** and clears the blind;
the student's Two Pair scores **18,304** and immediately ends the game. The
recorded action is outside the ten valid candidates, while the student choice
is inside. Its saved probability for the recorded action is 0.08468. This is the
first new immediate loss in panel order, shown as an illustration rather than a
separately selected statistical test.

## Provenance and limits

The [prediction checks](../../eval/original_hand_cost_20260924/prediction_provenance_v1.json)
verify the original checkpoint hash, complete row identities and agreement with
the original aggregate accuracy, NLL and teacher probabilities. The historical
prediction-producing helper versions could not be recovered from their pinned
hashes. This audit evaluates the concrete saved actions; it does not claim an
independent neural-network forward pass. Every imported simulator source matches
the generation manifest. Neither the original corpus nor its labels were changed.

Two fingerprint-related attempts stopped before emitting counterfactual results.
The first used pickle round trips; the second found that pickle-byte equality
also rejects some direct copies. The final fingerprint compares stored values
and mutable references, normalizes object attribute order, and preserves mapping
and list order. Thirty replayed-state copy checks and seven mutation checks
passed, including hidden RNG and broken-alias detection. Each branch also restores
the factual process-global card-identity counter and restores the live counter
afterward. The failed attempts and successful preflights remain preserved.

The full worker exited zero after 577.1 seconds, within its 1,800-second budget.
The independent audit checks saved arithmetic, coverage and hashes; replay and
engine consequences are worker evidence, not a second independent engine run.
The candidate analysis finished separately and passed its coverage cross-check.
All diagnostic processes are terminal.

The subsequent [corrected-policy comparison](corrected_hand_cost_20260924.md)
uses all three saved Stage 2 policies on the same exposed held-out states. The
trainable encoder makes fewer low-scoring choices and clears 47 more blinds than
the frozen encoder, with 46 of that net gain at Ante 4 or later. Both still
introduce seven immediate losses where the teacher survives. Its different panel
cannot serve as a matched before/after control for this original-model audit.
Residual fitting and autonomous-path errors remain the priority. The failed
Stage 3 attempt remains closed, world-model work remains deferred, and the
long-horizon generalization goal remains active and unproven.
