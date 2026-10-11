# Residual hand-ranking failures after encoder training

We can now explain a specific lethal choice using visible, present-state
scoring inputs. With one hand left and 800 points needed, all three corrected
policies choose a pair of 7s scoring **795**, while the teacher's pair of 4s
scores **1,989**. The score gap is exactly accounted for by the visible
**Even Steven** Joker: each scoring even card adds four multiplier.

The learned subset correction also reverses a teacher-favoring pairwise
reference ranking in **two of the seven inspected fatal choices**. These are
concrete remaining scoring and ranking errors after the original output cap
was removed. They do not establish a general architectural incapacity, explain
the entire transfer gap, or demonstrate a new learned win improvement.
World-model work remains deferred.

The [complete analysis](../../eval/residual_hand_errors_20260924/analysis_v1.json)
and [compact report data](data/residual_hand_errors_20260924.json) preserve all
seven cases and all three policies, including cases where the correction helps.

## Which cases were inspected

The previous [paired cost comparison](corrected_hand_cost_20260924.md) found seven
states where the trainable encoder's saved mode immediately loses while the
teacher action survives. This study includes **all seven**. They are selected
because of those exposed errors, not sampled to estimate population frequencies.
All three policies already have corrected action heads. The starting and frozen
policies remain useful comparisons, not additional selection criteria.

The worker replays all 128 original Stage 2 teacher games in order. All 26,045
cached input arrays, actions and terminal outcomes match, and all seven full
pre-state fingerprints match the preceding diagnostic. Every teacher and policy
action inspected reproduces its complete saved post-state, not only its score.
Hand cards, Joker state, hand levels, candidate proposals and corresponding
input tokens are saved alongside these results.

The model worker uses each checkpoint's original 32-row MPS batches containing
the selected states. Across three policies it reproduces **672 saved predictions**
and their teacher NLLs exactly, with maximum NLL error zero. It runs inference
only. The probability decomposition therefore describes the actual saved
policies, rather than a separately trained or reconstructed approximation.

## A scoring error with the relevant inputs present

The first case in seed/step order is **850000015 / 28**, at Ante 2 against
The Needle. Every card is visible, ordinary and not debuffed. The hand contains
pairs of 7s and 4s. Pair is level 4 with 55 base chips and five base multiplier;
these values are present in the hand-level tokens. Banner adds 90 chips because
three discards remain. Even Steven's identity and numeric bonus are present in
the Joker tokens. The card ranks and hand slots are also present.

| Scoring calculation | Policy's pair of 7s | Teacher's pair of 4s |
| --- | ---: | ---: |
| Base chips | 55 | 55 |
| Scoring-card chips | 14 | 8 |
| Banner chips | 90 | 90 |
| Total chips | 159 | 153 |
| Base multiplier | 5 | 5 |
| Even Steven multiplier | 0 | 8 |
| Total multiplier | 5 | 13 |
| Final score | **795** | **1,989** |

The arithmetic matches the engine's chips, multiplier and final score exactly.
The inspected implementation applies
[Even Steven per scoring card](../../eval/pretraining_transfer_20260923/candidate/src/pylatro/scoring.py#L616)
and [Banner from remaining discards](../../eval/pretraining_transfer_20260923/candidate/src/pylatro/scoring.py#L697).
This explains the observed score difference without requiring a future rollout.
Space Joker is also present, so this calculation describes the reproduced
outcomes; it is not a claim that every possible RNG realization is identical.

The trainable policy assigns the teacher action 20.3% conditional play
probability and the losing action 72.3%. The teacher is its second-ranked play.
Compared with the frozen encoder's 2.9% versus 92.4%, training moved probability
in the right direction but did not change the selected action. The learned
correction still favors the losing action relative to the teacher.

This one case does not prove that every required input is available in every
state, or determine whether representation, data coverage or optimization is
responsible. It does show that the remaining transfer failure includes a
present-state Joker/card scoring interaction whose relevant inputs are encoded.

## All seven fatal choices

The table reports the trainable policy. All teacher actions clear the blind;
all selected policy actions enter `GAME_OVER` with one hand remaining.

| Seed / step | Points needed | Teacher hand and score | Policy hand and score | Teacher's play rank |
| --- | ---: | --- | --- | ---: |
| 850000015 / 28 | 800 | Pair: 1,989 | Pair: 795 | 2 |
| 850000027 / 121 | 11,000 | Pair: 14,859 | Three of a Kind: 5,382 | 3 |
| 850000042 / 113 | 11,000 | Pair: 14,364 | Three of a Kind: 8,510 | 2 |
| 850000076 / 120 | 3,217 | Two Pair: 7,893 | Pair: 2,693 | 7 |
| 850000077 / 77 | 5,000 | Full House: 24,308 | Straight: 1,280 | 4 |
| 850000093 / 57 | 887 | Full House: 2,760 | Three of a Kind: 672 | 7 |
| 850000096 / 130 | 20,000 | Pair: 102,316 | High Card: 9,838 | 2 |

The final score comparisons use the complete game execution, including Joker
ordering, held-card effects and randomness. Only the first case receives the
explicit component-by-component arithmetic explanation above; the table does
not assign an unverified single-Joker cause to every other failure.

## Where the learned ranking goes wrong

The corrected head normalizes **reference log probability + learned subset
energy** over legal plays. The reference retains the original candidate/AR
mixture, but the energy correction can reweight every legal hand. Therefore
the original outside-candidate 0.5 ceiling is not a hard limit on the final
corrected policy. The teacher actions below are legal and have nonzero support.
The reference here uses each endpoint's current representation; it is not the
original epoch-10 network's unchanged distribution.

Each number below is **teacher minus selected action**, in log-probability or
energy units. A positive number favors the teacher within this pair. The first
two columns sum to the final difference, up to floating-point rounding.

| Seed / step | Reference difference | Learned correction difference | Final difference |
| --- | ---: | ---: | ---: |
| 850000015 / 28 | −0.342 | −0.927 | −1.269 |
| 850000027 / 121 | −2.459 | +0.058 | −2.401 |
| 850000042 / 113 | +0.411 | −0.433 | −0.022 |
| 850000076 / 120 | −3.001 | +1.354 | −1.647 |
| 850000077 / 77 | +1.444 | −2.666 | −1.222 |
| 850000093 / 57 | −0.070 | −1.346 | −1.415 |
| 850000096 / 130 | −0.626 | −0.445 | −1.071 |

In cases 850000042/113 and 850000077/77, the reference favors the teacher over
the ultimately selected action and the learned correction reverses that ordering.
Both the starting and frozen policies choose the teacher action in those states.
The first reversal is near a tie; the second is substantial. This is a measured
local regression in the trainable endpoint, alongside its previously established
aggregate improvement. The decomposition does not prove that removing the
correction would select the teacher over *every* other legal action or improve
autonomous wins.

In cases 850000027/121 and 850000076/120, the correction helps the teacher but
does not overcome the reference preference. Overall it favors the selected
losing action over the teacher in five of the seven selected cases. These
conditional counts must not be presented as the correction's overall error rate.

Five teacher actions are outside the structural candidate list; four selected
actions are inside it. Candidate coverage is therefore neither a complete
explanation nor a reliable fix by itself. In case 850000077/77, the teacher is
already the first candidate and the selected losing Straight is candidate five.
In case 850000093/57, the teacher is candidate two and the losing choice is
outside the list.

The candidate token's `estimated_score` is a
[structural heuristic](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/hand_candidates.py#L314):
it includes large hand-type and level bonuses and coarse Joker terms. It is not
the engine's predicted score. The first case's selected Pair has heuristic value
257,350 despite scoring 795. That scale is part of the existing observation
contract. This inspection does not show that the proxy causes the model's error
and does not change its semantics.

## Verification and next question

An independent stdlib analyzer verifies source and input hashes, saved outcome
identity, case coverage, ranking arithmetic and the first case's score/token
arithmetic. **1,937 checks pass.** The state worker, inference worker and analyzer
all actually exited zero; their terminal receipts are preserved. The first
inference attempt rejected an incorrectly assumed batch size of 64 before any
model import or forward. The versioned amendment uses the pinned original size
of 32, with cases, checkpoints, metrics and budgets unchanged.

The subsequent [coverage and fitting audit](even_steven_coverage_20260924.md)
checks these interactions in both the original 5,000-game corpus and the later
1,000-game training corpus. It finds few matching last-hand examples and confirms
lower-scoring choices on two repeatedly presented training states. Limited
coverage and incomplete fitting therefore coexist; their causal contributions
remain unresolved. A small-case diagnosis is not evidence that rare combinations
explain the whole corpus gap, and memorizing these seven held-out errors would
not demonstrate generalization.

The broader requirement still needs replicated fresh learned win improvements
and a controlled benefit from consequential early decisions. No new win estimate
or confirmation is supplied here. The failed Stage 3 attempt and its unopened
fresh-outcome gate remain untouched.
