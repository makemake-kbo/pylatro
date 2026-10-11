# What the Card Sharp training examples actually teach

**The training corpus contains the Pair-versus-triple distinction, but only one of its three last-hand examples separates survival from immediate defeat. Every saved model selects the surviving action there.** The complete controlled replay now measures all 145 previously fixed training states, all 156 structural alternatives and all five saved policies. This closes the missing training-consequence check; it does not establish fresh generalization.

Of the 156 alternatives, all are legal, 155 score below the teacher’s Pair, and one scores above it. The 155 lower scores cover 144 states in 88 games. There are 52 missed teacher clears across 49 states in 36 games; twelve alternatives clear along with the teacher, and none produces an extra clear. Only one alternative creates an immediate loss. These are overlapping descriptive counts, not independent trials or full-game regret.

| Structural alternatives | Count / 156 |
|---|---:|
| Legal | 156 |
| Lower immediate score | 155 |
| Equal immediate score | 0 |
| Higher immediate score | 1 |
| Miss a teacher clear | 52 |
| Both actions clear | 12 |
| New immediate loss | 1 |

All three last-hand comparisons are retained below. Every model—base, both ordinary continuations and both preservation continuations—matches the teacher in every row.

| Training seed / step | Points required | Teacher Pair | Alternative triple | Immediate result |
|---|---:|---:|---:|---|
| 4000471 / 116 | 20,000 | 33,948 | 9,386 | Pair clears; triple loses |
| 4000769 / 223 | 67,341 | 39,123 | 8,664 | Both lose |
| 4000995 / 235 | 128,664 | 37,338 | 7,964 | Both lose |

The decisive training example, 4000471 / 116, uses a level-5 Pair against a level-1 triple with Card Sharp and a prior Pair in the encoded history. It chooses action 145 over 139, the same action indices as the newly diagnosed held-out failure. This is shared structure, not identity of the full states. The teacher Pair scores 33,948 against a 20,000-point requirement; the triple scores 9,386 and loses.

The earlier consequence-weighting replay assigned this example severity zero and raw weight 1.0 because the base’s selected action already matched the teacher. Thus its weighting rule did not emphasize this unchosen fatal alternative. Ordinary teacher NLL still penalizes competing actions implicitly; the gap is the absence of an explicit consequence-sensitive comparison, not absence of any learning signal. This observation does not prove that a different loss will transfer.

The lone higher-scoring triple is also retained: at 4000723 / 157 it scores 8,140 versus the teacher’s 7,800, with 20,000 points required and five hands left. Neither clears. A rule that labels every structural triple as inferior would therefore be wrong even for immediate score. The eventual game value of either action is unmeasured.

All policy choices were replayed from the identical pre-state. The teacher clears 61 of the 145 states. Scores below/equal/above the teacher exclude non-play choices; non-play transitions are measured with no fabricated hand score.

| Policy | Exact matches | Below / equal / above teacher score | Non-play | Clears | Missed clears | New losses |
|---|---:|---:|---:|---:|---:|---:|
| Base | 71 | 22 / 111 / 9 | 3 | 56 | 5 | 0 |
| Ordinary 0 | 98 | 12 / 122 / 8 | 3 | 57 | 4 | 0 |
| Ordinary 1 | 99 | 14 / 124 / 5 | 2 | 58 | 3 | 0 |
| Preservation 0 | 85 | 18 / 114 / 10 | 3 | 57 | 4 | 0 |
| Preservation 1 | 86 | 16 / 116 / 10 | 3 | 57 | 4 | 0 |

Exact-action disagreement is not identical to an immediate scoring error. The base differs from the teacher 74 times, but 40 of those choices produce the same score and nine produce a higher score. The corresponding equal-score, nonmatching counts are 24 / 25 for ordinary continuation and 29 / 30 for preservation. Equal scores do not establish equivalent remaining cards, Joker effects or future value. Conversely, all five policies still miss some available training clears despite their exact-label fitting gains.

Together with the [held-out diagnosis](preservation_regression_trace_20260926.md), this supports testing how supervision ranks consequential alternatives across contexts. It rules out a complete absence of the broad comparison from training and separates the three last-hand examples into one survival-versus-loss comparison plus two losing states. It does not isolate representation, optimization or data coverage as the sole cause. A prospective comparison objective should use measured outcomes, retain ordinary fitting as a matched control, and distinguish training success from untouched-game and early-decision transfer. No new loss has been implemented or tested in this diagnostic. World-model work remains deferred.

Both replay phases reproduce all 18,857 factual decisions in the selected 88 games and all 4,003 historical teacher/base play controls. The separate auditor verifies the legal actions, original observations and terminal metadata, independently recounts consequences, and requires byte-identical control transitions between the control and measurement phases. All four child phases have actual successful direct-parent exits; the detached driver’s own exit code is not claimed. All five recorded processes were subsequently absent. This verifies recorded outcomes and bookkeeping, not an independent second game engine.

Artifacts: [frozen protocol](../../eval/card_sharp_training_cost_20260926/protocol_v1.json), [raw alternatives](../../eval/card_sharp_training_cost_20260926/panel_v1/comparisons.jsonl), [independent verification](../../eval/card_sharp_training_cost_20260926/panel_verification_v1.json), [actual phase exits](../../eval/card_sharp_training_cost_20260926/execution_v1/complete.json), [machine-readable report](data/card_sharp_training_cost_20260926.json).
