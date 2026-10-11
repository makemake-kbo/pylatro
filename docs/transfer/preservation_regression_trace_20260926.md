# A Card Sharp ranking failure: score, history and corpus coverage

**The new fatal regression has an exact scoring explanation, and the broad comparison exists in both training corpora. Last-hand coverage is sparse, and learning it in three training examples does not transfer reliably to the two exposed held-out examples.** No new model was trained and no fresh win improvement is claimed. The original long-horizon goal remains unmet; world models stay deferred.

This follows the [full-panel consequence replay](preservation_hand_cost_20260926.md). At seed 850000034 / step 104, preservation replica 1 replaces the teacher’s 36,011-point Pair with a 9,291-point Three of a Kind. It has one hand left and needs 13,272 points. The base, both ordinary continuations and preservation replica 0 all select the clearing Pair. The new inspection includes all nine fatal states in the completed panel, not just this case.

The teacher selects two Aces and a non-scoring 4; the losing choice replaces the 4 with a third Ace. Pair is level 5 while Three of a Kind is level 1. Banner contributes 60 chips, Square Joker contributes 80, Half Joker adds 20 multiplier, and Ramen multiplies by 1.99. Card Sharp supplies a further ×3 only for the Pair. The runner automatically reorders Jokers before scoring; the table follows the reproduced order Banner, Square, Half, Ramen, Card Sharp.

| Component | Clearing Pair | Losing Three of a Kind |
|---|---:|---:|
| Hand level | 5 | 1 |
| Base chips | 70 | 30 |
| Scoring-card chips | 22 | 33 |
| Banner + Square chips | 60 + 80 | 60 + 80 |
| Total chips | 232 | 203 |
| Base multiplier | 6 | 3 |
| Half Joker | +20 | +20 |
| Ramen | ×1.99 | ×1.99 |
| Card Sharp | ×3 | ×1 |
| Final multiplier | 155.22 | 45.77 |
| Floored score | **36,011** | **9,291** |

Independent Decimal arithmetic reproduces both engine scores. Removing only the Pair’s Card Sharp factor gives 12,003 points in the arithmetic, below the target; this is not a separately replayed branch. The repeat interaction is therefore material to the observed clearing margin.

The archived engine resets `played_this_round` after the Boss blind, so the counter spans an ante here. The current Big Blind history contains four High Card plays; the immediately preceding Small Blind history contains the Pair. These two saved history frames exactly reconstruct all twelve engine hand-type counters, with no omitted events. Pair’s count is one and Three of a Kind’s is zero. The hand-level tokens themselves encode level, chips, multiplier and total plays, but do not directly encode this repeat counter. The existing history inputs contain the relevant prior play and are passed into the model. This case supplies no evidence that a longer history window is required. It does not establish that every relevant fact is represented in every state.

The fixed coverage comparison requires active Card Sharp, visible non-debuffed non-Stone cards, no forced selection or active Splash, a three-card teacher Pair with one filler, and an unselected third card of the paired rank. Replacing the filler produces a structural triple. These structural alternatives are not automatically legal, lower-scoring or harmful. Refinements describe the actual encoded hand levels and history; a Pair anywhere in the three-blind history is not assumed to activate the current counter in every row.

The original corpus has 5,000 games, 1,126,644 decisions and 236,278 plays. The later training set has 1,000 games, 202,141 decisions and 47,149 plays. The exposed held-out set has 128 games, 26,045 decisions and 6,113 plays. Each cell below is states / distinct games. All ten fixed strata are retained.

| Predicate | Original corpus | Later training | Exposed held-out |
|---|---:|---:|---:|
| All plays | 236,278 / 5,000 | 47,149 / 1,000 | 6,113 / 128 |
| Active Card Sharp | 28,205 / 1,476 | 5,610 / 302 | 926 / 45 |
| Visible-card restriction | 22,156 / 1,433 | 4,464 / 295 | 738 / 44 |
| Core Pair-versus-triple comparison | 719 / 433 | 145 / 88 | 19 / 14 |
| Core: Pair level higher | 688 / 419 | 140 / 85 | 17 / 13 |
| Core: Pair in retained/omitted history | 702 / 423 | 145 / 88 | 19 / 14 |
| Core: Pair in previous blind history | 681 / 415 | 144 / 87 | 18 / 14 |
| Core: one hand left | 8 / 8 | 3 / 3 | 2 / 2 |
| Core: higher Pair level and previous-blind Pair | 657 / 406 | 139 / 84 | 16 / 13 |
| Exact five active Joker identities, any order | 24 / 1 | 0 / 0 | 5 / 1 |

The broad comparison appears 719 times in the original corpus and 145 times in the later training set. Most later examples also have a higher Pair level and a Pair in the immediately previous blind’s history. The exact five-Joker inventory is absent from the later training set but occurs in one original training game. Absence of an exact inventory is not evidence that exact copies are necessary for generalization.

Only eight original and three later training examples of the core comparison have one hand remaining. All five saved policies match the teacher in the three later last-hand examples. In the two exposed last-hand examples, all policies fail at 850000027 / 121 (5,382 instead of 14,859 against an 11,000-point target); preservation replica 1 additionally fails at the new 850000034 / 104 case. These are two error-selected held-out cases, not an estimated population failure rate.

The following table includes every saved policy. Matches are exact teacher-action matches; triple choices are the defined structural substitution, not a general error label. No new original-corpus model predictions were made.

| Policy | Training matches / 145 | Training triple choices | Training NLL | Held-out matches / 19 | Held-out triple choices | Held-out NLL |
|---|---:|---:|---:|---:|---:|---:|
| Base | 71 | 9 | 1.507577 | 7 | 4 | 2.128763 |
| Ordinary 0 | 98 | 7 | 0.929140 | 10 | 3 | 2.444453 |
| Ordinary 1 | 99 | 6 | 0.957406 | 8 | 3 | 2.238754 |
| Preservation 0 | 85 | 9 | 1.069171 | 8 | 3 | 2.153239 |
| Preservation 1 | 86 | 8 | 1.055198 | 6 | 5 | 2.062421 |

Training fit improves across all four continuations, while many training choices still differ from the teacher. In this small held-out comparison set, preservation replica 1 lowers NLL relative to the base but loses an exact match and introduces the additional fatal choice. This supports distinguishing likelihood fitting, action ranking and consequences. It does not establish a statistically resolved stratum-wide regression or identify a unique architectural cause.

Joining all nineteen held-out comparison states to the already verified full replay gives these immediate outcomes. The teacher clears eleven. These are descriptive teacher-path interventions; they do not measure full-game value.

| Policy | Blind clears / 19 | New immediate losses | Scores below teacher |
|---|---:|---:|---:|
| Base | 8 | 1 | 5 |
| Ordinary 0 | 9 | 1 | 3 |
| Ordinary 1 | 8 | 1 | 4 |
| Preservation 0 | 9 | 1 | 3 |
| Preservation 1 | 7 | 2 | 5 |

The evidence narrows the next intervention: it must improve transfer of score-sensitive comparisons across contexts and preserve already-correct critical rankings. Broad coverage exists, last-hand coverage is sparse, and further imitation fitting alone has not ensured that transfer. The broad comparison’s unmeasured training alternatives still need consequence checks before being used as negative training targets. Neither longer context nor a world model is justified by this case.

Verification reproduced all 26,045 factual decisions and 23 distinct actions across the nine selected states, and all eight historical state descriptions match. A separate reread checks all 289,540 play rows across the three corpora, 883 structural comparison records and 266,310 saved prediction rows. The source schemas and vocabulary match the original generation. All processes exited; no fitting or autonomous evaluation ran. The first state verifier omitted the engine’s `T` spelling for ten and failed; its versioned correction adds that spelling only, with the failed attempt retained. No replay or measured outcome was changed.

Artifacts: [state verification](../../eval/preservation_regression_trace_20260926/states_verification_v2.json), [score/history arithmetic](../../eval/preservation_regression_trace_20260926/score_explanation_v1.json), [fixed coverage definitions](../../eval/preservation_regression_trace_20260926/coverage_protocol_v1.json), [full coverage results](../../eval/preservation_regression_trace_20260926/coverage_v1/result.json), [independent coverage verification](../../eval/preservation_regression_trace_20260926/coverage_verification_v1.json), [actual verification exit](../../eval/preservation_regression_trace_20260926/coverage_verification_terminal_v1.json), [machine-readable report](data/preservation_regression_trace_20260926.json).
