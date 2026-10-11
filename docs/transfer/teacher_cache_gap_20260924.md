# A stale teacher cache in the pretraining pipeline

A reproduced label discrepancy exposed a concrete bug in the teacher used for
the original pretraining corpus. Its Joker-name cache retained a reference to
`state.joker_keys`, a list that purchases and sales mutate in place. The cache
key therefore changed along with the state while the cached names stayed stale.
The working-tree fix snapshots the key as a tuple. The native heuristic extension
was rebuilt, and 129 targeted tests passed, including three new regressions that
failed before the fix.

The completed original-corpus replay now reproduces all **3,909 decisions from
20 games**. Correcting this cache changes **13 labels across two games**: 0.33%
of this fixed, outcome-balanced panel. All six changes from one played hand to
another increase the exact immediate score. This establishes a label-quality
issue in the original data, with a narrow observed behavioral effect. It does
not establish a population frequency or explain the student's win-rate gap.
The original teacher achieved 74.56% corpus wins with the defective cache;
improving that teacher does not by itself explain why the transformer failed to
match its existing performance. No policy was trained and no new autonomous wins
were evaluated. World-model work remains deferred.

## Completed original-corpus measurement

The [replay result](../../eval/teacher_information_20260924/original_panel_v3/result.json)
uses the same 20 original HF games selected by the earlier historical diagnostic:
ten recorded losses and ten wins. It restores the generation-pinned simulator,
runner, tokenizer and teacher sources. All 14 saved observation arrays, action
legality, historical teacher labels and terminal outcomes match. The historical
live teacher supplies the control. A second teacher differs only in the replaced
cache method and is queried sequentially on copies of those factual states,
including the replacement for nested teacher instances. Recorded actions always
drive the replay; corrected proposals never replace them.

| Measurement | Decisions |
| --- | ---: |
| Historical teacher reproduces the recorded label | 3,909 / 3,909 |
| Cached names differ from actual inventory after querying | 3,755 / 3,909 |
| Relevant special-Joker flags differ | 382 / 3,909 |
| Corrected teacher proposes a different action | 13 / 3,909 |
| Changed play-to-play pairs with a higher immediate score | 6 / 6 |

The cache's consumers use four special-Joker flags: Smeared Joker, Four Fingers,
Shortcut and Pareidolia. Other scoring paths read actual inventory or call the
engine directly. Five of the twenty games hold at least one of these four Jokers;
this panel contains Four Fingers and Shortcut but no Smeared Joker or Pareidolia.
All thirteen changed decisions have relevant flag disagreement. A stale cache
after a query does not establish that the decision used that cache.

A [separate coupled-copy intervention](../../eval/teacher_information_20260924/original_cache_coupled_v1/result.json)
confirms the cache alone causes **all thirteen differences**. It first replays
the two affected original games, matching all 431 input digests, historical labels
and terminal outcomes. At each target it copies the teacher and simulator state
together, preserving the defective cache-key alias. The control reproduces the
saved label; changing only `_joker_names_cache_key` to an invalid key produces
the panel's corrected action. Paired full-state hashes match, and neither query
mutates its simulator state. The six paired play scores also reproduce. This
check completed in 38.5 seconds; its owner exited zero and its
[saved-evidence verification](../../eval/teacher_information_20260924/original_cache_coupled_verification_v1.json)
passed 181 checks. Targets were chosen from observed panel changes, so this is
causal follow-up evidence, not an independent frequency sample.

Twelve changes occur in loss seed 2075092: six discard-subset changes and six
play-subset changes. The remaining change is a discard-to-play proposal at step
20 of win seed 2025274. The six paired play scores are:

| Step in seed 2075092 | Recorded hand score | Corrected hand score |
| --- | ---: | ---: |
| 72 | 1,848 | 3,640 |
| 73 | 4,680 | 4,752 |
| 94 | 3,496 | 6,006 |
| 103 | 3,420 | 6,700 |
| 117 | 2,686 | 5,586 |
| 138 | 1,222 | 10,400 |

The scores are engine evaluations of each proposed hand on the identical state,
including attainable Joker ordering. They do not measure later draw quality,
resource use, terminal regret, or the value of the seven discard-related changes.
These are exposed, outcome-stratified games. The 0.33% rate describes this panel,
not all 5,000 games, and individual decisions are not independent game samples.

The [independent arithmetic and provenance audit](../../eval/teacher_information_20260924/original_panel_arithmetic_audit_v1.json)
passes 548 checks covering complete row identities, per-game/family/outcome
counts, paired-score coverage, pinned inputs and imported source hashes. It
checks saved evidence rather than rerunning the teacher or simulator. The native
owner exited zero after 774 seconds, within its fixed 1,800-second allowance.

Runtime reconstruction recovered 123 of 125 generation-pinned Python files,
including all required replay files; the unavailable historical `model_generate`
and `supervised` versions are not imported. Every imported project source matches
its original generation hash. Native extensions were rebuilt locally. Two earlier
reader preflights crashed in the filtered Arrow dataset scanner; direct Parquet
row-group decoding then passed a 364-decision preflight without simulator changes.
An additional synthetic dispatch test invoked the replacement method but failed
an overstrict expectation that its example hand must change. The launch
orchestration mistakenly continued after that failed assertion and a dependent
receipt write. The [retrospective launch review](../../eval/teacher_information_20260924/original_panel_launch_review_v1.json)
preserves this procedural error and the successful prelaunch replay. This remains
an exploratory diagnostic, with no claim of pristine confirmatory admission.

## Reproduced causal example

The [paired audit](../../eval/teacher_information_20260924/joker_cache_causal_audit_v1.json)
replayed 54 actions from already-exposed Stage 2 teacher game 850000002. All 55
observations through the examined decision matched the saved trajectory. At
step 54, Ante 4, the actual inventory contained Shortcut, Golden Joker, Abstract
Joker, Smiley Face and Gluttonous Joker. The teacher's cached name set was empty.

Two copies preserved the identical full simulator state and teacher state,
including the defective shared list reference. The treatment invalidated only
`_joker_names_cache_key` before the query; every other teacher field was retained.
Neither query mutated its simulator state.

| Query | Action | Selected hand slots, zero-based | Exact immediate score |
| --- | ---: | --- | ---: |
| Existing cached teacher | 4390 | 1, 4, 5, 6, 7 | 315 |
| Only name-cache key invalidated | 3224 | 0, 3, 5, 6, 7 | 1,342 |

The control exactly reproduces the recorded label. This is a difference in
immediate score, not measured terminal regret or evidence of a win-rate gain.
Other teacher code still reads the actual Jokers and their effects directly;
the bug does not mean that every part of the teacher ignores its inventory.

The saved Stage 2 trainable-encoder endpoint predicts action 4390 on this state
and assigns the recorded label probability 0.61096. It is counted as correct by
the imitation metric. The frozen endpoint predicts a different action and assigns
the label probability 0.20830. This example shows why increased agreement with
the recorded teacher is not automatically improved decision quality. We have
not evaluated the frozen endpoint's alternative action or the terminal outcomes
of these choices.

## Relation to the original corpus

The archived heuristic source has SHA256
`f4c11f67a5294f308b99e7ccd0876f103eb81a44a9c44ef98cc7bf5cb30da371`, exactly matching
the source recorded in the original 5,000-game generation manifest. The
[provenance receipt](../../eval/teacher_information_20260924/cache_finding_provenance_v1.json)
pins that match and both replay routes. The demonstrated state above comes from
the later Stage 2 dataset; it must not be represented as an original-corpus row
or a prevalence estimate for that corpus.

The discrepancy also explains why copying states can change teacher behavior:
repeated queries on the live state preserve the aliased list, while queries on
independent deep copies can refresh the cached names after inventory changes.
The [live-state replay](../../eval/teacher_information_20260924/replay_live_a.json)
reproduces label 4390; the [copy-state replay](../../eval/teacher_information_20260924/replay_copy_a.json)
chooses 3224 despite matching all saved observations. A source hash alone does
not establish equivalence between those calling conventions.

## Input and RNG probe that exposed the issue

The first preflight rejected draw-order shuffling as a hidden-information
intervention: it changed 66 token values at the examined state. The student
receives draw-pile cards in their current sequence, subject to deck-token capacity,
and its embedding layer adds absolute positions. Both the tokenizer and embedding
files exactly match those recorded for the original pretraining. The
[visibility artifact](../../eval/teacher_information_20260924/smoke_v2/draw_order_visibility.json)
records the changed input; this is information availability, not evidence that
the trained model uses the order effectively.

The corrected RNG-only smoke replayed all 221 decisions of one exposed game and
passed six observation-preserving interventions. Its separate arithmetic audit
passed. The planned main panel was 178 states from 16 exposed games with four
RNG interventions each. It stopped after 26 state records because the copied-state
teacher failed to reproduce the third game's label. That failed panel is preserved
and supplies no completed frequency estimate. It led to the cache diagnosis;
it must not be promoted to a completed information-sufficiency study.

## Fix and remaining measurement

The [source fix](../../src/pylatro_agent/heuristic.py) changes only the Joker-name
cache key to an immutable snapshot. The
[new regressions](../../tests/test_heuristic_joker_cache.py) cover first purchase,
sale and replacement with unchanged inventory length. The existing reproducibility,
shop-search, discard-search and scoring-copy tests also pass: 129 tests total.
Ruff passes for the edited source and tests. The
[verification receipt](../../eval/teacher_information_20260924/cache_fix_verification_v1.json)
records the actual build and test exits, source/native hashes, and example policy
predictions. All 248 archived runtime files were checked unchanged. Existing
datasets, checkpoints, study protocols and exposed outcomes remain preserved.

The completed panel establishes original-data label effects while preserving
historical labels. A new training and gameplay comparison would be required to
attribute student win gains to correcting this defect; this result alone does
not justify treating corrected-label training as the main solution. Residual
fitting error and the consequences of student mistakes remain the central
transfer questions. The broader long-horizon goal remains unproven.
