# Return to the pretraining transfer gap

**Current status, September 28: action-support repair and earlier stopping help the new scorer, but retention still fails.** The [controlled loss experiment](consequence_full_support_results_20260928.md) recovers 363 clears versus the original pooled scorer. Its [saved-duration follow-up](consequence_full_support_duration_results_20260928.md) selects epoch 10 without another fit and recovers 45 more. It still clears 2,573 states versus base 2,699, with 952 early clears versus 1,066 and nine new immediate losses versus seven. All four prediction gates pass; all four retention gates fail. No integration, autonomous evaluation or confirmation advances.

The [completed residual diagnosis](consequence_full_support_residual_results_20260928.md) locates 123 of the remaining 126 net lost clears in actions outside the original per-state proposals. Prediction errors on those selected alternatives are substantially worse than on the original proposals. The clear head already prefers base on 110 of 219 lost-clear cases, but only one of four newly fatal corrections versus surviving base. Both consequence prediction and preference selection remain problematic. The [completed training-only coverage audit](consequence_training_coverage_results_20260928.md) finds 5,932 of 46,427 eligible training choices also lack consequence labels. A [fixed training-only expansion](consequence_action_coverage_20260928.md) is now collecting 47,125 additional high-ranked action outcomes; no development-tuned gate or new fit has started. These teacher-state findings do not establish autonomous wins, long-horizon generalization, or the cause of the original base's pretraining gap. The full-support fit's final validator failure and separate zero-training recovery remain disclosed.

The [matched architectural readout study](consequence_readout_results_20260928.md) and its [earlier-checkpoint evaluation](consequence_prefix_duration_results_20260928.md) remain failed retention experiments. Their [support diagnosis](consequence_support_results_20260928.md) motivated the controlled loss repair above.

The [completed earlier-stopping study](continuation_duration_results_20260927.md) also failed its advancement screen: its candidate won 42/128 sampled games versus base 39 and ordinary 40, below both +5-point thresholds. Greedy wins were 57 versus 59/57; the early-only hybrid won 39 versus 37, with uncertainty spanning zero. All 384 new episodes, 640 reused controls, 159,597 action records and seven comparisons passed independent checks. That duration candidate remains closed. The subsequent [consequence preparation](consequence_preflight_results_20260927.md) completed label/RNG audits and shared frozen features, leading to the readout results above.

The [completed current-rule data-scale experiment](rule_aligned_data_scale_results_20260927.md) improves imitation but fails the clear-retention screen. One matched-exposure fit over 5,000 games lowers exposed held-out NLL to 0.583929 versus base 0.639106 and ordinary continuation 0.690329. It clears 2,677/6,113 teacher-play states versus base 2,699, including nine fewer early clears. Gains on the 339-state preference cohort (+28) are outweighed by losses where base equals teacher (−45) and other disagreements (−5). All process and independent audit checks pass; no autonomous games or confirmation fit are admitted for this fixed endpoint. The subsequent duration diagnostic below is a separate intervention; the fixed endpoint remains failed. No result establishes the long-horizon goal.

The preceding single exploratory fit combining ranking and preservation also failed the development readiness screen. All 1,024 game episodes and 160,169 action records passed independent verification. Sampled wins were 34/128, versus 39 for base and 40 for ordinary continuation; the early hybrid and ordinary control both won 37/128. All intervals span zero. The [complete report](ranking_preservation_development_20260927.md) retains all seven autonomous contrasts alongside the preference, replay and score diagnostics. No confirmation fit is warranted by this result, and no result yet establishes the long-horizon goal.

The [score diagnosis](ranking_preservation_score_decomposition_20260927.md) narrows a correction-transfer gap. The combined model improves the preferred-action margin on 1,394/1,395 teacher-better training pairs, versus 131/233 exposed held-out pairs. Both internal reference and correction terms transfer much less strongly. On the complete teacher-play replay, targeted gains and losses elsewhere nearly cancel: combined gains 6 net immediate clears over base, with an interval spanning zero. These are teacher-path and selected-pair findings. The completed game study supplies no autonomous or early-decision benefit; it does not isolate a parameter cause for the remaining gap.

Keep the two corpora distinct when interpreting these findings:

| Corpus | Games | Recorded decisions | Teacher wins | Role |
|---|---:|---:|---:|---|
| Original pretraining corpus | 5,000 | 1,126,644 | 3,728 (74.56%) | Original trainable-encoder pretraining, with the old action readouts |
| Later demonstration corpus | 1,000 | 202,141 | 728 (72.8%) | Stage 2 fitting and the subsequent ordinary, weighting, preservation, ranking and combined continuations |

The original [corpus metadata](../../cache/datasets/5k-balatro-games/metadata/summary.json), later [metadata](../../eval/learning_bottleneck_20260913/data/training/summary.json), and frozen [Stage 2 manifest](../../eval/pretraining_transfer_20260923/stage2_manifest_frozen_v1.json) confirm this distinction. The later seeds are 4000000–4000999. Recent models inherit weights originating in the original run, so those 5,000 games were not wholly unseen; however, the recent corrected-policy fits are not a controlled refit of the corrected architecture over all original 5,000 games. Their failures do not establish full-corpus convergence or an architectural limit. Decisive-context coverage, changed representations, and action consequences remain separate questions.

Apply the [current experiment workflow](experiment_workflow.md): one exploratory fit per change, then one independent confirmation fit only after a final candidate appears ready and its recipe is frozen. The combined candidate is not ready. The [complete original-corpus census](original_corpus_feasibility_20260927.md) now verifies all 1,126,644 model predictions. Original agreement is 84.32%, or 83.27% under the exposed evaluation set's action-family mix, versus 85.77% for later training and 82.72% for evaluation under that same mix. Plays, discards and Joker sales remain weak in the original corpus.

The follow-up found a concrete runtime-domain mismatch despite matching observation formats: **97,809 original records (8.68%) are post-claim empty-pack cleanup actions**, absent from both later corpora. Excluding those rows for analysis lowers original agreement to **82.83%**. Another **16,146 claim transitions add the claimed consumable to inventory**, consistent with an original engine branch that allowed banking pack consumables; the current engine requires immediate use. The later generation uses the current pack-engine source. A larger-data intervention therefore needs demonstrations aligned with the evaluation rules; simply reusing the original 5,000 games or dropping cleanup rows would not isolate data quantity. The win-rate effect of these differences is not established, and the current-rule transfer gap remains.

The [controlled data-scale experiment](rule_aligned_data_scale_20260927.md) is now complete. It adds 4,000 current-rule games / 811,414 decisions to the later 1,000-game corpus, yielding 5,000 games / 1,013,555 decisions. All wins and losses were retained; new teacher wins were 2,948/4,000 (73.7%). Its one ordinary-imitation fit matches the existing control's 1,280,000 presentations with unchanged architecture. Full dataset, exposure, checkpoint, prediction and replay checks pass. The [result above](rule_aligned_data_scale_results_20260927.md) supports a coverage benefit for imitation while exposing lost existing clears; it does not isolate diversity from reduced repetition or establish an architectural limit. No autonomous win-rate result was measured. Final success still requires matched-control win-rate gains, greedy guardrails and early-decision attribution. World-model work remains deferred.

## Historical record

The notes below preserve earlier milestones. Statements that a previous study was running describe that milestone; the current status above and each linked final report supersede them.

The [fixed counterfactual-ranking experiment](counterfactual_ranking_study_20260926.md) has started. Two4,000-update continuation replicas and all23 fresh evaluation panels/9,200episodes are frozen. The full worker reproduces both technical pilots, ordinary control provenance and exposure pass, and unused confirmation/early seed panels are reserved. Durable serial supervision records actual exits through fitting, independent fit audit, evaluation and independent result verification. No new win-rate or long-horizon benefit is available yet; all original completion criteria remain required. World models remain deferred.

The [counterfactual-ranking objective and native preflight](counterfactual_ranking_preparation_20260926.md) are implemented and independently verified. The fixed loss adds four times the average measured clear-preference loss to teacher NLL, using all 2,027 comparisons at 2,022 training states, including 582 pairs where the alternative clears and the teacher does not. Same-status pairs remain unranked. Independent mathematical, gradient, checkpoint, optimizer and exposure checks pass; the paired twenty-update ordinary control reproduces its prior trajectory. No scientific fit or fresh-game benefit is claimed. The next experiment retains the two-replica fresh-win and early-attribution bar. World models remain deferred.

The [completed Card Sharp training-consequence replay](card_sharp_training_cost_20260926.md) verifies all 145 fixed training states and 156 legal triple alternatives. Of those alternatives, 155 score below the teacher, 52 miss a teacher clear across 49 states, and one creates an immediate loss. Only one of the three last-hand training comparisons separates survival from defeat; all five models fit it, while the other two states lose with either action. The earlier weighting rule left that decisive example at raw weight 1 because the base already chose correctly. Forty of the base’s 74 exact-action disagreements nevertheless match the teacher’s immediate score. This sharpens the distinction between label fitting, consequential ranking and transfer without establishing a new win gain. World models remain deferred.

The [completed Card Sharp regression diagnosis](preservation_regression_trace_20260926.md) reconstructs the 36,011-versus-9,291 scores exactly and finds the needed prior Pair already present in the model’s saved history. The structural Pair-versus-triple comparison occurs 719 times in the original corpus and 145 times in later training, but only eight / three are last-hand examples. All five saved models fit the three later last-hand examples; preservation replica 1 nevertheless fails on both exposed last-hand comparisons. Full coverage, saved fitting metrics and all nine fatal states are verified. This separates available context, broad training coverage, sparse decisive contexts and unreliable transfer; it does not prove a unique architectural cause or new long-horizon benefit. World models remain deferred.

The [completed full-panel preservation replay](preservation_hand_cost_20260926.md) independently verifies all 6,113 teacher-play states and all eighteen fixed comparisons. Preservation improves immediate clears over ordinary continuation by +0.417 points [0.168, 0.659], mostly at Ante 4+, but its +0.057-point difference from the base remains unresolved. New immediate losses are four / seven across replicas, versus five / five for ordinary continuation. Both preservation replicas retain the diagnosed 6,313-point fatal choice; replica 1 also introduces a 9,291-point loss where every other policy clears with 36,011. In that new case, teacher-action probability improves over the base while the selected action becomes fatal. Thus reduced average forgetting does not reliably preserve critical action rankings. Fresh-win and early-attribution criteria remain failed; world models remain deferred.

The [completed preservation experiment](policy_preservation_results_20260926.md) passed independent verification of all 9,200 game episodes and 1,396,901 action records, but **four of five scientific criteria failed**. Sampled preservation wins were 27.0625%, versus 26.0625% for ordinary continuation and 27.25% for the base. Preservation − ordinary was +1.0 points [−1.6266, +3.6875], with opposite replica directions; the early hybrid was −1.25 points [−3.625, +1.125], negative in both replicas. The [teacher-state diagnostic](policy_preservation_mechanism_20260926.md) still shows less forgetting, but this objective did not establish a fresh-win or early-decision benefit. The original pretraining gap remains only partly explained; the long-horizon goal is unproven and world models remain deferred.

The [policy-preservation objective and native preflight](policy_preservation_preparation_20260925.md) are now implemented and independently verified. The fixed proposal adds 1.0 × full forward KL to the starting policy while keeping the actor and encoder trainable. Exhaustive legal-action values and gradients pass; the paired twenty-update preflight reproduces ordinary fitting, matches row exposure, and preserves the critic and reference. A byte-equality check was replaced with a fixed numerical tolerance only after identical-original repeats demonstrated comparable native variation; failed attempts are retained. At that preparation milestone, no scientific fit or fresh-game evaluation had started and no benefit was established. The subsequent execution above now uses the frozen matched manifest and fresh seed reservation. World models remain deferred; the long-horizon goal remains unproven.

The [completed full-panel component diagnostic](component_panel_20260925.md) extends the encoder finding to all 26,045 held-out teacher decisions. Encoder rollback improves overall NLL by 0.034034 for ordinary continuations and 0.039667 for weighted continuations, with paired intervals excluding zero; early and late likelihood both improve. Ordinary exact-agreement recovery remains unresolved, while weighted agreement improves by 0.420426 points [0.219281, 0.624166]. Rollback also removes many newly learned matches, and every tested rollback remains below the base overall. All 339,609 new predictions, 104,180 reused endpoint rows, 192 transition groups and 76 fixed intervals pass independent verification. This supports a retention problem under further fitting, not a global encoder-freeze remedy. The preservation objective and technical preflight above now implement that proposed intervention; the completed fresh-outcome result above failed four of five scientific criteria. World models remain deferred; long-horizon generalization remains unproven.

The [completed component intervention](correction_component_attribution_20260925.md) finds that changed encoder features are sufficient to reproduce the shared fatal choice with the old correction head. Restoring the base encoder fixes that choice in all four continuations; restoring only the head does not. Head updates and encoder–head interactions amplify the regression. The other seven selected cases show mixed effects: the same encoder rollback brings back another fatal choice in all four continuations. All five endpoint controls reproduce exactly and independent tensor, prediction and probability checks pass. This localizes a retention failure under continued fitting without justifying a global encoder freeze. The subsequent full-panel comparison above now quantifies those losses and gains. World models remain deferred; long-horizon generalization remains unproven.

The [completed regression trace](continuation_regression_trace_20260925.md) localizes the shared fatal choice to the learned subset correction’s output contribution. The reference policy still prefers the 13,296-point Pair at every inspected checkpoint; the correction overturns it by the first saved update, 1,000, in all four continuations. The broader structural choice appears in 142 current training states, and its training matches improve from 127 to 135–137, while exposed held-out matches fall from 13/16 to 11–12/16. Only five training examples have one hand left, and none within this motif has a Red-seal Bonus card in the Pair. Full state, ranking and coverage checks pass. These findings identify loss of an already correct behavior alongside sparse coverage. The subsequent component intervention above separates correction-head updates from changes to its encoder inputs. Neither study establishes a new win gain. World models remain deferred.

The [completed continuation hand replay](continuation_hand_cost_20260925.md) finds a mixed consequence gap on 6,113 identical teacher-play states. Ordinary continuation clears 2,671 / 2,683 blinds versus the base’s 2,699, but introduces five immediate losses per replica versus seven. Weighting matches fewer teacher actions yet clears 2,687 / 2,692 blinds and introduces three / five immediate losses; its advantage over ordinary fitting remains unresolved. All four continuations share a new last-hand failure: a clearing 13,296-point Pair becomes a losing 6,313-point High Card against an 11,000-point target. Full historical controls and independent verification of all eighteen fixed comparisons pass. Worse imitation therefore cannot alone explain the sampled-win gap. The completed trace above follows that regression through the saved checkpoints and training coverage. World models remain deferred; long-horizon generalization remains unproven.

The [completed held-out continuation diagnosis](continuation_transfer_diagnostic_20260925.md) finds **overfitting before autonomous trajectory drift is needed**. In both ordinary continuations, training NLL falls from 0.496954 to about 0.3717 while held-out NLL rises from 0.639106 to about 0.6904: a pooled **+0.051260 [0.043211, 0.059862]** regression. Weighting worsens held-out play agreement by **1.235 points [0.734, 1.742]** relative to ordinary fitting. The full baseline reproduces exactly; all 130,225 prediction rows and 24 paired comparisons pass independent verification. This diagnoses these continuations on an already exposed 128-game panel, not the entire original pretraining failure or an architectural limit. The completed hand replay above follows those saved choices and finds mixed immediate consequences. World models remain deferred and the long-horizon goal remains unproven.

The [matched consequence-weighting experiment](consequence_weighting_study_20260924.md)
is complete and independently verified, and **four of five scientific gates
failed**. Ordinary continuation improves training agreement by about three
points and lowers NLL from 0.496954 to about 0.3717, yet sampled fresh wins are
28.19% versus 28.00% for the base: +0.188 points [−2.938, +3.252]. Weighting fits
the selected costly errors better in both replicas but reaches 26.81% sampled
wins: −1.375 points [−3.875, +1.063] against ordinary fitting. Its early-only
hybrid difference is +2.063 points [−0.563, +4.688], still unresolved. All 16
checkpoints, 9,200 played game episodes, 1,391,954 actions and fixed statistical
comparisons passed independent checks. These negative or unresolved results
establish that training-fit gains alone did not reliably transfer under this
recipe; they do not identify the remaining cause or an architectural limit.
The subsequent diagnostic above measures the fixed endpoints on exposed
held-out teacher states and finds that ordinary fitting itself regresses. The long-horizon goal remains
unproven. The full replay below is the diagnostic basis for the completed test.

The latest [full training-cost replay](consequence_weighting_preparation_20260924.md)
reproduces all 202,141 decisions in the later 1,000-game corpus. The saved
corrected model misses 1,395 teacher blind clears and introduces 33 immediate
losses through alternative played hands. These are one-step teacher-path
interventions, not autonomous win rates. A fixed training-only consequence rule
upweights 5,850 decisions, with verified weights and a passed paired training
smoke. The matched experiment above tests supervision allocation within
existing coverage. World-model work stays deferred.

The subsequent [trajectory audit](fitting_trajectory_20260924.md) finds continued
training improvement with weakening transfer. From updates 2,000 to 4,000,
trainable training NLL falls by 0.06991, but exposed held-out NLL falls by only
0.00437; the late held-out play improvement is unresolved. The fatal Even Steven
choice remains wrong at every saved snapshot, despite rising teacher-action
probability. All four intermediate checkpoints are measured, and the full
26,045-row endpoint control reproduces exactly. This supports keeping ordinary
additional fitting as a measured control when testing coverage or weighting;
neither convergence nor the sufficiency of longer training is established.
No new gameplay result or long-horizon generalization claim is supplied.

The completed [coverage and fitting audit](even_steven_coverage_20260924.md)
now checks the original 5,000 games as well as the later training and exposed
held-out corpora. Only 1.90% of original recorded plays occur with one hand left.
Of 834 ordinary-hand Even Steven states with both even and odd pairs available,
nineteen have one hand left and only one of those has a single-pair teacher action.
The later trainable policy also continues to choose a lower-scoring pair in two
of three relevant training examples after six or seven presentations. Neither
substitution immediately loses its training game. These results identify sparse
coverage of the diagnosed decision context alongside incomplete fitting; they
do not establish which causes the held-out error or how much of the overall win
gap either explains. Separate those two mechanisms before changing architecture.

The subsequent [residual-error inspection](residual_hand_errors_20260924.md)
reproduces all seven remaining fatal choices and 672 saved neural predictions.
One shared failure selects a 795-point Pair instead of a 1,989-point Pair; visible
card ranks, Pair level, Banner and Even Steven exactly account for the score gap.
In two selected cases the learned subset correction reverses a teacher-favoring
pairwise reference ranking. This localizes concrete errors after the original
output cap was removed, while leaving their data/representation/optimization
cause open. The coverage and fitting audit above follows those interactions
through the existing corpora. These conditional cases are not a population-frequency study,
an architectural impossibility result or a new generalization claim.

The completed [corrected-policy hand-cost comparison](corrected_hand_cost_20260924.md)
now measures all three saved Stage 2 policies on 6,113 identical exposed held-out
teacher-play states. Trainable versus frozen encoder yields 47 more immediate
blind clears: +0.769 percentage points [0.200, 1.336]. Of those net additional
clears, 46 occur at Ante 4 or later; the early difference is unresolved. Choices
scoring at most half the teacher decrease from 511 to 408, but both policies
introduce seven immediate losses where the teacher survives. This identifies a
limited improvement in immediate hand selection alongside persistent errors.
It does not attribute the earlier autonomous win gain to better early decisions.
All 26,045 factual decisions reproduce, all saved-row arithmetic and provenance
checks pass, and both diagnostic processes are terminal. Next, explain residual
hand-ranking errors under the existing inputs and fitting recipe, and distinguish
them from errors on autonomous student paths. World-model work remains deferred.

The completed [original-model hand-cost audit](original_hand_cost_20260924.md)
now measures consequences of the saved epoch-10 predictions on original training
states. All 44,348 decisions in the fixed 200-game panel reproduce. Among 4,445
different played hands, 3,559 score lower, 531 tie and 355 score higher; 885 of the
lower scores still clear the blind in both branches. There are 918 missed teacher
blind clears versus 158 extra student clears, and 23 new immediate losses across
21 games. These are individual interventions on teacher paths, not autonomous
win-rate estimates. Some 805 of 918 missed clears lie outside the original valid
candidate set, connecting the error costs to the known output restriction while
leaving causal attribution unresolved. The corrected-policy comparison above
measures remaining errors on its own later held-out panel; it does not use the
different original-corpus panel as a matched control for the head intervention.

On September 24, a [reproduced teacher-cache defect](teacher_cache_gap_20260924.md)
added a concrete label-quality issue to this investigation. A mutable cache key
could leave the teacher's Joker names stale after purchases and sales. Invalidating
only that key changed a recorded later teacher decision from a 315-point hand
to a 1,342-point hand on the identical state. The affected source exactly matches
the original generation manifest. The working-tree fix passes 129 targeted tests;
historical artifacts remain unchanged. A completed replay now reproduces all
3,909 observations and historical labels in twenty original-corpus games. The
cache fix changes thirteen decisions across two games; all six play-to-play
changes improve immediate score. This is 0.33% of a fixed ten-loss/ten-win panel,
not a corpus-wide rate or a student win result. The original teacher's 74.56%
win rate already includes the bug, so fixing it does not by itself explain the
transformer's failure to match that performance.

The user's September 23 correction is the current research priority: explain why
the strong demonstration corpus did not produce a comparably strong transformer
policy. Later PPO gains and observation experiments do not by themselves answer
that question. The [replicated confirmation attempt](pretraining_confirmation_20260923.md)
ended incomplete on September 24 after host sleep consumed its fixed wall-clock
allowance. It supplies a repeated training-fitting benefit at update 1,000,
but no fresh confirmation or early-decision result. Its processes are terminal;
its no-retry/no-resume contract and unopened fresh-outcome gate remain intact.
The chronology below preserves earlier measurements; the latest completed
gameplay evidence is the Stage 2 report at the end of this document.

The [observation evaluation](observation_generalization_results_20260923.md) completed all 26 planned panels and
10,400 games on September 23. The required external control validation passed,
but all three scientific acceptance gates failed. Full-information sampled wins
were 40.1875%, versus 40.625% for the matched ablation and 40.75% for the parent.
The full-minus-ablation difference was -0.4375 percentage points (paired 95%
interval [-3.0625, +2.25]); the early-policy difference was +0.8125 points
[-1.875, +3.5625]. This intervention did not establish the three missing fields
as the performance bottleneck; it does not prove that observation sufficiency
or longer training is irrelevant to the original pretraining failure. An
independent analysis reproduced every raw-game count, comparison, interval and
gate, and verified all panel and frozen-source hashes.

The user explicitly instructed: “until we figure out the existing gaps dont
bother with world models.” All further world-model research, design and
experiments are deferred. Keep work focused on the observed fitting failure,
demonstration/student information differences, and the gap between teacher-state
imitation and autonomous play. A completed diagnostic is not automatically a
complete explanation of the transfer failure.

## What the existing evidence establishes

- The original `cache/datasets/5k-balatro-games/metadata/summary.json` records
  5,000 games, 1,126,644 decisions and 3,728 wins: **74.56%**, rather than 85%.
  The V68 teacher used Blue Deck, White Stake, Ante 8, search shops and scaler
  growth. Its full simulator/scoring/RNG access must be distinguished from the
  student's input. The matched teacher later won 143/200 games (71.5%).
- The completed original run is
  `logs/supervised_5k_v68_20260911_181359.log` and the corresponding TensorBoard
  directory under `runs/`. It completed ten epochs and 35,210 updates, with
  batch size 320 and approximately 11.27 million row exposures. Training action
  accuracy rose from 62.10% to 78.83%; action NLL fell from 1.640 to 0.6904.
  There were no original validation tags or held-out validation loop. Continued
  loss improvement does not establish convergence or diagnose its absence.
- The [original transfer audit](report.md) evaluated all 44,348 decisions from
  200 sampled training games. Epoch 10 matched 50.27% of plays and 40.18% of
  discards, despite 79.56% overall agreement. The
  [learning-bottleneck study](learning_bottleneck.md) found only 6.22% exact
  agreement on 1,237 audited Joker sales. Exact disagreement is not automatically
  a harmful action, but the aggregate metric concealed important weaknesses.
- The original fixed candidate/AR mixture imposed a probability ceiling on
  teacher hand actions outside its candidate list. The original sale readout
  poorly selected Joker entities. These are specific limitations; they do not
  establish a general inability of transformers to represent strategy.
- The [action-head study](action_refactor.md) trained corrected subset and
  entity-sale heads while freezing the original encoder and heads. Independent
  confirmation improved sampled wins from 3.58% to 19.33% and greedy wins from
  7.75% to 32.12%. This demonstrates recoverable information in the representation
  and a consequential action-readout/training bottleneck. The treatment bundles
  architecture, new trainable capacity and focused supervision; it does not
  separately identify each contribution.
- On September 23 the root agent reread all twelve historical action-head
  confirmation files, checked their recorded manifest/outcome hashes and 400
  distinct expected seeds per file, and reproduced the raw counts: sampled
  control 43/1200, candidate replicas 237/1200 and 227/1200; greedy control
  31/400, candidates 122/400 and 135/400. This check did not open current-study
  outcomes or rerun gameplay.

## What remains unresolved

We have a partial explanation of the original transfer failure, not an account
of the full remaining teacher–student gap. Existing evidence does not separate
full-corpus fitting/optimization, generalization to new observable states,
teacher information unavailable to the student, and errors caused by visiting
states outside the demonstrations. The bounded 320-observation memorization
test passed, but it is not complete-game imitation or full-task convergence.
The historical head correction initially froze the encoder. Subsequent
32-game and 1,000-game comparisons, followed by the matched continuation study,
demonstrate additional fitting headroom with the encoder trainable. They still
do not establish convergence or fully explain the remaining fresh-state and
autonomous-play gaps.

## Next diagnostic priority

### Saved corrected-head curves narrow the next question

The complete training/validation curves already exist for the historical head
experiments. A September 23 stdlib-only reanalysis of all four curves is saved at
`eval/pretraining_transfer_20260923/analysis/existing_fit_summary_v1.json`
(SHA256 `a1e4f1a20ffe41f0be1ffc779e5b7780a3f854613c82ef95bc23c4f68777d8d0`).
It retains source hashes, all recorded updates, both weighting conventions and
the play/discard strata. Endpoint conditional-choice accuracy was:

| Corrected head | Training, replicas 0 / 1 | Validation, replicas 0 / 1 |
| --- | --- | --- |
| Complete hand subset | 65.47% / 65.36% | 62.90% / 62.91% |
| Joker selection when selling | 57.85% / 57.66% | 53.30% / 54.17% |

These are conditional choices given the teacher action family, not complete
action agreement or wins. Validation used different game seeds but selected the
historical checkpoints, so it is exposed development evidence. The substantial
training error and comparatively modest split gap make fitting with a trainable
encoder a direct next contrast. They do not establish why fitting is incomplete,
prove convergence, or exclude larger errors on autonomously visited states.

### Matched complete-game fitting

The completed diagnostic is specified in
`eval/pretraining_transfer_20260923/stage1_protocol_draft.md`. Both arms start
from the same corrected supervised checkpoint with a fresh optimizer; they train
the same actor heads and differ only in whether inherited embeddings and the
transformer backbone are trainable. Keep v13 inputs fixed. The exported corrected
checkpoints lack the original supervised configuration and optimizer, so this
must be declared as a new training run, not an exact resume. Its manifest and
launch configuration are now frozen in `eval/pretraining_transfer_20260923/`.
Both actual-device gradient/movement smokes and both fixed 1,000-update arms
completed within the two-hour ceiling (3,619.77 seconds total). Exact training
action agreement was **83.82% with the encoder frozen versus 98.75% with it
trainable**; NLL was 0.58228 versus 0.04691. Play, discard and sale agreement
reached 99.93%, 98.80% and 99.07% respectively in the unfrozen arm. Pack claims
account for 71 of its 82 remaining disagreements. These results identify a
substantial frozen-representation fitting restriction under this recipe, not
fresh-state generalization or autonomous wins. See the [complete-game fitting
report](complete_game_fitting_results_20260923.md).

Both arms' saved metrics, matched baseline and exposures passed independent
artifact audits. Residual classification localizes 70 errors to Standard playing
card claims and one to a Planet claim. Scores were not saved, so frequent
first-slot predictions do not prove ties. The missing enhancement feature is
known, but its contribution to these specific errors remains unmeasured.

The 32 selected complete games come from the later 1,000-game demonstration
dataset used for the action-head investigation, not directly from the original
5,000-game HF corpus. This diagnostic tests a remaining encoder-fitting
constraint in the corrected policy. It does not reconstruct the original
pretraining run or identify every cause of its transfer failure. Original-corpus
training error, the action-head intervention, and this new comparison must retain
their separate data provenance when interpreting the result.

The [static runtime review](../../eval/pretraining_transfer_20260923/analysis/runtime_parity_review_v1.md)
identified a known deck-exhaustion correction in the CLI controller that the
earlier simulator-only hash comparison omitted. The current controller, wrappers
and native fast runner match the saved post-fix reproduction exactly. Future
fresh teacher and student evaluation should use that same corrected candidate;
restoring the old broken controller would be inappropriate. The bounded replay
of two existing, exposed trajectories (367 decisions) passed after fitting: saved
observations, FastRunner/Gym, rewards, legality, states and terminal outcomes
matched, as did ordinary evaluation with zero and eight workers. It took 23.31
seconds and involved no policy inference. This bounded result does not certify
all reachable states or quantify the historical correction’s contribution to
the original transfer gap.

First reconcile existing original training curves with phase-specific training
and whole-game held-out imitation measurements. A further controlled diagnostic
should distinguish three failures: failure to fit decisions from complete
training games, failure on held-out teacher states, and failure in autonomous
rollouts despite adequate teacher-state predictions. Use complete game seed
splits, a declared information regime, and fixed train/validation/rollout settings.
Track teacher-action probabilities and important action families as well as
argmax agreement; measure the gameplay consequences of disagreements rather
than assuming every alternative is wrong.

The [larger-corpus protocol](../../eval/pretraining_transfer_20260923/stage2_protocol_frozen_v1.md)
is now frozen and running under its own supervisor. It resets both arms to the
common initial corrected checkpoint, uses all 1,000 later demonstration games
(202,141 decisions), and fixes 4,000 updates per arm. It then measures fresh
teacher-state imitation and 768 autonomous games across the initial and both
endpoint policies, using 128 reserved game seeds and both greedy and sampled
play. The primary contrast is unfrozen minus frozen; comparison with the initial
policy is secondary. No fresh outcomes are opened until both fits finish.

The ceilings are eight cumulative hours for training/imitation and two for CPU
collection/gameplay, within ten hours overall. Seed separation, bounded memory,
helper interfaces and subprocess ownership checks passed before the actual native
launch. Final runtime success and endpoint results remain pending. This single-replica
diagnostic will not by itself satisfy the broader replicated early-decision bar.

The first frozen-encoder full-corpus snapshot is complete at 1,000 updates
(320,000 row presentations; every training row seen once or twice). Its saved
[metric report](../../eval/pretraining_transfer_20260923/stage2/arms/frozen_encoder/metrics_1000.json)
shows a small, mixed early change:

| Training action agreement | Initial | Frozen encoder, update 1,000 |
| --- | ---: | ---: |
| All 202,141 decisions | 82.16% | 82.32% |
| Plays | 67.23% | 67.75% |
| Discards | 48.63% | 48.20% |
| Joker sales | 19.81% | 21.03% |

Overall NLL fell from 0.672312 to 0.659596, while mean teacher-action probability
changed from 0.746972 to 0.746434. These measure different properties; the snapshot
does not show uniform improvement. The worker records movement in all 52 actor
tensors, with the backbone, embeddings and critic unchanged. This is an interim
training result after approximately 1.58 corpus passes, not the fixed endpoint,
a convergence claim, an encoder comparison, or evidence of fresh-state or win-rate
generalization. The 32-game diagnostic had many more presentations per row at the
same update count and must not be treated as an equivalent fitting budget.

An independent [interim arithmetic audit](../../eval/pretraining_transfer_20260923/analysis/stage2_frozen_interim_1000_arithmetic_audit_1790179574250427000.json)
passed all 23 checks: both snapshots contain all 202,141 ordered decisions from
the 1,000 complete training games; overall, all 14 action-family, per-game and
support-count summaries reproduce from saved predictions. Exposure is exactly
320,000: 84,282 rows once and 117,859 twice. The audit verifies artifact arithmetic
and coverage, not an independent model forward or gameplay effect.

An exploratory [flat-action error breakdown](../../eval/pretraining_transfer_20260923/analysis/training_family_errors_0000_1000_v1.md)
further separates the remaining disagreements. At update 1,000, play labels have
703 predictions in another action family and 14,502 incorrect card choices within
the play family; discard labels have 2,345 and 3,464 respectively. Joker-sale labels
have 3,938 predictions in another family and 1,128 wrong choices within the sale
family. Cross-family sale predictions include 1,733 buys, 967 shop exits and 820
rerolls. Across the corpus, the 317 net reduction in exact-action disagreements
comprises 18 fewer cross-family disagreements and 299 fewer within-family ones.
This describes the most probable complete action. The implementation selects a
flat-action mode, so it does not identify a wrong macro-head argmax; the saved
predictions lack macro and conditional distributions. These counts also do not
establish that an alternative action is harmful in gameplay.

The [2,000-update snapshot](../../eval/pretraining_transfer_20260923/stage2/arms/frozen_encoder/metrics_2000.json)
is also complete. With 640,000 presentations (each row seen three or four times),
agreement reaches 82.3861% and NLL reaches 0.654788; mean teacher-action probability
is 0.747750. The agreement gain from update 1,000 is 0.0663 percentage points.
Play, discard and Joker-sale agreement are 67.85%, 48.50% and 20.47%, respectively.
This extends the evidence of slow, mixed fitting improvement with the encoder
frozen. It is still an interim measurement, with both fixed 4,000-update endpoints
and the fresh-state/autonomous comparisons pending; it does not prove convergence
or establish the encoder as the remaining bottleneck.
Its [minimal independent check](../../eval/pretraining_transfer_20260923/analysis/stage2_frozen_interim_2000_arithmetic_audit_1790181162700245000.json)
reproduced all three overall metrics, matched every ordered teacher-row identity
to the audited baseline, and verified exposure: 168,564 rows three times and
33,577 four times. All nine checks passed. Full paired endpoint analysis remains pending.

The frozen arm has now completed the fixed **4,000-update endpoint**. After
1,280,000 presentations, exact agreement is **82.4459%**, NLL is **0.647823**,
and mean teacher-action probability is **0.746863**. Agreement improved by just
0.2830 percentage points from the initial checkpoint and 0.0599 points from
update 2,000. Plays reach 67.94%, discards 48.85%, Joker sales 20.67%, and pack
claims 83.12%. This is substantial remaining training error after approximately
6.33 corpus passes under the fixed recipe. It does not prove convergence, an
irreducible error floor, or the cause of the error.

The [independent endpoint check](../../eval/pretraining_transfer_20260923/analysis/stage2_frozen_endpoint_4000_arithmetic_audit_1790184034882365000.json)
passed all 20 checks. It reproduced overall and four critical-family metrics,
matched all 202,141 ordered row identities to baseline, and verified the exposure
total: 134,987 rows six times and 67,154 seven times. Completion metadata and
consumed artifact hashes agree. The check covers saved training arithmetic and
metadata, without an independent model forward or checkpoint decoding.

The unfrozen arm started after the frozen worker completed successfully. Its
[actual admission check](../../eval/pretraining_transfer_20260923/analysis/actual_stage2_unfrozen_admission_1790184103223218000.json)
passed 20 metadata checks: the same original checkpoint reference, recipe,
training-row identities and action-head trainability, a fresh optimizer, and
matching recorded module origins and cache array hashes. The 52 actor, 146
backbone and 112 embedding tensors are trainable; eight critic tensors remain
frozen. The subsequent [baseline and first-update check](../../eval/pretraining_transfer_20260923/analysis/stage2_paired_baseline_first_update_audit_1790184583527102000.json)
passed all 16 checks: both baseline prediction files are byte-identical across
202,141 rows, complete saved baseline metrics match, exposure is zero, and all
318 parameter tensors report no initial movement. The first update records
finite positive gradients and movement in actor, backbone and embedding groups,
with frozen tensors unchanged. These tensor checks validate worker records,
without independent tensor recomputation. Unfrozen training is underway; its
fixed endpoint, fresh-state imitation and autonomous wins remain pending.

The first larger-corpus **matched 1,000-update comparison** is now available.
Both arms have seen the same 320,000 presentations, approximately 1.58 passes:

| Training metric at update 1,000 | Frozen encoder | Trainable encoder |
| --- | ---: | ---: |
| All 202,141 decisions: exact agreement | 82.32% | 83.37% |
| All decisions: NLL | 0.659596 | 0.607324 |
| Mean teacher-action probability | 0.746434 | 0.756508 |
| Play agreement | 67.75% | 70.25% |
| Discard agreement | 48.20% | 50.41% |
| Joker-sale agreement | 21.03% | 24.07% |
| Pack-claim agreement | 83.18% | 83.73% |

The trainable encoder improves agreement by 1.0512 percentage points and reduces
NLL by 0.052272 relative to the frozen arm at the same update and exposure count.
This is early evidence that allowing the representation to adapt improves fitting
under the tested recipe. It leaves substantial training error and does not prove
convergence, explain all of the original pretraining failure, or demonstrate
generalization. The much smaller 32-game diagnostic had many more presentations
per row at update 1,000. The fixed 4,000-update comparison and subsequent fresh
imitation and autonomous evaluation remain necessary. Training continues without
changing the recipe or selecting an interim checkpoint.

An [independent check of the unfrozen snapshot](../../eval/pretraining_transfer_20260923/analysis/stage2_unfrozen_interim_1000_arithmetic_audit_1790187054215188000.json)
passed all 19 checks. Overall and four critical-family metrics reproduce from
saved predictions, every ordered row identity matches baseline, and exposure
files are byte-identical to the audited frozen snapshot: 84,282 rows once and
117,859 twice. Recorded row-stream state and digest also match. The worker
reports movement in all 52 actor and 146 backbone tensors and 109 embedding
tensors, with all eight critic tensors unchanged. Movement and gradient checks
validate saved worker records, not an independent tensor or optimizer replay.
No fresh outcomes or predictions have been opened.

The matched **2,000-update comparison** extends this fitting benefit. Both arms
have received 640,000 presentations, approximately 3.17 passes over the same
202,141 training decisions:

| Training metric at update 2,000 | Frozen encoder | Trainable encoder |
| --- | ---: | ---: |
| Exact agreement | 82.39% | 84.24% |
| NLL | 0.654788 | 0.566865 |
| Mean teacher-action probability | 0.747750 | 0.772186 |
| Play agreement | 67.85% | 72.33% |
| Discard agreement | 48.50% | 52.59% |
| Joker-sale agreement | 20.47% | 24.41% |
| Pack-claim agreement | 83.23% | 84.09% |

The trainable encoder's agreement advantage increases to 1.8507 percentage
points, with NLL lower by 0.087922. Its overall fitting continues to improve
between updates 1,000 and 2,000, but the improvement is not uniform: Joker-sale
NLL rises from 1.919177 to 1.964885 despite a small increase in sale agreement.
Substantial training error remains, particularly for discards and sales. These
results concern the later corrected-head corpus; the original pretraining also
trained its encoder, so this contrast does not establish that encoder freezing
caused the original failure. The fixed 4,000-update endpoint and fresh-state
imitation and autonomous wins remain pending.

The [minimal independent 2,000-update check](../../eval/pretraining_transfer_20260923/analysis/stage2_unfrozen_interim_2000_arithmetic_audit_1790189816415713000.json)
passed all ten checks. The three overall metrics reproduce from predictions,
every ordered teacher-row identity matches the audited baseline, and exposure
is byte-identical to frozen update 2,000: 168,564 rows seen three times and
33,577 seen four times. Recorded row-stream state and digest also match. All
seven consumed input files remained unchanged. This check concerns saved
training arithmetic and exposure; fresh outcomes remain unopened.

World-model work remains deferred until the existing transfer gaps are explained.
The broader long-horizon generalization goal remains active and unproven.

The **fixed 4,000-update fitting comparison is complete**. Each arm received
1,280,000 presentations, approximately 6.33 passes over the same 202,141 decisions
from 1,000 training games. The
[trainable-encoder endpoint](../../eval/pretraining_transfer_20260923/stage2/arms/unfrozen_encoder/metrics_4000.json)
compares with the
[frozen endpoint](../../eval/pretraining_transfer_20260923/stage2/arms/frozen_encoder/metrics_4000.json)
as follows:

| Training metric at update 4,000 | Frozen encoder | Trainable encoder |
| --- | ---: | ---: |
| Exact agreement | 82.45% | 85.83% |
| NLL | 0.647823 | 0.496954 |
| Mean teacher-action probability | 0.746863 | 0.787056 |
| Play agreement | 67.94% | 75.14% |
| Discard agreement | 48.85% | 57.94% |
| Joker-sale agreement | 20.67% | 37.69% |
| Pack-claim agreement | 83.12% | 85.14% |

Allowing the representation to adapt improves training agreement by 3.3838
percentage points and lowers NLL by 0.150869 under this recipe. The advantage
has grown from 1.0512 points at update 1,000 and 1.8507 at update 2,000. Sale
agreement improves sharply between 2,000 and 4,000, from 24.41% to 37.69%, and
sale NLL falls from 1.964885 to 1.565952. Play and discard agreement improve by
7.21 and 9.09 points relative to the frozen endpoint. Improvement is not uniform
across all families, and substantial training error remains. Continuing gains
do not establish convergence or an irreducible error floor.

This extends the small-set fitting result to the later 1,000-game corrected-head
corpus. It does not reconstruct the original 5,000-game pretraining, whose encoder
was also trainable, or establish fresh-state transfer or wins. The unfrozen fit
completed in 9,982.11 seconds; its child exited successfully before the supervisor
started fresh imitation. The reserved fresh-state comparison has completed successfully, and autonomous
evaluation is now running. Fresh outcome and prediction contents remain
unopened pending completion of the whole run. World-model work stays deferred,
and the broader goal remains unproven.

The [independent unfrozen endpoint check](../../eval/pretraining_transfer_20260923/analysis/stage2_unfrozen_endpoint_4000_arithmetic_audit_1790194018805692000.json)
passed all 21 checks. Overall and four critical-family decision-weighted and
equal-game metrics reproduce; all 202,141 ordered teacher-row identities match
the audited baseline. Exposure is byte-identical to the frozen endpoint:
134,987 rows seen six times and 67,154 seen seven times, totaling 1,280,000.
Recorded row-stream state and digest match, completion metadata agrees, and
all ten consumed inputs remained unchanged. This verifies saved training
arithmetic and completion consistency, without independently replaying the
optimizer or model, and supplies no fresh-game evidence.

The whole Stage 2 run has now completed successfully. Its
[final transfer report](pretraining_transfer_results_20260923.md) records all
26,045 fresh teacher decisions and 768 autonomous games, with independent
imitation and gameplay checks. The trainable-minus-frozen fresh NLL difference is
-0.031796 [-0.038712, -0.024599], and fresh agreement
improves by 0.7026 points. The much larger 3.3838-point training agreement gain
therefore transfers only partly. Fresh Joker-sale NLL is slightly worse despite
higher agreement.

Against the matched frozen endpoint, greedy wins improve from 37/128 to 51/128,
paired +10.94 points [2.34, 19.53]. Sampled wins improve from 27/128 to 33/128,
+4.69 [-3.91, 13.28], leaving that difference unresolved. Initial-policy wins
are 41/128 greedy and 20/128 sampled; those are secondary contrasts. The matched
teacher wins 95/128 with privileged information and computation. All fixed
modes and endpoints remain reported; no outcome selected a checkpoint or recipe.

This establishes partial transfer under the tested recipe. Independent training
replication, fresh confirmation and a controlled early-decision contribution
remain necessary. The subsequent confirmation attempt is incomplete as described
above. The immediate diagnostic priority remains separating residual fitting
error, unavailable teacher information and errors on student-visited states.
No world-model work is being pursued.
