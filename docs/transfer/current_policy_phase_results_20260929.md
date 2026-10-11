# Current-policy headroom extends beyond played-hand selection

The matched current-policy diagnostic finds substantial headroom in **both
hand and shop decisions**. On 64 development seeds, the student wins 11 games;
teacher help on agreed played subsets wins 20, whole-hand help 29, shop help
29, and combined hand/shop help 54. Whole-hand assistance adds another nine
wins beyond the restricted played-subset intervention. Further play-only
training would address only part of the demonstrated gap.

All **384 games finish uncensored**. Independent native replay verifies every
one of their **67,214 decisions**, including sampled proposals, teacher
proposals, intervention rules, state/input/RNG hashes and terminal outcomes.
The complete owner exits successfully before these results are admitted.

| Arm | Wins / 64 | Win rate | Difference from student, percentage points [paired 95% interval] |
| --- | ---: | ---: | --- |
| Student | 11 | 17.19% | Reference |
| Agreed played subsets | 20 | 31.25% | +14.06 [+1.56, +26.56] |
| Whole hand phase | 29 | 45.31% | +28.12 [+15.62, +40.62] |
| Shop phase | 29 | 45.31% | +28.12 [+15.62, +42.19] |
| Hand and shop | 54 | 84.38% | +67.19 [+56.25, +78.12] |
| Full teacher | 47 | 73.44% | +56.25 [+43.75, +68.75] |

The [fixed protocol](current_policy_phase_diagnostic_20260929.md) uses the
current corrected-head, adapted-encoder checkpoint, not the older models in
previous phase studies. The 64 seeds are selected without outcomes from the
already exposed 128-game development population. Each arm has one trajectory
per seed, CPU FP32, batch one, temperature 1, mixture 0.5, Blue deck, White
stake and Ante 8. Per-game policy RNG streams differ from production batch-32
evaluation. The student percentage here does not replace earlier full-panel
estimates, and these are not fresh confirmation games.

Agreed-play assistance replaces a subset only when both policies propose a
play. Whole-hand assistance covers every CHOOSE_ACTION decision, including
discards and consumables. Shop assistance covers SHOP, leaving packs to the
student. Every hybrid samples the student before possible replacement. The
teacher queries a private simulator copy; its privileged information never
enters the learned policy. The replay also verifies **3,680 shared prefix
decisions across 256 student/hybrid pairs**, through their first changed action.

Whole-hand minus agreed-play is **+14.06 points [+1.56, +26.56]**. Shop minus
hand is **0 [-15.62, +15.62]**, so their equal win counts do not establish equal
quality or rank the two phases. Combined assistance adds **+39.06 points
[+26.56, +51.56]** over either single-phase arm. The factorial interaction is
**+10.94 [-6.25, +28.12]**; a positive superadditive interaction is not
established. Full teacher minus combined assistance is **−10.94
[−21.88, 0]**, also lacking a strictly separated interval. Do not infer that
the hybrid reliably surpasses the teacher.

Intervals use 10,000 paired seed bootstrap draws, seed 1729. They are
exploratory marginal intervals across multiple comparisons, not simultaneous
inference or estimates of training-seed uncertainty.

The [saved-trace recount](../../eval/consequence_prediction_20260927/current_phase_trace_summary_v1.json)
helps define the next questions without attributing wins to individual changes:

- Whole-hand help makes 1,945 replacements: 1,191 change a played subset,
  372 change a discarded subset, and 334 change the action family. Among the
  family changes, 164 switch play to discard and 141 discard to play. Another
  48 change a conditional choice within other families. There are 729 early
  replacements, at Ante 1–3.
- Shop help makes 648 replacements: **445 change the action family**, 129
  change the purchased item and 74 change the Joker sold. There are 242 early
  replacements. This includes both deciding what kind of shop action to take
  and choosing its item; it is not just a sale-slot problem.
- Agreed-play help makes 1,004 replacements, all played-subset changes, of
  which 348 occur early. Different arms visit different later states, so
  subtracting replacement counts does not isolate the causal contribution of
  a family.

The current architecture already scores purchases from contextual shop tokens
and includes the repaired entity-sale residual. A generic proposal to add an
entity purchase or sale head would overlook existing code. See the frozen
[shop readout](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/action_grammar.py).

The next diagnostic should distinguish **action-family choice from conditional
item/subset choice**, and **represented-but-poorly-fitted decisions from
held-out transfer errors**, using compatible existing training/development
predictions first. The shop result warrants investigation alongside hand
timing and discards; these aggregate effects do not justify ranking shops
above hands or immediately selecting another architecture. No further fit or
collection is selected by this report.

Measurement takes **2,334 seconds** and replay verification **2,311 seconds**,
within their separate 6,024-second limits. A subsequent standard-library-only
recount verifies all traces and replacement totals without model or simulator
calls. No model is trained or changed. This localizes teacher-assisted
headroom; it does not prove a missing observation, the cause of the remaining
learning gap, or an autonomous generalization gain. The original fresh-win,
greedy-retention, early-policy attribution and final independent-fit bar
remains unmet. World models remain deferred.

Artifacts: [whole-owner receipt](../../eval/consequence_prediction_20260927/current_phase_owner_terminal_v1.json),
[frozen protocol](../../eval/consequence_prediction_20260927/current_phase_protocol_v1.json),
[results](../../eval/consequence_prediction_20260927/current_phase_v1/result_v1.json),
and [native audit](../../eval/consequence_prediction_20260927/current_phase_v1/verification_v1.json).
