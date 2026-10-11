# Broader survival comparisons expose an unsupervised preference relation

The admitted training labels contain 3,396 survivor–fatal pairs on 430 last-hand states across 332 games. Only 33 pair identities overlap the earlier ranking objective. Of the relational model's 320 preference-ranking errors on original training proposals, 270 involve a surviving action that is not the teacher action. Teacher cross-entropy promotes the teacher over alternatives but does not explicitly order these non-teacher pairs.

The [inventory](consequence_survival_pair_inventory_20260928.md) uses the same 245,637 original and 47,125 added labels, without collecting outcomes or running a model. Of the pairs, 2,366 involve only original proposals and 1,030 involve an added action. The added labels create ten additional mixed-outcome states beyond the original 420. There are 1,052 early pairs and 2,344 later pairs; 1,300 have the teacher as survivor and 2,096 have a non-teacher survivor. None has the teacher as the fatal action. Labels remain single original-RNG realizations, not conditional outcome probabilities.

| Original-proposal pair group | Pairs | Pooled preference errors | Relational preference errors | Relational clear-head errors |
| --- | ---: | ---: | ---: | ---: |
| Teacher survives, alternative fails | 1,102 | 44 | 50 | 79 |
| Non-teacher survives, another action fails | 1,264 | 285 | 270 | 142 |

Thus 84.4% of relational preference errors occur in the second group. Its raw error rate there is 21.4%, versus 4.5% for teacher-survivor pairs. This supports testing direct survival comparisons as a distinct objective change; it does not prove that missing pair supervision causes all errors. The 50 remaining teacher-survivor errors already have direct teacher preference supervision, and consequence predictions also misrank actions.

The prior ranking study used 2,027 teacher/alternative clear comparisons, including 34 fatal alternatives, and trained the full policy. It largely fitted those comparisons but failed fresh-game and early-attribution requirements. The new inventory contains 33 of those 34 fatal pairs. The missing pair, seed 4000471 step 116, compares actions 145 and 139; action 139 is absent from this state's current labelled proposal set. No missing outcome is inferred or added. Broader labelled alternatives, survival-specific targets and an unchanged frozen readout distinguish a possible new controlled experiment from rerunning the old full-policy ranking recipe.

Inventory v1 failed its final row recount because its output writer was not flushed before a second reader opened it. Its failed exit and artifacts are preserved. [Version 2](consequence_survival_pair_inventory_correction_20260928.md) changes file visibility and output paths only. It completed in 1.785 seconds, reproduced all original pair/ranking counts from the prior alignment audit, and recounted every emitted target from terminal labels. The fully closed v1 and v2 target files are byte-identical. No fit, inference, new simulator outcome or development evaluation occurred.

- [Result](../../eval/consequence_prediction_20260927/survival_pair_inventory_v2/result_v1.json), SHA256 `c1333fa0a5bcd872d1a49fafb882693e9063b0555d4d920c439270447d226a87`.
- [Admitted target rows](../../eval/consequence_prediction_20260927/survival_pair_inventory_v2/targets.jsonl), SHA256 `ccd84371d722eee74688a366cf04a7f5eb9292e615ce37fac97d06b0210cecaa`.
- [Owner receipt](../../eval/consequence_prediction_20260927/survival_pair_inventory_owner_terminal_v2.json), actual exit `87b734`, admission `518e74`; SHA256 `231d915dd59e9ea144241d215cbf57728a89f39f2d2465183d8c4e43384592f6`.

The inventory is diagnostic evidence, not a candidate or a generalization result. Any objective experiment must keep the existing feature pathway, action support, labels, exposure and final success gates fixed.
