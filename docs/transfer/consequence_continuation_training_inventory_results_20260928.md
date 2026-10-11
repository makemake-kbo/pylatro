# Existing training labels support a three-action continuation pilot

The training-only inventory identifies **45,852 states across all 1,000 training games** with three distinct labelled plays and at least one branch that is not already terminal. This establishes an available action population, not the value or learnability of new continuation targets. No continuation was simulated and no model was trained.

| Training group | Eligible states | Source games | States with differing immediate clear outcomes | States with differing immediate loss outcomes |
| --- | ---: | ---: | ---: | ---: |
| Early, base differs from teacher | 2,873 | 863 | 986 | 18 |
| Early, base matches teacher | 13,115 | 1,000 | 5,006 | 84 |
| Later, base differs from teacher | 7,959 | 968 | 1,742 | 39 |
| Later, base matches teacher | 21,905 | 975 | 5,856 | 138 |

The [prespecified rule](consequence_continuation_training_inventory_20260928.md) retains teacher and base actions, then the previously assessed added action, deduplicating identities and filling to three from existing proposals in ascending action ID. This gives 137,556 state/action entries, including 136,460 nonterminal branches. Of all 47,149 original training play states, 722 are excluded because base chooses a non-play action, 15 have fewer than three available plays, and 560 have all three selected plays already terminal. The 245,637 original and 47,125 added labels are reconciled exactly.

The runtime pilot is fixed at eight states from eight distinct games, with three actions each. It samples two states from each table row using the declared hash order within the first 64 training seeds. Selected identities are `4000046:10`, `4000036:19`, `4000014:21`, `4000055:23`, `4000040:66`, `4000045:200`, `4000049:228`, and `4000020:126`. The last required factual source seed is 4,000,055. The pilot's full action identities, original outcomes and snapshot hashes are saved before collecting any new continuation outcome.

A first reader failed on the first state because 14,047 historical labels omit a redundant `game_won` field. The [separate compatibility correction](consequence_continuation_training_inventory_correction_20260928.md) derives terminal status from the stored phase and checks every available win flag against it. All 292,762 labels pass phase/loss/clear consistency checks. No missing win flag is treated as a loss. The failed v1 remains preserved; v2's output, protocol and owner are separate. Sampling and budgets are unchanged, and v1 wrote no eligible-state rows or pilot.

The corrected standard-library inventory completed in 2.024 seconds, actually exited zero in tool chunk `32b97f`, and was admitted in `5920f2`. Its emitted state list is recounted and the pilot is verified as a subset with the required distinct action, group and source-game counts. There are no engine or model imports. The next step is a separately frozen runtime and label-diversity pilot using the existing immutable base follow-up controller. A full collection population, fitting recipe and new model remain undecided; immediate-outcome diversity is not evidence of terminal-value diversity. Existing gates and final independent confirmation requirements are unchanged.

- [Inventory result](../../eval/consequence_prediction_20260927/continuation_training_inventory_v2/result_v1.json), SHA256 `7798423633c688835acb4b509d2b48f064e259934ef28edd8f2176bc0394195b`.
- [Selected pilot](../../eval/consequence_prediction_20260927/continuation_training_inventory_v2/pilot_targets.json), SHA256 `2f613555ea631da013f33347726775ce5f0a674e1662d1614ee555abd6160eaa`.
- [Owner](../../eval/consequence_prediction_20260927/continuation_training_inventory_owner_terminal_v2.json), SHA256 `a9b3e2d4a100ec59899ef288d7f462b2a55674596099ba170be353a151d8ed98`.
- [Protocol](../../eval/consequence_prediction_20260927/continuation_training_inventory_protocol_v2.json), SHA256 `9effd21d3b441f4a9a3bed05f09c90643b3f71c27668cf33c93defed61d7813c`.
