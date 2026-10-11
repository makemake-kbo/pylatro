# Preservation and the eight previously diagnosed hand failures

Both preservation replicas still choose the losing 6,313-point High Card at seed 850000064 / step 138, where the base and teacher choose a 13,296-point Pair against an 11,000-point target. The preservation objective raises the winning action’s probability relative to ordinary continuation, but does not restore its top rank. This is a limitation of the tested intervention on an exposed diagnostic case, not proof that retention is irrelevant to generalization.

| Policy | Chosen action | Known score | Teacher-action probability |
|---|---:|---:|---:|
| base | 415 | 13,296 | 19.1845% |
| ordinary_bc_0 | 434 | 6,313 | 11.9454% |
| ordinary_bc_1 | 434 | 6,313 | 11.8160% |
| policy_preservation_bc_0 | 434 | 6,313 | 15.7988% |
| policy_preservation_bc_1 | 434 | 6,313 | 15.0374% |

All eight cases were selected before this preservation experiment and are retained below. C means the previously replayed action clears the current blind; L means it immediately loses. The action identifier precedes that outcome. These are fixed teacher states, not autonomous win rates.

| Seed / step | Base | Ordinary 0 | Ordinary 1 | Preservation 0 | Preservation 1 |
|---|---|---|---|---|---|
| 850000015 / 28 | 2519 L | 2519 L | 2519 L | 2519 L | 2519 L |
| 850000027 / 121 | 534 L | 534 L | 534 L | 534 L | 534 L |
| 850000042 / 113 | 5600 L | 534 C | 534 C | 534 C | 534 C |
| 850000064 / 138 | 415 C | 434 L | 434 L | 434 L | 434 L |
| 850000076 / 120 | 2519 L | 2519 L | 4940 C | 4940 C | 4940 C |
| 850000077 / 77 | 6105 L | 4989 C | 6635 L | 4989 C | 6105 L |
| 850000093 / 57 | 699 L | 3884 C | 989 C | 3884 C | 699 L |
| 850000096 / 130 | 4895 L | 4895 L | 4895 L | 4895 L | 4895 L |

The tradeoff remains visible. Preservation replica 0 changes the ordinary replica-0 loss at 850000076 / 120 into a clear. Preservation replica 1 changes the ordinary replica-1 clear at 850000093 / 57 back into the base’s losing choice. Both retain the ordinary correction at 850000042 / 113. These selected cases cannot estimate a population rate or explain the size of the full-game win gap.

The check joins saved predictions to the same verified observation cache, exact row indices, seeds, steps and teacher actions. All 24 base/ordinary historical choices and teacher NLL values reproduce. A separate direct recount checks all 40 saved case-policy records and their exact previously replayed outcomes. Every chosen action here has a known replayed outcome; the method leaves unseen actions unscored. No new neural inference or engine replay was performed.

The first lookup helper stopped because older ordinary prediction files omit the Ante field. Its failed exit and source are retained. Version 2 handles that legacy schema explicitly while requiring the same cache index, state identity, teacher action and historical action/NLL controls. This correction did not change fitting, gameplay, primary analysis or the independently verified fresh results.

The [completed fresh-game experiment](policy_preservation_results_20260926.md) failed four of five scientific criteria. The present follow-up sharpens the remaining question: how do we improve consequential hand rankings while retaining useful corrections? It does not justify world-model work or establish a new architecture as necessary.

- [All eight cases and source bindings](../../eval/policy_preservation_20260925/selected_cases_followup_v2.json)
- [Successful lookup exit](../../eval/policy_preservation_20260925/selected_cases_followup_terminal_v2.json)
- [Retained failed lookup](../../eval/policy_preservation_20260925/selected_cases_followup_terminal_v1.json)
- [Prior component diagnosis](correction_component_attribution_20260925.md)
