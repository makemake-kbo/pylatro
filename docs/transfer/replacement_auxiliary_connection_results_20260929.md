# Purchase prediction does not connect reliably to the preceding sale

On the 469 held-out replacement states with multiple offers, the auxiliary model predicts the intended purchase correctly in 441. It nevertheless mismatches the preceding sale in 325 of those 441 states (73.70%): 113 sell another Joker and 212 select another action family. The base already predicts the purchase correctly in 428 states, with 309 sale mismatches among those. The remaining problem is therefore common even when the prospective choice is correct.

The auxiliary model corrects 13 base purchase predictions and loses none on this development group. Those 13 corrections yield zero new exact sale matches and lose two existing sale matches. Against ordinary1k, it corrects 15 future predictions, gains zero sale matches in those states and loses three. These small, outcome-conditioned cells are descriptive; they are not a causal mediation experiment.

Across all 716 development SHOP-sale states, the auxiliary model makes 158 exact matches, 170 different-Joker sales and 388 other-family choices. Base counts are 167, 166 and 383; ordinary1k counts are 170, 173 and 373. Both sale-family choice and owned-Joker choice remain relevant. These are families of the joint flat mode, not direct macro-head argmax measurements.

## What the executed source explains

The purchase logits are a linear readout of contextual shop tokens. The sale logits combine a pooled-state head with an existing residual conditioned on the pooled state and each contextual owned-Joker token. The macro head reads the pooled state. The added prospective loss reads only purchase logits. From this dependency graph, that loss has no direct gradient to the macro head or the two sale readouts; transfer into sale decisions depends on shared encoder features. Ordinary action NLL still trains all relevant sale readouts.

The sale readout does not explicitly consume the predicted purchase or an owned-Joker/offer pair. Its transformer features already include shop context, so this is not missing observation information or proof of architectural inexpressivity. The result rejects this particular auxiliary-supervision route at the fixed endpoint; it does not reject every auxiliary objective or establish a unique bottleneck.

A focused next architectural candidate is an explicit relation between a proposed owned-Joker sale and the offered replacement, with that score connected to both whether to sell and which Joker to sell. Any policy experiment needs non-sale SHOP examples as well as sale examples to learn when a replacement is warranted. It must preserve meaningful order, use future choices only as training targets, and compare with the same ordinary control. This is a proposed mechanism, not a frozen experiment or authorization to skip the existing qualification criteria. No new fit, coefficient sweep, longer training run or confirmation was launched.

## Complete categorical counts

Each row reports exact sale / different Joker sale / other action family. Correct/wrong refers to prospective purchase prediction and is a model-dependent condition. Singleton future choices are automatic. All predeclared groups, including empty cells, follow.

### Training

| Group | Policy | Future condition | States | Exact sale | Different Joker | Other family | Sale NLL |
|---|---|---|---:|---:|---:|---:|---:|
| all_sales | base | all | 5471 | 1963 | 671 | 2837 | 1.610484 |
| all_sales | ordinary1k | all | 5471 | 2103 | 576 | 2792 | 1.513097 |
| all_sales | auxiliary | all | 5471 | 2063 | 542 | 2866 | 1.526982 |
| multiple_offers | base | all | 3494 | 1290 | 485 | 1719 | 1.550691 |
| multiple_offers | base | future_correct | 3198 | 1195 | 452 | 1551 | 1.531508 |
| multiple_offers | base | future_wrong | 296 | 95 | 33 | 168 | 1.757943 |
| multiple_offers | ordinary1k | all | 3494 | 1371 | 424 | 1699 | 1.458298 |
| multiple_offers | ordinary1k | future_correct | 3203 | 1280 | 387 | 1536 | 1.432426 |
| multiple_offers | ordinary1k | future_wrong | 291 | 91 | 37 | 163 | 1.743073 |
| multiple_offers | auxiliary | all | 3494 | 1351 | 388 | 1755 | 1.473431 |
| multiple_offers | auxiliary | future_correct | 3252 | 1284 | 358 | 1610 | 1.447067 |
| multiple_offers | auxiliary | future_wrong | 242 | 67 | 30 | 145 | 1.827715 |
| other_sales | base | all | 288 | 51 | 7 | 230 | 2.340765 |
| other_sales | ordinary1k | all | 288 | 54 | 7 | 227 | 2.274389 |
| other_sales | auxiliary | all | 288 | 52 | 8 | 228 | 2.249173 |
| replacements | base | all | 5183 | 1912 | 664 | 2607 | 1.569905 |
| replacements | base | future_correct | 4887 | 1817 | 631 | 2439 | 1.558516 |
| replacements | base | future_wrong | 296 | 95 | 33 | 168 | 1.757943 |
| replacements | ordinary1k | all | 5183 | 2049 | 569 | 2565 | 1.470795 |
| replacements | ordinary1k | future_correct | 4892 | 1958 | 532 | 2402 | 1.454598 |
| replacements | ordinary1k | future_wrong | 291 | 91 | 37 | 163 | 1.743073 |
| replacements | auxiliary | all | 5183 | 2011 | 534 | 2638 | 1.486852 |
| replacements | auxiliary | future_correct | 4941 | 1944 | 504 | 2493 | 1.470158 |
| replacements | auxiliary | future_wrong | 242 | 67 | 30 | 145 | 1.827715 |
| singleton_offers | base | all | 1689 | 622 | 179 | 888 | 1.609654 |
| singleton_offers | base | future_correct | 1689 | 622 | 179 | 888 | 1.609654 |
| singleton_offers | base | future_wrong | 0 | 0 | 0 | 0 | — |
| singleton_offers | ordinary1k | all | 1689 | 678 | 145 | 866 | 1.496645 |
| singleton_offers | ordinary1k | future_correct | 1689 | 678 | 145 | 866 | 1.496645 |
| singleton_offers | ordinary1k | future_wrong | 0 | 0 | 0 | 0 | — |
| singleton_offers | auxiliary | all | 1689 | 660 | 146 | 883 | 1.514617 |
| singleton_offers | auxiliary | future_correct | 1689 | 660 | 146 | 883 | 1.514617 |
| singleton_offers | auxiliary | future_wrong | 0 | 0 | 0 | 0 | — |

### Future-choice transitions: training

| Group and control | Future transition | States | Both sale-exact | Gained sale-exact | Lost sale-exact | Neither sale-exact | Sale NLL difference |
|---|---|---:|---:|---:|---:|---:|---:|
| multiple_offers_vs_base | auxiliary_corrects | 84 | 25 | 7 | 9 | 43 | -0.087256 |
| multiple_offers_vs_base | auxiliary_loses | 30 | 7 | 3 | 2 | 18 | -0.045568 |
| multiple_offers_vs_base | both_correct | 3168 | 1044 | 208 | 142 | 1774 | -0.080459 |
| multiple_offers_vs_base | both_wrong | 212 | 49 | 8 | 12 | 143 | -0.029983 |
| multiple_offers_vs_ordinary1k | auxiliary_corrects | 81 | 25 | 5 | 6 | 45 | -0.076776 |
| multiple_offers_vs_ordinary1k | auxiliary_loses | 32 | 8 | 2 | 1 | 21 | -0.031809 |
| multiple_offers_vs_ordinary1k | both_correct | 3171 | 1141 | 113 | 130 | 1787 | +0.018988 |
| multiple_offers_vs_ordinary1k | both_wrong | 210 | 52 | 5 | 8 | 145 | -0.000489 |
| replacements_vs_base | auxiliary_corrects | 84 | 25 | 7 | 9 | 43 | -0.087256 |
| replacements_vs_base | auxiliary_loses | 30 | 7 | 3 | 2 | 18 | -0.045568 |
| replacements_vs_base | both_correct | 4857 | 1605 | 307 | 203 | 2742 | -0.085528 |
| replacements_vs_base | both_wrong | 212 | 49 | 8 | 12 | 143 | -0.029983 |
| replacements_vs_ordinary1k | auxiliary_corrects | 81 | 25 | 5 | 6 | 45 | -0.076776 |
| replacements_vs_ordinary1k | auxiliary_loses | 32 | 8 | 2 | 1 | 21 | -0.031809 |
| replacements_vs_ordinary1k | both_correct | 4860 | 1748 | 166 | 201 | 2745 | +0.018635 |
| replacements_vs_ordinary1k | both_wrong | 210 | 52 | 5 | 8 | 145 | -0.000489 |
| singleton_offers_vs_base | auxiliary_corrects | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_base | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_base | both_correct | 1689 | 561 | 99 | 61 | 968 | -0.095037 |
| singleton_offers_vs_base | both_wrong | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_ordinary1k | auxiliary_corrects | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_ordinary1k | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_ordinary1k | both_correct | 1689 | 607 | 53 | 71 | 958 | +0.017972 |
| singleton_offers_vs_ordinary1k | both_wrong | 0 | 0 | 0 | 0 | 0 | — |

### Development

| Group | Policy | Future condition | States | Exact sale | Different Joker | Other family | Sale NLL |
|---|---|---|---:|---:|---:|---:|---:|
| all_sales | base | all | 716 | 167 | 166 | 383 | 2.311783 |
| all_sales | ordinary1k | all | 716 | 170 | 173 | 373 | 2.394442 |
| all_sales | auxiliary | all | 716 | 158 | 170 | 388 | 2.406345 |
| multiple_offers | base | all | 469 | 125 | 114 | 230 | 2.110984 |
| multiple_offers | base | future_correct | 428 | 119 | 101 | 208 | 2.001606 |
| multiple_offers | base | future_wrong | 41 | 6 | 13 | 22 | 3.252783 |
| multiple_offers | ordinary1k | all | 469 | 123 | 125 | 221 | 2.206798 |
| multiple_offers | ordinary1k | future_correct | 426 | 116 | 111 | 199 | 2.134366 |
| multiple_offers | ordinary1k | future_wrong | 43 | 7 | 14 | 22 | 2.924387 |
| multiple_offers | auxiliary | all | 469 | 120 | 120 | 229 | 2.212852 |
| multiple_offers | auxiliary | future_correct | 441 | 116 | 113 | 212 | 2.146901 |
| multiple_offers | auxiliary | future_wrong | 28 | 4 | 7 | 17 | 3.251578 |
| other_sales | base | all | 42 | 2 | 5 | 35 | 4.286565 |
| other_sales | ordinary1k | all | 42 | 3 | 4 | 35 | 4.153390 |
| other_sales | auxiliary | all | 42 | 3 | 3 | 36 | 4.172667 |
| replacements | base | all | 674 | 165 | 161 | 348 | 2.188725 |
| replacements | base | future_correct | 633 | 159 | 148 | 326 | 2.119805 |
| replacements | base | future_wrong | 41 | 6 | 13 | 22 | 3.252783 |
| replacements | ordinary1k | all | 674 | 167 | 169 | 338 | 2.284834 |
| replacements | ordinary1k | future_correct | 631 | 160 | 155 | 316 | 2.241251 |
| replacements | ordinary1k | future_wrong | 43 | 7 | 14 | 22 | 2.924387 |
| replacements | auxiliary | all | 674 | 155 | 167 | 352 | 2.296278 |
| replacements | auxiliary | future_correct | 646 | 151 | 160 | 335 | 2.254872 |
| replacements | auxiliary | future_wrong | 28 | 4 | 7 | 17 | 3.251578 |
| singleton_offers | base | all | 205 | 40 | 47 | 118 | 2.366583 |
| singleton_offers | base | future_correct | 205 | 40 | 47 | 118 | 2.366583 |
| singleton_offers | base | future_wrong | 0 | 0 | 0 | 0 | — |
| singleton_offers | ordinary1k | all | 205 | 44 | 44 | 117 | 2.463365 |
| singleton_offers | ordinary1k | future_correct | 205 | 44 | 44 | 117 | 2.463365 |
| singleton_offers | ordinary1k | future_wrong | 0 | 0 | 0 | 0 | — |
| singleton_offers | auxiliary | all | 205 | 35 | 47 | 123 | 2.487141 |
| singleton_offers | auxiliary | future_correct | 205 | 35 | 47 | 123 | 2.487141 |
| singleton_offers | auxiliary | future_wrong | 0 | 0 | 0 | 0 | — |

### Future-choice transitions: development

| Group and control | Future transition | States | Both sale-exact | Gained sale-exact | Lost sale-exact | Neither sale-exact | Sale NLL difference |
|---|---|---:|---:|---:|---:|---:|---:|
| multiple_offers_vs_base | auxiliary_corrects | 13 | 1 | 0 | 2 | 10 | +0.029763 |
| multiple_offers_vs_base | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| multiple_offers_vs_base | both_correct | 428 | 91 | 24 | 28 | 285 | +0.106726 |
| multiple_offers_vs_base | both_wrong | 28 | 2 | 2 | 1 | 23 | +0.061084 |
| multiple_offers_vs_ordinary1k | auxiliary_corrects | 15 | 2 | 0 | 3 | 10 | +0.081846 |
| multiple_offers_vs_ordinary1k | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| multiple_offers_vs_ordinary1k | both_correct | 426 | 97 | 17 | 19 | 293 | +0.001949 |
| multiple_offers_vs_ordinary1k | both_wrong | 28 | 1 | 3 | 1 | 23 | +0.027896 |
| replacements_vs_base | auxiliary_corrects | 13 | 1 | 0 | 2 | 10 | +0.029763 |
| replacements_vs_base | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| replacements_vs_base | both_correct | 633 | 118 | 32 | 41 | 442 | +0.111205 |
| replacements_vs_base | both_wrong | 28 | 2 | 2 | 1 | 23 | +0.061084 |
| replacements_vs_ordinary1k | auxiliary_corrects | 15 | 2 | 0 | 3 | 10 | +0.081846 |
| replacements_vs_ordinary1k | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| replacements_vs_ordinary1k | both_correct | 631 | 127 | 22 | 33 | 449 | +0.009040 |
| replacements_vs_ordinary1k | both_wrong | 28 | 1 | 3 | 1 | 23 | +0.027896 |
| singleton_offers_vs_base | auxiliary_corrects | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_base | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_base | both_correct | 205 | 27 | 8 | 13 | 157 | +0.120558 |
| singleton_offers_vs_base | both_wrong | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_ordinary1k | auxiliary_corrects | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_ordinary1k | auxiliary_loses | 0 | 0 | 0 | 0 | 0 | — |
| singleton_offers_vs_ordinary1k | both_correct | 205 | 30 | 5 | 14 | 156 | +0.023776 |
| singleton_offers_vs_ordinary1k | both_wrong | 0 | 0 | 0 | 0 | 0 | — |

## Verification and limits

The whole saved-data process exited successfully in 3.267 seconds. All 684,558 prediction identities and future labels match the original arrays; 24 admitted aggregate controls reproduce. Independent array counts verify 114 categorical groups, including exhaustive future-choice transitions. No model, training, simulator or teacher was run. Source files match the archived runtime manifest.

These are teacher-state imitation measurements on the existing exposed development panel. They neither measure superior action value nor prove fresh-game or long-horizon generalization. The failed auxiliary endpoint remains ineligible for game integration or confirmation. The original fresh-win, greedy retention, early-attribution and final independent-fit requirements remain unchanged.

## Artifacts

- [result_v1.json](../../eval/consequence_prediction_20260927/replacement_aux_connection_v1/result_v1.json)
- [replacement_aux_connection_owner_terminal_v1.json](../../eval/consequence_prediction_20260927/replacement_aux_connection_owner_terminal_v1.json)
- [replacement_auxiliary_connection_diagnostic_20260929.md](replacement_auxiliary_connection_diagnostic_20260929.md)
- [replacement_auxiliary_results_20260929.md](replacement_auxiliary_results_20260929.md)
- [action_grammar.py](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/action_grammar.py)
- [entity_heads.py](../../eval/pretraining_transfer_20260923/candidate/src/pylatro_agent/entity_heads.py)
- [replacement_aux_fit_v3.py](../../eval/consequence_prediction_20260927/replacement_aux_fit_v3.py)
