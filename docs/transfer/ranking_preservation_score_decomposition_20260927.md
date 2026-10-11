# Where the learned corrections stop transferring

**The combined model moves action scores toward the better action on nearly every targeted training pair, but only a small majority of the corresponding held-out pairs.** Relative to the base, its preferred-action margin increases on 1,394/1,395 training cases where the teacher action clears and the base action does not. On the exposed held-out population, that happens on 131/233 cases. The resulting correct pair rankings are 1,053/1,395 in training and 34/233 held out. This narrows the existing correction-transfer gap; it does not yet identify its parameter cause.

These populations apply the same selection rule to different games: the comparison action is the base mode, and replaying it and the teacher action gives different immediate-clear outcomes. All 1,977 matched training pairs and 339 held-out pairs are retained in the data, including the 582 and 106 cases where the base action is better. The complete original 2,027-pair training population is also included. This is a descriptive follow-up using saved scores, with no new training, model evaluation, or gameplay.

For two legal play actions, the policy's log-probability difference is the sum of its internal reference margin and its learned subset correction margin. The common play-family probability and normalization cancel. **Internal reference means the evaluated model's trainable candidate/autoregressive distribution before the correction. It is not the frozen base policy used by the KL loss.**

The actual changes from base to combined, on teacher-better pairs, are:

| Score component | Training: positive changes / pairs | Held out: positive changes / pairs | Mean training change | Mean held-out change |
|---|---:|---:|---:|---:|
| Internal reference margin | 1281/1395 | 131/233 | +1.0024 | +0.0261 |
| Correction energy margin | 1383/1395 | 132/233 | +1.6359 | +0.0861 |
| Final log-probability margin | 1394/1395 | 131/233 | +2.6383 | +0.1122 |

Both terms therefore show substantially weaker transfer of the learned score changes. Their contributions add algebraically; they cannot be assigned to separate parameter causes because shared encoder and projection features affect both. The teacher-better population starts with zero correct base pair rankings by construction, so these figures do not measure continuation's retention of already correct base choices. The [complete action replay](ranking_preservation_development_20260927.md) measures that separately.

Inside the combined endpoint, the correction's effect also differs sharply between training and held-out corrections:

| Population | Pairs | Reference correct | Final correct | Repairs | Reversals |
|---|---:|---:|---:|---:|---:|
| Training, teacher better | 1395 | 513 | 1053 | 572 | 32 |
| Held out, teacher better | 233 | 62 | 34 | 16 | 44 |
| Training, base action better | 582 | 472 | 579 | 108 | 1 |
| Held out, base action better | 106 | 91 | 102 | 15 | 4 |
| Training, all matched pairs | 1977 | 985 | 1632 | 680 | 33 |
| Held out, all pairs | 339 | 153 | 136 | 31 | 48 |

A repair changes an incorrect internal reference ranking into a correct final ranking; a reversal does the opposite. On teacher-better training pairs, the correction repairs 572/882 reference errors (64.85%) and reverses 32/513 reference successes (6.24%). Held out, it repairs 16/171 errors (9.36%) and reverses 44/62 successes (70.97%). This takes teacher-better pair accuracy from 36.77% to 75.48% in training, versus 26.61% to 14.59% held out.

The mismatch appears in both periods. On early teacher-better training states, there are 203 repairs and 9 reversals; held out, 7 repairs and 15 reversals. Late training gives 369 repairs and 23 reversals; late held-out gives 9 repairs and 29 reversals. The combined model has no reference or final margins within 0.00001 of zero on either population, so these counts are not near-tie artifacts.

This pattern is not unique to the combined objective. All five policies are retained below, on the same teacher-better pairs:

| Policy | Training reference → final correct / 1395 | Held-out reference → final correct / 233 | Held-out repairs / reversals |
|---|---:|---:|---:|
| Base | 265 → 0 | 54 → 0 | 0 / 54 |
| Ordinary | 367 → 514 | 63 → 40 | 18 / 41 |
| Ranking | 582 → 1195 | 69 → 47 | 20 / 42 |
| Preservation | 315 → 358 | 60 → 27 | 13 / 46 |
| Combined | 513 → 1053 | 62 → 34 | 16 / 44 |

Higher average margins also need not produce more correct rankings. Relative to ordinary, combined improves the mean held-out teacher-better margin by 0.4007, yet gains only 6 correct rankings while losing 12. Relative to ranking, the mean increases by 0.1825, but it gains 1 correct ranking and loses 14. The complete exports keep all four endpoint comparisons, both preference directions, and every early/late stratum.

The selected-pair decomposition does **not** establish that deleting the correction head improves the policy. It excludes most actions and states, and a pairwise winner can still lose to a third global action. Nor does a correction reversal isolate changed head weights: the [earlier parameter intervention](correction_component_attribution_20260925.md) reproduced a fatal reversal with the old head and a changed encoder. The [full-panel rollback study](component_panel_20260925.md) recovered old matches while losing new ones, with every rollback still worse than the base overall. The evidence currently supports a correction-transfer and retention problem, not a demonstrated longer-context or world-model bottleneck.

The independent stdlib recount checked all 11,830 saved model-pair rows and reproduced 1,650 existing aggregate counts. Maximum score-decomposition error was 0.0000018552 against the existing 0.00001 tolerance. The endpoint-change recount reproduced 240 archived contrast counts/means and checked 69,784 stratum-pair delta identities, including overlapping strata; maximum error was 0.000002563 against 0.00002. Both exited successfully. These checks reuse the already completed native-probability audits; they do not constitute another fit, a new parameter intervention, or independent confirmation of generalization. The subsequently completed [autonomous evaluation](ranking_preservation_development_20260927.md) failed the development readiness screen, with no sampled point improvement and equal early-hybrid/control wins.

Evidence: [score decomposition data](data/ranking_preservation_score_decomposition_20260927.json), [actual endpoint score changes](data/ranking_preservation_score_changes_20260927.json), [decomposition recount](../../eval/ranking_preservation_20260927/recount_score_decomposition_v1.py), [endpoint-change recount](../../eval/ranking_preservation_20260927/recount_score_changes_v1.py), [actual decomposition exit](../../eval/ranking_preservation_20260927/score_decomposition_terminal_v1.json), [actual endpoint-change exit](../../eval/ranking_preservation_20260927/score_changes_terminal_v1.json).
