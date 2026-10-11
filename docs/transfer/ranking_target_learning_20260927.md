# Ranking targets were largely learned; gameplay transfer remains unproven

The ranking continuations learned most of their intended training comparisons: **89.49% and 91.02%** rank the preferred action above its measured alternative, versus **45.73% and 44.55%** for matched ordinary continuation. The objective was effective on this training population. This does not resolve the transfer gap: the completed [fresh-game study](counterfactual_ranking_results_20260927.md) failed all five generalization criteria, including the early-decision test.

This is a read-only census of all **2,027 frozen preference pairs at 2,022 training states**, across the base and four existing continuation endpoints. No model was trained, no new game was played, and no checkpoint was selected. “Preferred” means the action cleared the current blind when the comparison action did not; it does not mean a higher eventual game-win value.

| Model | Preferred action outranks comparison | Preferred action is global mode | Mean pair loss |
|---|---:|---:|---:|
| Base | 632/2027 (31.18%) | 619/2027 (30.54%) | 1.6377 |
| Ordinary 0 | 927/2027 (45.73%) | 779/2027 (38.43%) | 1.3016 |
| Ordinary 1 | 903/2027 (44.55%) | 771/2027 (38.04%) | 1.3096 |
| Ranking 0 | 1814/2027 (89.49%) | 1485/2027 (73.26%) | 0.3148 |
| Ranking 1 | 1845/2027 (91.02%) | 1519/2027 (74.94%) | 0.2844 |

The comparison margin is log p(preferred) − log p(other); the loss is softplus(−margin). All counts and percentages in the tables weight pairs equally. The structured data also retain state-weighted results, averaging comparisons within each state as the training loss does. The five extra pairs do not change the conclusion. These are exact measurements of the fixed training population, not estimates of general performance.

| Model | Teacher preferred | Alternative preferred | Early, Ante < 4 | Fatal alternative |
|---|---:|---:|---:|---:|
| Base | 50/1445 (3.46%) | 582/582 (100.00%) | 220/680 (32.35%) | 1/34 |
| Ordinary 0 | 564/1445 (39.03%) | 363/582 (62.37%) | 299/680 (43.97%) | 11/34 |
| Ordinary 1 | 565/1445 (39.10%) | 338/582 (58.08%) | 278/680 (40.88%) | 14/34 |
| Ranking 0 | 1245/1445 (86.16%) | 569/582 (97.77%) | 601/680 (88.38%) | 30/34 |
| Ranking 1 | 1265/1445 (87.54%) | 580/582 (99.66%) | 601/680 (88.38%) | 31/34 |

The teacher-preferred and alternative-preferred split matters. Ordinary teacher imitation improves some teacher-preferred comparisons but loses 219 / 244 of the 582 alternative-preferred rankings that the base already got right. Ranking continuation retains 569 / 580 of those 582 rankings and learns far more teacher-preferred comparisons. Relative to its matched ordinary model, ranking corrects 888 / 942 pairwise errors while introducing 1 / 0 new pairwise errors on this population.

However, learning a comparison does not ensure the preferred action becomes the model's overall choice. Ranking chooses a third action in **409 / 402 comparisons (20.18% / 19.83%)**. Those third choices were not scored by this diagnostic; they may be better or worse. Among the 34 comparisons where the measured alternative immediately loses the game, ranking orders 30 / 31 pairs correctly but chooses the known clearing action in only **22 / 25 states**. It still chooses the known fatal alternative in **2 / 1 states** and chooses a third action in **10 / 8**. Neither ranking counts nor third-action counts establish how many losses its actual choices cause.

The remaining gaps are now more specific. Pairwise fitting is incomplete even on these training states, and pairwise success differs from choosing a clearing action among all legal actions. Strong improvements on the 680 early training comparisons also failed to establish a benefit in the separate early-game evaluation. We have not measured whether the learned rankings transfer to comparable states outside training, or whether new third choices preserve the intended immediate benefit. This diagnostic does not identify either mechanism as the cause of the gameplay result.

The target population was constructed largely from disagreements between the base and teacher, plus additional structural alternatives. Its low base accuracy is therefore selected by construction; it is not an unbiased error rate. The reference-score versus residual-energy decomposition is retained in the data, but is an output decomposition rather than an intervention that isolates encoder or head causality. All labels are immediate-clear preferences. World-model work remains deferred.

Next, measure preference transfer on a fixed population of already-exposed held-out states and replay the actual third choices where their consequences are unknown. These are diagnostics using existing checkpoints. Further exploratory changes follow [one fit per change, with final confirmation after candidate freeze](experiment_workflow.md).

Independent numerical verification passed. It reconstructed selected action probabilities and all 10,135 comparison margins in float64 from the complete legal-play reference scores, residual energies and macro logits. All **10,110 archived global modes matched exactly**. Maximum archived teacher-NLL error was 5.2452087e-06; reconstructed log-probability error was 1.1841482e-06; margin error was 1.2144446e-06, each below the existing 1e−5 bound. No pair was within that bound of a tie. Checkpoint, cache and archived-source hashes remained unchanged, and all model state tensors were unchanged after inference.

Measurement and audit both have actual successful direct-parent exit receipts; the serial supervisor also exited successfully. The [structured results](data/ranking_target_learning_20260927.json) retain all models, ten strata, six paired model contrasts, both weighting schemes, numerical checks and raw-pair pins. Further artifacts: [frozen protocol](../../eval/ranking_target_learning_20260927/protocol_v1.json), [independent audit](../../eval/ranking_target_learning_20260927/audit_result_v1.json), [completed execution](../../eval/ranking_target_learning_20260927/complete_v1.json), and [handoff](../../eval/ranking_target_learning_20260927/RESUME.md).
