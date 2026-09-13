# Learning bottleneck investigation

**Status and interpretation**

This follow-up reserves nine additional MPS hours. PPO remains stopped. Original HF data, checkpoints, and the first transfer report are preserved. Memorization and offline agreement are diagnostics; gameplay improvements require independent confirmation.
The preceding investigation, including historical replay, matched teacher evaluation, critic removal, and student-state corrections, is retained in [the first report](report.md). Its candidate-coverage figures predate the mask correction below.

Accounted accelerator time: **8.302 / 9 hours**, including a two-minute overhead reserve.

| Accelerator stage | Hours used | Hours allocated |
| --- | ---: | ---: |
| diagnostics | 1.440 | 1.500 |
| training | 3.329 | 3.500 |
| development | 1.266 | 1.500 |
| confirmation | 2.267 | 2.500 |

Stage reallocations, if any, are retained in `budget_adjustments.json`; the total remains nine hours.

Confirmation status: **inconclusive**. Sampled candidate versus control: **-0.12 pp [-0.96, +0.67]**. Greedy: -0.62 pp [-1.62, +0.25].

Sampled differences for training seeds 0 and 1: -0.50 pp, +0.25 pp.

The sampled interval's upper bound is below the predeclared five-point improvement target for these continuation settings and two training replicates.

Supplementary comparison with the original checkpoint on 400 seeds: selected continuation sampled +0.25 pp [-0.75, +1.17]; greedy -0.12 pp [-2.88, +2.62]. Full outcomes and both continuation contrasts appear below.

**Cause and evidence**

| Finding | Evidence | Interpretation |
| --- | --- | --- |
| Hand and shop policies both limit gameplay | On the same CPU seed panel, student wins 6.67%; teacher hand replacement 22.67%; teacher shop replacement 23.33%; both 67.33%. Joint gain: +60.67 pp [+49.33, +72.00]. | Controlled phase replacements substantially improve wins. Neither alone closes the gap; this does not distinguish poor imitation from unavailable inputs. |
| Shop-sale decisions contribute to lost games | Replacing only teacher-prescribed shop Joker sales gives +8.00 pp [+2.67, +14.00] versus the sampled student. | Exploratory controlled evidence on 50 exposed seeds. The intervention includes sale timing and Joker selection, uses full-state teacher information, and changes later states. |
| Conditional Joker choice matters | With the teacher's sale-timing rule in both policies, student conditional Joker choice minus teacher choice gives -12.00 pp [-18.00, -6.67]. | Exploratory controlled evidence for the value of Joker selection in this setting. It does not prove that a particular architecture change will learn the teacher choice. |
| Coverage diagnostic defect | Valid candidates cover 37.91% of teacher plays and 33.11% of discards. The previous metric ignored invalid slots. | Confirmed metric defect; corrected with a failing/passing regression. |
| Fixed mixture probability cap | With candidates present and the teacher action absent, its conditional probability is at most 0.5 and NLL is at least log(2). | Mathematical limitation of the configured policy. Does not establish that alternative actions lose games. |
| Average teacher-probability ceiling on audited hands | Valid candidate coverage is 36.99%. With candidates present on every audited hand, the .5 mixture limits mean conditional teacher-action probability to at most 68.49%. | A probability ceiling on these labels, not a bound on greedy agreement or game wins. |
| Component specialization | AR conditional teacher probability is 8.86% on candidate-covered labels versus 49.49% on uncovered labels. Mean AR gradient responsibility on covered labels is 10.11%. | The pretrained AR component has not learned all covered labels well; changing only inference routing exposes this weakness. |
| Rare shop actions are poorly imitated | On 1,237 HF teacher Joker sales, exact agreement is 6.22% and teacher-action probability is 10.41%. | Aggregate shop accuracy masks this weakness. The sale-conditioned CPU intervention below tests its gameplay consequences; accuracy alone is not causal evidence. |
| Conflicting identical observations | 0 conflicts among 44,348 audited decisions. | Absence of exact collisions does not prove observation sufficiency. |
| Current-rule pipeline equivalence | 20 complete games and 3,910 decisions replayed through FastRunner and BalatroEnv, including saved observations and rewards. | No divergence in the selected balanced panel; this is bounded evidence, not a proof for all states. |
| Gradient flow | Component gradients and actual parameter updates recorded for backbone, candidate heads, AR heads, and critic. | A 32-state local diagnostic; not proof about the entire training history. |

**Bounded memorization**

Same 320 distinct HF observations: 160 plays and 160 discards. Full model, 300 updates, FP32, batch 320, LR 3e-4, 20-update warmup, no dropout, critic, entropy, weight decay, or outcome weighting. These observations are not held out.

| Initialization | AR mixture | Accuracy | Teacher probability | NLL | NLL above mixture floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| epoch7 | 0.5 | 80.62% | 60.91% | 0.5419 | 0.0697 |
| epoch7 | 1.0 | 100.00% | 99.97% | 0.0003 | 0.0003 |
| random | 0.5 | 78.75% | 48.68% | 0.8072 | 0.3350 |
| random | 1.0 | 100.00% | 99.91% | 0.0009 | 0.0009 |

Both pure-AR controls crossed the prespecified memorization thresholds: at least 99% accuracy and more than 95% teacher-action probability. The conditional free-logit/CPU-MPS failure investigation was therefore unnecessary. The model and optimizer can fit this small panel; this does not establish sufficient capacity or effective optimization on the full task.

The initial pretrained memorization runs retained attention dropout .1 while standard dropout was disabled. They are preserved under their original names, excluded from this controlled table, and superseded by the `dropout0_v2` runs. A regression verifies deterministic training-mode forwards with attention and standard dropout both disabled; gradient computation remains enabled.

Same-weights routing counterfactuals on the same memorization observations:

| Initialization / training mixture | Evaluation mixture | Accuracy | Teacher probability |
| --- | ---: | ---: | ---: |
| memorize_epoch7_mixture0.5_dropout0_v2 | 1.0 | 91.56% | 91.52% |
| memorize_epoch7_mixture1.0_dropout0_v2 | 0.5 | 98.44% | 59.05% |
| memorize_random_mixture0.5 | 1.0 | 79.69% | 78.34% |
| memorize_random_mixture1.0 | 0.5 | 100.00% | 51.68% |

For a hand-action family, the fixed mixture is `p(action) = (1-epsilon) * p_candidate(action) + epsilon * p_AR(action)`. A label outside the valid candidate set receives probability only from AR. For covered labels, the AR gradient responsibility is `epsilon * p_AR(label) / p(label)`, so a 50/50 mixture does not imply equal useful imitation gradients for both components. Greedy agreement can remain high while sampling frequently chooses other actions. These measurements establish the probability effect on the selected states; gameplay confirmation tests its practical impact.

**Fresh data and matched training**

Training seeds 4000000-4000999; offline validation 4100000-4100199; development 120000-120199; confirmation 130000-130399. Every panel is checked against prior training manifests; old evaluation panels remain excluded. Teacher settings: current pinned sources, Blue, White Stake, Ante 8, search shops, scaler growth enabled, no blind rollout, gamma .997, reward snapshot version 3.

Four arms cross mixture {0.5,1.0} with play/discard BC weight {1,2}. Both training seeds start from epoch 7 and use identical data order, 750 updates, physical batch 320, FP32, AdamW LR 3e-5, weight decay .01, 100-update warmup and cosine decay. Dropout .1, critic coefficients .5/.1, entropy .001, outcome weighting beta 1.5 and clipping [.25,4] stay fixed. Effective BC weights are renormalized.
The loader shuffles Parquet row groups and then decisions within each group to preserve read locality. These new shards contain one game each, so batches can contain relatively few distinct games. This ordering is held fixed across arms; broadly mixed minibatches are not tested here. The continuation length is explicitly 750 updates; inherited epoch-budget metadata is not used by this loop.

Intermediate offline curves use the same fixed panel of up to 16 decisions per game: 100 training games and all 200 validation games. Diagnostics preserve training RNG and mode. They are recorded at updates 0/250/500/750.

Training dataset: 202,141 decisions, 728/1000 teacher wins. This generation win rate is not a matched gameplay contrast.
Each arm processes 240,000 examples, 1.19 times the dataset's record count. This is a short continuation; flattening late in its cosine schedule does not establish convergence.
Validation dataset: 40,341 decisions, 143/200 teacher wins. This generation win rate is not a matched gameplay contrast.

CPU-only generation took 2.28 wall-clock hours; it is excluded from MPS accounting.

A lightweight-summary defect stopped the first continuation attempt before update 1: the summary expected candidate details omitted by the lightweight observer. A failing/passing regression fixed the summary. The zero-update attempt is preserved under `failures/`, its 10.9 seconds are charged, and the complete matched runs start again from the original checkpoint.

Fixed-panel learning curves (initial → latest completed diagnostic):

| Arm / replicate | Latest update | Training NLL | Validation NLL | Validation play accuracy | Validation discard accuracy | Validation teacher-play probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control_seed0 | 750 | 0.911 → 0.889 | 0.856 → 0.847 | 42.96% → 44.83% | 30.87% → 29.53% | 29.24% → 29.63% |
| control_seed1 | 750 | 0.911 → 0.888 | 0.856 → 0.843 | 42.96% → 45.21% | 30.87% → 28.86% | 29.24% → 29.64% |
| hand_weight_seed0 | 750 | 0.911 → 0.887 | 0.856 → 0.848 | 42.96% → 45.33% | 30.87% → 28.86% | 29.24% → 29.74% |
| hand_weight_seed1 | 750 | 0.911 → 0.886 | 0.856 → 0.843 | 42.96% → 45.21% | 30.87% → 28.86% | 29.24% → 29.74% |
| pure_ar_seed0 | 750 | 1.697 → 1.081 | 1.604 → 1.034 | 41.97% → 50.06% | 36.24% → 36.24% | 29.38% → 26.31% |
| pure_ar_seed1 | 750 | 1.697 → 1.082 | 1.604 → 1.029 | 41.97% → 50.68% | 36.24% → 35.57% | 29.38% → 26.30% |
| combined_seed0 | 750 | 1.697 → 1.076 | 1.604 → 1.032 | 41.97% → 49.81% | 36.24% → 34.90% | 29.38% → 26.67% |
| combined_seed1 | 750 | 1.697 → 1.078 | 1.604 → 1.027 | 41.97% → 50.19% | 36.24% → 35.57% | 29.38% → 26.77% |

NLL averages negative log teacher-action probability; the probability column averages the probability itself. Reducing a few extremely low probabilities can improve NLL while reducing the arithmetic mean probability. Greedy agreement only requires the teacher action to rank first. Neither metric alone establishes that sampled actions or complete games improved.


**Every-decision offline validation**

All decisions in the 200 fresh validation games, with no training on those seeds. Accuracy and NLL below are decision-weighted; the structured summaries also retain game-weighted results and action-family/ante/outcome strata. Differences between mixtures include their probability parameterization.

| Policy | Overall accuracy | Play accuracy / NLL | Discard accuracy / NLL | Hand teacher probability |
| --- | ---: | ---: | ---: | ---: |
| combined_seed0 | 77.60% | 50.84% / 2.600 | 39.53% / 3.276 | 26.34% |
| combined_seed1 | 77.71% | 51.22% / 2.599 | 39.48% / 3.266 | 26.37% |
| control_seed0 | 76.62% | 48.00% / 1.803 | 33.13% / 3.215 | 28.52% |
| control_seed1 | 76.53% | 47.59% / 1.803 | 33.32% / 3.205 | 28.50% |
| hand_weight_seed0 | 76.58% | 47.96% / 1.799 | 33.13% / 3.213 | 28.63% |
| hand_weight_seed1 | 76.56% | 47.71% / 1.800 | 33.36% / 3.202 | 28.62% |
| original_m0.5 | 76.02% | 45.49% / 1.835 | 34.79% / 3.213 | 28.18% |
| original_m1.0 | 75.34% | 41.66% / 5.113 | 38.91% / 3.291 | 28.68% |
| pure_ar_seed0 | 77.55% | 50.50% / 2.618 | 39.48% / 3.280 | 25.99% |
| pure_ar_seed1 | 77.65% | 50.95% / 2.614 | 39.29% / 3.271 | 25.96% |

Teacher-action probability on hand decisions, split by valid candidate coverage:

| Policy | Covered labels | Uncovered labels |
| --- | ---: | ---: |
| combined_seed0 | 16.52% | 32.54% |
| combined_seed1 | 17.02% | 32.27% |
| control_seed0 | 39.33% | 21.69% |
| control_seed1 | 39.28% | 21.68% |
| hand_weight_seed0 | 39.43% | 21.80% |
| hand_weight_seed1 | 39.40% | 21.81% |
| original_m0.5 | 39.19% | 21.22% |
| original_m1.0 | 6.88% | 42.45% |
| pure_ar_seed0 | 16.12% | 32.22% |
| pure_ar_seed1 | 16.59% | 31.87% |

Relative to the original checkpoint using autoregressive routing, the two pure-AR continuations raise mean teacher-action probability on covered hand labels from 6.88% to 16.36%, while lowering it on uncovered labels from 42.45% to 32.05%. This is a redistribution of prediction quality across label groups; the reduction in large NLL errors does not imply uniformly sharper teacher-action probabilities. The sampled-game comparisons determine its gameplay value.

Shop action-family agreement on the same complete validation panel. The sale family can also occur outside the shop phase:

| Policy | Buy | Leave | Reroll | Sell Joker | Teacher-sale probability |
| --- | ---: | ---: | ---: | ---: | ---: |
| combined_seed0 | 93.32% | 94.04% | 94.80% | 7.08% | 10.35% |
| combined_seed1 | 93.57% | 94.21% | 94.14% | 7.08% | 10.62% |
| control_seed0 | 93.48% | 94.02% | 94.83% | 7.00% | 10.32% |
| control_seed1 | 93.49% | 94.42% | 94.22% | 7.08% | 10.73% |
| hand_weight_seed0 | 93.38% | 94.07% | 94.86% | 6.60% | 10.33% |
| hand_weight_seed1 | 93.61% | 94.47% | 94.00% | 7.08% | 10.65% |
| original_m0.5 | 93.35% | 94.70% | 94.22% | 5.42% | 9.94% |
| original_m1.0 | 93.35% | 94.70% | 94.22% | 5.50% | 9.94% |
| pure_ar_seed0 | 93.43% | 94.04% | 94.83% | 7.15% | 10.34% |
| pure_ar_seed1 | 93.54% | 94.33% | 94.16% | 7.31% | 10.70% |

These are exact teacher-action matches. Different Joker sales can be equally useful; weak sale agreement alone does not establish a causal contribution to lost games.

**Gameplay comparisons**

Primary comparisons use temperature 1.0, Blue Deck, White Stake, Ante 8, batch 32, four workers, and sampling seeds 0/1/2. The inference-only temperature-.5 sweep is separate. Sampling and training repeats are averaged within game seed before 10,000 paired bootstrap resamples. Two training replicates do not estimate uncertainty across a broad population of training runs. A win advances the engine to ante 9, so mean final ante can exceed 8.

**Inference-only screen**

| Treatment versus control | Sampled difference | Greedy difference |
| --- | ---: | ---: |
| inference_m0.5_t1.0 | +0.00 pp [+0.00, +0.00] | +0.00 pp [+0.00, +0.00] |
| inference_m0.5_t0.5 | +0.00 pp [-4.00, +4.00] | -4.00 pp [-12.00, +4.00] |
| inference_m1.0_t1.0 | -0.67 pp [-4.67, +3.33] | -4.00 pp [-16.00, +8.00] |
| inference_m1.0_t0.5 | +4.00 pp [-2.00, +10.67] | -4.00 pp [-14.00, +8.00] |

**Training screen**

| Treatment versus control | Sampled difference | Greedy difference |
| --- | ---: | ---: |
| hand_weight | +0.67 pp [-1.67, +3.33] | +1.00 pp [-3.00, +5.00] |
| pure_ar | -2.33 pp [-4.67, -0.33] | +0.00 pp [-6.00, +6.00] |
| combined | -1.33 pp [-3.67, +1.00] | -3.00 pp [-8.00, +2.00] |

development_comparison: sampled -0.67 pp [-1.83, +0.42]; greedy +2.50 pp [+0.25, +4.75].

**Confirmation outcomes**

| Policy / replicate / mode | Wins / games | Win rate [cluster 95% CI] | Mean final ante | Stalls |
| --- | ---: | ---: | ---: | ---: |
| control/seed0/sampled | 33/1200 | 2.75% [1.75%, 3.75%] | 3.587 | 0 |
| control/seed0/greedy | 31/400 | 7.75% [5.25%, 10.50%] | 4.838 | 0 |
| control/seed1/sampled | 28/1200 | 2.33% [1.42%, 3.33%] | 3.539 | 0 |
| control/seed1/greedy | 43/400 | 10.75% [7.75%, 14.00%] | 4.950 | 0 |
| hand_weight/seed0/sampled | 27/1200 | 2.25% [1.42%, 3.17%] | 3.499 | 0 |
| hand_weight/seed0/greedy | 28/400 | 7.00% [4.50%, 9.50%] | 4.775 | 0 |
| hand_weight/seed1/sampled | 31/1200 | 2.58% [1.67%, 3.67%] | 3.620 | 0 |
| hand_weight/seed1/greedy | 41/400 | 10.25% [7.50%, 13.25%] | 4.985 | 0 |

Ante-clear rates use all games in each row as their denominator:

| Policy / replicate / mode | Ante 1 | Ante 2 | Ante 3 | Ante 4 | Ante 5 | Ante 6 | Ante 7 | Ante 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control/seed0/sampled | 79.92% | 65.00% | 49.58% | 32.00% | 16.42% | 8.42% | 4.67% | 2.75% |
| control/seed0/greedy | 94.25% | 85.00% | 74.00% | 52.75% | 36.50% | 21.50% | 12.00% | 7.75% |
| control/seed1/sampled | 78.17% | 63.50% | 49.92% | 31.50% | 16.58% | 7.92% | 4.00% | 2.33% |
| control/seed1/greedy | 94.50% | 86.75% | 74.50% | 54.00% | 37.00% | 23.50% | 14.00% | 10.75% |
| hand_weight/seed0/sampled | 78.50% | 63.42% | 48.25% | 30.83% | 15.58% | 7.33% | 3.75% | 2.25% |
| hand_weight/seed0/greedy | 93.50% | 85.00% | 73.75% | 51.25% | 35.00% | 20.75% | 11.25% | 7.00% |
| hand_weight/seed1/sampled | 80.50% | 66.42% | 49.67% | 32.58% | 17.08% | 8.42% | 4.75% | 2.58% |
| hand_weight/seed1/greedy | 94.50% | 88.00% | 75.00% | 54.50% | 37.50% | 23.75% | 15.00% | 10.25% |


**Supplementary original-reference comparison**

On 400 confirmation seeds, the original checkpoint uses mixture .5, temperature 1, three sampling repeats, and greedy evaluation. The training replicates share that original baseline; the bootstrap unit remains the game seed. This supplementary comparison and its time-only choice of 200 or 400 seeds were planned during development and do not alter primary acceptance.

| Continuation versus original reference | Sampled difference | Greedy difference |
| --- | ---: | ---: |
| hand_weight | +0.25 pp [-0.75, +1.17] | -0.12 pp [-2.88, +2.62] |
| control | +0.38 pp [-0.67, +1.38] | +0.50 pp [-2.12, +3.12] |

| Original reference mode | Wins / games | Win rate [cluster 95% CI] | Mean final ante |
| --- | ---: | ---: | ---: |
| sampled | 26/1200 | 2.17% [1.33%, 3.08%] | 3.612 |
| greedy | 35/400 | 8.75% [6.00%, 11.50%] | 4.798 |

**Exploratory phase replacements (CPU only)**

Original epoch 7, Blue/White/Ante 8, mixture .5, temperature 1.0, CPU batch 1, 50 separate seeds 5000000-5000049 and sampling repeats 0/1/2. The full teacher runs once per seed. Each hybrid draws the student proposal before replacing it. Hand replacement covers every CHOOSE_ACTION decision, including consumables and reordering; shop replacement covers SHOP, while booster packs remain with the student. Compare arms within this experiment, since its batching and per-game sampling streams differ from the primary MPS evaluations.

| Controller | Wins / games | Win rate [cluster 95% CI] | Mean final ante | Paired gain over student |
| --- | ---: | ---: | ---: | ---: |
| student | 10/150 | 6.67% [2.00%, 12.67%] | 3.980 | reference |
| hand | 34/150 | 22.67% [15.33%, 30.67%] | 6.267 | +16.00 pp [+9.33, +23.33] |
| shop | 35/150 | 23.33% [15.33%, 32.00%] | 5.220 | +16.67 pp [+9.33, +24.67] |
| hand_shop | 101/150 | 67.33% [56.00%, 78.67%] | 8.147 | +60.67 pp [+49.33, +72.00] |
| teacher_all | 36/50 | 72.00% [60.00%, 84.00%] | 8.380 | +65.33 pp [+52.67, +77.33] |

Exploratory conditional contrasts on the same games: adding teacher shop decisions when hands already use the teacher yields +44.67 pp [+34.00, +55.33]; adding teacher hand decisions when shops already use the teacher yields +44.00 pp [+34.00, +54.00]. Both phases versus full teacher: -4.67 pp [-15.33, +6.00].

Teacher queries use isolated copies and check that the live state and RNG stay unchanged. The teacher sees full simulator state, so a successful phase replacement cannot separate poor imitation from omitted observation inputs. Replacing a policy also changes later visited states. ShopSearch uses public, seed-independent deck samples; exact hand scoring can use copied run RNG state. The contribution of that stochastic information was not isolated. These interventions localize gameplay bottlenecks; they do not validate a learned candidate or supply training labels. All traces and canonical-order statistics are preserved.

CPU diagnostic wall time: 1.01 hours; zero accelerator workload. These seeds remain excluded from training.

**Exploratory Joker-sale follow-up**

The HF audit found 6.22% exact agreement on 1,237 teacher Joker sales, compared with about 94% on buying, leaving, and rerolling. This motivated a post-hoc CPU experiment on the same 50 exploratory seeds and three sampling repeats. In each shop state, an isolated teacher proposal determines which intervention applies: replace only teacher-prescribed Joker sales, or replace only other teacher shop actions. All other decisions use the sampled student. The student proposal is drawn before replacement. These interventions do not select a model.

The sale family spans phases: 1,059 of the audited sales occur in SHOP (5.85% agreement), 148 in booster packs, and 30 during hand play. Phase fields were joined from the prior audit only after all 44,348 decision identifiers, action families, accuracies, and NLLs matched exactly. The intervention below is restricted to SHOP.

| Teacher replacements | Wins / 150 | Paired gain over student | Difference from full shop oracle |
| --- | ---: | ---: | ---: |
| teacher_sale | 22 / 150 | +8.00 pp [+2.67, +14.00] | -8.67 pp [-15.33, -2.67] |
| teacher_other_shop | 16 / 150 | +4.00 pp [-0.67, +9.33] | -12.67 pp [-19.33, -6.67] |

The partitions depend on the teacher proposal at each visited state; their win-rate effects are not additive. Teacher-sale replacement does not suppress a student sale when the teacher would take a different action. The teacher retains full-state information. This is an exploratory follow-up with exposed seeds, not independent confirmation or a learned improvement.

Additional CPU wall time: 0.37 hours; zero MPS workload.

**Where sale probability is lost**

A CPU pass over all 1,237 audited sale labels decomposes the complete-action probability into choosing the sale family and choosing the Joker slot. On the 1,059 SHOP labels, mean sale-family probability is 48.32%; conditional teacher-Joker probability is 21.36%, versus 20.09% for a uniform legal-Joker choice. The resulting mean complete teacher-action probability is 10.17%. Forced-family Joker accuracy is 28.33%.

The current sale head predicts fixed slot logits from the pooled state; buying scores individual item-token embeddings. A Joker-token-conditioned sale head is therefore a specific architecture hypothesis, not an established fix or evidence that v13 observations are insufficient. CPU/MPS NLL differences on these labels are at most 5.7e-06; teacher-agreement indicators match.

**Sale timing versus Joker choice**

A further post-hoc CPU panel uses the teacher's SHOP sale-timing rule but samples which Joker to sell from the student's conditional head. An independent target-selection RNG leaves the main student sampling stream unchanged. The control is the completed teacher-sale intervention with both teacher timing and teacher Joker choice, on the same 50 seeds and three repeats.

Teacher timing with student Joker choice wins **4/150** (2.67%). Paired difference versus the original student: **-4.00 pp [-8.00, -0.67]**. Student Joker choice minus teacher Joker choice, with both using the teacher's sale-timing rule: **-12.00 pp [-18.00, -6.67]**.

This tests conditional Joker choice within a teacher-timing policy. It does not isolate a specific observation input; changing a sale also changes subsequent states. A confidence interval containing zero would not establish equivalence between the choice policies. The exploratory intervals are not adjusted for the sequence of follow-up comparisons.

**Verification**

114 distinct relevant regression tests passed across the recorded suites. Coverage includes deck propagation, checkpoint/schema restoration, observations and replay, policy probabilities and legality, snapshot/RNG isolation, label provenance, paired statistics, budget accounting, phase intervention boundaries, and the confirmed diagnostic defects.

The acceptance helper previously relied on the artifact verifier to reject stalled or censored games. A failing/passing regression now enforces rejection in the helper itself. The healthy comparison across 800 recorded outcomes is unchanged. This is a verification fix and does not change policy behavior or explain poor gameplay.

Final artifact verification passed: **True**. It checks 40 preserved original artifacts, generated data, matched training configurations and sources, raw game panels, phase traces, paired acceptance, and the nine-hour budget.

**Acceptance, limitations, and artifacts**

Validation requires a sampled gain of at least five percentage points, paired 95% interval above zero, positive sampled gains in both training replicates, greedy interval ruling out a regression beyond five points, and no new legality/nonfinite/infrastructure failures. Lower loss alone never validates gameplay.

The training factorial isolates mixture and hand weighting on shared current-rule data; it does not isolate data regeneration itself. Memorizing 320 states does not establish capacity or convergence on the full task. HF observations retain historical-rule limitations. `macro_accuracy` measures the family of the joint greedy action; `marginal_macro_accuracy`, where present, measures the macro head's own argmax.

Full artifacts and reproducible per-game results: `eval/learning_bottleneck_20260913`. Commands: `tools/learning_bottleneck.py --help`, `tools/run_learning_bottleneck.py --help`, and `tools/report_learning_bottleneck.py --require-complete`. PPO remains stopped.

**Next training recommendation**

No continuation in this study has met every validation criterion. Preserve the original checkpoint as the reference; do not promote a trial based on its lower imitation loss.

Compare the current pooled-slot Joker-sale head against a residual score conditioned on each Joker token and the global state. Initialize the added output to zero so initial policy probabilities match, and record the architecture variant explicitly in checkpoint metadata.
Use pinned current-rule records, unchanged v13 observations, two matched training seeds, identical data ordering, objectives, optimizer settings, and update counts. Keep the rest of the policy trainable under the same settings in both arms.

Track conditional Joker-selection probability and sale decisions on all visited states, including excess sales when the teacher would choose another action. Evaluate sampled end-to-end games as the primary outcome; teacher-hand-assisted games can diagnose the shop contribution. Reserve new confirmation seeds because this study's panels are exposed.

The rationale is the controlled shop-sale gain, the timing-versus-choice intervention, and the near-uniform conditional Joker choice, together with the existing difference between pooled sale logits and token-based buy logits. This proposes a test of a specific architecture hypothesis. The present experiments do not establish convergence on the full dataset, identify an indispensable missing feature, or justify starting PPO.
