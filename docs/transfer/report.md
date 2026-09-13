Gameplay result: **inconclusive**. Candidate arm: `actor_only`. Sampled wins: 62/2400 (2.58%) versus 73/2400 (3.04%) for the matched control. Confirmation-panel sampled difference versus matched continuation control: **-0.46 pp [-1.38, +0.46]** (paired 95% interval).

The original HF dataset, ten checkpoints, and previous logs are preserved. PPO was not started. The reference is Blue Deck, White Stake, Ante 8, mixture 0.5, temperature 1.0, FP32, batch 32, four CPU engine workers, and MPS policy inference. Wins finish at engine ante 9, so mean final ante can exceed 8.

Matched V68 teacher: **143/200 (71.50%)**, Wilson 95% interval [64.88%, 77.30%], 0 stalls. Search shops and scaler growth are explicit; blind rollouts are off. The dataset rate is 3,728/5,000 (74.56%) on different seeds and simulator sources.

**Cause and evidence assessment**

| Candidate cause | Evidence | What is established |
| --- | --- | --- |
| Simulator drift | Current replay diverges in all 20 selected HF games; reconstructed historical sources match all 3,909 decisions. | Saved actions are not reproducible under current rules. The current wrappers agree on observations and state fields for 476 common prefix steps. Effect on policy wins requires matched historical/current evaluation below. |
| Reward-input defects | Fast generation omitted boss identity and shop-event reward fields, and computed action diagnostics after the action. A purchase in seed 2015075 lost a 0.4 Joker-upgrade reward. | Corrected for future generation with regression tests and matching rewards on all 476 legal prefix steps. New records use reward snapshot version 3. Original HF targets remain fixed in the trials; these fixes do not change the evaluated policy's observations or actions. |
| Evaluation configuration | Old CLI omitted deck/mixture controls, silently fell back from malformed metadata, and offered a legacy teacher. | Measurement defects fixed. These are not established causes of the previously reported custom Blue/.5 evaluation. |
| Imperfect imitation | Training-state action metrics below. | Critical hand choices are much less accurate than blind/shop choices. Exact teacher disagreement alone is not proof of a bad action. |
| State distribution | Same selected checkpoint evaluated on HF teacher states and current student states; correction trial below. | Agreement difference is descriptive. Historical/current rule differences also affect these panels and cannot be separated by agreement alone. |
| Missing/quantized inputs | Candidate score tokens are structural proxies; card permanent bonuses and Joker features are quantized; RNG state is absent. | No observation feature was changed. Exact-score action interventions test an oracle action rule, not a trained feature migration. |
| Critic-loss continuation | Development sampled difference versus matched continuation control: -0.08 pp [-1.33, +1.17]. | Tests removing critic losses for 1,000 continuation updates; it does not isolate their effect during original pretraining. Confirmation is reported separately below. |
| Student-state correction replacement | Development sampled difference versus matched continuation control: -1.58 pp [-2.67, -0.58]. | Tests replacing half the HF examples with the finite correction set; label quality, HF exposure, and reuse can interact. Confirmation is reported separately below. |

**Recorded-action divergence examples**

| HF seed | First divergent step | Field/index | Saved → reconstructed | First illegal recorded-action step |
| ---: | ---: | --- | --- | ---: |
| 2035106 | 5 | tokens[9, 0] | 3 → 2 | 5 |
| 2045118 | 40 | tokens[10, 0] | 11 → 8 | 41 |
| 2035067 | 19 | scalars[8] | 5.1984968185424805 → 5.420535087585449 | 22 |

Source provenance identifies scoring/boss-rule changes in `bf3a6e5` and pack handling changes in `81b4afd` after data generation. Historical reconstruction verifies each restored file against its generation SHA-256; per-file matching revisions are retained in structured results. The exact historical replay establishes source drift. A current-prefix replay ending at an illegal saved action does not establish full-episode reward parity.

**What epoch 10 learned on 200 training games**

Uniform seed sampling without replacement used NumPy RNG 1729. All 44,348 decisions were retained (147 winning games, 53 losses). These are training-data diagnostics, not held-out accuracy.

| Decisions | Count | Accuracy | NLL | Teacher-action probability | Candidate coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 44348 | 79.56% | 0.6847 | 72.44% | 40.46% |
| family/play_subset | 9336 | 50.27% | 1.6359 | 33.24% | 40.36% |
| family/discard_subset | 2220 | 40.18% | 2.7232 | 24.23% | 40.86% |
| family/shop_buy | 8192 | 94.40% | 0.1810 | 89.12% | n/a |
| family/shop_leave | 4284 | 94.47% | 0.1824 | 88.87% | n/a |
| family/pack_claim | 4117 | 83.56% | 0.3918 | 79.32% | n/a |
| family/pack_skip | 5298 | 98.85% | 0.0528 | 96.52% | n/a |

Game-weighted accuracy is 78.85%; NLL is 0.7234. No teacher labels were unreachable under mixture .5. Candidate coverage only counts play/discard labels; the autoregressive component supplies support outside the candidate list. Phase, ante, outcome, family confusion, and per-seed/step rows are retained in the diagnostic artifacts.

**Checkpoint screen: first 50 development seeds**

| Epoch | Sampled wins / 150 | Sampled win rate | Greedy wins / 50 |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0.00% | 1 |
| 2 | 3 | 2.00% | 1 |
| 3 | 4 | 2.67% | 2 |
| 4 | 0 | 0.00% | 4 |
| 5 | 3 | 2.00% | 7 |
| 6 | 2 | 1.33% | 4 |
| 7 | 4 | 2.67% | 4 |
| 8 | 3 | 2.00% | 5 |
| 9 | 2 | 1.33% | 3 |
| 10 | 3 | 2.00% | 3 |

**Full development expansion**

| Epoch | Sampled wins / 600 | Sampled win rate | Greedy wins / 200 |
| ---: | ---: | ---: | ---: |
| 7 | 16 | 2.67% | 13 |
| 3 | 2 | 0.33% | 5 |

Epoch 7 was selected by full-panel sampled win rate, with mean final ante as the declared tiebreaker.

**Epoch-10 deck/mixture control: first 100 development seeds**

| Deck | Mixture | Sampled wins / 300 | Greedy wins / 100 |
| --- | ---: | ---: | ---: |
| b_red | 0.1 | 5 | 2 |
| b_red | 0.5 | 4 | 4 |
| b_blue | 0.1 | 8 | 5 |
| b_blue | 0.5 | 6 | 8 |

**Previously reported seed-panel regression:** greedy 8/50; sampled 0/25 with sampling seed 31013. This rerun uses the original 32-step idle limit and the 10000-series seeds. The previous custom evaluator reported 8/50 and 0/25 respectively.

On the matched development panel, selected student minus V68 teacher sampled difference is -68.83 pp [-75.00, -62.50]. The deterministic teacher outcome is reused only to align sampling repeats; the bootstrap still resamples 200 game seeds.

The teacher's mean final ante is 8.230; the selected sampled student's is 3.737. Student stalls: 0; censored games: 0/600 (teacher: 0/200). Ante-clear denominators are all games in each panel.

| Ante cleared | V68 teacher | Epoch 7 sampled |
| ---: | ---: | ---: |
| 1 | 99.50% | 82.67% |
| 2 | 98.00% | 65.00% |
| 3 | 97.50% | 52.50% |
| 4 | 96.00% | 36.00% |
| 5 | 93.00% | 19.67% |
| 6 | 88.00% | 9.50% |
| 7 | 79.50% | 5.67% |
| 8 | 71.50% | 2.67% |

**Autonomous play on the 200 HF demonstration seeds:** sampled 16/600 (2.67%), greedy 26/200 (13.00%). These are fresh resets under current rules on training-game seeds, not recorded-action replay. Compare the fresh-seed Blue/.5 epoch-10 factorial above; the seed panels differ, so this comparison is unpaired.

**Selected checkpoint agreement by visited-state source**

| States | Decisions | Games | Decision accuracy | Game accuracy | Decision NLL | Game NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HF teacher | 44348 | 200 | 78.65% | 77.91% | 0.7267 | 0.7662 |
| Sampled student | 9284 | 100 | 66.54% | 67.59% | 1.2757 | 1.3314 |

The aggregate agreement drop partly reflects a different mix of decisions. Reweighting common teacher-action families to HF frequencies gives 78.65% HF accuracy versus 75.79% student-state accuracy (HF coverage 99.94%, student coverage 100.00%). This is descriptive standardization; ante, build state, and rule differences remain.

| Teacher-action family | HF-state accuracy | Student-state accuracy |
| --- | ---: | ---: |
| play_subset | 47.74% | 53.02% |
| discard_subset | 38.92% | 27.34% |
| shop_buy | 93.87% | 88.32% |
| shop_leave | 94.40% | 93.84% |
| pack_claim | 83.94% | 81.25% |
| pack_skip | 98.00% | 90.91% |

**Simulator-rule intervention, identical epoch 10 and 100 development seeds:** current rules 2.00%; historical rules 2.67%; current minus historical = -0.67 pp [-2.33, +1.00]. This changes the simulator rules while holding policy and evaluation settings fixed; it is not a proposal to restore incorrect rules in production.

The matched greedy current-minus-historical difference is -4.00 pp [-9.00, +0.00].

Within b_blue, mixture .5 minus .1 sampled difference: -0.67 pp [-3.00, +1.67].

Within b_red, mixture .5 minus .1 sampled difference: -0.33 pp [-2.67, +2.00].

At mixture 0.1, Blue minus Red sampled difference: +1.00 pp [-1.00, +3.33].

At mixture 0.5, Blue minus Red sampled difference: +0.67 pp [-1.67, +3.00].

**One-action interventions on student disagreements**

One disagreement per correction game is reservoir-sampled using RNG 1729. Both branches start from the same snapshot and use the same continuation sampling seed at batch 1. Selection is uniform within each game, not uniform over all decisions across games.

| Intervention | Games | Win difference | Blind-survival difference | Final-ante difference |
| --- | ---: | ---: | ---: | ---: |
| teacher | 100 | +0.0300 [-0.0200, +0.0800] | +0.0900 [+0.0200, +0.1700] | +0.3400 [-0.0500, +0.7500] |
| exact_candidate_oracle | 49 | -0.0204 [-0.0612, +0.0000] | +0.1020 [+0.0000, +0.2245] | +0.3878 [-0.0612, +0.8571] |

Exact scoring was probed in 49 play-disagreement states. Structural and exact candidate maxima differ in 12; 41 have equal score-token values for candidates with different exact scores. The per-state audit records permanent-bonus quantization and Joker extra keys without dedicated numeric features. Other tokens and history can carry related information: these observations do not establish whole-observation aliasing. An exact-oracle action effect cannot by itself separate missing information from imperfect use of existing inputs.

Each arm uses training seeds 0 and 1, FP32, physical batch 320, 1,000 AdamW updates, peak LR 3e-5, weight decay .01, and 100 warmup updates followed by cosine decay. Dropout .1, entropy coefficient .001, HF outcome-weight beta 1.5 and clipping [.25, 4], and mixture .5 stay fixed. The control retains critic coefficients .5/.1; both other arms use zero. Correction batches contain 160 HF and 160 unit-weight teacher-correction rows.

**Controlled continuation development comparisons**

| Arm versus continuation control | Sampled paired difference | Greedy paired difference |
| --- | ---: | ---: |
| actor_only | -0.08 pp [-1.33, +1.17] | -1.00 pp [-3.00, +1.00] |
| correction | -1.58 pp [-2.67, -0.58] | +1.75 pp [-1.50, +5.00] |

**Untouched confirmation panel**

400 game seeds, three sampling seeds per training replicate, two training replicates per arm. Repeats and training replicates are averaged within game seed before the paired bootstrap. The experiment does not estimate uncertainty over a broad population of training seeds.

Sampled gains for training seeds 0 and 1: -0.75 pp, -0.17 pp. Greedy paired difference: +1.75 pp [-0.25, +3.75]. Validation additionally requires no new legality, non-finite-output, or infrastructure failures.

| Arm / training seed / mode | Wins / games | Win rate | Game-cluster 95% interval | Mean final ante | Stalls |
| --- | ---: | ---: | --- | ---: | ---: |
| actor_only/seed0/sampled | 32/1200 | 2.67% | [1.75%, 3.75%] | 3.638 | 0 |
| actor_only/seed0/greedy | 53/400 | 13.25% | [10.00%, 16.50%] | 5.115 | 0 |
| actor_only/seed1/sampled | 30/1200 | 2.50% | [1.58%, 3.58%] | 3.607 | 0 |
| actor_only/seed1/greedy | 48/400 | 12.00% | [9.00%, 15.25%] | 5.128 | 0 |
| control/seed0/sampled | 41/1200 | 3.42% | [2.33%, 4.67%] | 3.644 | 0 |
| control/seed0/greedy | 46/400 | 11.50% | [8.50%, 14.75%] | 5.098 | 0 |
| control/seed1/sampled | 32/1200 | 2.67% | [1.67%, 3.75%] | 3.598 | 0 |
| control/seed1/greedy | 41/400 | 10.25% | [7.25%, 13.25%] | 5.003 | 0 |

Confirmation ante-clear rates, with all games in each arm as the denominator:

| Ante cleared | actor_only/seed0/sampled | actor_only/seed1/sampled | control/seed0/sampled | control/seed1/sampled |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 78.67% | 77.58% | 77.58% | 76.17% |
| 2 | 63.00% | 64.42% | 62.50% | 62.42% |
| 3 | 51.67% | 50.83% | 51.83% | 50.75% |
| 4 | 35.00% | 33.00% | 34.92% | 34.17% |
| 5 | 18.42% | 17.92% | 18.42% | 19.50% |
| 6 | 9.00% | 9.67% | 10.25% | 9.17% |
| 7 | 5.33% | 4.75% | 5.50% | 4.92% |
| 8 | 2.67% | 2.50% | 3.42% | 2.67% |

The secondary development comparison of teacher-correction continuation against actor-only HF continuation is -1.50 pp [-2.67, -0.42] sampled. This tests the correction-data contribution beyond removing critic losses; it is not an independent confirmation result.

**Compute accounting**

Diagnostic accelerator wall time: 2.185 hours. Continuation/intervention/development/confirmation wall time: 7.093 hours. The maximum is 12 accelerator hours; three hours were reserved for confirmation. CPU teacher search is accounted separately.

Confirmation used 1.912 hours. All six training runs processed 320,000 examples each (1,920,000 total) in 3.164 hours. Per-run learning curves are retained in `continuations/*_seed*/learning.jsonl` and TensorBoard events.

The matched V68 teacher panel used 0.452 hours of CPU wall time with 4 workers; this is excluded from accelerator usage. The continuation ledger conservatively includes CPU orchestration and the diagnostic pause within its wall time.

**Verification and reproducibility**

146 targeted regression tests passed at the implementation checkpoint; final verification is recorded in the artifact directory. Tests cover deck propagation, strict metadata, Parquet preservation, policy-path agreement, action support/modes, snapshot isolation, label provenance, seed separation, duplicate/missing paired rows, clustered statistics, actor-only gradients, and the reward reproducer. Additional regressions cover acceptance at exactly five points and exact zero paired gains despite floating-point cancellation. See [the reproducible workflow](README.md) for commands and trial settings.

Final artifact verification passed: ten original checkpoints and the HF files retain their hashes; correction labels retain their pre-training hashes and legality; all six training runs have matching settings; and acceptance was recomputed from all 6,400 raw confirmation games. Conservative compute accounting, including the overhead reserve, totals 9.378 of 12 hours. Confirmation stalls: 0; censored games: 0.

A diagnostic wrapper exceeded its initial 20-minute stage timeout. Its already-running greedy evaluator finished the full panel, and the resumed pipeline validated every completed panel before continuation training. No historical games were discarded or rerun; the error and recovery logs are preserved.

**Remaining uncertainty and next training**

The finite seed panels and two training replicates limit statistical power. HF teacher states were generated under historical rules, so state-distribution and rule-alignment effects can interact. Exact-score interventions do not isolate an individual missing feature. A null actor-only continuation cannot rule out critic interference during the original ten-epoch training: this intervention tests 1,000 further updates from epoch 7. The correction comparison tests replacement of half the HF examples by a finite 100-game correction set; it does not isolate label quality from reduced HF exposure or correction-set reuse. Keep tokenizer v13 unchanged unless a versioned migration and matched new-input training experiment are justified. Generate future teacher demonstrations under pinned current engine sources with corrected reward snapshots; keep whole-game seed splits and the confirmation panel excluded from training. Do not promote an unvalidated continuation or start PPO on the basis of imitation loss alone.

Recommended next experiment: generate a separate current-rule V68 dataset with the pinned Blue/search/scaler configuration and corrected reward snapshots. On that same dataset, compare the existing supervised objective with an otherwise identical objective that doubles play/discard BC weights, retaining the same normalization, optimizer settings, 1,000-update budget, and two training replicates. This targets the weak hand-decision imitation and the short-term benefit from teacher-action substitutions; its gameplay benefit is untested. Keep v13 inputs fixed and reserve a new confirmation panel before the experiment, since this investigation's confirmation panel is now exposed. Rebuilding the data addresses established source/reward inconsistencies; it is not itself a claim that those inconsistencies explain the sampled win-rate gap.

Full local artifacts: `eval/transfer_20260912`. Compact structured results: [results.json](results.json).
