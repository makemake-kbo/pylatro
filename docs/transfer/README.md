The ongoing [complete-cohort PPO experiment](complete_cohort.md) tests complete terminal-outcome labels against completion-window labels under matched collection and optimization. Its [interim confirmation snapshot](complete_cohort_interim.md) records substantially reduced average start-state pessimism and a +4.125-point sampled difference on one replica and two sampling repeats. The remaining confirmation results and greedy guardrail are required before a gameplay verdict. Committed [per-game evidence](data/complete_cohort_interim_games.jsonl) and [metadata](data/complete_cohort_interim_manifest.json) reproduce the interim comparison without large checkpoints.

The preceding [outcome-only PPO investigation](outcome_ppo.md) is complete. Its [confirmation report](outcome_ppo_results.md) shows sampled wins improving from 18.17% to 29.54% and greedy wins from 34.50% to 46.38%, with both training replicas positive. The matched teacher wins 74.75%, so substantial weaknesses remain. PPO uses actual student outcomes without teacher labels, action replacements or imitation loss. The study used 11.160 of its 12 additional MPS hours. Committed [game rows](data/outcome_confirmation_games.jsonl) and [metadata](data/outcome_confirmation_manifest.json) reproduce all 5,200 confirmation and teacher games without loading the checkpoints.

The original transfer investigation compared the ten HF-pretrained checkpoints under matched gameplay conditions and kept PPO stopped. Its generated results live in `eval/transfer_20260912`; original checkpoints, HF Parquet files, and prior logs remain unchanged.

Read the [evidence and gameplay report](report.md) and [structured results](results.json). The report compiler includes only completed evaluation panels.

The subsequent [action architecture study](action_refactor.md) produced a validated improvement using the existing observations and a frozen encoder. Its [400-seed confirmation results](action_refactor_results.md) show sampled wins increasing from 3.58% to 19.33% and greedy wins from 7.75% to 32.12%, averaged across two training replicas. The methods report links the preselected checkpoint and records regression tests, causal evidence, limitations and the next training recommendation.

The additional nine-hour investigation is recorded separately in [the learning bottleneck report](learning_bottleneck.md).
It tests memorization and the fixed candidate/AR mixture, then crosses mixture `.5/1.0` with hand-action BC weight `1/2`
on current-rule demonstrations. Original reports and artifacts remain unchanged. For a fresh reproduction, initialize
a new output directory first:

```sh
.venv/bin/python tools/learning_bottleneck.py init --root eval/new_learning_study
```

Then run CPU generation and the serialized MPS pipeline in separate terminals:

```sh
.venv/bin/python tools/learning_bottleneck.py generate --generation-workers 8 --root eval/new_learning_study
.venv/bin/python tools/run_learning_bottleneck.py --root eval/new_learning_study
```

The additional CPU-only phase-replacement diagnostic can run while the fixed MPS trials train:

```sh
.venv/bin/python tools/phase_oracle_experiment.py --root eval/new_learning_study
```

It uses 50 separate exploratory seeds (`5000000–5000049`), three sampling repeats, CPU batch 1, and the original
epoch-7 checkpoint. It compares student control with teacher replacement in hand, shop, both, and all phases.
It checks every teacher query for live-state/RNG isolation and saves decision traces. It does not supply training
examples or choose the confirmation candidate. Its absolute bootstrap intervals are regenerated in canonical seed
order from the per-game files; the original completion-order summary is also preserved.

After that diagnostic completes, the exploratory sale-conditioned follow-up is reproducible with:

```sh
.venv/bin/python tools/sale_oracle_experiment.py --root eval/new_learning_study
```

It reuses the completed student and shop-oracle baselines and adds two 150-game CPU panels: replace only
teacher-prescribed Joker sales, or only other teacher shop actions. This follow-up was motivated by the exposed
HF audit and is explicitly post-hoc. It neither trains a model nor selects the main confirmation candidate.
Final verification checks the intervention boundaries, trace hashes, and identical student proposals up to the
first action replacement in each paired trajectory.

`tools/audit_sale_head.py --root eval/new_learning_study` decomposes the original checkpoint's sale-family and
conditional Joker-selection probabilities on all sale labels in the fixed HF audit. It runs on CPU, checks the
probability factorization and agreement with the saved MPS NLLs, and preserves per-seed/step rows. This diagnoses
the existing head; it does not change the model or validate an architecture replacement.

After the sale-oracle baseline completes, `tools/sale_choice_oracle_experiment.py --root eval/new_learning_study`
adds a 150-game CPU panel with the teacher's sale-timing rule and student conditional Joker choice. A separate seeded
generator draws the Joker index without advancing the main policy RNG. Comparing this panel with the teacher-sale
baseline separates the value of Joker choice within the teacher-timing policy. It is explicitly post-hoc.

The pipeline waits for complete generation before training and preserves a 2.5-hour confirmation reserve inside its
nine-hour accelerator budget. It skips completed stages, refuses to overwrite partial training runs, and records
failed accelerator jobs in the ledger. Inspect failures before resuming; an interrupted running ledger entry requires
explicit crash accounting. CPU generation defaults to four workers; `--generation-workers 8` uses eight independently
seeded workers and resumes hash-checked completed games. Generation attempts and elapsed CPU wall time are recorded.
It uses an explicit list of 1,000 training and 200 validation
seeds. New gameplay panels are `120000–120199` and `130000–130399`; they never supply training examples.

This investigation also fixed a supplementary original-reference check during development. To reproduce it, copy
`eval/learning_bottleneck_20260913/incumbent_reference_plan.json` into the new study root before confirmation.
It compares the selected continuation and control with original epoch 7 at mixture .5 after every required
400-seed panel finishes. The time-only gate selects 400 reference seeds if at least 3,192 accelerator seconds
remain, otherwise 200 seeds if at least 1,656 seconds remain, otherwise a skipped-budget decision. It does not
change candidate selection or primary acceptance.

After the pipeline finishes, verify hashes, configurations, seed panels, and raw game metrics, then compile the report:

```sh
.venv/bin/python tools/report_learning_bottleneck.py --root eval/new_learning_study \
  --require-complete --output docs/transfer/new_learning_study.md
```

The reference is Blue Deck, White Stake, Ante 8, temperature 1.0, mixture 0.5, batch 32, four engine workers, and FP32. The new checkpoint evaluator restores architecture and reward settings strictly from checkpoint metadata. It rejects incompatible schemas, missing configurations, duplicate seeds, incomplete paired panels, and existing output directories. Each run records effective settings, explicit CLI overrides, source/checkpoint/data hashes, individual outcomes, timing, confidence intervals, and TensorBoard events.

For the first investigation, development seeds are 100000–100199 and confirmation seeds are 110000–110399. The experiment tools check both against the HF generation manifest, HF game manifest, and the local V68 bootstrap seed manifest. Correction data uses 3000000–3000099. Sampling repeats are 0, 1, and 2. Repeated games are averaged by game seed before bootstrapping. Single-repeat win rates also have Wilson intervals, which remain informative for zero wins. Bootstrap intervals can degenerate at all-zero/all-one empirical outcomes; they should not be read as certainty about population performance.

Run a matched checkpoint evaluation:

```sh
.venv/bin/python tools/evaluate_checkpoint.py \
  --checkpoint checkpoints/supervised_5k_v68_20260911_181359/supervised_epoch10.pt \
  --panel development --mode sampled --sampling-seed 0 --device mps \
  --training-manifest cache/datasets/5k-balatro-games/metadata/generation.json \
  --training-manifest docs/heuristic/pretraining_v68_seeds.json \
  --output-dir eval/new_transfer_run/epoch10_sample0
```

Use `--seeds` or `--seed-file` for explicit panels, `--deck b_red` and `--hand-ar-mixture-eps 0.1` for controlled overrides, and `--kind demonstration_seed` for autonomous play on seeds appearing in training manifests. `--paired-outcomes` accepts a reference `outcomes.jsonl` with identical game seeds and sampling repeats. The historical `--fixed-seeds` alias selects the 10000-series regression panel. Architecture overrides and silent configuration fallbacks were removed.

The matched V68 teacher requires explicit generation metadata:

```sh
.venv/bin/python tools/evaluate_checkpoint.py \
  --checkpoint checkpoints/supervised_5k_v68_20260911_181359/supervised_epoch10.pt \
  --heuristic-baseline --device cpu \
  --teacher-generation cache/datasets/5k-balatro-games/metadata/generation.json \
  --training-manifest cache/datasets/5k-balatro-games/metadata/generation.json \
  --output-dir eval/new_transfer_run/teacher
```

The teacher uses CPU simulator search, not a neural accelerator forward pass. Model evaluations, interventions, continuation training, and confirmation use MPS on the investigation host.

The experiment entry points are:

| Entry point | Measurement or operation |
| --- | --- |
| `tools/diagnose_transfer.py` | Uniform seed sample of 200 HF games, all decisions, and 20 balanced recorded-action replays. Offline metrics are explicitly training-data diagnostics. |
| `tools/reconstruct_generation.py` | Isolated historical simulator sources, each verified against its HF-generation SHA-256. Run the Cython compiler with its working directory set to the reconstruction root. |
| `tools/historical_transfer_evaluation.py` | Identical checkpoint and development seeds under the reconstructed generation rules. |
| `tools/diagnose_reward_transfer.py` | Minimal, repeatable boss-key reward-input defect and corrected potential comparison. |
| `tools/replay_reward_prefixes.py` | Compare current generator/environment rewards on every legal prefix step, even when historical recorded actions later become illegal. |
| `tools/transfer_experiments.py` | Ten-checkpoint screen, two-checkpoint expansion, complete deck/mixture factorial, and demonstration-seed evaluation. |
| `tools/continue_transfer.py` | Corrections, one-action interventions, six supervised trials, development selection, and untouched confirmation. |
| `tools/intervene_transfer.py` | Restore one reservoir-sampled disagreement per correction game; substitute one action and continue both branches with identical sampling seeds. |
| `tools/run_transfer_pipeline.py` | Sequence regression checks, historical MPS evaluation, and continuations after development; enforce the accelerator budget and confirmation reserve. |
| `tools/report_transfer.py` | Compile completed artifacts and paired comparisons without treating partial runs as results. |
| `tools/verify_transfer.py` | Pin correction hashes before training; verify original artifacts, correction provenance, matched trials, raw confirmation panels, acceptance, and compute accounting. |

Each continuation uses 1,000 updates, physical batch 320, AdamW LR 3e-5, weight decay .01, 100-step warmup followed by cosine decay, and training seeds 0 and 1. Control and actor-only arms use identical HF ordering; critic losses are disabled only in actor-only arms. The correction arm uses 160 HF and 160 correction examples per update. Original dropout, entropy coefficient, and HF outcome weighting stay fixed. Correction rows have unit BC weight, teacher/source/student-action provenance, and no return, reward, or terminal-outcome labels. Dummy collator values are masked and never supply critic losses.

Fast generation omitted `boss_key` and shop-event reward fields, and computed action diagnostics after the action. The corrected generator supplies those inputs, preserves the pre-sale Joker replacement baseline, and captures action diagnostics before cards or shop slots change. New records use `reward_snapshot_version=3` (version 2 was the intermediate boss-key-only fix). Existing HF records and return targets retain their original values. Tokenizer-v13 features are unchanged; historical reconstruction retains the old generator for reproducibility.

A candidate is validated only on confirmation: sampled improvement at least five percentage points, paired 95% interval above zero, positive improvement for both training seeds, and a greedy paired interval ruling out a regression greater than five points. Incomplete evaluations or policy/infrastructure errors prevent validation. Development threshold checks are never labeled validated.

The correction experiment is motivated by state-distribution dependence in [Ross, Gordon, and Bagnell (2011)](https://proceedings.mlr.press/v15/ross11a.html). Explicit evaluation uncertainty is motivated by [Jordan et al. (2020)](https://proceedings.mlr.press/v119/jordan20a.html). This investigation's game-cluster bootstrap is its own specified analysis, rather than an implementation of either paper's complete methodology. Oracle action interventions are exploratory; success does not by itself prove a particular omitted feature is causal or justify changing the observation schema.
