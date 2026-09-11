# Audit fixes and migration — 2026-09-06

This implements the approved follow-up to the [frozen-run audit](2026-09-06-model-training-audit.md).
The source changes are local. The existing SSH training job, its source, weights,
and service configuration were not modified. No new learning run has tested the
effect on fresh-run wins yet.

## What changed

1. **PT/PPO observation parity.** PPO's phase-aware interpretation was correct.
   Teacher tokenization now uses the same phase semantics and resets stale
   round-score progress outside active hand play. A real teacher trajectory is
   replayed through FastRunner and Gym, comparing every observation field.
   This fixes the mismatch; it does not recalibrate the analytic risk estimator.
2. **Observable economic and card state.** Eight new scalars expose reroll cost,
   free rerolls, post-reroll cash, current/post-reroll cash-only interest,
   interest paid per $5 tier, pack choices, and joker capacity. Deck embeddings now
   consume debuff, face-down, permanent bonus, and forced-selection fields.
   There is no fixed $5/$17 rule or new reward for spending. Income, lost interest,
   build needs, and imminent failure remain decisions for the policy to learn.
   The interest features are current cash entitlements, not forecasts of future
   income. Existing permanent-bonus quantization is unchanged.
3. **Correct atomic-action entropy.** Compute actual candidate/AR mixture
   probabilities over legal subsets, then their entropy. Conditional occupancy
   remains differentiable. Duplicate proposal slots are aggregated, and both
   consumable slot and target entropy are included. Tests compare values and
   every gradient against individually enumerated atomic log probabilities.
   This changes the entropy-controller signal; old entropy curves are not
   directly comparable. Inference-warmed static caches are safe for backward.
4. **Separate validation.** The old replay-only split is explicitly called
   `replay_holdout/*`, not a generalization holdout. Independent fresh games
   produce outcome NLL/Brier and win Brier metrics. `eval/sampled/*` follows the
   actual temperature-adjusted stochastic policy; greedy evaluation is separate.
   Truncated episodes are censored for critic scoring, with coverage reported.
   PPO excludes the evaluation seed panel from fresh starts and archive replay;
   the PT→PPO launcher also excludes it during teacher generation. Strict resume
   pins that panel. Unknown historical PT/transfer provenance stays explicitly
   unverified in logs, JSONL, and checkpoints; reservation cannot undo past use.
   This is a development validation panel, not an untouched final test set.
5. **Archive ancestry reaches early actions.** Boundary snapshots retain the
   ancestor seed/action path. A genuine archive win submits only its early
   prefix for budgeted deterministic reconstruction. The rebuilt boundary must
   exactly match a hash of all observation fields before compressed observations
   enter replay. A separate, small actor imitation loss trains those prefixes.
   They never enter PPO ratios, GAE, value targets, terminal replay, or validation.
   The current continuation still trains through on-policy PPO. This is a
   positive-path imitation heuristic, not an unbiased policy-gradient estimator
   or proof that the ancestor decisions caused the eventual win.
6. **Useful, faster evaluation.** New runs record an update-zero baseline and
   per-game JSONL outcomes, fresh Ante reach/clear rates, cash, terminal blind,
   jokers, actions, critic forecasts, and timing. Periodic best-checkpoint
   selection remains win-rate-first, then breaks ties with fresh progression
   and stall rate. Batched exact decoding replaces per-row subset enumeration.
   A compact legal-support trie avoids work for absent card slots without
   limiting the AR policy to candidate proposals. Long evaluations log progress
   heartbeats. Learned policy/value outputs, not analytic risk, are scored by
   the new independent critic metrics.

The PPO clipped surrogate, GAE, and reward schedule were not changed in this
follow-up. Correct entropy and return-path imitation are the actor-objective
changes; the new information enters through small embedding adapters, not a
larger transformer.

## Defaults and limits

- Archive return-path BC coefficient: `0.02`, only active when archives are
  enabled. `--return-path-coeff 0` fully disables this auxiliary objective.
- At most one prefix batch per PPO update, not per microbatch/epoch. Batch cap
  64, at most 8 observations per episode, 32 verified prefixes, 32 pending
  recipes, maximum ancestry length 2,048 actions. Reconstruction is capped at
  64 simulator actions/update (`--return-path-rebuild-steps`). No valid prefixes
  means no BC step. Verified data and its sampler state are checkpointed;
  interrupted reconstruction restarts from the recipe.
- BC shares the PPO optimizer, global clipping, and hard-KL rollback. It adds no
  critic loss, although shared-trunk learning can indirectly change the critic.
  Old two-way SIL/PPO gradient attribution is disabled when the third objective
  is present. Watch `return_path/*`, fresh progression, and KL for over-imitation.
- Evaluation baseline is enabled by default (`PPOConfig.eval_before_training`).
  Up to 25 additional sampled evaluation games use a subset of the reserved
  greedy panel (`eval_sampled_games=0` disables them). Sampled comparisons need
  the same RNG seed and batch size. Evaluation restores the model's mode and
  sampling RNG state and never optimizes these episodes.
- Metrics average critic scores within each episode, then equally across
  episodes. Repeated states from a long run do not become independent trials.
- Deferred: online analytic-risk recalibration, unseen final test panels,
  evaluation cancellation, measured BC coefficient ablations, and new long-run
  evidence that archive-prefix imitation improves fresh Ante-8 wins.

## SSH decoder benchmark

Executed on the requested SSH machine, isolated in
`/tmp/pylatro-grammar-benchmark.1I26qA`, using `CUDA_VISIBLE_DEVICES=1`, one CPU
thread, and `/venv/main/bin/python`. Native NVML and CUDA both identify the
hardware as **RTX 3090**, not 4090. Before benchmarking, GPU 0 had 19,468 MiB in
use and GPU 1 had 466 MiB. No learner was started or moved.

PyTorch `2.11.0+cu128`; five timed repetitions after warmup; synchronized wall
time includes Python/kernel-launch overhead. Synthetic eight-card inputs:
75% hand states, 25% shop states, some hand-targeted consumables, unique
candidate proposals, candidate/AR epsilon 0.1. FP32 grammar logits match the
production BF16-transformer/FP32-head boundary.

| Batch | Old exact mode | New exact mode | Old entropy + backward | Correct entropy + backward |
| --- | ---: | ---: | ---: | ---: |
| 32 | 140.96 ms | 7.83 ms | 16.89 ms | 20.26 ms |
| 160 | 748.01 ms | 7.72 ms | 17.10 ms | 23.62 ms |

Exact-mode actions and selected log probabilities matched the frozen source
for both batches. Entropy deliberately differs: the old implementation is not
the correctness reference. At batch 160, peak allocated memory was 56.36 MiB
for new mode and 78.80 MiB for new entropy/backward. These are decoder-only
allocations/timings, not transformer training VRAM or end-to-end PPO speedups.
The shared machine and small synthetic workload limit extrapolation to 4090
or full evaluation throughput.

Reproducer: [scripts/bench_action_grammar.py](../../scripts/bench_action_grammar.py).
Supply `--baseline` with the frozen v12 grammar and `--candidate` with the new
grammar. It never loads a checkpoint or modifies the training run.

Benchmarked grammar SHA-256:

- Frozen: `c01889f06cc551266d5eb272475ae62af2e0e90d7f5b62d9d8c722457f97e272`.
- New: `38b00fc0a42b6d444bdb58d4f535b7561cdfd56d778200ae115971d5371df825`.

## Checkpoint backup and migration

The remote-selected best checkpoint was downloaded and loaded successfully:
[local backup and provenance](../../runs/checkpoint-backup-2026-09-06.4xHsMC/README.md).
Remote/local SHA-256 agree:
`2315b0be9476ec0bfff4564ba8485493b1a097c7bba942a7fd9d44b2def2cb45`.
It is update 10, 40,960 steps, selected at **0/100 greedy wins**; later tied
evaluations did not replace it. “Best” is the run's stored selection, not an
established ranking of all its checkpoints.

These fixes bump observations to **tokenizer v13**, 33 scalars, and archives to
version 2. The backup and old teacher records remain v12. Do not rename their
schema tags or pad cached data to bypass validation.

For the next experiment, regenerate PT data with the updated tokenizer, reserve
the evaluation panel from the start, and run a new PT→PPO experiment with the
baseline and fresh/sampled metrics. Explicit v11/v12 actor transfer is also
supported with matching model dimensions: new feature adapters start at zero,
the old critic and optimizer are not restored, and source provenance is kept.
Phase/risk inputs changed, so this transfer is **not behavior-preserving** and
needs its own baseline. Old archives/replay do not migrate by strict resume.

The frozen source and historical audit probes remain under ignored `runs/`.
Those probes intentionally reproduce old defects; run them against their
frozen snapshot, not as acceptance tests for v13.

## Verification

- Full local suite: **859 passed, 3 skipped**, with one failure in the existing
  monitor-stop test. It exposed the new baseline evaluation starting despite a
  pending stop request. The baseline now skips that case; the unchanged stop
  test passes. The full-size, two-epoch supervised CPU integration test passed
  in this run, including inference-to-training cache reuse.
- After the fix and final cleanup: **45 passed** across archive, monitor,
  batched evaluation, checkpoint resume, and return-path tests.
- **35 action-grammar tests passed**, including four additional sparse-support
  cases at hand widths 1, 3, 8, and 16, added after the full run began.
- A further **78 tests passed** across PPO entropy/optimization, return-path
  replay, observation parity, PT→PPO launch configuration, and terminal replay.
  These sets overlap; the counts should not be added as independent tests.
- Changed files pass Ruff, and `git diff --check` is clean. Repository-wide
  Ruff still reports 12 pre-existing issues in untouched heuristic/vocabulary
  and engine/UI test files; these were not swept into the audit fix.
- CUDA performance was measured on SSH, not inferred from these CPU tests.
  The benchmarked candidate source checksum still matches the local grammar.

The entire suite was not rerun after the isolated monitor-stop fix; its affected
orchestration paths were rerun as listed above. These checks establish code
behavior and checkpoint safety, not learning gains or a full-model PPO speedup.
