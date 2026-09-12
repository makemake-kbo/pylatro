# PPO organization and performance checks

## Module ownership

`training/ppo.py` remains the public entry point for `PPOConfig`, `train_ppo`,
`evaluate_model`, `run_seed_evaluation`, `load_actor_transfer`, and the public
schedule resolvers. Private helpers now live with their owners; internal tools
and tests import them there.

| Module | Responsibility |
| --- | --- |
| `ppo.py` | Training orchestration, schedules, evaluation/checkpoint cadence |
| `ppo_config.py` | Configuration, validation, provenance, schedules, RNG seeding |
| `ppo_rollout.py` | Environment construction, persistent episode state, rollout collection |
| `ppo_observations.py` | Observation staging and vector-environment info handling |
| `ppo_policy.py` | Policy adapter, DataParallel handling, policy temperature |
| `ppo_optimization.py` | PPO/SIL updates, critic replay, KL rollback, entropy control |
| `ppo_checkpoint.py` | Strict loading, actor transfer, checkpoint writing |
| `ppo_evaluation.py` | Fresh evaluation and regression detection |
| `ppo_metrics.py` | Episode history, rollout diagnostics, TensorBoard reporting |

Archive environment lifetime remains with the trainer. `RolloutState` preserves
unfinished episode prefixes and pending forecasts between collection windows;
archived prefixes are not inserted into the fresh on-policy PPO buffer.

## First performance pass

- Attention no longer requests unused attention weights. This permits PyTorch's
  SDPA implementation while retaining checkpoint parameter names/shapes.
  See the [MultiheadAttention documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
- Ordered card legality reduces each compatible subset to its next card once,
  instead of scanning the subset table independently for all 16 slots. The
  reduction uses integer counts and does not change the sampling RNG calls.
- Rollout selected-action probability reuses `log_prob(actions).exp()`.
- PPO uses the grammar's existing action-type mask and transfers minibatch
  statistics together. Hard-KL checks, full-rollout soft-KL checks, rollback,
  accumulation boundaries, and loss formulas remain unchanged.
- CPU observations share NumPy/tensor storage. CUDA staging uses reusable pinned
  host buffers and nonblocking copies on the current stream, with an event wait
  before reusing host storage. Consumers must follow normal same-stream ordering.
  The wait matters because mutating pinned sources before a transfer completes
  can corrupt data; see [PyTorch's transfer guidance](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html).

No changes to rewards, GAE, archive sampling, optimizer hyperparameters, policy
temperature, or model dimensions are included in this pass.
Attention outputs/gradients are numerically close, not promised bit-identical
across backend implementations. Training-mode attention dropout may consume
random numbers differently; PPO already disables dropout during its updates.

## BF16 migration

Add `--precision bf16` to the supervised or PPO command on a CUDA GPU with native
BF16 support. For strict resume, an explicit flag changes compute precision
without converting checkpoint weights or resetting Adam. The change is logged.
Future resumes without the flag retain the checkpoint's saved precision;
older checkpoints and new runs still default to FP32. `--precision fp32` is the
explicit fallback. CPU/MPS evaluation of a BF16-configured checkpoint uses FP32.

BF16 autocast is scoped inside the transformer forward, including DataParallel
replicas. Embeddings, policy/value heads, probability calculations, PPO ratios,
losses, gradient accumulation, weights, and Adam state remain FP32. Supervised,
PPO, and terminal-replay steps reject nonfinite gradient norms in BF16 mode. No FP16
conversion or GradScaler is involved. See [PyTorch's autocast guidance](https://docs.pytorch.org/docs/2.11/amp.html)
for FP32 subregions and thread-local autocast behavior.

This is a numerically checked **opt-in**, not proof that long-run learning or
Ante-8 win rates are unchanged. Use the existing KL/calibration/fresh-run eval
guards when enabling it on a trained checkpoint; no private checkpoint was
uploaded for these tests.

Pure BF16 parameters plus ordinary Adam are **not** interchangeable with this
setup. Adam initializes its moments with the parameter dtype and does not keep
an FP32 master weight ([PyTorch 2.11 implementation](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/optim/adam.py)).
At the archive recipe's `3e-6` learning rate, a diagnostic Adam parameter near
`0.1` with constant gradient `1` changed on all 1,000 FP32 steps and on none of
the BF16 steps: each parameter update rounded away. This is a counterexample,
not a claim that every model weight freezes. A genuine low-precision-storage
optimizer needs a separately validated update-accumulation/rounding design.

## Sequential PT → PPO runs

The new-run launcher uses the full default 384-wide, 12-layer model, generates
1,000 teacher games with eight spawned workers, trains five supervised epochs,
then transfers the actor into 1,000 archive PPO updates (4,096,000 environment
steps). PPO uses logical batch 320 / physical batch 160, LR `3e-6`, and fresh
100-game evaluations every ten updates. Supervised uses physical batch 160.
The target is Ante 8 throughout. The PPO critic is fresh because its milestone
reward objective differs from the supervised shaped-return targets.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m pylatro_agent.training.pt_ppo_run --run-dir /path/to/new-run
```

Use a process supervisor for a remote run. The launcher writes `manifest.json`
and `status.json`, refuses to replace an existing manifest, stops on stage
failure, and starts each stage in a separate process so teacher-data memory
is released before PPO. It does not automatically resume or back up artifacts.
The teacher dataset stays in RAM; size it for the machine's cgroup limits,
not its advertised host RAM/CPU count. Override `--games`, `--epochs`, and
`--updates` for a smaller preflight; `--help` lists batch/worker options.

The first full teacher-data preflight exposed an existing terminal-label bug:
cash-out advances a winning Ante-8 game's counter to 9, which the outcome
validator rejected. The validator now accepts that post-win counter only for
wins; out-of-range death labels are still rejected. This is a correctness fix
found while preparing the run, not part of the kernel speed measurements below.

## SSH GPU measurements (2026-09-05)

Benchmarks ran on the supplied SSH host, on one isolated GPU (`cuda:0`). CUDA
reports **RTX 3090**, compute capability 8.6, 82 SMs, and 24,124 MiB, despite
`nvidia-smi` advertising RTX 4090. These are not RTX 4090 measurements.

PyTorch 2.11.0+cu128, CUDA 12.8, 384-wide/12-layer/8-head model, FF width 1536,
16 rollout rows, microbatch 160, logical batch 320, 320 synthetic PPO samples,
2 PPO epochs, one CPU thread. Medians: 5 repetitions after 2 warmups. Profiling
was outside timed regions. The baseline was frozen before performance changes;
all three variants used the same shapes and seed. Raw results are in
[the benchmark report](performance/2026-09-05-rtx3090.json).

| Case | Baseline FP32 | Optimized FP32 | Optimized BF16 |
| --- | ---: | ---: | ---: |
| Observation staging | 0.813 ms | 0.424 ms | 0.426 ms |
| Policy forward + sampling + scoring/telemetry | 125.43 ms | 53.78 ms | 56.71 ms |
| Model forward/backward | 388.10 ms | 365.48 ms | 190.12 ms |
| PPO update, including full-rollout KL evaluation | 2236.44 ms | 2036.11 ms | 1114.37 ms |
| Peak allocated memory during PPO | 10,032 MiB | 8,556 MiB | 5,169 MiB |

Combined changes give **2.01× PPO update speed** and **48.5% less peak allocated
memory** versus the baseline. BF16 alone gives 1.83× PPO speed over optimized
FP32, but is about 5% slower for this small rollout batch. Its transformer math
is cheaper, while additional casts/kernel launches matter at batch 16. Keep
rollout and PPO on the same configured precision for policy consistency.

The FP32 optimization reduces profiled PPO `cudaStreamSynchronize` calls from
320 to 236 and kernel launches from 33,542 to 22,894. Observation staging goes
from 14 stream waits to one host-buffer reuse event wait. Minibatch preparation
is unchanged (approximately 16–18 ms); remaining host/kernel-launch overhead is
a possible follow-up, not evidence of another implemented optimization.

At full model size, fixed-weight FP32/BF16 checks passed:

- Finite losses, gradients, parameters, and optimizer state.
- Gradient cosine similarity: 0.9999997.
- Policy-ratio error versus FP32: p99 0.0598%, maximum 0.0748%.
- Maximum expected-return difference: 0.0000663.
- BF16 rollout-versus-PPO spurious KL: approximately 3.4e-9 when scoring the
  same actions at rollout batch 16 versus gradient-enabled batch 160.

The fixture mixes reachable blind-selection, hand-play, and shop observations;
the PPO buffer is synthetic and excludes SIL and terminal-replay updates.
These results measure kernels and optimizer execution, not end-to-end environment
throughput or learning speed. CPU/CUDA reference tests additionally cover exact
legality masks, attention gradients, pinned-buffer reuse, BF16 dtypes/Adam state,
and unsupported-device rejection.

## CPU measurements (2026-09-05)

The before/after CPU check used PyTorch 2.11.0, one CPU thread, a 64-wide,
2-layer/4-head model with FF width 256, 16 observation rows, microbatches of 16,
logical batches of 32, 32 synthetic PPO samples, and 2 PPO epochs. Each timing
is the median of 5 repetitions after 2 warmups. Profiler collection is separate
from the timed repetitions.

| Case | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Observation staging | 0.177 ms | 0.102 ms | 1.73× |
| Policy forward + sample + scoring/selected-probability telemetry | 124.78 ms | 65.82 ms | 1.90× |
| Model forward/backward | 95.85 ms | 70.02 ms | 1.37× |
| PPO update, including rollout KL evaluations | 628.69 ms | 473.10 ms | 1.33× |
| Minibatch preparation (unchanged) | 0.347 ms | 0.345 ms | approximately unchanged |

In the profiled PPO update, `aten::item` calls decreased from 486 to 406.
Remaining calls include optimizer bookkeeping, batch metadata, and validation;
not every CPU scalar operation represents a GPU synchronization.

These are **small-model CPU microbenchmarks**, not end-to-end environment
throughput, learning curves, or RTX 4090 performance claims. The benchmark mixes
reachable blind-selection, hand-play, and shop observations, but uses a synthetic
PPO buffer without SIL or terminal replay updates.

## Reproducing the checks

With the project dependencies and Cython extensions installed:

```bash
PYTHONPATH=src python scripts/bench_model_ppo.py \
  --device cpu --d-model 64 --n-layers 2 --n-heads 4 --d-ff 256 \
  --envs 16 --micro-batch 16 --logical-batch 32 --samples 32 \
  --warmup 2 --repeats 5 --profile --output /tmp/ppo-cpu.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/bench_model_ppo.py \
  --device cuda:0 --precision bf16 --check-precision \
  --d-model 384 --n-layers 12 --n-heads 8 --d-ff 1536 \
  --envs 16 --micro-batch 160 --logical-batch 320 --samples 320 \
  --warmup 2 --repeats 5 --profile --output /tmp/ppo-gpu.json

pytest tests/test_model_performance.py tests/test_action_grammar.py \
  tests/test_ppo_entropy_control.py tests/test_sil.py tests/test_archive_return.py
```

Use identical arguments, seeds, device, precision settings, and dependency
versions before/after. Pin one idle GPU; do not run production training in
parallel with measurements. The GPU model configuration above is a benchmark
choice, not a recommendation to change the learner. The script does not load or
overwrite checkpoints or start an environment training run.

Reference tests compare legality masks exactly, attention outputs and gradients
within floating-point tolerance, and observation staging/snapshot ownership.
They also run on CUDA when available. Before kernel changes, a seeded 96-step
old/new trainer comparison matched every parameter and all 247 scalar metric
series exactly. After the PPO bookkeeping/staging changes, the same comparison
using the new kernels on both trainers still matched exactly.

Validation completed: the full local suite passed (816 passed, 2 skipped),
plus 21 passed/1 skipped in a separate final precision/archive regression
checks. The final remote CPU/CUDA suite passed all 15 tests, including BF16
critic contexts at Antes 1, 5, 6, and 8. Local skips include CUDA-only coverage;
that coverage was exercised on the remote GPU.

## CPU evaluation and critic replay

PPO evaluates active games with one central policy batch and four CPU engine
workers by default. `--eval-workers 0` keeps environments in the parent process,
which can be useful for very small panels. Workers are spawned without the
model, use one torch thread, and release games as they finish. Binary action
masks are bit-packed in transit and restored exactly before inference. Greedy and sampled
per-seed parity, callback errors, worker crashes, and cleanup have regression
coverage.

`--eval-cpu-threads 0` chooses one inference thread for model widths below 256
and up to eight for larger models, capped by CPU affinity. Positive values set
an explicit count. The trainer's thread count is restored afterward, including
on errors. This allows CPU evaluation to use multiple threads even when a GPU
launcher exports `OMP_NUM_THREADS=1`. CUDA/MPS model inference retains the
parent's thread setting. Thread counts can affect floating-point reductions;
use an explicit value for reproducible performance comparisons.

Analytic risk and score caches reuse identical calculations under immutable,
content-based keys. They invalidate on scoring-input changes and are bounded
per process. The engine RNG and scoring formulas are unchanged.

Terminal critic replay and bootstrap calls skip unused policy heads. Replay
also avoids the encoder autograd graph and expansion/transfer of unused action
masks. Ordinary PPO/SIL batches retain policy gradients and action masks.
Replay diagnostics transfer a compact payload once and compute their metric
buckets on CPU. Accelerator rollout statistics also share one host transfer;
CPU observations/statistics retain zero-copy views.

TensorBoard now includes `performance/rollout_wall_seconds`,
`performance/rollout_postprocess_wall_seconds`,
`performance/optimization_wall_seconds`,
`performance/terminal_replay_wall_seconds`, `performance/update_wall_seconds`,
and `performance/steps_per_second`. Total update time includes periodic
evaluation and checkpointing. These measure host wall time without introducing
extra device synchronization. Evaluation separately reports engine work and
batch stepping wall time (`eval/env_seconds` and `eval/env_wall_seconds`).

See the [CPU audit](performance/2026-09-11-ppo-cpu-audit.md) for reproducible
benchmarks, measured gains, validation scope, and remaining work. Local CPU
results do not establish production GPU throughput or long-run learning quality.
