# PPO CPU audit — in progress

Scope: remove PPO evaluation and CPU bottlenecks. Batch decisions actually
reached by active games; do not add exhaustive candidate lookahead. PPO-only
approximations are authorized if useful, but this first change is exact.

## Verified first change

`BalatroEnv._capture_state_info` recomputed analytic risk and weakest-joker
estimates for both the post-step state and the next pre-step state. A bounded
one-entry cache now keys these estimates on freshly captured state features.
An immutable serialized key detects direct engine mutations and avoids aliasing
when consumers mutate returned info. Nothing is deserialized. This applies to
training and evaluation environments; it changes neither engine RNG nor scoring.

CPU measurement: PyTorch 2.11.0+cu130, one torch thread, no usable CUDA device.
Fixed randomly initialized model (seed 0, width 32, one layer, two heads,
FF width 64), seeds 10000–10007, batch 8, Ante 2, stall limit 32; 308 steps.
Three alternating unprofiled measurements against the original method loaded
from Git HEAD:

| Method | Seconds | Median |
| --- | --- | --- |
| Original | 3.0192, 3.0467, 3.0624 | 3.0467 |
| Cached | 2.3357, 2.3476, 2.3423 | 2.3423 |

23.1% lower wall time, or 1.30x throughput, on this fixture only. This is not
evidence of the same speedup on a large pretrained model or CUDA training.

cProfile before/after: 9.481 / 6.729 seconds; analytic `_score_pass` calls fell
from 9,299 to 5,885; `estimate_clear_risk` calls fell from 641 to 333. Profiling
adds substantial overhead, so use unprofiled timings for speed comparisons.

63 targeted tests pass across batched evaluation, environment, risk, build value,
and the new cache suite. New tests cover input mutation, returned-info mutation,
and exact cached/uncached episode outcomes including critic forecasts.

Reproducible current-state benchmark:

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/bench_ppo_eval.py --output /tmp/eval.json
OMP_NUM_THREADS=1 .venv/bin/python scripts/bench_ppo_eval.py --profile /tmp/eval.prof
```

`--disable-estimate-cache` forces misses for correctness/performance diagnosis;
it retains key-construction overhead and is not the original Git baseline.
The benchmark uses a small random model, not a trained production policy.

## Parallel CPU evaluation

Evaluation now assigns live games to a bounded pool of spawned CPU processes.
Each worker advances a shard of the active batch; policy inference and sampling
remain in the parent process. Finished slots refill immediately, and outcomes
retain caller seed order. Worker pipes carry observations and the terminal
fields evaluation actually reads, excluding large training diagnostic payloads.
Workers close on normal completion, remote errors, process death, and exceptions
from result callbacks. Environment objects are released as games finish.

PPO, the PT-to-PPO launcher, and the checkpoint evaluator default to four
workers. `--eval-workers 0` selects local stepping. The lower-level evaluation
APIs retain their zero-worker default. Workers use one torch CPU thread each
and never receive the model or initialize CUDA. All interpreters launch before
receiving game data, avoiding serial startup blocked by large spawn arguments.

Fixed 64-game CPU fixture (same small architecture as above, batch 32, 2,202
actual decisions), two unprofiled runs per configuration:

| Environment stepping | Seconds | Mean |
| --- | --- | --- |
| Local, cached | 15.1859, 15.0724 | 15.1291 |
| Four workers, parallel startup, cached | 8.9645, 8.5381 | 8.7513 |

42.2% lower wall time (1.73x throughput), including process startup/shutdown.
Every non-timing result field matched between these runs. This measurement is
additional to the first cache comparison, uses a different panel size, and
must not be multiplied into a claimed end-to-end production speedup.

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/bench_ppo_eval.py \
  --games 64 --batch-size 32 --eval-workers 0 --repeats 2 --output /tmp/eval-local.json
OMP_NUM_THREADS=1 .venv/bin/python scripts/bench_ppo_eval.py \
  --games 64 --batch-size 32 --eval-workers 4 --repeats 2 --output /tmp/eval-workers.json
```

`env_seconds` remains summed engine CPU work. New `env_wall_seconds` measures
batch stepping wall time, including IPC; the distinction matters with parallel
workers. Overall evaluation wall time additionally includes startup and resets.
Checkpoint metadata records worker count and evaluation batch size.

146 targeted tests pass across parallel/batched evaluation, cache behavior,
environment/risk/build scoring, PPO checkpoints, entropy control, and launchers.
Parallel tests compare full greedy and sampled outcomes, critic forecasts, and
parent torch RNG states, including slot reuse. They also exercise worker errors,
forced worker termination, callback failure, model-mode restoration, and cleanup.
A full two-update CPU smoke run exercised four training envs x 32 rollout steps,
PPO optimization, terminal critic replay, initial/periodic greedy and sampled
evaluation, and checkpoint saving with two evaluation workers. The final
checkpoint reports update 2 / 256 steps; model weights and Adam tensors are
finite. The profiled run took 23.19 seconds after parallelizing startup versus
30.02 seconds before; this tiny, cold-start-heavy run is functional validation,
not a production throughput estimate.

## Reusing identical analytic score passes

A second exact cache now shares immutable score-pass results across risk,
hand planning, baseline/full/leave-one-out build scoring, and diagnostics.
Its key includes the full card and Joker descriptors, hand type/base chips and mult,
hands available, hand size, and Idol target. Cash, blind target/progress, and
shop offers do not affect a score pass; derived readiness ratios still use the
current target outside the cache. Keys snapshot mutable inputs, and custom
unserializable mappings fall back to uncached scoring. A lock protects cache
bookkeeping; forked children reset it so they cannot inherit a lock held by
a parent background thread. Each process is limited to 256 entries and 2 MiB of serialized keys
plus the small immutable results.

On the same 64-game fixture, after this cache:

| Environment stepping | Seconds | Mean |
| --- | --- | --- |
| Local | 12.1767, 11.8881 | 12.0324 |
| Four workers | 6.9209, 6.5570 | 6.7390 |

All non-timing outcomes match the preceding 64-game runs. This is about 20–23%
lower wall time than the corresponding configuration without the score cache,
and 55% lower than the 15.13-second local baseline that already had the first
state-estimate cache. This remains a small random-model benchmark.

Use `scripts/bench_ppo_eval.py --disable-score-cache` for local cache-ablation
measurements; overrides intentionally reject worker mode because monkeypatches
in the parent would not affect spawned children.

95 scoring, hand-planning, risk, and reward tests pass. Added checks cover
non-scoring state changes, nested card/Joker/hand mutations, hand size and
remaining hands, Idol targets, unserializable mappings, and bounded eviction.

## CPU thread measurements

A CPU policy microbenchmark using the production architecture (width 384,
12 layers, 8 heads, FF 1536, 16 concurrent observations) measured:

| Torch CPU threads | Median policy batch |
| --- | --- |
| 1 | 1,452.32 ms |
| 8 | 251.45 ms |

Both used the same command apart from `--threads`, one warmup and three timed
repeats, and matching preparation sizes:

```bash
.venv/bin/python scripts/bench_model_ppo.py --device cpu --case rollout_policy \
  --threads 8 --micro-batch 16 --logical-batch 16 --samples 16 --warmup 1 --repeats 3
```

The workload is sampled inference plus selected-action log probability, not a
whole training update. It shows that capping the parent model to one CPU thread
would be harmful here. Environment workers still use one thread. Some GPU
launchers export `OMP_NUM_THREADS=1`, so CPU evaluation overrides need separate
thread controls. PPO and checkpoint evaluation now use scoped
`--eval-cpu-threads`: zero chooses one thread for widths below 256 and up to
eight for larger models, capped by CPU affinity. Positive values override that
heuristic. Training thread settings are restored on success or error; CUDA/MPS
inference does not change the parent CPU thread count. Low-level APIs preserve
the caller setting unless explicitly requested.

After the score cache change, 286 broader environment/heuristic/evaluation tests
passed. A subsequent 95-test scoring/reward/risk pass checked normalized hand
keys, and ten cache-specific tests checked the fork-lock reset. A full PPO smoke
run with four AsyncVectorEnv training workers and two evaluation workers saved
update 2 / 256 steps with finite model and Adam tensors.

## Scoped evaluation and critic-only forwards

An end-to-end CPU evaluation with width 384 / 12 layers / 8 heads / FF 1536,
eight games, batch eight, and a four-step stall guard measured 5.02 seconds
with one inference thread and 1.26 seconds with automatic selection (eight).
Both runs made 56 decisions and returned identical non-timing outcomes. This
is a short random-policy fixture, not evidence of the same speedup on long
trained-policy games. Thread changes can alter floating-point reductions;
explicit thread settings remain available for reproducible comparisons.

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/bench_ppo_eval.py --games 8 \
  --batch-size 8 --d-model 384 --layers 12 --heads 8 --d-ff 1536 \
  --max-no-progress-steps 4 --eval-cpu-threads 0 --repeats 1
```

Terminal replay previously constructed the unused policy heads and retained
an encoder autograd graph despite requesting gradients only for terminal
critic parameters. `critic_only` now encodes under no-grad and skips policy
heads. Terminal replay and rollout bootstrap calls use this path. The ordinary
PPO forward remains differentiable. No checkpoint parameter names changed.

Production-sized CPU model, batch 16, eight threads, one warmup and three
repeats, critic loss plus gradients of the terminal heads:

| Replay forward | Median |
| --- | --- |
| Original policy/value path | 321.74 ms |
| Critic-only path | 287.56 ms |

10.6% lower wall time in this microbenchmark, in addition to avoiding encoder
activation graphs. GPU memory savings have not been measured. Critic replay
also skips expansion and transfer of 17,327-action masks (16.92 MiB per default
256-row batch). The sampler preserves identical row choices and remaining
tensors. Policy/SIL batches still receive their masks. Reproduce using
`scripts/bench_model_ppo.py --case critic_reference` and `--case critic_replay`
with `--device cpu --threads 8 --micro-batch 16 --logical-batch 16 --samples 16
--warmup 1 --repeats 3`.

The combined regression run passed 244 tests, with two CUDA-only checks
skipped. After omitting critic masks, the focused model/replay suite passed
21 checks with the same two skips.

Tests compare critic predictions and terminal-head gradients to the original
path, prohibit calls to the policy head, and verify encoder gradients are
absent. CPU DataParallel wrappers are covered; actual multi-GPU execution is
not available locally. Thread tests cover automatic bounds, explicit/nested
scopes, error restoration, and unchanged CUDA/MPS behavior.

The trainer now emits `performance/{rollout,rollout_postprocess,optimization,
terminal_replay,update}_wall_seconds` and `performance/steps_per_second`.
A full multiprocess smoke run saved update 2 / 256 steps with finite model and
Adam tensors and confirmed every timing tag in the actual TensorBoard events.
Before skipping replay mask allocation, update 2 recorded rollout 1.29 seconds,
rollout postprocessing 0.02, optimization 0.57, replay 1.01, total update 6.82
including the deliberately frequent small-panel evaluations and checkpoints.

Total update time includes periodic evaluation and checkpoints. These are host
wall measurements without added GPU synchronization, not kernel timings.

A refreshed eight-game CPU cProfile shows 896 actual score computations from
5,885 score requests, versus 9,299 computations before the optimizations.
Total profiled time is 3.10 seconds versus the original 9.48; engine stepping
is 2.44 seconds, score requests 1.25 seconds, actual score computation 0.79,
and policy inference 0.41. This supersedes earlier hotspot totals.

## Remaining audit and implementation

- Parallel CPU stepping is implemented and verified on bounded fixtures.
  Validate worker scaling, memory, startup amortization, and throughput on
  representative trained policies and long evaluation games.
- Analytic scoring is greatly reduced but still material in the small-model
  CPU fixture. Compare long trained-policy trajectories before deciding whether
  further descriptor preparation or PPO-only approximations are worthwhile.
- The small end-to-end CPU profile confirms analytic scoring remains expensive
  during rollouts. Expand profiling to representative model sizes and longer
  rollouts; investigate repeated hand/build scoring, CPU thread settings,
  observation staging/IPC, optimization, and critic replay. Repeated evaluation
  startup is significant on tiny panels; zero workers remains useful there.
  Existing `scripts/bench_model_ppo.py` covers model/update microbenchmarks.
- Validate on representative trained checkpoints and target accelerator hardware.
  Local CPU results do not establish GPU throughput or learning equivalence for
  any future approximations.

The full goal is not complete.

## GPU target availability

The local torch build reports no usable CUDA device. A read-only SSH status
check of the older documented endpoint (137.175.76.24:49022) could not establish
a session: the endpoint closed the connection. This does not establish whether
the historical training process is running or stopped. A current accessible
GPU target/run has been requested for representative profiling; no remote
training process was changed.

## Broad regression result

The complete non-slow repository suite passed after the scoped-thread,
critic-only, omitted-mask, and timing changes: **1,151 passed, 3 skipped** in
339.95 seconds. The long final segment was the existing end-to-end supervised
test, which trains the default 384-wide / 12-layer model for two epochs on CPU.
The updated four-worker AsyncVectorEnv PPO smoke run also completed update 2 /
256 steps and saved finite model and Adam tensors after mask omission.

The replay diagnostic transfer issue identified during the audit is addressed
below; accelerator throughput validation remains outstanding.

## Batched diagnostic transfers and descriptor preparation

Accelerator replay diagnostics now pack outcome probabilities, win probability,
and three small label/bucket fields into one CPU transfer before calculating
bucket masks, reductions, and scalar metrics. This removes per-bucket device
synchronization from that reporting path. CPU callers retain their existing
calculation path. A default batch's packed FP32 payload is 13 KiB.

Rollout log probabilities, values, chosen-action probabilities, macro maxima,
and survival forecasts now share one accelerator-to-CPU transfer instead of
five. Actions retain their integer transfer. CPU statistics keep zero-copy
NumPy views. GPU throughput gains are not claimed without hardware profiling.

Tests compare packed and original metrics for empty/singleton/mixed buckets
and FP32/FP64 inputs, and verify rollout statistic values/dtypes and CPU views.
The transfer/replay/critic/entropy suite passed 76 tests with two CUDA-only
checks skipped. A fresh four-training-worker/two-evaluation-worker PPO smoke
run completed update 2 / 256 steps with finite model and Adam tensors.

The current CPU profile also identified abstract Mapping checks during card
preparation. Native dicts now take a fast path; custom mappings and None retain
the original behavior. Two alternating 100,000-call runs on a 52-card descriptor
fixture measured 1.112/1.104 seconds for the original helper and 0.298/0.296 for
the initial fast path. The final tuple-based type check measured 0.322/0.313
seconds (about 3.5x versus the reference). This is an isolated helper measurement, not a claimed
end-to-end improvement. All 96 scoring/cache/hand-plan/risk/reward tests passed.

## Evaluation observation transport

Direct inspection corrected the earlier transport assumption: environment
observations contain int8 action masks, not the float32 masks used in replay
materialization. A reset observation serializes to roughly 53 KiB, with 17,327
bytes in its legality mask. Workers now bit-pack that binary mask to 2,166
bytes before sending and restore its exact shape, values, and int8 dtype in
the parent. Serialized observation size drops to about 38 KiB. Local evaluation
keeps the original observation path.

The same 64-game, batch-32, four-worker fixture measured:

| Transport | Wall seconds |
| --- | --- |
| Original | 6.8304, 6.8902 |
| Bit-packed masks, final repeat | 6.4358, 6.3505, 6.3737 |

The final median is 6.3737 seconds, approximately 7% below the recent baseline.
Every non-timing outcome field matched. Whole-reply zlib compression and an
explicit protocol-5 variant were also measured; both were slower than mask
packing alone, so neither is included.

22 transport, parallel/batched evaluation, and thread-scope tests passed.
Round-trip tests cover empty/full/random masks and the final action bit,
including every observation field's dtype/shape, serialization, and absence of
input mutation. Existing greedy/sampled parity and worker-failure cleanup tests
also passed on the final transport.

Representative accelerator and trained-policy profiling remain outstanding.

Final checkpoint audit found no compatible trained policy locally: the root
best-evaluation checkpoint uses tokenizer version 2, the archived PPO best
uses version 7, and the supervised checkpoint uses version 5; the current
schema is version 13. The root update-600 checkpoint is an unsupported raw
state dictionary. Strict loading rejects all four. Behavior-changing actor
transfer would not establish representative trained-policy performance.
CUDA remains unavailable locally and the documented remote SSH endpoint
closed the connection. Further production validation needs an accessible GPU
run with a current-schema trained checkpoint.

Evaluation batches contain the current decisions from active environments.
No exhaustive legal-candidate simulation or PPO-only math/RNG approximation
was introduced.
