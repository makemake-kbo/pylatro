# Public search-guided PPO continuation

The standalone runner compares ordinary PPO with PPO plus public sampled-world actor supervision, starting from `incumbents/ppo_replica1/checkpoint.pt`. Both branches preserve the parent weights and Adam state and collect their own sampled trajectories through the player's `PublicRunner` and public-information-v2 observations. Historical training defaults are unchanged.

```sh
.venv/bin/python tools/search_guided_ppo_study.py prepare
.venv/bin/python tools/search_guided_ppo_study.py run
.venv/bin/python tools/search_guided_ppo_study.py status
.venv/bin/python tools/search_guided_ppo_study.py report
```

Run with GPU access. Scientific training requires MPS FP32 and rejects `PYTORCH_ENABLE_MPS_FALLBACK=1`. CPU simulation and evaluation use at most four workers, each with one Torch thread. Preparation runs correctness checks, including actual-checkpoint CPU/MPS likelihood and gradient agreement, checks the compiled executor, audits retained and archived seed metadata without reading reserved outcome files, and freezes sources, recipes, checkpoint hashes, and disjoint integer seed panels. The 24-hour clock starts only with `run`'s technical pilot. Repeating `run` resumes the original clock; it never allocates a new budget.

Each PPO update collects 64 complete games in 32 logical environments, draws 8,192 training rows, and uses two epochs with logical batch 640 and microbatch 160. The parent learning rate, GAE, clipping, entropy, critic supervision, and KL limits are retained. Actual terminal rewards remain ten for an Ante-8 win and zero otherwise. Search actions never replace trajectory actions or acquire their rewards or advantages.

Guided cohorts retain the two smallest SHA-256 `(game seed, decision index)` keys among public hand-phase states in Antes 1–3 and two in Antes 4–8, before inspecting outcomes. Unsupported greedy proposals are rejected without substitution. A single six-candidate shortlist is compared in independent 16-world batches starting at indices 0 and 10,000, with ten seconds per batch and the existing continuation caps and score priorities. Both batches must finish all comparisons, have the same unique alternative winner, and improve blind-clear frequency over the greedy proposal by at least 2/16. The minimum improvement weights the accepted target. This is a consistency filter, not a confidence guarantee.

One independently seeded draw of 64 accepted labels contributes `0.02 × weighted action NLL` per logical optimizer group. Small accepted sets use replacement; empty sets contribute no loss. The actor and shared backbone receive these gradients; the critic head has no direct guidance loss. The existing optimizer step, gradient clipping, and KL checks remain in control. Labels are refreshed every cohort, and the separate label RNG is checkpointed.

The pilot runs one complete update per arm plus eight independent evaluation-timing games per mode. Its weights are never used for scientific training. At least 16 accepted pilot labels from eight games are required. Duration is locked to the largest feasible choice of 20, 12, or 8 updates using runtimes only, all possible training/evaluation/diagnostic work, and a 25% margin. Pilot wins are not reported before this lock. Development and confirmation gates, seed-paired bootstrap rules, and assisted and early-only attribution measurements are frozen in `frozen.json`.

Artifacts live in `runs/search_guided_ppo_20261004/`: immutable preparation metadata, `clock.json`, `duration.json`, branch `initial.pt` and `update_N.pt` checkpoints, public cohort/label records, evaluation games, fixed-state diagnostics, and `report.md`/`report.json`. A checkpoint rename commits a complete update. Resume checks recipe/provenance, RNGs, seed cursors, optimizer exposure, and committed evidence. Failed attempts are retained. Missing games stay incomplete. A KL rejection or partial optimizer exposure stops the comparison. CPU RNG restoration is exact; independent MPS executions are checked within tight FP32 tolerances because Metal reductions can differ by a few ULPs.

The final report distinguishes improved greedy weights, added benefit from guidance, and assisted strength. Guided simulation cost is additional compute, reported separately. Until the relevant confirmation gate succeeds, retain the incumbent command. Current-blind simulation scores do not establish optimal long-term strategy or a path to 90% wins.
