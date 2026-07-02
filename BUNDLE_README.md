# pylatro training bundle

Packaged so a downstream agent can diagnose what is going wrong with training.
This repo is a headless Python reimplementation of the Balatro card-game engine,
plus a transformer-based RL agent (supervised pretraining -> PPO fine-tuning).

## What's included

### Code
- `src/pylatro/`         - game engine (state, scoring, shop, jokers)
- `src/pylatro_agent/`   - model + supervised/PPO training code
- `src/pylatro_cli/`     - CLI
- `train.py`             - training entrypoint (phases: supervised | ppo | self_play | pretrain_from_checkpoint)
- `tests/`               - test suite (pytest)
- `tools/`               - `evaluate_checkpoint.py`, `ppo_health.py`
- `scripts/`             - cython compile, engine hotpath bench
- `bench_*.py`           - analysis / diagnostic scripts (death analysis, xmult, score gap, joker, etc.)
- `extract_tb_logs.py`   - pulls metrics out of TensorBoard event files
- `docs/`, `pyproject.toml`, `uv.lock`, `flake.nix`, `Dockerfile`

### Training data (main thing to analyze)
`runs/` - TensorBoard event files (`events.out.tfevents.*`) plus one `resume_train.log`:
- `runs/supervised/`                                    - supervised pretraining metrics
- `runs/ppo/`                                           - early PPO
- `runs/ppo_ft_ante4_shop_reward_v1/`                   - base shop-reward finetune
- `runs/ppo_ft_ante4_shop_reward_v1_resume/`
- `runs/ppo_ft_ante4_shop_reward_v1_resume_nodistill_long/`
- `runs/ppo_ft_ante4_shop_reward_v1_resume_nodistill_long_r500/`
- `runs/ppo_ft_ante4_entropy_rewardfix_main/`
- `runs/ppo_ft_ante4_recover_v2/`
- `runs/ppo_ft_ante4_recover_v3/`

Run names encode the experiment lineage (finetune from ante-4 checkpoint, shop-reward
tweak, entropy/reward fix, then two "recover" attempts). Newer/recover runs are the
ones most likely to show the current problem.

## What is NOT included (excluded for size)
- `checkpoints/` - model weights (~9.2 GB). Ask for separately if weight inspection is needed.
- `.venv/`, `build/`, `.git/`, `__pycache__/`, caches.

## Architecture (from README)
- Model: 12-layer transformer encoder (~40M params), d_model=512, 8 heads
- Observation: tokenized game state (cards, jokers, shop, blinds) + scalar features
- Action space: flat `Discrete(71)` with masking across 6 sub-phases
- Pipeline: supervised imitation learning -> PPO RL

## How to read the training logs
TensorBoard event files. Either:

```bash
pip install tensorboard
tensorboard --logdir runs
```

or from Python:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ea = EventAccumulator("runs/ppo_ft_ante4_recover_v3")
ea.Reload()
print(ea.Tags()["scalars"])        # available scalar names
series = ea.Scalars("rollout/ep_rew_mean")  # .wall_time, .step, .value
```

`extract_tb_logs.py` in the root does similar extraction.

## Default PPO config (from train.py + runs/.../resume_train.log)
- total_timesteps up to 30,000,000; steps_per_update 4096 (8 envs x 256 rollout) -> ~7325 planned updates
- ppo_epochs 4, lr 1e-4, entropy_coeff 1e-3, target_kl 0.05
- eval_interval 20, checkpoint_interval 50, max_no_progress_steps 256
- device: mps (Apple Silicon)

## Suggested analysis focus
1. Compare return/reward curves across `ppo_ft_ante4_*` (esp. recover_v2 vs recover_v3 vs entropy_rewardfix_main).
2. Watch for entropy collapse and KL-divergence blowups / early clipping.
3. Check whether fine-tuning regressed vs the supervised baseline (`runs/supervised/`).
4. Look at value-loss / advantage norms and explained-variance if tagged.
5. Run/inspect `tools/ppo_health.py` for the project's own health checks, and `bench_*.py` diagnostics.
