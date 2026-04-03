# pylatro

`pylatro` is a headless Python porting workspace for Balatro `1.0.1o-FULL`.

The upstream shipped Lua sources are unpacked into [`vendor/balatro_lua`](/Users/makemake/Documents/code/python/pylatro/vendor/balatro_lua) and treated as the reference implementation. The Python package in [`src/pylatro`](/Users/makemake/Documents/code/python/pylatro/src/pylatro) focuses on deterministic run generation, state transitions, and action handling so the project can later back a Gym-style environment for research.

## Development

```bash
nix develop
uv sync --group dev
```

The flake provides Python 3.12, `uv`, and a native C toolchain so Cython
extensions can be compiled in-place later.

For the RL stack as well:

```bash
uv sync --group dev --extra agent
```

The current implementation includes:

- upstream Lua table extraction for blinds, decks, cards, centers, stakes, tags, and seals
- Balatro-compatible pseudohash / pseudoseed state handling
- headless run initialization with stake modifiers and deck effects
- blind, voucher, tag, boss, shop, reroll, and pack generation logic
- ante progression through endless mode with Balatro-style `nan` overflow behavior in very large blind amounts
- headless hand evaluation and scoring for a growing subset of card, edition, deck, and joker interactions
- joker instance tracking with preserved in-run order for copy and position-sensitive effects
- differential and behavior tests over the deterministic core

---

## Pylatro Agent

Transformer-based RL agent that learns to play Balatro via supervised pretraining from a heuristic agent, then PPO fine-tuning.

### Architecture

- **Model**: 8-layer transformer encoder (~10M params), d_model=256, 8 heads
- **Observation**: Tokenized game state (cards, jokers, shop, blinds) + scalar features
- **Action space**: Flat `Discrete(71)` with masking across 6 sub-phases
- **Training**: Supervised imitation learning → PPO reinforcement learning

### Setup

```bash
uv sync --extra agent
```

### Training

#### Phase 1: Supervised Pretraining

Generates games from a rule-based heuristic agent and trains the model to imitate it.

```bash
# Quick test (100 games, 3 epochs)
uv run python train.py supervised --games 100 --epochs 3 --device mps

# Full pretraining (5000 games, 10 epochs)
uv run python train.py supervised --games 5000 --epochs 10 --device mps
```

Checkpoints saved to `checkpoints/supervised/`. TensorBoard logs in `runs/supervised/`.

#### Phase 2: PPO Fine-Tuning

Reinforcement learning from self-play, starting from the pretrained checkpoint.

```bash
# From pretrained checkpoint
uv run python train.py ppo \
  --pretrained checkpoints/supervised/supervised_epoch10.pt \
  --envs 16 --steps 4000000 --device mps

# Shorter run for testing
uv run python train.py ppo \
  --pretrained checkpoints/supervised/supervised_epoch5.pt \
  --envs 8 --steps 500000 --device mps
```

Checkpoints saved to `checkpoints/ppo/`. TensorBoard logs in `runs/ppo/`.

#### Monitoring

```bash
tensorboard --logdir runs/
```

### Docker (vast.ai)

Build and push:

```bash
docker build -t pylatro .
docker tag pylatro your-registry/pylatro:latest
docker push your-registry/pylatro:latest
```

On vast.ai, mount a persistent volume to `/data` so checkpoints and TensorBoard logs survive container restarts:

```bash
# Supervised pretraining
docker run --gpus all -v /workspace:/data pylatro \
  uv run python train.py supervised \
    --games 5000 --epochs 10 --device cuda \
    --checkpoint-dir /data/checkpoints/supervised \
    --log-dir /data/runs/supervised

# PPO fine-tuning
docker run --gpus all -v /workspace:/data pylatro \
  uv run python train.py ppo \
    --pretrained /data/checkpoints/supervised/supervised_epoch10.pt \
    --envs 16 --steps 4000000 --device cuda \
    --checkpoint-dir /data/checkpoints/ppo \
    --log-dir /data/runs/ppo
```

TensorBoard (run alongside training):

```bash
docker run --gpus all -v /workspace:/data -p 6006:6006 pylatro \
  uv run tensorboard --logdir /data/runs --host 0.0.0.0
```

### Playing

#### Trained model

```bash
# Evaluate a checkpoint over 20 games
uv run python play.py --checkpoint checkpoints/ppo/ppo_update100.pt --games 20

# Evaluate supervised checkpoint
uv run python play.py --checkpoint checkpoints/supervised/supervised_epoch10.pt --games 50
```

#### Heuristic baseline

```bash
uv run python play.py --heuristic --games 50
```

#### Options

```
--games N       Number of games to play (default: 10)
--seed N        Starting random seed (default: 0)
--device DEV    cpu, mps, or cuda (default: auto-detect)
```

### Agent Structure

```
src/pylatro_agent/
├── agent.py              # Top-level BalatroAgent nn.Module
├── env.py                # Gymnasium environment wrapping GameController
├── heuristic.py          # Rule-based agent for pretraining data
├── action.py             # Action encoding/decoding
├── masks.py              # Valid action mask computation
├── reward.py             # Reward shaping functions
├── tokenizer.py          # RunState → token/scalar arrays
├── vocab.py              # Vocabulary built from GameData
├── constants.py          # Action layout, dimensions, sub-phases
├── embeddings.py         # Per-token-type embedding layers
├── backbone.py           # Transformer encoder
├── action_heads.py       # Per-sub-phase action heads
├── value_head.py         # Win prob + expected score prediction
├── distributions.py      # Masked categorical distribution
└── training/
    ├── supervised.py     # Phase 1: imitation learning
    ├── ppo.py            # Phase 2: PPO training
    ├── rollout_buffer.py # Experience storage for PPO
    └── self_play.py      # Phase 3: self-play (WIP)
```
