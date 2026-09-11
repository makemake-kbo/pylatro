# pylatro

`pylatro` is a headless Python reimplementation of Balatro's game engine (`1.0.1o-FULL` mechanics).

The Python package in [`src/pylatro`](/Users/makemake/Documents/code/python/pylatro/src/pylatro) focuses on deterministic run generation, state transitions, and action handling so the project can back a Gym-style environment for research.

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

- game data for blinds, decks, cards, centers, stakes, tags, and seals
- Balatro-compatible pseudohash / pseudoseed state handling
- headless run initialization with stake modifiers and deck effects
- blind, voucher, tag, boss, shop, reroll, and pack generation logic
- ante progression through endless mode with Balatro-style `nan` overflow behavior in very large blind amounts
- headless hand evaluation and scoring for a growing subset of card, edition, deck, and joker interactions
- joker instance tracking with preserved in-run order for copy and position-sensitive effects
- differential and behavior tests over the deterministic core

The [wiki rules audit](docs/audits/2026-09-11-wiki-rules.md) records verified
mechanics, regression coverage, and known fidelity gaps. Pack consumable handling
and Perishable/Rental lifecycles still differ from Balatro; strict expected-failure
tests track these discrepancies.

## Seed search

`pylatro seed-search spec.json` finds run seeds whose vouchers, shops, blinds,
and skips match a JSON spec (e.g. "Charm tag on the ante-1 big blind whose Mega
Arcana pack contains The Soul → Perkeo, then Blueprint within 2 shop rolls in
ante 2"). Specs are validated against game rules before searching. See
[`docs/seed_search.md`](docs/seed_search.md).

## Seed walk

Seed walk is the interactive companion to seed search. Launch the terminal UI
and pick **Seed Walk** to step through a seed's shops and blind/ante selection
by hand — without playing any blinds — with free rerolls, a "how many rerolls
until this joker?" lookup, a per-ante voucher preview, and a report you can pin
findings into and export back as a seed-search spec. Buying/selling a joker
blocks it from later rolls exactly as in a real run. See
[`docs/seed_walk.md`](docs/seed_walk.md).

```bash
pylatro          # launch the TUI, then choose "Seed Walk"
```

---

## Pylatro Agent

Transformer-based RL agent that learns to play Balatro via supervised pretraining from a heuristic agent, then PPO fine-tuning.

### Architecture

- **Model**: 12-layer transformer encoder (~40M params), d_model=512, 8 heads
- **Observation**: Tokenized game state (cards, jokers, shop, blinds) + scalar features
- **Action space**: Flat `Discrete(71)` with masking across 6 sub-phases
- **Training**: Supervised imitation learning → PPO reinforcement learning

### Setup

```bash
uv sync --extra agent
```

### Running Post-Cython

The Cython workflow builds in-place extensions next to the Python modules in
`src/pylatro` and `src/pylatro_agent`. Once those `.so` files exist, Python will
import them automatically in preference to the `.py` sources.

```bash
# Enter the toolchain shell and install deps
nix develop
uv sync --group dev --extra agent

# Build the compiled modules
uv run python scripts/compile_cython.py

# Verify the compiled agent path
uv run pytest -q tests/test_tokenizer.py tests/test_fast_generate_regressions.py tests/test_gym_env.py

# Run the project normally; the compiled modules are picked up automatically
uv run python train.py supervised --games 100 --epochs 3 --device cpu
uv run python play.py --heuristic --games 10
```

If you change any of the Cythonized modules, rebuild before running again:

```bash
uv run python scripts/compile_cython.py
```

If you want to go back to pure Python or clear stale generated artifacts first:

```bash
uv run python scripts/compile_cython.py --clean
uv run python scripts/compile_cython.py
```

### Training

The fixed-Ante-8 archive-and-return experiment is documented in
[docs/archive_training.md](docs/archive_training.md). It adds intermediate-state
practice, explicit mutable Joker features, bounded boss-clear rewards, and an
actor-transfer path for compatible checkpoints.

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

For a run that explicitly learns to seek and use Tarots and Planets, enable
build-aware shaping, Planet alignment, and conservative exploration:

```bash
# First create the matching current-schema supervised source. Reward options are embedded
# in the checkpoint and must match PPO exactly.
uv run python train.py supervised \
  --games 5000 --epochs 10 --device cuda --win-ante 4 --gamma 0.997 \
  --outcome-loss-coeff 0.10 \
  --dense-reward-scale 0.25 \
  --strategic-event-reward-scale 1.0 \
  --score-build-potential --planet-match-shaping \
  --planet-unmatched-use-penalty-coeff 0.25 \
  --planet-unmatched-claim-penalty-coeff 0.10

uv run python train.py ppo \
  --pretrained checkpoints/supervised/supervised_epoch10.pt \
  --envs 16 --rollout-length 256 --batch 320 --micro-batch-size 160 --ppo-epochs 4 \
  --updates 2000 --device cuda --win-ante 4 \
  --gamma 0.997 --lr 3e-6 --clip-eps 0.1 \
  --outcome-loss-coeff 0.10 \
  --target-kl 0.03 --target-kl-p95 0.10 --target-kl-max 0.15 \
  --min-minibatch-fraction 0.50 \
  --dense-reward-scale 0.25 \
  --strategic-event-reward-scale 1.0 \
  --score-build-potential \
  --planet-match-shaping \
  --planet-unmatched-use-penalty-coeff 0.25 \
  --planet-unmatched-claim-penalty-coeff 0.10 \
  --entropy-coeff 0.01 --action-type-entropy-scale 0.25 \
  --danger-rollout-temperature 0.85 \
  --danger-death-probability-threshold 0.35 \
  --danger-shop-leave-logit-penalty 2.0 \
  --max-idle-steps 32 \
  --eval-games 100 --eval-interval 10 \
  --eval-regression-tolerance 0.05 --eval-regression-patience 2
```

`--score-build-potential` uses draw reliability and projected score to value Pair
and High Card as scalable fallbacks, Flush after suit fixing, and multiplicity
hands only after rank/card fixing. Two Pair receives a small conditional signal
only with a dedicated synergy Joker; Full House is not a strategic target. The
same bounded potential strongly values Blue/Purple seals, Tarot
deck fixing and money generation, score-improving Joker replacements, and
midgame Standard-pack searches when scoring is already safe. Planet shaping
rewards only these viable plans. The tokenizer also exposes a conservative,
boss-aware chance of clearing the immediate blind, five strategic opportunity
probabilities, four suit-target utilities shared with contextual Tarot rewards,
and the configured victory Ante. The critic embeds current and target
Ante together and predicts one conditional survival-hazard vector. That vector
defines a normalized distribution over death in each Ante or reaching the
target. Terminal value is the distribution-weighted outcome utility; a single
scalar residual accounts for discounting and dense shaping, and their sum is
the value used by GAE, bootstrapping, SIL, and explained variance.
Completed-blind chips are
discarded before the next shop's risk estimate. Danger, danger-weighted hands,
and danger-weighted discards feed the policy head directly; active hand
decisions are sharpened in Ante 1 or immediate danger. Unsafe shops retain
exploration but softly favor rerolling over leaving. A bounded survival
potential rewards actions that improve clear probability until the build
reaches the 65% safety threshold, then saturates.
Banked cash is valuable only when that chance is credible; unsafe rerolls and
realized, confidently modeled Joker upgrades get positive-only credit, with no
reward penalty for declining any individual offer.
One categorical terminal-outcome NLL trains the hazard model; stalls are
excluded as censored outcomes. Return Huber loss trains the residual against
`return_target - terminal_value.detach()`, so return noise cannot distort
outcome calibration. Complete-episode replay crosses rollout boundaries and
updates only the private outcome projection and hazard output layer. `win_prob` and `expected_score` remain
derived compatibility aliases for final-outcome mass and composed expected
return. The compact TensorBoard
shop/terminal metrics expose dying with money, weak full Joker slots, and
shop survival calibration. Joker ordering is not a policy decision: the harness
applies the best exact-scored arrangement before each play
(`pylatro_agent/joker_layout.py`), so `MOVE_JOKER` no longer exists in the
action space and there is no reorder loop to penalize. Action-family fractions
plus no-progress streaks make a shop loop directly visible.

Ordering optimizes for score by default and for money when - and only when -
the play still clears the blind. That is a computation, not a forecast: every
candidate order is simulated, so the exact chip total is known before
committing and `chips_forgone` can never turn a clearing play into a death.
The rule deliberately does not consult the learned critic; an ordering that
traded chips for cash on a critic's say-so would hand a control input to the
least reliable component in the system and make the transition function depend
on weights that change every update. Two scalars report what the ordering did
on the last play (objective taken, dollars banked) so harness-generated cash is
attributable to the policy rather than unexplained variance in its own dollars;
`joker_order/*` exposes the same on TensorBoard.

Every shop-leave risk forecast is resolved against its realized next-blind
outcome and appended to `<log-dir>/risk_forecasts.jsonl`. Earlier runs scored
only the *last* shop leave of each episode — about 8% of them, and the one
selected by imminent death — which made the calibration metrics read far more
optimistic than the model actually is. `tools/fit_risk_calibration.py` refits
the Platt constants in `pylatro_agent/risk.py` from those pairs; changing them
changes observation and reward semantics, so bump `REWARD_MODEL_VERSION` with
them.

Evaluation advances `--eval-batch-size` games in lockstep per policy forward
pass. Per-seed greedy results are unchanged by batching. A short idle horizon, action-family entropy bonus,
hard KL guard, and two-eval regression stop protect the pretrained policy. The
regression stop writes and mirrors an exact full resume checkpoint before
exiting, so a managed restart does not fall back to an older periodic save.
Strict loading and `--resume` require the current tokenizer schema (v12).
`--actor-transfer` accepts compatible v11/current actors for a new Ante-8 run,
with a fresh critic and optimizer; it is not a resume. Earlier schemas require
a separate migration or fresh supervised training. Full current-schema PPO
checkpoints restore model, optimizer, counters, entropy controller, RNG, and
schedule state. In-flight environments restart; configured archives persist.

Ante-1 TensorBoard reporting uses rollout-local counts. In particular,
`terminal/loss_ante/1_fraction` remains conditional on non-stall losses for
backwards compatibility, while
`terminal/ante1_death_per_nonstall_completed_episode` divides Ante-1 deaths by
all completed non-stall episodes; the adjacent `*_count` tags expose both
numerator and denominator and emit zero when no eligible episode completes.
Under `ante1/`, blind clear/death counts are split by small/big/boss. Clear
hands/discards means divide by `ante1/clear/count`; realized score progress and
hand shares divide by their adjacent play/hand denominator counts. The
`conservative_*` and `one_hand_clear_proxy/*` fields use the legal candidate
generator's Joker-blind chip proxy, not exact counterfactual scoring. Existing
chosen/best hand tags now rank only play actions admitted by the operative
action mask.

The shaped-reward CUDA baseline recipe is available as
`scripts/run_ppo_v14_safe_tarot_seal_strategy.sh` (the filename is retained for
deployment compatibility). It requires a pinned current-schema supervised checkpoint,
uses the hazard/residual critic, and may resume only a current-schema
`ppo_latest.pt`. Set `PYLATRO_SOURCE_SHA256` to the supervised checkpoint hash.
The launcher keeps a logical batch of `320` while accumulating two physical
microbatches of at most `160`, and binds checkpoints to the run UUID, source
hash, and the historical `pylatro-v8-conditional-survival-v1` recipe identity.
That recipe name and its ownership-marker filename are stable deployment
identifiers, not tokenizer versions. Use the [archive experiment guide](docs/archive_training.md)
for the fixed-Ante-8 archive/milestone recipe.

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

#### Live Balatro

Control a manually started vanilla Balatro run through the Steamodded bridge:

```bash
uv run --extra agent python play.py --live --heuristic
uv run --extra agent python play.py --live --checkpoint checkpoints/ppo/ppo_update100.pt
```

See [the live bridge setup and troubleshooting guide](docs/live_bridge.md).

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
├── value_head.py         # Win prob + expected score prediction
├── distributions.py      # Masked categorical distribution
└── training/
    ├── supervised.py     # Phase 1: imitation learning
    ├── ppo.py            # Phase 2: PPO training
    ├── rollout_buffer.py # Experience storage for PPO
    └── self_play.py      # Phase 3: self-play (WIP)
```
