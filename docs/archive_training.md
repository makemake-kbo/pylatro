# Archive-and-return PPO experiment

This experiment keeps Ante 8 as the victory condition and gives the policy
repeated practice from reachable intermediate shops and blind selections.
It retains the transformer and structured policy, adds explicit mutable Joker
features, and replaces strategic event shaping with a bounded progress reward.

## Start a run

Rebuild the Cython extensions after updating the source:

```bash
uv run python scripts/compile_cython.py
```

Transfer a compatible existing actor into a new run directory:

```bash
uv run python train.py ppo \
  --actor-transfer checkpoints/your_previous_run/ppo_best_eval.pt \
  --archive-return --win-ante 8 \
  --archive-return-probability 0.5 \
  --archive-min-ante 4 --archive-max-ante 8 \
  --archive-capacity-per-bucket 8 \
  --milestone-reward-budget 1.0 \
  --milestone-final-scale 0.0 --milestone-decay-fraction 0.8 \
  --envs 16 --rollout-length 256 --batch 320 --micro-batch-size 160 \
  --ppo-epochs 4 --updates 2000 --device cuda \
  --lr 3e-6 --gamma 0.997 \
  --eval-games 100 --eval-interval 10 \
  --checkpoint-dir checkpoints/ppo_archive_ante8 \
  --log-dir runs/ppo_archive_ante8
```

Use the source model's `--d-model`, `--n-layers`, `--n-heads`, and `--d-ff`
if they differ from the defaults. This is an initial experiment recipe, not
a claim that these hyperparameters are optimal. No training starts merely by
checking out the change.

`--archive-return` selects `--reward-objective milestone` by default. Explicit
`--reward-objective shaped` retains the previous reward for an archive-only
ablation. `--reward-objective milestone` without `--archive-return` is the
matched fresh-start baseline. Archive/milestone mode requires Ante 8.

## What is saved and sampled

Each training worker owns a bounded archive. A snapshot contains the controller,
engine RNG, card identities and pile aliases, played-hand history, phase,
last Joker-order outcome, idle count, and reward bookkeeping. Immutable game
data is shared rather than duplicated in each snapshot. Policy weights,
actions, and old action probabilities are not part of a snapshot.

Workers record transitions into shops and blind-selection boundaries in the
configured Ante range. Each Ante/phase bucket uses reservoir replacement;
returns select an Ante uniformly, then a phase, then a saved state. Repeated
visits to the same seed/Ante/phase/blind replace that entry instead of filling
the bucket with duplicate openings. The default limit is 80 snapshots per
worker (5 Antes × 2 phases × 8 entries).

On an ordinary training reset, the worker returns to an archive state with
probability 0.5 when its archive is nonempty. Otherwise it starts a fresh run.
This is a fraction of **starts**, not a fraction of environment steps. The
archive fills from the current policy's actual trajectories; it does not
construct synthetic strong builds or filter entries to winners.

The current policy generates new actions after a return, with the same rollout
and PPO probability calculation used for fresh runs. A continuation episode
contains only its newly collected suffix. Its return never includes rewards
or actions from the archived prefix. A restored game's future RNG is unchanged;
diversity comes from seed coverage and new policy actions, not invented card draws.

## Rewards

In milestone mode, intermediate bosses 1–7 each pay `budget / 7` on their first
clear. Ante 8 pays the true win reward of 10. Ordinary failures and stalls pay
zero terminal reward. Invalid actions retain the environment's existing error
penalty. Cash, seal, Planet, Tarot, reroll, tempo, and potential rewards are
disabled in this mode regardless of their legacy coefficient flags.

The set of already-paid bosses is saved in every snapshot. Restoring a state
pays nothing; reclearing a previously paid Ante after an Ante-lowering voucher
also pays nothing. No continuation can receive the original game's past rewards.

The milestone multiplier starts at 1 and linearly approaches
`--milestone-final-scale` over `--milestone-decay-fraction` of the saved training
step horizon. Its default is zero after 80% of that horizon. It is a global
training schedule, so returns do not rewind it. Use a final scale of 1 for a
constant-reward control. Progress reward temporarily changes the objective;
judge the experiment by actual full-run wins, not the shaped training return.

The outcome critic uses zero utility for death and 10 for an Ante-8 win in
milestone mode. Its existing return residual accounts for progress rewards
and discounting. Changing the actor/critic topology is a separate experiment.

## Checkpoints and transfer

Tokenizer v12 appends 35 numeric Joker fields to the original 12 token fields.
These include separate chip/Mult/XMult channels, growing/decaying `extra`
values, growth increments, and activation counters. Values use signed log
encoding. The appended feature projection starts at zero so transferred actors
can initially reproduce their prior behavior and learn to use the added data.

`--actor-transfer` accepts tokenizer-v11 or current-schema checkpoints with
matching actor parameter shapes. It copies the actor and shared representation,
keeps a freshly initialized critic, and starts new optimizer state and counters.
For a source trained to an earlier target, the source target-Ante embedding is
copied to the Ante-8 row. The checkpoint records the source path, hash, and schema.
Tokenizers before v11 have additional incompatible observation/action changes
and are rejected; they require a separate migration or current-schema pretraining.

Full PPO checkpoints include every worker's archive, reservoir counts, and
archive RNG. Resume with the same archive settings, number of environments,
reward settings, and milestone schedule, replacing `--actor-transfer` with
`--resume checkpoints/ppo_archive_ante8/ppo_latest.pt`. As in the existing
trainer, in-flight episodes restart on process resume; archives persist.
The saved annealing horizon is preserved unless `--reset-schedules` is passed.
Archive payloads use pickle and should only be loaded from trusted local runs.

## Evaluate the experiment

Evaluation creates fresh environments without archives. Explicit seeded resets
also always start fresh. Only full-run evaluation chooses `ppo_best_eval.pt`.
Keep a separate unseen seed set for final assessment.

TensorBoard adds:

- `archive/states`, `archive/states_anteN`, `archive/returns`, and
  `archive/fresh_starts` for archive coverage and usage.
- `curriculum/fresh_win_rate` and `curriculum/archive_win_rate`, measured over
  the last 100 completed episodes of each origin, plus cumulative episode counts.
- `curriculum/survive_anteN` for realized survival conditional on reaching each
  Ante, over its last 100 uncensored outcomes. This mixes fresh and archive
  suffixes and must not be interpreted as full-run success.
- `curriculum/milestone_scale` and `reward_boss_milestone` in the existing
  reward-component diagnostics.

Compare the archive run with a fresh-start milestone run from the same actor
and at equal environment-step and wall-clock budgets. If conditional Ante-6
survival improves without full-run wins improving, move the minimum archive
Ante earlier in a new experiment to practice the build decisions that precede
the failure. Do not call archive-start win rate the model's game win rate.
