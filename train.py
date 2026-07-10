#!/usr/bin/env python3
"""Training entrypoint for the Balatro agent.

Usage:
    uv run python train.py supervised [--games 1000] [--epochs 5] [--device mps]
    uv run python train.py ppo [--pretrained PATH] [--steps 1000000] [--device mps]
    uv run python train.py self_play [--pretrained PATH] [--device mps]
    uv run python train.py pretrain_from_checkpoint \\
        --inference-checkpoint PATH [--min-ante 5] [--games 1000] [--data-path PATH]
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from pylatro_agent._cython_check import check_cython_freshness  # noqa: E402

check_cython_freshness()


def main():
    parser = argparse.ArgumentParser(description="Train the Balatro agent")
    parser.add_argument(
        "phase",
        choices=["supervised", "ppo", "self_play", "pretrain_from_checkpoint"],
    )
    parser.add_argument("--games", type=int, default=1000, help="Heuristic games for supervised (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, help="Supervised epochs (default: 5)")
    parser.add_argument(
        "--supervised-entropy-coeff",
        type=float,
        default=0.001,
        help="Entropy bonus coefficient for supervised behavior cloning (default: 0.001)",
    )
    parser.add_argument(
        "--hand-ar-mixture-eps",
        type=float,
        default=None,
        help=(
            "Mixture weight on the autoregressive hand/discard head (Phase 1). 0.0 "
            "reproduces the historical candidate-only support; the default depends on "
            "phase: 0.5 for supervised (trains the AR head on every label), 0.1 for PPO "
            "(targeted exploration). Pass explicitly to override."
        ),
    )
    parser.add_argument(
        "--outcome-weight-beta",
        type=float,
        default=1.5,
        help=(
            "Phase 5: AWR-style outcome weight beta for BC. w = exp(beta * normalized_outcome) "
            "clamped to [0.25, 4]. Tilts imitation toward successful trajectories while "
            "preserving full state coverage. 0.0 = uniform weights (no tilt). Default: 1.5."
        ),
    )
    parser.add_argument(
        "--min-ante",
        type=int,
        default=1,
        help=(
            "Minimum ante a heuristic game must reach to be kept for supervised pretraining. "
            "Phase 5: default changed from 5 to 1 (no filter) — hard outcome filtering creates "
            "survivorship bias. Outcome weighting (--outcome-weight-beta) replaces the filter."
        ),
    )
    parser.add_argument("--steps", type=int, default=1_000_000, help="PPO total timesteps (default: 1000000)")
    parser.add_argument("--envs", type=int, default=8, help="Parallel envs for PPO (default: 8)")
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=256,
        help="PPO rollout length per env before each update (default: 256)",
    )
    parser.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update (default: 4)")
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint (weights-only init; optimizer/counters start fresh)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Strictly resume a PPO run: restores model weights, optimizer (Adam moments), "
            "update counter, total_steps, entropy controller, RNG state, and return normalization. "
            "Requires a full PPO checkpoint (checkpoint_format == 'ppo_full'). Use --pretrained "
            "for weights-only initialization instead."
        ),
    )
    parser.add_argument(
        "--additional-updates",
        type=int,
        default=None,
        help=(
            "With --resume: run this many more PPO updates after the checkpoint's saved update "
            "count. The target update count becomes saved_update_count + N. Requires --resume."
        ),
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help=(
            "Run exactly N PPO updates from scratch, converting to N * envs * rollout_length "
            "timesteps internally. Overrides --steps. Combine with --pretrained for weights-only "
            "fine-tuning of a fixed length, or with --resume to set the total target instead of --additional-updates."
        ),
    )
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, mps, cuda (default: auto-detect)")
    parser.add_argument("--lr", type=float, default=5e-5, help="PPO learning rate (default: 5e-5)")
    parser.add_argument(
        "--clip-eps",
        type=float,
        default=0.1,
        help=(
            "PPO clipped-surrogate range (default: 0.1, tighter than the usual 0.2). "
            "Raise toward 0.2 to let updates move the policy more when approx_kl sits "
            "well below --target-kl."
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda for advantage estimation (default: 0.95).",
    )
    parser.add_argument(
        "--value-loss-coeff",
        type=float,
        default=0.25,
        help="Weight on the critic (value) loss (default: 0.25).",
    )
    parser.add_argument("--d-model", type=int, default=384, help="Model dimension (default: 384)")
    parser.add_argument("--n-layers", type=int, default=12, help="Transformer layers (default: 12)")
    parser.add_argument("--n-heads", type=int, default=8, help="Transformer attention heads (default: 8)")
    parser.add_argument("--d-ff", type=int, default=1536, help="Transformer feed-forward dimension (default: 1536)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Base dir for checkpoints (default: checkpoints/<phase>)",
    )
    parser.add_argument("--workers", type=int, default=0, help="CPU workers for game generation (default: all cores)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Generate games in chunks of this size, writing each to disk to limit memory (default: 10000)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Base dir for TensorBoard logs (default: runs/<phase>)",
    )
    parser.add_argument(
        "--sync-envs",
        action="store_true",
        help=(
            "Use SyncVectorEnv instead of AsyncVectorEnv for PPO rollouts. "
            "Async is faster but worker exceptions surface as EOFError/"
            "BrokenPipe in the parent with no traceback. Pass --sync-envs "
            "to reproduce env-worker crashes synchronously in the parent "
            "process for debugging. Default: async."
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="PPO console log interval in updates (default: 10)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="PPO checkpoint interval in updates (default: 50)",
    )
    parser.add_argument("--eval-interval", type=int, default=5, help="PPO eval interval in updates (default: 5)")
    parser.add_argument(
        "--max-idle-steps",
        type=int,
        default=256,
        help="Terminate PPO episodes only after this many consecutive no-progress steps (default: 256)",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=0.15,
        help="PPO target normalized entropy ratio in [0, 1] when adaptive entropy is enabled (default: 0.15)",
    )
    parser.add_argument(
        "--win-ante",
        type=int,
        default=None,
        help=(
            "Curriculum: cap the run's victory ante below the engine default of 8. "
            "Heuristic-teacher win rates: ~39%% at ante 4, ~12%% at 5, ~2%% at 6, "
            "~1%% at 8. Lower values give PPO frequent positive-reward terminals "
            "to learn from; raise as the policy improves. Default: engine default (8)."
        ),
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Games per PPO eval pass (default: 50). Wins are rare; small samples are noisy.",
    )
    parser.add_argument(
        "--eval-device",
        type=str,
        default=None,
        help=(
            "Optional device override for the PPO eval pass (cpu, mps, cuda). "
            "Useful when training on MPS: eval runs serially in-process and "
            "its forward-pass allocations compete with idle AsyncVectorEnv "
            "workers for unified memory. Setting --eval-device cpu moves "
            "those allocations off the GPU entirely at the cost of slower "
            "eval. Default: same as --device."
        ),
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature applied at all PPO distribution sites (rollout sampling, "
            "training forward pass, value bootstraps). Set to 1.0 to disable sharpening "
            "(default: 1.0). The 0.7 crutch is obsolete post-Phase-1 BC."
        ),
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="PPO entropy coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.05,
        help="Stop each PPO epoch early when approximate KL exceeds this value; <=0 disables (default: 0.05)",
    )
    parser.add_argument(
        "--target-kl-p95",
        type=float,
        default=None,
        help=(
            "Optional hard guard: warn (and log ppo/stop_reason_kl_p95) when the 95th "
            "percentile minibatch approx_kl exceeds this. Does not stop training by "
            "itself; pair with --target-kl for early stopping. Default: disabled."
        ),
    )
    parser.add_argument(
        "--target-kl-max",
        type=float,
        default=None,
        help=(
            "Optional hard guard: warn when the max minibatch approx_kl exceeds this "
            "(logged as ppo/stop_reason_kl_max). Use ~0.25 to surface dangerous KL spikes. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--min-minibatch-fraction",
        type=float,
        default=None,
        help=(
            "Optional hard guard: warn when ppo/minibatch_fraction falls below this "
            "(target_kl is stopping updates too early). Use ~0.50. Default: disabled."
        ),
    )
    parser.add_argument(
        "--entropy-ema-beta",
        type=float,
        default=0.6,
        help="EMA smoothing for PPO entropy control signal (default: 0.6)",
    )
    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=1e-2,
        help="Learning rate for adaptive entropy coefficient updates (default: 1e-2)",
    )
    parser.add_argument(
        "--action-type-entropy-scale",
        type=float,
        default=0.0,
        help="Extra PPO bonus scale for normalized entropy over action types (default: 0.0)",
    )
    parser.add_argument(
        "--adaptive-entropy",
        action="store_true",
        help="Enable adaptive entropy tuning. Disabled by default to preserve pretrained policies.",
    )
    parser.add_argument(
        "--no-adaptive-entropy",
        action="store_true",
        help="Deprecated compatibility flag; adaptive entropy is disabled unless --adaptive-entropy is passed.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="PPO discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--heuristic-distill-coeff",
        type=float,
        default=0.3,
        help=(
            "Initial coefficient for the heuristic-teacher distillation loss "
            "(NLL of HeuristicAgent.select_action under the policy). Decays "
            "linearly to --heuristic-distill-min over total_timesteps. "
            "Set to <= 0 to disable distillation entirely. Default: 0.3."
        ),
    )
    parser.add_argument(
        "--heuristic-distill-min",
        type=float,
        default=0.0,
        help=(
            "Floor for the distillation coefficient after linear decay (default: 0.0). "
            "A nonzero floor anchors the policy to the heuristic forever. "
            "Ignored when --heuristic-distill-coeff <= 0. Clamped to the start "
            "coefficient if it would otherwise exceed it."
        ),
    )
    parser.add_argument(
        "--teacher-rollout-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of executing the heuristic action during PPO rollout collection "
            "when available. Useful for curriculum/DAgger-style smoke runs; default keeps "
            "strict sampled-policy rollouts."
        ),
    )
    parser.add_argument(
        "--teacher-rollout-final-prob",
        type=float,
        default=None,
        help=(
            "Optional final teacher rollout probability. When set, teacher rollout probability "
            "linearly anneals from --teacher-rollout-prob after warmup."
        ),
    )
    parser.add_argument(
        "--teacher-rollout-warmup-fraction",
        type=float,
        default=0.0,
        help="Fraction of training to keep initial teacher rollout probability before annealing (default: 0.0).",
    )
    parser.add_argument(
        "--teacher-rollout-decay-fraction",
        type=float,
        default=1.0,
        help="Fraction of training used to anneal teacher rollout probability to final prob (default: 1.0).",
    )
    parser.add_argument(
        "--dagger-bc-epochs",
        type=int,
        default=0,
        help=(
            "Online DAgger behavior-cloning epochs over each PPO rollout before the PPO update. "
            "Teacher-forced samples are imitation-only for policy learning. Default: 0."
        ),
    )
    parser.add_argument(
        "--dagger-bc-coeff",
        type=float,
        default=1.0,
        help="Multiplier for the online DAgger BC loss (default: 1.0).",
    )
    parser.add_argument(
        "--dagger-bc-lr-mult",
        type=float,
        default=1.0,
        help=(
            "Temporary learning-rate multiplier used only during online DAgger BC updates. "
            "Useful for a strong imitation phase while keeping PPO LR conservative. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--dense-reward-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier for dense shaping rewards. Terminal win/loss reward is unchanged. "
            "Set < 1.0 (e.g. 0.25) to shrink shaping so the policy optimizes winning rather "
            "than farming bounded shaping. Default: 1.0 (no change to existing behavior)."
        ),
    )
    parser.add_argument(
        "--local-hand-reward-scale",
        type=float,
        default=1.0,
        help=(
            "Group dense-scale for already-solved local hand-play shaping "
            "(hand_subset/top1/top3 bonuses). Lower (e.g. 0.10) to stop local hand "
            "play from drowning out strategic shop/economy signal. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--progression-reward-scale",
        type=float,
        default=1.0,
        help=(
            "Group dense-scale for blind-clear / ante-advance / score & pressure "
            "progress shaping. Lower to de-emphasize progression farming relative "
            "to terminal win/loss. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--shop-strategy-reward-scale",
        type=float,
        default=1.0,
        help="Group dense-scale for strategic shop buy/reroll/leave shaping. Default: 1.0.",
    )
    parser.add_argument(
        "--joker-strategy-reward-scale",
        type=float,
        default=1.0,
        help="Group dense-scale for joker slot-fill / xmult / sell shaping. Default: 1.0.",
    )
    parser.add_argument(
        "--economy-reward-scale",
        type=float,
        default=1.0,
        help="Group dense-scale for interest-progress / overspend shaping. Default: 1.0.",
    )
    parser.add_argument(
        "--consumable-reward-scale",
        type=float,
        default=1.0,
        help="Group dense-scale for planet/tarot/spectral improvement shaping. Default: 1.0.",
    )
    parser.add_argument(
        "--planet-unmatched-use-penalty-coeff",
        type=float,
        default=0.0,
        help=(
            "Per-step penalty coefficient for using a planet whose hand is not the "
            "main played hand, weighted by (1 - play_share) so leveling a strong "
            "secondary hand is nearly free, and ramped by ante progress toward "
            "win_ante so early pivot planning (leveling a hand you intend to play) "
            "is not taxed. Defaults to 0.0 (disabled). Default: 0.0."
        ),
    )
    parser.add_argument(
        "--planet-unmatched-claim-penalty-coeff",
        type=float,
        default=0.0,
        help=(
            "Per-step penalty coefficient for claiming a planet whose hand is not the "
            "main played hand, weighted by (1 - play_share) and ramped by ante "
            "progress like the use penalty. Defaults to 0.0 (disabled). Default: 0.0."
        ),
    )
    parser.add_argument(
        "--disable-shop-strategy-rewards",
        action="store_true",
        help="Disable context-aware strategic shop/economy/joker rewards (legacy flat shaping only).",
    )
    parser.add_argument(
        "--reward-v2",
        action="store_true",
        help=(
            "Use the Phase 2 PPO_V2_REWARD_CONFIG: kills all prescriptive heuristic-"
            "agreement shaping and replaces it with policy-invariant potential-based "
            "shaping. Terminal reward uses the halved v2 scale so outcome dominates "
            "return. The gamma from --gamma is plumbed into the potential telescope."
        ),
    )
    parser.add_argument(
        "--planet-match-shaping",
        action="store_true",
        help=(
            "With --reward-v2: re-enable the planet-alignment shaping component "
            "(bonuses for using/claiming planets that match played hand types, plus "
            "the --planet-unmatched-*-penalty-coeff penalties). The sparse v2 signal "
            "cannot credit-assign planet choices; without this agents drift to ~90%% "
            "unmatched planet use. No effect without --reward-v2."
        ),
    )
    parser.add_argument(
        "--build-curve-shaping",
        action="store_true",
        help=(
            "With --reward-v2: re-enable the joker build-curve shaping component — "
            "a positive-only bonus for acquiring jokers whose scoring profile fits "
            "the ante phase (chip scaling in antes 1-3, additive mult by ante 4, "
            "xmult from ante 6 or any earlier point). Which joker profile to buy "
            "when is invisible to the sparse win signal. No effect without --reward-v2."
        ),
    )
    parser.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=0,
        help=(
            "Phase 3.2: number of PPO updates to train only the critic (+survival head) "
            "with the policy frozen. After a reward-function change, advantages are "
            "garbage until the critic tracks the new return distribution; warming it up "
            "on-policy removes the window in which PPO earnestly optimizes noise. "
            "Pair with --reinit-value-head. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--critic-warmup-min-ev",
        type=float,
        default=0.7,
        help=(
            "Explained-variance gate for unfreezing the policy after critic warmup. "
            "The policy stays frozen until rollout EV clears this (capped at 4x "
            "--critic-warmup-updates). Pass 0 for a purely count-based warmup. "
            "Default: 0.7."
        ),
    )
    parser.add_argument(
        "--reinit-value-head",
        action="store_true",
        help=(
            "Phase 3.2/2.4: reinitialize the value head when loading --pretrained. "
            "Required after any reward-function change so the critic doesn't start from "
            "a stale return mapping."
        ),
    )
    parser.add_argument(
        "--reset-best-eval",
        action="store_true",
        help=(
            "With --resume: discard the checkpoint's best_eval_win_rate so "
            "ppo_best_eval.pt selection restarts from scratch. Use when the eval "
            "task changes (e.g. a --win-ante bump), otherwise no best-eval "
            "checkpoint is written until the harder task beats the old record."
        ),
    )
    parser.add_argument(
        "--sil-coeff",
        type=float,
        default=0.0,
        help=(
            "Self-imitation coefficient: behavior-clone the agent's own winning "
            "episodes from a FIFO buffer. The NLL term is added to the PPO "
            "minibatch objective, so it trades off directly against the policy "
            "loss and rides under the target-kl guard. Densifies the sparse win "
            "signal at high --win-ante targets without touching the reward "
            "function (only genuine wins enter the buffer). 0 disables. The "
            "buffer is in-memory only and refills after a resume."
        ),
    )
    parser.add_argument(
        "--sil-buffer-episodes",
        type=int,
        default=64,
        help="Max winning episodes kept in the SIL buffer (FIFO).",
    )
    parser.add_argument(
        "--sil-batch-size",
        type=int,
        default=64,
        help="SIL transitions sampled per PPO micro-batch.",
    )
    parser.add_argument(
        "--sil-min-episodes",
        type=int,
        default=8,
        help="Skip the SIL pass until the buffer holds this many wins.",
    )
    parser.add_argument(
        "--inference-checkpoint",
        type=str,
        default=None,
        help="Checkpoint used to generate games for pretrain_from_checkpoint",
    )
    parser.add_argument(
        "--inf-d-model",
        type=int,
        default=None,
        help="Model dim for the inference checkpoint (defaults to --d-model)",
    )
    parser.add_argument(
        "--inf-n-layers",
        type=int,
        default=None,
        help="Transformer layers for the inference checkpoint (defaults to --n-layers)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Load records from this path if present; otherwise save generated records here",
    )
    parser.add_argument(
        "--sample-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for sampled actions during generation (default: 1.0)",
    )
    parser.add_argument(
        "--max-no-progress-steps-gen",
        type=int,
        default=2000,
        help="Per-env stall limit during data generation (default: 2000)",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logging.info(f"Using device: {device}")

    from pylatro_agent.agent import AgentConfig
    agent_config = AgentConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads, d_ff=args.d_ff)

    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir

    if args.phase == "supervised":
        from pylatro_agent.training.supervised import SupervisedConfig, train_supervised
        sup_eps = args.hand_ar_mixture_eps if args.hand_ar_mixture_eps is not None else 0.5
        train_supervised(
            SupervisedConfig(
                num_games=args.games,
                batch_size=args.batch,
                max_epochs=args.epochs,
                action_entropy_coeff=args.supervised_entropy_coeff,
                num_workers=args.workers,
                min_ante=args.min_ante,
                chunk_size=args.chunk_size,
                log_interval=args.log_interval,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/supervised",
                log_dir=log_dir or "runs/supervised",
                hand_ar_mixture_eps=sup_eps,
                outcome_weight_beta=args.outcome_weight_beta,
            ),
            agent_config=agent_config,
        )

    elif args.phase == "ppo":
        from pylatro_agent.reward import PPO_V2_REWARD_CONFIG, RewardConfig
        from pylatro_agent.training.ppo import PPOConfig, train_ppo
        if args.resume and args.pretrained:
            parser.error("Pass either --pretrained or --resume, not both.")
        if args.additional_updates is not None and not args.resume:
            parser.error("--additional-updates requires --resume PATH.")
        if args.resume and args.updates is not None and args.additional_updates is not None:
            parser.error(
                "--updates and --additional-updates are mutually exclusive with --resume "
                "(--updates is an absolute target, --additional-updates is relative)."
            )
        if args.reward_v2 and args.pretrained and not args.reinit_value_head:
            parser.error(
                "--reward-v2 with --pretrained requires --reinit-value-head (Phase 2.4): "
                "the pretrained value head regresses returns from the old reward "
                "function, producing systematically wrong advantages that can destroy "
                "the BC policy before the critic re-converges. Pair it with "
                "--critic-warmup-updates to warm the fresh head."
            )
        # --updates N overrides --steps and is converted internally to
        # N * envs * rollout_length timesteps via PPOConfig.total_updates.
        ppo_eps = args.hand_ar_mixture_eps if args.hand_ar_mixture_eps is not None else 0.1
        train_ppo(
            PPOConfig(
                num_envs=args.envs,
                rollout_length=args.rollout_length,
                total_timesteps=args.steps,
                total_updates=args.updates,
                ppo_epochs=args.ppo_epochs,
                mini_batch_size=args.batch,
                lr=args.lr,
                clip_epsilon=args.clip_eps,
                gae_lambda=args.gae_lambda,
                value_loss_coeff=args.value_loss_coeff,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/ppo",
                log_dir=log_dir or "runs/ppo",
                log_interval=args.log_interval,
                checkpoint_interval=args.checkpoint_interval,
                eval_interval=args.eval_interval,
                max_no_progress_steps=args.max_idle_steps,
                entropy_coeff=args.entropy_coeff,
                target_kl=None if args.target_kl <= 0.0 else args.target_kl,
                target_kl_p95=args.target_kl_p95,
                target_kl_max=args.target_kl_max,
                min_minibatch_fraction=args.min_minibatch_fraction,
                adaptive_entropy=args.adaptive_entropy and not args.no_adaptive_entropy,
                target_entropy=args.target_entropy,
                entropy_ema_beta=args.entropy_ema_beta,
                alpha_lr=args.alpha_lr,
                action_type_entropy_scale=args.action_type_entropy_scale,
                gamma=args.gamma,
                async_envs=not args.sync_envs,
                heuristic_distill_coeff=args.heuristic_distill_coeff,
                heuristic_distill_min=args.heuristic_distill_min,
                teacher_rollout_prob=args.teacher_rollout_prob,
                teacher_rollout_final_prob=args.teacher_rollout_final_prob,
                teacher_rollout_warmup_fraction=args.teacher_rollout_warmup_fraction,
                teacher_rollout_decay_fraction=args.teacher_rollout_decay_fraction,
                dagger_bc_epochs=args.dagger_bc_epochs,
                dagger_bc_coeff=args.dagger_bc_coeff,
                dagger_bc_lr_mult=args.dagger_bc_lr_mult,
                rollout_temperature=args.rollout_temperature,
                win_ante=args.win_ante,
                eval_games=args.eval_games,
                eval_device=args.eval_device,
                hand_ar_mixture_eps=ppo_eps,
                critic_warmup_updates=args.critic_warmup_updates,
                critic_warmup_min_ev=args.critic_warmup_min_ev,
                reinit_value_head=args.reinit_value_head,
                reset_best_eval=args.reset_best_eval,
                sil_coeff=args.sil_coeff,
                sil_buffer_episodes=args.sil_buffer_episodes,
                sil_batch_size=args.sil_batch_size,
                sil_min_episodes=args.sil_min_episodes,
                # A RewardConfig with all scales 1.0 and strategic rewards on is
                # field-for-field identical to DEFAULT_REWARD_CONFIG, so runs
                # without these flags keep their exact prior shaping behavior.
                # --reward-v2 overrides with the Phase 2 potential-based config.
                reward_config=(
                    PPO_V2_REWARD_CONFIG(
                        gamma=args.gamma,
                        win_ante=args.win_ante or 8,
                        planet_match_shaping=args.planet_match_shaping,
                        planet_unmatched_use_penalty_coeff=args.planet_unmatched_use_penalty_coeff,
                        planet_unmatched_claim_penalty_coeff=args.planet_unmatched_claim_penalty_coeff,
                        build_curve_shaping=args.build_curve_shaping,
                    )
                    if args.reward_v2
                    else RewardConfig(
                        dense_reward_scale=args.dense_reward_scale,
                        local_hand_reward_scale=args.local_hand_reward_scale,
                        progression_reward_scale=args.progression_reward_scale,
                        shop_strategy_reward_scale=args.shop_strategy_reward_scale,
                        joker_strategy_reward_scale=args.joker_strategy_reward_scale,
                        economy_reward_scale=args.economy_reward_scale,
                        consumable_reward_scale=args.consumable_reward_scale,
                        planet_unmatched_use_penalty_coeff=args.planet_unmatched_use_penalty_coeff,
                        planet_unmatched_claim_penalty_coeff=args.planet_unmatched_claim_penalty_coeff,
                        enable_shop_strategy_rewards=not args.disable_shop_strategy_rewards,
                        enable_economy_strategy_rewards=not args.disable_shop_strategy_rewards,
                        enable_joker_context_rewards=not args.disable_shop_strategy_rewards,
                    )
                ),
            ),
            agent_config=agent_config,
            pretrained_path=args.pretrained,
            resume_path=args.resume,
            additional_updates=args.additional_updates,
        )

    elif args.phase == "self_play":
        from pylatro_agent.training.self_play import SelfPlayConfig, train_self_play
        train_self_play(
            SelfPlayConfig(
                ppo_timesteps_per_stage=args.steps,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/self_play",
            ),
            agent_config=agent_config,
            pretrained_path=args.pretrained,
        )

    elif args.phase == "pretrain_from_checkpoint":
        if not args.inference_checkpoint:
            parser.error("pretrain_from_checkpoint requires --inference-checkpoint PATH")

        from pathlib import Path

        from pylatro import load_game_data
        from pylatro_agent.training.model_generate import (
            ModelGenerateConfig,
            generate_training_data_from_model,
            load_records,
            save_records,
        )
        from pylatro_agent.training.supervised import SupervisedConfig, train_supervised
        from pylatro_agent.vocab import build_vocab

        data_path = Path(args.data_path) if args.data_path else None
        if data_path is not None and data_path.exists():
            records = load_records(data_path)
        else:
            game_data = load_game_data()
            build_vocab(game_data)  # validate vocab builds
            inf_config = AgentConfig(
                d_model=args.inf_d_model if args.inf_d_model is not None else args.d_model,
                n_layers=args.inf_n_layers if args.inf_n_layers is not None else args.n_layers,
            )
            gen_log_dir = log_dir or "runs/pretrain_from_checkpoint"
            records = generate_training_data_from_model(
                ModelGenerateConfig(
                    checkpoint_path=args.inference_checkpoint,
                    num_games=args.games,
                    num_envs=args.envs,
                    min_ante=args.min_ante,
                    device=device,
                    sample_temperature=args.sample_temperature,
                    max_no_progress_steps=args.max_no_progress_steps_gen,
                    async_envs=not args.sync_envs,
                    log_dir=gen_log_dir,
                ),
                agent_config=inf_config,
                data=game_data,
            )
            if data_path is not None:
                save_records(records, data_path)

        train_supervised(
            SupervisedConfig(
                num_games=args.games,
                batch_size=args.batch,
                max_epochs=args.epochs,
                num_workers=args.workers,
                min_ante=args.min_ante,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/pretrain_from_checkpoint",
                log_dir=log_dir or "runs/pretrain_from_checkpoint",
            ),
            agent_config=agent_config,
            records=records,
        )


if __name__ == "__main__":
    main()
