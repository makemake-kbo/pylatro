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


def main():
    parser = argparse.ArgumentParser(description="Train the Balatro agent")
    parser.add_argument(
        "phase",
        choices=["supervised", "ppo", "self_play", "pretrain_from_checkpoint"],
    )
    parser.add_argument("--games", type=int, default=1000, help="Heuristic games for supervised (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, help="Supervised epochs (default: 5)")
    parser.add_argument(
        "--min-ante",
        type=int,
        default=5,
        help="Minimum ante a heuristic game must reach to be kept for supervised pretraining (default: 5)",
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
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, mps, cuda (default: auto-detect)")
    parser.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate (default: 1e-4)")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension (default: 512)")
    parser.add_argument("--n-layers", type=int, default=12, help="Transformer layers (default: 12)")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Base dir for checkpoints (default: checkpoints/<phase>)",
    )
    parser.add_argument("--workers", type=int, default=0, help="CPU workers for game generation (default: all cores)")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Base dir for TensorBoard logs (default: runs/<phase>)",
    )
    parser.add_argument("--sync-envs", action="store_true", help="Use SyncVectorEnv instead of AsyncVectorEnv for PPO")
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
    parser.add_argument("--eval-interval", type=int, default=50, help="PPO eval interval in updates (default: 50)")
    parser.add_argument(
        "--max-idle-steps",
        type=int,
        default=256,
        help="Terminate PPO episodes only after this many consecutive no-progress steps (default: 256)",
    )
    parser.add_argument(
        "--target-entropy",
        type=float,
        default=0.5,
        help="PPO target normalized entropy ratio in [0, 1] (default: 0.5)",
    )
    parser.add_argument(
        "--entropy-coeff",
        type=float,
        default=0.01,
        help="PPO entropy coefficient; set to 0 with --no-adaptive-entropy for ablation (default: 0.01)",
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
        default=0.5,
        help="Extra PPO bonus scale for normalized entropy over action types (default: 0.5)",
    )
    parser.add_argument(
        "--no-adaptive-entropy",
        action="store_true",
        help="Disable adaptive entropy tuning and keep entropy coefficient fixed",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="PPO discount factor (default: 0.99)",
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
    agent_config = AgentConfig(d_model=args.d_model, n_layers=args.n_layers)

    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir

    if args.phase == "supervised":
        from pylatro_agent.training.supervised import SupervisedConfig, train_supervised
        train_supervised(
            SupervisedConfig(
                num_games=args.games,
                batch_size=args.batch,
                max_epochs=args.epochs,
                num_workers=args.workers,
                min_ante=args.min_ante,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/supervised",
                log_dir=log_dir or "runs/supervised",
            ),
            agent_config=agent_config,
        )

    elif args.phase == "ppo":
        from pylatro_agent.training.ppo import PPOConfig, train_ppo
        train_ppo(
            PPOConfig(
                num_envs=args.envs,
                rollout_length=args.rollout_length,
                total_timesteps=args.steps,
                ppo_epochs=args.ppo_epochs,
                mini_batch_size=args.batch,
                lr=args.lr,
                device=device,
                save_dir=checkpoint_dir or "checkpoints/ppo",
                log_dir=log_dir or "runs/ppo",
                log_interval=args.log_interval,
                checkpoint_interval=args.checkpoint_interval,
                eval_interval=args.eval_interval,
                max_no_progress_steps=args.max_idle_steps,
                entropy_coeff=args.entropy_coeff,
                adaptive_entropy=not args.no_adaptive_entropy,
                target_entropy=args.target_entropy,
                entropy_ema_beta=args.entropy_ema_beta,
                alpha_lr=args.alpha_lr,
                action_type_entropy_scale=args.action_type_entropy_scale,
                gamma=args.gamma,
                async_envs=not args.sync_envs,
            ),
            agent_config=agent_config,
            pretrained_path=args.pretrained,
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
