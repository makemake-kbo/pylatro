#!/usr/bin/env python3
"""Training entrypoint for the Balatro agent.

Usage:
    uv run python train.py supervised [--games 1000] [--epochs 5] [--device mps]
    uv run python train.py ppo [--pretrained PATH] [--steps 200000] [--device mps]
    uv run python train.py self_play [--pretrained PATH] [--device mps]
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train the Balatro agent")
    parser.add_argument("phase", choices=["supervised", "ppo", "self_play"])
    parser.add_argument("--games", type=int, default=1000, help="Heuristic games for supervised (default: 1000)")
    parser.add_argument("--epochs", type=int, default=5, help="Supervised epochs (default: 5)")
    parser.add_argument("--steps", type=int, default=200_000, help="PPO total timesteps (default: 200000)")
    parser.add_argument("--envs", type=int, default=8, help="Parallel envs for PPO (default: 8)")
    parser.add_argument("--batch", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, mps, cuda (default: auto-detect)")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension (default: 256)")
    parser.add_argument("--n-layers", type=int, default=8, help="Transformer layers (default: 8)")
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

    if args.phase == "supervised":
        from pylatro_agent.training.supervised import SupervisedConfig, train_supervised
        train_supervised(
            SupervisedConfig(
                num_games=args.games,
                batch_size=args.batch,
                max_epochs=args.epochs,
                device=device,
            ),
            agent_config=agent_config,
        )

    elif args.phase == "ppo":
        from pylatro_agent.training.ppo import PPOConfig, train_ppo
        train_ppo(
            PPOConfig(
                num_envs=args.envs,
                rollout_length=128,
                total_timesteps=args.steps,
                mini_batch_size=min(32, args.batch),
                device=device,
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
            ),
            agent_config=agent_config,
            pretrained_path=args.pretrained,
        )


if __name__ == "__main__":
    main()
