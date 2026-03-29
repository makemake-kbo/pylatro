#!/usr/bin/env python3
"""Play a game of Balatro using a trained agent and print results.

Usage:
    uv run python play.py --checkpoint checkpoints/supervised/supervised_epoch10.pt
    uv run python play.py --checkpoint checkpoints/ppo/ppo_update100.pt --games 20 --seed 42
    uv run python play.py --heuristic --games 50
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.env import BalatroEnv
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.vocab import build_vocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def play_model(checkpoint: str, num_games: int, seed: int, device: str, d_model: int, n_layers: int) -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    config = AgentConfig(d_model=d_model, n_layers=n_layers)

    dev = torch.device(device)
    model = BalatroAgent(config, vocab).to(dev)
    model.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True))
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint} ({model.count_parameters():,} params)")

    wins, total_antes, total_steps = 0, 0, 0
    for i in range(num_games):
        env = BalatroEnv(seed=seed + i, data=data, vocab=vocab)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            with torch.no_grad():
                batch = {k: torch.tensor(v, device=dev).unsqueeze(0) for k, v in obs.items()
                         if k in ("tokens", "token_types", "scalars", "attention_mask", "action_mask")}
                batch["tokens"] = batch["tokens"].long()
                batch["token_types"] = batch["token_types"].long()
                batch["attention_mask"] = batch["attention_mask"].long()
                batch["action_mask"] = batch["action_mask"].float()
                from pylatro_agent.distributions import MaskedCategorical
                logits, _ = model(
                    batch["tokens"], batch["token_types"], batch["scalars"],
                    batch["attention_mask"], batch["action_mask"],
                )
                dist = MaskedCategorical(logits, batch["action_mask"])
                action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        won = info.get("won", False)
        ante = info.get("ante", 1)
        wins += int(won)
        total_antes += ante
        total_steps += steps
        status = "WIN" if won else f"LOSS (ante {ante})"
        logger.info(f"Game {i+1}/{num_games}: {status} in {steps} steps")

    print(f"\n{'='*40}")
    print(f"Results: {wins}/{num_games} wins ({wins/num_games:.1%})")
    print(f"Avg ante reached: {total_antes/num_games:.1f}")
    print(f"Avg steps: {total_steps/num_games:.0f}")


def play_heuristic(num_games: int, seed: int) -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    agent = HeuristicAgent()

    wins, total_antes, total_steps = 0, 0, 0
    for i in range(num_games):
        env = BalatroEnv(seed=seed + i, data=data, vocab=vocab)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            mask = obs["action_mask"]
            action = agent.select_action(
                env.state, env._sub_phase, mask,
                selected_cards=env._selected_cards,
                pending_action=env._pending_action,
                pending_consumable_slot=env._pending_consumable_slot,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        won = info.get("won", False)
        ante = info.get("ante", 1)
        wins += int(won)
        total_antes += ante
        total_steps += steps
        status = "WIN" if won else f"LOSS (ante {ante})"
        logger.info(f"Game {i+1}/{num_games}: {status} in {steps} steps")

    print(f"\n{'='*40}")
    print(f"Heuristic: {wins}/{num_games} wins ({wins/num_games:.1%})")
    print(f"Avg ante reached: {total_antes/num_games:.1f}")
    print(f"Avg steps: {total_steps/num_games:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Play Balatro with a trained agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--heuristic", action="store_true", help="Use rule-based heuristic agent")
    parser.add_argument("--games", type=int, default=10, help="Number of games (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="Starting seed (default: 0)")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, mps, cuda")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension (default: 256)")
    parser.add_argument("--n-layers", type=int, default=8, help="Transformer layers (default: 8)")
    args = parser.parse_args()

    if args.heuristic:
        play_heuristic(args.games, args.seed)
    elif args.checkpoint:
        device = args.device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        play_model(args.checkpoint, args.games, args.seed, device, args.d_model, args.n_layers)
    else:
        parser.error("Provide --checkpoint or --heuristic")


if __name__ == "__main__":
    main()
