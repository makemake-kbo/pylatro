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

import torch

from pylatro import load_game_data
from pylatro_agent.action import ActionType, decode_action
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.env import BalatroEnv
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.vocab import build_vocab

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _single_obs_to_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor(obs["tokens"], dtype=torch.long, device=device).unsqueeze(0),
        "token_types": torch.tensor(obs["token_types"], dtype=torch.long, device=device).unsqueeze(0),
        "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=device).unsqueeze(0),
        "attention_mask": torch.tensor(obs["attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
    }


def play_model(
    checkpoint: str,
    num_games: int,
    seed: int,
    device: str,
    d_model: int,
    n_layers: int,
    d_ff: int,
    sample: bool,
    temperature: float,
    audit: bool,
    win_ante: int | None,
) -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    config = AgentConfig(d_model=d_model, n_layers=n_layers, d_ff=d_ff)

    dev = torch.device(device)
    model = BalatroAgent(config, vocab).to(dev)
    payload = load_checkpoint_payload(checkpoint, dev)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint: {checkpoint} ({model.count_parameters():,} params)")

    wins, total_antes, total_steps = 0, 0, 0
    teacher_matches, audited_steps, play_steps, out_of_candidate_plays = 0, 0, 0, 0
    for i in range(num_games):
        env = BalatroEnv(seed=seed + i, data=data, vocab=vocab, win_ante=win_ante)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            with torch.no_grad():
                batch = _single_obs_to_batch(obs, dev)
                dist, _ = model.action_distribution(
                    batch["tokens"], batch["token_types"], batch["scalars"],
                    batch["attention_mask"], batch["action_mask"],
                    temperature=temperature,
                )
                action = (dist.sample() if sample else dist.mode()).item()

            obs, _reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if audit:
                audited_steps += 1
                teacher_matches += int(bool(info.get("teacher_action_match", False)))
                decoded = decode_action(action)
                if decoded.action_type == ActionType.PLAY_SUBSET:
                    play_steps += 1
                    out_of_candidate_plays += int(bool(info.get("hand_play_not_in_candidates", False)))

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
    if audit and audited_steps:
        print(f"Teacher match: {teacher_matches/audited_steps:.1%}")
        if play_steps:
            print(f"Out-of-candidate plays: {out_of_candidate_plays/play_steps:.1%}")


def play_heuristic(num_games: int, seed: int, win_ante: int | None) -> None:
    data = load_game_data()
    vocab = build_vocab(data)
    agent = HeuristicAgent()

    wins, total_antes, total_steps = 0, 0, 0
    for i in range(num_games):
        env = BalatroEnv(seed=seed + i, data=data, vocab=vocab, win_ante=win_ante)
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            mask = obs["action_mask"]
            action = agent.select_action(
                env.state, env._sub_phase, mask,
                selected_cards=env._selected_cards,
                pending_action=env._pending_action,
            )
            obs, _reward, terminated, truncated, info = env.step(action)
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
    parser.add_argument("--d-ff", type=int, default=1024, help="Feed-forward dimension (default: 1024)")
    parser.add_argument("--sample", action="store_true", help="Sample from the policy instead of greedy mode")
    parser.add_argument("--temperature", type=float, default=0.7, help="Structured policy temperature (default: 0.7)")
    parser.add_argument("--audit", action="store_true", help="Print teacher-match and hand-choice diagnostics")
    parser.add_argument("--win-ante", type=int, default=None, help="Curriculum victory ante override")
    args = parser.parse_args()

    if args.heuristic:
        play_heuristic(args.games, args.seed, args.win_ante)
    elif args.checkpoint:
        device = args.device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        play_model(
            args.checkpoint,
            args.games,
            args.seed,
            device,
            args.d_model,
            args.n_layers,
            args.d_ff,
            args.sample,
            args.temperature,
            args.audit,
            args.win_ante,
        )
    else:
        parser.error("Provide --checkpoint or --heuristic")


if __name__ == "__main__":
    main()
