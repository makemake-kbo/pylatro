#!/usr/bin/env python3
"""Play a game of Balatro using a trained agent and print results.

Usage:
    uv run python play.py --checkpoint checkpoints/supervised/supervised_epoch10.pt
    uv run python play.py --checkpoint checkpoints/ppo/ppo_update100.pt --games 20 --seed 42
    uv run python play.py --heuristic --games 50
    uv run --extra agent python play.py --live --heuristic
    uv run --extra agent python play.py --live --checkpoint checkpoints/ppo/ppo_update100.pt
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

from pylatro import load_game_data
from pylatro_agent.action import ActionType, decode_action
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.vocab import build_vocab

if TYPE_CHECKING:
    import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _single_obs_to_batch(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    import torch

    batch = {
        "tokens": torch.tensor(obs["tokens"], dtype=torch.long, device=device).unsqueeze(0),
        "token_types": torch.tensor(obs["token_types"], dtype=torch.long, device=device).unsqueeze(0),
        "scalars": torch.tensor(obs["scalars"], dtype=torch.float32, device=device).unsqueeze(0),
        "attention_mask": torch.tensor(obs["attention_mask"], dtype=torch.long, device=device).unsqueeze(0),
        "action_mask": torch.tensor(obs["action_mask"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    for key in (
        "history_events",
        "history_event_features",
        "history_cards",
        "history_card_mask",
        "history_jokers",
        "history_joker_mask",
        "history_event_mask",
        "history_round_mask",
        "history_omitted",
    ):
        dtype = torch.float32 if key in {"history_event_features", "history_omitted"} else torch.long
        batch[key] = torch.tensor(obs[key], dtype=dtype, device=device).unsqueeze(0)
    return batch


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
    import torch

    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.checkpoint import load_checkpoint_payload
    from pylatro_agent.env import BalatroEnv

    data = load_game_data()
    vocab = build_vocab(data)

    dev = torch.device(device)
    payload = load_checkpoint_payload(checkpoint, dev)
    saved_config = payload.get("agent_config")
    if isinstance(saved_config, dict):
        # PPO checkpoints persist their architecture (incl. value_bins for the
        # HL-Gauss head); trust it over CLI flags so any checkpoint replays
        # without the caller knowing its layer count or head shape.
        config = AgentConfig(**saved_config)
    else:
        config = AgentConfig(d_model=d_model, n_layers=n_layers, d_ff=d_ff)
    model = BalatroAgent(config, vocab).to(dev)
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
                    batch["tokens"],
                    batch["token_types"],
                    batch["scalars"],
                    batch["attention_mask"],
                    batch["action_mask"],
                    history_events=batch["history_events"],
                    history_event_features=batch["history_event_features"],
                    history_cards=batch["history_cards"],
                    history_card_mask=batch["history_card_mask"],
                    history_jokers=batch["history_jokers"],
                    history_joker_mask=batch["history_joker_mask"],
                    history_event_mask=batch["history_event_mask"],
                    history_round_mask=batch["history_round_mask"],
                    history_omitted=batch["history_omitted"],
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
        logger.info(f"Game {i + 1}/{num_games}: {status} in {steps} steps")

    print(f"\n{'=' * 40}")
    print(f"Results: {wins}/{num_games} wins ({wins / num_games:.1%})")
    print(f"Avg ante reached: {total_antes / num_games:.1f}")
    print(f"Avg steps: {total_steps / num_games:.0f}")
    if audit and audited_steps:
        print(f"Teacher match: {teacher_matches / audited_steps:.1%}")
        if play_steps:
            print(f"Out-of-candidate plays: {out_of_candidate_plays / play_steps:.1%}")


def play_heuristic(num_games: int, seed: int, win_ante: int | None) -> None:
    from pylatro_agent.env import BalatroEnv

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
                env.state,
                env._sub_phase,
                mask,
                round_score=env._controller.round_score,
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
        logger.info(f"Game {i + 1}/{num_games}: {status} in {steps} steps")

    print(f"\n{'=' * 40}")
    print(f"Heuristic: {wins}/{num_games} wins ({wins / num_games:.1%})")
    print(f"Avg ante reached: {total_antes / num_games:.1f}")
    print(f"Avg steps: {total_steps / num_games:.0f}")


def main():
    parser = argparse.ArgumentParser(description="Play Balatro with a trained agent")
    policy = parser.add_mutually_exclusive_group(required=True)
    policy.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    policy.add_argument("--heuristic", action="store_true", help="Use rule-based heuristic agent")
    parser.add_argument("--live", action="store_true", help="Control a running Balatro game through the bridge")
    parser.add_argument("--host", default="127.0.0.1", help="Live bind host (loopback only)")
    parser.add_argument("--port", type=int, default=43137, help="Live bridge port (default: 43137)")
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

    device = args.device or "cpu"
    if args.checkpoint and args.device is None:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

    if args.live:
        from pylatro_agent.live.policy import build_live_runner
        from pylatro_agent.live.server import serve_live

        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535")
        try:
            runner = build_live_runner(
                checkpoint=args.checkpoint,
                heuristic=args.heuristic,
                device=device,
                d_model=args.d_model,
                n_layers=args.n_layers,
                d_ff=args.d_ff,
                sample=args.sample,
                temperature=args.temperature,
            )
            serve_live(runner, host=args.host, port=args.port)
        except ValueError as exc:
            parser.error(str(exc))
    elif args.heuristic:
        play_heuristic(args.games, args.seed, args.win_ante)
    else:
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


if __name__ == "__main__":
    main()
