"""Phase 3: Self-play refinement with curriculum learning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from pylatro import GameData, load_game_data

from ..agent import AgentConfig, BalatroAgent
from ..checkpoint import save_checkpoint
from ..vocab import Vocab, build_vocab
from .ppo import PPOConfig, evaluate_model, train_ppo

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    initial_stake: int = 1
    max_stake: int = 8
    win_rate_threshold: float = 0.5
    eval_games: int = 50
    ppo_timesteps_per_stage: int = 500_000
    device: str = "cpu"
    save_dir: str = "checkpoints/self_play"


def train_self_play(
    config: SelfPlayConfig,
    agent_config: AgentConfig | None = None,
    pretrained_path: str | None = None,
    data: GameData | None = None,
) -> BalatroAgent:
    """Run self-play training with stake curriculum."""
    if data is None:
        data = load_game_data()
    vocab = build_vocab(data)
    if agent_config is None:
        agent_config = AgentConfig()

    device = torch.device(config.device)
    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    current_stake = config.initial_stake
    current_model_path = pretrained_path

    while current_stake <= config.max_stake:
        logger.info(f"=== Self-play stage: Stake {current_stake} ===")

        ppo_config = PPOConfig(
            total_timesteps=config.ppo_timesteps_per_stage,
            device=config.device,
            save_dir=str(save_path / f"stake_{current_stake}"),
        )

        # Train with PPO at current stake
        model = train_ppo(
            ppo_config,
            agent_config=agent_config,
            pretrained_path=current_model_path,
            data=data,
        )

        # Evaluate
        win_rate = evaluate_model(model, data, vocab, config.eval_games, device)
        logger.info(f"Stake {current_stake} win rate: {win_rate:.3f}")

        checkpoint_path = save_path / f"self_play_stake{current_stake}.pt"
        save_checkpoint(model, checkpoint_path)
        current_model_path = str(checkpoint_path)

        if win_rate >= config.win_rate_threshold:
            logger.info(f"Advancing to stake {current_stake + 1}")
            current_stake += 1
        else:
            logger.info(f"Win rate {win_rate:.3f} < {config.win_rate_threshold}, continuing at stake {current_stake}")

    logger.info("Self-play training complete")
    return model
