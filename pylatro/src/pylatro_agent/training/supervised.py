"""Phase 1: Supervised pretraining via imitation learning from heuristic agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pylatro import GameData, load_game_data

from ..agent import AgentConfig, BalatroAgent
from ..constants import SubPhase
from ..env import BalatroEnv
from ..heuristic import HeuristicAgent
from ..vocab import Vocab, build_vocab

logger = logging.getLogger(__name__)


@dataclass
class SupervisedConfig:
    num_games: int = 10000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 10
    value_loss_coeff: float = 0.5
    save_dir: str = "checkpoints/supervised"
    device: str = "cpu"


def generate_training_data(
    num_games: int,
    data: GameData | None = None,
    vocab: Vocab | None = None,
) -> list[dict[str, Any]]:
    """Run the heuristic agent for num_games and collect (obs, action, outcome) tuples."""
    if data is None:
        data = load_game_data()
    if vocab is None:
        vocab = build_vocab(data)

    agent = HeuristicAgent()
    records: list[dict[str, Any]] = []

    for game_idx in range(num_games):
        env = BalatroEnv(seed=game_idx, data=data, vocab=vocab)
        obs, info = env.reset()
        done = False
        game_records: list[dict[str, Any]] = []

        while not done:
            mask = obs["action_mask"]
            action = agent.select_action(
                env.state,
                env._sub_phase,
                mask,
                selected_cards=env._selected_cards,
                pending_action=env._pending_action,
                pending_consumable_slot=env._pending_consumable_slot,
            )
            game_records.append({"obs": obs, "action": action})
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        won = info.get("won", False)
        for rec in game_records:
            rec["won"] = won
            records.append(rec)

        if (game_idx + 1) % 100 == 0:
            logger.info(f"Generated {game_idx + 1}/{num_games} games, {len(records)} records")

    return records


def train_supervised(
    config: SupervisedConfig,
    agent_config: AgentConfig | None = None,
    data: GameData | None = None,
) -> BalatroAgent:
    """Run supervised pretraining."""
    if data is None:
        data = load_game_data()
    vocab = build_vocab(data)
    if agent_config is None:
        agent_config = AgentConfig()

    device = torch.device(config.device)
    model = BalatroAgent(agent_config, vocab).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info("Generating training data from heuristic agent...")
    records = generate_training_data(config.num_games, data=data, vocab=vocab)
    logger.info(f"Generated {len(records)} training records")

    # Shuffle and batch
    n = len(records)
    steps_per_epoch = (n + config.batch_size - 1) // config.batch_size
    total_steps = steps_per_epoch * config.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(config.max_epochs):
        indices = np.random.permutation(n)
        epoch_loss = 0.0
        epoch_action_correct = 0
        epoch_count = 0

        model.train()
        for batch_start in range(0, n, config.batch_size):
            batch_end = min(batch_start + config.batch_size, n)
            batch_idx = indices[batch_start:batch_end]
            batch_records = [records[i] for i in batch_idx]

            batch = _collate_batch(batch_records, device)
            dist, value_dict = model(
                batch["tokens"], batch["token_types"], batch["scalars"],
                batch["attention_mask"], batch["action_mask"],
            )

            # Action loss: cross-entropy
            action_loss = F.cross_entropy(dist.logits, batch["actions"])

            # Value loss: BCE on win prediction
            value_loss = F.binary_cross_entropy(
                value_dict["win_prob"], batch["won"].float()
            )

            loss = action_loss + config.value_loss_coeff * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Warmup
            if global_step < config.warmup_steps:
                warmup_factor = (global_step + 1) / config.warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = config.lr * warmup_factor
            else:
                scheduler.step()

            # Track accuracy
            predicted = dist.logits.argmax(dim=-1)
            correct = (predicted == batch["actions"]).sum().item()
            epoch_action_correct += correct
            epoch_count += len(batch_records)
            epoch_loss += loss.item() * len(batch_records)
            global_step += 1

        avg_loss = epoch_loss / max(epoch_count, 1)
        accuracy = epoch_action_correct / max(epoch_count, 1)
        logger.info(f"Epoch {epoch + 1}/{config.max_epochs}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        torch.save(model.state_dict(), save_path / f"supervised_epoch{epoch + 1}.pt")

    return model


def _collate_batch(records: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor(
            np.array([r["obs"]["tokens"] for r in records]), dtype=torch.long, device=device,
        ),
        "token_types": torch.tensor(
            np.array([r["obs"]["token_types"] for r in records]), dtype=torch.long, device=device,
        ),
        "scalars": torch.tensor(
            np.array([r["obs"]["scalars"] for r in records]), dtype=torch.float32, device=device,
        ),
        "attention_mask": torch.tensor(
            np.array([r["obs"]["attention_mask"] for r in records]), dtype=torch.long, device=device,
        ),
        "action_mask": torch.tensor(
            np.array([r["obs"]["action_mask"] for r in records]), dtype=torch.float32, device=device,
        ),
        "actions": torch.tensor(
            [r["action"] for r in records], dtype=torch.long, device=device,
        ),
        "won": torch.tensor(
            [float(r["won"]) for r in records], dtype=torch.float32, device=device,
        ),
    }
