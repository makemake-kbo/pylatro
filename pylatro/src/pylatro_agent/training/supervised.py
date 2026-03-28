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
    log_dir: str = "runs/supervised"
    device: str = "cpu"


def generate_training_data(
    num_games: int,
    data: GameData | None = None,
    vocab: Vocab | None = None,
    min_blinds_beaten: int = 2,
) -> list[dict[str, Any]]:
    """Run the heuristic agent for num_games and collect (obs, action, outcome) tuples.

    Only keeps games where at least min_blinds_beaten blinds were beaten,
    filtering out low-quality games that would teach bad strategy.
    """
    if data is None:
        data = load_game_data()
    if vocab is None:
        vocab = build_vocab(data)

    agent = HeuristicAgent()
    records: list[dict[str, Any]] = []
    games_kept = 0
    games_total = 0

    seed = 0
    while games_kept < num_games:
        env = BalatroEnv(seed=seed, data=data, vocab=vocab)
        obs, info = env.reset()
        done = False
        game_records: list[dict[str, Any]] = []
        blinds_beaten = 0

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
            if env._blind_just_beaten:
                blinds_beaten += 1

        seed += 1
        games_total += 1
        won = info.get("won", False)

        # Filter: only keep games that beat enough blinds
        if blinds_beaten >= min_blinds_beaten or won:
            games_kept += 1
            for rec in game_records:
                rec["won"] = won
                rec["blinds_beaten"] = blinds_beaten
                records.append(rec)

        if games_total % 200 == 0:
            logger.info(
                f"Played {games_total} games, kept {games_kept}/{num_games}, "
                f"{len(records)} records (filter rate: {games_kept/games_total:.0%})"
            )

    logger.info(
        f"Done: played {games_total} games to get {games_kept} quality games "
        f"({len(records)} records, keep rate: {games_kept/games_total:.0%})"
    )
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

    from torch.utils.tensorboard import SummaryWriter

    device = torch.device(config.device)
    model = BalatroAgent(agent_config, vocab).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    logger.info(f"Model parameters: {model.count_parameters():,}")
    logger.info("Generating training data from heuristic agent...")
    records = generate_training_data(config.num_games, data=data, vocab=vocab)
    logger.info(f"Generated {len(records)} training records")

    wins = sum(1 for r in records if r["won"])
    unique_games = config.num_games
    win_games = len({id(r) for r in records if r["won"]})  # approximate
    logger.info(f"Heuristic win rate (approx): {wins / max(len(records), 1):.3f} of records from winning games")

    # Shuffle and batch
    n = len(records)
    steps_per_epoch = (n + config.batch_size - 1) // config.batch_size
    total_steps = steps_per_epoch * config.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(config.log_dir)
    writer.add_scalar("data/num_records", n, 0)
    writer.add_scalar("data/num_games", config.num_games, 0)

    global_step = 0
    for epoch in range(config.max_epochs):
        indices = np.random.permutation(n)
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_value_loss = 0.0
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
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            batch_acc = correct / len(batch_records)

            epoch_action_correct += correct
            epoch_count += len(batch_records)
            epoch_loss += loss.item() * len(batch_records)
            epoch_action_loss += action_loss.item() * len(batch_records)
            epoch_value_loss += value_loss.item() * len(batch_records)

            # Per-step TensorBoard logging
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/action_loss", action_loss.item(), global_step)
            writer.add_scalar("train/value_loss", value_loss.item(), global_step)
            writer.add_scalar("train/accuracy", batch_acc, global_step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/entropy", dist.entropy().mean().item(), global_step)
            writer.add_scalar("train/win_prob_mean", value_dict["win_prob"].mean().item(), global_step)

            global_step += 1

        avg_loss = epoch_loss / max(epoch_count, 1)
        avg_action_loss = epoch_action_loss / max(epoch_count, 1)
        avg_value_loss = epoch_value_loss / max(epoch_count, 1)
        accuracy = epoch_action_correct / max(epoch_count, 1)

        writer.add_scalar("epoch/loss", avg_loss, epoch + 1)
        writer.add_scalar("epoch/action_loss", avg_action_loss, epoch + 1)
        writer.add_scalar("epoch/value_loss", avg_value_loss, epoch + 1)
        writer.add_scalar("epoch/accuracy", accuracy, epoch + 1)

        logger.info(
            f"Epoch {epoch + 1}/{config.max_epochs}: "
            f"loss={avg_loss:.4f}, action_loss={avg_action_loss:.4f}, "
            f"value_loss={avg_value_loss:.4f}, accuracy={accuracy:.4f}"
        )

        torch.save(model.state_dict(), save_path / f"supervised_epoch{epoch + 1}.pt")

    writer.close()
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
