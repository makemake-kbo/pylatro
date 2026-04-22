"""Phase 1: Supervised pretraining via imitation learning from heuristic agent."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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
from ..checkpoint import save_checkpoint
from ..distributions import MaskedCategorical
from ..survival import compute_ante_survival_targets
from ..vocab import build_vocab
from .fast_generate import generate_training_data

logger = logging.getLogger(__name__)


@dataclass
class SupervisedConfig:
    num_games: int = 10000
    batch_size: int = 256
    gamma: float = 0.995
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 10
    value_loss_coeff: float = 0.5
    num_workers: int = 0  # 0 = auto-detect (all available cores)
    min_ante: int = 5
    save_dir: str = "checkpoints/supervised"
    log_dir: str = "runs/supervised"
    device: str = "cpu"


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted reward-to-go targets for value pretraining."""
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running_return = rewards[idx] + gamma * running_return
        returns[idx] = running_return
    return returns


def train_supervised(
    config: SupervisedConfig,
    agent_config: AgentConfig | None = None,
    data: GameData | None = None,
    records: list[dict[str, Any]] | None = None,
) -> BalatroAgent:
    """Run supervised pretraining.

    If ``records`` is provided, heuristic data generation is skipped and the
    trainee is fit directly on the supplied transitions. This is used by
    pretraining workflows that build records from a separate model checkpoint.
    """
    if data is None:
        data = load_game_data()
    vocab = build_vocab(data)
    if agent_config is None:
        agent_config = AgentConfig()

    from torch.utils.tensorboard import SummaryWriter

    device = torch.device(config.device)
    model = BalatroAgent(agent_config, vocab).to(device)
    if config.device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    logger.info(f"Model parameters: {base_model.count_parameters():,}")
    if records is None:
        logger.info("Generating training data from heuristic agent...")
        t_gen_start = time.monotonic()
        records = generate_training_data(
            config.num_games,
            data=data,
            vocab=vocab,
            min_ante=config.min_ante,
            gamma=config.gamma,
            num_workers=config.num_workers,
        )
        t_gen_elapsed = time.monotonic() - t_gen_start
        logger.info("Generated %d training records in %.1fs", len(records), t_gen_elapsed)
    else:
        logger.info("Using %d pre-generated training records", len(records))

    if not records:
        logger.error("No training records provided — check min_ante or the upstream generator")
        return model

    wins = sum(1 for r in records if r["won"])
    max_antes = [r.get("max_ante", 1) for r in records]
    logger.info(
        "Data stats: win_rate=%.3f, mean_max_ante=%.1f, median_max_ante=%d",
        wins / len(records),
        sum(max_antes) / len(max_antes),
        sorted(max_antes)[len(max_antes) // 2],
    )

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
            logits, value_dict = model(
                batch["tokens"], batch["token_types"], batch["scalars"],
                batch["attention_mask"], batch["action_mask"],
            )
            dist = MaskedCategorical(logits, batch["action_mask"])

            # Action loss: cross-entropy
            action_loss = F.cross_entropy(logits, batch["actions"])

            # Value loss: BCE on win prediction + MSE on expected_score
            win_loss = F.binary_cross_entropy(
                value_dict["win_prob"], batch["won"].float()
            )
            # Train expected_score to predict approximate game return
            # This is CRITICAL — PPO uses expected_score as its value function
            score_loss = F.mse_loss(
                value_dict["expected_score"], batch["value_target"]
            )
            # Per-ante survival: BCE masked by observed antes.
            survival_mask = batch["ante_survival_mask"]
            survival_per_elem = F.binary_cross_entropy(
                value_dict["ante_survival"],
                batch["ante_survival_target"],
                reduction="none",
            )
            survival_loss = (survival_per_elem * survival_mask).sum() / survival_mask.sum().clamp(min=1.0)
            value_loss = win_loss + score_loss + survival_loss

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
            writer.add_scalar("train/expected_score_mean", value_dict["expected_score"].mean().item(), global_step)
            writer.add_scalar("train/survival_loss", survival_loss.item(), global_step)

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

        save_model = model.module if isinstance(model, nn.DataParallel) else model
        ckpt_path = save_path / f"supervised_epoch{epoch + 1}.pt"
        tmp_path = ckpt_path.with_suffix(".tmp")
        save_checkpoint(save_model, tmp_path)
        tmp_path.rename(ckpt_path)

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
        "value_target": torch.tensor(
            [
                r.get("return_target", 10.0 if r["won"] else (-10.0 + min(r.get("max_ante", 1), 8) * 1.0))
                for r in records
            ],
            dtype=torch.float32, device=device,
        ),
        "ante_survival_target": torch.tensor(
            np.array([_survival_target(r)[0] for r in records]),
            dtype=torch.float32, device=device,
        ),
        "ante_survival_mask": torch.tensor(
            np.array([_survival_target(r)[1] for r in records]),
            dtype=torch.float32, device=device,
        ),
    }


def _survival_target(record: dict) -> tuple[np.ndarray, np.ndarray]:
    if "ante_survival_target" in record and "ante_survival_mask" in record:
        return record["ante_survival_target"], record["ante_survival_mask"]
    return compute_ante_survival_targets(
        int(record.get("max_ante", 1)),
        bool(record.get("won", False)),
    )
