"""Phase 1: Supervised pretraining via imitation learning from heuristic agent."""

from __future__ import annotations

import logging
import time
from collections import Counter
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

from ..action import ActionType, decode_action
from ..agent import AgentConfig, BalatroAgent
from ..checkpoint import save_checkpoint
from ..constants import NUM_ACTIONS
from ..distributions import MaskedCategorical
from ..survival import compute_ante_survival_targets
from ..vocab import build_vocab
from .fast_generate import generate_training_data

logger = logging.getLogger(__name__)
_ACTION_ID_TO_TYPE = tuple(decode_action(action_id).action_type.value for action_id in range(NUM_ACTIONS))


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
    action_entropy_coeff: float = 0.001
    num_workers: int = 0  # 0 = auto-detect (all available cores)
    min_ante: int = 5
    save_dir: str = "checkpoints/supervised"
    log_dir: str = "runs/supervised"
    log_interval: int = 10
    device: str = "cpu"


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted reward-to-go targets for value pretraining."""
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running_return = rewards[idx] + gamma * running_return
        returns[idx] = running_return
    return returns


def _masked_action_loss(logits: torch.Tensor, action_mask: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return behavior-cloning NLL over the legal action support only."""
    dist = MaskedCategorical(logits, action_mask)
    return -dist.log_prob(actions).mean()


def _grammar_distribution(model: nn.Module, batch: dict[str, torch.Tensor]):
    """Return the structured policy distribution, unwrapping DataParallel if present."""
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    return base_model.action_distribution(
        batch["tokens"],
        batch["token_types"],
        batch["scalars"],
        batch["attention_mask"],
        batch["action_mask"],
    )


def _action_type_dataset_stats(records: list[dict[str, Any]]) -> dict[str, Counter]:
    """Summarize chosen and valid action families in generated BC records."""
    chosen: Counter = Counter()
    valid_states: Counter = Counter()
    for record in records:
        chosen[_ACTION_ID_TO_TYPE[int(record["action"])]] += 1
        valid_types = {
            _ACTION_ID_TO_TYPE[int(action_id)]
            for action_id in np.flatnonzero(record["obs"]["action_mask"])
        }
        valid_states.update(valid_types)
    return {"chosen": chosen, "valid_states": valid_states}


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
    action_type_stats = _action_type_dataset_stats(records)
    chosen_counts = action_type_stats["chosen"]
    valid_state_counts = action_type_stats["valid_states"]
    logger.info("Action family stats (chosen_fraction / valid_state_fraction):")
    for action_type in ActionType:
        name = action_type.value
        chosen_fraction = chosen_counts[name] / max(len(records), 1)
        valid_state_fraction = valid_state_counts[name] / max(len(records), 1)
        logger.info("  %-28s chosen=%.4f valid=%.4f", name, chosen_fraction, valid_state_fraction)
    hand_target_name = ActionType.USE_CONSUMABLE_HAND_SUBSET.value
    if valid_state_counts[hand_target_name] > 0 and chosen_counts[hand_target_name] == 0:
        logger.warning(
            "%s was valid in %d/%d records but never chosen; PPO will need to discover it from entropy alone.",
            hand_target_name,
            valid_state_counts[hand_target_name],
            len(records),
        )

    # Shuffle and batch
    n = len(records)
    steps_per_epoch = (n + config.batch_size - 1) // config.batch_size
    total_steps = steps_per_epoch * config.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    log_interval = max(int(config.log_interval), 1)

    save_path = Path(config.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(config.log_dir)
    writer.add_scalar("data/num_records", n, 0)
    writer.add_scalar("data/num_games", config.num_games, 0)
    for action_type in ActionType:
        name = action_type.value
        writer.add_scalar(f"data/action_chosen/{name}_fraction", chosen_counts[name] / n, 0)
        writer.add_scalar(f"data/action_valid/{name}_state_fraction", valid_state_counts[name] / n, 0)

    global_step = 0
    for epoch in range(config.max_epochs):
        indices = np.random.permutation(n)
        epoch_loss = torch.zeros((), device=device)
        epoch_action_loss = torch.zeros((), device=device)
        epoch_value_loss = torch.zeros((), device=device)
        epoch_action_correct = torch.zeros((), device=device)
        epoch_count = 0

        model.train()
        for batch_start in range(0, n, config.batch_size):
            batch_end = min(batch_start + config.batch_size, n)
            batch_idx = indices[batch_start:batch_end]
            batch_records = [records[i] for i in batch_idx]

            batch = _collate_batch(batch_records, device)
            dist, value_dict = _grammar_distribution(model, batch)

            # Action loss: behavior-cloning NLL over legal actions only.
            #
            # Raw cross-entropy treats every invalid action as a negative
            # class. In this action space that pushes rarely-valid heads
            # (notably hand-targeted consumables) to extreme negative logits
            # before PPO ever gets a chance to explore them.
            action_loss = -dist.log_prob(batch["actions"]).mean()

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

            entropy_bonus = dist.entropy().mean()
            loss = action_loss + config.value_loss_coeff * value_loss - config.action_entropy_coeff * entropy_bonus

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

            # Track epoch metrics on-device; synchronizing every batch is
            # expensive on MPS and visibly lowers GPU utilization.
            with torch.no_grad():
                predicted = dist.mode()
                correct = (predicted == batch["actions"]).sum()
                epoch_loss += loss.detach() * len(batch_records)
                epoch_action_loss += action_loss.detach() * len(batch_records)
                epoch_value_loss += value_loss.detach() * len(batch_records)
                epoch_action_correct += correct.detach()
            epoch_count += len(batch_records)

            if global_step % log_interval == 0:
                batch_acc = correct.float() / len(batch_records)
                # TensorBoard scalar writes call .item() under the hood; keep
                # them sparse so MPS can run without constant host syncs.
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/action_loss", action_loss, global_step)
                writer.add_scalar("train/value_loss", value_loss, global_step)
                writer.add_scalar("train/accuracy", batch_acc, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/entropy", entropy_bonus, global_step)
                writer.add_scalar(
                    "train/action_entropy_bonus",
                    config.action_entropy_coeff * entropy_bonus,
                    global_step,
                )
                writer.add_scalar("train/win_prob_mean", value_dict["win_prob"].mean(), global_step)
                writer.add_scalar("train/expected_score_mean", value_dict["expected_score"].mean(), global_step)
                writer.add_scalar("train/survival_loss", survival_loss, global_step)

            global_step += 1

        avg_loss = (epoch_loss / max(epoch_count, 1)).item()
        avg_action_loss = (epoch_action_loss / max(epoch_count, 1)).item()
        avg_value_loss = (epoch_value_loss / max(epoch_count, 1)).item()
        accuracy = (epoch_action_correct / max(epoch_count, 1)).item()

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
