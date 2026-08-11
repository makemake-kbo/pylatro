"""Phase 1: Supervised pretraining via imitation learning from heuristic agent."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from pylatro import GameData, load_game_data

from ..action import ActionType, decode_action
from ..action_grammar import ActionGrammarDistribution, ActionGrammarOutput
from ..agent import AgentConfig, BalatroAgent
from ..checkpoint import save_checkpoint
from ..constants import NUM_ACTIONS
from ..history import HistoryArrays
from ..reward import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    outcome_value,
    reward_checkpoint_metadata,
)
from ..survival import terminal_outcome_class, validate_critic_win_ante
from ..value_head import outcome_nll, return_huber_loss
from ..vocab import build_vocab
from .fast_generate import generate_training_data

logger = logging.getLogger(__name__)
_ACTION_ID_TO_TYPE = tuple(decode_action(action_id).action_type.value for action_id in range(NUM_ACTIONS))


@dataclass
class SupervisedConfig:
    num_games: int = 10000
    batch_size: int = 256
    gamma: float = 0.997  # Must match the PPO gamma of the next phase so value targets are calibrated
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 10
    value_loss_coeff: float = 0.5
    outcome_loss_coeff: float = 0.10
    action_entropy_coeff: float = 0.001
    num_workers: int = 0  # 0 = auto-detect (all available cores)
    min_ante: int = 1  # Phase 5: was 5; hard outcome filtering creates survivorship bias
    # Phase 5: outcome weighting replaces hard filtering. Every game is kept;
    # per-record BC loss weight w = exp(beta * normalized_outcome), clamped to
    # [bc_weight_min, bc_weight_max]. This tilts imitation toward successful
    # trajectories (AWR-style) while preserving full state coverage, including
    # the recovery states PPO visits. Applied to the BC NLL only, never to
    # value/win/survival losses (else the critic inherits the optimism bias).
    outcome_weight_beta: float = 1.5
    bc_weight_min: float = 0.25
    bc_weight_max: float = 4.0
    # Fraction of below-threshold games to keep in the training set,
    # used to break the survivorship bias of filtering exclusively for
    # successful runs (which left the model unable to recognize the
    # common bad early/mid states it has to recover from). 0.0 (default)
    # preserves the original strict-filter behavior.
    keep_below_threshold_ratio: float = 0.0
    chunk_size: int = 10_000
    save_dir: str = "checkpoints/supervised"
    log_dir: str = "runs/supervised"
    log_interval: int = 10
    device: str = "cpu"
    # Mixture weight on the autoregressive hand/discard head during BC. BC is
    # the only phase with dense, cheap labels for the AR head; at 0.5 the AR
    # component receives strong gradient on every hand-play label, including
    # the ~6% of teacher plays the candidate head can never match (those rows
    # now train the AR head exclusively). At PPO time this drops to 0.1.
    hand_ar_mixture_eps: float = 0.5
    win_ante: int = 8
    reward_config: RewardConfig | None = None


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """Compute discounted reward-to-go targets for value pretraining."""
    returns = [0.0] * len(rewards)
    running_return = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running_return = rewards[idx] + gamma * running_return
        returns[idx] = running_return
    return returns


def _grammar_distribution(model: nn.Module, batch: dict[str, torch.Tensor]):
    """Return the policy while keeping DataParallel on the actual forward path."""
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    history_kwargs = {
        "history_events": batch.get("history_events"),
        "history_event_features": batch.get("history_event_features"),
        "history_cards": batch.get("history_cards"),
        "history_card_mask": batch.get("history_card_mask"),
        "history_jokers": batch.get("history_jokers"),
        "history_joker_mask": batch.get("history_joker_mask"),
        "history_event_mask": batch.get("history_event_mask"),
        "history_round_mask": batch.get("history_round_mask"),
        "history_omitted": batch.get("history_omitted"),
    }
    if isinstance(model, nn.DataParallel) and isinstance(base_model, BalatroAgent):
        raw_output, value_dict = model(
            batch["tokens"],
            batch["token_types"],
            batch["scalars"],
            batch["attention_mask"],
            batch["action_mask"],
            return_raw_outputs=True,
            **history_kwargs,
        )
        return (
            ActionGrammarDistribution(
                ActionGrammarOutput(**raw_output),
                batch["action_mask"],
                tokens=batch["tokens"],
                hand_ar_mixture_eps=base_model.config.hand_ar_mixture_eps,
            ),
            value_dict,
        )
    return base_model.action_distribution(
        batch["tokens"],
        batch["token_types"],
        batch["scalars"],
        batch["attention_mask"],
        batch["action_mask"],
        **history_kwargs,
    )


def _action_type_dataset_stats(records: list[dict[str, Any]]) -> dict[str, Counter]:
    """Summarize chosen and valid action families in generated BC records."""
    chosen: Counter = Counter()
    valid_states: Counter = Counter()
    for record in records:
        chosen[_ACTION_ID_TO_TYPE[int(record["action"])]] += 1
        valid_types = {_ACTION_ID_TO_TYPE[int(action_id)] for action_id in np.flatnonzero(record["obs"]["action_mask"])}
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
    validate_critic_win_ante(config.win_ante)
    if data is None:
        data = load_game_data()
    vocab = build_vocab(data)
    if agent_config is None:
        agent_config = AgentConfig()
    # Apply the BC-phase mixture weight so the AR head trains alongside the
    # candidate head on every label (Phase 1.2).
    agent_config.hand_ar_mixture_eps = config.hand_ar_mixture_eps

    from torch.utils.tensorboard import SummaryWriter

    device = torch.device(config.device)
    model = BalatroAgent(agent_config, vocab).to(device)
    if config.device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    active_reward_config = replace(
        config.reward_config or DEFAULT_REWARD_CONFIG,
        gamma=config.gamma,
        potential_win_ante=config.win_ante,
    )
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
            keep_below_threshold_ratio=config.keep_below_threshold_ratio,
            chunk_size=config.chunk_size,
            win_ante=config.win_ante,
            reward_config=active_reward_config,
        )
        t_gen_elapsed = time.monotonic() - t_gen_start
        logger.info("Generated %d training records in %.1fs", len(records), t_gen_elapsed)
    else:
        logger.info("Using %d pre-generated training records", len(records))

    if not records:
        logger.error("No training records provided, check min_ante or the upstream generator")
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

            batch = _collate_batch(batch_records, device, config=config)
            dist, value_dict = _grammar_distribution(model, batch)

            # Action loss: behavior-cloning NLL over legal actions only.
            #
            # Raw cross-entropy treats every invalid action as a negative
            # class. In this action space that pushes rarely-valid heads
            # (notably hand-targeted consumables) to extreme negative logits
            # before PPO ever gets a chance to explore them.
            # Phase 5: outcome-weighted BC NLL. The weight tilts imitation toward
            # successful trajectories while preserving full state coverage. Applied
            # to the action loss ONLY; critic losses use unweighted
            # targets so the critic stays unbiased.
            bc_weights = batch["bc_weight"]
            logp = dist.log_prob(batch["actions"])
            with torch.no_grad():
                # Fraction of BC labels at the -1e8 grammar-unreachable floor.
                # These rows contribute a constant to the loss with zero useful
                # gradient (the -1e8 floor), and historically inflated
                # train/action_loss to ~6e6 while crushing healthy-row gradients
                # under clip_grad_norm. PPO's distill path already masks these;
                # supervised BC did not, which is why the poisoning was invisible
                # except through the absurd loss magnitude. Phase 1's mixture
                # makes this 0.0 by construction.
                unreachable_label_fraction = (logp < -1e7).float().mean()
            if config.hand_ar_mixture_eps > 0.0 and unreachable_label_fraction.item() > 0.0:
                # Phase 1.2 safety gate: with the AR mixture on, every legal label
                # must have finite log_prob. A nonzero fraction means the mixture
                # has a support bug (or a label is illegal under its own mask).
                raise RuntimeError(
                    f"{unreachable_label_fraction.item():.4f} of BC labels are at the "
                    f"-1e8 unreachable floor despite hand_ar_mixture_eps="
                    f"{config.hand_ar_mixture_eps}, the mixture has a support bug."
                )
            action_loss = -(logp * bc_weights).sum() / bc_weights.sum().clamp(min=1e-6)

            terminal_nll = outcome_nll(
                value_dict["outcome_probabilities"],
                batch["terminal_outcome_target"],
                batch["terminal_outcome_mask"],
            )
            return_loss = return_huber_loss(value_dict, batch["value_target"])
            critic_loss = (
                config.value_loss_coeff * return_loss
                + config.outcome_loss_coeff * terminal_nll
            )
            with torch.no_grad():
                terminal_mask = batch["terminal_outcome_mask"].float()
                terminal_denominator = terminal_mask.sum().clamp(min=1.0)
                one_hot_outcome = torch.nn.functional.one_hot(
                    batch["terminal_outcome_target"],
                    num_classes=value_dict["outcome_probabilities"].shape[1],
                ).to(dtype=value_dict["outcome_probabilities"].dtype)
                outcome_brier = (
                    (value_dict["outcome_probabilities"] - one_hot_outcome)
                    .square()
                    .sum(dim=-1)
                    .mul(terminal_mask)
                    .sum()
                    / terminal_denominator
                )
                win_target = (
                    batch["terminal_outcome_target"]
                    == value_dict["outcome_probabilities"].shape[1] - 1
                ).float()
                derived_win_brier = (
                    (value_dict["win_prob"] - win_target)
                    .square()
                    .mul(terminal_mask)
                    .sum()
                    / terminal_denominator
                )

            entropy_bonus = dist.entropy().mean()
            loss = action_loss + critic_loss - config.action_entropy_coeff * entropy_bonus

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
                epoch_value_loss += critic_loss.detach() * len(batch_records)
                epoch_action_correct += correct.detach()
            epoch_count += len(batch_records)

            if global_step % log_interval == 0:
                batch_acc = correct.float() / len(batch_records)
                # TensorBoard scalar writes call .item() under the hood; keep
                # them sparse so MPS can run without constant host syncs.
                writer.add_scalar("train/loss", loss, global_step)
                writer.add_scalar("train/action_loss", action_loss, global_step)
                writer.add_scalar("train/value_loss", critic_loss, global_step)
                writer.add_scalar("train/accuracy", batch_acc, global_step)
                writer.add_scalar("train/grad_norm", grad_norm, global_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("train/entropy", entropy_bonus, global_step)
                writer.add_scalar(
                    "train/action_entropy_bonus",
                    config.action_entropy_coeff * entropy_bonus,
                    global_step,
                )
                writer.add_scalar("critic/outcome_nll", terminal_nll, global_step)
                writer.add_scalar("critic/outcome_brier", outcome_brier, global_step)
                writer.add_scalar("critic/derived_win_brier", derived_win_brier, global_step)
                writer.add_scalar("critic/return_huber", return_loss, global_step)
                writer.add_scalar("critic/terminal_value_mean", value_dict["terminal_value"].mean(), global_step)
                writer.add_scalar("critic/return_residual_mean", value_dict["return_residual"].mean(), global_step)
                writer.add_scalar("critic/expected_return_mean", value_dict["expected_return"].mean(), global_step)
                writer.add_scalar("train/unreachable_label_fraction", unreachable_label_fraction, global_step)

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
        save_checkpoint(
            save_model,
            tmp_path,
            extra={
                "agent_config": asdict(save_model.config),
                **reward_checkpoint_metadata(active_reward_config),
            },
        )
        tmp_path.rename(ckpt_path)

    writer.close()
    return model


def _collate_batch(records: list[dict], device: torch.device, config: SupervisedConfig) -> dict[str, torch.Tensor]:
    bc_weights = np.array([_outcome_weight(record, config) for record in records], dtype=np.float32)
    empty_history = HistoryArrays.empty().as_dict()

    def history_values(key: str) -> np.ndarray:
        return np.array([record["obs"].get(key, empty_history[key]) for record in records])

    return {
        "tokens": torch.tensor(
            np.array([r["obs"]["tokens"] for r in records]),
            dtype=torch.long,
            device=device,
        ),
        "token_types": torch.tensor(
            np.array([r["obs"]["token_types"] for r in records]),
            dtype=torch.long,
            device=device,
        ),
        "scalars": torch.tensor(
            np.array([r["obs"]["scalars"] for r in records]),
            dtype=torch.float32,
            device=device,
        ),
        "attention_mask": torch.tensor(
            np.array([r["obs"]["attention_mask"] for r in records]),
            dtype=torch.long,
            device=device,
        ),
        "action_mask": torch.tensor(
            np.array([r["obs"]["action_mask"] for r in records]),
            dtype=torch.float32,
            device=device,
        ),
        "history_events": torch.tensor(history_values("history_events"), dtype=torch.long, device=device),
        "history_event_features": torch.tensor(
            history_values("history_event_features"), dtype=torch.float32, device=device
        ),
        "history_cards": torch.tensor(history_values("history_cards"), dtype=torch.long, device=device),
        "history_card_mask": torch.tensor(history_values("history_card_mask"), dtype=torch.long, device=device),
        "history_jokers": torch.tensor(history_values("history_jokers"), dtype=torch.long, device=device),
        "history_joker_mask": torch.tensor(history_values("history_joker_mask"), dtype=torch.long, device=device),
        "history_event_mask": torch.tensor(history_values("history_event_mask"), dtype=torch.long, device=device),
        "history_round_mask": torch.tensor(history_values("history_round_mask"), dtype=torch.long, device=device),
        "history_omitted": torch.tensor(history_values("history_omitted"), dtype=torch.float32, device=device),
        "actions": torch.tensor(
            [r["action"] for r in records],
            dtype=torch.long,
            device=device,
        ),
        "value_target": torch.tensor(
            [r["return_target"] for r in records],
            dtype=torch.float32,
            device=device,
        ),
        "terminal_outcome_target": torch.tensor(
            [
                int(
                    r.get(
                        "terminal_outcome_target",
                        terminal_outcome_class(
                            won=bool(r.get("won", False)),
                            final_ante=int(r.get("max_ante", 1)),
                        ),
                    )
                )
                for r in records
            ],
            dtype=torch.long,
            device=device,
        ),
        "terminal_outcome_mask": torch.tensor(
            [float(r.get("terminal_outcome_mask", 1.0)) for r in records],
            dtype=torch.float32,
            device=device,
        ),
        "bc_weight": torch.tensor(bc_weights, dtype=torch.float32, device=device),
    }


def _outcome_weight(record: dict, config: SupervisedConfig) -> float:
    """AWR-style outcome weight: w = exp(beta * normalized_outcome), clamped.

    normalized_outcome maps outcome_value to [0, 1]. Applied to the
    BC NLL only, never to return or terminal-outcome targets, so the critic stays
    unbiased while imitation tilts toward successful trajectories.
    """
    import math

    won = bool(record.get("won", False))
    max_ante = int(record.get("max_ante", 1))
    outcome = outcome_value(won=won, ante=max_ante)
    min_outcome = outcome_value(won=False, ante=1, stalled=True)
    max_outcome = outcome_value(won=True, ante=1)
    span = max(max_outcome - min_outcome, 1e-6)
    normalized = (outcome - min_outcome) / span
    weight = math.exp(config.outcome_weight_beta * normalized)
    return max(config.bc_weight_min, min(config.bc_weight_max, weight))
