#!/usr/bin/env python3
"""Compare scalar and HL-Gauss critics on one cached policy dataset.

The probe keeps the actor and transformer trunk fixed. It collects complete
episodes from a PPO checkpoint, caches one pooled transformer feature per
state, and fits both value-head parameterizations on identical PPO lambda
targets. Held-out complete episodes are also used to recompute the SIL gate
from discounted return-to-go.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.reward import PPO_V2_REWARD_CONFIG
from pylatro_agent.training.ppo import (
    _extract_step_info_value,
    _make_vectorized_envs,
    _ObsBuffer,
)
from pylatro_agent.training.sil import sil_percentile_gate
from pylatro_agent.value_head import ValueHead, hl_gauss_projection
from pylatro_agent.vocab import build_vocab

LOGGER = logging.getLogger("offline_critic_probe")

if TYPE_CHECKING:
    from collections.abc import Sequence


def _device_from_arg(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    result = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def _lambda_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_ids: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> np.ndarray:
    targets = np.empty_like(values, dtype=np.float32)
    for episode_id in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode_id)
        last_gae = 0.0
        for position in range(len(indices) - 1, -1, -1):
            index = int(indices[position])
            next_value = 0.0 if position == len(indices) - 1 else float(values[indices[position + 1]])
            delta = float(rewards[index]) + gamma * next_value - float(values[index])
            last_gae = delta + gamma * gae_lambda * last_gae
            targets[index] = last_gae + values[index]
    return targets


def _split_episodes(
    episode_ids: np.ndarray,
    episode_wins: dict[int, bool],
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    validation_episodes: list[int] = []
    for won in (False, True):
        group = np.asarray(
            [episode_id for episode_id, outcome in episode_wins.items() if outcome is won],
            dtype=np.int64,
        )
        rng.shuffle(group)
        count = (
            0
            if len(group) <= 1
            else min(len(group) - 1, max(1, round(len(group) * validation_fraction)))
        )
        validation_episodes.extend(group[:count].tolist())
    validation_mask = np.isin(episode_ids, np.asarray(validation_episodes, dtype=np.int64))
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def _collect_episodes(
    model: BalatroAgent,
    *,
    num_games: int,
    num_envs: int,
    device: torch.device,
    gamma: float,
    seed: int,
) -> dict[str, Any]:
    data = load_game_data()
    vocab = build_vocab(data)
    reward_config = PPO_V2_REWARD_CONFIG(
        gamma=gamma,
        win_ante=5,
        planet_match_shaping=True,
        planet_unmatched_use_penalty_coeff=0.4,
        planet_unmatched_claim_penalty_coeff=0.4,
        build_curve_shaping=True,
    )
    vec_env = _make_vectorized_envs(
        num_envs,
        data,
        vocab,
        max_no_progress_steps=256,
        use_async=num_envs > 1,
        win_ante=5,
        reward_config=reward_config,
        env_seed_base=seed * 10_000_019 + 700_000_001,
        enable_teacher=False,
    )
    obs_dict, _ = vec_env.reset()
    obs_buf = _ObsBuffer(num_envs, device)
    obs_buf.update(obs_dict)

    fields = ("tokens", "token_types", "scalars", "attention_mask")
    episode_obs: list[dict[str, list[np.ndarray]]] = [
        {field: [] for field in fields} for _ in range(num_envs)
    ]
    episode_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    flattened: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    flat_rewards: list[np.ndarray] = []
    flat_mc_returns: list[np.ndarray] = []
    flat_episode_ids: list[np.ndarray] = []
    flat_wins: list[np.ndarray] = []
    episode_wins: dict[int, bool] = {}
    completed = 0
    attempted = 0
    stalled_dropped = 0
    start = time.monotonic()

    try:
        while completed < num_games:
            pre_obs = obs_buf.as_numpy_dict()
            with torch.no_grad():
                distribution, _ = model.action_distribution(
                    obs_buf.tokens,
                    obs_buf.token_types,
                    obs_buf.scalars,
                    obs_buf.attention_mask,
                    obs_buf.action_mask,
                    temperature=1.0,
                )
                actions = distribution.sample()
            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions.cpu().numpy())
            dones = terminated | truncated

            for env_index in range(num_envs):
                for field in fields:
                    episode_obs[env_index][field].append(pre_obs[field][env_index].copy())
                episode_rewards[env_index].append(float(rewards[env_index]))
                if not dones[env_index]:
                    continue
                attempted += 1
                won = bool(
                    _extract_step_info_value(infos, "won", env_index, done=True, default=False)
                )
                stalled = bool(
                    _extract_step_info_value(infos, "stalled", env_index, done=True, default=False)
                )
                if stalled:
                    stalled_dropped += 1
                    episode_obs[env_index] = {field: [] for field in fields}
                    episode_rewards[env_index] = []
                    continue
                if completed < num_games:
                    episode_id = completed
                    rewards_array = np.asarray(episode_rewards[env_index], dtype=np.float32)
                    for field in fields:
                        flattened[field].append(np.stack(episode_obs[env_index][field]))
                    flat_rewards.append(rewards_array)
                    flat_mc_returns.append(_discounted_returns(rewards_array, gamma))
                    flat_episode_ids.append(
                        np.full(len(rewards_array), episode_id, dtype=np.int64)
                    )
                    flat_wins.append(np.full(len(rewards_array), won, dtype=np.bool_))
                    episode_wins[episode_id] = won
                    completed += 1
                    if completed % max(10, num_games // 4) == 0:
                        LOGGER.info("Collected %d/%d complete episodes", completed, num_games)
                episode_obs[env_index] = {field: [] for field in fields}
                episode_rewards[env_index] = []
            obs_buf.update(next_obs)
    finally:
        vec_env.close()

    elapsed = time.monotonic() - start
    result = {field: np.concatenate(flattened[field], axis=0) for field in fields}
    result.update(
        rewards=np.concatenate(flat_rewards),
        mc_returns=np.concatenate(flat_mc_returns),
        episode_ids=np.concatenate(flat_episode_ids),
        wins=np.concatenate(flat_wins),
        episode_wins=episode_wins,
        attempted_episodes=attempted,
        retained_episodes=completed,
        stalled_episodes_dropped=stalled_dropped,
        collection_seconds=elapsed,
        attempted_games=attempted,
    )
    return result


def _cache_features(
    model: BalatroAgent,
    dataset: dict[str, Any],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    feature_batches: list[torch.Tensor] = []
    prediction_batches: list[np.ndarray] = []
    total = len(dataset["rewards"])
    model.eval()
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(total, start + batch_size)
            tokens = torch.as_tensor(dataset["tokens"][start:stop], dtype=torch.long, device=device)
            token_types = torch.as_tensor(
                dataset["token_types"][start:stop], dtype=torch.long, device=device
            )
            scalars = torch.as_tensor(
                dataset["scalars"][start:stop], dtype=torch.float32, device=device
            )
            attention = torch.as_tensor(
                dataset["attention_mask"][start:stop], dtype=torch.long, device=device
            )
            backbone = model.embedding(tokens, token_types, scalars)
            backbone = model.backbone(backbone, padding_mask=attention == 0)
            mask = attention.unsqueeze(-1).float()
            pooled = (backbone * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            feature_batches.append(pooled.cpu())
            prediction = model.value_head(backbone, attention)["expected_score"]
            prediction_batches.append(prediction.float().cpu().numpy())
    return torch.cat(feature_batches, dim=0), np.concatenate(prediction_batches)


def _predict_head(
    head: ValueHead,
    features: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = features[batch_indices].to(device)
            mask = torch.ones((len(batch_indices), 1), dtype=torch.long, device=device)
            value = head(batch.unsqueeze(1), mask)["expected_score"]
            predictions.append(value.float().cpu().numpy())
    return np.concatenate(predictions)


def _fit_head(
    features: torch.Tensor,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    value_bins: int,
    initial_pool_state: dict[str, torch.Tensor],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[ValueHead, dict[str, float]]:
    torch.manual_seed(seed)
    head = ValueHead(d_model=features.shape[1], value_bins=value_bins).to(device)
    head.pool_proj.load_state_dict(copy.deepcopy(initial_pool_state))
    parameters = list(head.pool_proj.parameters()) + list(head.expected_score.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_mse = math.inf
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        head.train()
        for start in range(0, len(shuffled), batch_size):
            batch_indices = shuffled[start:start + batch_size]
            batch_features = features[batch_indices].to(device)
            batch_targets = torch.as_tensor(
                targets[batch_indices], dtype=torch.float32, device=device
            )
            mask = torch.ones((len(batch_indices), 1), dtype=torch.long, device=device)
            output = head(batch_features.unsqueeze(1), mask)
            if value_bins > 0:
                target_probabilities = hl_gauss_projection(
                    batch_targets, head.bin_edges, head.hl_gauss_sigma
                )
                loss = -(
                    target_probabilities
                    * F.log_softmax(output["expected_score_logits"], dim=-1)
                ).sum(dim=-1).mean()
            else:
                loss = F.mse_loss(output["expected_score"], batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        validation_predictions = _predict_head(
            head,
            features,
            validation_indices,
            batch_size=batch_size,
            device=device,
        )
        validation_mse = float(
            np.mean((validation_predictions - targets[validation_indices]) ** 2)
        )
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    assert best_state is not None
    head.load_state_dict(best_state)
    return head, {"best_validation_mse": best_mse, "best_epoch": best_epoch}


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = np.empty(len(left), dtype=np.float64)
    right_ranks = np.empty(len(right), dtype=np.float64)
    left_ranks[np.argsort(left, kind="stable")] = np.arange(len(left), dtype=np.float64)
    right_ranks[np.argsort(right, kind="stable")] = np.arange(len(right), dtype=np.float64)
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _prediction_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residual = targets - predictions
    target_variance = float(np.var(targets))
    explained_variance = (
        float("nan")
        if target_variance <= 1e-12
        else 1.0 - float(np.var(residual)) / target_variance
    )
    tail_cutoff = float(np.quantile(np.abs(targets), 0.9))
    tail = np.abs(targets) >= tail_cutoff
    return {
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "explained_variance": explained_variance,
        "rank_correlation": _rank_correlation(targets, predictions),
        "tail_mse": float(np.mean(residual[tail] ** 2)),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
    }


def _gate_subset(
    advantages: np.ndarray,
    *,
    open_percentile: float,
    saturation_percentile: float,
    advantage_floor: float,
) -> dict[str, float]:
    """Summarize the shared SIL gate over one advantage slice."""
    gate, info = sil_percentile_gate(
        advantages,
        open_percentile=open_percentile,
        saturation_percentile=saturation_percentile,
        advantage_floor=advantage_floor,
    )
    if advantages.size == 0:
        return {
            "states": 0,
            "advantage_mean": float("nan"),
            "advantage_p50": float("nan"),
            "advantage_p80": float("nan"),
            "advantage_p95": float("nan"),
            "positive_advantage_fraction": float("nan"),
            "open_threshold": float(info["open_threshold"]),
            "saturation_threshold": float(info["saturation_threshold"]),
            "gate_mean": float("nan"),
            "gate_positive_fraction": float("nan"),
            "gate_saturation_fraction": float("nan"),
            "noise_floor_rejected_fraction": float("nan"),
        }
    return {
        "states": int(advantages.size),
        "advantage_mean": float(np.mean(advantages)),
        "advantage_p50": float(np.percentile(advantages, 50.0)),
        "advantage_p80": float(np.percentile(advantages, 80.0)),
        "advantage_p95": float(np.percentile(advantages, 95.0)),
        "positive_advantage_fraction": float(np.mean(advantages > 0.0)),
        "open_threshold": float(info["open_threshold"]),
        "saturation_threshold": float(info["saturation_threshold"]),
        "gate_mean": float(np.mean(gate)),
        "gate_positive_fraction": float(np.mean(gate > 0.0)),
        "gate_saturation_fraction": float(np.mean(gate >= 1.0)),
        "noise_floor_rejected_fraction": float(np.mean(gate == 0.0)),
    }


def _sil_gate_metrics(
    mc_returns: np.ndarray,
    predictions: np.ndarray,
    wins: np.ndarray,
    episode_ids: np.ndarray | None,
    *,
    open_percentile: float,
    saturation_percentile: float,
    advantage_floor: float,
) -> dict[str, Any]:
    """Evaluate the exact training SIL gate on validation transitions.

    Reports aggregate, win-only, and loss-only metrics using the shared
    :func:`sil_percentile_gate` helper so the probe and PPO cannot drift.
    Computes raw advantages (``R_mc - V``) over each subset and derives the
    percentile gate over that subset's advantages. Also attributes gate mass
    to wins vs losses and counts unique episodes carrying gate weight.
    """
    raw_advantage = mc_returns - predictions
    win_mask = wins.astype(bool)
    loss_mask = ~win_mask
    result: dict[str, Any] = {
        "aggregate": _gate_subset(
            raw_advantage,
            open_percentile=open_percentile,
            saturation_percentile=saturation_percentile,
            advantage_floor=advantage_floor,
        ),
        "win": _gate_subset(
            raw_advantage[win_mask],
            open_percentile=open_percentile,
            saturation_percentile=saturation_percentile,
            advantage_floor=advantage_floor,
        ),
        "loss": _gate_subset(
            raw_advantage[loss_mask],
            open_percentile=open_percentile,
            saturation_percentile=saturation_percentile,
            advantage_floor=advantage_floor,
        ),
    }
    # Gate-mass attribution across the aggregate gate.
    gate, _ = sil_percentile_gate(
        raw_advantage,
        open_percentile=open_percentile,
        saturation_percentile=saturation_percentile,
        advantage_floor=advantage_floor,
    )
    gate_mass = float(gate.sum())
    result["gate_weight_from_wins_fraction"] = (
        float((gate[win_mask]).sum()) / gate_mass if gate_mass > 0.0 else 0.0
    )
    if episode_ids is not None and gate_mass > 0.0:
        contributing = gate > 0.0
        result["unique_episodes_with_gate_mass"] = len(np.unique(episode_ids[contributing]))
    else:
        result["unique_episodes_with_gate_mass"] = 0
    result["winning_states"] = int(win_mask.sum())
    result["loss_states"] = int(loss_mask.sum())
    return result


def _target_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.5)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "below_hl_support_fraction": float(np.mean(values < -8.0)),
        "above_hl_support_fraction": float(np.mean(values > 12.0)),
    }


def _aggregate_section(runs: list[dict[str, Any]], section: str) -> dict[str, Any]:
    """Aggregate a metric section across runs, recursing one level for nesting.

    ``sil_gate`` now carries aggregate/win/loss sub-dicts plus flat attribution
    keys; ``critic`` stays flat. Both are handled by recursing into dict values.
    """
    keys = runs[0][section].keys()
    result: dict[str, Any] = {}
    for key in keys:
        sample = runs[0][section][key]
        if isinstance(sample, dict):
            sub_result: dict[str, Any] = {}
            for sub_key in sample:
                values = [
                    float(run[section][key][sub_key])
                    for run in runs
                    if sub_key in run[section][key]
                    and np.isfinite(run[section][key][sub_key])
                ]
                if values:
                    sub_result[sub_key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
            if sub_result:
                result[key] = sub_result
        elif isinstance(sample, (int, float)):
            values = [float(run[section][key]) for run in runs if section in run and key in run[section]]
            if values:
                result[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return result


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in ("critic", "sil_gate"):
        if section in runs[0]:
            aggregated = _aggregate_section(runs, section)
            if aggregated:
                result[section] = aggregated
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline scalar versus HL-Gauss critic and SIL-gate probe."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fit-seeds", default="0,1,2")
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument(
        "--sil-advantage-floor",
        type=float,
        default=0.25,
        help="Absolute advantage floor in raw reward units (default: 0.25).",
    )
    parser.add_argument(
        "--sil-gate-open-percentile",
        type=float,
        default=80.0,
        help="Percentile at which the gate opens (default: 80).",
    )
    parser.add_argument(
        "--sil-gate-saturation-percentile",
        type=float,
        default=95.0,
        help="Percentile at which the gate saturates (default: 95).",
    )
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 8:
        raise ValueError("--games must be at least 8")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    fit_seeds = [int(item) for item in args.fit_seeds.split(",")]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device_from_arg(args.device)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = load_checkpoint_payload(checkpoint, "cpu")
    config_fields = {
        field.name: payload["agent_config"][field.name]
        for field in dataclasses.fields(AgentConfig)
        if field.name in payload["agent_config"]
    }
    agent_config = AgentConfig(**config_fields)
    data = load_game_data()
    vocab = build_vocab(data)
    model = BalatroAgent(agent_config, vocab).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    LOGGER.info("Loaded %s on %s", checkpoint, device)

    dataset = _collect_episodes(
        model,
        num_games=args.games,
        num_envs=args.envs,
        device=device,
        gamma=0.99,
        seed=args.seed,
    )
    LOGGER.info(
        "Caching transformer features for %d states from %d games",
        len(dataset["rewards"]),
        args.games,
    )
    feature_start = time.monotonic()
    features, checkpoint_predictions = _cache_features(
        model,
        dataset,
        batch_size=args.feature_batch_size,
        device=device,
    )
    feature_seconds = time.monotonic() - feature_start
    lambda_targets = _lambda_returns(
        dataset["rewards"],
        checkpoint_predictions,
        dataset["episode_ids"],
        gamma=0.99,
        gae_lambda=0.95,
    )
    train_indices, validation_indices = _split_episodes(
        dataset["episode_ids"],
        dataset["episode_wins"],
        args.validation_fraction,
        args.seed,
    )
    validation_targets = lambda_targets[validation_indices]
    validation_mc = dataset["mc_returns"][validation_indices]
    validation_wins = dataset["wins"][validation_indices]
    validation_episode_ids = dataset["episode_ids"][validation_indices]

    checkpoint_result = {
        "critic": _prediction_metrics(
            validation_targets, checkpoint_predictions[validation_indices]
        ),
        "sil_gate": _sil_gate_metrics(
            validation_mc,
            checkpoint_predictions[validation_indices],
            validation_wins,
            validation_episode_ids,
            open_percentile=args.sil_gate_open_percentile,
            saturation_percentile=args.sil_gate_saturation_percentile,
            advantage_floor=args.sil_advantage_floor,
        ),
    }
    initial_pool_state = model.value_head.pool_proj.state_dict()
    fitted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fit_start = time.monotonic()
    for fit_seed in fit_seeds:
        for name, value_bins in (("scalar_refit", 0), ("hl_gauss", 51)):
            head, fit_info = _fit_head(
                features,
                lambda_targets,
                train_indices,
                validation_indices,
                value_bins=value_bins,
                initial_pool_state=initial_pool_state,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=fit_seed,
                device=device,
            )
            predictions = _predict_head(
                head,
                features,
                validation_indices,
                batch_size=args.batch_size,
                device=device,
            )
            fitted[name].append(
                {
                    "seed": fit_seed,
                    "fit": fit_info,
                    "critic": _prediction_metrics(validation_targets, predictions),
                    "sil_gate": _sil_gate_metrics(
                        validation_mc,
                        predictions,
                        validation_wins,
                        validation_episode_ids,
                        open_percentile=args.sil_gate_open_percentile,
                        saturation_percentile=args.sil_gate_saturation_percentile,
                        advantage_floor=args.sil_advantage_floor,
                    ),
                }
            )
            LOGGER.info(
                "%s seed=%d: val_mse=%.4f EV=%.3f SIL_gate=%.3f",
                name,
                fit_seed,
                fitted[name][-1]["critic"]["mse"],
                fitted[name][-1]["critic"]["explained_variance"],
                fitted[name][-1]["sil_gate"]["gate_mean"],
            )
    fit_seconds = time.monotonic() - fit_start

    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_update": payload.get("update_count"),
        "device": str(device),
        "config": vars(args),
        "dataset": {
            "games": args.games,
            "states": len(dataset["rewards"]),
            "wins": int(sum(dataset["episode_wins"].values())),
            "attempted_episodes": int(dataset["attempted_episodes"]),
            "retained_episodes": int(dataset["retained_episodes"]),
            "stalled_episodes_dropped": int(dataset["stalled_episodes_dropped"]),
            "validation_games": len(np.unique(dataset["episode_ids"][validation_indices])),
            "validation_states": len(validation_indices),
            "collection_seconds": dataset["collection_seconds"],
            "feature_seconds": feature_seconds,
            "fit_seconds": fit_seconds,
            "lambda_target_stats": _target_stats(lambda_targets),
            "mc_return_stats": _target_stats(dataset["mc_returns"]),
        },
        "checkpoint_scalar": checkpoint_result,
        "fits": dict(fitted),
        "aggregate": {name: _aggregate_runs(runs) for name, runs in fitted.items()},
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    scalar = result["aggregate"]["scalar_refit"]
    hl = result["aggregate"]["hl_gauss"]
    print(f"Dataset: {args.games} games, {result['dataset']['states']} states, {result['dataset']['wins']} wins")
    print(
        "Held-out critic MSE: "
        f"scalar={scalar['critic']['mse']['mean']:.4f}, "
        f"HL-Gauss={hl['critic']['mse']['mean']:.4f}"
    )
    print(
        "Held-out explained variance: "
        f"scalar={scalar['critic']['explained_variance']['mean']:.3f}, "
        f"HL-Gauss={hl['critic']['explained_variance']['mean']:.3f}"
    )
    print(
        "Winning-state SIL gate mean: "
        f"checkpoint={checkpoint_result['sil_gate']['gate_mean']:.3f}, "
        f"scalar={scalar['sil_gate']['gate_mean']['mean']:.3f}, "
        f"HL-Gauss={hl['sil_gate']['gate_mean']['mean']:.3f}"
    )
    print(f"Result: {output}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
