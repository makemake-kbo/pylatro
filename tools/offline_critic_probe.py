#!/usr/bin/env python3
"""Evaluate v8 outcome calibration and composed-return quality offline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.reward import (
    REWARD_MODEL_VERSION,
    RewardConfig,
    reward_config_fingerprint,
)
from pylatro_agent.survival import (
    DEFAULT_MAX_ANTES,
    terminal_outcome_class,
    validate_critic_win_ante,
)
from pylatro_agent.training.ppo_observations import (
    _extract_step_info_value,
    _ObsBuffer,
)
from pylatro_agent.training.ppo_rollout import _make_vectorized_envs
from pylatro_agent.training.sil import sil_percentile_gate
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


def _discounted_returns(
    rewards: Sequence[float] | np.ndarray,
    gamma: float,
) -> np.ndarray:
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
        count = 0 if len(group) == 0 else min(
            len(group),
            max(1, round(len(group) * validation_fraction)),
        )
        validation_episodes.extend(group[:count].tolist())
    validation_mask = np.isin(episode_ids, np.asarray(validation_episodes, dtype=np.int64))
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def _gate_subset(
    advantages: np.ndarray,
    *,
    open_percentile: float,
    saturation_percentile: float,
    advantage_floor: float,
) -> dict[str, float]:
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
            "open_threshold": float(info["open_threshold"]),
            "gate_mean": float("nan"),
            "gate_positive_fraction": float("nan"),
            "gate_saturation_fraction": float("nan"),
            "noise_floor_rejected_fraction": float("nan"),
        }
    return {
        "states": int(advantages.size),
        "advantage_mean": float(np.mean(advantages)),
        "open_threshold": float(info["open_threshold"]),
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
    advantages = mc_returns - predictions
    win_mask = wins.astype(bool)
    gate, _ = sil_percentile_gate(
        advantages,
        open_percentile=open_percentile,
        saturation_percentile=saturation_percentile,
        advantage_floor=advantage_floor,
    )
    gate_mass = float(gate.sum())
    return {
        "aggregate": _gate_subset(
            advantages,
            open_percentile=open_percentile,
            saturation_percentile=saturation_percentile,
            advantage_floor=advantage_floor,
        ),
        "win": _gate_subset(
            advantages[win_mask],
            open_percentile=open_percentile,
            saturation_percentile=saturation_percentile,
            advantage_floor=advantage_floor,
        ),
        "loss": _gate_subset(
            advantages[~win_mask],
            open_percentile=open_percentile,
            saturation_percentile=saturation_percentile,
            advantage_floor=advantage_floor,
        ),
        "gate_weight_from_wins_fraction": (
            float(gate[win_mask].sum()) / gate_mass if gate_mass > 0.0 else 0.0
        ),
        "unique_episodes_with_gate_mass": (
            len(np.unique(episode_ids[gate > 0.0]))
            if episode_ids is not None and gate_mass > 0.0
            else 0
        ),
        "winning_states": int(win_mask.sum()),
        "loss_states": int((~win_mask).sum()),
    }


def _target_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p50": float(np.quantile(values, 0.5)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def _checkpoint_reward_config(
    payload: dict[str, Any],
    *,
    checkpoint: str | Path = "checkpoint",
    gamma_override: float | None = None,
    win_ante_override: int | None = None,
) -> RewardConfig:
    """Reconstruct and verify the exact reward task saved by a checkpoint."""

    snapshot = payload.get("reward_config")
    field_names = {field.name for field in dataclasses.fields(RewardConfig)}
    if not isinstance(snapshot, dict) or set(snapshot) != field_names:
        raise RuntimeError(
            f"Checkpoint {checkpoint} lacks a complete reward_config; the probe cannot "
            "reconstruct the task used for critic targets."
        )
    saved_version = payload.get("reward_model_version")
    if saved_version != REWARD_MODEL_VERSION:
        raise RuntimeError(
            f"Checkpoint {checkpoint} uses reward_model_version={saved_version!r}, but "
            f"the active environment implements version {REWARD_MODEL_VERSION}; probing "
            "would measure a different reward task."
        )
    reward_config = RewardConfig(**snapshot)
    saved_fingerprint = payload.get("reward_fingerprint")
    expected_fingerprint = reward_config_fingerprint(reward_config)
    if saved_fingerprint != expected_fingerprint:
        raise RuntimeError(
            f"Checkpoint {checkpoint} has inconsistent reward metadata "
            f"(saved={str(saved_fingerprint)[:12]}, "
            f"computed={expected_fingerprint[:12]})."
        )

    win_ante = validate_critic_win_ante(
        reward_config.potential_win_ante,
        name="checkpoint reward_config.potential_win_ante",
    )
    if gamma_override is not None and not np.isclose(
        gamma_override,
        reward_config.gamma,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            f"--gamma={gamma_override} does not match checkpoint gamma="
            f"{reward_config.gamma}; overrides cannot change the probe task."
        )
    if win_ante_override is not None and win_ante_override != win_ante:
        raise ValueError(
            f"--win-ante={win_ante_override} does not match checkpoint victory Ante="
            f"{win_ante}; overrides cannot change the probe task."
        )
    return reward_config


def _collect_episodes(
    model: BalatroAgent,
    *,
    num_games: int,
    num_envs: int,
    device: torch.device,
    seed: int,
    reward_config: RewardConfig,
) -> dict[str, Any]:
    data = load_game_data()
    vocab = build_vocab(data)
    gamma = reward_config.gamma
    win_ante = reward_config.potential_win_ante
    vec_env = _make_vectorized_envs(
        num_envs,
        data,
        vocab,
        max_no_progress_steps=256,
        use_async=num_envs > 1,
        win_ante=win_ante,
        reward_config=reward_config,
        env_seed_base=seed * 10_000_019 + 700_000_001,
    )
    obs_dict, _ = vec_env.reset()
    obs_buf = _ObsBuffer(num_envs, device)
    obs_buf.update(obs_dict)
    per_env: list[dict[str, list[Any]]] = [
        {
            "rewards": [],
            "expected_returns": [],
            "terminal_values": [],
            "return_residuals": [],
            "outcome_probabilities": [],
        }
        for _ in range(num_envs)
    ]
    completed: list[dict[str, Any]] = []
    attempted = 0
    stalled_dropped = 0
    started = time.monotonic()
    model.eval()
    try:
        while len(completed) < num_games:
            with torch.inference_mode():
                distribution, values = model.action_distribution(
                    obs_buf.tokens,
                    obs_buf.token_types,
                    obs_buf.scalars,
                    obs_buf.attention_mask,
                    obs_buf.action_mask,
                    history_events=obs_buf.history_events,
                    history_event_features=obs_buf.history_event_features,
                    history_cards=obs_buf.history_cards,
                    history_card_mask=obs_buf.history_card_mask,
                    history_jokers=obs_buf.history_jokers,
                    history_joker_mask=obs_buf.history_joker_mask,
                    history_event_mask=obs_buf.history_event_mask,
                    history_round_mask=obs_buf.history_round_mask,
                    history_omitted=obs_buf.history_omitted,
                )
                actions = distribution.sample()
            value_arrays = {
                key: values[key].detach().float().cpu().numpy()
                for key in (
                    "expected_return",
                    "terminal_value",
                    "return_residual",
                    "outcome_probabilities",
                )
            }
            next_obs, rewards, terminated, truncated, infos = vec_env.step(actions.cpu().numpy())
            dones = terminated | truncated
            for env_index in range(num_envs):
                episode = per_env[env_index]
                episode["rewards"].append(float(rewards[env_index]))
                for key, array in value_arrays.items():
                    episode[f"{key}s" if key != "outcome_probabilities" else key].append(array[env_index].copy())
                if not dones[env_index]:
                    continue
                attempted += 1
                won = bool(_extract_step_info_value(infos, "won", env_index, done=True, default=False))
                stalled = bool(_extract_step_info_value(infos, "stalled", env_index, done=True, default=False))
                final_ante_value = _extract_step_info_value(
                    infos,
                    "ante",
                    env_index,
                    done=True,
                    default=1,
                )
                final_ante = int(final_ante_value if final_ante_value is not None else 1)
                if stalled:
                    stalled_dropped += 1
                elif len(completed) < num_games:
                    completed.append(
                        {
                            **episode,
                            "won": won,
                            "final_ante": final_ante,
                            "outcome_target": terminal_outcome_class(won=won, final_ante=final_ante),
                        }
                    )
                per_env[env_index] = {
                    "rewards": [],
                    "expected_returns": [],
                    "terminal_values": [],
                    "return_residuals": [],
                    "outcome_probabilities": [],
                }
            obs_buf.update(next_obs)
    finally:
        vec_env.close()

    episode_ids: list[np.ndarray] = []
    rewards_flat: list[np.ndarray] = []
    mc_returns: list[np.ndarray] = []
    expected_returns: list[np.ndarray] = []
    terminal_values: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    wins: list[np.ndarray] = []
    episode_wins: dict[int, bool] = {}
    for episode_id, episode in enumerate(completed):
        rewards = np.asarray(episode["rewards"], dtype=np.float32)
        length = len(rewards)
        rewards_flat.append(rewards)
        mc_returns.append(_discounted_returns(rewards, gamma))
        expected_returns.append(np.asarray(episode["expected_returns"], dtype=np.float32))
        terminal_values.append(np.asarray(episode["terminal_values"], dtype=np.float32))
        residuals.append(np.asarray(episode["return_residuals"], dtype=np.float32))
        probabilities.append(np.asarray(episode["outcome_probabilities"], dtype=np.float32))
        targets.append(np.full(length, episode["outcome_target"], dtype=np.int64))
        wins.append(np.full(length, episode["won"], dtype=np.bool_))
        episode_ids.append(np.full(length, episode_id, dtype=np.int64))
        episode_wins[episode_id] = bool(episode["won"])
    return {
        "rewards": np.concatenate(rewards_flat),
        "mc_returns": np.concatenate(mc_returns),
        "expected_returns": np.concatenate(expected_returns),
        "terminal_values": np.concatenate(terminal_values),
        "return_residuals": np.concatenate(residuals),
        "outcome_probabilities": np.concatenate(probabilities),
        "outcome_targets": np.concatenate(targets),
        "wins": np.concatenate(wins),
        "episode_ids": np.concatenate(episode_ids),
        "episode_wins": episode_wins,
        "attempted_episodes": attempted,
        "stalled_episodes_dropped": stalled_dropped,
        "collection_seconds": time.monotonic() - started,
    }


def _critic_metrics(dataset: dict[str, Any], indices: np.ndarray) -> dict[str, float]:
    probabilities = dataset["outcome_probabilities"][indices]
    targets = dataset["outcome_targets"][indices]
    one_hot = np.eye(DEFAULT_MAX_ANTES + 1, dtype=np.float32)[targets]
    outcome_brier = np.sum((probabilities - one_hot) ** 2, axis=1)
    selected = probabilities[np.arange(len(indices)), targets]
    climatology = one_hot.mean(axis=0, keepdims=True)
    climatology_brier = np.sum((one_hot - climatology) ** 2, axis=1)
    win_targets = targets == DEFAULT_MAX_ANTES
    win_probabilities = probabilities[:, -1]
    win_climatology = float(np.mean(win_targets))
    derived_win_brier = float(np.mean((win_probabilities - win_targets) ** 2))
    derived_win_climatology_brier = float(
        np.mean((win_targets - win_climatology) ** 2)
    )
    returns = dataset["mc_returns"][indices]
    predictions = dataset["expected_returns"][indices]
    terminal_values = dataset["terminal_values"][indices]
    residual_predictions = dataset["return_residuals"][indices]
    residual_targets = returns - terminal_values
    errors = returns - predictions
    return_variance = float(np.var(returns))
    return {
        "outcome_nll": float(np.mean(-np.log(np.clip(selected, 1e-7, 1.0)))),
        "outcome_brier": float(np.mean(outcome_brier)),
        "outcome_climatology_brier": float(np.mean(climatology_brier)),
        "outcome_brier_skill": (
            1.0 - float(np.mean(outcome_brier)) / float(np.mean(climatology_brier))
            if float(np.mean(climatology_brier)) > 0.0
            else float("nan")
        ),
        "derived_win_brier": derived_win_brier,
        "derived_win_climatology_brier": derived_win_climatology_brier,
        "derived_win_brier_skill": (
            1.0 - derived_win_brier / derived_win_climatology_brier
            if derived_win_climatology_brier > 0.0
            else float("nan")
        ),
        "residual_mae": float(np.mean(np.abs(residual_predictions - residual_targets))),
        "composed_return_mae": float(np.mean(np.abs(errors))),
        "explained_variance": (
            1.0 - float(np.var(errors)) / return_variance
            if return_variance > 1e-12
            else float("nan")
        ),
        "terminal_value_mean": float(np.mean(terminal_values)),
        "return_residual_mean": float(np.mean(residual_predictions)),
        "expected_return_mean": float(np.mean(predictions)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int, default=96)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional assertion; must match the checkpoint's saved gamma.",
    )
    parser.add_argument(
        "--win-ante",
        type=int,
        default=None,
        help="Optional assertion; must match the checkpoint's saved victory Ante.",
    )
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    return parser


def _json_safe(value: Any) -> Any:
    """Convert non-finite probe metrics to JSON null values."""

    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games < 8:
        raise ValueError("--games must be at least 8")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _device_from_arg(args.device)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint_payload(checkpoint, "cpu")
    reward_config = _checkpoint_reward_config(
        payload,
        checkpoint=checkpoint,
        gamma_override=args.gamma,
        win_ante_override=args.win_ante,
    )
    stored_config = payload.get("agent_config") or {}
    valid_fields = {field.name for field in dataclasses.fields(AgentConfig)}
    agent_config = AgentConfig(
        **{key: value for key, value in stored_config.items() if key in valid_fields}
    )
    model = BalatroAgent(agent_config, build_vocab(load_game_data())).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    dataset = _collect_episodes(
        model,
        num_games=args.games,
        num_envs=args.envs,
        device=device,
        seed=args.seed,
        reward_config=reward_config,
    )
    _, validation_indices = _split_episodes(
        dataset["episode_ids"],
        dataset["episode_wins"],
        args.validation_fraction,
        args.seed,
    )
    if validation_indices.size == 0:
        validation_indices = np.arange(len(dataset["rewards"]), dtype=np.int64)
    metrics = _critic_metrics(dataset, validation_indices)
    effective_config = vars(args).copy()
    effective_config.update(
        gamma=reward_config.gamma,
        win_ante=reward_config.potential_win_ante,
        reward_config=dataclasses.asdict(reward_config),
    )
    result = {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "checkpoint_update": payload.get("update_count"),
        "device": str(device),
        "config": effective_config,
        "dataset": {
            "games": args.games,
            "states": len(dataset["rewards"]),
            "validation_states": len(validation_indices),
            "wins": int(sum(dataset["episode_wins"].values())),
            "attempted_episodes": int(dataset["attempted_episodes"]),
            "stalled_episodes_dropped": int(dataset["stalled_episodes_dropped"]),
            "collection_seconds": float(dataset["collection_seconds"]),
            "return_target_stats": _target_stats(dataset["mc_returns"]),
        },
        "critic": metrics,
    }
    output.write_text(
        json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        "Held-out critic: "
        f"outcome_brier={metrics['outcome_brier']:.4f} "
        f"derived_win_brier={metrics['derived_win_brier']:.4f} "
        f"return_mae={metrics['composed_return_mae']:.4f} "
        f"EV={metrics['explained_variance']:.3f}"
    )
    print(f"Result: {output}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
