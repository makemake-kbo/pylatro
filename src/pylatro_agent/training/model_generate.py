"""Generate training data by running a trained checkpoint against vectorized envs.

Loads a BalatroAgent checkpoint and uses it as the actor driving
``BalatroEnv`` instances. Actions are sampled from the masked policy so the
collected trajectories are diverse, and full episodes are filtered by
``min_ante`` (or a win) before they are emitted as supervised-training
records.

Records match the schema produced by ``fast_generate.generate_training_data``
so downstream consumers (``train_supervised``) need no changes.
"""

from __future__ import annotations

import logging
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from gymnasium.vector.vector_env import AutoresetMode

from pylatro import GameData, load_game_data

from ..agent import AgentConfig, BalatroAgent
from ..env import BalatroEnv
from ..vocab import Vocab, build_vocab
from .ppo import _extract_step_info_value, _load_checkpoint_compatible, _ObsBuffer

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelGenerateConfig:
    checkpoint_path: str
    num_games: int = 1000
    num_envs: int = 16
    min_ante: int = 5
    gamma: float = 0.997
    sample_temperature: float = 1.0
    max_no_progress_steps: int = 2000
    device: str = "cpu"
    async_envs: bool = True
    progress_interval_s: float = 60.0
    log_dir: str | None = None


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns = [0.0] * len(rewards)
    running = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        running = rewards[idx] + gamma * running
        returns[idx] = running
    return returns


def _make_env_thunk(seed: int, data: GameData, vocab: Vocab, max_no_progress_steps: int):
    def _thunk():
        return BalatroEnv(seed=seed, data=data, vocab=vocab, max_steps=max_no_progress_steps)

    return _thunk


def _make_vectorized_envs(
    num_envs: int,
    data: GameData,
    vocab: Vocab,
    max_no_progress_steps: int,
    use_async: bool,
):
    import gymnasium

    env_fns = [
        _make_env_thunk(i, data, vocab, max_no_progress_steps) for i in range(num_envs)
    ]
    if use_async and num_envs > 1:
        return gymnasium.vector.AsyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    return gymnasium.vector.SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)


def _format_eta(seconds: float) -> str:
    if math.isinf(seconds) or math.isnan(seconds) or seconds < 0:
        return "--"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def generate_training_data_from_model(
    config: ModelGenerateConfig,
    agent_config: AgentConfig,
    data: GameData | None = None,
    vocab: Vocab | None = None,
) -> list[dict[str, Any]]:
    """Run the checkpointed agent against vectorized envs and collect records.

    Returns a list of transition dicts matching
    ``fast_generate.generate_training_data`` (``obs``, ``action``, ``reward``,
    ``won``, ``max_ante``, ``return_target``).
    """
    if data is None:
        data = load_game_data()
    if vocab is None:
        vocab = build_vocab(data)

    device = torch.device(config.device)
    model = BalatroAgent(agent_config, vocab).to(device)
    _load_checkpoint_compatible(model, config.checkpoint_path, device)
    model.eval()
    logger.info(
        "Loaded inference checkpoint %s (d_model=%d, n_layers=%d, params=%d)",
        config.checkpoint_path,
        agent_config.d_model,
        agent_config.n_layers,
        model.count_parameters(),
    )

    vec_env = _make_vectorized_envs(
        config.num_envs, data, vocab, config.max_no_progress_steps, config.async_envs,
    )
    obs_dict, _ = vec_env.reset()

    obs_buf = _ObsBuffer(config.num_envs, device)
    obs_buf.update(obs_dict)

    ep_obs: list[list[dict[str, np.ndarray]]] = [[] for _ in range(config.num_envs)]
    ep_actions: list[list[int]] = [[] for _ in range(config.num_envs)]
    ep_rewards: list[list[float]] = [[] for _ in range(config.num_envs)]
    ep_max_ante: list[int] = [1 for _ in range(config.num_envs)]

    records: list[dict[str, Any]] = []
    attempted = 0
    kept = 0
    wins = 0
    total_steps = 0
    total_ep_steps: list[int] = []  # steps per completed episode
    kept_antes: list[int] = []  # max ante of kept games
    all_antes: list[int] = []  # max ante of all completed games

    writer = None
    if config.log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(config.log_dir)

    logger.info(
        "Generating %d qualifying games with %d envs (min_ante=%d, temperature=%.2f)",
        config.num_games, config.num_envs, config.min_ante, config.sample_temperature,
    )
    start_time = time.monotonic()
    last_report = start_time
    temperature = max(config.sample_temperature, 1e-6)

    try:
        while kept < config.num_games:
            # Snapshot pre-step obs per env (obs_buf is overwritten on update).
            pre_tokens = obs_buf._np_tokens
            pre_token_types = obs_buf._np_token_types
            pre_scalars = obs_buf._np_scalars
            pre_attention = obs_buf._np_attention_mask
            pre_action_mask = obs_buf._np_action_mask

            with torch.no_grad():
                dist, _ = model.action_distribution(
                    obs_buf.tokens, obs_buf.token_types, obs_buf.scalars,
                    obs_buf.attention_mask, obs_buf.action_mask,
                    temperature=temperature,
                )
                actions = dist.sample()

            actions_np = actions.cpu().numpy()
            next_obs_dict, step_rewards, terminated, truncated, infos = vec_env.step(actions_np)
            dones = terminated | truncated
            total_steps += config.num_envs

            for i in range(config.num_envs):
                ep_obs[i].append({
                    "tokens": pre_tokens[i].copy(),
                    "token_types": pre_token_types[i].copy(),
                    "scalars": pre_scalars[i].copy(),
                    "attention_mask": pre_attention[i].copy(),
                    "action_mask": pre_action_mask[i].copy(),
                })
                ep_actions[i].append(int(actions_np[i]))
                ep_rewards[i].append(float(step_rewards[i]))

                step_done = bool(dones[i])
                ante = _extract_step_info_value(infos, "ante", i, done=step_done, default=0)
                if ante is not None:
                    ep_max_ante[i] = max(ep_max_ante[i], int(ante))

                if step_done:
                    attempted += 1
                    won = bool(_extract_step_info_value(infos, "won", i, done=True, default=False))
                    max_ante = ep_max_ante[i]
                    ep_len = len(ep_obs[i])
                    ep_return = sum(ep_rewards[i])
                    total_ep_steps.append(ep_len)
                    all_antes.append(max_ante)
                    if writer is not None:
                        writer.add_scalar("generate/ep_max_ante", max_ante, attempted)
                        writer.add_scalar("generate/ep_length", ep_len, attempted)
                        writer.add_scalar("generate/ep_return", ep_return, attempted)
                        writer.add_scalar("generate/ep_won", float(won), attempted)
                    if won or max_ante >= config.min_ante:
                        returns_list = _discounted_returns(ep_rewards[i], config.gamma)
                        for obs_rec, action_rec, reward_rec, return_rec in zip(
                            ep_obs[i], ep_actions[i], ep_rewards[i], returns_list, strict=True,
                        ):
                            records.append({
                                "obs": obs_rec,
                                "action": action_rec,
                                "reward": reward_rec,
                                "won": won,
                                "max_ante": max_ante,
                                "return_target": return_rec,
                            })
                        kept += 1
                        kept_antes.append(max_ante)
                        if won:
                            wins += 1

                    ep_obs[i] = []
                    ep_actions[i] = []
                    ep_rewards[i] = []
                    ep_max_ante[i] = 1

                    if kept >= config.num_games:
                        break

            obs_buf.update(next_obs_dict)

            now = time.monotonic()
            if now - last_report >= config.progress_interval_s:
                elapsed = now - start_time
                game_rate = kept / max(elapsed, 1e-6)
                step_rate = total_steps / max(elapsed, 1e-6)
                pct_valid = (kept / max(attempted, 1)) * 100
                win_rate = (wins / max(kept, 1)) * 100
                remaining = config.num_games - kept
                eta = remaining / game_rate if game_rate > 0 else float("inf")
                avg_ep_len = sum(total_ep_steps) / max(len(total_ep_steps), 1)
                avg_ante_kept = sum(kept_antes) / max(len(kept_antes), 1)
                avg_ante_all = sum(all_antes) / max(len(all_antes), 1)
                logger.info(
                    "Progress: %d/%d kept (%.1f%% of %d attempted) | "
                    "win_rate=%.1f%% | avg_ante=%.1f (kept=%.1f) | "
                    "avg_ep_len=%.0f | %.0f steps/s | %.2f games/s | "
                    "steps=%d | records=%d | elapsed=%s | ETA=%s",
                    kept, config.num_games, pct_valid, attempted,
                    win_rate, avg_ante_all, avg_ante_kept,
                    avg_ep_len, step_rate, game_rate,
                    total_steps, len(records), _format_eta(elapsed), _format_eta(eta),
                )
                if writer is not None:
                    writer.add_scalar("generate/kept_games", kept, total_steps)
                    writer.add_scalar("generate/attempted_games", attempted, total_steps)
                    writer.add_scalar("generate/pass_rate", pct_valid / 100, total_steps)
                    writer.add_scalar("generate/win_rate", win_rate / 100, total_steps)
                    writer.add_scalar("generate/avg_ante_all", avg_ante_all, total_steps)
                    writer.add_scalar("generate/avg_ante_kept", avg_ante_kept, total_steps)
                    writer.add_scalar("generate/avg_ep_length", avg_ep_len, total_steps)
                    writer.add_scalar("generate/steps_per_sec", step_rate, total_steps)
                    writer.add_scalar("generate/games_per_sec", game_rate, total_steps)
                    writer.add_scalar("generate/records", len(records), total_steps)
                last_report = now
    finally:
        vec_env.close()

    elapsed = time.monotonic() - start_time
    avg_ep_len = sum(total_ep_steps) / max(len(total_ep_steps), 1)
    avg_ante_kept = sum(kept_antes) / max(len(kept_antes), 1)
    avg_ante_all = sum(all_antes) / max(len(all_antes), 1)
    logger.info(
        "Done: %d records from %d kept games (%d attempted, %.1f%% pass rate) in %s | "
        "win_rate=%.1f%% | avg_ante=%.1f (kept=%.1f) | avg_ep_len=%.0f | "
        "%.0f steps/s | %.2f games/s",
        len(records), kept, attempted,
        (kept / max(attempted, 1)) * 100, _format_eta(elapsed),
        (wins / max(kept, 1)) * 100,
        avg_ante_all, avg_ante_kept,
        avg_ep_len,
        total_steps / max(elapsed, 1e-6),
        kept / max(elapsed, 1e-6),
    )
    if writer is not None:
        writer.add_scalar("generate/final/kept_games", kept, 0)
        writer.add_scalar("generate/final/attempted_games", attempted, 0)
        writer.add_scalar("generate/final/pass_rate", kept / max(attempted, 1), 0)
        writer.add_scalar("generate/final/win_rate", wins / max(kept, 1), 0)
        writer.add_scalar("generate/final/avg_ante_all", avg_ante_all, 0)
        writer.add_scalar("generate/final/avg_ante_kept", avg_ante_kept, 0)
        writer.add_scalar("generate/final/avg_ep_length", avg_ep_len, 0)
        writer.add_scalar("generate/final/total_records", len(records), 0)
        writer.add_scalar("generate/final/total_steps", total_steps, 0)
        writer.add_scalar("generate/final/elapsed_s", elapsed, 0)
        writer.close()
    return records


def save_records(records: list[dict[str, Any]], path: str | Path) -> None:
    """Persist generated records to disk using pickle (protocol 5)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)
    logger.info("Saved %d records to %s", len(records), path)


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load previously generated records from disk."""
    path = Path(path)
    with open(path, "rb") as f:
        records = pickle.load(f)
    logger.info("Loaded %d records from %s", len(records), path)
    return records
