"""Training environment construction and rollout collection."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector.vector_env import AutoresetMode

from ..action import ActionType
from ..constants import META_START
from ..env import BalatroEnv
from ..reward import REWARD_INFO_KEYS
from ..risk import uncalibrate_analytic_death_probability
from .ppo_metrics import (
    EpisodeHistory,
    _action_type_name,
    _next_blind_clear_outcome,
    _record_action_diagnostics,
    _RolloutMetrics,
)
from .ppo_observations import (
    _extract_step_info_value,
    _obs_dicts_to_batch,
    _ObsBuffer,
    _observation_buffer_batch,
    _ppo_terminal_flags,
)
from .ppo_policy import _grammar_distribution, _policy_temperature_for_scalars
from .rollout_buffer import RolloutBuffer
from .sil import EpisodeReplayBuffer, EpisodeTracker

if TYPE_CHECKING:
    from pylatro import GameData

    from ..archive import ArchiveConfig
    from ..reward import RewardConfig
    from ..vocab import Vocab
    from .ppo_config import PPOConfig

logger = logging.getLogger(__name__)


def _make_env(
    seed: int,
    stake: int,
    data: GameData,
    vocab: Vocab,
    max_no_progress_steps: int,
    win_ante: int | None,
    reward_config: RewardConfig | None = None,
    counterfactual_diagnostic_interval: int = 0,
    archive_config: ArchiveConfig | None = None,
    archive_index: int = 0,
    excluded_seeds: tuple[int, ...] = (),
):
    """Factory for creating a BalatroEnv (used by vectorized env wrappers)."""

    def _thunk():
        return BalatroEnv(
            seed=seed,
            stake=stake,
            data=data,
            vocab=vocab,
            max_steps=max_no_progress_steps,
            win_ante=win_ante,
            reward_config=reward_config,
            enable_teacher=False,
            counterfactual_diagnostic_interval=counterfactual_diagnostic_interval,
            archive_config=archive_config,
            archive_index=archive_index,
            excluded_seeds=excluded_seeds,
        )

    return _thunk


def _make_vectorized_envs(
    num_envs: int,
    data: GameData,
    vocab: Vocab,
    stake: int = 1,
    max_no_progress_steps: int = 256,
    use_async: bool = True,
    win_ante: int | None = None,
    reward_config: RewardConfig | None = None,
    env_seed_base: int = 0,
    counterfactual_diagnostic_interval: int = 0,
    archive_config: ArchiveConfig | None = None,
    excluded_seeds: tuple[int, ...] = (),
):
    """Create a gymnasium VectorEnv (async for multiprocess, sync for single-process)."""
    import gymnasium

    env_fns = [
        _make_env(
            env_seed_base + i,
            stake,
            data,
            vocab,
            max_no_progress_steps,
            win_ante,
            reward_config,
            counterfactual_diagnostic_interval=counterfactual_diagnostic_interval,
            archive_config=archive_config,
            archive_index=i,
            excluded_seeds=excluded_seeds,
        )
        for i in range(num_envs)
    ]

    if use_async and num_envs > 1:
        return gymnasium.vector.AsyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    else:
        return gymnasium.vector.SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)


class RolloutState:
    """Per-environment episode state that must survive rollout boundaries."""

    def __init__(self, config: PPOConfig):
        self.history = EpisodeHistory()
        self.tracker = EpisodeTracker(config.num_envs, gamma=config.gamma)
        self.episode_rewards = np.zeros(config.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(config.num_envs, dtype=np.int64)
        self.pending_shop_forecasts: list[list[dict]] = [[] for _ in range(config.num_envs)]
        self.episode_start_step = np.zeros(config.num_envs, dtype=np.int64)


_MAX_PENDING_SHOP_FORECASTS = 128


def collect_rollout(
    model: nn.Module,
    vec_env,
    obs_buf: _ObsBuffer,
    config: PPOConfig,
    state: RolloutState,
    episode_buffer: EpisodeReplayBuffer,
    *,
    update_count: int,
    risk_forecast_file=None,
    return_path_replay=None,
) -> tuple[RolloutBuffer, _RolloutMetrics, int]:
    """Collect a fresh on-policy suffix, label completions, and compute GAE.

    The caller owns environment/archive lifetime and reward schedules. Episode
    prefixes and forecasts persist in ``state``; the returned PPO buffer holds
    only this collection window. No archived prefix is replayed into it.
    """
    device = obs_buf.device
    effective_win_ante = config.win_ante or 8
    collected_steps = 0
    buffer = RolloutBuffer(
        num_envs=config.num_envs,
        rollout_length=config.rollout_length,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    state.episode_start_step[:] = 0
    rm = _RolloutMetrics()
    # === Collect rollouts (vectorized) ===
    model.eval()
    for step in range(config.rollout_length):
        with torch.no_grad():
            dist, value_dict = _grammar_distribution(
                model,
                _observation_buffer_batch(obs_buf),
                temperature=_policy_temperature_for_scalars(obs_buf.scalars, config),
            )
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            values = value_dict["expected_return"]
            chosen_action_probs = log_probs.exp()
            # Cheap macro-concentration telemetry. This is explicitly
            # not the maximum flat-action probability; computing the
            # exact hand-mixture mode on every rollout step would add
            # substantial evaluation-only work to training.
            max_action_type_probs = dist.action_type_probs.max(dim=-1).values

        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy()
        chosen_action_probs_np = chosen_action_probs.cpu().numpy()
        max_action_type_probs_np = max_action_type_probs.cpu().numpy()
        pre_scalars = obs_buf._np_scalars
        pre_tokens = obs_buf._np_tokens
        rm.clear_probabilities.extend(pre_scalars[:, 11].astype(np.float64).tolist())
        rm.immediate_death_probabilities.extend(pre_scalars[:, 12].astype(np.float64).tolist())
        survival_np = value_dict["ante_survival"].cpu().numpy()
        for env_idx, action_id in enumerate(actions_np):
            if _action_type_name(int(action_id)) != ActionType.SHOP_LEAVE.value:
                continue
            shop_ante = max(round(float(pre_scalars[env_idx, 2])), 1)
            survival_index = min(shop_ante - 1, survival_np.shape[1] - 1)
            calibrated_death = 1.0 - float(pre_scalars[env_idx, 11])
            raw_death = uncalibrate_analytic_death_probability(calibrated_death)
            pending = state.pending_shop_forecasts[env_idx]
            if len(pending) < _MAX_PENDING_SHOP_FORECASTS:
                pending.append(
                    {
                        "shop_ante": int(shop_ante),
                        "shop_blind_index": int(pre_tokens[env_idx, META_START + 3, 0]) // 100,
                        "survival_pred": float(survival_np[env_idx, survival_index]),
                        "analytic_clear_pred": float(pre_scalars[env_idx, 11]),
                        "raw_analytic_clear_pred": 1.0 - raw_death,
                    }
                )

        # Step all envs at once
        next_obs_dict, rewards, terminated, truncated, infos = vec_env.step(actions_np)
        dones = terminated | truncated
        ppo_terminated, ppo_truncated, stalled_flags = _ppo_terminal_flags(
            terminated,
            truncated,
            infos,
        )
        terminal_rewards_np = np.asarray(
            [
                float(
                    _extract_step_info_value(
                        infos,
                        "reward_terminal",
                        env_idx,
                        done=bool(dones[env_idx]),
                        default=0.0,
                    )
                    or 0.0
                )
                for env_idx in range(config.num_envs)
            ],
            dtype=np.float32,
        )
        reported_rewards_np = np.asarray(
            [
                float(
                    _extract_step_info_value(
                        infos,
                        "reward_total",
                        env_idx,
                        done=bool(dones[env_idx]),
                        default=rewards[env_idx],
                    )
                )
                for env_idx in range(config.num_envs)
            ],
            dtype=np.float32,
        )
        if not np.allclose(reported_rewards_np, rewards, rtol=0.0, atol=1e-6):
            max_error = float(np.max(np.abs(reported_rewards_np - rewards)))
            raise RuntimeError(
                f"Environment reward_total does not reconstruct the rollout reward (max absolute error {max_error:.3g})"
            )
        bootstrap_values_np = np.zeros(config.num_envs, dtype=np.float32)
        if np.any(ppo_truncated) and "final_obs" in infos:
            final_obs_arr = infos["final_obs"]
            truncated_indices = [idx for idx in np.where(ppo_truncated)[0] if final_obs_arr[idx] is not None]
            if truncated_indices:
                final_obs_batch = _obs_dicts_to_batch([final_obs_arr[idx] for idx in truncated_indices], device)
                with torch.no_grad():
                    _, truncated_value_dict = _grammar_distribution(
                        model,
                        final_obs_batch,
                        temperature=_policy_temperature_for_scalars(final_obs_batch["scalars"], config),
                    )
                truncated_vals = truncated_value_dict["expected_return"].cpu().numpy()
                bootstrap_values_np[np.asarray(truncated_indices, dtype=np.int64)] = truncated_vals

        # Store transition (using pre-step obs from obs_buf)
        buffer.add_batch(
            step=step,
            obs=obs_buf.as_numpy_dict(),
            actions=actions_np,
            rewards=rewards.astype(np.float32),
            values=values_np,
            log_probs=log_probs_np,
            terminated=ppo_terminated,
            truncated=ppo_truncated,
            bootstrap_values=bootstrap_values_np,
        )
        # Same pre-step observations the rollout buffer stores. The
        # tracker clones compact arrays before the shared _ObsBuffer is
        # overwritten and persists episode prefixes across updates.
        state.tracker.record_step(
            obs_buf.as_numpy_dict(),
            actions_np,
            rewards,
            terminal_rewards=terminal_rewards_np,
            behavior_log_probs=log_probs_np,
            policy_version=update_count,
        )

        # Track per-env episode stats
        state.episode_rewards += rewards
        state.episode_lengths += 1
        rm.step_rewards.extend(rewards.astype(np.float64).tolist())
        rm.chosen_action_probs.extend(chosen_action_probs_np.astype(np.float64).tolist())
        rm.max_action_type_probs.extend(max_action_type_probs_np.astype(np.float64).tolist())
        rm.done_flags.extend(dones.astype(np.float64).tolist())
        rm.terminated_flags.extend(terminated.astype(np.float64).tolist())
        rm.truncated_flags.extend(truncated.astype(np.float64).tolist())
        for action_id in actions_np:
            rm.action_type_counts[_action_type_name(int(action_id))] += 1
        for env_idx in range(config.num_envs):
            step_done = bool(dones[env_idx])
            pre_sub_phase = _extract_step_info_value(
                infos,
                "pre_sub_phase",
                env_idx,
                done=step_done,
                default="",
            )
            action_type_name = _action_type_name(int(actions_np[env_idx]))

            in_choose_action = pre_sub_phase == "choose_action"
            rm.pre_choose_action_flags.append(float(in_choose_action))
            if action_type_name == ActionType.PLAY_SUBSET.value:
                rm.play_subset_count += 1
            elif action_type_name == ActionType.DISCARD_SUBSET.value:
                rm.discard_subset_count += 1

            rm.progress_flags.append(
                float(
                    bool(
                        _extract_step_info_value(
                            infos,
                            "progress_made",
                            env_idx,
                            done=step_done,
                            default=False,
                        )
                    )
                )
            )
            rm.steps_since_progress.append(
                float(_extract_step_info_value(infos, "steps_since_progress", env_idx, done=step_done, default=0))
            )
            if action_type_name == ActionType.PLAY_SUBSET.value:
                objective = _extract_step_info_value(
                    infos, "joker_order_objective", env_idx, done=step_done, default=""
                )
                if objective:
                    rm.joker_order_money_flags.append(float(str(objective) == "money"))
                    rm.joker_order_clear_flags.append(
                        float(
                            bool(
                                _extract_step_info_value(
                                    infos, "joker_order_clears", env_idx, done=step_done, default=False
                                )
                            )
                        )
                    )
                    rm.joker_order_dollars_gained.append(
                        float(
                            _extract_step_info_value(
                                infos, "joker_order_dollars_gained", env_idx, done=step_done, default=0.0
                            )
                            or 0.0
                        )
                    )
                    rm.joker_order_chips_forgone.append(
                        float(
                            _extract_step_info_value(
                                infos, "joker_order_chips_forgone", env_idx, done=step_done, default=0.0
                            )
                            or 0.0
                        )
                    )
            for component_name in REWARD_INFO_KEYS:
                component_value = _extract_step_info_value(
                    infos,
                    component_name,
                    env_idx,
                    done=step_done,
                    default=None,
                )
                if component_value is not None:
                    rm.reward_component_values[component_name].append(float(component_value))

            _record_action_diagnostics(rm, infos, env_idx, done=step_done)

        # Handle completed episodes (vectorized envs auto-reset)
        for i in np.where(dones)[0]:
            ep_reward = float(state.episode_rewards[i])
            ep_length = int(state.episode_lengths[i])
            ep_won = bool(_extract_step_info_value(infos, "won", i, done=True, default=False))
            from_archive = bool(_extract_step_info_value(infos, "archive_start", i, done=True, default=False))
            start_ante = int(_extract_step_info_value(infos, "start_ante", i, done=True, default=1))
            state.history.origin_wins["archive" if from_archive else "fresh"].append(float(ep_won))
            ep_stalled = bool(stalled_flags[i])
            ep_ante = int(_extract_step_info_value(infos, "ante", i, done=True, default=1))
            if not ep_stalled:
                for ante in range(start_ante, min(ep_ante, 8) + 1):
                    state.history.continuation_survival[ante].append(float(ep_won or ep_ante > ante))
            ep_terminal_blind = str(
                _extract_step_info_value(
                    infos,
                    "blind_on_deck",
                    i,
                    done=True,
                    default="",
                )
                or ""
            ).lower()
            ep_tarot_uses = int(
                _extract_step_info_value(
                    infos,
                    "tarot_usage_total",
                    i,
                    done=True,
                    default=0,
                )
                or 0
            )
            state.history.rewards.append(ep_reward)
            state.history.lengths.append(ep_length)
            state.history.wins.append(ep_won)
            state.history.stalls.append(ep_stalled)
            state.history.antes.append(ep_ante)
            state.history.tarot_uses.append(ep_tarot_uses)
            rm.completed_episode_rewards.append(ep_reward)
            rm.completed_episode_lengths.append(ep_length)
            rm.completed_episode_wins.append(float(ep_won))
            rm.completed_episode_stalls.append(float(ep_stalled))
            rm.completed_episode_antes.append(ep_ante)
            rm.completed_episode_tarot_uses.append(ep_tarot_uses)
            if not ep_won and not ep_stalled:
                rm.terminal_loss_antes.append(ep_ante)
                terminal_dollars = float(_extract_step_info_value(infos, "dollars", i, done=True, default=0.0) or 0.0)
                terminal_full_weak = bool(
                    _extract_step_info_value(
                        infos,
                        "joker_full",
                        i,
                        done=True,
                        default=False,
                    )
                    and _extract_step_info_value(
                        infos,
                        "weak_confident_joker",
                        i,
                        done=True,
                        default=False,
                    )
                )
                rm.terminal_loss_dollars.append(terminal_dollars)
                rm.terminal_loss_cash_ge_10.append(float(terminal_dollars >= 10.0))
                rm.terminal_loss_joker_full_weak.append(float(terminal_full_weak))
                end_blind = ep_terminal_blind
                rm.terminal_loss_blind_counts[end_blind] += 1
                if ep_ante == 1:
                    rm.ante1_death_blind_counts[end_blind] += 1
                if end_blind == "boss":
                    boss_key = str(_extract_step_info_value(infos, "boss_key", i, done=True, default="") or "")
                    if boss_key:
                        rm.terminal_boss_loss_counts[boss_key] += 1
                blind_target = float(_extract_step_info_value(infos, "blind_target", i, done=True, default=0.0) or 0.0)
                round_score = float(_extract_step_info_value(infos, "round_score", i, done=True, default=0.0) or 0.0)
                if blind_target > 0.0:
                    rm.terminal_loss_score_ratios.append(round_score / blind_target)
                last_play_top1 = _extract_step_info_value(
                    infos,
                    "hand_play_legal_top1",
                    i,
                    done=True,
                    default=_extract_step_info_value(
                        infos,
                        "hand_play_top1",
                        i,
                        done=True,
                        default=None,
                    ),
                )
                if last_play_top1 is not None:
                    rm.terminal_loss_last_play_top1.append(float(bool(last_play_top1)))
                last_play_value_ratio = _extract_step_info_value(
                    infos,
                    "hand_play_legal_candidate_value_ratio",
                    i,
                    done=True,
                    default=_extract_step_info_value(
                        infos,
                        "hand_play_candidate_value_ratio",
                        i,
                        done=True,
                        default=None,
                    ),
                )
                if last_play_value_ratio is not None:
                    rm.terminal_loss_last_play_value_ratios.append(float(last_play_value_ratio))
            if state.pending_shop_forecasts[i]:
                if not ep_stalled:
                    terminal_blind = str(
                        _extract_step_info_value(
                            infos,
                            "blind_on_deck",
                            i,
                            done=True,
                            default="",
                        )
                        or ""
                    )
                    for forecast in state.pending_shop_forecasts[i]:
                        shop_ante = int(forecast["shop_ante"])
                        ante_outcome = float(ep_won or ep_ante > shop_ante)
                        prediction = float(forecast["survival_pred"])
                        rm.shop_survival_predictions.append(prediction)
                        rm.shop_survival_outcomes.append(ante_outcome)
                        rm.shop_survival_briers.append((prediction - ante_outcome) ** 2)
                        blind_outcome = _next_blind_clear_outcome(
                            shop_ante=shop_ante,
                            shop_blind_index=int(forecast["shop_blind_index"]),
                            final_ante=ep_ante,
                            won=ep_won,
                            terminal_blind=terminal_blind,
                        )
                        predicted_death = 1.0 - float(forecast["analytic_clear_pred"])
                        death_outcome = 1.0 - blind_outcome
                        rm.risk_shop_death_predictions.append(predicted_death)
                        rm.risk_shop_death_outcomes.append(death_outcome)
                        rm.risk_shop_death_briers.append((predicted_death - death_outcome) ** 2)
                        raw_predicted_death = 1.0 - float(forecast["raw_analytic_clear_pred"])
                        rm.risk_shop_raw_death_predictions.append(raw_predicted_death)
                        rm.risk_shop_raw_death_briers.append((raw_predicted_death - death_outcome) ** 2)
                        if shop_ante == 1 and death_outcome > 0.5:
                            rm.risk_ante1_false_safe_deaths.append(float(predicted_death <= 0.35))
                        if risk_forecast_file is not None:
                            risk_forecast_file.write(
                                json.dumps(
                                    {
                                        "update": int(update_count),
                                        "shop_ante": shop_ante,
                                        "shop_blind_index": int(forecast["shop_blind_index"]),
                                        "survival_pred": prediction,
                                        "calibrated_death_pred": predicted_death,
                                        "raw_death_pred": raw_predicted_death,
                                        "next_blind_death": death_outcome,
                                        "ante_survived": ante_outcome,
                                        "won": bool(ep_won),
                                        "final_ante": int(ep_ante),
                                    }
                                )
                                + "\n"
                            )
                state.pending_shop_forecasts[i].clear()
            # Insert wins and ordinary completed losses for terminal
            # critic replay. Infrastructure/no-progress stalls are
            # deliberately excluded because their final-Ante outcome
            # is censored rather than a policy loss.
            state.tracker.finish_episode(
                int(i),
                won=ep_won,
                stalled=ep_stalled,
                final_ante=ep_ante,
                win_ante=effective_win_ante,
                terminal_blind=ep_terminal_blind,
                buffer=episode_buffer,
            )
            if return_path_replay is not None and ep_won and not ep_stalled:
                recipe = _extract_step_info_value(infos, "winning_return_path", i, done=True, default=None)
                if recipe is not None:
                    return_path_replay.enqueue(recipe)
            # Stalled episodes are censored, so their outcome mask stays
            # zero. Complete wins and losses get one categorical label.
            if not ep_stalled:
                buffer.set_episode_outcome(
                    env_idx=int(i),
                    start_step=int(state.episode_start_step[i]),
                    end_step=step,
                    won=ep_won,
                    final_ante=ep_ante,
                )
            state.episode_start_step[i] = step + 1
            state.episode_rewards[i] = 0.0
            state.episode_lengths[i] = 0

        # Update obs buffer with new observations
        obs_buf.update(next_obs_dict)
        collected_steps += config.num_envs

    # Bootstrap values for GAE
    with torch.no_grad():
        _, value_dict = _grammar_distribution(
            model,
            _observation_buffer_batch(obs_buf),
            temperature=_policy_temperature_for_scalars(obs_buf.scalars, config),
        )
        last_values = value_dict["expected_return"].cpu().numpy()

    buffer.compute_returns_and_advantages(last_values=last_values)
    buffer.normalize_advantages(clip_sigma=config.advantage_clip_sigma)

    return buffer, rm, collected_steps
