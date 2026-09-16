"""Complete-cohort orchestration using the production policy and PPO optimizer."""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from pylatro import load_game_data

from ..agent import AgentConfig, BalatroAgent
from ..checkpoint import load_ppo_resume_payload, restore_rng_states
from ..learning_bottleneck import atomic_json
from ..transfer import restore_configuration
from ..vocab import build_vocab
from .cohort import CohortBuffer, CohortFailure, collect_complete_cohort, policy_digest
from .cohort_config import COHORT_VERSION, validate_cohort_resume
from .cohort_io import save_cohort
from .evaluation_envs import EvaluationEnvs
from .ppo_checkpoint import _mirror_latest_checkpoint, _save_checkpoint
from .ppo_config import _effective_reward_config, _seed_training_rngs, _validate_resume_provenance
from .ppo_evaluation import evaluate_model, training_seed_reservations, validate_resume_evaluation_seeds
from .ppo_optimization import _make_policy_optimizer, _restore_policy_optimizer_state, _run_ppo_update


def initial_cohort_state(config, update_count=0):
    return {
        "version": COHORT_VERSION,
        "seed_cursor": 0,
        "slot_counters": [0] * config.num_envs,
        "initial_update_count": update_count,
        "optimizer_examples": 0,
        "collected_transitions": 0,
    }


def _validate_state(state, config, update_count, planned):
    cursor = state["seed_cursor"]
    if type(cursor) is not int or cursor < 0 or cursor % config.cohort_games:
        raise ValueError("Invalid cohort seed cursor")
    if cursor // config.cohort_games != update_count - state["initial_update_count"]:
        raise ValueError("Cohort cursor disagrees with checkpoint update count")
    if len(state["slot_counters"]) != config.num_envs or any(
        type(s) is not int or s < 0 for s in state["slot_counters"]
    ):
        raise ValueError("Invalid cohort slot counters")
    if (
        type(state["collected_transitions"]) is not int
        or state["collected_transitions"] != sum(state["slot_counters"])
        or type(state["optimizer_examples"]) is not int
        or state["optimizer_examples"]
        != (cursor // config.cohort_games) * config.cohort_training_rows * config.ppo_epochs
    ):
        raise ValueError("Cohort transition or optimizer counts disagree with its committed state")
    if planned < update_count or cursor + (planned - update_count) * config.cohort_games > len(config.cohort_seeds):
        raise ValueError("Insufficient predeclared game seeds for the requested cohort updates")


def train_complete_cohort(
    config,
    agent_config=None,
    *,
    pretrained_path=None,
    resume_path=None,
    additional_updates=None,
    data=None,
    actor_transfer_path=None,
):
    if pretrained_path or actor_transfer_path:
        raise ValueError("Use an explicit full-state cohort branch to initialize from an existing checkpoint")
    if additional_updates is not None and (not resume_path or additional_updates <= 0):
        raise ValueError("Positive additional_updates requires a cohort resume checkpoint")
    config.reward_config = _effective_reward_config(config)
    config.precision = config.precision or "fp32"
    _seed_training_rngs(config.seed)
    data = data if data is not None else load_game_data()
    vocab = build_vocab(data)
    device = torch.device(config.device)
    payload = None
    update_count, total_steps = 0, 0
    state = initial_cohort_state(config)
    if resume_path:
        from .outcome_credit import validate_outcome_credit_sources

        payload = load_ppo_resume_payload(
            resume_path, "cpu", active_reward_config=config.reward_config, active_win_ante=config.win_ante
        )
        validate_outcome_credit_sources(payload)
        validate_cohort_resume(config, payload)
        validate_resume_evaluation_seeds(config, payload["ppo_config_fields"])
        _validate_resume_provenance(config, payload)
        saved_architecture, _ = restore_configuration(payload)
        if agent_config is not None and asdict(agent_config) != asdict(saved_architecture):
            raise ValueError("Cohort resume architecture differs from checkpoint metadata")
        agent_config = saved_architecture
        state = copy.deepcopy(payload["cohort_state"])
        update_count, total_steps = int(payload["update_count"]), int(payload["total_steps"])
    else:
        agent_config = copy.deepcopy(agent_config or AgentConfig())
        agent_config.hand_ar_mixture_eps = config.hand_ar_mixture_eps
        agent_config.win_only_value = True
        agent_config.return_value_mode = config.return_value_mode
        agent_config.precision = "fp32"
    if (
        agent_config.hand_ar_mixture_eps != config.hand_ar_mixture_eps
        or not agent_config.win_only_value
        or agent_config.return_value_mode != config.return_value_mode
    ):
        raise ValueError("Cohort policy configuration disagrees with the recipe")
    planned = update_count + additional_updates if additional_updates is not None else config.total_updates
    _validate_state(state, config, update_count, planned)
    # Persist the effective plan when callers extend via additional_updates.
    config.total_updates = planned
    model = BalatroAgent(agent_config, vocab).to(device)
    optimizer = _make_policy_optimizer(model.parameters(), config.lr)
    reserved = training_seed_reservations(config)
    model._training_reserved_seeds = reserved
    if payload is not None:
        model.load_state_dict(payload["state_dict"], strict=True)
        _restore_policy_optimizer_state(optimizer, payload, config, device)
        model._training_reserved_seeds = payload.get("training_reserved_seeds")
        if payload.get("actor_transfer"):
            model._actor_transfer_metadata = payload["actor_transfer"]
    directory, logs = Path(config.save_dir), Path(config.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    physical = min(config.micro_batch_size, config.mini_batch_size)
    last_committed_update = update_count

    def checkpoint():
        nonlocal last_committed_update
        # The file rename is the commit point; no live environments cross it.
        committed = directory / f"ppo_update{update_count}.pt"
        if committed.exists():
            raise FileExistsError(f"Preserving existing cohort checkpoint: {committed}")
        path = _save_checkpoint(
            model=model,
            optimizer=optimizer,
            save_path=directory,
            filename="pending.pt",
            update_count=update_count,
            total_steps=total_steps,
            planned_updates=planned,
            entropy_coeff=config.entropy_coeff,
            entropy_signal_ema=None,
            lr=config.lr,
            agent_config=agent_config,
            config=config,
            extra={
                "cohort_state": copy.deepcopy(state),
                "cohort_ppo_config": asdict(config),
                **({"cohort_branch": payload["cohort_branch"]} if payload and "cohort_branch" in payload else {}),
                **(
                    {"outcome_credit_branch": payload["outcome_credit_branch"]}
                    if payload and "outcome_credit_branch" in payload else {}
                ),
            },
        )
        path.replace(committed)
        _mirror_latest_checkpoint(committed, directory)
        last_committed_update = update_count
        return committed

    def evaluation(writer):
        # Shared evaluation protects the policy RNG. Keep training on its device.
        eval_device = torch.device(config.eval_device or config.device)
        try:
            evaluate_model(
                model.to(eval_device),
                data,
                vocab,
                config.eval_games,
                eval_device,
                seeds=config.eval_seeds,
                greedy=True,
                sampled_games=config.eval_sampled_games,
                sampling_seed=config.seed + 31013,
                deck_key=config.deck_key,
                stake=config.stake,
                win_ante=config.win_ante,
                temperature=config.rollout_temperature,
                eval_batch_size=config.eval_batch_size,
                eval_workers=config.eval_workers,
                eval_cpu_threads=config.eval_cpu_threads,
                reward_config=config.reward_config,
                max_no_progress_steps=config.max_no_progress_steps,
                policy_config=config,
                results_path=logs / "eval_outcomes.jsonl",
                writer=writer,
                update_count=update_count,
            )
        finally:
            model.to(device)

    from torch.utils.tensorboard import SummaryWriter

    envs = EvaluationEnvs(
        config.cohort_workers,
        data=data,
        vocab=vocab,
        deck_key=config.deck_key,
        stake=config.stake,
        win_ante=config.win_ante,
        max_steps=config.max_no_progress_steps,
        reward_config=config.reward_config,
        enable_teacher=False,
        excluded_seeds=tuple(reserved),
    )
    try:
        with SummaryWriter(str(logs / "tensorboard")) as writer:
            # Construction and worker startup must not perturb the resumed stream.
            if payload is not None:
                restore_rng_states(payload["rng_states"])
            if payload is None:
                checkpoint()
            if config.eval_before_training:
                evaluation(writer)
            while update_count < planned:
                if config.stop_request_path and Path(config.stop_request_path).exists():
                    break
                started = time.monotonic()
                cursor = state["seed_cursor"]
                seeds = config.cohort_seeds[cursor : cursor + config.cohort_games]
                episodes, counters = collect_complete_cohort(
                    model,
                    envs,
                    config,
                    seeds,
                    state["slot_counters"],
                    policy_version=update_count,
                )
                collection_seconds = time.monotonic() - started
                buffer = CohortBuffer(episodes, config)
                artifact = logs / "cohorts" / f"update_{update_count + 1}"
                if artifact.exists():
                    # Preserve an interrupted attempt and explicitly restart its
                    # same games from the last committed RNG/cursor checkpoint.
                    index = 1
                    while artifact.with_name(artifact.name + f"_attempt{index}").exists():
                        index += 1
                    artifact.rename(artifact.with_name(artifact.name + f"_attempt{index}"))
                save_cohort(artifact, episodes, buffer, config)
                optimize_started = time.monotonic()
                stats = _run_ppo_update(
                    model,
                    optimizer,
                    buffer,
                    config.entropy_coeff,
                    config,
                    math.ceil(config.mini_batch_size / physical),
                    physical,
                    device,
                    False,
                )
                if stats.kl_rollback or stats.ppo_samples_processed != config.ppo_epochs * config.cohort_training_rows:
                    raise RuntimeError("Cohort update failed its matched optimizer-exposure requirement")
                if any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values()):
                    raise RuntimeError("Non-finite model after cohort update")
                optimization_seconds = time.monotonic() - optimize_started
                update_count += 1
                total_steps += buffer.size
                state.update(
                    seed_cursor=cursor + config.cohort_games,
                    slot_counters=counters,
                    optimizer_examples=state["optimizer_examples"] + stats.ppo_samples_processed,
                    collected_transitions=state["collected_transitions"] + buffer.size,
                )
                metrics = {
                    "update": update_count,
                    "cohort_state": copy.deepcopy(state),
                    "collection_seconds": collection_seconds,
                    "optimization_seconds": optimization_seconds,
                    "elapsed_seconds": time.monotonic() - started,
                    "buffer": buffer.diagnostics(),
                    "optimizer": asdict(stats),
                    "policy_sha256": policy_digest(model),
                }
                atomic_json(artifact / "update.json", metrics)
                for key, value in {
                    "collected_transitions": state["collected_transitions"],
                    "optimizer_examples": state["optimizer_examples"],
                    "label_fraction": float(buffer.arrays["terminal_outcome_mask"].mean()),
                    "duplicate_draw_fraction": buffer.diagnostics()["duplicate_draw_fraction"],
                    "collection_seconds": collection_seconds,
                    "optimization_seconds": optimization_seconds,
                }.items():
                    writer.add_scalar("cohort/" + key, value, update_count)
                for key in ("policy_losses", "outcome_nlls", "return_hubers", "grad_norms", "approx_kls"):
                    if key == "return_hubers" and not stats.return_loss_enabled:
                        continue
                    values = getattr(stats, key)
                    if values:
                        writer.add_scalar("cohort/" + key, sum(values) / len(values), update_count)
                writer.add_scalar("cohort/return_loss_enabled", int(stats.return_loss_enabled), update_count)
                for key, value in buffer.diagnostics()["credit_targets"].items():
                    writer.add_scalar("cohort/credit_targets/" + key, value, update_count)
                checkpoint()
                if update_count % config.eval_interval == 0:
                    evaluation(writer)
                writer.flush()
            atomic_json(
                logs / "status.json",
                {
                    "status": "complete" if update_count == planned else "stopped",
                    "update_count": update_count,
                    "total_steps": total_steps,
                    "cohort_state": state,
                },
            )
    except BaseException as exc:
        failure = {
            "status": "failed",
            "error": repr(exc),
            "last_committed_update": last_committed_update,
            "diagnostics": exc.diagnostics if isinstance(exc, CohortFailure) else {},
        }
        failures = logs / "failures"
        failures.mkdir(exist_ok=True)
        index = 1
        while (failures / f"attempt{index}.json").exists():
            index += 1
        atomic_json(failures / f"attempt{index}.json", failure)
        atomic_json(logs / "failure.json", failure)
        raise
    finally:
        envs.close()
    return model
