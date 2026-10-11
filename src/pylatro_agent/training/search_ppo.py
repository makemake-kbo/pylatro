"""Resumable public PPO continuation, with optional sampled-world supervision."""

from __future__ import annotations

import copy
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch

from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import capture_rng_states, load_checkpoint_payload, restore_rng_states
from pylatro_agent.public_state import OBSERVATION_INTERPRETATION
from pylatro_agent.reward import RewardConfig
from pylatro_agent.search_ppo_study import atomic, check_deadline, digest, read, save_torch
from pylatro_agent.tokenizer import Tokenizer

from .cohort import CohortBuffer, collect_complete_cohort, policy_digest
from .cohort_config import validate_cohort_config
from .cohort_io import save_cohort
from .evaluation_envs import EvaluationEnvs
from .ppo_config import PPOConfig, _seed_training_rngs
from .ppo_optimization import (
    _clone_state_to_cpu,
    _make_policy_optimizer,
    _restore_policy_optimizer_state,
    _run_ppo_update,
)
from .public_training import PublicTrainingEnv
from .search_guidance import RECIPE, SearchGuidance, proposals


@dataclass
class PublicPPOConfig(PPOConfig):
    # Keep legacy PPOConfig defaults and resume schema unchanged.
    observation_mode: str = OBSERVATION_INTERPRETATION
    search_guidance: dict | None = None


def from_dict(values):
    values = copy.deepcopy(values)
    values["reward_config"] = RewardConfig(**values["reward_config"])
    return PublicPPOConfig(**values)


def require_mps():
    import os

    if not torch.backends.mps.is_available():
        raise RuntimeError("Scientific training requires MPS; no CPU fallback")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for scientific training")


def make_config(parent, *, guided, seeds, rng_seed, updates):
    fields = copy.deepcopy(parent["cohort_ppo_config"])
    fields.update(
        observation_mode=OBSERVATION_INTERPRETATION,
        search_guidance=RECIPE.metadata() if guided else None,
        cohort_seeds=list(seeds),
        seed=rng_seed,
        total_updates=updates,
        device="mps",
        eval_device="cpu",
        eval_cpu_threads=1,
        eval_before_training=False,
        cohort_workers=4,
    )
    config = from_dict(fields)
    validate_cohort_config(config)
    expected = dict(
        cohort_games=64,
        num_envs=32,
        cohort_training_rows=8192,
        ppo_epochs=2,
        mini_batch_size=640,
        micro_batch_size=160,
        lr=1e-5,
        gamma=1.0,
        gae_lambda=0.97,
        clip_epsilon=0.1,
        entropy_coeff=0.001,
        target_kl=0.015,
        target_kl_max=0.08,
        return_path_coeff=0,
        terminal_replay_updates_per_ppo_update=0,
        sil_coeff=0,
        return_value_mode="terminal_only",
        outcome_label_mode="complete_episode",
    )
    if any(getattr(config, k) != v for k, v in expected.items()):
        raise ValueError("Parent PPO recipe differs from the frozen study recipe")
    return config


def validate_resume(payload, identity, config):
    study = payload.get("search_ppo_study", {})
    if study.get("identity") != identity or payload.get("public_ppo_config") != asdict(config):
        raise ValueError("Search PPO checkpoint configuration or provenance changed")
    if payload.get("observation_mode") != OBSERVATION_INTERPRETATION:
        raise ValueError("Wrong observation interpretation")
    step = study["completed_updates"]
    if (
        type(step) is not int
        or not 0 <= step <= config.total_updates
        or study["seed_cursor"] != step * config.cohort_games
        or study["optimizer_examples"] != step * config.ppo_epochs * config.cohort_training_rows
        or len(study["slot_counters"]) != config.num_envs
    ):
        raise ValueError("Invalid training exposure or seed cursor")
    if "torch_mps" not in payload["rng_states"] or "guidance_rng" not in study:
        raise ValueError("Resume lacks required RNG state")


def initial_checkpoint(parent_path, path, config, identity):
    require_mps()
    if path.exists():
        payload = load_checkpoint_payload(path, "cpu")
        validate_resume(payload, identity, config)
        return payload
    payload = load_checkpoint_payload(parent_path, "cpu")
    _seed_training_rngs(config.seed)
    torch.mps.manual_seed(config.seed)
    payload.update(
        public_ppo_config=asdict(config),
        observation_mode=config.observation_mode,
        rng_states=capture_rng_states(),
        search_ppo_study=dict(
            identity=identity,
            completed_updates=0,
            seed_cursor=0,
            slot_counters=copy.deepcopy(payload["cohort_state"]["slot_counters"]),
            optimizer_examples=0,
            collected_transitions=0,
            guidance_rng=np.random.RandomState(config.seed + 40000).get_state(),
        ),
    )
    validate_resume(payload, identity, config)
    save_torch(path, payload)
    return payload


def train_to(
    parent_path,
    directory,
    config,
    identity,
    endpoint,
    *,
    data,
    vocab,
    deadline,
    teach_records,
    progress=lambda **kw: None,
):
    require_mps()
    torch.set_num_threads(1)
    if (directory / "rejected.json").exists():
        raise RuntimeError("Matched comparison stopped after an optimizer-exposure/KL rejection")
    directory.mkdir(parents=True, exist_ok=True)
    initial = directory / "initial.pt"
    initial_checkpoint(parent_path, initial, config, identity)
    candidates = [initial, *sorted(directory.glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))]
    if [int(p.stem.split("_")[-1]) for p in candidates[1:]] != list(range(1, len(candidates))):
        raise ValueError("Non-contiguous committed training updates")
    latest = candidates[-1]
    payload = load_checkpoint_payload(latest, "cpu")
    validate_resume(payload, identity, config)
    study = payload["search_ppo_study"]
    for relative, expected in study.get("evidence", {}).items():
        if digest(directory / relative) != expected:
            raise ValueError("Committed cohort evidence changed")
    if study["completed_updates"]:
        metrics_path = directory / "cohorts" / f"update_{study['completed_updates']}" / "update.json"
        if metrics_path.exists() and read(metrics_path) != study["metrics"]:
            raise ValueError("Committed update metrics changed")
        atomic(metrics_path, study["metrics"])
    if study["completed_updates"] >= endpoint:
        return directory / f"update_{endpoint}.pt" if endpoint else initial
    model = BalatroAgent(AgentConfig(**payload["agent_config"]), vocab).float().eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to("mps")
    optimizer = _make_policy_optimizer(model.parameters(), config.lr)
    _restore_policy_optimizer_state(optimizer, payload, config, torch.device("mps"))
    restore_rng_states(payload["rng_states"])
    label_rng = np.random.RandomState()
    label_rng.set_state(study["guidance_rng"])
    tokenizer = Tokenizer(vocab)
    try:
        while study["completed_updates"] < endpoint:
            check_deadline(deadline)
            update = study["completed_updates"] + 1
            started = time.monotonic()
            cursor = study["seed_cursor"]
            seeds = config.cohort_seeds[cursor : cursor + config.cohort_games]
            artifact = directory / "cohorts" / f"update_{update}"
            if artifact.exists():
                index = 1
                while artifact.with_name(f"update_{update}_attempt{index}").exists():
                    index += 1
                artifact.rename(artifact.with_name(f"update_{update}_attempt{index}"))
            progress(stage="collection", arm=identity["arm"], replica=identity["replica"], update=update)
            model.cpu()
            envs = EvaluationEnvs(
                config.cohort_workers,
                environment_class=PublicTrainingEnv,
                data=data,
                vocab=vocab,
                deck_key=config.deck_key,
                stake=config.stake,
                win_ante=config.win_ante,
                max_steps=config.cohort_max_episode_steps,
                capture_states=config.search_guidance is not None,
                deadline=deadline,
            )
            try:
                episodes, counters = collect_complete_cohort(
                    model, envs, config, seeds, study["slot_counters"], policy_version=payload["update_count"]
                )
                states = envs.public_states()
            finally:
                envs.close()
            collection_seconds = time.monotonic() - started
            buffer = CohortBuffer(episodes, config)
            save_cohort(artifact, episodes, buffer, config)
            labels, label_seconds = [], 0.0
            if config.search_guidance is not None:
                progress(stage="labels", arm=identity["arm"], replica=identity["replica"], update=update)
                label_start = time.monotonic()
                labels = teach_records(proposals(model, states, data, tokenizer), deadline)
                label_seconds = time.monotonic() - label_start
            atomic(artifact / "public_labels.json", labels)
            guidance = SearchGuidance(labels, data, tokenizer, label_rng) if config.search_guidance else None
            progress(stage="optimization", arm=identity["arm"], replica=identity["replica"], update=update)
            check_deadline(deadline)
            model.to("mps")
            optimize_start = time.monotonic()
            stats = _run_ppo_update(
                model,
                optimizer,
                buffer,
                config.entropy_coeff,
                config,
                math.ceil(config.mini_batch_size / config.micro_batch_size),
                config.micro_batch_size,
                torch.device("mps"),
                False,
                search_guidance=guidance,
                step_guard=lambda: check_deadline(deadline),
            )
            torch.mps.synchronize()
            optimization_seconds = time.monotonic() - optimize_start
            if stats.kl_rollback or stats.ppo_samples_processed != config.ppo_epochs * config.cohort_training_rows:
                atomic(directory / "rejected.json", dict(update=update, optimizer=asdict(stats)))
                raise RuntimeError("KL rejection or partial optimizer exposure stops the matched comparison")
            if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
                raise RuntimeError("Non-finite trained parameters")
            study.update(
                completed_updates=update,
                seed_cursor=cursor + config.cohort_games,
                slot_counters=counters,
                optimizer_examples=study["optimizer_examples"] + stats.ppo_samples_processed,
                collected_transitions=study["collected_transitions"] + buffer.size,
                guidance_rng=label_rng.get_state(),
            )
            metrics = dict(
                update=update,
                collection_seconds=collection_seconds,
                label_seconds=label_seconds,
                optimization_seconds=optimization_seconds,
                seconds=time.monotonic() - started,
                optimizer=asdict(stats),
                transitions=buffer.size,
                games=len(episodes),
                labels=len(labels),
                accepted=sum(r["accepted"] for r in labels),
                accepted_games=len({r["seed"] for r in labels if r["accepted"]}),
                rejections=dict(Counter(r["reason"] for r in labels if not r["accepted"])),
                guidance_draws=guidance.draws if guidance else 0,
                guidance_losses=guidance.losses if guidance else [],
                label_simulation_seconds=sum(r["seconds"] for r in labels),
                policy_sha256=policy_digest(model),
            )
            payload.update(
                state_dict=_clone_state_to_cpu(model.state_dict()),
                optimizer_state_dict=_clone_state_to_cpu(optimizer.state_dict()),
                update_count=payload["update_count"] + 1,
                total_steps=payload["total_steps"] + buffer.size,
                rng_states=capture_rng_states(),
                search_ppo_study=study,
            )
            # The portable full checkpoint is the commit; all its evidence is
            # hashed inside it. A crash before rename leaves no completed update.
            study["metrics"] = metrics
            study["evidence"] = {
                str(p.relative_to(directory)): digest(p)
                for p in (artifact / "manifest.json", artifact / "public_labels.json")
            }
            save_torch(directory / f"update_{update}.pt", payload)
            atomic(artifact / "update.json", metrics)
            progress(
                stage="update_complete",
                arm=identity["arm"],
                replica=identity["replica"],
                update=update,
                seconds=metrics["seconds"],
                accepted=metrics["accepted"],
            )
    finally:
        del model, optimizer
        torch.mps.empty_cache()
    return directory / f"update_{endpoint}.pt"
