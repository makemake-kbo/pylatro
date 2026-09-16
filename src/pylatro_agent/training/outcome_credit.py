"""Explicit, preparation-only migration to completed-outcome PPO credit."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np
import torch

from ..checkpoint import load_checkpoint_payload, load_ppo_resume_payload
from ..reward import RewardConfig
from ..transfer import restore_agent_configuration, restore_configuration, sha256, source_hashes
from .cohort_config import LEGACY_CREDIT_DEFAULTS, validate_cohort_resume
from .cohort_training import _validate_state
from .ppo_config import PPOConfig, _ppo_run_provenance, _validate_ppo_config, _validate_resume_provenance
from .ppo_evaluation import validate_resume_evaluation_seeds

_PRESERVED = (
    "state_dict",
    "optimizer_state_dict",
    "rng_states",
    "cohort_state",
    "total_steps",
    "update_count",
    "planned_updates",
    "training_reserved_seeds",
)


def validate_outcome_credit_sources(payload):
    """Require the prepared branch's exact implementation before collection."""

    branch = payload.get("outcome_credit_branch")
    if branch is None:
        return
    if not isinstance(branch, dict) or branch.get("version") != 1 or branch.get("kind") != "completed_outcome_credit":
        raise ValueError("Missing or incompatible outcome credit branch provenance")
    saved_sources = branch.get("source_hashes")
    if not isinstance(saved_sources, dict) or not saved_sources:
        raise ValueError("Outcome credit branch lacks source hashes")
    active_sources = source_hashes()
    if saved_sources != active_sources:
        changed = sorted(
            key for key in set(saved_sources) | set(active_sources) if saved_sources.get(key) != active_sources.get(key)
        )
        raise ValueError(f"Outcome credit branch source changed after preparation: {changed[:10]}")
    digest = hashlib.sha256(json.dumps(saved_sources, sort_keys=True).encode()).hexdigest()
    snapshots = (
        (branch, "source_sha256"),
        (payload.get("ppo_run_provenance"), "source_sha256"),
        (payload.get("cohort_ppo_config"), "ppo_source_sha256"),
        (payload.get("ppo_config_fields"), "ppo_source_sha256"),
    )
    if any(not isinstance(snapshot, dict) or snapshot.get(key) != digest for snapshot, key in snapshots):
        raise ValueError("Outcome credit branch source digest disagrees with checkpoint provenance")


def _same_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same_state(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_same_state(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _parent_configuration(payload):
    saved = payload.get("cohort_ppo_config")
    expected = {field.name for field in fields(PPOConfig)}
    if not isinstance(saved, dict) or set(saved) - expected or expected - set(saved) - LEGACY_CREDIT_DEFAULTS.keys():
        raise ValueError("Outcome credit branch requires the complete known cohort configuration schema")
    missing = expected - set(saved)
    values = copy.deepcopy(saved)
    values.update({key: LEGACY_CREDIT_DEFAULTS[key] for key in missing})
    if not isinstance(values.get("reward_config"), dict):
        raise ValueError("Outcome credit branch requires explicit reward configuration")
    values["reward_config"] = RewardConfig(**values["reward_config"])
    config = PPOConfig(**values)
    if config.collection_mode != "complete_cohort":
        raise ValueError("Outcome credit branch requires an existing complete_cohort checkpoint")
    _validate_ppo_config(config)
    validate_cohort_resume(config, payload)
    _validate_resume_provenance(config, payload)
    validate_resume_evaluation_seeds(config, payload.get("ppo_config_fields") or {})
    active = asdict(config)
    for key, value in (payload.get("ppo_config_fields") or {}).items():
        if key not in active or value != active[key]:
            raise ValueError(f"Inconsistent parent PPO configuration metadata: {key}")
    return config, [
        {"field": f"cohort_ppo_config.{key}", "value": LEGACY_CREDIT_DEFAULTS[key]} for key in sorted(missing)
    ]


def prepare_outcome_credit_branch(parent, destination, *, parent_sha256=None):
    """Create a new CPU-only full-state branch; never allocate compute or run PPO.

    The inherited seed plan, including its consumed prefix, is unchanged. The
    Launch and accelerator budgeting are separate operations. No tensor or
    optimizer state is reset: inactive residual parameters retain their state.
    """

    parent, destination = Path(parent).resolve(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink() or parent == destination.resolve():
        raise FileExistsError(destination)
    digest = sha256(parent)
    if parent_sha256 is not None and parent_sha256 != digest:
        raise ValueError("Parent checkpoint checksum mismatch")
    original = load_checkpoint_payload(parent, "cpu")
    architecture, reward = restore_configuration(original)
    config, migrations = _parent_configuration(original)
    load_ppo_resume_payload(parent, "cpu", active_reward_config=config.reward_config, active_win_ante=config.win_ante)
    if reward != config.reward_config or not architecture.win_only_value:
        raise ValueError("Outcome credit branch requires the parent's unchanged win-only reward and architecture")
    if architecture.hand_ar_mixture_eps != config.hand_ar_mixture_eps:
        raise ValueError("Parent architecture and PPO policy mixture disagree")
    if architecture.return_value_mode != config.return_value_mode:
        raise ValueError("Parent architecture and PPO return_value_mode disagree")
    if any(key not in original for key in _PRESERVED):
        raise ValueError("Outcome credit branch requires complete optimizer, RNG, counters and seed reservations")
    if not original["optimizer_state_dict"] or not original["rng_states"]:
        raise ValueError("Outcome credit branch requires full optimizer and RNG state")
    _validate_state(original["cohort_state"], config, original["update_count"], original["planned_updates"])
    old_config, old_architecture = asdict(config), asdict(architecture)
    sources = source_hashes()
    source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    config = replace(
        config,
        return_value_mode="terminal_only",
        advantage_estimator="complete_episode_mc",
        value_loss_coeff=0.0,
        outcome_label_mode="complete_episode",
        # Older additional_updates checkpoints retained the parent's total.
        # Record normalization to their effective plan as an explicit override.
        total_updates=original["planned_updates"],
        save_dir=str(destination.parent / "checkpoints"),
        log_dir=str(destination.parent / "logs"),
        stop_request_path=str(destination.parent / "STOP"),
        ppo_run_uuid=str(uuid.uuid4()),
        ppo_source_sha256=source_digest,
        ppo_recipe_id="completed_outcome_credit_v1",
    )
    architecture = replace(architecture, return_value_mode="terminal_only")
    _validate_ppo_config(config)
    for directory in (Path(config.save_dir), Path(config.log_dir)):
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Branch runtime directory already exists: {directory}")
    _validate_state(original["cohort_state"], config, original["update_count"], original["planned_updates"])
    updated = asdict(config)
    _unused, architecture_migrations = restore_agent_configuration(original["agent_config"])
    payload = dict(original)
    payload.update(
        agent_config=asdict(architecture),
        ppo_config_fields={**original["ppo_config_fields"], **updated},
        cohort_ppo_config=updated,
        ppo_run_provenance=_ppo_run_provenance(config),
    )
    payload["outcome_credit_branch"] = {
        "version": 1,
        "kind": "completed_outcome_credit",
        "status_scope": "at_preparation",
        "parent": str(parent),
        "parent_sha256": digest,
        "parent_provenance": original.get("ppo_run_provenance"),
        "source_hashes": sources,
        "source_sha256": source_digest,
        "configuration_migrations": [*migrations, *architecture_migrations],
        "ppo_overrides": {
            key: {"from": old_config[key], "to": value} for key, value in updated.items() if old_config[key] != value
        },
        "agent_overrides": {
            key: {"from": old_architecture[key], "to": value}
            for key, value in asdict(architecture).items()
            if old_architecture[key] != value
        },
        "preserved_state": list(_PRESERVED),
        "preserved_consumed_seed_count": original["cohort_state"]["seed_cursor"],
        "remaining_predeclared_games": len(config.cohort_seeds) - original["cohort_state"]["seed_cursor"],
        "training_started": False,
        "accelerator_budget_assigned": False,
    }
    validate_cohort_resume(config, payload)
    _validate_resume_provenance(config, payload)
    validate_outcome_credit_sources(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".outcome-credit-", suffix=".pt", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        torch.save(payload, temporary)
        reloaded = load_checkpoint_payload(temporary, "cpu")
        for key in _PRESERVED:
            if not _same_state(original[key], reloaded[key]):
                raise ValueError(f"Outcome credit branch changed preserved state: {key}")
        if sha256(parent) != digest:
            raise ValueError("Parent checkpoint changed while preparing the branch")
        # Same-filesystem hard link is atomic and fails if the destination was
        # created concurrently. Unlike replace(), it can never overwrite it.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"checkpoint": str(destination), "sha256": sha256(destination), **payload["outcome_credit_branch"]}
