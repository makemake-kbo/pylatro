"""PPO initialization, actor transfer, and checkpoint persistence."""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..reward import DEFAULT_REWARD_CONFIG, RewardConfig
from .ppo_config import (
    PPOConfig,
    _effective_reward_config,
    _ppo_run_provenance,
)
from .ppo_policy import (
    _unwrap_model,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ..agent import AgentConfig

logger = logging.getLogger(__name__)


def _load_state_dict_into_model(model: nn.Module, state_dict: dict, checkpoint_path: str) -> None:
    """Strictly load weights, allowing only a DataParallel prefix change."""
    has_module_prefix = any(k.startswith("module.") for k in state_dict)
    is_wrapped = isinstance(model, nn.DataParallel)

    if has_module_prefix and not is_wrapped:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif not has_module_prefix and is_wrapped:
        state_dict = {f"module.{k}": v for k, v in state_dict.items()}

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is architecture-incompatible and does not "
            "exactly match the configured model. Use the source model dimensions "
            "or a matching checkpoint."
        ) from exc


def _load_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    *,
    active_reward_config: RewardConfig | None = None,
) -> None:
    """Load current-schema weights, optionally requiring matching rewards."""
    from ..checkpoint import load_checkpoint_payload
    from ..reward import reward_config_fingerprint

    payload = load_checkpoint_payload(checkpoint_path, device)
    if active_reward_config is not None:
        saved_fingerprint = payload.get("reward_fingerprint")
        active_fingerprint = reward_config_fingerprint(active_reward_config)
        if saved_fingerprint and saved_fingerprint != active_fingerprint:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} was trained with a different reward "
                "fingerprint. Use matching reward settings, or --actor-transfer "
                "to start a new Ante-8 PPO run with a fresh critic and optimizer."
            )
        if not saved_fingerprint:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has no reward fingerprint, so loading its "
                "critic cannot be verified. Use a matching checkpoint or "
                "--actor-transfer to start a new Ante-8 PPO run with a fresh critic."
            )
    _load_state_dict_into_model(model, payload["state_dict"], checkpoint_path)
    _unwrap_model(model)._training_reserved_seeds = payload.get("training_reserved_seeds")


def load_actor_transfer(model: nn.Module, path: str, device: torch.device) -> None:
    """Explicit weights-only migration; never restore old reward/optimizer state.

    Newly exposed features receive zero adapters on explicit legacy transfer.
    v13 also repairs phase/risk semantics, so transfer from older schemas is
    NOT behavior-preserving and requires a new baseline/teacher dataset.
    """
    import hashlib

    from ..schema import ACTOR_TRANSFER_SCHEMAS

    payload = torch.load(path, map_location=device, weights_only=False)
    version = payload.get("tokenizer_version")
    if version not in ACTOR_TRANSFER_SCHEMAS or payload.get("tokenizer_semantics") != ACTOR_TRANSFER_SCHEMAS[version]:
        raise ValueError("Actor transfer supports tokenizer v11, v12, or the current schema")
    saved = {key.removeprefix("module."): value for key, value in payload["state_dict"].items()}
    base_model = _unwrap_model(model)
    current = base_model.state_dict()
    # Discover the module's registered path rather than assuming a naming alias.
    adapters = {key for key in current if key.endswith("state_proj.weight") and "joker" in key}
    observation_adapters = {
        key for key in current if key.endswith(("economy_proj.weight", "card_state_proj.weight"))
    }
    for key, value in current.items():
        if key.startswith("value_head."):
            continue
        if key not in saved:
            if (version == 11 and key in adapters) or (version < 13 and key in observation_adapters):
                current[key] = torch.zeros_like(value)
                continue
            raise ValueError(f"Actor transfer missing {key}; architecture must match the source")
        if value.shape != saved[key].shape:
            raise ValueError(f"Actor transfer shape mismatch for {key}; use source model dimensions")
        current[key] = saved[key]
    extra_keys = set(saved) - set(current)
    if any(not key.startswith("value_head.") for key in extra_keys):
        raise ValueError(f"Actor transfer contains incompatible actor parameters: {sorted(extra_keys)}")
    saved_goal = int(
        (payload.get("ppo_config_fields") or {}).get("win_ante")
        or (payload.get("reward_config") or {}).get("potential_win_ante")
        or 8
    )
    if not 1 <= saved_goal <= 8:
        raise ValueError("Actor transfer requires a source target Ante between 1 and 8")
    if saved_goal != 8:
        for key, value in current.items():
            if key.endswith("win_ante_emb.weight"):
                value = value.clone()
                value[8] = value[saved_goal]
                current[key] = value
    base_model.load_state_dict(current, strict=True)
    with open(path, "rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    base_model._actor_transfer_metadata = {
        "path": str(path),
        "sha256": digest,
        "tokenizer_version": version,
        "source_win_ante": saved_goal,
        "critic_reset": True,
        "training_reserved_seeds": payload.get("training_reserved_seeds"),
    }
    base_model._training_reserved_seeds = payload.get("training_reserved_seeds")
    logger.info("Transferred actor from %s (tokenizer %s); critic and optimizer start fresh", path, version)
    if version < 13:
        logger.warning("Observation semantics changed in v13; regenerate PT data and evaluate transfer before PPO")


def _save_checkpoint(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    save_path: Path,
    update_count: int,
    total_steps: int,
    planned_updates: int,
    entropy_coeff: float,
    entropy_signal_ema: float | None,
    lr: float,
    log_alpha: torch.Tensor | None = None,
    alpha_optimizer: torch.optim.Optimizer | None = None,
    agent_config: AgentConfig | None = None,
    config: PPOConfig | None = None,
    filename: str | None = None,
    extra: dict | None = None,
    vector_env=None,
) -> Path:
    """Persist a full PPO resume checkpoint and return its path."""
    from ..checkpoint import save_ppo_checkpoint

    checkpoint_path = save_path / (filename or f"ppo_update{update_count}.pt")
    # Snapshot the PPOConfig fields that affect resume correctness so a future
    # resume can warn on mismatched hyperparameters (kept lightweight: only
    # scalar scheduling fields, not the full dataclass).
    config_fields: dict[str, object] = {}
    if config is not None:
        for key in (
            "seed",
            "ppo_epochs",
            "mini_batch_size",
            "micro_batch_size",
            "precision",
            "lr",
            "clip_epsilon",
            "gae_lambda",
            "rollout_temperature",
            "danger_rollout_temperature",
            "danger_death_probability_threshold",
            "action_type_entropy_scale",
            "max_no_progress_steps",
            "eval_regression_tolerance",
            "eval_regression_patience",
            "ppo_run_uuid",
            "ppo_source_sha256",
            "ppo_recipe_id",
            "gamma",
            "win_ante",
            "target_entropy",
            "entropy_ema_beta",
            "adaptive_entropy",
            "milestone_final_scale",
            "milestone_decay_fraction",
            "return_path_coeff",
            "return_path_capacity",
            "return_path_batch_size",
            "return_path_samples_per_episode",
            "return_path_rebuild_steps",
            "eval_games",
            "eval_sampled_games",
            "eval_seeds",
        ):
            config_fields[key] = getattr(config, key)
        config_fields["archive_config"] = asdict(config.archive_config) if config.archive_config else None
    active_reward_config = _effective_reward_config(config) if config is not None else DEFAULT_REWARD_CONFIG
    checkpoint_extra = dict(extra or {})
    checkpoint_extra["training_reserved_seeds"] = getattr(_unwrap_model(model), "_training_reserved_seeds", None)
    best_eval_rank = getattr(_unwrap_model(model), "_best_eval_rank", None)
    if best_eval_rank is not None:
        checkpoint_extra["best_eval_rank"] = tuple(best_eval_rank)
    return_paths = getattr(_unwrap_model(model), "_return_path_replay", None)
    if return_paths is not None:
        checkpoint_extra["return_path_replay"] = return_paths.state_dict()
    transfer_metadata = getattr(_unwrap_model(model), "_actor_transfer_metadata", None)
    if transfer_metadata is not None:
        checkpoint_extra["actor_transfer"] = transfer_metadata
    if config is not None and config.archive_config is not None:
        if vector_env is None:
            raise ValueError("Archive checkpoints require the training environments")
        checkpoint_extra["archive_states"] = list(vector_env.call("archive_state_dict"))
    checkpoint_extra["ppo_active_lr"] = float(optimizer.param_groups[0]["lr"])
    if config is not None:
        provenance = _ppo_run_provenance(config)
        if provenance is not None:
            checkpoint_extra["ppo_run_provenance"] = provenance
    save_ppo_checkpoint(
        _unwrap_model(model),
        checkpoint_path,
        optimizer=optimizer,
        update_count=update_count,
        total_steps=total_steps,
        planned_updates=planned_updates,
        entropy_coeff=entropy_coeff,
        entropy_signal_ema=entropy_signal_ema,
        lr=lr,
        log_alpha=log_alpha,
        alpha_optimizer=alpha_optimizer,
        agent_config=agent_config,
        reward_config=active_reward_config,
        ppo_config_fields=config_fields,
        extra=checkpoint_extra,
    )
    return checkpoint_path


def _mirror_latest_checkpoint(checkpoint_path: Path, save_path: Path) -> Path:
    """Mirror a committed checkpoint to the stable strict-resume path."""

    latest_path = save_path / "ppo_latest.pt"
    try:
        latest_path.unlink(missing_ok=True)
        shutil.copy2(checkpoint_path, latest_path)
    except OSError:
        logger.warning("Could not mirror latest checkpoint to %s", latest_path)
    return latest_path
