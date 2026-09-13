#!/usr/bin/env python3
"""Bounded outcome-only PPO legs with pinned recipes and immutable resume copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, outcome_reward
from pylatro_agent.training.ppo import train_ppo
from pylatro_agent.training.ppo_config import PPOConfig, _ppo_run_provenance
from pylatro_agent.transfer import load_policy, restore_configuration, sha256, source_hashes

RECIPES = {"gae097": 0.97, "gae100": 1.0, "gae097_lr3e5": 0.97, "gae100_lr3e5": 1.0,
           "gae097_terminal": 0.97, "gae100_terminal": 1.0,
           "gae097_terminal_lr3e5": 0.97, "gae100_terminal_lr3e5": 1.0}


def same_state_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same_state_tree(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same_state_tree(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right


def validate_branch_recipe(parent_recipe, config, intervention="lr"):
    saved, active = parent_recipe["ppo_config"], asdict(config)
    if intervention not in ("lr", "terminal_critic"):
        raise ValueError("Unknown branch intervention")
    setting = "lr" if intervention == "lr" else "value_loss_coeff"
    permitted = {setting, "ppo_run_uuid", "ppo_source_sha256", "ppo_recipe_id",
                 "save_dir", "log_dir", "stop_request_path"}
    if set(saved) != set(active):
        raise ValueError("Branch requires the same complete PPO configuration schema")
    changed = [key for key in saved if key not in permitted and saved[key] != active[key]]
    if changed or parent_recipe["source_core_sha256"] != config.ppo_source_sha256:
        raise ValueError(f"Branch must preserve the complete recipe and core code: {changed}")


def branch_checkpoint(source, destination, config, intervention="lr", parent_config=None):
    """Fork a declared intervention while preserving unrelated policy and optimizer state."""
    if destination.exists():
        raise FileExistsError(destination)
    original = load_checkpoint_payload(source, "cpu")
    provenance = original.get("ppo_run_provenance") or {}
    if provenance.get("source_sha256") != config.ppo_source_sha256:
        raise ValueError("Branch source uses different core code")
    saved, active = original["ppo_config_fields"], asdict(config)
    identity_keys = {"ppo_run_uuid", "ppo_source_sha256", "ppo_recipe_id"}
    changed_setting = "lr" if intervention == "lr" else "value_loss_coeff"
    if intervention not in ("lr", "terminal_critic"):
        raise ValueError("Unknown branch intervention")
    undeclared = [key for key, value in saved.items()
                  if key not in identity_keys | {changed_setting} and active.get(key) != value]
    if undeclared:
        raise ValueError(f"Undeclared branch configuration changes: {undeclared}")
    if not original.get("optimizer_state_dict") or not original.get("rng_states"):
        raise ValueError("Branch requires a full optimizer and RNG checkpoint")
    payload = dict(original)
    payload["ppo_config_fields"] = {**saved, **{key: active[key] for key in identity_keys}}
    payload["ppo_run_provenance"] = _ppo_run_provenance(config)
    changed_names, changed_ids = set(), set()
    details = {"lr": {"from": saved["lr"], "to": config.lr}}
    if intervention == "terminal_critic":
        from pylatro import load_game_data
        from pylatro_agent.vocab import build_vocab

        if config.gamma != 1 or config.reward_config != outcome_reward() or config.value_loss_coeff != 0:
            raise ValueError("Terminal-only value requires the undiscounted pure-win task and no return loss")
        if parent_config is None:
            raise ValueError("Terminal critic preparation requires the complete parent recipe")
        model, reward, _ = load_policy(source, build_vocab(load_game_data()), torch.device("cpu"), mixture=.5)
        if reward != outcome_reward() or not model.config.win_only_value:
            raise ValueError("Parent critic is not the pure-win task")
        optimizer = original["optimizer_state_dict"]
        groups = optimizer["param_groups"]
        parameters = list(model.named_parameters())
        if len(groups) != 1 or len(groups[0]["params"]) != len(parameters):
            raise ValueError("Cannot establish the saved Adam parameter-name mapping")
        payload["state_dict"] = dict(original["state_dict"])
        payload["optimizer_state_dict"] = {**optimizer, "state": dict(optimizer["state"])}
        for parameter_id, (name, _) in zip(groups[0]["params"], parameters, strict=True):
            if not name.startswith("value_head.return_residual."):
                continue
            changed_names.add(name)
            changed_ids.add(parameter_id)
            payload["state_dict"][name] = torch.zeros_like(original["state_dict"][name])
            if parameter_id in optimizer["state"]:
                payload["optimizer_state_dict"]["state"][parameter_id] = {
                    key: torch.zeros_like(value) if key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") else value
                    for key, value in optimizer["state"][parameter_id].items()
                }
        if changed_names != {"value_head.return_residual.weight", "value_head.return_residual.bias"}:
            raise ValueError("Return correction parameters were not identified exactly")
        details = dict(value_loss_coeff={"from": parent_config["value_loss_coeff"], "to": 0.},
                       return_residual="Zero weights, bias and their Adam moments; all other state preserved",
                       value_definition="10 * outcome_head_probability_of_winning")
    payload["outcome_study_branch"] = dict(
        parent_checkpoint=str(source), parent_checkpoint_sha256=sha256(source), parent_provenance=provenance,
        intervention=details, kind=intervention,
        note="Explicit experiment copy; unrelated weights, moments, RNG and counters are preserved.",
    )
    torch.save(payload, destination)
    reloaded = load_checkpoint_payload(destination, "cpu")
    for name, tensor in original["state_dict"].items():
        other = reloaded["state_dict"][name]
        if (torch.count_nonzero(other).item() != 0 if name in changed_names else not same_state_tree(tensor, other)):
            raise ValueError(f"Unexpected branch parameter change: {name}")
    for parameter_id, state in original["optimizer_state_dict"]["state"].items():
        other = reloaded["optimizer_state_dict"]["state"][parameter_id]
        expected = payload["optimizer_state_dict"]["state"][parameter_id] if parameter_id in changed_ids else state
        if not same_state_tree(expected, other):
            raise ValueError("Unexpected branch optimizer-state change")
    if not same_state_tree(original["optimizer_state_dict"]["param_groups"],
                           reloaded["optimizer_state_dict"]["param_groups"]):
        raise ValueError("Branch changed optimizer groups")
    for key in ("rng_states", "update_count", "total_steps",
                "schedule_total_steps", "training_reserved_seeds"):
        if not same_state_tree(original.get(key), reloaded.get(key)):
            raise ValueError(f"Branch modified training state: {key}")
    return payload["outcome_study_branch"]


def recipe_config(root, recipe, seed, device):
    protocol = read_json(root / "manifest.json")
    directory = root / "trials" / f"{recipe}_seed{seed}"
    held_out = protocol["confirmation_seeds"] + protocol["critic_validation_seeds"] + protocol["diagnostic_seeds"]
    return PPOConfig(
        seed=seed,
        deck_key="b_blue",
        stake=1,
        win_ante=8,
        device=device,
        precision="fp32",
        num_envs=32,
        rollout_length=256,
        total_updates=1000,
        ppo_epochs=2,
        mini_batch_size=640,
        micro_batch_size=320,
        lr=3e-5 if recipe.endswith("_lr3e5") else 1e-5,
        gamma=1.0,
        gae_lambda=RECIPES[recipe],
        clip_epsilon=0.1,
        target_kl=0.015,
        target_kl_max=0.08,
        max_grad_norm=0.5,
        entropy_coeff=0.001,
        adaptive_entropy=False,
        action_type_entropy_scale=0.0,
        value_loss_coeff=0. if "_terminal" in recipe else .25,
        outcome_loss_coeff=0.1,
        advantage_clip_sigma=0.0,
        reward_config=outcome_reward(),
        hand_ar_mixture_eps=0.5,
        rollout_temperature=1.0,
        danger_rollout_temperature=None,
        sil_coeff=0,
        return_path_coeff=0,
        archive_config=None,
        terminal_replay_batch_size=256,
        terminal_replay_updates_per_ppo_update=4,
        terminal_replay_holdout_fraction=0.1,
        terminal_replay_row_uniform=True,
        eval_games=50,
        eval_sampled_games=50,
        eval_seeds=protocol["development_seeds"],
        training_excluded_seeds=held_out,
        eval_before_training=False,
        eval_interval=100000,
        eval_regression_tolerance=None,
        eval_device="cpu",
        eval_workers=4,
        eval_batch_size=32,
        eval_cpu_threads=4,
        max_no_progress_steps=256,
        save_dir=str(directory / "checkpoints"),
        log_dir=str(directory / "tensorboard"),
        checkpoint_interval=5,
        log_interval=1,
        risk_forecast_log=False,
        training_episode_log=True,
        stop_request_path=str(directory / "stop_requested"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--seed", choices=(0, 1), type=int, required=True)
    parser.add_argument("--hours", type=float, default=0.75)
    parser.add_argument("--updates", type=int, default=10)
    initialize = parser.add_mutually_exclusive_group()
    initialize.add_argument("--resume", type=Path)
    initialize.add_argument("--branch-from", type=Path,
                            help="Declared LR or critic intervention preserving unrelated checkpoint state")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    if args.hours <= 0 or args.updates <= 0:
        parser.error("Hours and update count must be positive")
    torch.set_num_threads(4)
    root = args.root
    source = root / f"critic_seed{args.seed}" / "checkpoint.pt"
    source_metadata = read_json(source.parent / "manifest.json")
    if not source_metadata["actor_unchanged"] or sha256(source) != source_metadata["checkpoint_sha256"]:
        raise ValueError("Unverified critic preparation")
    config = recipe_config(root, args.recipe, args.seed, args.device)
    directory = Path(config.save_dir).parent
    directory.mkdir(parents=True, exist_ok=True)
    sources = source_hashes()
    core = {key: value for key, value in sources.items() if key.startswith("src/")}
    core_digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    recipe_path = directory / "recipe.json"
    if recipe_path.exists():
        recipe = read_json(recipe_path)
        if args.resume is None or recipe["source_core_sha256"] != core_digest:
            raise ValueError("Existing recipe requires a matching-source explicit resume")
        if recipe["critic_checkpoint_sha256"] != sha256(source):
            raise ValueError("Critic starting point changed")
        expected_config = asdict(config)
        expected_config.update(
            {key: recipe["ppo_config"][key] for key in ("ppo_run_uuid", "ppo_source_sha256", "ppo_recipe_id")}
        )
        if expected_config != recipe["ppo_config"]:
            raise ValueError("Recipe configuration changed between legs")
        identity = recipe["run_uuid"]
    else:
        if args.resume is not None:
            raise ValueError("Cannot resume a recipe without its initial protocol")
        identity = str(uuid.uuid4())
    config.ppo_run_uuid = identity
    config.ppo_source_sha256 = core_digest
    config.ppo_recipe_id = f"outcome_gamma1_{args.recipe}_v1"
    resume = args.resume
    branch = None
    if args.branch_from:
        parent_recipe = args.branch_from.parent.parent / "recipe.json"
        parent = read_json(parent_recipe)
        intervention = "terminal_critic" if args.recipe.endswith("_terminal") else "lr"
        validate_branch_recipe(parent, config, intervention)
        resume = directory / "branch_initial.pt"
        branch = branch_checkpoint(args.branch_from, resume, config, intervention, parent["ppo_config"])
        branch.update(parent_recipe=str(parent_recipe), parent_recipe_sha256=sha256(parent_recipe))
    if not recipe_path.exists():
        atomic_json(
            recipe_path,
            dict(
                run_uuid=identity,
                recipe=args.recipe,
                training_seed=args.seed,
                source_core_sha256=core_digest,
                source_hashes=sources,
                ppo_config=asdict(config),
                critic_checkpoint_sha256=sha256(source),
                study_manifest_sha256=sha256(root / "manifest.json"),
                evaluation="Separate CPU development panels; periodic checkpoints do not use confirmation",
                branch=branch,
            ),
        )
    payload = load_checkpoint_payload(resume or source, "cpu")
    agent_config, reward = restore_configuration(payload, mixture=0.5)
    if reward != outcome_reward() or not agent_config.win_only_value:
        raise ValueError("PPO initialization does not use the pure outcome task")
    if "_terminal" in args.recipe and any(
        torch.count_nonzero(value).item() for name, value in payload["state_dict"].items()
        if name.startswith("value_head.return_residual.")
    ):
        raise ValueError("Terminal-only critic requires an explicit zero-residual preparation")
    start_update = int(payload.get("update_count", 0))
    target_update = start_update + args.updates
    leg = directory / f"leg_from{start_update}_to{target_update}"
    leg.mkdir(exist_ok=False)
    config.total_updates = target_update
    stop_path = Path(config.stop_request_path)
    stop_path.unlink(missing_ok=True)
    max_seconds = args.hours * 3600
    ledger = read_json(root / "compute.json")
    available = (
        ledger["limit_seconds"]
        - ledger["elapsed_seconds"]
        - read_json(root / "manifest.json")["confirmation_reserve_seconds"]
    )
    if args.device == "mps" and max_seconds + 600 > available:
        raise TimeoutError("This training leg would spend the confirmation reserve")

    def stop():
        stop_path.write_text("Allocated leg time reached; save at the next completed update.\n")

    timer = threading.Timer(max_seconds, stop)
    timer.daemon = True
    metadata = dict(
        status="running",
        start_update=start_update,
        target_update=target_update,
        recipe_sha256=sha256(recipe_path),
        source_checkpoint=str(resume or source),
        source_checkpoint_sha256=sha256(resume or source),
        wall_limit_seconds=max_seconds,
    )
    atomic_json(leg / "manifest.json", metadata)
    started = time.monotonic()
    timer.start()
    try:
        kwargs = dict(config=config, agent_config=agent_config)
        if resume:
            kwargs.update(resume_path=str(resume))
        else:
            kwargs.update(pretrained_path=str(source))
        if args.device == "mps":
            with accelerator_job(root, f"ppo/{args.recipe}/seed{args.seed}/from{start_update}", max_seconds):
                train_ppo(**kwargs)
        else:
            train_ppo(**kwargs)
        latest = Path(config.save_dir) / "ppo_latest.pt"
        final = load_checkpoint_payload(latest, "cpu")
        if "_terminal" in args.recipe and any(
            torch.count_nonzero(value).item() for name, value in final["state_dict"].items()
            if name.startswith("value_head.return_residual.")
        ):
            raise ValueError("Disabled return correction changed during PPO")
        checkpoint = leg / "checkpoint.pt"
        shutil.copy2(latest, checkpoint)
        metadata.update(
            status="complete",
            update_count=final["update_count"],
            total_steps=final["total_steps"],
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
        )
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        timer.cancel()
        metadata["elapsed_seconds"] = time.monotonic() - started
        atomic_json(leg / "manifest.json", metadata)
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
