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

import torch

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, outcome_reward
from pylatro_agent.training.ppo import train_ppo
from pylatro_agent.training.ppo_config import PPOConfig
from pylatro_agent.transfer import restore_configuration, sha256, source_hashes

RECIPES = {"gae097": 0.97, "gae100": 1.0}


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
        lr=1e-5,
        gamma=1.0,
        gae_lambda=RECIPES[recipe],
        clip_epsilon=0.1,
        target_kl=0.015,
        target_kl_max=0.08,
        max_grad_norm=0.5,
        entropy_coeff=0.001,
        adaptive_entropy=False,
        action_type_entropy_scale=0.0,
        value_loss_coeff=0.25,
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
    parser.add_argument("--resume", type=Path)
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
            ),
        )
    payload = load_checkpoint_payload(args.resume or source, "cpu")
    agent_config, reward = restore_configuration(payload, mixture=0.5)
    if reward != outcome_reward() or not agent_config.win_only_value:
        raise ValueError("PPO initialization does not use the pure outcome task")
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
        source_checkpoint=str(args.resume or source),
        source_checkpoint_sha256=sha256(args.resume or source),
        wall_limit_seconds=max_seconds,
    )
    atomic_json(leg / "manifest.json", metadata)
    started = time.monotonic()
    timer.start()
    try:
        kwargs = dict(config=config, agent_config=agent_config)
        if args.resume:
            kwargs.update(resume_path=str(args.resume))
        else:
            kwargs.update(pretrained_path=str(source))
        if args.device == "mps":
            with accelerator_job(root, f"ppo/{args.recipe}/seed{args.seed}/from{start_update}", max_seconds):
                train_ppo(**kwargs)
        else:
            train_ppo(**kwargs)
        latest = Path(config.save_dir) / "ppo_latest.pt"
        final = load_checkpoint_payload(latest, "cpu")
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
