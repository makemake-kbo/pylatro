#!/usr/bin/env python3
"""Prepare outcome-only policy trials without teacher labels or reward shaping."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.checkpoint import save_checkpoint
from pylatro_agent.constants import CURRENT_ANTE_SCALAR_INDEX
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import (
    BASELINE,
    ROOT,
    actor_digest,
    complete_game_targets,
    critic_from_pooled,
    critic_metrics,
    outcome_reward,
    pooled_critic_inputs,
    study_protocol,
)
from pylatro_agent.reward import reward_checkpoint_metadata
from pylatro_agent.training.evaluation_envs import EvaluationEnvs
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch
from pylatro_agent.training.ppo_policy import _grammar_distribution
from pylatro_agent.transfer import check_seed_separation, load_policy, sha256, source_hashes
from pylatro_agent.value_head import ValueHead, outcome_nll
from pylatro_agent.vocab import build_vocab


def initialize(root):
    root.mkdir(parents=True, exist_ok=False)
    protocol = study_protocol()
    previous = read_json(Path("eval/action_refactor_20260913/manifest.json"))
    manifests = [Path(item["path"]) for item in previous["seed_separation"]]
    seeds = [
        seed
        for name, values in protocol.items()
        if name.endswith("_seeds") and name != "sampling_seeds"
        for seed in values
    ]
    assert len(seeds) == len(set(seeds))
    protocol.update(
        status="initialized",
        created_unix=time.time(),
        checkpoint_sha256=sha256(BASELINE),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_hashes=source_hashes(),
        seed_separation=check_seed_separation(seeds, manifests),
    )
    atomic_json(root / "manifest.json", protocol)
    atomic_json(root / "compute.json", dict(limit_seconds=12 * 3600, elapsed_seconds=0, jobs=[]))
    for name, values in protocol.items():
        if name.endswith("_seeds") and isinstance(values, list):
            atomic_json(root / f"{name}.json", values)
    print(
        json.dumps({k: protocol[k] for k in ("checkpoint", "checkpoint_sha256", "accelerator_limit_seconds")}),
        flush=True,
    )


def collect(root, device, count=None):
    protocol = read_json(root / "manifest.json")
    training = protocol["critic_training_seeds"]
    validation = protocol["critic_validation_seeds"]
    seeds = training + validation
    if count is not None:
        seeds = seeds[:count]
    destination = root / ("critic_games" if count is None else f"critic_smoke_{count}")
    destination.mkdir(exist_ok=False)
    data = load_game_data()
    vocab = build_vocab(data)
    model, _, _ = load_policy(BASELINE, vocab, device, mixture=0.5)
    model.eval()
    torch.manual_seed(1729)
    np.random.seed(1729)
    captured = {}

    def capture(module, args, kwargs):
        captured["x"] = pooled_critic_inputs(args, kwargs).detach().cpu().numpy().copy()

    hook = model.value_head.register_forward_pre_hook(capture, with_kwargs=True)
    envs = EvaluationEnvs(
        4,
        data=data,
        vocab=vocab,
        enable_teacher=False,
        deck_key="b_blue",
        stake=1,
        win_ante=8,
        max_steps=256,
        reward_config=outcome_reward(),
    )
    active, records, completed = {}, {}, []
    next_index = 0
    started = time.monotonic()
    metadata = dict(
        status="running",
        checkpoint_sha256=sha256(BASELINE),
        source_hashes=source_hashes(),
        seeds=seeds,
        device=str(device),
        reward_config=asdict(outcome_reward()),
        batch_size=32,
        workers=4,
        sampling_seed=1729,
        observation="Exact frozen ValueHead pooled encoder features",
        targets="Only outcomes of actions executed by this student; complete episodes; gamma=1",
    )
    atomic_json(destination / "manifest.json", metadata)
    try:
        while next_index < len(seeds) or active:
            while len(active) < 32 and next_index < len(seeds):
                seed = seeds[next_index]
                obs, phase = envs.reset(seed)
                active[seed] = (obs, phase)
                records[seed] = {
                    key: [] for key in ("x", "ante", "action", "phase", "original_win_prob", "original_return")
                }
                next_index += 1
            order = list(active)
            batch = _obs_dicts_to_batch([active[seed][0] for seed in order], device)
            with torch.inference_mode():
                dist, values = _grammar_distribution(model, batch, temperature=1.0)
                selected = dist.sample()
                assert torch.isfinite(dist.log_prob(selected)).all()
                assert torch.isfinite(values["expected_return"]).all()
                probabilities = values["win_prob"].cpu().numpy()
                returns = values["expected_return"].cpu().numpy()
                actions = selected.cpu().tolist()
            for index, seed in enumerate(order):
                obs, phase = active[seed]
                action = actions[index]
                assert obs["action_mask"][action]
                record = records[seed]
                for key, value in dict(
                    x=captured["x"][index],
                    ante=obs["scalars"][CURRENT_ANTE_SCALAR_INDEX],
                    action=action,
                    phase=phase,
                    original_win_prob=probabilities[index],
                    original_return=returns[index],
                ).items():
                    record[key].append(value)
            stepped = envs.step(list(zip(order, actions, strict=True)))
            for seed in order:
                obs, terminated, truncated, info, _ = stepped[seed]
                if terminated or truncated:
                    if truncated or info.get("stalled"):
                        raise RuntimeError(f"Incomplete training trajectory seed={seed}: {info}")
                    record = records.pop(seed)
                    length = len(record["action"])
                    targets, outcomes = complete_game_targets(bool(info["won"]), int(info["ante"]), length)
                    path = destination / f"game_{seed}.npz"
                    np.savez_compressed(
                        path,
                        **{key: np.asarray(value) for key, value in record.items()},
                        returns=targets,
                        target=outcomes,
                        step=np.arange(length),
                        seed=np.full(length, seed),
                    )
                    result = dict(
                        seed=seed,
                        won=bool(info["won"]),
                        final_ante=int(info["ante"]),
                        steps=length,
                        split="training" if seed in training else "validation",
                        path=str(path),
                        sha256=sha256(path),
                    )
                    completed.append(result)
                    with (destination / "outcomes.jsonl").open("a") as stream:
                        stream.write(json.dumps(result) + "\n")
                    del active[seed]
                    if len(completed) % 16 == 0:
                        print(
                            "critic_collection",
                            len(completed),
                            len(seeds),
                            "wins",
                            sum(r["won"] for r in completed),
                            "seconds",
                            round(time.monotonic() - started, 1),
                            flush=True,
                        )
                        atomic_json(
                            root / "status.json",
                            dict(stage="critic_collection", games=len(completed), target=len(seeds)),
                        )
                else:
                    active[seed] = obs, info["phase"]
        metadata.update(
            status="complete",
            games=len(completed),
            states=sum(r["steps"] for r in completed),
            wins=sum(r["won"] for r in completed),
            outcomes_sha256=sha256(destination / "outcomes.jsonl"),
        )
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        envs.close()
        hook.remove()
        metadata["elapsed_seconds"] = time.monotonic() - started
        atomic_json(destination / "manifest.json", metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "source_hashes" and k != "seeds"}), flush=True)


def load_cache(root):
    directory = root / "critic_games"
    metadata = read_json(directory / "manifest.json")
    if metadata["status"] != "complete" or metadata["checkpoint_sha256"] != sha256(BASELINE):
        raise ValueError("Unverified or incomplete critic data")
    rows = [json.loads(line) for line in (directory / "outcomes.jsonl").read_text().splitlines()]
    assert sha256(directory / "outcomes.jsonl") == metadata["outcomes_sha256"]
    assert sorted(r["seed"] for r in rows) == sorted(metadata["seeds"])
    split = {}
    for name in ("training", "validation"):
        arrays = {key: [] for key in ("x", "ante", "returns", "target", "step", "seed")}
        for row in rows:
            if row["split"] != name:
                continue
            assert sha256(row["path"]) == row["sha256"]
            with np.load(row["path"], allow_pickle=False) as data:
                for key in arrays:
                    arrays[key].append(data[key])
        split[name] = {key: torch.from_numpy(np.concatenate(values)) for key, values in arrays.items()}
        split[name]["x"] = split[name]["x"].float()
        split[name]["ante"] = split[name]["ante"].float()
    assert not set(split["training"]["seed"].tolist()) & set(split["validation"]["seed"].tolist())
    return split


def fit(root, seed):
    data = load_cache(root)
    model, _, _ = load_policy(BASELINE, build_vocab(load_game_data()), torch.device("cpu"), mixture=0.5)
    before = actor_digest(model)
    model.config = replace(model.config, win_only_value=True)
    previous = model.value_head.state_dict()
    model.value_head = ValueHead(model.config.d_model, win_only=True)
    model.value_head.load_state_dict(previous)
    torch.nn.init.zeros_(model.value_head.return_residual.weight)
    torch.nn.init.zeros_(model.value_head.return_residual.bias)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.value_head.parameters():
        parameter.requires_grad_(True)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.value_head.parameters(), lr=3e-4, weight_decay=0.01)
    destination = root / f"critic_seed{seed}"
    destination.mkdir(exist_ok=False)
    best, best_state, best_step = float("inf"), None, None
    started = time.monotonic()
    for update in range(2001):
        if update in (0, 100, 300, 600, 1000, 1500, 2000):
            metrics = critic_metrics(model.value_head, data["validation"])
            row = dict(update=update, elapsed_seconds=time.monotonic() - started, validation=metrics)
            with (destination / "learning.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print("critic_fit", seed, update, json.dumps(metrics), flush=True)
            if metrics["selection_loss"] < best:
                best, best_step = metrics["selection_loss"], update
                best_state = {key: value.detach().clone() for key, value in model.value_head.state_dict().items()}
        if update == 2000:
            break
        indices = torch.randint(len(data["training"]["x"]), (320,), generator=generator)
        batch = {key: value[indices] for key, value in data["training"].items()}
        result = critic_from_pooled(model.value_head, batch["x"], batch["ante"])
        loss = (
            outcome_nll(result["outcome_probabilities"], batch["target"])
            + 0.1 * (result["expected_return"] - batch["returns"]).square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    model.value_head.load_state_dict(best_state)
    assert before == actor_digest(model), "Critic fitting changed the actor"
    protocol = read_json(root / "manifest.json")
    reserved = sorted(
        set(
            protocol["development_seeds"]
            + protocol["confirmation_seeds"]
            + protocol["critic_validation_seeds"]
            + protocol["diagnostic_seeds"]
        )
    )
    checkpoint = destination / "checkpoint.pt"
    save_checkpoint(
        model,
        checkpoint,
        extra=dict(
            agent_config=asdict(model.config),
            **reward_checkpoint_metadata(outcome_reward()),
            training_reserved_seeds=reserved,
            critic_warmup=dict(
                source=str(BASELINE),
                source_sha256=sha256(BASELINE),
                seed=seed,
                selected_update=best_step,
                actor_sha256=before,
                source_manifest_sha256=sha256(root / "critic_games/manifest.json"),
            ),
        ),
    )
    metadata = dict(
        status="complete",
        seed=seed,
        selected_update=best_step,
        updates=2000,
        batch_size=320,
        examples_processed=640000,
        device="cpu",
        elapsed_seconds=time.monotonic() - started,
        checkpoint=str(checkpoint),
        checkpoint_sha256=sha256(checkpoint),
        actor_unchanged=True,
        validation=critic_metrics(model.value_head, data["validation"]),
        source_hashes=source_hashes(),
        reward_config=asdict(outcome_reward()),
    )
    atomic_json(destination / "manifest.json", metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "source_hashes"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "collect", "fit"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "init":
        initialize(args.root)
    elif args.command == "collect":
        device = torch.device(args.device)
        if device.type == "mps":
            with accelerator_job(args.root, "critic_collection", 1800):
                collect(args.root, device, args.count)
        else:
            collect(args.root, device, args.count)
    else:
        fit(args.root, args.seed)


if __name__ == "__main__":
    main()
