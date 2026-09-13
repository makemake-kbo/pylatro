"""Controlled supervised continuations; teacher corrections never acquire student outcome targets."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from .checkpoint import save_checkpoint
from .reward import reward_checkpoint_metadata
from .training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from .transfer import CONFIRMATION_SEEDS, DEVELOPMENT_SEEDS, load_policy, sha256, write_json
from .value_head import outcome_nll, return_huber_loss


def correction_record(obs, action, *, seed, step, student_action, checkpoint_sha256, teacher_sources):
    if seed in DEVELOPMENT_SEEDS or seed in CONFIRMATION_SEEDS:
        raise ValueError("Evaluation seeds cannot supply teacher corrections")
    if not obs["action_mask"][action]:
        raise ValueError("Teacher correction must be legal in the recorded observation")
    return {
        "obs": {k: v.copy() for k, v in obs.items()},
        "action": int(action),
        "seed": int(seed),
        "step": int(step),
        "label_source": "V68_student_visited",
        "bc_weight": 1.0,
        "executed_student_action": int(student_action),
        "student_checkpoint_sha256": checkpoint_sha256,
        "teacher_source_sha256": teacher_sources,
    }


def continuation_batch(rows, device, config):
    prepared = []
    for row in rows:
        if row.get("label_source") == "V68_student_visited":
            if any(key in row for key in ("won", "max_ante", "return_target", "terminal_outcome_target", "reward")):
                raise ValueError("Unexecuted teacher actions cannot carry student outcome/critic targets")
            if row.get("bc_weight") != 1.0:
                raise ValueError("Correction rows require unit BC weight")
            prepared.append({**row, "return_target": 0.0, "terminal_outcome_target": 0, "terminal_outcome_mask": 0.0})
        else:
            prepared.append(row)
    batch = _collate_batch(prepared, device, config)
    correction = torch.tensor([r.get("label_source") == "V68_student_visited" for r in rows], device=device)
    batch["bc_weight"][correction] = 1.0
    batch["critic_mask"] = ~correction
    return batch


def continuation_loss(model, batch, config, *, actor_only, hand_weight=1.0):
    from .action import ActionRange

    if not math.isfinite(hand_weight) or hand_weight <= 0:
        raise ValueError("Hand BC weight must be positive and finite")
    dist, values = _grammar_distribution(model, batch)
    logp = dist.log_prob(batch["actions"])
    if not torch.isfinite(logp).all() or (logp < -1e7).any():
        raise RuntimeError("Non-finite or unreachable BC targets")
    actions = batch["actions"]
    hand = ((actions >= int(ActionRange.PLAY_SUBSET_START)) & (actions <= int(ActionRange.PLAY_SUBSET_END))) | (
        (actions >= int(ActionRange.DISCARD_SUBSET_START)) & (actions <= int(ActionRange.DISCARD_SUBSET_END))
    )
    weights = batch["bc_weight"] * torch.where(hand, hand_weight, 1.0)
    bc = -(logp * weights).sum() / weights.sum()
    entropy = dist.entropy().mean()
    critic = torch.zeros((), device=logp.device)
    if not actor_only:
        if not batch["critic_mask"].all():
            raise ValueError("The control objective accepts only executed HF trajectories")
        critic = config.value_loss_coeff * return_huber_loss(values, batch["value_target"])
        critic = critic + config.outcome_loss_coeff * outcome_nll(
            values["outcome_probabilities"], batch["terminal_outcome_target"], batch["terminal_outcome_mask"]
        )
    loss = bc + critic - config.action_entropy_coeff * entropy
    return loss, {"bc_nll_weighted": bc, "critic_loss": critic, "entropy": entropy, "loss": loss}


def train_continuation(
    checkpoint,
    records,
    vocab,
    device,
    output,
    *,
    arm,
    seed,
    corrections=None,
    updates=1000,
    batch_size=320,
    max_seconds=3600,
    mixture=0.5,
    hand_weight=1.0,
    intermediate_steps=(),
    observer=None,
    reserved_seeds=(),
    data_path=None,
    num_games=None,
):
    from torch.utils.tensorboard import SummaryWriter

    if arm not in ("control", "actor_only", "correction"):
        raise ValueError("Unknown continuation arm")
    if arm == "correction" and not corrections:
        raise ValueError("Correction arm requires student-visited labels")
    if batch_size % 2 or updates < 1:
        raise ValueError("Positive updates and even physical batch required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, reward, payload = load_policy(checkpoint, vocab, device, mixture=mixture)
    if model.config.precision != "fp32":
        raise ValueError("Controlled continuations require an FP32 checkpoint")
    saved = payload.get("supervised_config")
    if not saved:
        raise ValueError("Continuation requires the original supervised objective metadata")
    # Only approved trial parameters change. Dropout, entropy and HF weighting stay fixed.
    config = SupervisedConfig(**{**saved, "reward_config": reward})
    config = replace(
        config,
        lr=3e-5,
        weight_decay=0.01,
        warmup_steps=100,
        batch_size=batch_size,
        hand_ar_mixture_eps=mixture,
        deck_key="b_blue",
        stake=1,
        seed=seed,
        device=str(device),
        save_dir=str(output),
        log_dir=str(output / "tensorboard"),
        data_path=str(data_path) if data_path is not None else config.data_path,
        num_games=num_games if num_games is not None else config.num_games,
        excluded_seeds=tuple(sorted(set(tuple(config.excluded_seeds) + tuple(reserved_seeds)))),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # Identical HF order and dropout RNG sequence for control and actor-only replicas.
    indices = records.shuffled_indices()
    correction_rng = np.random.default_rng(seed)
    correction_order = correction_rng.permutation(len(corrections)) if corrections else None
    cursor = 0
    correction_cursor = 0
    started = time.monotonic()
    metadata = {
        "arm": arm,
        "training_seed": seed,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "configuration": asdict(config),
        "optimizer": {
            "name": "AdamW",
            "betas": list(optimizer.defaults["betas"]),
            "eps": optimizer.defaults["eps"],
            "amsgrad": optimizer.defaults["amsgrad"],
            "weight_decay": config.weight_decay,
            "peak_lr": config.lr,
        },
        "learning_rate_schedule": "100-update linear warmup followed by cosine decay over remaining updates",
        "gradient_clip_norm": 1.0,
        "updates_requested": updates,
        "physical_batch": batch_size,
        "precision": "fp32",
        "hf_fraction": 0.5 if corrections else 1.0,
        "hf_order_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "effective_value_loss_coeff": config.value_loss_coeff if arm == "control" else 0.0,
        "effective_outcome_loss_coeff": config.outcome_loss_coeff if arm == "control" else 0.0,
        "status": "running",
        "hand_weight": hand_weight,
        "intermediate_steps": list(intermediate_steps),
    }
    write_json(output / "manifest.json", metadata)
    completed = 0
    def snapshot(step):
        save_checkpoint(
            model, output / f"checkpoint_{step}.pt",
            extra={
                "agent_config": asdict(model.config), "supervised_config": asdict(config),
                "training_reserved_seeds": sorted(set(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS + tuple(reserved_seeds))),
                "continuation": {**metadata, "updates_completed": step},
                **reward_checkpoint_metadata(reward),
            },
        )
        if observer is not None:
            # Diagnostics must not advance training RNG or leave dropout disabled.
            cpu_rng = torch.get_rng_state()
            mps_rng = torch.mps.get_rng_state() if device.type == "mps" else None
            numpy_rng, python_rng = np.random.get_state(), random.getstate()
            was_training = model.training
            try:
                observer(model, step, output)
            finally:
                torch.set_rng_state(cpu_rng)
                if mps_rng is not None:
                    torch.mps.set_rng_state(mps_rng)
                np.random.set_state(numpy_rng)
                random.setstate(python_rng)
                model.train(was_training)
    try:
        with (
            SummaryWriter(str(output / "tensorboard")) as writer,
            (output / "learning.jsonl").open("x", buffering=1) as stream,
        ):
            model.train()
            if 0 in intermediate_steps:
                snapshot(0)
            for update in range(updates):
                if time.monotonic() - started > max_seconds:
                    raise TimeoutError("Continuation budget exhausted; partial arm cannot enter confirmation")
                count = batch_size // 2 if corrections else batch_size
                if cursor + count > len(indices):
                    indices = records.shuffled_indices()
                    cursor = 0
                rows = [records[int(i)] for i in indices[cursor : cursor + count]]
                cursor += count
                if corrections:
                    extra = []
                    while len(extra) < count:
                        if correction_cursor >= len(correction_order):
                            correction_order = correction_rng.permutation(len(corrections))
                            correction_cursor = 0
                        take = min(count - len(extra), len(correction_order) - correction_cursor)
                        extra.extend(
                            corrections[int(i)] for i in correction_order[correction_cursor : correction_cursor + take]
                        )
                        correction_cursor += take
                    rows.extend(extra)
                batch = continuation_batch(rows, device, config)
                loss, metrics = continuation_loss(
                    model, batch, config, actor_only=arm != "control", hand_weight=hand_weight
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                if update < 100:
                    lr = config.lr * (update + 1) / 100
                else:
                    lr = config.lr * (1 + math.cos(math.pi * (update - 100) / max(updates - 100, 1))) / 2
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                completed = update + 1
                if completed in intermediate_steps:
                    snapshot(completed)
                if completed % 10 == 0 or completed == 1:
                    row = {
                        "update": completed,
                        "examples": completed * batch_size,
                        "lr": lr,
                        "elapsed_seconds": time.monotonic() - started,
                        "gradient_norm": float(norm),
                        **{k: float(v.detach()) for k, v in metrics.items()},
                    }
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    for key, value in row.items():
                        writer.add_scalar("continuation/" + key, value, completed)
                    if completed % 100 == 0 or completed == 1:
                        print(arm, seed, row, flush=True)
        metadata.update(
            status="complete",
            updates_completed=completed,
            examples_processed=completed * batch_size,
            elapsed_seconds=time.monotonic() - started,
        )
        save_checkpoint(
            model,
            output / "checkpoint.pt",
            extra={
                "agent_config": asdict(model.config),
                "supervised_config": asdict(config),
                "training_reserved_seeds": sorted(set(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS + tuple(reserved_seeds))),
                "continuation": metadata,
                **reward_checkpoint_metadata(reward),
            },
        )
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        metadata.update(
            updates_completed=completed,
            examples_processed=completed * batch_size,
            elapsed_seconds=time.monotonic() - started,
        )
        write_json(output / "manifest.json", metadata)
    return output / "checkpoint.pt"
