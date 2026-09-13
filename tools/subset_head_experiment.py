#!/usr/bin/env python3
"""Cache frozen hand representations and fit a full-subset energy residual."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.action import ActionType
from pylatro_agent.action_grammar import ACTION_TYPE_TO_GRAMMAR_INDEX
from pylatro_agent.constants import ActionRange
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import CHECKPOINT, atomic_json, read_json
from pylatro_agent.subset_residual import (
    SubsetResidualHead,
    corrected_subset_log_probs,
    reference_hand_log_probs,
    subset_support,
)
from pylatro_agent.training.model_generate import load_records
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import load_policy, sha256, source_hashes
from pylatro_agent.vocab import build_vocab

PRIOR_DATA = Path("eval/learning_bottleneck_20260913/data")
SNAPSHOTS = (0, 100, 300, 1000, 2000)


def initialize(root):
    path = root / "subset_manifest.json"
    if path.exists():
        return
    parent = read_json(root / "manifest.json")
    atomic_json(
        path,
        {
            "status": "active",
            "source_checkpoint": str(CHECKPOINT),
            "source_checkpoint_sha256": sha256(CHECKPOINT),
            "source_hashes": source_hashes(),
            "parent_manifest_sha256": sha256(root / "manifest.json"),
            "development_seeds": parent["development_seeds"],
            "confirmation_seeds": parent["confirmation_seeds"],
            "observation_version": 13,
            "reference_mixture": 0.5,
            "reference_temperature": 1.0,
            "encoder_and_original_heads": "frozen in evaluation mode",
            "residual": (
                "64 hidden units, invariant selected-card pooling, play/discard joint energies, zero output initializer"
            ),
            "training_seeds": [0, 1],
            "updates": 2000,
            "batch_size": 320,
            "lr": 3e-4,
            "weight_decay": 0.01,
            "warmup": 100,
            "optimizer": "AdamW",
            "schedule": "linear warmup then cosine",
            "loss": (
                "unit-weight conditional teacher subset NLL on every recorded play/discard; no macro or critic loss"
            ),
            "snapshots": SNAPSHOTS,
            "selection": "minimum validation conditional NLL; earlier update breaks ties",
            "data": {split: sha256(PRIOR_DATA / split / "summary.json") for split in ("training", "validation")},
        },
    )


def hand_indices(records):
    indices, offset = [], 0
    for file in records._files:
        actions = file.read(columns=["action"])["action"].to_numpy()
        selected = (actions >= int(ActionRange.PLAY_SUBSET_START)) & (actions <= int(ActionRange.DISCARD_SUBSET_END))
        indices.extend((np.flatnonzero(selected) + offset).tolist())
        offset += len(actions)
    if offset != len(records):
        raise ValueError("Parquet action offsets do not cover all records")
    return indices


def cache(root, device_name):
    marker = root / "subset_cache_complete.json"
    if marker.exists():
        for split, digest in read_json(marker)["hashes"].items():
            if sha256(root / f"subset_{split}.pt") != digest:
                raise ValueError("Completed hand cache changed")
        return
    manifest = read_json(root / "subset_manifest.json")
    if sha256(CHECKPOINT) != manifest["source_checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed")
    context = accelerator_job(root, "cache_hand_representations", 600) if device_name == "mps" else nullcontext()
    started = time.monotonic()
    with context:
        device = torch.device(device_name)
        torch.set_num_threads(4)
        vocab = build_vocab(load_game_data())
        model, reward, payload = load_policy(CHECKPOINT, vocab, device, mixture=0.5)
        config = SupervisedConfig(reward_config=reward)
        captured = {}

        def state_hook(module, inputs):
            captured["state"] = inputs[0].detach()

        def hand_hook(module, inputs):
            captured["cards"], captured["present"] = module._gather_hand_slots(*inputs[:4])

        handles = [
            model.action_grammar_head.shop_global_head.register_forward_pre_hook(state_hook),
            model.action_grammar_head.register_forward_pre_hook(hand_hook),
        ]
        hashes, counts, bucket_counts = {}, {}, {}
        try:
            with torch.inference_mode():
                for split in ("training", "validation"):
                    destination = root / f"subset_{split}.pt"
                    if destination.exists():
                        raise FileExistsError("Partial hand cache requires inspection")
                    folder = PRIOR_DATA / split
                    if sha256(folder / "summary.json") != manifest["data"][split]:
                        raise ValueError("Hand data summary changed")
                    for name, digest in read_json(folder / "summary.json")["file_hashes"].items():
                        if sha256(folder / name) != digest:
                            raise ValueError("Hand data shard changed")
                    records = load_records(folder, reward_config=reward)
                    indices = hand_indices(records)
                    buckets = defaultdict(lambda: defaultdict(list))
                    for start in range(0, len(indices), 320):
                        rows = [copy.deepcopy(records[i]) for i in indices[start : start + 320]]
                        batch = _collate_batch(rows, device, config)
                        dist, _ = _grammar_distribution(model, batch)
                        family = (batch["actions"] >= int(ActionRange.DISCARD_SUBSET_START)).long()
                        base = torch.where(
                            family[:, None] == 0,
                            reference_hand_log_probs(dist, is_play=True),
                            reference_hand_log_probs(dist, is_play=False),
                        )
                        valid = torch.where(
                            family[:, None] == 0,
                            dist._hand_valid_subset_mask(True),
                            dist._hand_valid_subset_mask(False),
                        )
                        global_labels = batch["actions"] - torch.where(
                            family == 0, int(ActionRange.PLAY_SUBSET_START), int(ActionRange.DISCARD_SUBSET_START)
                        )
                        macro_indices = torch.where(
                            family == 0,
                            ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET],
                            ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.DISCARD_SUBSET],
                        )
                        macro = dist.action_type_probs.gather(1, macro_indices[:, None])[:, 0]
                        present = captured["present"]
                        widths = (present.bool() * torch.arange(1, present.shape[1] + 1, device=device)).amax(-1)
                        if not valid.gather(1, global_labels[:, None]).all() or (widths <= 0).any():
                            raise ValueError("Illegal teacher action or empty hand in cache")
                        ids = torch.tensor([[r["seed"], r["step"]] for r in rows], device=device)
                        for width_tensor in widths.unique():
                            width = int(width_tensor)
                            support, _ = subset_support(width, str(device))
                            selected = torch.where(widths == width)[0]
                            selected_base, selected_valid = base[selected][:, support], valid[selected][:, support]
                            labels = torch.searchsorted(support, global_labels[selected].contiguous())
                            if not torch.equal(support[labels], global_labels[selected]):
                                raise ValueError("Teacher label missing from compact support")
                            if not torch.allclose(
                                (selected_base.exp() * selected_valid).sum(-1),
                                torch.ones(len(selected), device=device),
                                atol=2e-5,
                                rtol=2e-5,
                            ):
                                raise ValueError("Reference distribution is not normalized")
                            values = dict(
                                state=captured["state"][selected],
                                cards=captured["cards"][selected, :width],
                                base_logp=selected_base,
                                valid=selected_valid,
                                labels=labels,
                                family=family[selected],
                                ids=ids[selected],
                                macro_probability=macro[selected],
                            )
                            for key, value in values.items():
                                if not torch.isfinite(value).all():
                                    raise ValueError(f"Nonfinite hand cache {key}")
                                buckets[width][key].append(value.detach().cpu().clone())
                        print("cache", split, min(start + 320, len(indices)), len(indices), flush=True)
                    data = {width: {k: torch.cat(v) for k, v in chunks.items()} for width, chunks in buckets.items()}
                    all_ids = torch.cat([values["ids"] for values in data.values()])
                    if len(all_ids) != len(indices) or len(all_ids.unique(dim=0)) != len(indices):
                        raise ValueError("Duplicate or missing hand cache decisions")
                    torch.save(data, destination)
                    hashes[split] = sha256(destination)
                    counts[split] = len(indices)
                    bucket_counts[split] = {width: len(values["labels"]) for width, values in data.items()}
        finally:
            for handle in handles:
                handle.remove()
        atomic_json(
            marker,
            dict(
                hashes=hashes,
                counts=counts,
                bucket_counts=bucket_counts,
                device=device_name,
                d_model=model.config.d_model,
                source_checkpoint_sha256=sha256(CHECKPOINT),
                source_hashes=source_hashes(),
                configuration_migrations=payload["configuration_migrations"],
                elapsed_seconds=time.monotonic() - started,
            ),
        )


def predict(head, width, batch):
    _, membership = subset_support(width)
    energies = head(batch["state"], batch["cards"], membership)
    selected = energies[torch.arange(len(energies)), batch["family"]]
    return corrected_subset_log_probs(batch["base_logp"], selected, batch["valid"])


def metrics(head, data):
    values, identities, families = [], [], []
    with torch.no_grad():
        for width, bucket in sorted(data.items()):
            for start in range(0, len(bucket["labels"]), 320):
                batch = {k: v[start : start + 320] for k, v in bucket.items()}
                logp = predict(head, width, batch)
                target = logp.gather(1, batch["labels"][:, None])[:, 0]
                values.append(
                    torch.stack(
                        (
                            -target,
                            target.exp(),
                            (logp.argmax(-1) == batch["labels"]).float(),
                            target.exp() * batch["macro_probability"],
                        ),
                        -1,
                    )
                )
                identities.append(batch["ids"])
                families.append(batch["family"])
    values, identities, families = torch.cat(values), torch.cat(identities), torch.cat(families)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite hand metrics")
    names = ("conditional_nll", "conditional_teacher_probability", "conditional_accuracy", "joint_teacher_probability")
    result = {}
    for label, mask in (
        ("all", torch.ones(len(values), dtype=torch.bool)),
        ("play", families == 0),
        ("discard", families == 1),
    ):
        selected, seeds = values[mask], identities[mask, 0]
        game_means = torch.stack([selected[seeds == seed].mean(0) for seed in seeds.unique()])
        result[label] = dict(
            decisions=len(selected),
            games=len(game_means),
            decision_weighted=dict(zip(names, selected.mean(0).tolist(), strict=True)),
            game_weighted=dict(zip(names, game_means.mean(0).tolist(), strict=True)),
        )
    return result


def train(root, seed):
    destination = root / f"subset_residual_v1_seed{seed}"
    if (destination / "adapter_complete.json").exists():
        return
    destination.mkdir(exist_ok=False)
    info = read_json(root / "subset_cache_complete.json")
    for split, digest in info["hashes"].items():
        if sha256(root / f"subset_{split}.pt") != digest:
            raise ValueError("Hand representation cache changed")
    training, validation = (
        torch.load(root / f"subset_{split}.pt", weights_only=True) for split in ("training", "validation")
    )
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    head = SubsetResidualHead(info["d_model"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    index = torch.cat(
        [
            torch.stack((torch.full((len(b["labels"]),), w), torch.arange(len(b["labels"]))), -1)
            for w, b in sorted(training.items())
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(index), generator=generator)
    cursor, best, started = 0, (float("inf"), -1), time.monotonic()
    ordering = hashlib.sha256()
    with (destination / "learning.jsonl").open("x") as stream:
        for update in range(2001):
            if update in SNAPSHOTS:
                train_metrics, val_metrics = metrics(head, training), metrics(head, validation)
                row = dict(
                    update=update,
                    training=train_metrics,
                    validation=val_metrics,
                    elapsed_seconds=time.monotonic() - started,
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                torch.save(head.state_dict(), destination / f"adapter_step{update}.pt")
                key = (val_metrics["all"]["decision_weighted"]["conditional_nll"], update)
                if key < best:
                    best = key
                    torch.save(head.state_dict(), destination / "adapter_best.pt")
                print("subset", seed, update, val_metrics["all"]["decision_weighted"], flush=True)
            if update == 2000:
                break
            if cursor + 320 > len(order):
                order = torch.cat((order[cursor:], torch.randperm(len(index), generator=generator)))
                cursor = 0
            selected = order[cursor : cursor + 320]
            cursor += 320
            ordering.update(selected.numpy().tobytes())
            rows = index[selected]
            lr = (
                3e-4
                * min(1.0, (update + 1) / 100)
                * (1.0 if update < 100 else 0.5 * (1 + math.cos(math.pi * (update - 100) / 1900)))
            )
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for width_tensor in rows[:, 0].unique():
                width = int(width_tensor)
                indices = rows[rows[:, 0] == width, 1]
                batch = {k: v[indices] for k, v in training[width].items()}
                logp = predict(head, width, batch)
                loss = -logp.gather(1, batch["labels"][:, None]).sum() / 320
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite subset loss")
                loss.backward()
                loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if (update + 1) % 100 == 0:
                print("update", seed, update + 1, loss_value, time.monotonic() - started, flush=True)
    atomic_json(
        destination / "adapter_complete.json",
        dict(
            training_seed=seed,
            best_validation_update=best[1],
            best_validation_nll=best[0],
            updates=2000,
            examples_processed=640000,
            ordering_sha256=ordering.hexdigest(),
            cache_hashes=info["hashes"],
            source_checkpoint_sha256=info["source_checkpoint_sha256"],
            source_hashes=source_hashes(),
            protocol_sha256=sha256(root / "subset_manifest.json"),
            elapsed_cpu_seconds=time.monotonic() - started,
            accelerator_seconds=0,
            adapter_sha256=sha256(destination / "adapter_best.pt"),
            status="trained adapter; must export a strict checkpoint and verify gameplay",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("init", "cache", "train"))
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--seed", type=int, choices=(0, 1))
    args = parser.parse_args()
    initialize(args.root)
    if args.task == "cache":
        cache(args.root, args.device)
    elif args.task == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        train(args.root, args.seed)


if __name__ == "__main__":
    main()
