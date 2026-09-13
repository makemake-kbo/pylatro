#!/usr/bin/env python3
"""Frozen-representation sale-head trials, followed by shared gameplay evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pylatro import load_game_data
from pylatro_agent.architecture_transfer import add_sale_residual
from pylatro_agent.checkpoint import save_checkpoint
from pylatro_agent.constants import JOKER_START, MAX_JOKER_SLOTS, ActionRange
from pylatro_agent.entity_heads import SaleResidualHead
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import CHECKPOINT, atomic_json, read_json
from pylatro_agent.reward import reward_checkpoint_metadata
from pylatro_agent.training.model_generate import load_records
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import check_seed_separation, load_policy, sha256, source_hashes, write_json
from pylatro_agent.vocab import build_vocab

PRIOR = Path("eval/learning_bottleneck_20260913")
DEV_SEEDS = tuple(range(160000, 160200))
CONFIRM_SEEDS = tuple(range(170000, 170400))
VARIANTS = ("pooled_sale_residual_v1", "entity_sale_residual_v1")
SNAPSHOTS = (0, 100, 300, 1000, 2000)


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        return
    prior = read_json(PRIOR / "manifest.json")
    manifests = [Path(item["path"]) for item in prior["seed_separation"]]
    for split in ("training", "validation"):
        summary = read_json(PRIOR / "data" / split / "summary.json")
        path = root / f"{split}_seeds.json"
        seeds = [int(Path(name).stem.split("_")[-1]) for name in summary["file_hashes"]]
        write_json(path, {"seeds": sorted(seeds), "summary_sha256": sha256(PRIOR / "data" / split / "summary.json")})
        manifests.append(path)
    separation = check_seed_separation(DEV_SEEDS + CONFIRM_SEEDS, manifests)
    reserved = {s for key, seeds in prior["panels"].items() if key != "training" for s in seeds}
    if reserved.intersection(DEV_SEEDS + CONFIRM_SEEDS):
        raise ValueError("New gameplay panels overlap the previous study")
    prior_compute = read_json(PRIOR / "compute.json")
    prior_used = sum(j["elapsed_seconds"] for j in prior_compute["jobs"])
    if any(j["status"] == "running" for j in prior_compute["jobs"]):
        raise RuntimeError("Previous accelerator study is still running")
    atomic_json(
        root / "compute.json",
        {
            "limit_seconds": max(0, 32400 - prior_used),
            "elapsed_seconds": 0,
            "jobs": [],
            "authorization": "Remaining portion of the prior nine-hour limit; additional budget question pending",
            "prior_compute_sha256": sha256(PRIOR / "compute.json"),
        },
    )
    atomic_json(
        root / "manifest.json",
        {
            "objective": "Improve sampled gameplay through verified changes to action architecture and learning",
            "status": "active",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "created_unix": time.time(),
            "source_hashes": source_hashes(),
            "seed_separation": separation,
            "development_seeds": DEV_SEEDS,
            "confirmation_seeds": CONFIRM_SEEDS,
            "probe": {
                "variants": VARIANTS,
                "training_seeds": [0, 1],
                "updates": 2000,
                "batch_size": 320,
                "optimizer": "AdamW",
                "lr": 3e-4,
                "weight_decay": 0.01,
                "warmup": 100,
                "objective": "Unit-weight conditional teacher-Joker NLL on all recorded Joker sales",
                "encoder_and_original_heads": "frozen in evaluation mode",
                "dropout": 0,
                "critic_and_macro_losses": "none",
                "snapshots": SNAPSHOTS,
                "snapshot_selection": "minimum validation conditional NLL; earlier update breaks ties",
                "interpretation": "Representation and action-head probe; offline gains do not validate gameplay",
            },
            "references": ["https://arxiv.org/abs/1506.03134", "https://github.com/google-deepmind/alphastar"],
        },
    )


def _sale_indices(records):
    result, offset = [], 0
    for file in records._files:
        actions = file.read(columns=["action"])["action"].to_numpy()
        selected = (actions >= int(ActionRange.SHOP_SELL_JOKER_START)) & (
            actions <= int(ActionRange.SHOP_SELL_JOKER_END)
        )
        result.extend((np.flatnonzero(selected) + offset).tolist())
        offset += len(actions)
    if offset != len(records):
        raise ValueError("Record offsets do not match the Parquet dataset")
    return result


def cache_representations(root):
    marker = root / "cache_complete.json"
    if marker.exists():
        for split, digest in read_json(marker)["hashes"].items():
            if sha256(root / f"{split}.pt") != digest:
                raise ValueError("Completed representation cache changed")
        return
    with accelerator_job(root, "cache_sale_representations", 240):
        device = torch.device("mps")
        vocab = build_vocab(load_game_data())
        model, reward, payload = load_policy(CHECKPOINT, vocab, device, mixture=0.5)
        config = SupervisedConfig(reward_config=reward)
        captured = {}

        def state_hook(module, inputs):
            captured["state"] = inputs[0].detach().cpu()

        def token_hook(module, inputs):
            captured["jokers"] = inputs[0][:, JOKER_START : JOKER_START + MAX_JOKER_SLOTS].detach().cpu()

        handles = [
            model.action_grammar_head.shop_global_head.register_forward_pre_hook(state_hook),
            model.action_grammar_head.register_forward_pre_hook(token_hook),
        ]
        hashes, counts = {}, {}
        try:
            with torch.inference_mode():
                for split in ("training", "validation"):
                    folder = PRIOR / "data" / split
                    for name, expected in read_json(folder / "summary.json")["file_hashes"].items():
                        if sha256(folder / name) != expected:
                            raise ValueError("Current-rule data changed")
                    records = load_records(folder, reward_config=reward)
                    indices = _sale_indices(records)
                    chunks = defaultdict(list)
                    for start in range(0, len(indices), 320):
                        rows = [copy.deepcopy(records[i]) for i in indices[start : start + 320]]
                        batch = _collate_batch(rows, device, config)
                        dist, _ = _grammar_distribution(model, batch)
                        lo, hi = int(ActionRange.SHOP_SELL_JOKER_START), int(ActionRange.SHOP_SELL_JOKER_END)
                        logits = dist.output.shop_sell_joker_logits.detach().cpu()
                        mask = batch["action_mask"][:, lo : hi + 1].bool().cpu()
                        labels = batch["actions"].cpu() - lo
                        if not mask.gather(1, labels[:, None]).all():
                            raise ValueError("Illegal cached teacher sale")
                        conditional = F.log_softmax(logits.masked_fill(~mask, -1e8), -1).gather(1, labels[:, None])[
                            :, 0
                        ]
                        macro = (dist.log_prob(batch["actions"]).cpu() - conditional).exp()
                        values = dict(
                            captured,
                            base_logits=logits,
                            mask=mask,
                            labels=labels,
                            macro_probability=macro,
                            ids=torch.tensor([[r["seed"], r["step"]] for r in rows]),
                            phase=batch["scalars"][:, 7].cpu().long(),
                        )
                        for key, value in values.items():
                            if not torch.isfinite(value).all():
                                raise ValueError(f"Nonfinite cached {key}")
                            chunks[key].append(value.clone())
                        print("cache", split, min(start + 320, len(indices)), len(indices), flush=True)
                    tensors = {key: torch.cat(value) for key, value in chunks.items()}
                    if len(torch.unique(tensors["ids"], dim=0)) != len(indices):
                        raise ValueError("Duplicate cached decision IDs")
                    destination = root / f"{split}.pt"
                    if destination.exists():
                        raise FileExistsError("Partial cache requires explicit inspection")
                    torch.save(tensors, destination)
                    hashes[split] = sha256(destination)
                    counts[split] = len(indices)
        finally:
            for handle in handles:
                handle.remove()
        atomic_json(
            marker,
            {
                "hashes": hashes,
                "counts": counts,
                "d_model": model.config.d_model,
                "checkpoint_sha256": sha256(CHECKPOINT),
                "source_hashes": source_hashes(),
                "configuration_migrations": payload["configuration_migrations"],
            },
        )


def metrics(head, data):
    rows = []
    with torch.no_grad():
        for start in range(0, len(data["labels"]), 320):
            batch = {key: value[start : start + 320] for key, value in data.items()}
            logits = batch["base_logits"] + head(batch["state"], batch["jokers"])
            logp = F.log_softmax(logits.masked_fill(~batch["mask"], -1e8), -1)
            target = logp.gather(1, batch["labels"][:, None])[:, 0]
            rows.append(
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
    values = torch.cat(rows)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite validation metrics")
    names = ("conditional_nll", "conditional_teacher_probability", "conditional_accuracy", "joint_teacher_probability")
    seeds = data["ids"][:, 0]
    game_means = torch.stack([values[seeds == seed].mean(0) for seed in seeds.unique()])
    return {
        "decisions": len(values),
        "games": len(game_means),
        "decision_weighted": dict(zip(names, values.mean(0).tolist(), strict=True)),
        "game_weighted": dict(zip(names, game_means.mean(0).tolist(), strict=True)),
    }


def train_probe(root, variant, seed):
    destination = root / f"{variant}_seed{seed}"
    if (destination / "complete.json").exists():
        return
    destination.mkdir(exist_ok=False)
    cache = read_json(root / "cache_complete.json")
    for split, digest in cache["hashes"].items():
        if sha256(root / f"{split}.pt") != digest:
            raise ValueError("Cache changed before training")
    train, validation = (torch.load(root / f"{split}.pt", weights_only=True) for split in ("training", "validation"))
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    head = SaleResidualHead(cache["d_model"], entity_conditioned=variant == "entity_sale_residual_v1")
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train["labels"]), generator=generator)
    cursor, best, started = 0, (float("inf"), -1), time.monotonic()
    ordering = hashlib.sha256()
    with (destination / "learning.jsonl").open("x") as stream:
        for update in range(2001):
            if update in SNAPSHOTS:
                train_metrics, val_metrics = metrics(head, train), metrics(head, validation)
                row = dict(
                    update=update,
                    training=train_metrics,
                    validation=val_metrics,
                    elapsed_seconds=time.monotonic() - started,
                )
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                torch.save(head.state_dict(), destination / f"adapter_step{update}.pt")
                key = (val_metrics["decision_weighted"]["conditional_nll"], update)
                if key < best:
                    best = key
                    torch.save(head.state_dict(), destination / "adapter_best.pt")
                print(variant, seed, update, val_metrics["decision_weighted"], flush=True)
            if update == 2000:
                break
            # Epoch boundaries preserve every example; concatenate the next
            # permutation when a physical batch crosses an epoch boundary.
            if cursor + 320 > len(order):
                order = torch.cat((order[cursor:], torch.randperm(len(train["labels"]), generator=generator)))
                cursor = 0
            indices = order[cursor : cursor + 320]
            cursor += 320
            ordering.update(indices.numpy().tobytes())
            lr = (
                3e-4
                * min(1.0, (update + 1) / 100)
                * (1.0 if update < 100 else 0.5 * (1 + math.cos(math.pi * (update - 100) / 1900)))
            )
            optimizer.param_groups[0]["lr"] = lr
            logits = train["base_logits"][indices] + head(train["state"][indices], train["jokers"][indices])
            loss = F.cross_entropy(logits.masked_fill(~train["mask"][indices], -1e8), train["labels"][indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite conditional loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
    vocab = build_vocab(load_game_data())
    source, reward, _ = load_policy(CHECKPOINT, vocab, "cpu", mixture=0.5)
    model, migration = add_sale_residual(source, vocab, variant)
    model.action_grammar_head.sale_residual.load_state_dict(
        torch.load(destination / "adapter_best.pt", weights_only=True)
    )
    provenance = {
        "variant": variant,
        "seed": seed,
        "best_validation_update": best[1],
        "updates": 2000,
        "examples_processed": 2000 * 320,
        "ordering_sha256": ordering.hexdigest(),
        "cache_hashes": cache["hashes"],
        "source_checkpoint_sha256": sha256(CHECKPOINT),
        "source_hashes": source_hashes(),
        "architecture_migration": migration,
        "elapsed_cpu_seconds": time.monotonic() - started,
        "accelerator_seconds": 0,
        "objective": read_json(root / "manifest.json")["probe"],
    }
    checkpoint = save_checkpoint(
        model,
        destination / "checkpoint.pt",
        extra={
            "agent_config": asdict(model.config),
            **reward_checkpoint_metadata(reward),
            "head_probe": provenance,
            "training_reserved_seeds": list(DEV_SEEDS + CONFIRM_SEEDS),
        },
    )
    provenance["checkpoint_sha256"] = sha256(checkpoint)
    atomic_json(destination / "complete.json", provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("init", "cache", "train"))
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=(0, 1))
    args = parser.parse_args()
    initialize(args.root)
    if args.task == "cache":
        cache_representations(args.root)
    elif args.task == "train":
        if args.variant is None or args.seed is None:
            parser.error("train requires --variant and --seed")
        train_probe(args.root, args.variant, args.seed)


if __name__ == "__main__":
    main()
