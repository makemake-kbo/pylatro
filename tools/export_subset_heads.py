#!/usr/bin/env python3
"""Export trained subset adapters and neutral sampling controls as strict checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from pylatro import load_game_data
from pylatro_agent.architecture_transfer import add_sale_residual, add_subset_residual
from pylatro_agent.checkpoint import save_checkpoint
from pylatro_agent.learning_bottleneck import CHECKPOINT, atomic_json, read_json
from pylatro_agent.reward import reward_checkpoint_metadata
from pylatro_agent.transfer import load_policy, sha256, source_hashes
from pylatro_agent.vocab import build_vocab


def write_checkpoint(model, path, reward, provenance):
    if path.exists():
        raise FileExistsError(f"Checkpoint export already exists: {path}")
    save_checkpoint(
        model,
        path,
        extra={
            "agent_config": asdict(model.config),
            **reward_checkpoint_metadata(reward),
            "architecture_experiment": provenance,
        },
    )
    return sha256(path)


def export(root, family, seed):
    subset_folder = root / f"subset_residual_v1_seed{seed}"
    source_path = CHECKPOINT if family == "subset" else root / f"entity_sale_residual_v1_seed{seed}" / "checkpoint.pt"
    destination = (
        root / f"{'subset_residual_v1' if family == 'subset' else 'entity_sale_subset_residual_v1'}_seed{seed}"
    )
    destination.mkdir(exist_ok=True)
    if (destination / "complete.json").exists():
        if sha256(destination / "checkpoint.pt") != read_json(destination / "complete.json")["checkpoint_sha256"]:
            raise ValueError("Completed subset checkpoint changed")
        return
    trained = read_json(subset_folder / "adapter_complete.json")
    adapter = subset_folder / "adapter_best.pt"
    if sha256(adapter) != trained["adapter_sha256"]:
        raise ValueError("Trained subset adapter changed")
    if sha256(CHECKPOINT) != trained["source_checkpoint_sha256"]:
        raise ValueError("Frozen encoder source changed")
    vocab = build_vocab(load_game_data())
    source, reward, _ = load_policy(source_path, vocab, "cpu", mixture=0.5)
    original, _, _ = load_policy(CHECKPOINT, vocab, "cpu", mixture=0.5)
    for key, value in original.state_dict().items():
        if not torch.equal(source.state_dict()[key], value):
            raise ValueError(f"Cached representations incompatible with modified source parameter: {key}")
    model, migration = add_subset_residual(source, vocab)
    model.action_grammar_head.subset_residual.load_state_dict(torch.load(adapter, weights_only=True))
    provenance = dict(
        family=family,
        seed=seed,
        source_checkpoint=str(source_path),
        source_checkpoint_sha256=sha256(source_path),
        frozen_representation_checkpoint_sha256=trained["source_checkpoint_sha256"],
        subset_training_manifest_sha256=sha256(subset_folder / "adapter_complete.json"),
        subset_adapter_sha256=sha256(adapter),
        architecture_migration=migration,
        source_hashes=source_hashes(),
        observations="unchanged v13",
        best_validation_update=trained["best_validation_update"],
    )
    provenance["checkpoint_sha256"] = write_checkpoint(model, destination / "checkpoint.pt", reward, provenance)
    atomic_json(destination / "complete.json", provenance)


def neutral_control(root, family):
    path = root / f"{family}_neutral_control.pt"
    marker = root / f"{family}_neutral_control.json"
    if marker.exists():
        if sha256(path) != read_json(marker)["checkpoint_sha256"]:
            raise ValueError("Neutral checkpoint changed")
        return
    torch.manual_seed(1729)
    vocab = build_vocab(load_game_data())
    source, reward, _ = load_policy(CHECKPOINT, vocab, "cpu", mixture=0.5)
    migrations = []
    if family == "combined":
        source, migration = add_sale_residual(source, vocab, "entity_sale_residual_v1")
        migrations.append(migration)
    model, migration = add_subset_residual(source, vocab)
    migrations.append(migration)
    provenance = dict(
        family=family,
        neutral_control=True,
        source_checkpoint_sha256=sha256(CHECKPOINT),
        architecture_migrations=migrations,
        source_hashes=source_hashes(),
        interpretation=(
            "Original policy probabilities with the same exact categorical sampler as the trained subset policy"
        ),
    )
    provenance["checkpoint_sha256"] = write_checkpoint(model, path, reward, provenance)
    atomic_json(marker, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--family", choices=("subset", "combined"), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1))
    args = parser.parse_args()
    torch.set_num_threads(4)
    neutral_control(args.root, args.family)
    for seed in (0, 1) if args.seed is None else (args.seed,):
        export(args.root, args.family, seed)


if __name__ == "__main__":
    main()
