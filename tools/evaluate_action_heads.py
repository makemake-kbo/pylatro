#!/usr/bin/env python3
"""Matched development gameplay for the frozen-encoder sale-head experiment."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
from pathlib import Path

import torch
from action_head_experiment import DEV_SEEDS, VARIANTS, initialize

from pylatro import load_game_data
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import CHECKPOINT, atomic_json, read_json
from pylatro_agent.transfer import evaluate_artifact, load_policy, paired_statistics, sha256, source_hashes, summarize
from pylatro_agent.vocab import build_vocab


def run_panel(root, name, checkpoint, seeds, device):
    folder = root / "evaluations" / name
    digest = sha256(checkpoint)
    completed, pending = {}, []
    for sampling in (0, 1, 2, None):
        destination = folder / (f"sample{sampling}" if sampling is not None else "greedy")
        if (destination / "summary.json").exists():
            manifest = read_json(destination / "manifest.json")
            expected = dict(
                status="complete",
                checkpoint_sha256=digest,
                seeds=list(seeds),
                sampling_seed=sampling,
                device=str(device),
                temperature=1.0,
                deck="b_blue",
                batch_size=32,
                workers=4,
            )
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Completed evaluation configuration differs: {destination}")
            completed[sampling] = [
                json.loads(line) for line in (destination / "outcomes.jsonl").read_text().splitlines()
            ]
        else:
            if destination.exists():
                raise FileExistsError(f"Inspect incomplete evaluation before resume: {destination}")
            pending.append((sampling, destination))
    if pending:
        context = (
            accelerator_job(root, f"evaluate/{name}", len(seeds) * len(pending) * 1.8)
            if device.type == "mps"
            else contextlib.nullcontext()
        )
        with context:
            data = load_game_data()
            model, reward, payload = load_policy(checkpoint, build_vocab(data), device, mixture=0.5)
            for sampling, destination in pending:
                atomic_json(root / "evaluation_status.json", dict(status="running", panel=name, sampling_seed=sampling))
                completed[sampling] = evaluate_artifact(
                    model,
                    reward,
                    data,
                    build_vocab(data),
                    seeds,
                    device,
                    destination,
                    greedy=sampling is None,
                    sampling_seed=sampling,
                    metadata={
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest,
                        "source_hashes": source_hashes(),
                        "study_manifest_sha256": sha256(root / "manifest.json"),
                        "cache_manifest_sha256": sha256(root / "cache_complete.json"),
                        "configuration_migrations": payload["configuration_migrations"],
                        "explicit_overrides": {"mixture": 0.5, "temperature": 1.0, "device": str(device)},
                    },
                )
            del model
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
    sampled = [row for sampling in (0, 1, 2) for row in completed[sampling]]
    greedy = completed[None]
    for rows in (sampled, greedy):
        if any(row.get("stalled") or row.get("censored") for row in rows):
            raise ValueError("Incomplete games cannot enter gameplay comparisons")
        summarize(rows)  # validates duplicate/missing sampling repeats
    return sampled, greedy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    root, device = args.root, torch.device(args.device)
    initialize(root)
    results = {"original": run_panel(root, "screen_original", CHECKPOINT, DEV_SEEDS[:50], device)}
    for variant in VARIANTS:
        results[variant] = {}
        for seed in (0, 1):
            checkpoint = root / f"{variant}_seed{seed}" / "checkpoint.pt"
            manifest = read_json(checkpoint.parent / "complete.json")
            if sha256(checkpoint) != manifest["checkpoint_sha256"]:
                raise ValueError("Probe checkpoint changed")
            results[variant][seed] = run_panel(root, f"screen_{variant}_seed{seed}", checkpoint, DEV_SEEDS[:50], device)
    comparisons = {}
    for variant in VARIANTS:
        comparisons[variant] = {"per_replicate": {}}
        for seed in (0, 1):
            comparisons[variant]["per_replicate"][seed] = {
                mode: paired_statistics(results[variant][seed][i], results["original"][i])
                for i, mode in enumerate(("sampled", "greedy"))
            }
        for i, mode in enumerate(("sampled", "greedy")):
            candidate = [
                dict(row, sampling_seed=f"train{s}/sample{row['sampling_seed']}")
                for s in (0, 1)
                for row in results[variant][s][i]
            ]
            control = [
                dict(row, sampling_seed=f"train{s}/sample{row['sampling_seed']}")
                for s in (0, 1)
                for row in results["original"][i]
            ]
            comparisons[variant][mode] = paired_statistics(candidate, control)
    direct = {}
    for i, mode in enumerate(("sampled", "greedy")):
        arms = [
            [
                dict(row, sampling_seed=f"train{s}/sample{row['sampling_seed']}")
                for s in (0, 1)
                for row in results[variant][s][i]
            ]
            for variant in VARIANTS
        ]
        direct[mode] = paired_statistics(arms[1], arms[0])
    atomic_json(
        root / "screen_comparisons.json",
        {
            "versus_original": comparisons,
            "entity_versus_pooled": direct,
            "independent_game_seeds": 50,
            "confirmation": False,
            "note": "The original baseline is shared across training replicates; bootstrap units remain game seeds.",
        },
    )
    atomic_json(root / "evaluation_status.json", dict(status="complete", stage="development_screen"))
    print(json.dumps(direct, indent=2), flush=True)


if __name__ == "__main__":
    main()
