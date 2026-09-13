#!/usr/bin/env python3
"""Development expansion and locked confirmation of refactored action heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from evaluate_action_heads import run_panel

from pylatro_agent.learning_bottleneck import CHECKPOINT, atomic_json, read_json
from pylatro_agent.transfer import paired_statistics, sha256, summarize


def checkpoints(root, family):
    if family == "sale":
        return CHECKPOINT, [root / f"entity_sale_residual_v1_seed{s}" / "checkpoint.pt" for s in (0, 1)]
    prefix = "subset_residual_v1" if family == "subset" else "entity_sale_subset_residual_v1"
    return root / f"{family}_neutral_control.pt", [root / f"{prefix}_seed{s}" / "checkpoint.pt" for s in (0, 1)]


def compare(results):
    report = {"per_replicate": {}}
    for seed in (0, 1):
        report["per_replicate"][seed] = {
            mode: paired_statistics(results[seed][index], results["control"][index])
            for index, mode in enumerate(("sampled", "greedy"))
        }
    for index, mode in enumerate(("sampled", "greedy")):
        candidate = [
            dict(row, sampling_seed=f"train{s}/sample{row['sampling_seed']}")
            for s in (0, 1)
            for row in results[s][index]
        ]
        control = [
            dict(row, sampling_seed=f"train{s}/sample{row['sampling_seed']}")
            for s in (0, 1)
            for row in results["control"][index]
        ]
        report[mode] = dict(
            paired=paired_statistics(candidate, control), candidate=summarize(candidate), control=summarize(control)
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--family", choices=("sale", "subset", "combined"), required=True)
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--parallel-arms", type=int, choices=(1, 3),
        help="Independent CPU policy processes; defaults to three for confirmation and one for development",
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    manifest = read_json(root / "manifest.json")
    seeds = manifest[f"{args.stage}_seeds"]
    count = len(seeds) if args.count is None else args.count
    if count < 1 or count > len(seeds) or (args.stage == "confirmation" and count != 400):
        parser.error("Invalid panel size; confirmation requires all 400 reserved seeds")
    seeds = seeds[:count]
    control, candidates = checkpoints(root, args.family)
    digests = {"control": sha256(control), **{f"seed{s}": sha256(p) for s, p in enumerate(candidates)}}
    if args.stage == "confirmation":
        selection = read_json(root / "candidate_selection.json")
        if selection.get("family") != args.family or selection.get("checkpoint_hashes") != digests:
            raise ValueError("Confirmation checkpoints do not match the frozen development selection")
    name = f"{args.device}_{args.stage}_{args.family}_{count}"
    result_path = root / f"{name}_comparisons.json"
    protocol_path = root / f"{name}_protocol.json"
    protocol = dict(
        stage=args.stage,
        family=args.family,
        device=args.device,
        seeds=seeds,
        checkpoint_hashes=digests,
        sampling_seeds=[0, 1, 2],
        greedy=True,
        seed_panel_manifest_sha256=sha256(root / "manifest.json"),
        baseline_shared_across_training_replicates=True,
        bootstrap_unit="game seed",
    )
    if protocol_path.exists():
        if read_json(protocol_path) != protocol:
            raise ValueError("Existing panel protocol differs")
    else:
        atomic_json(protocol_path, protocol)
    parallel = args.parallel_arms or (3 if args.stage == "confirmation" and args.device == "cpu" else 1)
    if parallel == 3:
        if args.device != "cpu":
            parser.error("Parallel arms are supported only on CPU")
        from parallel_cpu_panels import run_parallel_panels

        results = run_parallel_panels(root, name, control, candidates, seeds)
    else:
        results = {"control": run_panel(root, f"{name}_control", control, seeds, torch.device(args.device))}
        for seed, checkpoint in enumerate(candidates):
            results[seed] = run_panel(root, f"{name}_seed{seed}", checkpoint, seeds, torch.device(args.device))
    report = dict(protocol=protocol, results=compare(results))
    sampled = report["results"]["sampled"]["paired"]
    greedy = report["results"]["greedy"]["paired"]
    report["acceptance"] = dict(
        confirmation=args.stage == "confirmation",
        sampled_gain_at_least_five_points=sampled["sampled_or_greedy_delta"] >= 0.05,
        sampled_paired_ci_excludes_zero=sampled["paired_ci95"][0] > 0,
        both_replicates_positive=all(
            report["results"]["per_replicate"][s]["sampled"]["sampled_or_greedy_delta"] > 0 for s in (0, 1)
        ),
        greedy_rules_out_five_point_regression=greedy["paired_ci95"][0] >= -0.05,
        no_stalled_or_censored_games=True,
    )
    report["validated"] = all(report["acceptance"].values())
    atomic_json(result_path, report)
    atomic_json(root / "evaluation_status.json", dict(status="complete", stage=name))
    print(json.dumps({"sampled": sampled, "greedy": greedy, "acceptance": report["acceptance"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
