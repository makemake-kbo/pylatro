#!/usr/bin/env python3
"""Gameplay panels for outcome-only PPO, with strict checkpoint and seed provenance."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import torch

from pylatro import load_game_data
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import BASELINE, ROOT
from pylatro_agent.transfer import evaluate_artifact, load_policy, paired_statistics, sha256, source_hashes, summarize
from pylatro_agent.vocab import build_vocab


def run_evaluation(args):
    protocol = read_json(args.root / "manifest.json")
    seeds = protocol[f"{args.stage}_seeds"][: args.games]
    if len(seeds) != args.games or args.games < 1:
        raise ValueError("Invalid seed count")
    if args.stage == "confirmation":
        selection = read_json(args.root / "candidate_selection.json")
        if args.games != 400 or sha256(args.checkpoint) not in selection["checkpoint_hashes"].values():
            raise ValueError("Confirmation requires the full panel and a frozen checkpoint")
        if args.device != selection.get("confirmation_device", "cpu"):
            raise ValueError("Confirmation device differs from its frozen protocol")
    directory = args.root / "evaluations" / args.name
    data = load_game_data()
    vocab = build_vocab(data)
    device = torch.device(args.device)
    model, reward, payload = load_policy(args.checkpoint, vocab, device, mixture=0.5)
    results = {}
    for sampling in (0, 1, 2, None):
        mode = "greedy" if sampling is None else f"sample{sampling}"
        if args.only_mode is not None and args.only_mode != mode:
            continue
        destination = directory / mode
        if destination.exists():
            metadata = read_json(destination / "manifest.json")
            expected = dict(
                status="complete",
                checkpoint_sha256=sha256(args.checkpoint),
                seeds=seeds,
                sampling_seed=sampling,
                greedy=sampling is None,
                device=args.device,
                deck="b_blue",
                temperature=1.0,
                batch_size=32,
                workers=4,
            )
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("An existing panel has different settings or is incomplete")
            rows = [json.loads(line) for line in (destination / "outcomes.jsonl").read_text().splitlines()]
        else:
            rows = evaluate_artifact(
                model,
                reward,
                data,
                vocab,
                seeds,
                device,
                destination,
                greedy=sampling is None,
                sampling_seed=sampling,
                cpu_threads=4,
                metadata=dict(
                    checkpoint=str(args.checkpoint),
                    checkpoint_sha256=sha256(args.checkpoint),
                    source_hashes=source_hashes(),
                    study_manifest_sha256=sha256(args.root / "manifest.json"),
                    configuration_migrations=payload["configuration_migrations"],
                    explicit_overrides=dict(mixture=0.5, device=args.device, temperature=1.0),
                ),
            )
        if any(row.get("stalled") or row.get("censored") for row in rows):
            raise ValueError("Infrastructure failures prevent a completed gameplay panel")
        results[mode] = rows
        print(args.name, mode, sum(row["won"] for row in rows), len(rows), flush=True)
    if args.only_mode is not None:
        return
    sampled = [row for mode in ("sample0", "sample1", "sample2") for row in results[mode]]
    result = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint),
        stage=args.stage,
        device=args.device,
        sampled=summarize(sampled),
        greedy=summarize(results["greedy"]),
    )
    if args.paired:
        reference = {}
        for mode in results:
            path = args.paired / mode / "outcomes.jsonl"
            reference[mode] = [json.loads(line) for line in path.read_text().splitlines()]
        result["paired_sampled"] = paired_statistics(
            sampled, [row for mode in ("sample0", "sample1", "sample2") for row in reference[mode]]
        )
        result["paired_greedy"] = paired_statistics(results["greedy"], reference["greedy"])
    atomic_json(directory / "comparison.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, default=BASELINE)
    parser.add_argument("--name", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--parallel-modes", choices=(1, 2, 4), type=int, default=2)
    parser.add_argument("--only-mode", choices=("sample0", "sample1", "sample2", "greedy"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device == "mps":
        # Includes model initialization and every mode. The persisted ledger is
        # shared with training, so evaluation cannot run over the allowance.
        with accelerator_job(args.root, f"evaluation/{args.name}", 60 + args.games * 4 * 1.5):
            run_evaluation(args)
    else:
        if args.only_mode is None and args.parallel_modes > 1:
            # Every process keeps four CPU threads, batch 32, four engine
            # workers and its own explicit sampling seed. Parallelism changes
            # scheduling, not an individual game's policy or random stream.
            command = [sys.executable, "-u", str(Path(__file__)), "--root", str(args.root),
                       "--checkpoint", str(args.checkpoint), "--name", args.name, "--games", str(args.games),
                       "--stage", args.stage, "--device", "cpu", "--parallel-modes", "1"]

            def evaluate_mode(mode):
                subprocess.run([*command, "--only-mode", mode], check=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel_modes) as pool:
                list(pool.map(evaluate_mode, ("sample0", "sample1", "sample2", "greedy")))
        run_evaluation(args)


if __name__ == "__main__":
    main()
