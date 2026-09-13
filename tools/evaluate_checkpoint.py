#!/usr/bin/env python3
"""Strict checkpoint evaluation using the shared PPO evaluator and auditable seed panels."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from pylatro import load_game_data
from pylatro_agent.eval import EVAL_SEEDS_V1
from pylatro_agent.transfer import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    check_demonstration_seeds,
    check_seed_separation,
    evaluate_artifact,
    evaluate_teacher_artifact,
    load_policy,
    paired_statistics,
    sha256,
    source_hashes,
    write_json,
)
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    panel = parser.add_mutually_exclusive_group()
    panel.add_argument("--seeds", type=int, nargs="+")
    panel.add_argument("--seed-file", type=Path)
    panel.add_argument("--panel", choices=["development", "confirmation", "regression"], default="development")
    panel.add_argument("--fixed-seeds", action="store_true", help="Alias for --panel regression")
    parser.add_argument("--games", type=int, default=None, help="Optional prefix of selected panel")
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument("--mode", choices=["greedy", "sampled"], default="sampled")
    parser.add_argument("--deck", choices=["b_blue", "b_red"], default="b_blue")
    parser.add_argument("--hand-ar-mixture-eps", type=float, default=None, help="Explicit checkpoint config override")
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-device", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--eval-cpu-threads", type=int, default=0)
    parser.add_argument("--max-idle-steps", type=int, default=256)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--stake", type=int, default=1)
    parser.add_argument("--win-ante", type=int, default=8)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory; existing artifacts are never replaced"
    )
    parser.add_argument("--training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--kind", choices=["fresh_start", "demonstration_seed"], default="fresh_start")
    parser.add_argument("--paired-outcomes", type=Path, help="Reference outcomes.jsonl with identical seeds/repeats")
    parser.add_argument("--heuristic-baseline", action="store_true")
    parser.add_argument("--teacher-generation", type=Path)
    args = parser.parse_args()
    if args.seed_file:
        seeds = json.loads(args.seed_file.read_text())
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = {"development": DEVELOPMENT_SEEDS, "confirmation": CONFIRMATION_SEEDS, "regression": EVAL_SEEDS_V1}[
            "regression" if args.fixed_seeds else args.panel
        ]
    seeds = [int(s) for s in seeds]
    if args.games is not None:
        if not 0 < args.games <= len(seeds):
            parser.error("--games must be positive and no larger than the seed panel")
        seeds = seeds[: args.games]
    if args.kind == "fresh_start" and not args.training_manifest:
        parser.error("Fresh-start evaluation requires --training-manifest to verify seed separation")
    if args.kind == "demonstration_seed" and not args.training_manifest:
        parser.error("Demonstration evaluation requires its training manifest")
    separation = (
        check_seed_separation(seeds, args.training_manifest)
        if args.kind == "fresh_start"
        else check_demonstration_seeds(seeds, args.training_manifest)
    )
    if args.rollout_temperature <= 0 or args.eval_batch_size < 1 or args.eval_workers < 0:
        parser.error("Temperature/batch must be positive and workers nonnegative")
    device = torch.device(
        args.eval_device
        or args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    data = load_game_data()
    vocab = build_vocab(data)
    model, reward, payload = load_policy(args.checkpoint, vocab, device, mixture=args.hand_ar_mixture_eps)
    explicit_names = {value.split("=", 1)[0] for value in sys.argv if value.startswith("--")}
    data_hashes = {str(p): sha256(p) for p in args.training_manifest}
    for manifest in args.training_manifest:
        summary_path = manifest.parent / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            data_hashes[str(summary_path)] = sha256(summary_path)
            for shard in summary.get("shards", []):
                shard_path = manifest.parent.parent / shard["file"]
                actual = sha256(shard_path)
                if actual != shard["sha256"]:
                    raise ValueError(f"Training shard hash mismatch: {shard_path}")
                data_hashes[str(shard_path)] = actual
    metadata = {
        "configuration_migrations": payload["configuration_migrations"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_hashes": source_hashes(),
        "entrypoint": str(Path(__file__).resolve()),
        "entrypoint_sha256": sha256(__file__),
        "seed_separation": separation,
        "data_hashes": data_hashes,
        "argv": sys.argv,
        "explicit_overrides": {
            key: json.loads(json.dumps(value, default=str))
            for key, value in vars(args).items()
            if "--" + key.replace("_", "-") in explicit_names
        },
    }
    if args.heuristic_baseline:
        if not args.teacher_generation:
            parser.error("--heuristic-baseline requires --teacher-generation")
        evaluate_teacher_artifact(
            seeds, reward, args.teacher_generation, args.output_dir, workers=args.eval_workers, metadata=metadata
        )
        print((args.output_dir / "summary.json").read_text())
        return
    rows = evaluate_artifact(
        model,
        reward,
        data,
        vocab,
        seeds,
        device,
        args.output_dir,
        greedy=args.mode == "greedy",
        sampling_seed=args.sampling_seed,
        deck=args.deck,
        temperature=args.rollout_temperature,
        stake=args.stake,
        win_ante=args.win_ante,
        batch_size=args.eval_batch_size,
        workers=args.eval_workers,
        cpu_threads=args.eval_cpu_threads,
        max_idle_steps=args.max_idle_steps,
        metadata=metadata,
        kind=args.kind,
    )
    if args.paired_outcomes:
        other = [json.loads(line) for line in args.paired_outcomes.read_text().splitlines()]
        write_json(args.output_dir / "paired.json", paired_statistics(rows, other))
    print((args.output_dir / "summary.json").read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
