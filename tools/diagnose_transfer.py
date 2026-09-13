#!/usr/bin/env python3
"""Replay balanced HF games and measure every decision in 200 uniformly selected HF games."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.training.model_generate import load_records
from pylatro_agent.transfer import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    check_seed_separation,
    load_policy,
    sha256,
    source_hashes,
    write_json,
)
from pylatro_agent.transfer_diagnostics import offline_predictions, replay_game, selected_records
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--offline-games", type=int, default=200)
    parser.add_argument("--replay-games", type=int, default=20)
    parser.add_argument("--skip-offline", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data = load_game_data()
    vocab = build_vocab(data)
    device = torch.device(args.device)
    model, reward, _payload = load_policy(args.checkpoint, vocab, device)
    games = [json.loads(line) for line in (args.dataset / "metadata/games.jsonl").read_text().splitlines()]
    rng = np.random.default_rng(1729)
    offline_seeds = sorted(map(int, rng.choice(sorted(g["seed"] for g in games), args.offline_games, replace=False)))
    replay_seeds = []
    for won in (False, True):
        panel = sorted(g["seed"] for g in games if g["won"] == won)
        replay_seeds.extend(map(int, rng.choice(panel, args.replay_games // 2, replace=False)))
    manifests = [
        args.dataset / "metadata/generation.json",
        args.dataset / "metadata/games.jsonl",
        Path("docs/heuristic/pretraining_v68_seeds.json"),
    ]
    separation = check_seed_separation(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS, manifests)
    sources = source_hashes()
    generation = json.loads((args.dataset / "metadata/generation.json").read_text())
    drift = {
        key: {"generation": value, "current": sources.get(key)}
        for key, value in generation["sources"].items()
        if sources.get(key) != value
    }
    write_json(
        args.output_dir / "manifest.json",
        {
            "checkpoint": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "offline_seeds": offline_seeds,
            "replay_seeds": replay_seeds,
            "selection_rng_seed": 1729,
            "source_hashes": sources,
            "source_drift": drift,
            "seed_separation": separation,
            "data_hashes": {str(p): sha256(p) for p in sorted(args.dataset.rglob("*")) if p.is_file()},
            "kind": "training_data_diagnostics",
            "device": args.device,
        },
    )
    records = load_records(args.dataset, reward_config=reward)
    selected = selected_records(records, set(offline_seeds + replay_seeds))
    replay = []
    for seed in replay_seeds:
        result = replay_game(selected[seed], data, vocab, reward)
        replay.append(result)
        print(json.dumps(result), flush=True)
        write_json(args.output_dir / "replay.json", replay)
    if not args.skip_offline:
        offline_predictions(model, [r for seed in offline_seeds for r in selected[seed]], device, args.output_dir)
    write_json(args.output_dir / "timing.json", {"elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
