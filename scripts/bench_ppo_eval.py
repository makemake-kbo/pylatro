#!/usr/bin/env python3
"""Bounded CPU evaluation profile; fixed seeds and model, no training updates."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import statistics
import time
from pathlib import Path

import torch

import pylatro_agent.build_value as build_value
from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.ppo_evaluation import run_seed_evaluation
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--max-no-progress-steps", type=int, default=32)
    parser.add_argument("--eval-cpu-threads", type=int)
    parser.add_argument("--profile", type=Path, help="Profile the parent; use zero workers to include engine calls.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--disable-estimate-cache", action="store_true")
    parser.add_argument("--disable-score-cache", action="store_true")
    args = parser.parse_args()
    if min(args.games, args.batch_size, args.repeats, args.threads) < 1:
        parser.error("games, batch-size, repeats and threads must be positive")
    if args.eval_workers < 0:
        parser.error("eval-workers must be non-negative")
    if args.eval_workers and (args.disable_estimate_cache or args.disable_score_cache):
        parser.error("cache overrides only support local profiling")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    data = load_game_data()
    vocab = build_vocab(data)
    model = BalatroAgent(
        AgentConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, d_ff=args.d_ff or args.d_model * 2),
        vocab,
    )
    original_score = build_value._score_pass
    if args.disable_score_cache:
        build_value._score_pass = build_value._score_pass_uncached
    original_capture = BalatroEnv._capture_state_info

    def uncached(env):
        env._state_estimate_cache = None
        return original_capture(env)

    if args.disable_estimate_cache:
        BalatroEnv._capture_state_info = uncached
    durations = []
    outcomes = None
    profiler = cProfile.Profile() if args.profile else None
    try:
        for _ in range(args.repeats):
            if profiler:
                profiler.enable()
            started = time.perf_counter()
            rows = run_seed_evaluation(
                model,
                data,
                vocab,
                list(range(10000, 10000 + args.games)),
                torch.device("cpu"),
                max_no_progress_steps=args.max_no_progress_steps,
                win_ante=2,
                batch_size=args.batch_size,
                eval_workers=args.eval_workers,
                eval_cpu_threads=args.eval_cpu_threads,
            )
            durations.append(time.perf_counter() - started)
            if profiler:
                profiler.disable()
            current = [{k: v for k, v in row.items() if not k.endswith("_seconds")} for row in rows]
            if outcomes is not None and current != outcomes:
                raise RuntimeError("Fixed-seed evaluation changed between repeats")
            outcomes = current
    finally:
        BalatroEnv._capture_state_info = original_capture
        build_value._score_pass = original_score
    if profiler:
        profiler.dump_stats(str(args.profile))
        pstats.Stats(profiler).sort_stats("cumtime").print_stats(30)
    result = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "seconds": durations,
        "median_seconds": statistics.median(durations),
        "steps": sum(row["steps"] for row in outcomes),
        "outcomes": outcomes,
        "torch_version": torch.__version__,
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
