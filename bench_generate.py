#!/usr/bin/env python3
"""Benchmark: old (Gymnasium) vs new (FastRunner) training data generation."""

from __future__ import annotations

import time
import multiprocessing
from pylatro import load_game_data
from pylatro_agent.vocab import build_vocab
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.env import BalatroEnv
from pylatro_agent.training.fast_runner import FastRunner

NUM_GAMES = 50
MIN_ANTE = 3


def bench_old():
    data = load_game_data()
    vocab = build_vocab(data)
    agent = HeuristicAgent()
    records = 0

    t0 = time.perf_counter()
    for seed in range(NUM_GAMES):
        env = BalatroEnv(seed=seed, data=data, vocab=vocab)
        obs, info = env.reset()
        done = False
        while not done:
            mask = obs["action_mask"]
            action = agent.select_action(
                env.state, env._sub_phase, mask,
                selected_cards=env._selected_cards,
                pending_action=env._pending_action,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            records += 1
            done = terminated or truncated
    elapsed = time.perf_counter() - t0
    return elapsed, records


def bench_new():
    data = load_game_data()
    agent = HeuristicAgent()
    steps = 0

    t0 = time.perf_counter()
    for seed in range(NUM_GAMES):
        runner = FastRunner(seed, data)
        while not runner.done:
            mask = runner.compute_mask()
            action = agent.select_action(
                runner.state, runner.sub_phase, mask,
                selected_cards=runner.selected_cards,
                pending_action=runner.pending_action,
            )
            runner.step(action)
            steps += 1
    elapsed = time.perf_counter() - t0
    return elapsed, steps


def main():
    print(f"Benchmark: {NUM_GAMES} games, min_ante={MIN_ANTE}")
    print()

    print("Warming up...")
    bench_old()
    bench_new()

    print("Running OLD (Gymnasium + obs every step)...")
    old_time, old_steps = bench_old()
    print(f"  {old_time:.2f}s, {old_steps} steps, {old_steps / old_time:.0f} steps/s")

    print("Running NEW (FastRunner, no obs)...")
    new_time, new_steps = bench_new()
    print(f"  {new_time:.2f}s, {new_steps} steps, {new_steps / new_time:.0f} steps/s")

    if new_time > 0:
        speedup = old_time / new_time
        print(f"\nSpeedup (game logic only, no obs): {speedup:.2f}x")


if __name__ == "__main__":
    main()
