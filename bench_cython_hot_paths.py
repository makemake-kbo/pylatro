#!/usr/bin/env python3
"""Benchmark hot paths in the pretraining pipeline for Cython optimization.

Profiles game creation and filtering to identify which uncompiled functions
would benefit most from Cython.  Already-compiled engine modules (scoring,
flow, rng, instances, runtime, shop, pool, blind, consumables, _helpers)
are included as baseline reference points.

Usage:
    uv run python bench_cython_hot_paths.py
    uv run python bench_cython_hot_paths.py --json results.json
    uv run python bench_cython_hot_paths.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from statistics import mean, stdev

import numpy as np

from pylatro import (
    create_run_state,
    evaluate_poker_hand,
    load_game_data,
    select_blind,
    start_blind,
)
from pylatro.rng import PseudorandomState, pseudohash
from pylatro.scoring import (
    _evaluate_joker,
    get_poker_hand_info,
    score_hand,
)
from pylatro_agent.constants import NUM_ACTIONS, ActionRange, SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_generate import (
    _capture_info,
    _discounted_returns,
    _info_signature,
    _run_game_single_pass,
)
from pylatro_agent.training.fast_runner import (
    FastRunner,
    _blind_target,
)
from pylatro_agent.vocab import build_vocab


@dataclass
class BenchResult:
    name: str
    iterations: int
    total_sec: float
    mean_us: float
    stddev_us: float
    ops_per_sec: float


@dataclass
class BenchGroup:
    label: str
    results: list[BenchResult] = field(default_factory=list)


def _bench(fn, iterations: int, warmup: int = 100) -> BenchResult:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    total = sum(times)
    avg = mean(times)
    sd = stdev(times) if len(times) > 1 else 0.0
    return BenchResult(
        name="",
        iterations=iterations,
        total_sec=total,
        mean_us=avg * 1e6,
        stddev_us=sd * 1e6,
        ops_per_sec=1.0 / avg if avg > 0 else 0,
    )


def _print_result(r: BenchResult) -> None:
    print(f"  {r.mean_us:10.1f} +/- {r.stddev_us:7.1f} us   {r.ops_per_sec:12,.0f} ops/s   {r.name}")


def _print_group(g: BenchGroup) -> None:
    print(f"\n-- {g.label} --")
    for r in g.results:
        _print_result(r)


def _make_data():
    return load_game_data()


def _make_state(data, seed: str = "bench42"):
    state = create_run_state(seed, 1, "b_red", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    return state


def _make_runner_at_choose_action(data, seed: int = 42):
    runner = FastRunner(seed, data)
    runner.step(ActionRange.BLIND_PLAY)
    return runner


def _make_runner_at_blind_select(data, seed: int = 42):
    return FastRunner(seed, data)


def _advance_to_shop(data, seed: int = 42):
    runner = FastRunner(seed, data)
    agent = HeuristicAgent()
    for _ in range(500):
        if runner.done:
            break
        if runner.sub_phase == SubPhase.SHOP:
            return runner
        mask = runner.compute_mask()
        action = agent.select_action(runner.state, runner.sub_phase, mask)
        runner.step(action)
    return runner


def _play_one_game(agent, data, seed: int) -> int:
    runner = FastRunner(seed, data)
    steps = 0
    while not runner.done:
        mask = runner.compute_mask()
        action = agent.select_action(runner.state, runner.sub_phase, mask)
        runner.step(action)
        steps += 1
    return steps


# ── A. RNG  [compiled baseline] ────────────────────────────────────────────


def bench_rng(data) -> BenchGroup:
    g = BenchGroup("A. RNG  [compiled baseline]")
    rng = PseudorandomState("bench_seed_12345")

    r = _bench(lambda: rng.pseudoseed("test_key"), 5000)
    r.name = "pseudoseed()"
    g.results.append(r)

    r = _bench(lambda: rng.pseudorandom("bench_rnd"), 5000)
    r.name = "pseudorandom(str)"
    g.results.append(r)

    r = _bench(lambda: rng.pseudorandom(0.12345), 5000)
    r.name = "pseudorandom(float)"
    g.results.append(r)

    r = _bench(lambda: pseudohash("some_key_string"), 5000)
    r.name = "pseudohash()"
    g.results.append(r)

    return g


# ── B. Scoring  [compiled baseline] ────────────────────────────────────────


def bench_scoring(data) -> BenchGroup:
    g = BenchGroup("B. Scoring  [compiled baseline]")
    state = _make_state(data)
    hand = list(state.hand_cards[:5])

    r = _bench(lambda: evaluate_poker_hand(state, hand), 3000)
    r.name = "evaluate_poker_hand()  [5 cards, no jokers]"
    g.results.append(r)

    r = _bench(lambda: get_poker_hand_info(state, hand), 3000)
    r.name = "get_poker_hand_info()  [5 cards]"
    g.results.append(r)

    r = _bench(lambda: score_hand(state, hand), 2000)
    r.name = "score_hand()  [5 cards, no jokers]"
    g.results.append(r)

    if state.jokers:
        joker = state.jokers[0]
        r = _bench(
            lambda: _evaluate_joker(
                state,
                joker,
                index=0,
                phase="main",
                full_hand=hand,
                scoring_hand=hand,
                held_hand=[],
                scoring_name="High Card",
                poker_hands={},
            ),
            5000,
        )
        r.name = "_evaluate_joker()  [main phase, 1 joker]"
        g.results.append(r)

    return g


# ── C. FastRunner.compute_mask() ────────────────────────────────────────────


def bench_compute_mask(data) -> BenchGroup:
    g = BenchGroup("C. FastRunner.compute_mask()")

    runner_blind = _make_runner_at_blind_select(data)
    r = _bench(lambda: runner_blind.compute_mask(), 5000)
    r.name = "compute_mask()  [BLIND_SELECT]"
    g.results.append(r)

    runner_action = _make_runner_at_choose_action(data)
    r = _bench(lambda: runner_action.compute_mask(), 5000)
    r.name = "compute_mask()  [CHOOSE_ACTION]"
    g.results.append(r)

    runner_shop = _advance_to_shop(data)
    if runner_shop.sub_phase == SubPhase.SHOP:
        r = _bench(lambda: runner_shop.compute_mask(), 5000)
        r.name = "compute_mask()  [SHOP]"
        g.results.append(r)

    return g


# ── D. FastRunner internals ─────────────────────────────────────────────────


def bench_fast_runner_internals(data) -> BenchGroup:
    g = BenchGroup("D. FastRunner internals")

    runner = _make_runner_at_choose_action(data)
    state = runner.state

    r = _bench(runner._progress_signature, 10000)
    r.name = "_progress_signature()"
    g.results.append(r)

    r = _bench(lambda: _blind_target(state), 10000)
    r.name = "_blind_target()"
    g.results.append(r)

    r = _bench(lambda: _capture_info(runner), 5000)
    r.name = "_capture_info()"
    g.results.append(r)

    info = _capture_info(runner)
    r = _bench(lambda: _info_signature(info), 10000)
    r.name = "_info_signature()"
    g.results.append(r)

    mask = runner.compute_mask()
    r = _bench(lambda: [i for i in range(len(mask)) if mask[i] == 1], 10000)
    r.name = "mask valid indices  [list comprehension scan]"
    g.results.append(r)

    return g


# ── E. HeuristicAgent ───────────────────────────────────────────────────────


def bench_heuristic(data) -> BenchGroup:
    g = BenchGroup("E. HeuristicAgent")

    agent = HeuristicAgent()
    runner_blind = _make_runner_at_blind_select(data)
    mask_blind = runner_blind.compute_mask()

    r = _bench(
        lambda: agent.select_action(runner_blind.state, SubPhase.BLIND_SELECT, mask_blind),
        5000,
    )
    r.name = "select_action()  [BLIND_SELECT]"
    g.results.append(r)

    runner_action = _make_runner_at_choose_action(data)
    mask_action = runner_action.compute_mask()

    r = _bench(
        lambda: agent.select_action(runner_action.state, SubPhase.CHOOSE_ACTION, mask_action),
        5000,
    )
    r.name = "select_action()  [CHOOSE_ACTION]"
    g.results.append(r)

    hand = list(runner_action.state.hand_cards)

    r = _bench(lambda: agent._find_best_hand(runner_action.state, hand), 2000)
    r.name = "_find_best_hand()  [full hand, typical]"
    g.results.append(r)

    r = _bench(lambda: agent._find_worst_cards(runner_action.state, hand, 3), 3000)
    r.name = "_find_worst_cards()  [3 discards]"
    g.results.append(r)

    r = _bench(lambda: agent._quick_hand_quality(runner_action.state, hand), 2000)
    r.name = "_quick_hand_quality()"
    g.results.append(r)

    agent2 = HeuristicAgent()
    agent2._cached_best_hand(runner_action.state, hand)
    r = _bench(
        lambda: agent2._cached_best_hand(runner_action.state, hand),
        5000,
    )
    r.name = "_cached_best_hand()  [warm cache]"
    g.results.append(r)

    r = _bench(
        lambda: evaluate_poker_hand(runner_action.state, hand[:5]),
        3000,
    )
    r.name = "evaluate_poker_hand()  [heuristic path, 5 cards]"
    g.results.append(r)

    return g


# ── F. Tokenizer ────────────────────────────────────────────────────────────


def bench_tokenizer(data) -> BenchGroup:
    g = BenchGroup("F. Tokenizer")
    vocab = build_vocab(data)
    tokenizer = Tokenizer(vocab=vocab)
    mask = np.ones(NUM_ACTIONS, dtype=np.int8)

    state = _make_state(data)
    r = _bench(
        lambda: tokenizer.tokenize(state, SubPhase.CHOOSE_ACTION, action_mask=mask),
        1000,
    )
    r.name = "tokenize()  [CHOOSE_ACTION]"
    g.results.append(r)

    runner_shop = _advance_to_shop(data)
    if runner_shop.sub_phase == SubPhase.SHOP:
        r = _bench(
            lambda: tokenizer.tokenize(runner_shop.state, SubPhase.SHOP, action_mask=mask),
            1000,
        )
        r.name = "tokenize()  [SHOP]"
        g.results.append(r)

    runner_blind = _make_runner_at_blind_select(data)
    r = _bench(
        lambda: tokenizer.tokenize(runner_blind.state, SubPhase.BLIND_SELECT, action_mask=mask),
        1000,
    )
    r.name = "tokenize()  [BLIND_SELECT]"
    g.results.append(r)

    return g


# ── G. fast_generate helpers ────────────────────────────────────────────────


def bench_fast_generate_helpers(data) -> BenchGroup:
    g = BenchGroup("G. fast_generate helpers")

    rewards = [0.1] * 50 + [1.0] * 10
    r = _bench(lambda: _discounted_returns(rewards, 0.995), 5000)
    r.name = "_discounted_returns()  [60 rewards]"
    g.results.append(r)

    rewards_short = [0.1] * 5
    r = _bench(lambda: _discounted_returns(rewards_short, 0.995), 10000)
    r.name = "_discounted_returns()  [5 rewards]"
    g.results.append(r)

    return g


# ── H. Run creation ────────────────────────────────────────────────────────


def bench_run_creation(data) -> BenchGroup:
    g = BenchGroup("H. Run creation")

    r = _bench(lambda: create_run_state("bench_create", 1, "b_red", data=data), 500)
    r.name = "create_run_state()"
    g.results.append(r)

    def _full_init():
        s = create_run_state("bench_init", 1, "b_red", data=data)
        select_blind(s, "Small")
        start_blind(s, "Small")

    r = _bench(_full_init, 500)
    r.name = "create_run_state() + select_blind + start_blind"
    g.results.append(r)

    r = _bench(lambda: FastRunner(42, data), 500)
    r.name = "FastRunner.__init__()"
    g.results.append(r)

    return g


# ── I. Full game  [no obs -- filter pass] ──────────────────────────────────


def bench_full_game_no_obs(data) -> BenchGroup:
    g = BenchGroup("I. Full game  [no obs -- filter pass]")

    agent = HeuristicAgent()
    steps_per_game: list[int] = []

    def _bench_game():
        steps = _play_one_game(agent, data, len(steps_per_game))
        steps_per_game.append(steps)

    r = _bench(_bench_game, 50, warmup=5)
    avg_steps = mean(steps_per_game) if steps_per_game else 0
    r.name = f"full game  [no obs, ~{avg_steps:.0f} steps/game]"
    g.results.append(r)

    per_step = r.mean_us / max(avg_steps, 1)
    sr = BenchResult(
        name="  per-step overhead  [no obs]",
        iterations=r.iterations,
        total_sec=0,
        mean_us=per_step,
        stddev_us=0,
        ops_per_sec=1e6 / per_step if per_step > 0 else 0,
    )
    g.results.append(sr)

    return g


def bench_full_game_with_obs(data) -> BenchGroup:
    g = BenchGroup("J. Full game  [with obs -- training pass]")
    vocab = build_vocab(data)
    tokenizer = Tokenizer(vocab=vocab)
    agent = HeuristicAgent()
    steps_per_game: list[int] = []

    def _bench_game():
        seed = len(steps_per_game)
        records, _, _ = _run_game_single_pass(seed, data, tokenizer, agent, 0.995)
        steps_per_game.append(len(records))

    r = _bench(_bench_game, 20, warmup=2)
    avg_steps = mean(steps_per_game) if steps_per_game else 0
    r.name = f"full game  [with obs, ~{avg_steps:.0f} steps/game]"
    g.results.append(r)

    per_step = r.mean_us / max(avg_steps, 1)
    sr = BenchResult(
        name="  per-step overhead  [with obs]",
        iterations=r.iterations,
        total_sec=0,
        mean_us=per_step,
        stddev_us=0,
        ops_per_sec=1e6 / per_step if per_step > 0 else 0,
    )
    g.results.append(sr)

    return g


# ── K. Filter throughput  [min_ante gate] ──────────────────────────────────


def _bench_filter_pass(agent, data, min_ante, iterations, warmup):
    games_run = 0
    games_passed = 0
    total_steps = 0

    def _run():
        nonlocal games_run, games_passed, total_steps
        runner = FastRunner(games_run, data)
        steps = 0
        while not runner.done:
            mask = runner.compute_mask()
            action = agent.select_action(runner.state, runner.sub_phase, mask)
            runner.step(action)
            steps += 1
        games_run += 1
        total_steps += steps
        if runner.max_ante >= min_ante or runner.won:
            games_passed += 1

    r = _bench(_run, iterations, warmup)
    return r, games_run, games_passed, total_steps


def bench_filter_throughput(data) -> BenchGroup:
    g = BenchGroup("K. Filter throughput  [min_ante gate]")
    agent = HeuristicAgent()

    for min_ante in (1, 3, 5, 7):
        r, total, passed, total_steps = _bench_filter_pass(
            agent,
            data,
            min_ante,
            50,
            3,
        )
        pct = passed / total * 100 if total else 0
        avg_steps = total_steps // max(total, 1)
        r.name = f"filter pass  min_ante={min_ante}  [{pct:.0f}% pass, {avg_steps} steps/game]"
        g.results.append(r)

    return g


# ── L. Per-step breakdown ──────────────────────────────────────────────────


def bench_per_step_breakdown(data) -> BenchGroup:
    g = BenchGroup("L. Per-step breakdown  [estimated from micro-benchmarks]")

    agent = HeuristicAgent()
    runner = _make_runner_at_choose_action(data)
    vocab = build_vocab(data)
    tokenizer = Tokenizer(vocab=vocab)
    state = runner.state
    mask_arr = runner.compute_mask()

    r = _bench(lambda: runner.compute_mask(), 5000)
    r.name = "1. compute_mask()"
    g.results.append(r)

    r = _bench(
        lambda: agent.select_action(state, SubPhase.CHOOSE_ACTION, mask_arr),
        5000,
    )
    r.name = "2. select_action()  [CHOOSE_ACTION]"
    g.results.append(r)

    r = _bench(lambda: runner.step(ActionRange.BLIND_PLAY), 5000)
    r.name = "3. step()  [BLIND_PLAY]"
    g.results.append(r)

    r = _bench(
        lambda: tokenizer.tokenize(state, SubPhase.CHOOSE_ACTION, action_mask=mask_arr),
        2000,
    )
    r.name = "4. tokenize()  [CHOOSE_ACTION]"
    g.results.append(r)

    info = _capture_info(runner)
    r = _bench(lambda: _info_signature(info), 5000)
    r.name = "5. _info_signature()"
    g.results.append(r)

    r = _bench(lambda: _capture_info(runner), 5000)
    r.name = "6. _capture_info()"
    g.results.append(r)

    r = _bench(runner._progress_signature, 5000)
    r.name = "7. _progress_signature()"
    g.results.append(r)

    return g


# ── serialization ──────────────────────────────────────────────────────────


def _results_to_dict(groups: list[BenchGroup]) -> dict:
    return {
        "groups": [
            {
                "label": g.label,
                "results": [
                    {
                        "name": r.name,
                        "iterations": r.iterations,
                        "total_sec": round(r.total_sec, 6),
                        "mean_us": round(r.mean_us, 3),
                        "stddev_us": round(r.stddev_us, 3),
                        "ops_per_sec": round(r.ops_per_sec, 1),
                    }
                    for r in g.results
                ],
            }
            for g in groups
        ],
    }


def _compare(before_path: str, after_path: str) -> None:
    with open(before_path) as f:
        before = json.load(f)
    with open(after_path) as f:
        after = json.load(f)

    before_by_name: dict[str, dict] = {}
    for g in before.get("groups", []):
        for r in g.get("results", []):
            before_by_name[r["name"]] = r

    print()
    print("=" * 78)
    print("  CYTHON SPEEDUP REPORT")
    print("=" * 78)
    print()
    print(f"  {'BEFORE (us)':>14s}  {'AFTER (us)':>14s}  {'SPEEDUP':>10s}  FUNCTION")
    print(f"  {'-' * 14}  {'-' * 14}  {'-' * 10}  {'-' * 40}")

    total_before = 0.0
    total_after = 0.0

    for g in after.get("groups", []):
        for r in g.get("results", []):
            name = r["name"]
            after_us = r["mean_us"]
            b = before_by_name.get(name)
            if not b:
                continue
            before_us = b["mean_us"]
            if before_us <= 0:
                continue
            speedup = before_us / after_us
            marker = "  + " if speedup > 1.05 else "  - " if speedup < 0.95 else "  = "
            print(f"{marker}{before_us:14.1f}  {after_us:14.1f}  {speedup:9.2f}x  {name}")
            total_before += before_us
            total_after += after_us

    if total_after > 0:
        overall = total_before / total_after
        print(f"\n  Overall weighted speedup: {overall:.2f}x")


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pretraining pipeline hot paths")
    parser.add_argument("--json", type=str, help="Write results to JSON file")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two result files",
    )
    args = parser.parse_args()

    if args.compare:
        _compare(args.compare[0], args.compare[1])
        return

    print("Loading game data...")
    data = _make_data()
    print("Running benchmarks...\n")

    groups: list[BenchGroup] = [
        bench_rng(data),
        bench_scoring(data),
        bench_compute_mask(data),
        bench_fast_runner_internals(data),
        bench_heuristic(data),
        bench_tokenizer(data),
        bench_fast_generate_helpers(data),
        bench_run_creation(data),
        bench_per_step_breakdown(data),
        bench_full_game_no_obs(data),
        bench_full_game_with_obs(data),
        bench_filter_throughput(data),
    ]

    for g in groups:
        _print_group(g)

    print("\n-- Summary --")
    print("\nAgent-side hot paths under test:")
    print("  - pylatro_agent/training/fast_runner.py  (compute_mask, step, _execute)")
    print("  - pylatro_agent/heuristic.py             (select_action, _find_best_hand)")
    print("  - pylatro_agent/tokenizer.py              (tokenize)")
    print("  - pylatro_agent/training/fast_generate.py (_capture_info, _info_signature)")
    print("\nCore engine baselines:")
    print("  - pylatro/rng.py, scoring.py, flow.py, instances.py, runtime.py")
    print("  - pylatro/shop.py, pool.py, blind.py, consumables.py, _helpers.py")

    if args.json:
        d = _results_to_dict(groups)
        with open(args.json, "w") as f:
            json.dump(d, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
