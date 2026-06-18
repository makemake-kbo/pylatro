#!/usr/bin/env python3
"""Micro-benchmark for the pure-engine hot paths touched by Cython @cython.locals
typing: rng.pseudohash and the scoring poker-hand evaluation.  Imports only
`pylatro` (no torch/agent stack), so it runs anywhere the core is installed.

Usage:
    uv run python scripts/bench_engine_hotpaths.py
    uv run python scripts/bench_engine_hotpaths.py --json out.json
    uv run python scripts/bench_engine_hotpaths.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from statistics import mean

from pylatro import (
    create_run_state,
    evaluate_poker_hand,
    load_game_data,
    select_blind,
    start_blind,
)
from pylatro.rng import pseudohash
from pylatro.scoring import _card_nominal, _get_straight, get_poker_hand_info, score_hand


def _bench(fn, iterations: int, warmup: int = 200) -> float:
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return mean(times) * 1e6  # microseconds


def _make_state():
    data = load_game_data()
    state = create_run_state("bench42", 1, "b_red", data=data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    return state


def run() -> dict[str, float]:
    state = _make_state()
    hand = list(state.hand_cards[:5])
    if len(hand) < 5:
        raise RuntimeError("not enough cards dealt for the benchmark")

    cases = {
        "pseudohash(short)": lambda: pseudohash("lucky_mult"),
        "pseudohash(seed8)": lambda: pseudohash("AAAAAAAA"),
        "_card_nominal()": lambda: _card_nominal(state, hand[0]),
        "_get_straight()": lambda: _get_straight(state, hand),
        "evaluate_poker_hand()": lambda: evaluate_poker_hand(state, hand),
        "get_poker_hand_info()": lambda: get_poker_hand_info(state, hand),
        "score_hand()": lambda: score_hand(state, list(hand)),
    }
    iters = {
        "pseudohash(short)": 50000,
        "pseudohash(seed8)": 50000,
        "_card_nominal()": 50000,
        "_get_straight()": 20000,
        "evaluate_poker_hand()": 10000,
        "get_poker_hand_info()": 10000,
        "score_hand()": 5000,
    }
    return {name: _bench(fn, iters[name]) for name, fn in cases.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = ap.parse_args()

    if args.compare:
        before = json.loads(open(args.compare[0]).read())
        after = json.loads(open(args.compare[1]).read())
        print(f"{'function':<26}{'before us':>12}{'after us':>12}{'speedup':>10}")
        for name in before:
            b, a = before[name], after.get(name, 0.0)
            spd = b / a if a else float("inf")
            print(f"{name:<26}{b:>12.3f}{a:>12.3f}{spd:>9.2f}x")
        return

    results = run()
    print(f"{'function':<26}{'mean us':>12}")
    for name, us in results.items():
        print(f"{name:<26}{us:>12.3f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
