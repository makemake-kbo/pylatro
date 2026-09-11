#!/usr/bin/env python3
"""Compare Python/native state copying and optionally complete training records.

Build with scripts/compile_cython.py first. Example:
    .venv/bin/python scripts/bench_state_copy.py --games 1 --output /tmp/copy.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import time
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

import pylatro._state_copy as native
import pylatro.models as models
from pylatro import add_joker, create_run_state, load_game_data, start_blind
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_simulation import copy_for_scoring
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_generate import _run_game_single_pass
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--copies", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 0 or args.copies <= 0 or args.repeats <= 0:
        parser.error("games must be nonnegative; copies and repeats must be positive")
    if not any(str(native.__file__).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        parser.error("build the optional _state_copy extension with scripts/compile_cython.py first")

    source = Path(__file__).resolve().parents[1] / "src/pylatro/_state_copy.py"
    spec = importlib.util.spec_from_file_location("copy_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    implementations = {"python": reference.copy_state_object, "native": native.copy_state_object}
    data = load_game_data()
    state = create_run_state("copy_native", deck_key="b_blue", data=data)
    start_blind(state, "Small")
    for key in ("j_scholar", "j_hologram", "j_supernova", "j_blackboard", "j_card_sharp"):
        add_joker(state, key)
    tokenizer = Tokenizer(vocab=build_vocab(data)) if args.games else None
    rows = []

    def record(**row):
        rows.append(row)
        print(json.dumps(row), flush=True)

    original = models.copy_state_object
    try:
        for repeat in range(args.repeats):
            copies = []
            for label in (("python", "native") if repeat % 2 == 0 else ("native", "python")):
                models.copy_state_object = implementations[label]
                for _ in range(100):
                    copy_for_scoring(state)
                started = time.perf_counter()
                for _ in range(args.copies):
                    clone = copy_for_scoring(state)
                elapsed = time.perf_counter() - started
                copies.append(pickle.dumps(clone))
                record(mode="copy", repeat=repeat, variant=label, seconds=elapsed, copies=args.copies)
            assert copies[0] == copies[1], "copy state mismatch"

        for offset in range(args.games):
            seed = args.seed + offset
            outputs = []
            for label in (("python", "native") if offset % 2 == 0 else ("native", "python")):
                models.copy_state_object = implementations[label]
                started = time.perf_counter()
                result = _run_game_single_pass(
                    seed, data, tokenizer, HeuristicAgent(shop_policy="search", grow_scalers=True),
                    0.995, deck_key="b_blue",
                )
                elapsed = time.perf_counter() - started
                outputs.append(result)
                record(mode="generate", seed=seed, variant=label, seconds=elapsed, records=len(result[0]))
            # Includes every observation array, reward, return, action, outcome
            # and metadata field, outside the measured interval.
            np.testing.assert_equal(outputs[0], outputs[1], err_msg=f"seed {seed} record mismatch")
    finally:
        models.copy_state_object = original

    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "extension_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
        "parity_passed": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
