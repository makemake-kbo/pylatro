#!/usr/bin/env python3
"""Bounded hardware reproducer for MPS random-state checkpoint restoration."""

from __future__ import annotations

import argparse
import ast
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pylatro_agent.checkpoint import capture_rng_states, restore_rng_states
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json
from pylatro_agent.transfer import sha256


def original_functions(path):
    tree = ast.parse(path.read_text())
    names = {"capture_rng_states", "restore_rng_states"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("Expected original checkpoint RNG functions")
    namespace = dict(torch=torch, random=random, np=np, Any=Any)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["capture_rng_states"], namespace["restore_rng_states"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-checkpoint-source", type=Path, required=True)
    args = parser.parse_args()
    destination = args.root / "mps_rng_hardware"
    destination.mkdir(exist_ok=False)
    old_capture, old_restore = original_functions(args.original_checkpoint_source)
    with accelerator_job(args.root, "mps_rng_checkpoint_reproducer", 60):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS hardware is required")
        probabilities = torch.full((512, 16), 1 / 16, device="mps")
        results = {}
        for name, capture, restore in (("before", old_capture, old_restore),
                                        ("after", capture_rng_states, restore_rng_states)):
            torch.manual_seed(1729)
            torch.rand(13, device="mps")
            state = capture()
            path = destination / f"{name}.pt"
            torch.save(state, path)
            expected = torch.multinomial(probabilities, 1)
            torch.rand(257, device="mps")
            # The production resume loader maps every tensor to the active
            # backend; restore must convert RNG ByteTensors back to CPU.
            loaded = torch.load(path, map_location="mps", weights_only=False)
            restore(loaded)
            actual = torch.multinomial(probabilities, 1)
            results[name] = dict(
                saved_rng_keys=sorted(state), samples=512,
                matching_actions=int((expected == actual).sum().item()),
                exactly_reproduced=bool(torch.equal(expected, actual)),
                checkpoint_sha256=sha256(path),
            )
        if results["before"]["exactly_reproduced"] or not results["after"]["exactly_reproduced"]:
            raise AssertionError("Expected the original failure and an exact fixed MPS replay")
        result = dict(
            passed=True, device="mps", torch_version=torch.__version__, results=results,
            original_source=str(args.original_checkpoint_source),
            original_source_sha256=sha256(args.original_checkpoint_source),
            fixed_source_sha256=sha256(Path(__file__).parents[1] / "src/pylatro_agent/checkpoint.py"),
            script_sha256=sha256(__file__),
            interpretation="Checkpoint RNG persistence only: unchanged categorical probabilities, 512 samples. "
                           "The original implementation fails exact replay; the fixed implementation restores it. "
                           "This does not establish a gameplay gain or restore an omitted legacy state.",
        )
        atomic_json(destination / "result.json", result)
        print(result, flush=True)


if __name__ == "__main__":
    main()
