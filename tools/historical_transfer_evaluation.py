#!/usr/bin/env python3
"""Evaluate the same checkpoint under reconstructed generation rules on development seeds."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    env = {**os.environ, "PYTHONPATH": str(args.reconstruction / "src")}
    for mode, sample in [("sampled", 0), ("sampled", 1), ("sampled", 2), ("greedy", 0)]:
        subprocess.run(
            [
                sys.executable,
                "tools/evaluate_checkpoint.py",
                "--checkpoint",
                str(args.checkpoint),
                "--games",
                "100",
                "--mode",
                mode,
                "--sampling-seed",
                str(sample),
                "--device",
                args.device,
                "--eval-cpu-threads",
                "4",
                "--training-manifest",
                "cache/datasets/5k-balatro-games/metadata/generation.json",
                "--training-manifest",
                "docs/heuristic/pretraining_v68_seeds.json",
                "--output-dir",
                str(args.output_dir / f"{mode}{sample}"),
            ],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
