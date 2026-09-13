#!/usr/bin/env python3
"""Create TensorBoard learning curves from preserved CPU trial measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.transfer import sha256


def scalars(value, prefix=""):
    for key, child in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict):
            yield from scalars(child, name)
        elif isinstance(child, (int, float)) and not isinstance(child, bool):
            yield name, child


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    args = parser.parse_args()
    sources = sorted(args.root.glob("*_seed*/learning.jsonl"))
    sources += sorted((args.root / "hand_exposure_control").glob("pure_ar_seed*/learning.jsonl"))
    if len(sources) != 8:
        raise ValueError("Expected six main CPU training trials and two offline exposure controls")
    hashes = {str(path): sha256(path) for path in sources}
    destination = args.root / "learning_tensorboard"
    marker = destination / "provenance.json"
    if marker.exists():
        if read_json(marker)["source_hashes"] != hashes:
            raise ValueError("Learning measurements changed since TensorBoard export")
        return
    destination.mkdir(exist_ok=False)
    for path in sources:
        with SummaryWriter(str(destination / path.parent.name)) as writer:
            writer.add_text("provenance/source", f"{path}; SHA-256 {hashes[str(path)]}")
            for line in path.read_text().splitlines():
                row = json.loads(line)
                for phase in ("training", "validation"):
                    if phase not in row:
                        continue
                    values = row[phase] if "all" in row[phase] else {"all": row[phase]}
                    for name, value in scalars(values):
                        writer.add_scalar(f"{phase}/{name}", value, row["update"])
                elapsed = row.get("elapsed_cpu_seconds", row.get("elapsed_seconds"))
                writer.add_scalar("compute/elapsed_cpu_seconds", elapsed, row["update"])
    atomic_json(marker, dict(
        source_hashes=hashes, curves=len(sources), accelerator_seconds=0,
        note="Derived from original measurements; TensorBoard event timestamps are export times, not training times",
    ))
    print(f"TensorBoard curves saved: {destination}")


if __name__ == "__main__":
    main()
