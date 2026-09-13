#!/usr/bin/env python3
"""Create an isolated simulator reconstruction, verifying every restored source against the HF manifest."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from pylatro_agent.transfer import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(root / "src", args.destination / "src", ignore=shutil.ignore_patterns("*.so", "__pycache__", "*.c"))
    shutil.copytree(root / "tools", args.destination / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (args.destination / "scripts").mkdir()
    shutil.copy2(root / "scripts/compile_cython.py", args.destination / "scripts/compile_cython.py")
    generation = json.loads(args.generation.read_text())
    replaced = {}
    matched_revisions = {}
    for path, expected in generation["sources"].items():
        if not path.endswith(".py"):
            continue
        if not (
            path.startswith("src/pylatro/")
            or path
            in {
                "src/pylatro_cli/controller.py",
                "src/pylatro_agent/env.py",
                "src/pylatro_agent/build_value.py",
                "src/pylatro_agent/training/fast_runner.py",
                "src/pylatro_agent/training/fast_generate.py",
            }
        ):
            continue
        target = args.destination / path
        import hashlib

        revisions = subprocess.check_output(["git", "rev-list", "--all", "--", path], cwd=root, text=True).splitlines()
        for revision in revisions:
            blob = subprocess.check_output(["git", "show", revision + ":" + path], cwd=root)
            if hashlib.sha256(blob).hexdigest() == expected:
                target.write_bytes(blob)
                matched_revisions[path] = revision
                break
        else:
            raise ValueError(f"No historical source matches dataset: {path}")
        replaced[path] = expected
    write_json(
        args.destination / "reconstruction.json",
        {
            "generation_sha256": sha256(args.generation),
            "matching_commits": matched_revisions,
            "restored_sources": replaced,
            "scope": "Historical engine/wrappers; current policy architecture and measurement tooling retained",
        },
    )
    print(args.destination)


if __name__ == "__main__":
    main()
