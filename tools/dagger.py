#!/usr/bin/env python3
"""Run, resume, or report an iterative DAgger-from-V68 experiment.

    python tools/dagger.py run --checkpoint incumbents/ppo_replica1/checkpoint.pt --out runs/dagger_v68
    python tools/dagger.py report --out runs/dagger_v68

Re-running ``run`` with the same ``--out`` resumes: finished collections,
checkpoints and panel evaluations are reused, and an interrupted collection
replays its seeds. Create ``<out>/STOP`` to stop after the current stage.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pylatro_agent.dagger import DaggerConfig, load_exclusions, read_json, render_report, run  # noqa: E402


def default_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("run", help="run or resume an experiment")
    start.add_argument("--checkpoint", type=Path, required=True, help="starting student checkpoint")
    start.add_argument("--out", type=Path, required=True, help="run directory (created or resumed)")
    start.add_argument(
        "--exclude-seeds",
        type=Path,
        nargs="*",
        default=[],
        help="JSON/JSONL seed lists that panel and training seeds must avoid",
    )
    # Unset flags take the run directory's saved value when resuming, and the
    # DaggerConfig default otherwise.
    defaults = DaggerConfig()
    for field in fields(DaggerConfig):
        flag = "--" + field.name.replace("_", "-")
        value = getattr(defaults, field.name)
        if field.name == "panel_modes":
            start.add_argument(flag, nargs="+", choices=("greedy", "sample"), help=f"default: {' '.join(value)}")
        elif field.name == "collection_device":
            start.add_argument(flag, help="inference device for collection and panels (default: --device)")
        elif field.name == "device":
            start.add_argument(flag, help="training device (default: cuda, then mps, then cpu)")
        else:
            start.add_argument(flag, type=type(value), help=f"default: {value}")
    report = commands.add_parser("report", help="print the learning curve of a run directory")
    report.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "report":
        config = DaggerConfig.from_json(read_json(args.out / "config.json"))
        print(render_report(read_json(args.out / "learning_curve.json"), config), end="")
        return
    saved = args.out / "config.json"
    base = DaggerConfig.from_json(read_json(saved)) if saved.exists() else DaggerConfig(device=default_device())
    values = {field.name: getattr(base, field.name) for field in fields(DaggerConfig)}
    values.update(
        {
            field.name: getattr(args, field.name)
            for field in fields(DaggerConfig)
            if getattr(args, field.name) is not None
        }
    )
    values["panel_modes"] = tuple(values["panel_modes"])
    config = DaggerConfig(**values)
    curve = run(args.checkpoint, args.out, config, exclusions=load_exclusions(args.exclude_seeds))
    print(render_report(curve, config), end="")


if __name__ == "__main__":
    main()
