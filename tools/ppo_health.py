#!/usr/bin/env python3
"""Audit the tail of a PPO run for trust-region / exploration / planet health.

Reads TensorBoard scalar event files for a run and prints the last N values of
the metrics that gate the smoke-run pass/fail decision, so a run can be
audited without opening TensorBoard.

Example:

    uv run --extra agent python tools/ppo_health.py --log-dir runs/ppo_ft_ante4_run --last 20
"""

from __future__ import annotations

import argparse
import sys

# Metrics to surface, in the order the smoke-run gates are described.
HEALTH_TAGS = [
    "ppo/approx_kl",
    "ppo/approx_kl_p95",
    "ppo/approx_kl_max",
    "ppo/minibatch_fraction",
    "ppo/clip_fraction",
    "ppo/clip_fraction_max",
    "ppo/entropy_normalized",
    "ppo/action_type_entropy_normalized",
    "debug/chosen_action_prob_mean",
    "rollout/win_rate",
    "eval/win_rate",
    "planet/claim_main_hand_match_fraction",
    "planet/use_main_hand_match_fraction",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the last N values of key PPO health metrics.")
    parser.add_argument("--log-dir", type=str, required=True, help="TensorBoard run directory.")
    parser.add_argument("--last", type=int, default=20, help="Number of trailing updates to show (default: 20).")
    args = parser.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(args.log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    found_any = False
    for tag in HEALTH_TAGS:
        if tag not in available:
            print(f"{tag}: [NOT FOUND]")
            continue
        found_any = True
        events = ea.Scalars(tag)
        tail = events[-args.last :]
        if not tail:
            print(f"{tag}: [NO DATA]")
            continue
        values = [e.value for e in tail]
        steps = [e.step for e in tail]
        vmin, vmax, vmean = min(values), max(values), sum(values) / len(values)
        formatted = ", ".join(f"{v:.4g}" for v in values)
        print(f"{tag}  (steps {steps[0]}..{steps[-1]})")
        print(f"    min={vmin:.4g}  max={vmax:.4g}  mean={vmean:.4g}")
        print(f"    tail: {formatted}")

    if not found_any:
        print(f"No matching scalar tags found in {args.log_dir}.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
