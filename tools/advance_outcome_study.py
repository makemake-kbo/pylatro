#!/usr/bin/env python3
"""Advance a completed replicated GAE comparison to declared, matched interventions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256

if __package__:
    from .report_outcome_ppo import compare_panel
else:
    from report_outcome_ppo import compare_panel


def select_development(results):
    """Prefer the greedy guardrail, then rank sampled strength across both replicas."""
    if not results:
        raise ValueError("No complete replicated development comparison")
    eligible = [item for item in results
                if item["acceptance"]["greedy_rules_out_five_point_regression"]]
    pool = eligible or results
    return max(pool, key=lambda item: (
        item["results"]["sampled"]["candidate"]["win_rate"],
        item["results"]["greedy"]["candidate"]["win_rate"],
        item["recipe"],
    )), bool(eligible)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--wait-for-stage", default="pilot20")
    parser.add_argument("--hours-per-leg", type=float, default=.85)
    parser.add_argument("--adaptive-rule", default="adaptive_trial_rule_v4.json")
    args = parser.parse_args()
    if args.hours_per_leg <= 0:
        parser.error("A positive per-leg time allowance is required")
    root = args.root
    selection_path = root / "continuation_selection.json"
    if selection_path.exists() or (root / "interventions30").exists():
        raise FileExistsError("A continuation selection or stage already exists; inspect before recovery")
    rule = root / args.adaptive_rule
    critic_suffix = read_json(rule).get("critic_recipe_suffix", "terminal")
    if critic_suffix not in ("terminal", "complete"):
        raise ValueError("Unknown declared critic intervention")
    while True:
        metadata = read_json(root / args.wait_for_stage / "manifest.json")
        status = metadata["status"]
        if status == "complete":
            break
        if status != "running":
            raise RuntimeError(f"Prior stage did not complete: {status}")
        print(f"{datetime.now(UTC).isoformat()} Waiting for all replicated development panels", flush=True)
        time.sleep(45)
    comparisons = [compare_panel(root, recipe, 20, "development") for recipe in ("gae100", "gae097")]
    for result in comparisons:
        atomic_json(root / f"development_{result['recipe']}_u20_comparison.json", result)
    chosen, guardrail_met = select_development(comparisons)
    base = chosen["recipe"]
    selection = dict(
        recorded_utc=datetime.now(UTC).isoformat(),
        rule=str(rule), rule_sha256=sha256(rule),
        selected_base=base, starting_update=20, target_update=30,
        recipes=[base, f"{base}_{critic_suffix}", f"{base}_lr3e5"], training_replicates=[0, 1],
        selection_rule="Prefer recipes whose development greedy interval rules out a >5-point regression; "
                       "then maximize mean sampled win rate across both replicas, with greedy win rate as tie-break. "
                       "If neither passes the guardrail, retain the strongest sampled recipe for diagnostic trials.",
        selected_base_passes_greedy_guardrail=guardrail_met,
        comparisons={item["recipe"]: {
            "path": str(root / f"development_{item['recipe']}_u20_comparison.json"),
            "sha256": sha256(root / f"development_{item['recipe']}_u20_comparison.json"),
            "sampled": item["results"]["sampled"], "greedy": item["results"]["greedy"],
        } for item in comparisons},
        confirmation_used=False,
    )
    atomic_json(selection_path, selection)
    print(json.dumps(selection), flush=True)
    if (chosen["results"]["sampled"]["candidate"]["win_rate"] >= .5
            and chosen["acceptance"]["sampled_paired_ci_excludes_zero"] and guardrail_met):
        print("Majority-winning development result: stop here for confirmation selection", flush=True)
        return
    command = [sys.executable, "-u", "tools/run_outcome_pilots.py", "--root", str(root),
               "--stage-name", "interventions30", "--updates", "30", "--hours-per-leg", str(args.hours_per_leg),
               "--recipes", *selection["recipes"], "--branch-base", base, "--branch-update", "20"]
    with (root / "interventions30.log").open("x") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    for recipe in selection["recipes"]:
        result = compare_panel(root, recipe, 30, "development")
        atomic_json(root / f"development_{recipe}_u30_comparison.json", result)
    print("All declared interventions and replicated development panels are complete", flush=True)


if __name__ == "__main__":
    main()
