#!/usr/bin/env python3
"""Declare and run one longer matched comparison after all update-30 panels finish."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256

if __package__:
    from .advance_outcome_study import select_development
    from .outcome_resource_plan import RESOURCE_PLAN, cpu_confirmation_plan, training_reserve
    from .report_outcome_ppo import compare_interventions, compare_panel
    from .run_outcome_pilots import checkpoint_at_update
else:
    from advance_outcome_study import select_development
    from outcome_resource_plan import RESOURCE_PLAN, cpu_confirmation_plan, training_reserve
    from report_outcome_ppo import compare_interventions, compare_panel
    from run_outcome_pilots import checkpoint_at_update


def fitting_extension(seconds_per_update, remaining, reserve, maximum_updates):
    """Admit every matched leg assuming each spends its full 10%-padded wall limit."""
    if not seconds_per_update or any(not math.isfinite(value) or value <= 0
                                     for value in seconds_per_update.values()):
        raise ValueError("Measured positive per-leg costs are required")
    if not math.isfinite(remaining) or not math.isfinite(reserve) or reserve < 0:
        raise ValueError("Finite remaining time and a nonnegative reserve are required")
    for updates in range(maximum_updates, 4, -1):
        seconds = {key: 1.1 * cost * updates for key, cost in seconds_per_update.items()}
        available = remaining
        fits = True
        for wall_limit in seconds.values():
            if wall_limit + 600 + reserve > available or 1.2 * wall_limit + 5 > available:
                fits = False
                break
            available -= wall_limit
        if fits:
            return updates, {key: value / 3600 for key, value in seconds.items()}
    return 0, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    allocation, resource_evidence = cpu_confirmation_plan(root)
    rule_path = root / "longer_continuation_rule.json"
    if rule_path.exists() or (root / "longer_continuation_selection.json").exists():
        raise FileExistsError("A longer-continuation rule already exists; inspect it before recovery")
    rule = dict(
        recorded_utc=datetime.now(UTC).isoformat(), resource_allocation=resource_evidence,
        source_sha256=sha256(__file__),
        selection="Choose among all three update-30 replicated development comparisons using the existing "
                  "greedy-guardrail then sampled-win-rate ranking. Preserve both replicas; if a treatment wins, "
                  "also continue the original-critic recipe for the same additional updates.",
        duration="One additional leg per replica and recipe. Maximum 30 further updates for the original recipe "
                 "alone, or 15 for a treatment plus matched original controls. Choose the largest common integer "
                 "count of at least five that fits all measured per-leg costs with 10% padding and every normal "
                 "budget admission check, assuming prior legs spend their full limits.",
        timing="Use each replica's own completed update-20-to-30 wall seconds per update, including internal CPU "
               "evaluation. This extrapolation is conservative about evaluation overhead for longer legs.",
        stopping="Skip the extension if development already reaches 50% sampled wins with a positive paired "
                 "interval and the greedy guardrail, or if five matched further updates do not fit. Do not add "
                 "new recipes or inspect confirmation games. Select the final candidate from completed replicated "
                 "development results at all measured update counts, using the same ranking.",
        confirmation_device=allocation["confirmation_device"],
    )
    atomic_json(rule_path, rule)
    while True:
        stage = read_json(root / "interventions30/manifest.json")
        if stage["status"] == "complete":
            break
        if stage["status"] != "running":
            raise RuntimeError("The intervention stage requires recovery")
        print(f"{datetime.now(UTC).isoformat()} Waiting for all update-30 development panels", flush=True)
        time.sleep(45)
    reserve, current_evidence = training_reserve(root, root / RESOURCE_PLAN)
    if current_evidence != resource_evidence:
        raise ValueError("The resource allocation changed while waiting")
    base = read_json(root / "continuation_selection.json")["selected_base"]
    comparisons = [compare_panel(root, recipe, 30, "development")
                   for recipe in (base, f"{base}_complete", f"{base}_lr3e5")]
    for item in comparisons:
        atomic_json(root / f"development_{item['recipe']}_u30_comparison.json", item)
    atomic_json(root / "intervention_comparison_u30.json", compare_interventions(root, base, 30))
    chosen, guardrail = select_development(comparisons)
    recipe = chosen["recipe"]
    recipes = [base] if recipe == base else [base, recipe]
    ledger = read_json(root / "compute.json")
    if any(job["status"] == "running" for job in ledger["jobs"]):
        raise RuntimeError("The longer comparison requires an idle accelerator")
    costs, parents = {}, {}
    for seed in (0, 1):
        for name in recipes:
            key = f"{name}_seed{seed}"
            directory = root / "trials" / key
            checkpoint = checkpoint_at_update(directory, 30)
            leg = read_json(checkpoint.parent / "manifest.json")
            if leg["start_update"] != 20:
                raise ValueError("Timing must come from the common update-20-to-30 interval")
            costs[key] = leg["elapsed_seconds"] / 10
            parents[key] = dict(path=str(checkpoint), sha256=sha256(checkpoint), leg=leg)
    additional, hours = fitting_extension(costs, ledger["limit_seconds"] - ledger["elapsed_seconds"],
                                          reserve, 30 if recipe == base else 15)
    majority = (chosen["results"]["sampled"]["candidate"]["win_rate"] >= .5
                and chosen["acceptance"]["sampled_paired_ci_excludes_zero"] and guardrail)
    if majority:
        additional, hours = 0, {}
    selection = dict(
        recorded_utc=datetime.now(UTC).isoformat(), rule_sha256=sha256(rule_path),
        selected_recipe=recipe, recipes=recipes, starting_update=30, additional_updates=additional,
        target_update=30 + additional, parents=parents, seconds_per_update=costs,
        wall_limit_hours=hours, accelerator_reserve_seconds=reserve, compute_before=ledger,
        comparisons={item["recipe"]: dict(
            path=str(root / f"development_{item['recipe']}_u30_comparison.json"),
            sha256=sha256(root / f"development_{item['recipe']}_u30_comparison.json")) for item in comparisons},
        skipped_reason="majority-winning development result" if majority else
                       "insufficient allowance for five matched updates" if not additional else None,
    )
    atomic_json(root / "longer_continuation_selection.json", selection)
    print(json.dumps(selection), flush=True)
    if not additional:
        return
    hours_path = root / "longer_continuation_hours.json"
    atomic_json(hours_path, hours)
    command = [sys.executable, "-u", "tools/run_outcome_pilots.py", "--root", str(root),
               "--stage-name", "longer_continuation", "--updates", str(30 + additional),
               "--recipes", *recipes, "--resource-plan", str(root / RESOURCE_PLAN),
               "--leg-hours-file", str(hours_path)]
    with (root / "longer_continuation.log").open("x") as handle:
        subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
    for name in recipes:
        result = compare_panel(root, name, 30 + additional, "development")
        atomic_json(root / f"development_{name}_u{30 + additional}_comparison.json", result)
    print("The longer matched comparison and all development panels are complete", flush=True)


if __name__ == "__main__":
    main()
