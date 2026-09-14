#!/usr/bin/env python3
"""Describe remaining confirmation failures without collecting games or changing selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256

if __package__:
    from .export_outcome_confirmation import verify_export
else:
    from export_outcome_confirmation import verify_export


def failure_profile(rows, seeds):
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected distinct game seeds")
    groups = {}
    for row in rows:
        arm = row["arm"]
        key = row["seed"], row["sampling_seed"]
        group = groups.setdefault(arm, {})
        if key in group:
            raise ValueError(f"Duplicate game in {arm}: {key}")
        group[key] = row
    required = {"control", "seed0", "seed1", "teacher"}
    if set(groups) not in (required, required | {"matched_control_seed0", "matched_control_seed1"}):
        raise ValueError("Expected both selected replicas, initial control and teacher")
    for arm, group in groups.items():
        repeats = (None,) if arm == "teacher" else (None, 0, 1, 2)
        if set(group) != {(seed, repeat) for seed in seeds for repeat in repeats}:
            raise ValueError(f"Missing or unexpected comparison seeds/repeats in {arm}")

    sampled = [row for arm in ("seed0", "seed1") for row in groups[arm].values()
               if row["sampling_seed"] is not None]
    greedy = [row for arm in ("seed0", "seed1") for row in groups[arm].values()
              if row["sampling_seed"] is None]
    wins_per_seed = {}
    greedy_per_seed = {}
    teacher_wins = set()
    for seed in seeds:
        wins_per_seed[seed] = sum(groups[arm][seed, repeat]["won"]
                                  for arm in ("seed0", "seed1") for repeat in (0, 1, 2))
        greedy_per_seed[seed] = sum(groups[arm][seed, None]["won"] for arm in ("seed0", "seed1"))
        if groups["teacher"][seed, None]["won"]:
            teacher_wins.add(seed)

    def membership(values):
        return dict(count=len(values), seeds=sorted(values))

    def losses(values):
        lost = [row for row in values if not row["won"]]
        return dict(games=len(values), wins=sum(row["won"] for row in values), losses=len(lost),
                    highest_ante_reached_on_losses=dict(sorted(Counter(row["max_ante"] for row in lost).items())),
                    stalls=sum(row["stalled"] for row in values), censored=sum(row["censored"] for row in values))

    never_won = {seed for seed, count in wins_per_seed.items() if count == 0}
    return dict(
        game_seeds=len(seeds), sampled_games_per_seed=6, greedy_games_per_seed=2,
        selected_sampled=losses(sampled), selected_greedy=losses(greedy),
        sampled_wins_per_seed_histogram={count: sum(value == count for value in wins_per_seed.values())
                                         for count in range(7)},
        teacher_outcome_by_sampled_wins_per_seed={
            outcome: {count: sum((seed in teacher_wins) == outcome and wins_per_seed[seed] == count
                                 for seed in seeds) for count in range(7)}
            for outcome in (False, True)
        },
        no_sampled_wins=membership(never_won),
        teacher_wins_with_no_sampled_wins=membership(never_won & teacher_wins),
        any_greedy_win_with_no_sampled_wins=membership(
            {seed for seed in never_won if greedy_per_seed[seed] > 0}),
        interpretation="Post-confirmation descriptive counts, conditional on the two fixed trained replicas. "
                       "Six sampled losses do not establish that a seed is impossible for the policy. "
                       "Winning at least one of six games is not a per-game win rate. Teacher access is privileged. "
                       "Loss antes localize termination but do not identify the decisions that caused failure. "
                       "These counts do not alter candidate selection or acceptance.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-directory", type=Path, default=Path("docs/transfer/data"))
    args = parser.parse_args()
    verify_export(args.data_directory)
    games = args.data_directory / "outcome_confirmation_games.jsonl"
    metadata = args.data_directory / "outcome_confirmation_manifest.json"
    rows = [json.loads(line) for line in games.read_text().splitlines()]
    protocol = read_json(args.root / "manifest.json")
    result = dict(results=failure_profile(rows, protocol["confirmation_seeds"]),
                  games_sha256=sha256(games), export_manifest_sha256=sha256(metadata),
                  source_sha256=sha256(__file__), accelerator_seconds=0)
    path = args.root / "confirmation_failure_profile.json"
    if path.exists():
        if read_json(path) != json.loads(json.dumps(result)):
            raise ValueError("Existing failure profile differs; preserve it and inspect the changed evidence")
    else:
        atomic_json(path, result)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
