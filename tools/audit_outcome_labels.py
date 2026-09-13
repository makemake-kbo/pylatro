#!/usr/bin/env python3
"""Reconstruct which completed-game states had outcome labels during their PPO update."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256


def reconstruct_labels(rows, start_update, rollout_length):
    clocks = defaultdict(int)
    result = []
    for row in rows:
        environment = row["environment"]
        end = clocks[environment] + row["steps"]
        update = start_update + (end - 1) // rollout_length + 1
        if update != row["update"]:
            raise ValueError("Episode lengths do not reproduce the recorded rollout boundary")
        suffix = min(row["steps"], (end - 1) % rollout_length + 1)
        result.append(dict(**row, label_available_states=suffix,
                           label_unavailable_prefix_states=row["steps"] - suffix,
                           initial_state_label_available=suffix == row["steps"]))
        clocks[environment] = end
    return result


def label_metrics(rows):
    result = {}
    for name, group in (("all", rows), ("winning", [row for row in rows if row["won"]]),
                        ("losing", [row for row in rows if not row["won"]])):
        states = sum(row["steps"] for row in group)
        labeled = sum(row["label_available_states"] for row in group)
        initial_labeled = sum(row["initial_state_label_available"] for row in group)
        result[name] = dict(games=len(group), states=states, labeled_states=labeled,
                            label_fraction=labeled / states if states else None,
                            initial_states_labeled=initial_labeled,
                            initial_state_label_fraction=initial_labeled / len(group) if group else None,
                            mean_game_length=states / len(group) if group else None)
    labeled_starts = [row for row in rows if row["initial_state_label_available"]]
    result["observed_win_rate"] = sum(row["won"] for row in rows) / len(rows) if rows else None
    result["win_rate_in_labeled_initial_states"] = (
        sum(row["won"] for row in labeled_starts) / len(labeled_starts) if labeled_starts else None
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    destination = args.root / f"{args.name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    directory = args.root / "trials" / args.trial
    recipe = read_json(directory / "recipe.json")
    source = directory / "tensorboard/training_episodes.jsonl"
    episodes = [json.loads(line) for line in source.read_text().splitlines()]
    legs = [read_json(path) for path in directory.glob("leg_*/manifest.json")]
    if not legs or any(leg["status"] != "complete" for leg in legs):
        raise ValueError("This audit requires completed training legs")
    annotated = []
    for leg in sorted(legs, key=lambda item: item["start_update"]):
        rows = [row for row in episodes if leg["start_update"] < row["update"] <= leg["update_count"]]
        annotated.extend(reconstruct_labels(rows, leg["start_update"], recipe["ppo_config"]["rollout_length"]))
    if len(annotated) != len(episodes):
        raise ValueError("Missing or duplicated training episodes")
    labeled_by_update = defaultdict(int)
    for row in annotated:
        labeled_by_update[row["update"]] += row["label_available_states"]
    events = EventAccumulator(str(directory / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    logged = {row.step: row.value for row in events.Scalars("critic/rollout_labels/labeled_rows")}
    if dict(labeled_by_update) != logged:
        raise ValueError("Reconstructed label counts differ from the actual PPO buffer metrics")
    forecast_panels = {}
    names = ["baseline_dev200", f"{args.trial}_u{max(leg['update_count'] for leg in legs)}_dev200"]
    for name in names:
        path = args.root / "evaluations" / name
        if not (path / "comparison.json").exists():
            continue
        rows = [json.loads(line) for sampling in (0, 1, 2)
                for line in (path / f"sample{sampling}/outcomes.jsonl").read_text().splitlines()]
        predicted = np.array([row["critic_forecasts"][0]["outcome_probabilities"][-1] for row in rows])
        actual = np.array([row["won"] for row in rows], dtype=float)
        forecast_panels[name] = dict(
            games=len(rows), game_seeds=len({row["seed"] for row in rows}),
            mean_start_win_forecast=float(predicted.mean()), observed_win_rate=float(actual.mean()),
            start_win_brier=float(np.square(predicted - actual).mean()),
            comparison_sha256=sha256(path / "comparison.json"),
        )
    result = dict(
        trial=args.trial, recipe_sha256=sha256(directory / "recipe.json"),
        episode_log_sha256=sha256(source), source_sha256=sha256(__file__),
        source_rollout_sha256=sha256("src/pylatro_agent/training/ppo_rollout.py"),
        metrics=label_metrics(annotated), labeled_rows_by_update=labeled_by_update,
        exact_match_to_logged_ppo_label_counts=True, forecast_panels=forecast_panels,
        interpretation="Reconstructs episode boundaries from each environment's cumulative completed episode lengths, "
                       "resetting at every training-leg restart. A completed episode labels only its suffix in the "
                       "rollout where it ended. Earlier prefixes can later enter complete-episode critic replay, "
                       "but were unlabeled when PPO consumed their rollout. Exact logged label counts match. "
                       "Different policy/phase/length distributions remain confounders; these counts and calibration "
                       "drift do not isolate the gameplay effect of the outcome-loss mask.",
    )
    atomic_json(destination, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
