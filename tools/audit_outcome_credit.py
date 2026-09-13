#!/usr/bin/env python3
"""Audit complete-game credit targets without treating one realized outcome as an action oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, critic_from_pooled
from pylatro_agent.transfer import load_policy, sha256
from pylatro_agent.vocab import build_vocab


def complete_advantages(rewards, values, trace):
    result = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        running = rewards[index] + next_value - values[index] + trace * running
        result[index] = running
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.root
    path = root / "critic_seed0/checkpoint.pt"
    model, _, _ = load_policy(path, build_vocab(load_game_data()), torch.device("cpu"), mixture=0.5)
    rows = [json.loads(line) for line in (root / "critic_games/outcomes.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == "validation"]
    records = []
    for row in rows:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError("Critic trajectory changed")
        with np.load(row["path"], allow_pickle=False) as data:
            with torch.no_grad():
                outputs = critic_from_pooled(
                    model.value_head, torch.from_numpy(data["x"]).float(), torch.from_numpy(data["ante"]).float()
                )
                values = outputs["expected_return"].numpy()
            rewards = np.zeros(len(values))
            rewards[-1] = 10 * row["won"]
            short = complete_advantages(rewards, values, 0.97)
            full = complete_advantages(rewards, values, 1.0)
            np.testing.assert_allclose(full + values, 10 * row["won"], atol=1e-6)
            records.extend(
                dict(
                    seed=row["seed"],
                    step=index,
                    ante=int(data["ante"][index]),
                    phase=str(data["phase"][index]),
                    won=row["won"],
                    short=float(short[index]),
                    full=float(full[index]),
                    distance=len(values) - 1 - index,
                )
                for index in range(len(values))
            )

    def summarize(group):
        if not group:
            return None
        by_game = {}
        for record in group:
            by_game.setdefault(record["seed"], []).append(record)
        return dict(
            games=len(by_game),
            decisions=len(group),
            sign_disagreement_decision_weighted=float(np.mean([(r["short"] > 0) != (r["full"] > 0) for r in group])),
            sign_disagreement_game_weighted=float(
                np.mean([np.mean([(r["short"] > 0) != (r["full"] > 0) for r in game]) for game in by_game.values()])
            ),
            short_advantage_mean=float(np.mean([r["short"] for r in group])),
            full_advantage_mean=float(np.mean([r["full"] for r in group])),
        )

    result = dict(
        checkpoint=str(path),
        checkpoint_sha256=sha256(path),
        source_manifest_sha256=sha256(root / "critic_games/manifest.json"),
        critic_selection=read_json(root / "critic_seed0/manifest.json")["selected_update"],
        scope="128 critic-validation games; also used to select critic checkpoint; not independent confirmation",
        gamma=1.0,
        trace_parameters=[0.97, 1.0],
        interpretation=(
            "Raw pre-normalization advantages on complete games. The PPO trial still bootstraps rollout "
            "boundaries. A realized win/loss does not establish that every preceding action was good/bad. "
            "Gameplay experiments, not sign agreement, test whether the changed credit helps."
        ),
        aggregate=summarize(records),
        by_ante={str(ante): summarize([r for r in records if r["ante"] == ante]) for ante in range(1, 9)},
        by_phase={
            phase: summarize([r for r in records if r["phase"] == phase])
            for phase in sorted({r["phase"] for r in records})
        },
        distant_winning_decisions=summarize([r for r in records if r["won"] and r["distance"] >= 100]),
        distant_losing_decisions=summarize([r for r in records if not r["won"] and r["distance"] >= 100]),
        direct_terminal_coefficient_100_actions={
            "old_gamma0997_lambda097": (0.997 * 0.97) ** 100,
            "gamma1_lambda097": 0.97**100,
            "gamma1_lambda1": 1,
        },
    )
    atomic_json(root / "credit_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
