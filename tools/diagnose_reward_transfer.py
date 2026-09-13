#!/usr/bin/env python3
"""Reproduce the boss-key reward discrepancy on an explicit recorded action prefix."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from pylatro import load_game_data
from pylatro_agent.env import BalatroEnv
from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, default_reward_components, state_potential_breakdown
from pylatro_agent.training.fast_generate import _capture_info
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.transfer import source_hashes, write_json

ACTIONS = [0, 9788, 2663, 2519, 17297, 13771, 17296, 13771, 17320, 0, 2519, 2519, 5600, 2519]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_game_data()
    runner = FastRunner(2045118, data, deck_key="b_blue", raise_errors=True)
    env = BalatroEnv(seed=2045118, data=data, deck_key="b_blue", enable_teacher=False)
    env.reset()
    rows = []
    reward_pairs = []

    def paired_reward(state, previous, current, terminated, won, config):
        corrected = default_reward_components(state, previous, current, terminated, won, config)
        omitted = default_reward_components(
            state,
            {k: v for k, v in previous.items() if k != "boss_key"},
            {k: v for k, v in current.items() if k != "boss_key"},
            terminated,
            won,
            config,
        )
        reward_pairs.append({"with_boss_key": corrected["total"], "without_boss_key": omitted["total"]})
        return corrected

    for step, action in enumerate(ACTIONS):
        runner.step(action)
        with patch("pylatro_agent.env.default_reward_components", paired_reward):
            env.step(action)
        fast = _capture_info(runner)
        historical = {k: v for k, v in fast.items() if k != "boss_key"}
        current = env._capture_state_info()
        rows.append(
            {
                "step": step,
                "action": action,
                "boss": current["boss_key"],
                "legacy_fast_potential": state_potential_breakdown(historical, DEFAULT_REWARD_CONFIG),
                "corrected_fast_potential": state_potential_breakdown(fast, DEFAULT_REWARD_CONFIG),
                "env_potential": state_potential_breakdown(current, DEFAULT_REWARD_CONFIG),
                "paired_reward_boss_key_intervention": reward_pairs[-1],
            }
        )
    env.close()
    write_json(
        args.output,
        {
            "seed": 2045118,
            "rows": rows,
            "source_hashes": source_hashes(),
            "note": "New fast-generated records have reward_snapshot_version=3; original HF rewards remain unchanged.",
        },
    )
    print(json.dumps(rows[-1], indent=2))


if __name__ == "__main__":
    main()
