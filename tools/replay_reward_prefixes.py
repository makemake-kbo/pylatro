#!/usr/bin/env python3
"""Compare current FastRunner/environment rewards on legal recorded-action prefixes."""

import argparse
import json
from contextlib import suppress
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pylatro import load_game_data
from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.env import BalatroEnv
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training import fast_generate
from pylatro_agent.training.model_generate import load_records
from pylatro_agent.transfer import restore_configuration, sha256, source_hashes
from pylatro_agent.transfer_diagnostics import selected_records
from pylatro_agent.vocab import build_vocab


class PrefixFinished(Exception):
    pass


def capture_reward(original, collected, *args, **kwargs):
    value = original(*args, **kwargs)
    collected.append(value)
    return value


class PrefixActions:
    def __init__(self, rows):
        self.rows, self.step = rows, 0

    def select_action(self, state, phase, mask, **kwargs):
        if self.step == len(self.rows):
            raise PrefixFinished
        action = self.rows[self.step]["action"]
        if not mask[action]:
            raise ValueError("FastRunner rejected an action legal in the environment prefix")
        self.step += 1
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, reward = restore_configuration(load_checkpoint_payload(args.checkpoint, "cpu"))
    seeds = json.loads(args.diagnostic_manifest.read_text())["replay_seeds"]
    records = selected_records(load_records(args.dataset, reward_config=reward), seeds)
    data = load_game_data()
    vocab = build_vocab(data)
    results = []
    for seed in seeds:
        rows = records[seed]
        env = BalatroEnv(
            seed=seed,
            data=data,
            vocab=vocab,
            deck_key=rows[0]["deck_key"],
            stake=rows[0]["stake"],
            reward_config=reward,
            enable_teacher=False,
            max_steps=2000,
        )
        env.reset()
        env_rewards = []
        try:
            for row in rows:
                if not env.action_masks()[row["action"]]:
                    break
                _, value, terminated, truncated, _ = env.step(row["action"])
                env_rewards.append(value)
                if terminated or truncated:
                    break
        finally:
            env.close()
        fast_rewards = []
        original = fast_generate.default_reward

        with (
            patch(
                "pylatro_agent.training.fast_generate.default_reward", partial(capture_reward, original, fast_rewards)
            ),
            suppress(PrefixFinished),
        ):
            fast_generate._run_game_single_pass(
                seed,
                data,
                Tokenizer(vocab=vocab),
                PrefixActions(rows[: len(env_rewards)]),
                reward.gamma,
                reward_config=reward,
                deck_key=rows[0]["deck_key"],
                stake=rows[0]["stake"],
            )
        if len(fast_rewards) != len(env_rewards):
            raise ValueError("Reward comparison returned unequal legal prefixes")
        result = {"seed": seed, "steps": len(env_rewards), "first_fast_env": None, "first_saved_current": None}
        for step, (fast, gym) in enumerate(zip(fast_rewards, env_rewards, strict=True)):
            if result["first_fast_env"] is None and not np.isclose(fast, gym, rtol=1e-6, atol=1e-6):
                result["first_fast_env"] = {"step": step, "fast": fast, "env": gym}
            saved = rows[step]["reward"]
            if result["first_saved_current"] is None and not np.isclose(fast, saved, rtol=1e-6, atol=1e-6):
                result["first_saved_current"] = {"step": step, "saved": saved, "current": fast}
        results.append(result)
    with args.output.open("x") as stream:
        json.dump(
            {
                "results": results,
                "source_hashes": source_hashes(),
                "checkpoint_sha256": sha256(args.checkpoint),
                "diagnostic_manifest_sha256": sha256(args.diagnostic_manifest),
                "scope": "Legal current-rule prefixes only; original HF targets are untouched",
            },
            stream,
            indent=2,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                "games": len(results),
                "steps": sum(row["steps"] for row in results),
                "fast_env_divergences": sum(row["first_fast_env"] is not None for row in results),
            }
        )
    )


if __name__ == "__main__":
    main()
