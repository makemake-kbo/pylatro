"""Replay a checkpoint's next frozen cohort and preserve failed-game actions."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.checkpoint import load_checkpoint_payload, restore_rng_states
from pylatro_agent.cohort_study import config_from_dict
from pylatro_agent.learning_bottleneck import atomic_json
from pylatro_agent.training.cohort import CohortFailure, collect_complete_cohort
from pylatro_agent.training.evaluation_envs import EvaluationEnvs
from pylatro_agent.training.ppo_evaluation import training_seed_reservations
from pylatro_agent.transfer import load_policy, sha256, source_hashes
from pylatro_agent.vocab import build_vocab


class RecordingEnvs:
    def __init__(self, envs):
        self.envs = envs
        self.actions = {}
        self.observations = {}
        self.phases = {}

    def reset(self, seed):
        obs, phase = self.envs.reset(seed)
        self.actions[seed] = []
        self.observations[seed] = obs
        self.phases[seed] = phase
        return obs, phase

    def step_training(self, requests):
        results = self.envs.step_training(requests)
        for seed, action in requests:
            self.actions[seed].append(action)
            obs, _reward, terminated, truncated, info, _seconds = results[seed]
            self.observations[seed] = obs
            self.phases[seed] = info.get("phase")
            if terminated or truncated:
                del self.observations[seed]
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(1)
    payload = load_checkpoint_payload(args.checkpoint, "cpu")
    config = config_from_dict(payload["cohort_ppo_config"])
    data = load_game_data()
    vocab = build_vocab(data)
    model, _reward, _payload = load_policy(args.checkpoint, vocab, config.device)
    envs = EvaluationEnvs(
        config.cohort_workers,
        data=data,
        vocab=vocab,
        deck_key=config.deck_key,
        stake=config.stake,
        win_ante=config.win_ante,
        max_steps=config.max_no_progress_steps,
        reward_config=config.reward_config,
        enable_teacher=False,
        excluded_seeds=tuple(training_seed_reservations(config)),
    )
    recorded = RecordingEnvs(envs)
    cursor = payload["cohort_state"]["seed_cursor"]
    seeds = config.cohort_seeds[cursor : cursor + config.cohort_games]
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_hashes": source_hashes(),
        "device": config.device,
        "seeds": seeds,
        "update": payload["update_count"],
        "training": False,
    }
    print(f"Replaying {len(seeds)} games at update {payload['update_count']} on {config.device}", flush=True)
    try:
        restore_rng_states(payload["rng_states"])
        episodes, _ = collect_complete_cohort(
            model,
            recorded,
            config,
            seeds,
            payload["cohort_state"]["slot_counters"],
            policy_version=payload["update_count"],
        )
        report.update(status="complete", outcomes=[e.outcome for e in episodes])
    except CohortFailure as error:
        report.update(status="failed", error=str(error), diagnostics=error.diagnostics)
        report["unfinished"] = {}
        for seed, obs in recorded.observations.items():
            np.savez_compressed(args.output / f"observation_{seed}.npz", **obs)
            report["unfinished"][str(seed)] = {
                "actions": recorded.actions[seed],
                "phase": recorded.phases[seed],
                "legal_actions": np.flatnonzero(obs["action_mask"]).tolist(),
            }
    finally:
        envs.close()
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / "result.json", report)
    print({k: report[k] for k in ("status", "elapsed_seconds")}, flush=True)


if __name__ == "__main__":
    main()
