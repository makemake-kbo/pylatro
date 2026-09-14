#!/usr/bin/env python3
"""Prepare bounded, replay-verified observations for the final CPU/MPS head comparison."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.constants import ActionRange
from pylatro_agent.env import BalatroEnv
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, outcome_reward
from pylatro_agent.transfer import sha256, source_hashes
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    destination = root / "backend_observations.pt"
    if destination.exists():
        raise FileExistsError(destination)
    data = load_game_data()
    vocab = build_vocab(data)
    rows, provenance = [], []
    seen = {name: set() for name in ("play", "discard", "sale")}
    bounds = dict(play=(ActionRange.PLAY_SUBSET_START, ActionRange.PLAY_SUBSET_END),
                  discard=(ActionRange.DISCARD_SUBSET_START, ActionRange.DISCARD_SUBSET_END),
                  sale=(ActionRange.SHOP_SELL_JOKER_START, ActionRange.SHOP_SELL_JOKER_END))
    counts = dict(play=4, discard=4, sale=2)
    decisions = 0
    for seed in read_json(root / "manifest.json")["diagnostic_seeds"][:12]:
        path = root / "phase_oracle_cpu/games" / f"student_{seed}_sample0.json"
        trace = path.with_suffix(".jsonl.gz")
        outcome = read_json(path)
        if outcome["trace_sha256"] != sha256(trace) or outcome["stalled"] or outcome["censored"]:
            raise ValueError("A source student trajectory changed or is incomplete")
        env = BalatroEnv(seed=seed, data=data, vocab=vocab, enable_teacher=False, deck_key="b_blue",
                         stake=1, win_ante=8, max_steps=256, reward_config=outcome_reward())
        try:
            obs, _ = env.reset(seed=seed)
            rewards = 0.0
            terminated = truncated = False
            steps = 0
            with gzip.open(trace, "rt") as handle:
                for step, line in enumerate(handle):
                    record = json.loads(line)
                    if (terminated or truncated or step != record["step"]
                            or not obs["action_mask"][record["action"]]
                            or record["phase"] != str(env._sub_phase)
                            or record["ante"] != int(obs["scalars"][2])):
                        raise ValueError("Recorded action, phase, ante or terminal state diverged")
                    for category, (start, end) in bounds.items():
                        if seed in seen[category] or len(seen[category]) >= counts[category]:
                            continue
                        legal = np.flatnonzero(obs["action_mask"][start:end + 1]) + start
                        if len(legal) < (2 if category == "sale" else 1):
                            continue
                        actions = [int(legal[0]), int(legal[-1])] if category == "sale" else [int(legal[0])]
                        for action in actions:
                            rows.append(dict(obs=copy.deepcopy(obs), action=action, seed=seed, step=step,
                                             recorded_action=record["action"], category=category))
                        seen[category].add(seed)
                    # Probe actions are never executed or assigned critic targets.
                    obs, reward, terminated, truncated, info = env.step(record["action"])
                    rewards += reward
                    steps += 1
            if (not terminated or truncated or info["won"] != outcome["won"]
                    or steps != outcome["steps"] or rewards != 10 * outcome["won"]):
                raise ValueError("Whole-game replay differs from its recorded terminal result")
            decisions += steps
            provenance.append(dict(path=str(path), sha256=sha256(path), trace_sha256=sha256(trace),
                                   replayed_decisions=steps))
        finally:
            env.close()
        if all(len(seen[name]) == count for name, count in counts.items()):
            break
    if len(rows) != 12 or any(len(seen[name]) != count for name, count in counts.items()):
        raise ValueError("The bounded student panel did not cover play, discard and multi-Joker sales")
    core = {key: value for key, value in source_hashes().items() if key.startswith("src/")}
    metadata = dict(source_core_sha256=hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest(),
                    source_games=provenance, replayed_decisions=decisions, observation_count=len(rows),
                    rows=[{key: value for key, value in row.items() if key != "obs"} for row in rows],
                    source_sha256=sha256(__file__), accelerator_seconds=0,
                    scope="Read-only backend observations from whole student-game replays. Four play, four discard, "
                          "and four alternative sales across two multi-Joker states. No critic targets or training.")
    torch.save(dict(metadata=metadata, rows=rows), destination)
    atomic_json(root / "backend_observations.json", dict(**metadata, cache_sha256=sha256(destination)))
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
