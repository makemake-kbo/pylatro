#!/usr/bin/env python3
"""Rebuild supervised records from complete, verified heuristic action traces.

No policy decisions are resampled. Every episode (including losses) is retained;
this exports existing development games, not a fresh win-rate evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.reward import DEFAULT_REWARD_CONFIG
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_generate import _run_game_single_pass
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.training.model_generate import load_records, save_records
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch
from pylatro_agent.vocab import build_vocab


def action_id(event):
    typ, index, detail = re.search(
        r"ActionType\.(\w+):.*index=(\d+), detail=(\d+)", event["action"]
    ).groups()
    return encode_action(ActionType[typ], int(index), int(detail))


class ReplayAgent:
    _shop_policy = "search"
    _grow_scalers = True

    def __init__(self, row):
        self.row = row
        self.position = 0

    def select_action(self, state, sub_phase, mask, *, round_score):
        event = self.row["events"][self.position]
        assert event["ante"] == state.round_resets.ante
        assert event["blind"] == state.blind_on_deck
        assert event["score"] == round_score
        assert event["dollars"] == state.dollars
        assert [j["key"] for j in event["jokers"]] == list(state.joker_keys)
        action = action_id(event)
        assert mask[action], (self.row["seed"], self.position, "illegal action")
        self.position += 1
        return action


def audit(row, data):
    runner = FastRunner(row["seed"], data, deck_key="b_blue", raise_errors=True)
    agent = ReplayAgent(row)
    hands, late_hands, actions = Counter(), Counter(), Counter()
    for event in row["events"]:
        action = agent.select_action(runner.state, runner.sub_phase,
                                     runner.compute_mask(), round_score=runner.round_score)
        result = runner.step(action)
        typ = re.search(r"ActionType\.(\w+):", event["action"])[1]
        actions[typ] += 1
        if typ == "PLAY_SUBSET":
            hands[result.score.hand_name] += 1
            if event["ante"] >= 6:
                late_hands[result.score.hand_name] += 1
    assert runner.done and runner.won == row["won"]
    assert runner.round_score == row["final"]["score"]
    return dict(seed=row["seed"], won=runner.won, steps=len(row["events"]),
                max_ante=runner.max_ante, terminal_ante=runner.state.round_resets.ante,
                played_hands=dict(hands), late_played_hands=dict(late_hands),
                actions=dict(actions), final_boss=row["final"]["boss"],
                final_jokers=sorted(j["key"] for j in row["final"]["jokers"]))


def export_shard(task):
    index, rows, output = task
    torch.set_num_threads(1)
    data = load_game_data()
    tokenizer = Tokenizer(vocab=build_vocab(data))
    records, audits = [], []
    for row in rows:
        audits.append(audit(row, data))
        agent = ReplayAgent(row)
        episode, ante, won = _run_game_single_pass(
            row["seed"], data, tokenizer, agent, .997,
            reward_config=DEFAULT_REWARD_CONFIG, deck_key="b_blue", stake=1,
        )
        assert agent.position == len(row["events"]) == len(episode)
        assert won == row["won"]
        # The benchmark's final snapshot precedes the last action. A victory
        # cash-out can advance the terminal state to Ante 9; compare the two
        # completed runners, rather than this pre-action Ante 8 snapshot.
        assert ante == audits[-1]["max_ante"], (row["seed"], ante, audits[-1])
        for record in episode:
            assert record["obs"]["action_mask"][record["action"]]
            assert record["terminal_outcome_mask"] == 1
            assert np.isfinite(record["return_target"])
            for value in record["obs"].values():
                assert np.isfinite(value).all()
            record["teacher_action_source"] = "archived_v68_trace"
        records.extend(episode)
        print(f"shard {index:03d}: seed {row['seed']}, {len(episode)} records verified", flush=True)
    path = Path(output) / f"shard_{index:03d}.pkl"
    save_records(records, path, reward_config=DEFAULT_REWARD_CONFIG)
    count = len(records)
    del records
    loaded = load_records(path, reward_config=DEFAULT_REWARD_CONFIG)
    assert len(loaded) == count
    # Exercise the actual training collator across every episode in the shard.
    batch = _collate_batch(loaded[::max(1, count // 32)], torch.device("cpu"), SupervisedConfig())
    assert all(torch.isfinite(value).all() for value in batch.values())
    return dict(file=path.name, records=count, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), games=audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reserved-seeds", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games-per-shard", type=int, default=10)
    args = parser.parse_args()
    if args.workers < 1 or args.games_per_shard < 1:
        parser.error("workers and games-per-shard must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for path in args.traces for line in path.read_text().splitlines()]
    seeds = [r["seed"] for r in rows]
    assert len(seeds) == len(set(seeds))
    reserved = json.loads(args.reserved_seeds.read_text())
    if isinstance(reserved, dict):
        reserved = reserved["seeds"]
    assert not set(seeds).intersection(reserved)
    assert not set(seeds).intersection(range(10000, 11000))
    root = Path(__file__).resolve().parents[1]
    sources = [path for package in ("pylatro", "pylatro_agent", "pylatro_cli")
               for path in (root / "src" / package).rglob("*")
               if path.suffix in {".py", ".so", ".json"}]
    sources.append(Path(__file__).resolve())
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    profile = dict(policy="V68", deck="b_blue", stake=1, win_ante=8, gamma=.997,
                   shop_policy="search", grow_scalers=True, blind_rollout=None,
                   teacher_information="full_simulator_state_with_current_score_RNG_oracle",
                   source="replay of complete development games; not fresh evaluation",
                   outcome_filter=None, seeds=seeds,
                   reserved_seeds_sha256=hashlib.sha256(args.reserved_seeds.read_bytes()).hexdigest(),
                   traces={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.traces},
                   implementation=manifest)
    (args.output / "manifest.json").write_text(json.dumps(profile, indent=2) + "\n")
    with tarfile.open(args.output / "sources.tar.gz", "w:gz") as archive:
        for path in sources:
            if path.suffix != ".so":
                archive.add(path, arcname=str(path.relative_to(root)))
    load_game_data()
    tasks = [(i, rows[start:start + args.games_per_shard], str(args.output))
             for i, start in enumerate(range(0, len(rows), args.games_per_shard))]
    shards = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(export_shard, task) for task in tasks]):
            result = future.result()
            shards.append(result)
            (args.output / "progress.json").write_text(json.dumps(shards, indent=2) + "\n")
            print(f"{len(shards)}/{len(tasks)} shards; {sum(s['records'] for s in shards)} verified records", flush=True)
    games = [g for s in shards for g in s["games"]]
    wins = [g for g in games if g["won"]]
    def combined(field, subset):
        counter = Counter()
        for game in subset:
            counter.update(game[field])
        return dict(counter.most_common())
    summary = dict(games=len(games), wins=len(wins), losses=len(games)-len(wins),
                   records=sum(s["records"] for s in shards), bytes=sum(s["bytes"] for s in shards),
                   unique_winning_final_joker_sets=len({tuple(g["final_jokers"]) for g in wins}),
                   winning_joker_presence=combined("final_jokers", wins),
                   winning_bosses=dict(Counter(g["final_boss"] for g in wins)),
                   winning_played_hands=combined("played_hands", wins),
                   winning_late_played_hands=combined("late_played_hands", wins),
                   winning_dominant_late_hand=dict(Counter(max(g["late_played_hands"], key=g["late_played_hands"].get) for g in wins)),
                   actions=combined("actions", games),
                   validation="all actions legal, finite observations/returns, terminal replay parity, every shard reloaded and collated",
                   shards=sorted(shards, key=lambda s: s["file"]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "shards"}), flush=True)


if __name__ == "__main__":
    main()
