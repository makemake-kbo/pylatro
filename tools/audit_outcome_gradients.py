#!/usr/bin/env python3
"""CPU gradient attribution on replayed student games, without changing the policy."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.env import BalatroEnv
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, actor_digest, outcome_reward
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch
from pylatro_agent.training.ppo_optimization import _per_state_normalized_entropy
from pylatro_agent.training.ppo_policy import _grammar_distribution
from pylatro_agent.transfer import load_policy, sha256, source_hashes
from pylatro_agent.value_head import outcome_nll, return_huber_loss
from pylatro_agent.vocab import build_vocab


def gradient_summary(names, actor, returns, outcomes):
    result = {}
    for group in ("all", "embedding", "backbone", "action_grammar_head", "value_head"):
        indices = [index for index, name in enumerate(names) if group == "all" or name.startswith(group + ".")]
        aa = rr = oo = cc = ac = combined = 0.0
        for index in indices:
            a, r, o = actor[index].double(), returns[index].double(), outcomes[index].double()
            c = r + o
            aa += float(a.square().sum())
            rr += float(r.square().sum())
            oo += float(o.square().sum())
            cc += float(c.square().sum())
            ac += float((a * c).sum())
            combined += float((a + c).square().sum())
        result[group] = dict(
            actor_norm=math.sqrt(aa),
            weighted_return_gradient_norm=math.sqrt(rr),
            weighted_outcome_gradient_norm=math.sqrt(oo),
            critic_norm=math.sqrt(cc),
            combined_norm=math.sqrt(combined),
            critic_to_actor_norm=math.sqrt(cc / aa) if aa else None,
            actor_critic_cosine=ac / math.sqrt(aa * cc) if aa * cc else None,
        )
    result["global_clip_multiplier"] = min(1.0, 0.5 / (result["all"]["combined_norm"] + 1e-6))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--examples", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--name", default="gradient_audit_initial")
    args = parser.parse_args()
    if min(args.games, args.examples, args.batch) < 1:
        parser.error("Positive game, example and batch counts required")
    torch.set_num_threads(4)
    root = args.root
    checkpoint = args.checkpoint or root / "critic_seed0/checkpoint.pt"
    destination = root / args.name
    destination.mkdir(exist_ok=False)
    started = time.monotonic()
    protocol = read_json(root / "manifest.json")
    seeds = protocol["diagnostic_seeds"][:args.games]
    if len(seeds) != args.games:
        raise ValueError("Diagnostic game panel exceeded")
    traces = root / "phase_oracle_cpu/games"
    provenance = []
    reservoir = []
    seen = 0
    rng = np.random.default_rng(1729)
    data = load_game_data()
    vocab = build_vocab(data)
    # Reservoir selection is uniform over every decision in the specified games.
    for seed in seeds:
        for sampling in (0, 1, 2):
            path = traces / f"student_{seed}_sample{sampling}.json"
            trace_path = path.with_suffix(".jsonl.gz")
            outcome = read_json(path)
            if sha256(trace_path) != outcome["trace_sha256"] or outcome["stalled"] or outcome["censored"]:
                raise ValueError("Incomplete or modified source trajectory")
            provenance.append(dict(path=str(path), sha256=sha256(path), trace_sha256=sha256(trace_path)))
            env = BalatroEnv(seed=seed, data=data, vocab=vocab, enable_teacher=False, deck_key="b_blue",
                             stake=1, win_ante=8, max_steps=256, reward_config=outcome_reward())
            try:
                obs, _ = env.reset(seed=seed)
                rewards = 0.0
                with gzip.open(trace_path, "rt") as handle:
                    for step, line in enumerate(handle):
                        record = json.loads(line)
                        if step != record["step"] or not obs["action_mask"][record["action"]]:
                            raise ValueError("Recorded student action is unreachable on replay")
                        if str(env._sub_phase) != record["phase"] or int(obs["scalars"][2]) != record["ante"]:
                            raise ValueError("Recorded phase or ante diverged")
                        seen += 1
                        index = seen - 1 if seen <= args.examples else int(rng.integers(seen))
                        if index < args.examples:
                            row = dict(obs=copy.deepcopy(obs), seed=seed, sampling_seed=sampling, step=step,
                                       phase=record["phase"], action=record["action"], won=outcome["won"],
                                       target=8 if outcome["won"] else outcome["final_ante"] - 1)
                            if index == len(reservoir):
                                reservoir.append(row)
                            else:
                                reservoir[index] = row
                        obs, reward, terminated, truncated, info = env.step(record["action"])
                        rewards += reward
                if not terminated or truncated or info["won"] != outcome["won"] or step + 1 != outcome["steps"]:
                    raise ValueError("Student trajectory terminal result diverged")
                if rewards != 10 * outcome["won"]:
                    raise ValueError("Replayed reward is not the pure outcome target")
            finally:
                env.close()
    if len(reservoir) != args.examples:
        raise ValueError("Insufficient decisions for the requested sample")
    atomic_json(destination / "selection.json", dict(
        population_decisions=seen, selected_decisions=len(reservoir), sampling_rng_seed=1729,
        source_games=provenance,
        rows=[{key: value for key, value in row.items() if key != "obs"} for row in reservoir],
    ))
    print(f"Replayed {len(provenance)} whole games, selected {len(reservoir)} of {seen} decisions", flush=True)
    model, reward, _ = load_policy(checkpoint, vocab, torch.device("cpu"), mixture=0.5)
    if reward != outcome_reward():
        raise ValueError("Gradient audit requires the outcome-only critic")
    before = actor_digest(model)
    predicted = []
    entropy_rows = []
    with torch.no_grad():
        for start in range(0, len(reservoir), args.batch):
            rows = reservoir[start:start + args.batch]
            batch = _obs_dicts_to_batch([row["obs"] for row in rows], torch.device("cpu"))
            dist, values = _grammar_distribution(model, batch)
            predicted.append(values["expected_return"])
            logits = dist._t(dist.output.macro_logits).masked_fill(~dist.macro_mask, -1e9)
            probabilities = logits.softmax(-1)
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
            counts = dist.macro_mask.sum(-1)
            normalized = torch.where(counts > 1, entropy / counts.clamp_min(2).float().log(), 0)
            for row, count, value in zip(rows, counts.tolist(), normalized.tolist(), strict=True):
                entropy_rows.append(dict(phase=row["phase"], legal_families=count, entropy=value))
    targets = torch.tensor([10.0 * row["won"] for row in reservoir])
    raw_advantages = targets - torch.cat(predicted)
    advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std(unbiased=False) + 1e-8)
    named = list(model.named_parameters())
    names, parameters = zip(*named, strict=True)
    accumulated = [[torch.zeros_like(parameter) for parameter in parameters] for _ in range(3)]
    for start in range(0, len(reservoir), args.batch):
        rows = reservoir[start:start + args.batch]
        batch = _obs_dicts_to_batch([row["obs"] for row in rows], torch.device("cpu"))
        dist, values = _grammar_distribution(model, batch)
        actions = torch.tensor([row["action"] for row in rows])
        log_probs = dist.log_prob(actions)
        ratio = (log_probs - log_probs.detach()).exp()
        actor_loss = -(ratio * advantages[start:start + len(rows)]).mean()
        actor_loss -= 0.001 * _per_state_normalized_entropy(dist.entropy(), batch["action_mask"]).mean()
        return_loss = 0.25 * return_huber_loss(values, targets[start:start + len(rows)])
        terminal_loss = 0.1 * outcome_nll(
            values["outcome_probabilities"], torch.tensor([row["target"] for row in rows])
        )
        for component, loss in enumerate((actor_loss, return_loss, terminal_loss)):
            grads = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=component < 2)
            for accumulator, grad in zip(accumulated[component], grads, strict=True):
                if grad is not None:
                    accumulator.add_(grad, alpha=len(rows) / len(reservoir))
        print(f"Gradient rows {min(start + len(rows), len(reservoir))}/{len(reservoir)}", flush=True)
    if actor_digest(model) != before:
        raise ValueError("Read-only audit modified actor parameters")
    entropy_report = {}
    for phase in ["all", *sorted({row["phase"] for row in entropy_rows})]:
        rows = entropy_rows if phase == "all" else [row for row in entropy_rows if row["phase"] == phase]
        choices = [row["entropy"] for row in rows if row["legal_families"] > 1]
        entropy_report[phase] = dict(rows=len(rows), forced_family_rows=sum(row["legal_families"] == 1 for row in rows),
                                     all_row_mean=float(np.mean([row["entropy"] for row in rows])),
                                     choice_row_mean=float(np.mean(choices)) if choices else None)
    result = dict(
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), source_hashes=source_hashes(),
        selection_sha256=sha256(destination / "selection.json"), population_games=len(provenance),
        population_decisions=seen, sampled_decisions=len(reservoir),
        selected_game_counts=dict(Counter(str(r["seed"]) for r in reservoir)),
        gradient=gradient_summary(names, *accumulated), normalized_macro_entropy=entropy_report,
        raw_advantage_mean=float(raw_advantages.mean()), raw_advantage_std=float(raw_advantages.std(unbiased=False)),
        actor_unchanged=True, device="cpu", accelerator_seconds=0, elapsed_seconds=time.monotonic() - started,
        interpretation=("Gradient attribution at the initial policy on a uniform decision sample from fixed, complete "
                        "student diagnostic games. Uses Monte Carlo advantages normalized over this sample and known "
                        "terminal targets for all rows, not an actual fragmented PPO rollout. Clipping is inactive at "
                        "ratio one. Adam preconditioning and subsequent updates can change the effect. Norms or cosine "
                        "alone do not establish interference as a gameplay cause. "
                        "No teacher labels or optimizer steps."),
    )
    atomic_json(destination / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
