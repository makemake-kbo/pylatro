#!/usr/bin/env python3
"""Bounded CPU/MPS probability, gradient, and gameplay check for a subset checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.constants import ActionRange
from pylatro_agent.env import BalatroEnv
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch
from pylatro_agent.training.ppo_policy import _grammar_distribution as ppo_distribution
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import evaluate_artifact, load_policy, sha256, source_hashes, summarize
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--observation-probe", type=Path,
                        help="Additional replay-verified states for the entity-sale and hand heads")
    args = parser.parse_args()
    root = args.root
    marker = root / "subset_backend_verification.json"
    if marker.exists():
        raise FileExistsError("Backend verification already exists; inspect it before repeating")
    torch.set_num_threads(4)
    data = load_game_data()
    vocab = build_vocab(data)
    seeds = read_json(root / "manifest.json")["development_seeds"][:16]
    rows = []
    for index, seed in enumerate(seeds[:4]):
        env = BalatroEnv(seed=seed, data=data, vocab=vocab, enable_teacher=False, deck_key="b_blue")
        obs, _ = env.reset()
        if index:
            obs, *_ = env.step(int(ActionRange.BLIND_PLAY))
        rows.append(
            dict(obs=obs, action=int(np.flatnonzero(obs["action_mask"])[0]), return_target=0.0, won=False, max_ante=1)
        )
        env.close()
    probe_metadata = None
    if args.observation_probe:
        expected_probe = read_json(args.observation_probe.with_suffix(".json"))
        if sha256(args.observation_probe) != expected_probe["cache_sha256"]:
            raise ValueError("The replay-verified observation cache changed")
        probe = torch.load(args.observation_probe, map_location="cpu", weights_only=False)
        probe_metadata = probe["metadata"]
        if probe_metadata != {key: value for key, value in expected_probe.items() if key != "cache_sha256"}:
            raise ValueError("Observation-cache metadata differs from its replay record")
        core = {key: value for key, value in source_hashes().items() if key.startswith("src/")}
        digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
        if probe_metadata["source_core_sha256"] != digest or len(probe["rows"]) != 12:
            raise ValueError("The observation probe uses different core code or lacks the declared coverage")
        if {category: sum(row["category"] == category for row in probe["rows"])
                for category in ("play", "discard", "sale")} != dict(play=4, discard=4, sale=4):
            raise ValueError("The observation probe lacks hand and multi-Joker sale coverage")
        for row in probe["rows"]:
            if not row["obs"]["action_mask"][row["action"]]:
                raise ValueError("A replay-probe action is illegal")
            # The supervised collator requires these placeholders; this check
            # computes only action log probabilities, never a critic loss.
            rows.append(dict(obs=row["obs"], action=row["action"], return_target=0., won=False, max_ante=1))
    model, reward, _ = load_policy(args.checkpoint, vocab, "cpu", mixture=0.5)
    if model.config.action_head_variant not in ("subset_residual_v1", "entity_sale_subset_residual_v1"):
        raise ValueError("Backend probe requires a subset architecture")
    batch = _collate_batch(rows, torch.device("cpu"), SupervisedConfig(reward_config=reward))
    cpu, _ = _grammar_distribution(model, batch)
    cpu_logp = cpu.log_prob(batch["actions"])
    with torch.no_grad():
        ppo, _ = ppo_distribution(model, _obs_dicts_to_batch([row["obs"] for row in rows], torch.device("cpu")))
        ppo_logp = ppo.log_prob(batch["actions"])
        torch.testing.assert_close(ppo_logp, cpu_logp, rtol=1e-6, atol=1e-6)
    def gradient_parameters(agent):
        params = {"subset": agent.action_grammar_head.subset_residual.score[-1].weight}
        if args.observation_probe:
            params["sale"] = agent.action_grammar_head.sale_residual.net[-1].weight
        return params

    cpu_gradients = torch.autograd.grad(-cpu_logp.mean(), tuple(gradient_parameters(model).values()))
    with accelerator_job(root, "subset_backend_verification", 120):
        mps_model = copy.deepcopy(model).to("mps")
        mps_batch = {key: value.to("mps") for key, value in batch.items()}
        mps, _ = _grammar_distribution(mps_model, mps_batch)
        mps_logp = mps.log_prob(mps_batch["actions"])
        mps_gradients = torch.autograd.grad(-mps_logp.mean(), tuple(gradient_parameters(mps_model).values()))
        torch.testing.assert_close(mps_logp.cpu(), cpu_logp, rtol=3e-4, atol=3e-4)
        gradient_errors = {}
        for name, mps_gradient, cpu_gradient in zip(
                gradient_parameters(model), mps_gradients, cpu_gradients, strict=True):
            torch.testing.assert_close(mps_gradient.cpu(), cpu_gradient, rtol=2e-3, atol=3e-4)
            gradient_errors[name] = dict(max_absolute_error=float((mps_gradient.cpu() - cpu_gradient).abs().max()),
                                         cpu_norm=float(cpu_gradient.norm()), mps_norm=float(mps_gradient.norm()))
        errors = {}
        for family in (True, False):
            left, right = (
                cpu._hand_probabilities(is_play=family).detach(),
                mps._hand_probabilities(is_play=family).detach().cpu(),
            )
            torch.testing.assert_close(left, right, rtol=3e-4, atol=3e-5)
            errors["play" if family else "discard"] = float((left - right).abs().max())
        with torch.inference_mode():
            if not torch.equal(cpu.mode(), mps.mode().cpu()):
                raise ValueError("CPU and MPS greedy decisions disagree on the bounded probe")
        started = time.monotonic()
        outcomes = evaluate_artifact(
            mps_model,
            reward,
            data,
            vocab,
            seeds,
            torch.device("mps"),
            root / "evaluations" / "subset_backend_smoke",
            greedy=False,
            sampling_seed=0,
            metadata={
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": sha256(args.checkpoint),
                "source_hashes": source_hashes(),
                "scope": "backend and throughput smoke; not a strength comparison",
            },
        )
        if any(row.get("stalled") or row.get("censored") for row in outcomes):
            raise ValueError("Backend probe has incomplete games")
        report = dict(
            passed=True,
            checkpoint_sha256=sha256(args.checkpoint),
            architecture=model.config.action_head_variant,
            supervised_ppo_logp_max_absolute_error=float((ppo_logp - cpu_logp.detach()).abs().max()),
            action_logp_max_absolute_error=float((mps_logp.detach().cpu() - cpu_logp.detach()).abs().max()),
            hand_probability_max_absolute_errors=errors,
            gradient_max_absolute_error=gradient_errors["subset"]["max_absolute_error"],
            gradient_comparison=gradient_errors,
            observation_count=len(rows),
            observation_probe=(dict(path=str(args.observation_probe), sha256=sha256(args.observation_probe),
                                    metadata=probe_metadata) if args.observation_probe else None),
            gameplay_games=len(outcomes),
            gameplay_seconds=time.monotonic() - started,
            gameplay=summarize(outcomes),
            source_hashes=source_hashes(),
            torch_version=torch.__version__,
            interpretation=(
                "CPU/MPS implementation verification only; these 16 development games do not establish a gain"
            ),
        )
        atomic_json(marker, report)
        print({key: value for key, value in report.items() if key not in ("source_hashes", "gameplay")}, flush=True)


if __name__ == "__main__":
    main()
