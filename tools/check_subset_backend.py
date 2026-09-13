#!/usr/bin/env python3
"""Bounded CPU/MPS probability, gradient, and gameplay check for a subset checkpoint."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.constants import ActionRange
from pylatro_agent.env import BalatroEnv
from pylatro_agent.experiment_budget import accelerator_job
from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import evaluate_artifact, load_policy, sha256, source_hashes, summarize
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    model, reward, _ = load_policy(args.checkpoint, vocab, "cpu", mixture=0.5)
    if model.config.action_head_variant not in ("subset_residual_v1", "entity_sale_subset_residual_v1"):
        raise ValueError("Backend probe requires a subset architecture")
    batch = _collate_batch(rows, torch.device("cpu"), SupervisedConfig(reward_config=reward))
    cpu, _ = _grammar_distribution(model, batch)
    cpu_logp = cpu.log_prob(batch["actions"])
    cpu_gradient = torch.autograd.grad(-cpu_logp.mean(), model.action_grammar_head.subset_residual.score[-1].weight)[0]
    with accelerator_job(root, "subset_backend_verification", 120):
        mps_model = copy.deepcopy(model).to("mps")
        mps_batch = {key: value.to("mps") for key, value in batch.items()}
        mps, _ = _grammar_distribution(mps_model, mps_batch)
        mps_logp = mps.log_prob(mps_batch["actions"])
        mps_gradient = torch.autograd.grad(
            -mps_logp.mean(), mps_model.action_grammar_head.subset_residual.score[-1].weight
        )[0]
        torch.testing.assert_close(mps_logp.cpu(), cpu_logp, rtol=3e-4, atol=3e-4)
        torch.testing.assert_close(mps_gradient.cpu(), cpu_gradient, rtol=2e-3, atol=3e-4)
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
            action_logp_max_absolute_error=float((mps_logp.detach().cpu() - cpu_logp.detach()).abs().max()),
            hand_probability_max_absolute_errors=errors,
            gradient_max_absolute_error=float((mps_gradient.cpu() - cpu_gradient).abs().max()),
            observation_count=4,
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
