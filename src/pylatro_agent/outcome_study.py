"""Pinned outcome-only PPO study inputs and complete-game critic diagnostics."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .reward import RewardConfig

ROOT = Path("eval/outcome_ppo_20260913")
BASELINE = Path("eval/action_refactor_20260913/entity_sale_subset_residual_v1_seed0/checkpoint.pt")
DEVELOPMENT = list(range(190000, 190200))
CONFIRMATION = list(range(200000, 200400))
CRITIC_TRAIN = list(range(6100000, 6100256))
CRITIC_VALIDATION = list(range(6200000, 6200128))
DIAGNOSTIC = list(range(6300000, 6300050))


def outcome_reward():
    """Ten points for an Ante-8 win, zero otherwise; no strategic bonuses."""
    return RewardConfig(objective="milestone", milestone_reward_budget=0.0, gamma=1.0)


def actor_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith("value_head."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def complete_game_targets(won: bool, final_ante: int, length: int):
    if length <= 0 or not 1 <= final_ante <= 9:
        raise ValueError("Invalid completed trajectory")
    if not won and final_ante > 8:
        raise ValueError("A loss must end in Antes 1-8")
    return np.full(length, 10.0 * won, np.float32), np.full(length, 8 if won else final_ante - 1, np.int64)


def pooled_critic_inputs(args, kwargs):
    """The exact masked pooling used by ValueHead; caller controls detachment."""
    encoded, attention = args[:2]
    mask = attention.unsqueeze(-1).to(encoded.dtype)
    return (encoded * mask).sum(1) / mask.sum(1).clamp(min=1)


def critic_from_pooled(head, pooled, antes):
    """Use the production critic with a single already-pooled token."""
    return head(pooled[:, None, :], torch.ones_like(pooled[:, :1]), antes, torch.full_like(antes, 8))


def critic_metrics(head, dataset, *, batch=2048):
    import torch.nn.functional as F

    outputs = []
    with torch.no_grad():
        for start in range(0, len(dataset["x"]), batch):
            outputs.append(
                critic_from_pooled(head, dataset["x"][start : start + batch], dataset["ante"][start : start + batch])
            )
    probabilities = torch.cat([item["outcome_probabilities"] for item in outputs])
    predicted = torch.cat([item["expected_return"] for item in outputs])
    targets, returns = dataset["target"], dataset["returns"]
    wins = (targets == 8).float()
    errors = returns - predicted
    variance = returns.var(unbiased=False)
    starts = dataset["step"] == 0
    one_hot = F.one_hot(targets, num_classes=9).float()
    nll = -probabilities.gather(1, targets[:, None]).clamp_min(1e-7).log().squeeze(1)
    return {
        "states": len(targets),
        "games": int(starts.sum()),
        "outcome_nll": float(nll.mean()),
        "outcome_brier": float((probabilities - one_hot).square().sum(-1).mean()),
        "win_brier": float((probabilities[:, -1] - wins).square().mean()),
        "return_mse": float(errors.square().mean()),
        "return_mae": float(errors.abs().mean()),
        "explained_variance": float(1 - errors.var(unbiased=False) / variance) if variance > 0 else None,
        "start_predicted_win_rate": float(probabilities[starts, -1].mean()),
        "start_observed_win_rate": float(wins[starts].mean()),
        "start_predicted_return": float(predicted[starts].mean()),
        "selection_loss": float(nll.mean() + 0.1 * errors.square().mean()),
    }


def study_protocol():
    return {
        "checkpoint": str(BASELINE),
        "development_seeds": DEVELOPMENT,
        "confirmation_seeds": CONFIRMATION,
        "critic_training_seeds": CRITIC_TRAIN,
        "critic_validation_seeds": CRITIC_VALIDATION,
        "diagnostic_seeds": DIAGNOSTIC,
        "reward_config": asdict(outcome_reward()),
        "teacher_action_supervision": False,
        "heuristic_reward_bonuses": False,
        "sampling_seeds": [0, 1, 2],
        "training_replicates": [0, 1],
        "architecture": "entity_sale_subset_residual_v1",
        "mixture": 0.5,
        "temperature": 1.0,
        "deck": "b_blue",
        "stake": 1,
        "win_ante": 8,
        "precision": "fp32",
        "accelerator_limit_seconds": 12 * 3600,
        "confirmation_reserve_seconds": 3 * 3600,
        "interpretation": (
            "25% is an intermediate result, not a target or a presumed ceiling; "
            "seek reliable large gains and identify unresolved failure mechanisms"
        ),
        "selection": (
            "Choose recipe and matched update count using development sampled wins across both replicas; "
            "freeze hashes before confirmation"
        ),
        "substantial_step": (
            "At least +10 percentage points sampled with positive paired 95% CI, both replicas positive, "
            "greedy regression no worse than five points; report absolute win rate even if this passes"
        ),
        "desired_direction": (
            "Policies that win a majority of sampled games; no guarantee this budget will reach that level"
        ),
    }
