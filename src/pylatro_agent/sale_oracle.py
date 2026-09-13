"""Exploratory sale-conditioned shop interventions; original study paths stay fixed."""

import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from . import phase_oracle as base
from .action import ActionType, decode_action
from .constants import SubPhase
from .env import BalatroEnv
from .heuristic import HeuristicAgent
from .phase_oracle import isolated_teacher_action, policy_seed
from .training.ppo_observations import _single_obs_to_batch
from .training.ppo_policy import _grammar_distribution
from .transfer import sha256, write_json

SALE_ARMS = ("teacher_sale", "teacher_other_shop")


def replace_shop_action(arm, phase, teacher_action):
    if arm not in SALE_ARMS:
        raise ValueError(f"Unknown sale-oracle arm: {arm}")
    if phase != SubPhase.SHOP:
        return False
    if teacher_action is None:
        raise ValueError("A SHOP decision requires an isolated teacher proposal")
    is_sale = decode_action(teacher_action).action_type == ActionType.SHOP_SELL_JOKER
    return is_sale if arm == "teacher_sale" else not is_sale


def run_game(task):
    arm, seed, sampling, destination = task
    folder = Path(destination)
    name = f"{arm}_{seed}_sample{sampling}"
    outcome_path = folder / f"{name}.json"
    trace_path = folder / f"{name}.jsonl.gz"
    if outcome_path.exists():
        result = json.loads(outcome_path.read_text())
        if result["trace_sha256"] != sha256(trace_path):
            raise RuntimeError("Completed sale-oracle trace changed")
        return result
    if trace_path.exists():
        raise RuntimeError("Interrupted sale-oracle trace requires inspection")
    rng_seed = policy_seed(seed, sampling if sampling is not None else 0)
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(rng_seed).get_state())
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    env = BalatroEnv(
        seed=seed,
        data=base._data,
        vocab=base._vocab,
        enable_teacher=False,
        deck_key="b_blue",
        stake=1,
        win_ante=8,
        max_steps=256,
        reward_config=base._reward,
    )
    teacher = HeuristicAgent(shop_policy="search", grow_scalers=True)
    started = time.monotonic()
    teacher_steps = disagreements = steps = 0
    try:
        obs, _ = env.reset(seed=seed)
        with torch.inference_mode(), gzip.open(trace_path, "xt") as trace:
            while True:
                phase = env._sub_phase
                student = None
                # Always draw the student's proposal in hybrid arms, even when replaced.
                dist, _ = _grammar_distribution(base._model, _single_obs_to_batch(obs, torch.device("cpu")))
                sampled = dist.sample()
                if not torch.isfinite(dist.log_prob(sampled)).all():
                    raise RuntimeError("Nonfinite student policy output")
                student = int(sampled.item())
                if not obs["action_mask"][student]:
                    raise RuntimeError("Illegal student proposal")
                teacher_action = isolated_teacher_action(teacher, env) if phase == SubPhase.SHOP else None
                controlled = replace_shop_action(arm, phase, teacher_action)
                action = teacher_action if controlled else student
                teacher_steps += int(controlled)
                disagreements += int(controlled and student is not None and action != student)
                trace.write(
                    json.dumps(
                        dict(
                            seed=seed,
                            step=steps,
                            phase=str(phase),
                            ante=int(obs["scalars"][2]),
                            action=action,
                            student_proposal=student,
                            teacher_proposal=teacher_action,
                            source="teacher" if controlled else "student",
                        )
                    )
                    + "\n"
                )
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated or truncated:
                    result = dict(
                        arm=arm,
                        seed=seed,
                        sampling_seed=sampling,
                        policy_rng_seed=rng_seed,
                        won=bool(info["won"]),
                        max_ante=info["ante"],
                        final_ante=info["ante"],
                        steps=steps,
                        teacher_steps=teacher_steps,
                        teacher_replacements=disagreements,
                        stalled=bool(info.get("stalled")),
                        censored=bool(truncated),
                        cleared_antes=sorted(env._cleared_boss_antes | ({8} if info["won"] else set())),
                        elapsed_seconds=time.monotonic() - started,
                    )
                    if result["stalled"] or result["censored"]:
                        raise RuntimeError(f"Incomplete sale-oracle game: {result}")
                    break
                if steps > 10000:
                    raise RuntimeError("Phase-oracle trajectory exceeded its bounded step limit")
        result["trace_sha256"] = sha256(trace_path)
        write_json(outcome_path, result)
        return result
    finally:
        env.close()
