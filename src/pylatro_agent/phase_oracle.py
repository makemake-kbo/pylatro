"""Exploratory CPU phase replacements, separate from candidate confirmation."""

from __future__ import annotations

import copy
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .constants import SubPhase
from .env import BalatroEnv
from .heuristic import HeuristicAgent
from .training.ppo_observations import _single_obs_to_batch
from .training.ppo_policy import _grammar_distribution
from .transfer import load_policy, sha256, write_json
from .transfer_diagnostics import live_state_digest

ORACLE_ARMS = ("student", "hand", "shop", "hand_shop", "teacher_all")
ORACLE_SEEDS = tuple(range(5000000, 5000050))


def teacher_controls_phase(arm, phase):
    if arm not in ORACLE_ARMS:
        raise ValueError(f"Unknown phase-oracle arm: {arm}")
    return (
        arm == "teacher_all"
        or (arm in ("hand", "hand_shop") and phase == SubPhase.CHOOSE_ACTION)
        or (arm in ("shop", "hand_shop") and phase == SubPhase.SHOP)
    )


def policy_seed(game_seed, sampling_seed):
    return int(np.random.SeedSequence([game_seed, sampling_seed, 1729]).generate_state(1)[0])


def isolated_teacher_action(teacher, env):
    before = live_state_digest(env)
    state = copy.deepcopy(env.state, {id(env._data): env._data})
    action = int(
        teacher.select_action(
            state,
            env._sub_phase,
            env.action_masks(),
            round_score=env._controller.round_score,
        )
    )
    if live_state_digest(env) != before:
        raise RuntimeError("Teacher query changed the live game or RNG")
    if not env.action_masks()[action]:
        raise RuntimeError("Illegal phase-oracle teacher action")
    return action


def worker_init(checkpoint):
    from pylatro import load_game_data

    from .vocab import build_vocab

    global _model, _reward, _data, _vocab
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    _data = load_game_data()
    _vocab = build_vocab(_data)
    _model, _reward, _ = load_policy(checkpoint, _vocab, torch.device("cpu"), mixture=0.5)
    _model.eval()


def run_game(task):
    arm, seed, sampling, destination = task
    folder = Path(destination)
    name = f"{arm}_{seed}_sample{sampling}"
    outcome_path = folder / f"{name}.json"
    trace_path = folder / f"{name}.jsonl.gz"
    if outcome_path.exists():
        result = json.loads(outcome_path.read_text())
        if result["trace_sha256"] != sha256(trace_path):
            raise RuntimeError("Completed phase-oracle trace changed")
        return result
    if trace_path.exists():
        raise RuntimeError("Interrupted phase-oracle trace requires inspection")
    rng_seed = policy_seed(seed, sampling if sampling is not None else 0)
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(rng_seed).get_state())
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    env = BalatroEnv(
        seed=seed,
        data=_data,
        vocab=_vocab,
        enable_teacher=False,
        deck_key="b_blue",
        stake=1,
        win_ante=8,
        max_steps=256,
        reward_config=_reward,
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
                if arm != "teacher_all":
                    dist, _ = _grammar_distribution(_model, _single_obs_to_batch(obs, torch.device("cpu")))
                    sampled = dist.sample()
                    if not torch.isfinite(dist.log_prob(sampled)).all():
                        raise RuntimeError("Nonfinite student policy output")
                    student = int(sampled.item())
                    if not obs["action_mask"][student]:
                        raise RuntimeError("Illegal student proposal")
                controlled = teacher_controls_phase(arm, phase)
                action = isolated_teacher_action(teacher, env) if controlled else student
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
                        raise RuntimeError(f"Incomplete phase-oracle game: {result}")
                    break
                if steps > 10000:
                    raise RuntimeError("Phase-oracle trajectory exceeded its bounded step limit")
        result["trace_sha256"] = sha256(trace_path)
        write_json(outcome_path, result)
        return result
    finally:
        env.close()
