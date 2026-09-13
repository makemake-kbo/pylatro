"""CPU teacher queries and isolated snapshots for centrally batched student trajectories."""

from __future__ import annotations

import copy
import gzip
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .action import ActionType, decode_action
from .heuristic import HeuristicAgent
from .training.evaluation_envs import EvaluationEnvs, _Environments
from .training.model_generate import save_records
from .training.ppo_observations import _obs_dicts_to_batch
from .training.ppo_policy import _grammar_distribution
from .transfer import CORRECTION_SEEDS, sha256, write_json
from .transfer_training import correction_record


def query_teacher(self, seeds):
    if not hasattr(self, "_correction_teachers"):
        self._correction_teachers = {}
    result = []
    for seed in seeds:
        env = self.envs[seed]
        teacher = self._correction_teachers.setdefault(seed, HeuristicAgent(shop_policy="search", grow_scalers=True))
        state = copy.deepcopy(env.state, {id(env._data): env._data})
        action = int(
            teacher.select_action(state, env._sub_phase, env.action_masks(), round_score=env._controller.round_score)
        )
        if not env.action_masks()[action]:
            raise RuntimeError(f"Illegal teacher correction on seed {seed}")
        result.append((seed, action))
    return result


def save_diagnostic_snapshot(self, requests):
    for seed, path in requests:
        # This is a diagnostic any-phase copy, not an archive training reset.
        env = self.envs[seed]
        Path(path).write_bytes(gzip.compress(pickle.dumps(env, protocol=5)))
    return []


def install_collection_commands():
    _Environments.query_teacher = query_teacher
    _Environments.save_diagnostic_snapshot = save_diagnostic_snapshot


def _worker_with_collection(connection):
    from .training.evaluation_envs import _worker

    install_collection_commands()
    _worker(connection)


class CollectionEnvs(EvaluationEnvs):
    def __init__(self, workers, **kwargs):
        from .training import evaluation_envs

        original = evaluation_envs._worker
        evaluation_envs._worker = _worker_with_collection
        try:
            install_collection_commands()
            super().__init__(workers, **kwargs)
        finally:
            evaluation_envs._worker = original

    def command(self, name, requests, *, snapshot=False):
        if self.local is not None:
            return dict(getattr(self.local, name)(requests))
        shards = [[] for _ in self.processes]
        for request in requests:
            seed = request[0] if snapshot else request
            shards[self.owners[seed]].append(request)
        for index, shard in enumerate(shards):
            if shard:
                self.connections[index].send((name, shard))
        results = {}
        for index, shard in enumerate(shards):
            if shard:
                results.update(self._receive(index))
        return results


def collect_corrections(
    model,
    reward,
    data,
    vocab,
    checkpoint,
    device,
    output,
    *,
    seeds=CORRECTION_SEEDS,
    workers=4,
    batch_size=32,
    sampling_seed=0,
    max_idle_steps=256,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "snapshots").mkdir()
    torch.manual_seed(sampling_seed)
    if device.type == "mps":
        torch.mps.manual_seed(sampling_seed)
    model.eval()
    checkpoint_hash = sha256(checkpoint)
    teacher_hash = sha256(Path(__file__).parent / "heuristic.py")
    started = time.monotonic()
    rng = np.random.default_rng(1729)
    pending = iter(seeds)
    steps = {s: 0 for s in seeds}
    disagreements = {s: 0 for s in seeds}
    snapshots = {}
    records, outcomes, agreement = [], [], []
    buffer = []
    shard = 0
    envs = CollectionEnvs(
        workers,
        data=data,
        vocab=vocab,
        deck_key="b_blue",
        stake=1,
        win_ante=8,
        max_steps=max_idle_steps,
        reward_config=reward,
        enable_teacher=False,
    )
    try:
        slots = []
        for _ in range(min(batch_size, len(seeds))):
            seed = next(pending)
            obs, _ = envs.reset(seed)
            slots.append((seed, obs))
        with torch.inference_mode(), (output / "agreement.jsonl").open("x", buffering=1) as stream:
            while slots:
                batch = _obs_dicts_to_batch([obs for _, obs in slots], device)
                dist, _ = _grammar_distribution(model, batch)
                actions = dist.sample().cpu().tolist()
                modes = dist.mode().cpu().tolist()
                teachers = envs.command("query_teacher", [seed for seed, _ in slots])
                teacher_actions = torch.tensor([teachers[seed] for seed, _ in slots], device=device)
                logp = dist.log_prob(teacher_actions).cpu().tolist()
                candidates = {
                    kind: dist._candidate_distribution(is_play=kind)[2].cpu().numpy() for kind in (True, False)
                }
                snapshot_requests = []
                for i, (seed, obs) in enumerate(slots):
                    teacher = teachers[seed]
                    family = decode_action(teacher).action_type
                    is_play = family == ActionType.PLAY_SUBSET
                    is_hand = is_play or family == ActionType.DISCARD_SUBSET
                    row = correction_record(
                        obs,
                        teacher,
                        seed=seed,
                        step=steps[seed],
                        student_action=actions[i],
                        checkpoint_sha256=checkpoint_hash,
                        teacher_sources=teacher_hash,
                    )
                    buffer.append(row)
                    measured = {
                        "seed": seed,
                        "step": steps[seed],
                        "ante": int(obs["scalars"][2]),
                        "phase": int(obs["scalars"][7]),
                        "teacher_action": teacher,
                        "student_action": actions[i],
                        "greedy_agreement": modes[i] == teacher,
                        "sampled_agreement": actions[i] == teacher,
                        "teacher_probability": float(np.exp(logp[i])),
                        "nll": -logp[i],
                        "teacher_family": family.value,
                        "predicted_family": decode_action(modes[i]).action_type.value,
                        "accuracy": float(modes[i] == teacher),
                        "unreachable": float(logp[i] < -1e7),
                        "candidate_covered": float(teacher in candidates[is_play][i]) if is_hand else None,
                    }
                    agreement.append(measured)
                    stream.write(json.dumps(measured, allow_nan=False) + "\n")
                    if teacher != actions[i]:
                        disagreements[seed] += 1
                        if rng.random() < 1 / disagreements[seed]:
                            path = output / "snapshots" / f"{seed}.pkl.gz"
                            snapshot_requests.append((seed, str(path.resolve())))
                            snapshots[seed] = {
                                **measured,
                                "path": str(path.resolve()),
                                "continuation_sampling_seed": 500000 + seed,
                            }
                    steps[seed] += 1
                if snapshot_requests:
                    envs.command("save_diagnostic_snapshot", snapshot_requests, snapshot=True)
                results = envs.step([(seed, actions[i]) for i, (seed, _) in enumerate(slots)])
                next_slots = []
                for seed, _ in slots:
                    obs, terminated, truncated, info, _ = results[seed]
                    if terminated or truncated:
                        outcomes.append({"seed": seed, "won": bool(info["won"]), "final_ante": info["ante"]})
                        next_seed = next(pending, None)
                        if next_seed is not None:
                            obs, _ = envs.reset(next_seed)
                            next_slots.append((next_seed, obs))
                    else:
                        next_slots.append((seed, obs))
                slots = next_slots
                if len(buffer) >= 1000 or not slots:
                    save_records(buffer, output / f"corrections_{shard:03d}.parquet", reward_config=reward)
                    records.extend(buffer)
                    buffer = []
                    shard += 1
                    write_json(output / "snapshots.json", list(snapshots.values()))
                    print(f"Corrections: {len(records)} rows, {len(outcomes)}/{len(seeds)} games", flush=True)
    finally:
        envs.close()
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "student_checkpoint_sha256": checkpoint_hash,
            "seeds": list(seeds),
            "sampling_seed": sampling_seed,
            "batch_size": batch_size,
            "workers": workers,
            "device": str(device),
            "deck": "b_blue",
            "stake": 1,
            "win_ante": 8,
            "temperature": 1.0,
            "agent_config": asdict(model.config),
            "reward_config": asdict(reward),
            "examples": len(records),
            "elapsed_seconds": time.monotonic() - started,
            "teacher": {"shop_policy": "search", "grow_scalers": True, "blind_rollout": None},
            "outcome_targets": "absent: correction actions were not executed",
            "snapshot_selection": "one uniform reservoir-sampled disagreement per game, RNG 1729",
            "snapshot_count": len(snapshots),
        },
    )
    write_json(output / "student_outcomes.json", outcomes)
    from .transfer_diagnostics import summarize_predictions

    # Outcomes are joined only for diagnostic stratification, never correction targets.
    game_outcomes = {row["seed"]: row["won"] for row in outcomes}
    diagnostic = summarize_predictions([{**row, "won": game_outcomes[row["seed"]]} for row in agreement])
    diagnostic["interpretation"] = "Student-visited-state diagnostics; outcomes are not teacher-label targets"
    write_json(output / "agreement_strata.json", diagnostic)
    write_json(
        output / "agreement_summary.json",
        {
            key: float(np.mean([row[key] for row in agreement]))
            for key in ("greedy_agreement", "sampled_agreement", "teacher_probability", "nll")
        },
    )
    return records
