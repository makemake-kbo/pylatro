"""Offline and action-replay diagnostics on explicitly identified demonstration games."""

from __future__ import annotations

import copy
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .action import ActionType, decode_action
from .env import BalatroEnv
from .heuristic import HeuristicAgent
from .tokenizer import Tokenizer
from .training.fast_generate import _run_game_single_pass
from .training.ppo_policy import _grammar_distribution
from .training.supervised import SupervisedConfig, _collate_batch
from .transfer import write_json


def selected_records(records, seeds):
    """Scan only seed columns before loading observations; preserve published step IDs."""
    selected = set(seeds)
    found = defaultdict(list)
    offset = 0
    for file in records._files:
        file_seeds = file.read(columns=["seed"]).column("seed").to_numpy()
        for i in np.flatnonzero(np.isin(file_seeds, list(selected))):
            row = records[offset + int(i)]
            if "step" not in row:
                raise ValueError("Diagnostics require published seed/step identifiers")
            found[int(row["seed"])].append(row)
        offset += len(file_seeds)
    if set(found) != selected:
        raise ValueError("Requested diagnostic games missing from dataset")
    for seed, rows in found.items():
        rows.sort(key=lambda row: row["step"])
        if [r["step"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"Duplicate or missing steps in seed {seed}")
    return found


def first_observation_difference(a, b):
    if set(a) != set(b):
        return {"field": "observation_keys", "left": sorted(a), "right": sorted(b)}
    for key in a:
        left, right = np.asarray(a[key]), np.asarray(b[key])
        if left.shape != right.shape:
            return {"field": key, "left_shape": list(left.shape), "right_shape": list(right.shape)}
        if not np.array_equal(left, right):
            indices = np.argwhere(left != right)
            idx = tuple(indices[0])
            return {
                "field": key,
                "index": list(map(int, idx)),
                "left": left[idx].item(),
                "right": right[idx].item(),
                "different_elements": len(indices),
                "max_abs_difference": float(np.max(np.abs(left.astype(float) - right.astype(float)))),
            }
    return None


class RecordedActions:
    def __init__(self, rows):
        self.rows = rows
        self.step = 0

    def select_action(self, state, sub_phase, mask, **kwargs):
        if self.step >= len(self.rows):
            raise RuntimeError("Recorded actions exhausted before game termination")
        action = int(self.rows[self.step]["action"])
        if not mask[action]:
            raise RuntimeError(f"Illegal recorded action at step {self.step}: {action}")
        self.step += 1
        return action


def replay_game(rows, data, vocab, reward):
    from .training.fast_generate import _build_obs
    from .training.fast_runner import FastRunner

    seed = int(rows[0]["seed"])
    result = {
        "seed": seed,
        "recorded_won": bool(rows[0]["won"]),
        "recorded_steps": len(rows),
        "first_saved_fast": None,
        "first_fast_env": None,
        "first_reward_difference": None,
        "first_saved_reward": None,
        "first_state_difference": None,
    }
    runner = FastRunner(seed, data, deck_key=rows[0]["deck_key"], stake=rows[0]["stake"], raise_errors=True)
    tokenizer = Tokenizer(vocab=vocab)
    env = BalatroEnv(
        seed=seed,
        data=data,
        vocab=vocab,
        deck_key=rows[0]["deck_key"],
        stake=rows[0]["stake"],
        reward_config=reward,
        enable_teacher=False,
        max_steps=2000,
    )
    obs, _ = env.reset(seed=seed)
    terminal = False
    env_rewards = []
    info = {}
    try:
        for step, saved in enumerate(rows):
            reconstructed = _build_obs(runner, tokenizer)
            difference = first_observation_difference(saved["obs"], reconstructed)
            if difference and result["first_saved_fast"] is None:
                result["first_saved_fast"] = {"step": step, **difference}
            difference = first_observation_difference(reconstructed, obs)
            if difference and result["first_fast_env"] is None:
                result["first_fast_env"] = {"step": step, **difference}
            action = int(saved["action"])
            if runner.done or terminal or not reconstructed["action_mask"][action] or not obs["action_mask"][action]:
                result["replay_error"] = {
                    "step": step,
                    "fast_terminal": runner.done,
                    "env_terminal": terminal,
                    "action": action,
                    "fast_legal": bool(reconstructed["action_mask"][action]),
                    "env_legal": bool(obs["action_mask"][action]),
                }
                break
            runner.step(action)
            obs, value, terminated, truncated, info = env.step(action)
            env_rewards.append(value)
            terminal = terminated or truncated
            fields = {
                "round_score": (runner.round_score, env._controller.round_score),
                "joker_order": (list(runner.state.joker_keys), list(env.state.joker_keys)),
                "phase": (str(runner.sub_phase), str(env._sub_phase)),
                "ante": (runner.state.round_resets.ante, env.state.round_resets.ante),
                "terminal": (runner.done, terminal),
            }
            for key, (left, right) in fields.items():
                if left != right and result["first_state_difference"] is None:
                    result["first_state_difference"] = {"step": step, "field": key, "fast": left, "env": right}
        result.update(
            fast_won=runner.won,
            fast_final_ante=runner.max_ante,
            env_won=bool(info.get("won")),
            env_final_ante=info.get("ante"),
            env_terminal=terminal,
            replayed_steps=len(env_rewards),
        )
    finally:
        env.close()
    if len(env_rewards) == len(rows):
        fast, _, _ = _run_game_single_pass(
            seed,
            data,
            tokenizer,
            RecordedActions(rows),
            reward.gamma,
            reward_config=reward,
            deck_key=rows[0]["deck_key"],
            stake=rows[0]["stake"],
        )
        for step, (saved, current, env_value) in enumerate(zip(rows, fast, env_rewards, strict=True)):
            for name, left, right in [
                ("first_saved_reward", saved["reward"], current["reward"]),
                ("first_reward_difference", current["reward"], env_value),
            ]:
                if not np.isclose(left, right, rtol=1e-6, atol=1e-6) and result[name] is None:
                    result[name] = {"step": step, "left": left, "right": right}
    return result


def summarize_predictions(rows):
    groups = defaultdict(list)
    confusion = defaultdict(int)
    for row in rows:
        for key in (
            "overall",
            f"phase/{row['phase']}",
            f"ante/{row['ante']}",
            f"family/{row['teacher_family']}",
            f"outcome/{row['won']}",
        ):
            groups[key].append(row)
        confusion[row["teacher_family"] + " -> " + row["predicted_family"]] += 1
    metrics = ("accuracy", "nll", "teacher_probability", "unreachable", "candidate_covered")
    summary = {}
    for key, group in groups.items():
        games = defaultdict(list)
        for row in group:
            games[row["seed"]].append(row)
        summary[key] = {"decisions": len(group), "games": len(games)}
        for weighting in ("decision", "game"):
            summary[key][weighting + "_weighted"] = {}
            for metric in metrics:
                values = [r[metric] for r in group if r[metric] is not None]
                if weighting == "game":
                    values = [
                        np.mean([r[metric] for r in rs if r[metric] is not None])
                        for rs in games.values()
                        if any(r[metric] is not None for r in rs)
                    ]
                summary[key][weighting + "_weighted"][metric] = float(np.mean(values)) if values else None
    return {
        "strata": summary,
        "family_confusion_counts": dict(confusion),
        "interpretation": "Training-data diagnostics, not held-out accuracy; candidate coverage only for play/discard",
    }


def offline_predictions(model, records, device, output, batch_size=320):
    output = Path(output)
    rows = []
    model.eval()
    with torch.inference_mode(), (output / "predictions.jsonl").open("x", buffering=1) as stream:
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            batch = _collate_batch(chunk, device, SupervisedConfig())
            dist, _ = _grammar_distribution(model, batch)
            logp = dist.log_prob(batch["actions"]).cpu().numpy()
            actions = dist.mode().cpu().numpy()
            candidates = {}
            for kind in (True, False):
                _, valid, mapped = dist._candidate_distribution(is_play=kind)
                candidates[kind] = mapped.masked_fill(~valid, -1).cpu().numpy()
            for i, record in enumerate(chunk):
                teacher = decode_action(record["action"]).action_type
                play = teacher == ActionType.PLAY_SUBSET
                hand = play or teacher == ActionType.DISCARD_SUBSET
                row = {
                    "seed": int(record["seed"]),
                    "step": int(record["step"]),
                    "phase": int(record["obs"]["scalars"][7]),
                    "ante": int(record["obs"]["scalars"][2]),
                    "won": bool(record["won"]),
                    "teacher_action": int(record["action"]),
                    "predicted_action": int(actions[i]),
                    "teacher_family": teacher.value,
                    "predicted_family": decode_action(int(actions[i])).action_type.value,
                    "accuracy": float(actions[i] == record["action"]),
                    "nll": -float(logp[i]),
                    "teacher_probability": float(np.exp(logp[i])),
                    "unreachable": float(logp[i] < -1e7),
                    "candidate_covered": float(record["action"] in candidates[play][i]) if hand else None,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
            if (start // batch_size) % 20 == 0:
                print(f"Offline predictions: {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    report = summarize_predictions(rows)
    write_json(output / "offline_summary.json", report)
    return report


def clone_diagnostic_env(env):
    """Any-phase diagnostic copy; archive boundary semantics remain unchanged."""
    return copy.deepcopy(env, {id(env._data): env._data, id(env._vocab): env._vocab})


def isolated_teacher_action(env):
    """Query V68 on a private simulator copy, without attaching student outcomes."""
    clone = clone_diagnostic_env(env)
    teacher = HeuristicAgent(shop_policy="search", grow_scalers=True)
    action = int(
        teacher.select_action(
            clone.state, clone._sub_phase, clone.action_masks(), round_score=clone._controller.round_score
        )
    )
    if not env.action_masks()[action]:
        raise RuntimeError("V68 returned an illegal correction")
    return action


def live_state_digest(env):
    return pickle.dumps((env.state, env._sub_phase, env._controller.round_score, env._history), protocol=5)
