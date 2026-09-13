"""Reproducible transfer measurements; evaluation panels never supply training rows."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, fields, replace
from pathlib import Path

import numpy as np

DEVELOPMENT_SEEDS = tuple(range(100000, 100200))
CONFIRMATION_SEEDS = tuple(range(110000, 110400))
CORRECTION_SEEDS = tuple(range(3000000, 3000100))
SAMPLING_SEEDS = (0, 1, 2)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {
        str(p.relative_to(root)): sha256(p)
        for folder in ("src", "tools")
        for p in sorted((root / folder).rglob("*"))
        if p.suffix in {".py", ".so", ".json"}
    }


def manifest_seeds(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [int(json.loads(line)["seed"]) for line in path.read_text().splitlines() if line]
    payload = json.loads(path.read_text())
    values = payload if isinstance(payload, list) else payload.get("seeds")
    if values is None:
        raise ValueError(f"{path} has no explicit training seeds")
    return [int(seed) for seed in values]


def check_demonstration_seeds(seeds, manifests):
    training = {seed for path in manifests for seed in manifest_seeds(path)}
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= training:
        raise ValueError("Demonstration evaluation requires distinct seeds present in the training manifests")
    return [{"path": str(path), "sha256": sha256(path)} for path in manifests]


def check_seed_separation(seeds, manifests):
    """Require explicit complete training seed lists, never infer disjointness from ranges."""
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and distinct")
    evidence = []
    for path in manifests:
        path = Path(path)
        training = manifest_seeds(path)
        overlap = sorted(set(seeds) & set(training))
        if overlap:
            raise ValueError(f"Evaluation/training seed overlap in {path}: {overlap[:10]}")
        evidence.append({"path": str(path), "sha256": sha256(path), "training_games": len(set(training))})
    return evidence


def restore_configuration(payload, *, mixture=None):
    from .agent import AgentConfig
    from .reward import REWARD_MODEL_VERSION, RewardConfig, reward_config_fingerprint

    configs = []
    for key, cls in (("agent_config", AgentConfig), ("reward_config", RewardConfig)):
        saved = payload.get(key)
        expected = {field.name for field in fields(cls)}
        # The archived, exact pre-variant config identifies the original head
        # unambiguously. This is the only supported missing-field migration;
        # all other omissions and unknown fields still fail closed.
        if key == "agent_config" and isinstance(saved, dict) and set(saved) == expected - {"action_head_variant"}:
            saved = dict(saved, action_head_variant="legacy_v1")
        if not isinstance(saved, dict) or set(saved) != expected:
            raise ValueError(f"Checkpoint {key} must contain exactly the current configuration fields")
        configs.append(cls(**saved))
    agent, reward = configs
    if payload.get("reward_model_version") != REWARD_MODEL_VERSION or payload.get(
        "reward_fingerprint"
    ) != reward_config_fingerprint(reward):
        raise ValueError("Checkpoint reward schema/fingerprint mismatch")
    if mixture is not None:
        if not 0 <= mixture <= 1:
            raise ValueError("Mixture must lie in [0, 1]")
        agent = replace(agent, hand_ar_mixture_eps=mixture)
    return agent, reward


def load_policy(checkpoint, vocab, device, *, mixture=None):
    from .agent import BalatroAgent
    from .checkpoint import load_checkpoint_payload

    payload = load_checkpoint_payload(checkpoint, "cpu")
    config, reward = restore_configuration(payload, mixture=mixture)
    model = BalatroAgent(config, vocab).to(device)
    state = payload["state_dict"]
    if state and all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    payload = dict(payload)
    payload["configuration_migrations"] = (
        [{"field": "agent_config.action_head_variant", "value": "legacy_v1", "source": "exact archived config schema"}]
        if "action_head_variant" not in payload["agent_config"] else []
    )
    return model, reward, payload


def _cluster(rows):
    if not rows:
        raise ValueError("No outcomes")
    groups = defaultdict(dict)
    for row in rows:
        seed, repeat = int(row["seed"]), row.get("sampling_seed")
        if repeat in groups[seed]:
            raise ValueError(f"duplicate seed/repeat: {seed}/{repeat}")
        groups[seed][repeat] = row
    repeats = set(next(iter(groups.values())))
    if any(set(group) != repeats for group in groups.values()):
        raise ValueError("missing sampling repeats for some game seeds")
    return groups


def _bootstrap(values, *, resamples=10000):
    values = np.asarray(values, dtype=float)
    if resamples < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite nonempty values and positive resamples")
    rng = np.random.default_rng(1729)
    # Bound memory even for larger panels.
    means = np.concatenate(
        [
            values[rng.integers(len(values), size=(min(256, resamples - i), len(values)))].mean(1)
            for i in range(0, resamples, 256)
        ]
    )
    # Suppress machine-epsilon remnants at an exact zero boundary.
    return np.quantile(means, [0.025, 0.975]).round(15).tolist()


def summarize(rows):
    groups = _cluster(rows)
    wins = [np.mean([float(r["won"]) for r in group.values()]) for group in groups.values()]
    n = len(groups)
    mean = float(np.mean(wins))
    # Wilson interval is informative at zero/all wins for single-repeat panels.
    z = 1.959963984540054
    centre = (mean + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(mean * (1 - mean) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "game_seeds": n,
        "games": len(rows),
        "wins": sum(int(r["won"]) for r in rows),
        "win_rate": mean,
        "win_rate_cluster_ci95": _bootstrap(wins),
        "win_rate_wilson_ci95": [centre - half, centre + half] if len(rows) == n else None,
        "mean_final_ante": float(np.mean([r.get("final_ante", r["max_ante"]) for r in rows])),
        "stalls": sum(bool(r.get("stalled")) for r in rows),
        "censored": sum(bool(r.get("censored")) for r in rows),
        "ante_clear_rates": {
            str(a): sum(a in r.get("cleared_antes", []) for r in rows) / len(rows) for a in range(1, 9)
        },
    }


def paired_statistics(candidate, control):
    a, b = _cluster(candidate), _cluster(control)
    if set(a) != set(b) or any(set(a[s]) != set(b[s]) for s in a):
        raise ValueError("Paired panels must have the same game seeds and sampling repeats")
    differences = [np.mean([int(a[s][r]["won"]) - int(b[s][r]["won"]) for r in a[s]]) for s in sorted(a)]
    # Equal repeat counts are enforced above, so integer win totals give the
    # same seed-weighted point estimate without cancellation/order roundoff.
    win_difference = sum(int(row["won"]) for row in candidate) - sum(int(row["won"]) for row in control)
    return {
        "game_seeds": len(a),
        "games_per_arm": len(candidate),
        "sampled_or_greedy_delta": win_difference / len(candidate),
        "paired_ci95": _bootstrap(differences),
        "unit": "game seed; sampling repeats averaged before resampling",
    }


def evaluate_artifact(
    model,
    reward,
    data,
    vocab,
    seeds,
    device,
    output,
    *,
    greedy=False,
    sampling_seed=0,
    deck="b_blue",
    temperature=1.0,
    stake=1,
    win_ante=8,
    batch_size=32,
    workers=4,
    cpu_threads=0,
    max_idle_steps=256,
    metadata=None,
    kind="fresh_start",
):
    import time

    from torch.utils.tensorboard import SummaryWriter

    from .training.ppo_evaluation import evaluate_model

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        **(metadata or {}),
        "kind": kind,
        "seeds": list(seeds),
        "greedy": greedy,
        "sampling_seed": None if greedy else sampling_seed,
        "deck": deck,
        "stake": stake,
        "win_ante": win_ante,
        "temperature": temperature,
        "batch_size": batch_size,
        "workers": workers,
        "cpu_threads": cpu_threads,
        "max_idle_steps": max_idle_steps,
        "device": str(device),
        "agent_config": asdict(model.config),
        "reward_config": asdict(reward),
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    try:
        with SummaryWriter(str(output / "tensorboard")) as writer:
            evaluate_model(
                model,
                data,
                vocab,
                len(seeds),
                device,
                seeds=list(seeds),
                greedy=greedy,
                sampling_seed=sampling_seed,
                deck_key=deck,
                stake=stake,
                win_ante=win_ante,
                temperature=temperature,
                eval_batch_size=batch_size,
                eval_workers=workers,
                eval_cpu_threads=cpu_threads,
                reward_config=reward,
                max_no_progress_steps=max_idle_steps,
                results_path=output / "outcomes.jsonl",
                writer=writer,
            )
        rows = [json.loads(line) for line in (output / "outcomes.jsonl").read_text().splitlines()]
        if len(rows) != len(seeds) or {r["seed"] for r in rows} != set(seeds):
            raise RuntimeError("Evaluation returned incomplete seed panel")
        metrics = summarize(rows)
        write_json(output / "summary.json", metrics)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)
    return rows


def _teacher_game(task):
    import time

    from pylatro import load_game_data

    from .env import BalatroEnv
    from .heuristic import HeuristicAgent
    from .vocab import build_vocab

    seed, config, reward = task
    data = load_game_data()
    env = BalatroEnv(
        seed=seed,
        data=data,
        vocab=build_vocab(data),
        enable_teacher=False,
        deck_key=config["deck"],
        stake=config["stake"],
        win_ante=config["win_ante"],
        max_steps=config["max_idle_steps"],
        reward_config=reward,
    )
    teacher = HeuristicAgent(shop_policy=config["shop_policy"], grow_scalers=config["grow_scalers"])
    started = time.monotonic()
    steps = 0
    try:
        obs, _ = env.reset(seed=seed)
        while True:
            action = int(
                teacher.select_action(
                    env.state, env._sub_phase, env.action_masks(), round_score=env._controller.round_score
                )
            )
            if not obs["action_mask"][action]:
                raise RuntimeError(f"Illegal teacher action: seed={seed}, step={steps}")
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                return {
                    "seed": seed,
                    "sampling_seed": None,
                    "won": bool(info["won"]),
                    "max_ante": info["ante"],
                    "final_ante": info["ante"],
                    "round_score": info["round_score"],
                    "steps": steps,
                    "stalled": bool(info.get("stalled")),
                    "censored": bool(truncated),
                    "cleared_antes": sorted(env._cleared_boss_antes | ({8} if info["won"] else set())),
                    "joker_keys": list(env.state.joker_keys),
                    "elapsed_seconds": time.monotonic() - started,
                }
    finally:
        env.close()


def evaluate_teacher_artifact(seeds, reward, generation_path, output, *, workers=4, metadata=None):
    import multiprocessing as mp
    import time

    from torch.utils.tensorboard import SummaryWriter

    generation = json.loads(Path(generation_path).read_text())
    required = {"policy": "V68", "shop_policy": "search", "grow_scalers": True, "blind_rollout": None}
    if any(generation.get(key) != value for key, value in required.items()):
        raise ValueError("Matched teacher baseline requires explicit V68 search/scaler settings, blind rollouts off")
    config = {key: generation[key] for key in ("deck", "stake", "win_ante", "shop_policy", "grow_scalers")}
    config["max_idle_steps"] = 256
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        **(metadata or {}),
        "kind": "matched_V68_fresh_start",
        "configuration": config,
        "generation_sha256": sha256(generation_path),
        "seeds": list(seeds),
        "workers": workers,
        "source_hashes": source_hashes(),
        "status": "running",
    }
    write_json(output / "manifest.json", manifest)
    rows = []
    try:
        with (
            mp.get_context("spawn").Pool(workers) as pool,
            (output / "outcomes.jsonl").open("x", buffering=1) as stream,
        ):
            for row in pool.imap_unordered(_teacher_game, [(seed, config, reward) for seed in seeds]):
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"Teacher games: {len(rows)}/{len(seeds)}, wins={sum(r['won'] for r in rows)}", flush=True)
        metrics = summarize(rows)
        write_json(output / "summary.json", metrics)
        with SummaryWriter(str(output / "tensorboard")) as writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar("teacher/" + key, value, 0)
        manifest["status"] = "complete"
    except BaseException as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)
    return rows
