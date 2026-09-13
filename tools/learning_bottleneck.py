#!/usr/bin/env python3
"""Reproducible nine-hour MPS study. Each subcommand preserves completed artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.learning_bottleneck import (
    ARMS,
    CHECKPOINT,
    CONFIRM_SEEDS,
    DEV_SEEDS,
    HF_DATA,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    accelerator_job,
    balanced_probe,
    collision_audit,
    disable_dropout,
    measure,
    memorization_name,
    read_json,
    reserve_diagnostic_completion,
    reserve_evaluation_capacity,
)
from pylatro_agent.training.model_generate import load_records, save_records
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    check_seed_separation,
    evaluate_artifact,
    load_policy,
    paired_statistics,
    sha256,
    source_hashes,
    write_json,
)
from pylatro_agent.transfer_diagnostics import replay_game, selected_records
from pylatro_agent.transfer_training import train_continuation
from pylatro_agent.vocab import build_vocab


def initialize(root):
    environment_path = root / "environment.json"
    if not environment_path.exists():
        import os
        import platform
        import sys

        import pyarrow

        write_json(
            environment_path,
            dict(
                python=sys.version,
                platform=platform.platform(),
                machine=platform.machine(),
                torch=str(torch.__version__),
                torch_git_revision=torch.version.git_version,
                numpy=np.__version__,
                pyarrow=pyarrow.__version__,
                mps_built=torch.backends.mps.is_built(),
                default_dtype=str(torch.get_default_dtype()),
                environment={
                    key: os.environ.get(key)
                    for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "OMP_NUM_THREADS")
                },
            ),
        )
    originals_path = root / "original_hashes.json"
    if not originals_path.exists():
        originals = {
            str(p): sha256(p)
            for folder in (CHECKPOINT.parent, HF_DATA)
            for p in sorted(folder.rglob("*"))
            if p.is_file()
        }
        write_json(originals_path, originals)
    path = root / "manifest.json"
    if path.exists():
        return read_json(path)
    manifests = list(HF_DATA.glob("metadata/generation.json")) + list(HF_DATA.glob("metadata/games.jsonl"))
    manifests.append(Path("docs/heuristic/pretraining_v68_seeds.json"))
    # Previous corrections are training data as well.
    for candidate in Path("eval/transfer_20260912").rglob("manifest.json"):
        payload = read_json(candidate)
        if "seeds" in payload and ("correction" in str(candidate) or payload.get("kind") == "training"):
            manifests.append(candidate)
    panels = TRAIN_SEEDS + VALIDATION_SEEDS + DEV_SEEDS + CONFIRM_SEEDS
    evidence = check_seed_separation(panels, manifests)
    manifest = dict(
        study="learning_bottleneck_20260913",
        checkpoint=str(CHECKPOINT),
        checkpoint_sha256=sha256(CHECKPOINT),
        source_hashes=source_hashes(),
        seed_separation=evidence,
        panels={
            "training": TRAIN_SEEDS,
            "validation": VALIDATION_SEEDS,
            "development": DEV_SEEDS,
            "confirmation": CONFIRM_SEEDS,
        },
        arms=ARMS,
        training_seeds=[0, 1],
        updates=750,
        physical_batch=320,
        budget_seconds=32400,
        confirmation_reserve_seconds=9000,
        reference=dict(
            deck="b_blue",
            stake=1,
            win_ante=8,
            temperature=1.0,
            sampling_seeds=[0, 1, 2],
            batch_size=32,
            workers=4,
            precision="fp32",
        ),
        prior_reserved_seeds=DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS,
    )
    write_json(path, manifest)
    return manifest


def generation_init():
    global _data, _vocab, _reward
    from pylatro_agent.transfer import restore_configuration

    torch.set_num_threads(1)
    _data = load_game_data()
    _vocab = build_vocab(_data)
    from pylatro_agent.checkpoint import load_checkpoint_payload

    _, _reward = restore_configuration(load_checkpoint_payload(CHECKPOINT, "cpu"))


def generate_one(task):
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.tokenizer import Tokenizer
    from pylatro_agent.training.fast_generate import _run_game_single_pass

    seed, destination = task
    path = Path(destination) / f"game_{seed}.parquet"
    summary_path = path.with_suffix(".json")
    if summary_path.exists():
        summary = read_json(summary_path)
        if summary["sha256"] != sha256(path):
            raise ValueError("Completed generated game changed")
        return summary
    if path.exists():
        raise ValueError("Uncommitted generated game requires inspection")
    started = time.monotonic()
    rows, ante, won = _run_game_single_pass(
        seed,
        _data,
        Tokenizer(vocab=_vocab),
        HeuristicAgent(shop_policy="search", grow_scalers=True),
        0.997,
        8,
        _reward,
        deck_key="b_blue",
        stake=1,
    )
    for step, row in enumerate(rows):
        row["step"] = step
        if not row["obs"]["action_mask"][row["action"]]:
            raise RuntimeError("Illegal generated label")
    if not rows or any(row["terminal_outcome_mask"] != 1.0 for row in rows):
        raise RuntimeError(f"Incomplete/stalled teacher game {seed}")
    save_records(rows, path, reward_config=_reward)
    summary = dict(
        seed=seed,
        steps=len(rows),
        max_ante=ante,
        won=won,
        sha256=sha256(path),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(summary_path, summary)
    return summary


def generate(root, workers=4):
    if workers < 1:
        raise ValueError("Generation requires at least one CPU worker")
    attempts_path = root / "generation_attempts.json"
    attempts = read_json(attempts_path) if attempts_path.exists() else []
    attempt = dict(workers=workers, started_unix=time.time(), status="running")
    attempts.append(attempt)
    write_json(attempts_path, attempts)
    started = time.monotonic()
    try:
        _generate(root, workers)
        attempt["status"] = "complete"
    except BaseException as exc:
        attempt.update(status="interrupted", error=repr(exc))
        raise
    finally:
        attempt["elapsed_seconds"] = time.monotonic() - started
        write_json(attempts_path, attempts)
    write_json(
        root / "generation_complete.json",
        dict(elapsed_seconds=sum(row["elapsed_seconds"] for row in attempts), attempts=len(attempts)),
    )


def _generate(root, workers):
    initialize(root)
    started = time.monotonic()
    sources = source_hashes()
    for name, seeds in (("training", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        destination = root / "data" / name
        destination.mkdir(parents=True, exist_ok=True)
        generation_path = destination / "generation.json"
        if not generation_path.exists():
            write_json(
                generation_path,
                dict(
                    seeds=seeds,
                    sources=sources,
                    deck="b_blue",
                    stake=1,
                    win_ante=8,
                    shop_policy="search",
                    grow_scalers=True,
                    blind_rollout=None,
                    gamma=0.997,
                    reward_snapshot_version=3,
                ),
            )
        elif read_json(generation_path)["sources"] != sources:
            # Only the pinned game/observation/teacher sources matter on a generation resume.
            original = read_json(generation_path)["sources"]
            changed = [
                p
                for p, h in original.items()
                if p.startswith("src/")
                and sources.get(p) != h
                and not any(s in p for s in ("learning_bottleneck", "transfer_training", "transfer_diagnostics"))
            ]
            if changed:
                raise RuntimeError(f"Generator source drift on resume: {changed}")
        results = []
        with mp.get_context("spawn").Pool(workers, initializer=generation_init) as pool:
            for row in pool.imap_unordered(generate_one, [(s, str(destination)) for s in seeds], chunksize=1):
                results.append(row)
                if len(results) % 10 == 0:
                    print(
                        "generate", name, len(results), len(seeds), "wins", sum(r["won"] for r in results), flush=True
                    )
        results.sort(key=lambda r: r["seed"])
        if [r["seed"] for r in results] != list(seeds):
            raise RuntimeError("Missing generated games")
        with (destination / "games.jsonl").open("w") as stream:
            for row in results:
                stream.write(json.dumps(row) + "\n")
        write_json(
            destination / "summary.json",
            dict(
                games=len(results),
                wins=sum(r["won"] for r in results),
                records=sum(r["steps"] for r in results),
                elapsed_seconds=time.monotonic() - started,
                file_hashes={f"game_{r['seed']}.parquet": r["sha256"] for r in results},
            ),
        )


def hf_selection(reward):
    games = [json.loads(line) for line in (HF_DATA / "metadata/games.jsonl").read_text().splitlines()]
    seeds = sorted(
        map(int, np.random.default_rng(1729).choice(sorted(int(game["seed"]) for game in games), 200, replace=False))
    )
    previous = Path("eval/transfer_20260912/diagnostics_v2/manifest.json")
    if previous.exists() and seeds != read_json(previous)["offline_seeds"]:
        raise ValueError("Recomputed HF diagnostic selection differs from the original study")
    records = load_records(HF_DATA, reward_config=reward)
    selected = selected_records(records, seeds)
    return [row for seed in seeds for row in selected[seed]]


def audit(root, device):
    with accelerator_job(root, "audit", "diagnostics", 350):
        vocab = build_vocab(load_game_data())
        model, reward, _ = load_policy(CHECKPOINT, vocab, device)
        rows = hf_selection(reward)
        out = root / "audit"
        out.mkdir(exist_ok=False)
        write_json(out / "collisions.json", collision_audit(rows))
        for mixture in (0.5, 1.0):
            model.config.hand_ar_mixture_eps = mixture
            measure(model, rows, device, out / f"mixture{mixture}", detailed=True)
        probe = balanced_probe(rows)
        save_records(probe, out / "probe.parquet", reward_config=reward)
        write_json(out / "probe_ids.json", [(row["seed"], row["step"]) for row in probe])


def replay(root, *, pilot=False):
    if not pilot and not (root / "generation_complete.json").exists():
        raise RuntimeError("Generation must finish first")
    vocab = build_vocab(load_game_data())
    _, reward, _ = load_policy(CHECKPOINT, vocab, torch.device("cpu"))
    records = load_records(root / "data/training", reward_config=reward)
    games = [read_json(p) for p in sorted((root / "data/training").glob("game_*.json"))]
    rng = np.random.default_rng(1729)
    seeds = []
    for won in (False, True):
        seeds.extend(
            map(int, rng.choice(sorted(r["seed"] for r in games if r["won"] == won), 5 if pilot else 10, replace=False))
        )
    rows = selected_records(records, seeds)
    results = []
    data = load_game_data()
    for seed in seeds:
        result = replay_game(rows[seed], data, vocab, reward)
        results.append(result)
        print("replay", result, flush=True)
        write_json(root / ("pilot_replay.json" if pilot else "current_replay.json"), results)
        if any(value is not None for key, value in result.items() if key.startswith("first_")):
            raise RuntimeError(f"Current generated trajectory diverged: {result}")
    write_json(
        root / ("pilot_replay_complete.json" if pilot else "replay_complete.json"),
        dict(games=len(results), seeds=seeds),
    )


def gradient_audit(root, device):
    from pylatro_agent.value_head import outcome_nll, return_huber_loss

    with accelerator_job(root, "gradient_audit", "diagnostics", 45):
        vocab = build_vocab(load_game_data())
        model, reward, payload = load_policy(CHECKPOINT, vocab, device)
        records = load_records(root / "audit/probe.parquet", reward_config=reward)
        rows = [records[i] for i in list(range(16)) + list(range(160, 176))]
        config = SupervisedConfig(**{**payload["supervised_config"], "reward_config": reward})
        batch = _collate_batch(rows, device, config)
        model.eval()
        dist, values = _grammar_distribution(model, batch)
        logp = dist.log_prob(batch["actions"])
        losses = dict(
            bc=-(logp * batch["bc_weight"]).sum() / batch["bc_weight"].sum(),
            critic=config.value_loss_coeff * return_huber_loss(values, batch["value_target"])
            + config.outcome_loss_coeff
            * outcome_nll(
                values["outcome_probabilities"], batch["terminal_outcome_target"], batch["terminal_outcome_mask"]
            ),
            entropy=-config.action_entropy_coeff * dist.entropy().mean(),
        )
        named = list(model.named_parameters())
        gradients = {}
        for component, loss in losses.items():
            gradients[component] = [
                g.detach() if g is not None else torch.zeros_like(p)
                for g, (_, p) in zip(
                    torch.autograd.grad(loss, [p for _, p in named], retain_graph=True, allow_unused=True),
                    named,
                    strict=True,
                )
            ]
        grouped = {}
        for group, predicate in {
            "all": lambda n: True,
            "backbone": lambda n: n.startswith("backbone"),
            "embedding": lambda n: n.startswith("embedding"),
            "candidate_heads": lambda n: "action_grammar_head.candidate_" in n,
            "candidate_embeddings": lambda n: n.startswith("embedding") and "candidate_" in n,
            "autoregressive": lambda n: "action_grammar_head.hand_" in n,
            "critic": lambda n: n.startswith("value_head"),
        }.items():
            indices = [i for i, (name, _) in enumerate(named) if predicate(name)]
            vectors = {name: torch.cat([values[i].flatten() for i in indices]) for name, values in gradients.items()}
            if any(not torch.isfinite(v).all() for v in vectors.values()):
                raise RuntimeError("Nonfinite component gradient")
            grouped[group] = dict(
                norms={name: float(v.norm()) for name, v in vectors.items()},
                cosines={
                    f"{a}/{b}": float(torch.nn.functional.cosine_similarity(vectors[a], vectors[b], dim=0))
                    for a, b in (("bc", "critic"), ("bc", "entropy"), ("critic", "entropy"))
                },
            )
        before = {name: p.detach().clone() for name, p in named}
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        optimizer.zero_grad()
        sum(losses.values()).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        updates = {name: float((p.detach() - before[name]).norm()) for name, p in named}
        write_json(
            root / "gradients_v2.json",
            dict(
                losses={k: float(v.detach()) for k, v in losses.items()},
                groups=grouped,
                parameter_update_norms=updates,
                total_gradient_norm=float(norm),
                record_ids=[(r["seed"], r["step"]) for r in rows],
            ),
        )


def memorize(root, device, initialization, mixture):
    from pylatro_agent.agent import BalatroAgent
    from pylatro_agent.checkpoint import save_checkpoint
    from pylatro_agent.reward import reward_checkpoint_metadata

    name = memorization_name(initialization, mixture)
    with accelerator_job(root, name, "diagnostics", 650):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        vocab = build_vocab(load_game_data())
        model, reward, payload = load_policy(CHECKPOINT, vocab, device, mixture=mixture)
        if initialization == "random":
            torch.manual_seed(0)
            model = BalatroAgent(replace(model.config, dropout=0.0), vocab).to(device)
        disable_dropout(model)
        rows = list(load_records(root / "audit/probe.parquet", reward_config=reward))
        config = SupervisedConfig(**{**payload["supervised_config"], "reward_config": reward})
        batch = _collate_batch(rows, device, config)
        output = root / name
        output.mkdir(exist_ok=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
        started = time.monotonic()
        with (output / "learning.jsonl").open("x", buffering=1) as stream:
            for update in range(301):
                if update % 25 == 0:
                    summary = measure(model, rows, device, output / f"step{update}", detailed=True)
                    row = dict(
                        update=update,
                        elapsed_seconds=time.monotonic() - started,
                        **summary["strata"]["overall"]["metrics"],
                    )
                    stream.write(json.dumps(row) + "\n")
                    print(name, row, flush=True)
                if update == 300:
                    break
                model.train()
                dist, _ = _grammar_distribution(model, batch)
                loss = -dist.log_prob(batch["actions"]).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite memorization loss")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                for group in optimizer.param_groups:
                    group["lr"] = 3e-4 * min((update + 1) / 20, 1.0)
                optimizer.step()
        save_checkpoint(
            model,
            output / "checkpoint.pt",
            extra={
                "agent_config": asdict(model.config),
                "diagnostic_only": True,
                "initialization": initialization,
                "updates": 300,
                **reward_checkpoint_metadata(reward),
            },
        )
        write_json(
            output / "complete.json",
            dict(
                updates=300,
                elapsed_seconds=time.monotonic() - started,
                initialization=initialization,
                mixture=mixture,
                probe_sha256=sha256(root / "audit/probe.parquet"),
                settings=dict(batch=320, lr=3e-4, warmup=20, dropout=0, critic=0, entropy=0, weighting="unit"),
            ),
        )


def expressivity(root, device):
    """Conditional follow-up: independent per-state outputs and CPU/MPS backprop."""
    from pylatro_agent.action_grammar import ActionGrammarDistribution, ActionGrammarOutput

    with accelerator_job(root, "expressivity_and_device_gradients", "diagnostics", 150):
        vocab = build_vocab(load_game_data())
        model, reward, _ = load_policy(CHECKPOINT, vocab, device, mixture=1.0)
        model.eval()
        rows = list(load_records(root / "audit/probe.parquet", reward_config=reward))
        batch = _collate_batch(rows, device, SupervisedConfig())
        with torch.no_grad():
            dist, _ = _grammar_distribution(model, batch)
        output = ActionGrammarOutput(
            **{
                key: torch.nn.Parameter(getattr(dist.output, key).detach().clone())
                for key in dist.output.__dataclass_fields__
            }
        )
        parameters = [getattr(output, key) for key in output.__dataclass_fields__]
        optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.0)
        curves = []
        for update in range(1001):
            dist = ActionGrammarDistribution(
                output, batch["action_mask"], tokens=batch["tokens"], hand_ar_mixture_eps=1.0
            )
            logp = dist.log_prob(batch["actions"])
            loss = -logp.mean()
            if update % 100 == 0:
                with torch.no_grad():
                    curves.append(
                        dict(
                            update=update,
                            nll=float(loss.detach()),
                            probability=float(logp.exp().mean()),
                            accuracy=float((dist.mode() == batch["actions"]).float().mean()),
                        )
                    )
                print("free_outputs", curves[-1], flush=True)
            if update == 1000:
                break
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
        write_json(
            root / "free_output_expressivity.json",
            dict(
                curves=curves,
                mixture=1.0,
                explanation="Independent logits per observation; removes the encoder and shared feature mapping.",
            ),
        )
        del model, dist, output, parameters, optimizer, batch
        gc.collect()
        torch.mps.empty_cache()
        gradients = {}
        losses = {}
        tiny = rows[:4] + rows[160:164]
        for backend in ("cpu", "mps"):
            model, _, _ = load_policy(CHECKPOINT, vocab, torch.device(backend), mixture=1.0)
            model.eval()
            batch = _collate_batch(tiny, torch.device(backend), SupervisedConfig())
            dist, _ = _grammar_distribution(model, batch)
            loss = -dist.log_prob(batch["actions"]).mean()
            loss.backward()
            gradients[backend] = torch.cat(
                [p.grad.detach().cpu().flatten() for p in model.parameters() if p.grad is not None]
            )
            losses[backend] = float(loss.detach())
            del model, dist, loss, batch
            gc.collect()
            torch.mps.empty_cache()
        difference = gradients["mps"] - gradients["cpu"]
        relative = float(difference.norm() / gradients["cpu"].norm().clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(gradients["mps"], gradients["cpu"], dim=0))
        result = dict(
            losses=losses,
            gradient_relative_l2=relative,
            gradient_cosine=cosine,
            gradient_max_abs=float(difference.abs().max()),
            passed=bool(
                np.isclose(losses["cpu"], losses["mps"], rtol=1e-4, atol=1e-4) and relative < 0.01 and cosine > 0.999
            ),
        )
        write_json(root / "cpu_mps_gradients.json", result)
        if not result["passed"]:
            raise RuntimeError("CPU/MPS gradient discrepancy requires investigation before training")


def memorization_cross_eval(root, device):
    with accelerator_job(root, "memorization_same_weights_routing", "diagnostics", 20):
        vocab = build_vocab(load_game_data())
        comparisons = {}
        for initialization in ("epoch7", "random"):
            for trained_mixture in (0.5, 1.0):
                name = memorization_name(initialization, trained_mixture)
                checkpoint = root / name / "checkpoint.pt"
                evaluated_mixture = 1.0 if trained_mixture == 0.5 else 0.5
                model, reward, _ = load_policy(checkpoint, vocab, device, mixture=evaluated_mixture)
                rows = load_records(root / "audit/probe.parquet", reward_config=reward)
                summary = measure(model, rows, device, root / "memorization_cross_eval" / name, detailed=True)
                comparisons[name] = dict(
                    trained_mixture=trained_mixture,
                    evaluated_mixture=evaluated_mixture,
                    checkpoint_sha256=sha256(checkpoint),
                    summary=summary,
                )
                del model, rows
                gc.collect()
                torch.mps.empty_cache()
        write_json(root / "memorization_cross_eval.json", comparisons)


def eval_one(root, checkpoint, output_name, seeds, *, mixture=None, temperature=1.0, stage="development"):
    output = root / "evaluations" / output_name
    model, reward, _ = load_policy(checkpoint, build_vocab(load_game_data()), torch.device("mps"), mixture=mixture)
    data = load_game_data()
    vocab = build_vocab(data)
    checkpoint_hash = sha256(checkpoint)
    sampled, greedy = [], []
    for sampling in (0, 1, 2, None):
        destination = output / (f"sample{sampling}" if sampling is not None else "greedy")
        if (destination / "summary.json").exists():
            manifest = read_json(destination / "manifest.json")
            if (
                manifest["status"] != "complete"
                or manifest["seeds"] != list(seeds)
                or manifest["checkpoint_sha256"] != checkpoint_hash
                or manifest["temperature"] != temperature
                or manifest["sampling_seed"] != sampling
                or manifest["agent_config"]["hand_ar_mixture_eps"] != model.config.hand_ar_mixture_eps
            ):
                raise RuntimeError("Evaluation resume mismatch")
            rows = [json.loads(s) for s in (destination / "outcomes.jsonl").read_text().splitlines()]
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            reserve_evaluation_capacity(root, stage, len(seeds) * 1.6)
            with accelerator_job(root, f"eval/{output_name}/{sampling}", stage, len(seeds) * 1.6):
                rows = evaluate_artifact(
                    model,
                    reward,
                    data,
                    vocab,
                    seeds,
                    torch.device("mps"),
                    destination,
                    greedy=sampling is None,
                    sampling_seed=sampling,
                    temperature=temperature,
                    metadata=dict(
                        checkpoint=str(checkpoint),
                        checkpoint_sha256=checkpoint_hash,
                        study_manifest_sha256=sha256(root / "manifest.json"),
                        source_hashes=source_hashes(),
                        explicit_overrides=dict(mixture=mixture, temperature=temperature),
                    ),
                )
        (greedy if sampling is None else sampled).extend(rows)
    del model
    gc.collect()
    torch.mps.empty_cache()
    return sampled, greedy


def inference(root):
    results = {}
    for mixture in (0.5, 1.0):
        for temperature in (1.0, 0.5):
            key = f"inference_m{mixture}_t{temperature}"
            results[key] = eval_one(
                root, CHECKPOINT, key, DEV_SEEDS[:50], mixture=mixture, temperature=temperature, stage="diagnostics"
            )
    reference = results["inference_m0.5_t1.0"]
    write_json(
        root / "inference_comparisons.json",
        {
            name: dict(
                sampled=paired_statistics(rows[0], reference[0]), greedy=paired_statistics(rows[1], reference[1])
            )
            for name, rows in results.items()
        },
    )


def fixed_learning_panel(records, seeds, per_game):
    # Pick uniformly spaced decisions from each whole game, fixed across all arms.
    selected = selected_records(records, seeds)
    result = []
    for seed in seeds:
        rows = selected[seed]
        indices = np.unique(np.linspace(0, len(rows) - 1, min(len(rows), per_game), dtype=int))
        result.extend(rows[i] for i in indices)
    return result


def verify_new_data(root):
    marker = root / "data_verified_before_training.json"
    if marker.exists():
        for name in ("training", "validation"):
            if read_json(marker)[name]["summary_sha256"] != sha256(root / "data" / name / "summary.json"):
                raise RuntimeError("Verified dataset manifest changed")
        return
    from pylatro_agent.checkpoint import load_checkpoint_payload
    from pylatro_agent.transfer import restore_configuration

    _, reward = restore_configuration(load_checkpoint_payload(CHECKPOINT, "cpu"))
    result = {}
    for name, seeds in (("training", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        folder = root / "data" / name
        summary = read_json(folder / "summary.json")
        if sorted(summary["file_hashes"]) != [f"game_{seed}.parquet" for seed in seeds]:
            raise RuntimeError("Dataset does not contain exactly the declared game seeds")
        count = 0
        for seed in seeds:
            path = folder / f"game_{seed}.parquet"
            if sha256(path) != summary["file_hashes"][path.name]:
                raise RuntimeError("Generated dataset hash changed before training")
            rows = list(load_records(path, reward_config=reward))
            running = 0.0
            for step in range(len(rows) - 1, -1, -1):
                row = rows[step]
                if row["seed"] != seed or row["step"] != step:
                    raise RuntimeError("Duplicate/missing recorded step")
                expected = dict(
                    deck_key="b_blue",
                    stake=1,
                    heuristic_shop_policy="search",
                    heuristic_grow_scalers=True,
                    heuristic_blind_rollout=None,
                    reward_snapshot_version=3,
                    tokenizer_version=13,
                    terminal_outcome_mask=1.0,
                )
                if any(row[key] != value for key, value in expected.items()):
                    raise RuntimeError(f"Incorrect demonstration metadata at {seed}/{step}")
                if not row["obs"]["action_mask"][row["action"]]:
                    raise RuntimeError("Illegal demonstration label")
                running = row["reward"] + 0.997 * running
                if not np.isclose(running, row["return_target"], rtol=1e-10, atol=1e-10):
                    raise RuntimeError("Recorded reward-to-go does not match executed rewards")
            count += len(rows)
        if count != summary["records"]:
            raise RuntimeError("Dataset decision denominator mismatch")
        result[name] = dict(summary_sha256=sha256(folder / "summary.json"), games=len(seeds), decisions=count)
    write_json(marker, result)


def train(root, device, arm, seed):
    if not (root / "replay_complete.json").exists():
        raise RuntimeError("Current-rule replay must pass before training")
    verify_new_data(root)
    mixture, weight = ARMS[arm]
    output = root / "training" / f"{arm}_seed{seed}"
    if (output / "manifest.json").exists():
        manifest = read_json(output / "manifest.json")
        if manifest["status"] == "complete" and manifest["updates_completed"] == 750:
            return
        raise RuntimeError("Partial training arm is not silently restarted")
    # Include data loading, diagnostics, and checkpoint writing in the job's charge.
    ledger = read_json(root / "compute.json")
    previous = [j["elapsed_seconds"] for j in ledger["jobs"] if j["stage"] == "training" and j["status"] == "complete"]
    estimate = max(previous, default=1600)
    with accelerator_job(root, f"train/{arm}/{seed}", "training", estimate) as remaining:
        vocab = build_vocab(load_game_data())
        temporary_model, reward, _ = load_policy(CHECKPOINT, vocab, device)
        del temporary_model
        torch.mps.empty_cache()
        records = load_records(root / "data/training", reward_config=reward)
        validation = load_records(root / "data/validation", reward_config=reward)
        train_probe = fixed_learning_panel(records, TRAIN_SEEDS[:100], 16)
        validation_probe = fixed_learning_panel(validation, VALIDATION_SEEDS, 16)

        def observe(model, update, destination):
            for label, panel in (("training", train_probe), ("validation", validation_probe)):
                measure(model, panel, device, destination / "offline" / f"{label}_{update}")

        checkpoint = train_continuation(
            CHECKPOINT,
            records,
            vocab,
            device,
            output,
            arm="control",
            seed=seed,
            updates=750,
            batch_size=320,
            max_seconds=remaining - 120,
            mixture=mixture,
            hand_weight=weight,
            intermediate_steps=(0, 250, 500, 750),
            observer=observe,
            reserved_seeds=VALIDATION_SEEDS + DEV_SEEDS + CONFIRM_SEEDS,
            data_path=root / "data/training",
            num_games=len(TRAIN_SEEDS),
        )
        provenance = dict(
            experiment_arm=arm,
            checkpoint_sha256=sha256(checkpoint),
            training_data_sha256=sha256(root / "data/training/summary.json"),
            validation_data_sha256=sha256(root / "data/validation/summary.json"),
            source_hashes=source_hashes(),
            study_manifest_sha256=sha256(root / "manifest.json"),
        )
        write_json(output / "provenance.json", provenance)


def combined_comparison(candidate, control):
    from continue_transfer import acceptance

    return acceptance(
        {i: candidate[i][0] for i in (0, 1)},
        {i: control[i][0] for i in (0, 1)},
        {i: candidate[i][1] for i in (0, 1)},
        {i: control[i][1] for i in (0, 1)},
    )


def development(root):
    offline_final(root)
    results = {}
    for arm in ARMS:
        results[arm] = {}
        for seed in (0, 1):
            checkpoint = root / "training" / f"{arm}_seed{seed}" / "checkpoint.pt"
            if read_json(checkpoint.parent / "manifest.json")["status"] != "complete":
                raise RuntimeError("Incomplete training arm")
            results[arm][seed] = eval_one(root, checkpoint, f"screen_{arm}_seed{seed}", DEV_SEEDS[:50])
    comparisons = {arm: combined_comparison(results[arm], results["control"]) for arm in ARMS if arm != "control"}
    rank = []
    for index, arm in enumerate(ARMS):
        if arm == "control":
            continue
        ante = np.mean([row["max_ante"] for seed in (0, 1) for row in results[arm][seed][0]])
        delta = comparisons[arm]["paired"]["sampled"]["sampled_or_greedy_delta"]
        rank.append((-delta, -float(ante), index, arm))
    selected = min(rank)[-1]
    write_json(root / "screen_comparisons.json", comparisons)
    selection_path = root / "selection.json"
    selection = dict(
        arm=selected,
        rule="paired sampled gain; final ante; declared arm order",
        screen_comparisons_sha256=sha256(root / "screen_comparisons.json"),
    )
    if selection_path.exists() and read_json(selection_path) != selection:
        raise RuntimeError("Selection changed on resume")
    write_json(selection_path, selection)
    expanded = {}
    for arm in ("control", selected):
        expanded[arm] = {}
        for seed in (0, 1):
            more = eval_one(
                root,
                root / "training" / f"{arm}_seed{seed}" / "checkpoint.pt",
                f"expand_{arm}_seed{seed}",
                DEV_SEEDS[50:],
            )
            expanded[arm][seed] = tuple(results[arm][seed][i] + more[i] for i in (0, 1))
    write_json(root / "development_comparison.json", combined_comparison(expanded[selected], expanded["control"]))
    write_json(root / "development_complete.json", dict(selected=selected, selection_sha256=sha256(selection_path)))


def offline_final(root):
    marker = root / "offline_final_complete.json"
    if marker.exists():
        return
    device = torch.device("mps")
    vocab = build_vocab(load_game_data())
    references = [(f"original_m{mixture}", CHECKPOINT, mixture) for mixture in (0.5, 1.0)]
    references.extend(
        (f"{arm}_seed{seed}", root / "training" / f"{arm}_seed{seed}" / "checkpoint.pt", None)
        for arm in ARMS
        for seed in (0, 1)
    )
    for label, checkpoint, mixture in references:
        output = root / "full_validation" / label
        provenance = dict(
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint),
            validation_data_sha256=sha256(root / "data/validation/summary.json"),
            study_manifest_sha256=sha256(root / "manifest.json"),
            explicit_overrides={} if mixture is None else dict(hand_ar_mixture_eps=mixture),
        )
        if (output / "summary.json").exists():
            saved = read_json(output / "manifest.json")
            if saved["status"] != "complete" or any(saved.get(key) != value for key, value in provenance.items()):
                raise RuntimeError("Incomplete validation prediction requires inspection")
            continue
        completed = [
            j["elapsed_seconds"]
            for j in read_json(root / "compute.json")["jobs"]
            if j["name"].startswith("full_validation/") and j["status"] == "complete"
        ]
        estimate = max([120, *completed])
        reserve_diagnostic_completion(root, estimate)
        with accelerator_job(root, f"full_validation/{label}", "diagnostics", estimate):
            model, reward, _ = load_policy(checkpoint, vocab, device, mixture=mixture)
            records = load_records(root / "data/validation", reward_config=reward)
            measure(model, records, device, output, detailed=True)
            prediction_manifest = read_json(output / "manifest.json")
            prediction_manifest.update(provenance, source_hashes=source_hashes())
            write_json(output / "manifest.json", prediction_manifest)
            del model, records
            gc.collect()
            torch.mps.empty_cache()
    write_json(marker, dict(policies=10, seeds=VALIDATION_SEEDS))


def incumbent_reference(root, selected, candidates, controls):
    """Supplementary reference check, admitted only after every required confirmation panel."""
    plan_path = root / "incumbent_reference_plan.json"
    if not plan_path.exists():
        return
    plan = read_json(plan_path)
    result_path = root / "incumbent_reference_comparison.json"
    if result_path.exists():
        return
    ledger = read_json(root / "compute.json")
    remaining = 32400 - sum(job["elapsed_seconds"] for job in ledger["jobs"])
    if remaining < plan["budget_gate_seconds"]:
        write_json(
            result_path, dict(status="skipped_budget", remaining_seconds=remaining, plan_sha256=sha256(plan_path))
        )
        return
    panel = plan
    if plan.get("larger_panel") and remaining >= plan["larger_panel"]["budget_gate_seconds"]:
        panel = plan["larger_panel"]
    evaluation_name = f"confirm_original_reference{len(panel['seeds'])}"
    original = eval_one(root, CHECKPOINT, evaluation_name, panel["seeds"], mixture=0.5, stage="confirmation")
    comparisons = {}
    seeds = set(panel["seeds"])
    for label, runs in ((selected, candidates), ("control", controls)):
        comparisons[label] = {}
        for i, mode in enumerate(("sampled", "greedy")):
            candidate = [
                dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                for seed in (0, 1)
                for row in runs[seed][i]
                if row["seed"] in seeds
            ]
            baseline = [
                dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                for seed in (0, 1)
                for row in original[i]
            ]
            comparisons[label][mode] = paired_statistics(candidate, baseline)
    write_json(
        result_path,
        dict(
            status="complete",
            plan_sha256=sha256(plan_path),
            selected=selected,
            seeds=panel["seeds"],
            evaluation_name=evaluation_name,
            remaining_seconds_at_admission=remaining,
            comparisons=comparisons,
            note=(
                "Supplementary original-reference comparison. Panel size depends only on the predeclared time gate. "
                "Original outcomes are shared across training replicates; "
                "game seeds are bootstrap units. Primary acceptance is unchanged."
            ),
        ),
    )


def confirmation(root):
    from continue_transfer import acceptance

    if not (root / "development_complete.json").exists():
        raise RuntimeError("Development and selection must finish before confirmation")
    selected = read_json(root / "selection.json")["arm"]
    results = {}
    for arm in ("control", selected):
        results[arm] = {}
        for seed in (0, 1):
            results[arm][seed] = eval_one(
                root,
                root / "training" / f"{arm}_seed{seed}" / "checkpoint.pt",
                f"confirm_{arm}_seed{seed}",
                CONFIRM_SEEDS,
                stage="confirmation",
            )
    comparison = acceptance(
        {i: results[selected][i][0] for i in (0, 1)},
        {i: results["control"][i][0] for i in (0, 1)},
        {i: results[selected][i][1] for i in (0, 1)},
        {i: results["control"][i][1] for i in (0, 1)},
        confirmation=True,
    )
    write_json(root / "confirmation_comparison.json", comparison)
    if ARMS[selected][0] != 0.5:
        original = eval_one(
            root,
            CHECKPOINT,
            "confirm_original_same_mixture",
            CONFIRM_SEEDS,
            mixture=ARMS[selected][0],
            stage="confirmation",
        )
        candidate = tuple(
            [
                dict(row, sampling_seed=f"train{s}/sample{row['sampling_seed']}")
                for s in (0, 1)
                for row in results[selected][s][i]
            ]
            for i in (0, 1)
        )
        baseline = tuple(
            [dict(row, sampling_seed=f"train{s}/sample{row['sampling_seed']}") for s in (0, 1) for row in original[i]]
            for i in (0, 1)
        )
        write_json(
            root / "confirmation_training_beyond_routing.json",
            {
                "sampled": paired_statistics(candidate[0], baseline[0]),
                "greedy": paired_statistics(candidate[1], baseline[1]),
                "note": "Original checkpoint is shared across training replicates; game seeds are the bootstrap unit.",
            },
        )
    incumbent_reference(root, selected, results[selected], results["control"])
    write_json(root / "confirmation_complete.json", dict(selected=selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        choices=[
            "init",
            "generate",
            "audit",
            "replay",
            "replay_pilot",
            "gradients",
            "memorize",
            "inference",
            "train",
            "development",
            "confirmation",
            "expressivity",
            "memorization_cross_eval",
        ],
    )
    parser.add_argument("--root", type=Path, default=Path("eval/learning_bottleneck_20260913"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--generation-workers", type=int, default=4)
    parser.add_argument("--initialization", choices=["epoch7", "random"], default="epoch7")
    parser.add_argument("--mixture", type=float, default=0.5)
    parser.add_argument("--arm", choices=list(ARMS), default="control")
    parser.add_argument("--seed", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.task == "init":
        initialize(args.root)
    elif args.task == "generate":
        generate(args.root, args.generation_workers)
    elif args.task == "replay":
        replay(args.root)
    elif args.task == "replay_pilot":
        replay(args.root, pilot=True)
    else:
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable in this process; do not silently fall back")
        if args.task == "audit":
            audit(args.root, torch.device(args.device))
        elif args.task == "gradients":
            gradient_audit(args.root, torch.device(args.device))
        elif args.task == "memorize":
            memorize(args.root, torch.device(args.device), args.initialization, args.mixture)
        elif args.task == "inference":
            inference(args.root)
        elif args.task == "train":
            train(args.root, torch.device(args.device), args.arm, args.seed)
        elif args.task == "development":
            development(args.root)
        elif args.task == "confirmation":
            confirmation(args.root)
        elif args.task == "expressivity":
            expressivity(args.root, torch.device(args.device))
        elif args.task == "memorization_cross_eval":
            memorization_cross_eval(args.root, torch.device(args.device))


if __name__ == "__main__":
    main()
