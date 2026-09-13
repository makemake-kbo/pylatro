"""Independent CPU policy processes with isolated progress/artifact namespaces."""

from __future__ import annotations

import multiprocessing
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.transfer import sha256


def _worker(task):
    import torch
    from evaluate_action_heads import run_panel

    root, name, checkpoint, seeds = task
    torch.set_num_threads(4)
    results = run_panel(root, name, checkpoint, seeds, torch.device("cpu"))
    atomic_json(root / "evaluation_status.json", dict(status="complete", panel=name))
    return results


def run_parallel_panels(root, name, control, candidates, seeds):
    """Run one shared control and two candidates, each with four engine workers.

    Separate processes own their model and sampling RNG. Each writes progress
    into its own directory. Completed outputs are linked into the usual panel
    layout only after all workers succeed; partial runs remain inspectable.
    """
    from evaluate_action_heads import run_panel

    labels = ("control", "seed0", "seed1")
    checkpoints = (control, *candidates)
    if len(checkpoints) != 3:
        raise ValueError("Parallel confirmation requires one control and two candidates")
    plan = dict(
        backend="cpu", concurrent_policies=3, torch_threads_per_policy=4, engine_workers_per_policy=4,
        independent_process_rng=True, seeds=list(seeds),
        checkpoints={label: sha256(path) for label, path in zip(labels, checkpoints, strict=True)},
        artifact_layout="isolated worker directories; links published after all policies finish",
        note="Concurrency changes job overlap; policy, batching and per-policy RNG settings are unchanged",
    )
    plan_path = root / f"{name}_execution.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError("Parallel CPU execution plan changed")
    atomic_json(plan_path, plan)
    if all((root / "evaluations" / f"{name}_{label}").exists() for label in labels):
        import torch

        return {
            key: run_panel(root, f"{name}_{label}", checkpoint, seeds, torch.device("cpu"))
            for key, label, checkpoint in zip(("control", 0, 1), labels, checkpoints, strict=True)
        }
    tasks = []
    for label, checkpoint in zip(labels, checkpoints, strict=True):
        worker_root = root / "cpu_workers" / label
        worker_root.mkdir(parents=True, exist_ok=True)
        for metadata in ("manifest.json", "cache_complete.json"):
            destination = worker_root / metadata
            if destination.exists():
                if sha256(destination) != sha256(root / metadata):
                    raise ValueError("Worker metadata differs from the frozen study")
            else:
                shutil.copyfile(root / metadata, destination)
        tasks.append((worker_root, f"{name}_{label}", checkpoint, seeds))
    atomic_json(root / "evaluation_status.json", dict(status="running_parallel", panel=name, labels=labels))
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_worker, tasks))
    (root / "evaluations").mkdir(exist_ok=True)
    for worker_root, panel, _, _ in tasks:
        destination = root / "evaluations" / panel
        source = worker_root / "evaluations" / panel
        if destination.exists():
            if not destination.is_symlink() or destination.resolve() != source.resolve():
                raise ValueError("Existing panel conflicts with completed parallel output")
        else:
            destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=True)
    return dict(zip(("control", 0, 1), results, strict=True))
