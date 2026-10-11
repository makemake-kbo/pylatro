#!/usr/bin/env python3
"""Frozen 24-hour public PPO/search-supervision comparison.

Run prepare, then run (resumes automatically), status or report. Only the
technical pilot starts the immutable wall clock. Pilot outcomes do not select
duration. All scientific training requires MPS FP32, with CPU public execution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pylatro import load_game_data  # noqa: E402
from pylatro_agent.checkpoint import load_checkpoint_payload  # noqa: E402
from pylatro_agent.hybrid import HybridPolicy, PlannerConfig  # noqa: E402
from pylatro_agent.public_runtime import PublicNeuralPolicy  # noqa: E402
from pylatro_agent.public_state import OBSERVATION_INTERPRETATION  # noqa: E402
from pylatro_agent.search_ppo_study import (  # noqa: E402
    NAMESPACE,
    TRAINING_RNG,
    assert_separation,
    atomic,
    audit_metadata,
    check_deadline,
    digest,
    encode,
    exclusive,
    fingerprint,
    immutable,
    paired_claim,
    read,
    seed_panels,
    select_duration,
    start_clock,
)
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch  # noqa: E402
from pylatro_agent.training.ppo_policy import _grammar_distribution  # noqa: E402
from pylatro_agent.training.public_training import PublicTrainingEnv, observation_dict  # noqa: E402
from pylatro_agent.training.search_guidance import RECIPE, restore_snapshot, teach  # noqa: E402
from pylatro_agent.training.search_ppo import make_config, require_mps, train_to  # noqa: E402
from pylatro_agent.vocab import build_vocab  # noqa: E402

DEFAULT = ROOT / "runs/search_guided_ppo_20261004"
PARENT = ROOT / "incumbents/ppo_replica1/checkpoint.pt"
TESTS = [
    "tests/test_search_guided_ppo.py",
    "tests/test_public_hybrid.py",
    "tests/test_search_scaling.py",
    "tests/test_complete_cohort.py",
    "tests/test_complete_cohort_study.py",
    "tests/test_ppo_resume_checkpoint.py",
    "tests/test_action_masking.py",
    "tests/test_joker_layout.py",
    "tests/test_deck_exhaustion.py",
    "tests/test_rng.py",
    "tests/test_incumbent_reference.py",
]
_DATA = None
_POLICIES = {}


def sources():
    paths = [p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".json", ".so"}]
    paths += [
        Path(__file__),
        ROOT / "scripts/compile_cython.py",
        ROOT / "docs/transfer/search_guided_ppo_20261004.md",
        *[ROOT / p for p in TESTS],
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def worker_init():
    global _DATA
    torch.set_num_threads(1)
    _DATA = load_game_data()


def policy(checkpoint, sample=False):
    key = (str(checkpoint), sample)
    if key not in _POLICIES:
        # Early attribution needs exactly two policies; avoid retaining every
        # model across a long panel in each CPU worker.
        if len(_POLICIES) >= 2:
            _POLICIES.clear()
        _POLICIES[key] = PublicNeuralPolicy(checkpoint, _DATA, sample=sample)
    return _POLICIES[key]


def label_worker(job):
    record, deadline = job
    return teach(record, _DATA, deadline=deadline)


def game_worker(job):
    check_deadline(job["deadline"])
    path = Path(job["path"])
    started = time.monotonic()
    started_unix = time.time()
    selected = policy(job["checkpoint"], job["mode"] == "sampled")
    base = policy(job["base_checkpoint"]) if job["mode"] == "early" else selected
    seed = job["seed"]
    import hashlib

    rng_seed = int.from_bytes(hashlib.sha256(f"{NAMESPACE}:evaluation:{seed}".encode()).digest()[:8], "big")
    selected.reset_rng(rng_seed)
    env = PublicTrainingEnv(
        seed,
        _DATA,
        build_vocab(_DATA),
        max_steps=10000,
        capture_states=job.get("capture", False),
        deadline=job["deadline"],
    )
    env.reset(seed=seed)
    planner = HybridPolicy(selected, _DATA, PlannerConfig(mode="hand" if job["mode"] == "assisted" else "off"))
    changes, searches = 0, 0
    while not env.runner.done:
        check_deadline(job["deadline"])
        snapshot = env.runner.snapshot()
        if job["mode"] == "early" and snapshot.values["round_resets"]["ante"] > 3:
            action = base(snapshot)
        else:
            decision = planner.decide(snapshot)
            action = decision.action
            changes += int(decision.diagnostics["changed"])
            searches += int(decision.diagnostics["candidates"] > 0)
        _, _, terminal, truncated, _ = env.step(action)
        if truncated:
            raise RuntimeError(f"Incomplete evaluation game {seed}; no replacement or loss imputation")
    check_deadline(job["deadline"])
    if not terminal:
        raise RuntimeError("Missing terminal evaluation outcome")
    states = env.selected_public_states() if job.get("capture") else []
    if job.get("capture"):
        for row in states:
            row["proposal"] = selected(restore_snapshot(row))
        atomic(path.with_suffix(".states.json"), states)
    result = dict(
        identity=job["identity"],
        seed=seed,
        mode=job["mode"],
        won=env.runner.won,
        complete=True,
        seconds=time.monotonic() - started,
        started_unix=started_unix,
        finished_unix=time.time(),
        steps=env.decisions,
        searches=searches,
        changes=changes,
    )
    atomic(path, result)
    # No wins cross the process boundary during the timing pilot.
    return dict(seconds=result["seconds"], path=str(path), complete=True)


def pool_map(function, jobs):
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init
    ) as pool:
        futures = [pool.submit(function, job) for job in jobs]
        return [future.result() for future in futures]


def teach_records(records, deadline):
    if not records:
        return []
    return pool_map(label_worker, [(row, deadline) for row in records])


def progress(out, **values):
    values.update(unix=time.time(), pid=os.getpid())
    atomic(out / "status.json", values)
    print(encode(values).strip(), flush=True)


def prepare(out):
    if (out / "frozen.json").exists() or (out / "clock.json").exists():
        raise ValueError("Already frozen; use run to resume without resetting the clock")
    require_mps()
    out.mkdir(parents=True, exist_ok=True)
    progress(out, stage="correctness")
    with (out / "correctness.log").open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *TESTS, "-q"],
            cwd=ROOT,
            env={**os.environ, "SEARCH_PPO_ACTUAL_MPS_TEST": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(f"Correctness checks failed; see {out / 'correctness.log'}")
    from pylatro_agent.training import fast_runner

    if not fast_runner.__file__.endswith(".so"):
        raise RuntimeError("Correctness must exercise the compiled execution path")
    progress(out, stage="metadata_audit")
    audit, exclusions = audit_metadata(ROOT, out)
    parent = load_checkpoint_payload(PARENT, "cpu")

    # Explicit checkpoint reservations supplement metadata, without reading
    # any game outcome. Original PPO recipe and Adam remain pinned.
    def integers(value):
        if type(value) is int:
            return {value}
        if isinstance(value, dict):
            return set().union(*(integers(v) for v in value.values())) if value else set()
        if isinstance(value, (list, tuple)):
            return set().union(*(integers(v) for v in value)) if value else set()
        return set()

    exclusions.update(integers(parent.get("training_reserved_seeds", [])))
    exclusions.update(parent["cohort_ppo_config"]["cohort_seeds"])
    panels = seed_panels(exclusions)
    atomic(out / "metadata_audit.json", audit)
    atomic(out / "historical_integer_exclusions.json", sorted(exclusions))
    atomic(out / "seeds.json", panels)
    recipes = {
        arm: asdict(
            make_config(
                parent, guided=arm == "guided", seeds=panels["pilot"], rng_seed=TRAINING_RNG["pilot"], updates=1
            )
        )
        for arm in ("ordinary", "guided")
    }
    frozen = dict(
        namespace=NAMESPACE,
        parent=str(PARENT),
        parent_sha256=digest(PARENT),
        sources=sources(),
        created=time.time(),
        observation_mode=OBSERVATION_INTERPRETATION,
        recipes=recipes,
        guidance=RECIPE.metadata(),
        training_rng=TRAINING_RNG,
        seeds_sha256=digest(out / "seeds.json"),
        audit_sha256=digest(out / "metadata_audit.json"),
        exclusions_sha256=digest(out / "historical_integer_exclusions.json"),
        correctness_sha256=digest(out / "correctness.log"),
        protocol=dict(
            wall_seconds=86400,
            start="first technical pilot",
            workers=4,
            torch_threads=1,
            durations=[20, 12, 8],
            margin=0.25,
            minimum_pilot_labels=16,
            minimum_pilot_games=8,
            development_gain=0.05,
            confirmation_gain=0.05,
            bootstrap_draws=10000,
            bootstrap_seed=1729,
            duration_uses="runtimes only; pilot wins sealed until duration locked",
            tie="ordinary",
            diagnostic_checkpoints=["initial", "halfway", "endpoint"],
            early_attribution="candidate greedy decisions in Antes 1-3; frozen base thereafter",
            correction_filter="consistency filter, not statistical confidence",
            comparison="matched PPO games and optimizer exposure, additional search compute reported",
        ),
    )
    atomic(out / "frozen.json", frozen)
    progress(out, stage="prepared", clock_started=False)


def verify(out):
    frozen = read(out / "frozen.json")
    if frozen["sources"] != sources() or digest(PARENT) != frozen["parent_sha256"]:
        raise ValueError("Frozen sources or parent checkpoint changed")
    for name, key in (
        ("seeds.json", "seeds_sha256"),
        ("metadata_audit.json", "audit_sha256"),
        ("historical_integer_exclusions.json", "exclusions_sha256"),
        ("correctness.log", "correctness_sha256"),
    ):
        if digest(out / name) != frozen[key]:
            raise ValueError(f"Frozen artifact changed: {name}")
    panels = read(out / "seeds.json")
    assert_separation(panels, read(out / "historical_integer_exclusions.json"))
    return frozen, panels


def train(out, frozen, panels, replica, arm, updates, endpoint, deadline):
    parent = load_checkpoint_payload(PARENT, "cpu")
    seeds = panels["pilot" if replica == "pilot" else replica + "_training"][: updates * 64]
    config = make_config(parent, guided=arm == "guided", seeds=seeds, rng_seed=TRAINING_RNG[replica], updates=updates)
    identity = dict(
        frozen_sha256=digest(out / "frozen.json"),
        parent_sha256=frozen["parent_sha256"],
        replica=replica,
        arm=arm,
        config_sha256=fingerprint(asdict(config)),
    )
    data = load_game_data()
    return train_to(
        PARENT,
        out / replica / arm,
        config,
        identity,
        endpoint,
        data=data,
        vocab=build_vocab(data),
        deadline=deadline,
        teach_records=teach_records,
        progress=lambda **kw: progress(out, **kw),
    )


def evaluate(out, stage, models, modes, seeds, deadline, *, capture=False):
    jobs, all_paths = [], []
    for name, checkpoint in models.items():
        for mode in modes:
            identity = dict(
                frozen_sha256=digest(out / "frozen.json"),
                model=name,
                stage=stage,
                mode=mode,
                checkpoint_sha256=digest(checkpoint),
                capture=capture,
                seed_panel_sha256=fingerprint(seeds),
            )
            for seed in seeds:
                path = out / "evaluation" / stage / name / mode / f"{seed}.json"
                all_paths.append(path)
                if path.exists():
                    row = read(path)
                    if (
                        row.get("identity") != identity
                        or row.get("complete") is not True
                        or row.get("seed") != seed
                        or row.get("mode") != mode
                    ):
                        raise ValueError(f"Evaluation duplicate/configuration mismatch: {path}")
                    continue
                jobs.append(
                    dict(
                        path=str(path),
                        identity=identity,
                        checkpoint=str(checkpoint),
                        base_checkpoint=str(PARENT),
                        mode=mode,
                        seed=seed,
                        deadline=deadline,
                        capture=capture,
                    )
                )
    if jobs:
        progress(out, stage="evaluation", panel=stage, games_pending=len(jobs))
        pool_map(game_worker, jobs)
    return all_paths


def panel_wins(out, stage, model, mode, seeds):
    rows = [read(out / "evaluation" / stage / model / mode / f"{seed}.json") for seed in seeds]
    if len(set(seeds)) != len(seeds) or any(
        not r["complete"] or r["seed"] != s for r, s in zip(rows, seeds, strict=True)
    ):
        raise ValueError("Incomplete or duplicate panel")
    return [int(r["won"]) for r in rows]


def diagnostic_measure(checkpoint, labels):
    """Fixed public states and fixed teacher targets; no checkpoint selection."""
    data = load_game_data()
    base = PublicNeuralPolicy(checkpoint, data)
    rows = []
    for start in range(0, len(labels), 32):
        selected = labels[start : start + 32]
        batch = _obs_dicts_to_batch(
            [observation_dict(restore_snapshot(r), data, base.tokenizer) for r in selected], torch.device("cpu")
        )
        with torch.inference_mode():
            dist, _ = _grammar_distribution(base.model, batch, temperature=1.0)
            actions = dist.mode().tolist()
            targets = torch.tensor([r.get("target", r["proposal"]) for r in selected])
            nll = -dist.log_prob(targets)
        for row, action, loss in zip(selected, actions, nll.tolist(), strict=True):
            rows.append(
                dict(
                    ante=row["ante"],
                    corrected=row["accepted"],
                    teacher_agreement=action == row.get("target", row["proposal"]),
                    proposal_retained=action == row["proposal"],
                    target_nll=loss,
                )
            )
    summary = {}
    for name, group in (
        ("all", rows),
        ("early", [r for r in rows if r["ante"] <= 3]),
        ("late", [r for r in rows if r["ante"] >= 4]),
    ):
        corrected = [r for r in group if r["corrected"]]
        retained = [r for r in group if not r["corrected"]]
        summary[name] = dict(
            states=len(group),
            corrections=len(corrected),
            teacher_agreement=float(np.mean([r["teacher_agreement"] for r in group])) if group else None,
            correction_learning=float(np.mean([r["teacher_agreement"] for r in corrected])) if corrected else None,
            correction_nll=float(np.mean([r["target_nll"] for r in corrected])) if corrected else None,
            retained_behavior=float(np.mean([r["proposal_retained"] for r in retained])) if retained else None,
        )
    return summary


def diagnose(out, checkpoint, name, deadline):
    path = out / "diagnostics" / f"{name}.json"
    labels_path = out / "diagnostics/public_labels.json"
    identity = dict(checkpoint_sha256=digest(checkpoint), labels_sha256=digest(labels_path))
    if path.exists():
        if read(path)["identity"] != identity:
            raise ValueError("Diagnostic checkpoint changed")
        return
    check_deadline(deadline)
    atomic(path, dict(identity=identity, measurements=diagnostic_measure(checkpoint, read(labels_path))))


def run(out):
    require_mps()
    torch.set_num_threads(1)
    frozen, panels = verify(out)
    if (out / "decision.json").exists():
        return report(out)
    if list(out.glob("*/ordinary/rejected.json")) or list(out.glob("*/guided/rejected.json")):
        raise RuntimeError("Matched comparison was stopped by KL/optimizer exposure rejection")
    clock = start_clock(out)
    deadline = clock["deadline"]
    check_deadline(deadline)
    if not (out / "duration.json").exists():
        for arm in ("ordinary", "guided"):
            started = time.monotonic()
            checkpoint = train(out, frozen, panels, "pilot", arm, 1, 1, deadline)
            runtime_path = out / "pilot" / arm / "runtime.json"
            if not runtime_path.exists():
                # Includes complete checkpoint I/O and interrupted elapsed
                # time conservatively, even if a crash preceded this record.
                seconds = max(
                    time.monotonic() - started,
                    checkpoint.stat().st_mtime - checkpoint.with_name("initial.pt").stat().st_mtime,
                )
                immutable(runtime_path, dict(seconds=seconds))
        for mode in ("greedy", "sampled", "assisted"):
            started = time.monotonic()
            paths = evaluate(out, "timing", {"base": PARENT}, [mode], panels["timing"], deadline)
            runtime_path = out / "timing" / f"{mode}.json"
            if not runtime_path.exists():
                # Timing fields only. Interrupted spans are conservatively
                # retained, and wins remain unopened for duration selection.
                spans = [(read(p)["started_unix"], read(p)["finished_unix"]) for p in paths]
                seconds = max(time.monotonic() - started, max(b for _, b in spans) - min(a for a, _ in spans))
                immutable(runtime_path, dict(wall_seconds=seconds, games=len(paths), workers=4))
        pilot = {
            arm: load_checkpoint_payload(out / "pilot" / arm / "update_1.pt", "cpu")["search_ppo_study"]["metrics"]
            for arm in ("ordinary", "guided")
        }
        # Read only runtime from timing games, never pilot wins or training
        # outcomes, until this immutable choice is committed.
        timing = {
            mode + "_game": read(out / "timing" / f"{mode}.json")["wall_seconds"] / len(panels["timing"])
            for mode in ("greedy", "sampled", "assisted")
        }
        timing.update(
            ordinary_update=read(out / "pilot/ordinary/runtime.json")["seconds"],
            guided_update=read(out / "pilot/guided/runtime.json")["seconds"],
            label_state=20.0,
            diagnostic_inference_state=1.0,
        )
        choice = select_duration(timing, deadline - time.time())
        choice.update(
            pilot_labels=pilot["guided"]["accepted"],
            pilot_games=pilot["guided"]["accepted_games"],
            chosen_unix=time.time(),
        )
        immutable(out / "duration.json", choice)
    choice = read(out / "duration.json")
    if choice["pilot_labels"] < 16 or choice["pilot_games"] < 8:
        atomic(out / "decision.json", dict(decision="retain", reason="insufficient teaching signal", scientific_fits=0))
        return report(out)
    if choice["updates"] is None:
        atomic(
            out / "decision.json",
            dict(decision="incomplete", reason="eight updates infeasible within original deadline", scientific_fits=0),
        )
        return report(out)
    updates = choice["updates"]
    labels_path = out / "diagnostics/public_labels.json"
    if not labels_path.exists():
        paths = evaluate(out, "diagnostic", {"base": PARENT}, ["greedy"], panels["diagnostic"], deadline, capture=True)
        states = [r for p in paths for r in read(p.with_suffix(".states.json"))]
        progress(out, stage="diagnostic_labels", states=len(states))
        atomic(labels_path, teach_records(states, deadline))
    diagnose(out, PARENT, "initial", deadline)
    exploratory = {}
    for endpoint in (updates // 2, updates):
        for arm in ("ordinary", "guided"):
            checkpoint = train(out, frozen, panels, "exploratory", arm, updates, endpoint, deadline)
            diagnose(out, checkpoint, f"exploratory_{arm}_{endpoint}", deadline)
            exploratory[arm] = checkpoint
    evaluate(
        out,
        "development",
        {"base": PARENT, **exploratory},
        ["greedy", "sampled", "assisted"],
        panels["development"],
        deadline,
    )
    if not (out / "selection.json").exists():
        wins = {
            name: sum(panel_wins(out, "development", name, "greedy", panels["development"]))
            for name in ("base", "ordinary", "guided")
        }
        winner = max(("ordinary", "guided"), key=lambda name: wins[name])
        selected = winner if (wins[winner] - wins["base"]) / 96 >= 0.05 else None
        immutable(
            out / "selection.json",
            dict(
                selected=selected,
                wins=wins,
                updates=updates,
                checkpoint_hashes={name: digest(p) for name, p in exploratory.items()},
            ),
        )
    selection = read(out / "selection.json")
    selected = selection["selected"]
    if not selected:
        atomic(out / "decision.json", dict(decision="retain", reason="development greedy gain below five points"))
        return report(out)
    if selection["checkpoint_hashes"] != {name: digest(p) for name, p in exploratory.items()}:
        raise ValueError("Selected exploratory endpoint changed")
    controls = ("ordinary", "guided") if selected == "guided" else ("ordinary",)
    confirmation = {}
    for endpoint in (updates // 2, updates):
        for arm in controls:
            checkpoint = train(out, frozen, panels, "confirmation", arm, updates, endpoint, deadline)
            diagnose(out, checkpoint, f"confirmation_{arm}_{endpoint}", deadline)
            confirmation[arm] = checkpoint
    models = {
        "base": PARENT,
        **{f"exploratory_{a}": exploratory[a] for a in controls},
        **{f"confirmation_{a}": confirmation[a] for a in controls},
    }
    evaluate(out, "confirmation", models, ["greedy", "sampled"], panels["confirmation"], deadline)
    selected_models = {
        f"exploratory_{selected}": exploratory[selected],
        f"confirmation_{selected}": confirmation[selected],
    }
    evaluate(out, "confirmation", {"base": PARENT, **selected_models}, ["assisted"], panels["confirmation"], deadline)
    evaluate(out, "confirmation", selected_models, ["early"], panels["confirmation"], deadline)
    seeds = panels["confirmation"]

    def vector(name, mode="greedy"):
        return panel_wins(out, "confirmation", name, mode, seeds)

    selected_names = list(selected_models)
    claims = dict(
        base=paired_claim([vector(n) for n in selected_names], [vector("base")] * 2),
        assisted=paired_claim([vector(n, "assisted") for n in selected_names], [vector("base", "assisted")] * 2),
        early_attribution=paired_claim([vector(n, "early") for n in selected_names], [vector("base")] * 2),
    )
    if selected == "guided":
        claims["search_additional"] = paired_claim(
            [vector(n) for n in selected_names], [vector("exploratory_ordinary"), vector("confirmation_ordinary")]
        )
    atomic(
        out / "decision.json",
        dict(
            decision="adopt" if claims["base"]["qualifies"] else "retain",
            selected=selected,
            claims=claims,
            assisted_command_change=claims["base"]["qualifies"] and claims["assisted"]["qualifies"],
        ),
    )
    return report(out)


def report(out):
    decision = read(out / "decision.json") if (out / "decision.json").exists() else dict(decision="incomplete")
    clock = read(out / "clock.json") if (out / "clock.json").exists() else None
    duration = read(out / "duration.json") if (out / "duration.json").exists() else None
    # Do not open any pilot/timing wins before the runtime lock is committed.
    panels = read(out / "seeds.json") if (out / "seeds.json").exists() else {}
    measurements = []
    if duration is not None:
        for stage in ("development", "confirmation"):
            for model in sorted((out / "evaluation" / stage).glob("*")):
                for mode in sorted(model.glob("*")):
                    rows = [read(p) for p in mode.glob("*.json") if not p.name.endswith(".states.json")]
                    expected = panels[stage]
                    completed = {r["seed"] for r in rows if r.get("complete")}
                    complete = len(rows) == len(expected) and completed == set(expected)
                    measurements.append(
                        dict(
                            panel=stage,
                            model=model.name,
                            mode=mode.name,
                            complete=complete,
                            games=len(completed),
                            expected=len(expected),
                            wins=sum(r["won"] for r in rows) if complete else None,
                            rate=sum(r["won"] for r in rows) / len(expected) if complete else None,
                        )
                    )
    exposure = []
    for p in sorted(out.glob("*/*/cohorts/update_*/update.json")):
        if "attempt" not in str(p):
            m = read(p)
            exposure.append(
                dict(
                    path=str(p.relative_to(out)),
                    update=m["update"],
                    games=m["games"],
                    optimizer_examples=m["optimizer"]["ppo_samples_processed"],
                    collection_seconds=m["collection_seconds"],
                    optimization_seconds=m["optimization_seconds"],
                    label_wall_seconds=m["label_seconds"],
                    label_simulation_seconds=m["label_simulation_seconds"],
                    accepted=m["accepted"],
                    accepted_games=m["accepted_games"],
                    labels=m["labels"],
                    rejections=m["rejections"],
                )
            )
    result = dict(
        decision=decision,
        clock=clock,
        duration=duration,
        measurements=measurements,
        exposure=exposure,
        last_status=read(out / "status.json") if (out / "status.json").exists() else {},
        interpretation="Greedy base strength, incremental guidance benefit, and assisted strength are separate claims. "
        "Paired uncertainty conditions on two trained runs. Search consistency is not a confidence guarantee. "
        "Current-blind scores do not establish long-term optimality or a path to 90%.",
    )
    atomic(out / "report.json", result)
    text = [
        f"# Search-guided PPO study\n\nDecision: **{decision['decision']}**.",
        decision.get(
            "reason", "Scientific panels are not yet complete." if decision["decision"] == "incomplete" else ""
        ),
        f"\nObservation: {OBSERVATION_INTERPRETATION}. Training: MPS FP32; simulation/evaluation: CPU FP32.",
        "Ordinary and guided PPO have matched games and optimizer exposure. Guided simulation cost is additional.",
    ]
    if duration:
        text.append(
            f"\nLocked duration: {duration['updates']} updates. Pilot accepted {duration['pilot_labels']} "
            f"corrections from {duration['pilot_games']} games. Pilot weights are excluded from scientific fits."
        )
    if clock:
        text.append(f"\nOriginal start: {clock['started']}; deadline: {clock['deadline']} (Unix seconds; never reset).")
    if measurements:
        text.append("\n| Panel | Model | Mode | Wins / games | Win rate |\n|---|---|---|---:|---:|")
        for row in measurements:
            score = (
                f"{row['wins']} / {row['games']}"
                if row["complete"]
                else f"incomplete ({row['games']}/{row['expected']})"
            )
            rate = f"{row['rate']:.1%}" if row["complete"] else "—"
            text.append(f"| {row['panel']} | {row['model']} | {row['mode']} | {score} | {rate} |")
    for name, claim in decision.get("claims", {}).items():
        text.append(
            f"\n{name}: mean gain {claim['mean_gain']:.1%}; paired 95% interval "
            f"[{claim['ci95'][0]:.1%}, {claim['ci95'][1]:.1%}]; qualifies={claim['qualifies']}."
        )
    if exposure:
        text.append(
            f"\nCommitted updates (including pilot): {len(exposure)}; PPO examples: "
            f"{sum(r['optimizer_examples'] for r in exposure)}; additional label simulation: "
            f"{sum(r['label_simulation_seconds'] for r in exposure):.1f} CPU-worker seconds. "
            "Per-update costs, coverage, and rejection counts are in report.json; "
            "fixed-state diagnostics are in diagnostics/."
        )
    text.append("\n" + result["interpretation"])
    if decision.get("assisted_command_change"):
        selected = decision["selected"]
        endpoint = duration["updates"]
        text.append(
            f"\nRecommended assisted command: `.venv/bin/python play.py --checkpoint "
            f"{out / 'exploratory' / selected / f'update_{endpoint}.pt'} --hybrid hand`."
        )
    else:
        text.append("\nRetain the incumbent assisted-player command.")
    (out / "report.md").write_text("\n".join(text) + "\n")
    print(encode(dict(decision=decision, report=str(out / "report.md"))).strip(), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status", "report"))
    parser.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "status":
        print(
            encode(
                {
                    name: read(out / f"{name}.json")
                    for name in ("status", "clock", "duration", "decision")
                    if (out / f"{name}.json").exists()
                }
            )
        )
        return
    with exclusive(out):
        try:
            {"prepare": prepare, "run": run, "report": report}[args.command](out)
        except BaseException as exc:
            progress(out, stage="interrupted" if isinstance(exc, KeyboardInterrupt) else "error", error=repr(exc))
            atomic(
                out / f"error_{time.time_ns()}.json",
                dict(error=repr(exc), traceback=traceback.format_exc(), diagnostics=getattr(exc, "diagnostics", {})),
            )
            if (out / "frozen.json").exists():
                report(out)
            raise


if __name__ == "__main__":
    main()
