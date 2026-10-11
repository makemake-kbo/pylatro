#!/usr/bin/env python3
"""User-authorized four-hour continuation of the guided pilot, then greedy evaluation.

The prior study remains closed and immutable. This follow-up explicitly waives
its pilot label-coverage gate, preserves training/RNG state, and uses fresh seeds.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from pylatro import load_game_data  # noqa: E402
from pylatro_agent.checkpoint import load_checkpoint_payload  # noqa: E402
from pylatro_agent.search_ppo_study import (  # noqa: E402
    assert_separation,
    atomic,
    digest,
    encode,
    exclusive,
    fingerprint,
    immutable,
    metadata_integers,
    metadata_name,
    read,
    save_torch,
)
from pylatro_agent.training.search_ppo import from_dict, require_mps, train_to, validate_resume  # noqa: E402
from pylatro_agent.vocab import build_vocab  # noqa: E402
from tools import search_guided_ppo_study as prior  # noqa: E402

DEFAULT = ROOT / "runs/search_guided_ppo_4h_20261004"
PILOT = prior.DEFAULT / "pilot/guided/update_1.pt"
NAMESPACE = "SEARCH-GUIDED-PPO-4H-20261004-V1"
TRAINING_SECONDS = 4 * 3600
EVALUATION_SECONDS = 2 * 3600
CAPACITY_UPDATES = 64
EVALUATION_GAMES = 256
TESTS = ["tests/test_continue_search_guided_ppo.py", "tests/test_search_guided_ppo.py"]


def sources():
    return {**prior.sources(), **{str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / TESTS[0]]}}


def reserve(exclusions):
    used, panels = set(exclusions), {}
    for name, count in (("training", CAPACITY_UPDATES * 64), ("evaluation", EVALUATION_GAMES)):
        values, index = [], 0
        while len(values) < count:
            h = hashlib.sha256(f"{NAMESPACE}:{name}:{index}".encode()).digest()
            seed = 1_000_000_000 + int.from_bytes(h[:8], "big") % 1_147_483_647
            index += 1
            if seed not in used:
                values.append(seed)
                used.add(seed)
        panels[name] = values
    assert_separation(panels, exclusions)
    return panels


def historical_exclusions(out):
    """Reuse the verified archive audit, and audit current metadata afresh."""
    prior.verify(prior.DEFAULT)
    audit = read(prior.DEFAULT / "metadata_audit.json")
    for record in audit["archives"]:
        if digest(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("Previously audited archive changed")
    exclusions = set(read(prior.DEFAULT / "historical_integer_exclusions.json"))
    for panel in read(prior.DEFAULT / "seeds.json").values():
        exclusions.update(panel)
    records = []
    for folder in ("eval", "datasets", "incumbents", "runs", "docs", "cache", "data", "archives"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and out not in path.parents and metadata_name(path):
                blob = path.read_bytes()
                exclusions.update(metadata_integers(blob))
                records.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(blob).hexdigest()))
    return exclusions, dict(
        prior_audit_sha256=digest(prior.DEFAULT / "metadata_audit.json"),
        archives_verified=audit["archives"],
        current_metadata=records,
        prior_reservations_excluded=True,
        reserved_outcomes_opened=False,
    )


def continuation_config(parent, seeds):
    config = from_dict(parent["public_ppo_config"])
    config.cohort_seeds = list(seeds)
    config.total_updates = CAPACITY_UPDATES
    return config


def branch_payload(parent, config, identity):
    """Reset only this leg's counters; preserve weights, Adam and all RNGs."""
    payload = copy.deepcopy(parent)
    previous = copy.deepcopy(parent["search_ppo_study"])
    payload.update(
        public_ppo_config=asdict(config),
        search_ppo_study=dict(
            identity=identity,
            completed_updates=0,
            seed_cursor=0,
            optimizer_examples=0,
            collected_transitions=0,
            slot_counters=copy.deepcopy(previous["slot_counters"]),
            guidance_rng=copy.deepcopy(previous["guidance_rng"]),
        ),
        guided_continuation_parent=dict(
            checkpoint_sha256=identity["parent_sha256"],
            study_state=previous,
            public_ppo_config=parent["public_ppo_config"],
            authorization="Continue guided PPO for approximately four hours despite pilot coverage gate",
        ),
    )
    validate_resume(payload, identity, config)
    return payload


def prepare(out):
    if (out / "frozen.json").exists() or (out / "clock.json").exists():
        raise ValueError("Continuation already prepared; use run to resume")
    require_mps()
    out.mkdir(parents=True, exist_ok=True)
    prior.progress(out, stage="correctness")
    with (out / "correctness.log").open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *TESTS, "-q"],
            cwd=ROOT,
            env={**os.environ, "SEARCH_PPO_ACTUAL_MPS_TEST": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError("Continuation correctness checks failed")
    prior.progress(out, stage="seed_audit")
    exclusions, audit = historical_exclusions(out)
    panels = reserve(exclusions)
    atomic(out / "seeds.json", panels)
    atomic(out / "metadata_audit.json", audit)
    atomic(out / "historical_integer_exclusions.json", sorted(exclusions))
    parent = load_checkpoint_payload(PILOT, "cpu")
    config = continuation_config(parent, panels["training"])
    frozen = dict(
        namespace=NAMESPACE,
        parent=str(PILOT),
        parent_sha256=digest(PILOT),
        base=str(prior.PARENT),
        base_sha256=digest(prior.PARENT),
        created_unix=time.time(),
        prior_study_sha256=digest(prior.DEFAULT / "frozen.json"),
        prior_decision_sha256=digest(prior.DEFAULT / "decision.json"),
        prior_coverage_gate="Explicitly overridden by user: continue guided PPO for ~4 hours",
        guidance_rules="unchanged",
        config=asdict(config),
        sources=sources(),
        seeds_sha256=digest(out / "seeds.json"),
        audit_sha256=digest(out / "metadata_audit.json"),
        exclusions_sha256=digest(out / "historical_integer_exclusions.json"),
        correctness_sha256=digest(out / "correctness.log"),
        training_seconds=TRAINING_SECONDS,
        evaluation_seconds=EVALUATION_SECONDS,
        evaluation_games=EVALUATION_GAMES,
        evaluation_mode="greedy",
        controls=["original_incumbent", "starting_guided_pilot"],
        training_stop="last completed update before four-hour deadline; "
        "forecast next using 1.1 * maximum observed update wall time",
        initial_update_seconds=read(prior.DEFAULT / "pilot/guided/runtime.json")["seconds"],
        checkpoint_selection="last complete update by runtime only, no intermediate evaluation",
        comparison="one trained continuation; paired game-seed bootstrap 10000 draws, seed 1729; "
        "no matched ordinary continuation",
        improvement_threshold=dict(mean_gain=0.05, ci95_lower_strictly_positive=True),
    )
    atomic(out / "frozen.json", frozen)
    identity = dict(
        arm="guided",
        replica="four_hour_continuation",
        frozen_sha256=digest(out / "frozen.json"),
        parent_sha256=digest(PILOT),
        config_sha256=fingerprint(asdict(config)),
    )
    branch = branch_payload(parent, config, identity)
    save_torch(out / "guided/initial.pt", branch)
    atomic(out / "branch.json", dict(identity=identity, initial_sha256=digest(out / "guided/initial.pt")))
    prior.progress(out, stage="prepared", clock_started=False)


def verify(out):
    frozen = read(out / "frozen.json")
    if frozen["sources"] != sources() or digest(PILOT) != frozen["parent_sha256"]:
        raise ValueError("Continuation source or starting checkpoint changed")
    if digest(prior.PARENT) != frozen["base_sha256"]:
        raise ValueError("Incumbent checkpoint changed")
    for name, field in (
        ("seeds.json", "seeds_sha256"),
        ("metadata_audit.json", "audit_sha256"),
        ("historical_integer_exclusions.json", "exclusions_sha256"),
        ("correctness.log", "correctness_sha256"),
    ):
        if digest(out / name) != frozen[field]:
            raise ValueError(f"Frozen continuation artifact changed: {name}")
    branch = read(out / "branch.json")
    if digest(out / "guided/initial.pt") != branch["initial_sha256"]:
        raise ValueError("Continuation initial checkpoint changed")
    panels = read(out / "seeds.json")
    assert_separation(panels, read(out / "historical_integer_exclusions.json"))
    config = from_dict(frozen["config"])
    if branch["identity"]["frozen_sha256"] != digest(out / "frozen.json"):
        raise ValueError("Branch provenance changed")
    validate_resume(load_checkpoint_payload(out / "guided/initial.pt", "cpu"), branch["identity"], config)
    return frozen, panels, config, branch["identity"]


def clock(out, now=None):
    path = out / "clock.json"
    if path.exists():
        value = read(path)
    else:
        started = time.time() if now is None else now
        value = dict(
            started=started,
            training_deadline=started + TRAINING_SECONDS,
            evaluation_deadline=started + TRAINING_SECONDS + EVALUATION_SECONDS,
        )
        immutable(path, value)
    if (
        value["training_deadline"] - value["started"] != TRAINING_SECONDS
        or value["evaluation_deadline"] - value["training_deadline"] != EVALUATION_SECONDS
    ):
        raise ValueError("Continuation deadline changed")
    immutable(out / "execution.json", dict(clock_sha256=fingerprint(value)))
    return value


def can_start_update(now, deadline, durations, initial):
    return now + 1.1 * max([initial, *durations]) <= deadline


def paired_result(candidate, baseline):
    candidate, baseline = np.asarray(candidate, float), np.asarray(baseline, float)
    if candidate.ndim != 1 or candidate.shape != baseline.shape or not len(candidate):
        raise ValueError("Need complete paired game results")
    diff = candidate - baseline
    rng = np.random.default_rng(1729)
    draws = rng.integers(len(diff), size=(10000, len(diff)))
    low, high = np.quantile(diff[draws].mean(axis=1), [0.025, 0.975])
    gain = float(diff.mean())
    return dict(
        candidate_wins=int(candidate.sum()),
        baseline_wins=int(baseline.sum()),
        games=len(diff),
        candidate_rate=float(candidate.mean()),
        baseline_rate=float(baseline.mean()),
        gain=gain,
        ci95=[float(low), float(high)],
        improvement=bool(gain >= 0.05 and low > 0),
        uncertainty="Conditional on this single trained continuation; paired over game seeds",
    )


def completed(out):
    paths = sorted((out / "guided").glob("update_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    indices = [int(p.stem.split("_")[-1]) for p in paths]
    if indices != list(range(1, len(paths) + 1)):
        raise ValueError("Missing intermediate continuation checkpoint")
    return paths


def update_metrics(out, checkpoint):
    path = out / "guided/cohorts" / checkpoint.stem / "update.json"
    if not path.exists():
        # Recover an interruption after the full checkpoint's atomic commit
        # but before its derived metric file was written.
        payload = load_checkpoint_payload(checkpoint, "cpu")
        atomic(path, payload["search_ppo_study"]["metrics"])
    return read(path)


def run(out):
    require_mps()
    torch.set_num_threads(1)
    frozen, panels, config, identity = verify(out)
    if (out / "result.json").exists():
        return report(out)
    times = clock(out)
    data = load_game_data()
    if not (out / "endpoint.json").exists():
        paths = completed(out)
        durations = [update_metrics(out, p)["seconds"] for p in paths]
        stop = "four-hour training window"
        for update in range(len(paths) + 1, CAPACITY_UPDATES + 1):
            if (out / "guided/rejected.json").exists():
                stop = "KL or partial-exposure rejection; retained last complete checkpoint"
                break
            if not can_start_update(
                time.time(), times["training_deadline"], durations, frozen["initial_update_seconds"]
            ):
                break
            started = time.monotonic()
            try:
                train_to(
                    PILOT,
                    out / "guided",
                    config,
                    identity,
                    update,
                    data=data,
                    vocab=build_vocab(data),
                    deadline=times["training_deadline"],
                    teach_records=prior.teach_records,
                    progress=lambda **kw: prior.progress(out, **kw),
                )
            except TimeoutError:
                stop = "training deadline reached; retained last complete checkpoint"
                break
            except RuntimeError as exc:
                if time.time() >= times["training_deadline"]:
                    atomic(
                        out / "deadline_interruption.json",
                        dict(error=repr(exc), diagnostics=getattr(exc, "diagnostics", {})),
                    )
                    stop = "training deadline reached; retained last complete checkpoint"
                    break
                if not (out / "guided/rejected.json").exists():
                    raise
                stop = "KL or partial-exposure rejection; retained last complete checkpoint"
                break
            durations.append(time.monotonic() - started)
            atomic(out / "training_runtime.json", dict(update_wall_seconds=durations))
        paths = completed(out)
        endpoint = paths[-1] if paths else out / "guided/initial.pt"
        immutable(
            out / "endpoint.json",
            dict(
                checkpoint=str(endpoint),
                sha256=digest(endpoint),
                completed_updates=len(paths),
                reason=stop,
                frozen_unix=time.time(),
                elapsed_training_seconds=time.time() - times["started"],
            ),
        )
    endpoint = read(out / "endpoint.json")
    if digest(endpoint["checkpoint"]) != endpoint["sha256"]:
        raise ValueError("Frozen endpoint checkpoint changed")
    if time.time() >= times["evaluation_deadline"]:
        raise TimeoutError("Continuation evaluation window exhausted; missing games remain incomplete")
    models = dict(incumbent=prior.PARENT, pilot=PILOT, continued=Path(endpoint["checkpoint"]))
    prior.evaluate(out, "fresh_greedy", models, ["greedy"], panels["evaluation"], times["evaluation_deadline"])
    wins = {name: prior.panel_wins(out, "fresh_greedy", name, "greedy", panels["evaluation"]) for name in models}
    result = dict(
        endpoint=endpoint,
        comparisons={name: paired_result(wins["continued"], wins[name]) for name in ("incumbent", "pilot")},
        evaluation_complete_unix=time.time(),
        label_coverage_gate_waived=True,
        scope="Combined guided PPO continuation; no estimate of added search benefit without an ordinary control",
    )
    atomic(out / "result.json", result)
    prior.progress(
        out,
        stage="complete",
        completed_updates=endpoint["completed_updates"],
        improved_over_incumbent=result["comparisons"]["incumbent"]["improvement"],
    )
    return report(out)


def report(out):
    result = read(out / "result.json") if (out / "result.json").exists() else None
    endpoint = read(out / "endpoint.json") if (out / "endpoint.json").exists() else None
    updates = []
    for p in completed(out):
        row = update_metrics(out, p)
        updates.append(
            dict(
                update=row["update"],
                games=row["games"],
                ppo_examples=row["optimizer"]["ppo_samples_processed"],
                accepted=row["accepted"],
                accepted_games=row["accepted_games"],
                selected_labels=row["labels"],
                rejections=row["rejections"],
                collection_seconds=row["collection_seconds"],
                optimization_seconds=row["optimization_seconds"],
                label_seconds=row["label_seconds"],
                label_simulation_seconds=row["label_simulation_seconds"],
            )
        )
    report_value = dict(
        complete=result is not None,
        result=result,
        endpoint=endpoint,
        updates=updates,
        clock=read(out / "clock.json") if (out / "clock.json").exists() else None,
        status=read(out / "status.json") if (out / "status.json").exists() else None,
    )
    atomic(out / "report.json", report_value)
    lines = [
        "# Four-hour guided PPO continuation",
        "",
        "The user explicitly authorized continuing despite the original pilot coverage gate. "
        "The search filter, PPO recipe, rewards, and KL guards are unchanged. "
        "This branch preserves the guided pilot weights, Adam state, policy RNGs, and label RNG.",
        "",
        f"Completed continuation updates: {len(updates)}; games: {sum(r['games'] for r in updates)}; "
        f"PPO examples: {sum(r['ppo_examples'] for r in updates)}; accepted teaching states: "
        f"{sum(r['accepted'] for r in updates)}.",
    ]
    if endpoint:
        lines += [
            "",
            f"Training stop: {endpoint['reason']}. Elapsed training window: "
            f"{endpoint['elapsed_training_seconds'] / 3600:.2f} hours.",
            "",
            f"Endpoint: [{Path(endpoint['checkpoint']).name}]({endpoint['checkpoint']}).",
        ]
    if result:
        lines += [
            "",
            "Primary evaluation: 256 untouched common seeds, greedy play, CPU FP32, no search.",
            "",
            "| Comparison | Continued wins | Baseline wins | Gain | Paired 95% interval | Meets 5-point gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for name, row in result["comparisons"].items():
            lines.append(
                f"| {name} | {row['candidate_wins']}/{row['games']} | {row['baseline_wins']}/{row['games']} | "
                f"{row['gain']:.1%} | [{row['ci95'][0]:.1%}, {row['ci95'][1]:.1%}] | {row['improvement']} |"
            )
        lines += [
            "",
            "The paired bootstrap uses 10,000 draws and seed 1729. Uncertainty conditions on one trained "
            "continuation. This evaluates combined PPO continuation with optional search labels. "
            "An additional benefit from search alone is not established without a matched ordinary-PPO continuation. "
            "The incumbent and assisted-player recommendation have not been changed.",
        ]
    else:
        lines += ["", "Evaluation is incomplete; no win-rate conclusion is available."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(encode(dict(complete=result is not None, updates=len(updates), report=str(out / "report.md"))), flush=True)
    return report_value


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
                    for name in ("status", "clock", "endpoint", "result")
                    if (out / f"{name}.json").exists()
                }
            )
        )
        return
    with exclusive(out):
        try:
            {"prepare": prepare, "run": run, "report": report}[args.command](out)
        except BaseException as exc:
            prior.progress(out, stage="interrupted" if isinstance(exc, KeyboardInterrupt) else "error", error=repr(exc))
            atomic(out / f"error_{time.time_ns()}.json", dict(error=repr(exc), traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
