"""Frozen 24-hour search scaling and descriptive loss study.

Use: .venv/bin/python tools/search_scaling_study.py prepare|run|report|status
Implementation/correctness precede `run`. Only `run` starts the immutable clock.
No partial panel advances. Reserved historical outcomes are never read.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pylatro_agent.hybrid import HybridPolicy, PlannerConfig  # noqa: E402
from pylatro_agent.search_diagnostics import (  # noqa: E402
    historical_groups,
    inspect_trace,
    replay_difference,
    restore_snapshot,
)

DEFAULT = ROOT / "runs/search_scaling_20261003"
HISTORICAL = ROOT / "runs/public_hybrid_20261002"
NAMESPACE = "PUBLIC-SEARCH-SCALING-20261003-V1"
PROFILES = {
    "incumbent": dict(samples=4, seconds=3.0),
    "moderate": dict(samples=16, seconds=3.0),
    "high32": dict(samples=32, seconds=10.0),
    "high64": dict(samples=64, seconds=10.0),
}
TESTS = [
    "tests/test_public_hybrid.py",
    "tests/test_hybrid_study.py",
    "tests/test_search_scaling.py",
    "tests/test_action_masking.py",
    "tests/test_joker_layout.py",
    "tests/test_deck_exhaustion.py",
    "tests/test_rng.py",
    "tests/test_heuristic_joker_cache.py",
    "tests/test_incumbent_reference.py",
]


def native(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def encode(value):
    return json.dumps(value, default=native, sort_keys=True, allow_nan=False, indent=2) + "\n"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w") as f:
        f.write(encode(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def immutable(path, value):
    if path.exists():
        if read(path) != json.loads(encode(value)):
            raise ValueError(f"Mismatched frozen artifact: {path}")
    else:
        atomic(path, value)


@contextmanager
def exclusive(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "owner.lock").open("a+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def sources():
    paths = [p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".json", ".so"}]
    paths += [Path(__file__), ROOT / "play.py", ROOT / "scripts/compile_cython.py"]
    paths += [ROOT / p for p in TESTS]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def metadata_name(path):
    p = Path(path)
    # Only metadata by name; never open outcome/trajectory/label/result records.
    name = p.name.lower()
    return (
        p.suffix in {".json", ".jsonl", ".md", ".yaml", ".yml"}
        and not any(w in name for w in ("outcome", "result", "trajectory", "label", "games"))
        and any(
            w in name
            for w in ("seed", "manifest", "protocol", "config", "generation", "reservation", "exclusion", "registry")
        )
    )


def seed_panels(namespace=NAMESPACE):
    return {
        stage: [f"{namespace}-{stage}-{i:04d}" for i in range(n)]
        for stage, n in (("pilot", 8), ("development", 96), ("confirmation", 256))
    }


def assert_separation(panels, exclusions=()):
    flat = [s for rows in panels.values() for s in rows]
    if len(flat) != len(set(flat)) or not set(flat).isdisjoint(exclusions):
        raise ValueError("Seed reservations overlap")


def metadata_audit(out):
    """Stream archived metadata only; hash other archive content opaquely."""
    records = []
    namespace = NAMESPACE.encode()

    def check(name, blob):
        if namespace in blob:
            raise ValueError(f"New namespace already used in retained metadata: {name}")
        records.append(dict(path=name, sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob)))

    for folder in ("eval", "datasets", "incumbents", "runs", "docs", "cache"):
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and metadata_name(p) and out not in p.parents:
                check(str(p.relative_to(ROOT)), p.read_bytes())
    archive = ROOT / "archives/experiments_20261002/evidence.tar.gz"
    inventory = ROOT / "archives/experiments_20261002/inventory.jsonl"
    wanted = {
        r["path"]
        for line in inventory.open()
        if (r := json.loads(line))["action"] == "archive" and metadata_name(r["path"])
    }
    found = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name in wanted:
                check(f"{archive.relative_to(ROOT)}::{member.name}", tar.extractfile(member).read())
                found.add(member.name)
    if found != wanted:
        raise ValueError("Archived metadata inventory is incomplete")
    old = read(HISTORICAL / "reservation.json")["invalidated_attempt_seed_exclusions"]
    old_seeds = [s for panel in old.values() for s in panel]
    old_seeds += [s for panel in read(HISTORICAL / "seeds.json").values() for s in panel]
    assert_separation(seed_panels(), old_seeds)
    return dict(
        metadata=records,
        invalidated_and_completed_exclusions=old_seeds,
        archive_sha256=digest(archive),
        inventory_sha256=digest(inventory),
        rationale="Fresh nonnumeric namespace absent from retained training/evaluation metadata, "
        "including archived metadata; disjoint from decimal historical seeds and both "
        "invalidated reservations. Reserved historical outcomes were not opened.",
    )


def history():
    seeds = read(HISTORICAL / "seeds.json")["confirmation"]
    panels, pins = {}, {}
    for mode in ("off", "hand"):
        rows = []
        for seed in seeds:
            p = HISTORICAL / "confirmation/ppo_replica1" / mode / f"{seed}.json"
            row = read(p)
            if row["seed"] != seed or row["mode"] != mode or row["model"] != "ppo_replica1":
                raise ValueError("Historical identity mismatch")
            rows.append(row)
            pins[str(p.relative_to(ROOT))] = digest(p)
        panels[mode] = rows
    return historical_groups(panels["off"], panels["hand"]), pins


def prepare(out):
    if (out / "frozen.json").exists() or (out / "clock.json").exists():
        raise ValueError("Study is already frozen; use run to resume")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "correctness.log").open("w") as f:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *TESTS, "-q"], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT
        )
    atomic(
        out / "correctness.json",
        dict(exit_code=result.returncode, tests=TESTS, log_sha256=digest(out / "correctness.log")),
    )
    if result.returncode:
        raise RuntimeError("Preclock correctness tests failed")
    import pylatro.scoring as scoring
    import pylatro_agent.heuristic as heuristic
    import pylatro_agent.training.fast_runner as fast

    compiled = {m.__name__: m.__file__ for m in (fast, heuristic, scoring)}
    if not all(p.endswith(".so") for p in compiled.values()):
        raise RuntimeError("Expected compiled simulator path")
    print("Checking retained metadata and seed exclusions (no gameplay clock yet)", flush=True)
    exclusions = metadata_audit(out)
    hist, hist_pins = history()
    registry = read(ROOT / "incumbents/registry.json")
    checkpoint = registry["ppo_replica1"]
    if digest(ROOT / checkpoint["path"]) != checkpoint["sha256"]:
        raise ValueError("Checkpoint hash differs from incumbent registry")
    config = dict(
        version=1,
        checkpoint=checkpoint,
        profiles=PROFILES,
        workers=4,
        torch_threads=1,
        device="cpu",
        precision="fp32",
        mode="hand",
        observation="public-information-v2",
        deck="b_blue",
        stake=1,
        win_ante=8,
        greedy=True,
        temperature=1.0,
        hand_candidates=6,
        hand_steps=60,
        shop_steps=100,
        shop_candidates=4,
        wall_seconds=86400,
        bootstrap_draws=10000,
        bootstrap_seed=1729,
        advancement_points=5,
        development_games=96,
        confirmation_games=256,
        diagnostic_selection="all 32 regressions; SHA256 ascending: 16 shared losses, 8 rescues, 8 shared wins",
        diagnostic_batches=[0, 10000],
        high_selection="64 if timing projection plus 25% fits, else 32 if fits, else infeasible",
        runtime_projection="96 games per three arms + 256 games for incumbent and slower possible challenger; "
        "pilot mean game seconds, four workers, two maximum-game tail allowances, times 1.25",
        challenger_selection="most wins with at least 5 percentage points gain; latency then requested worlds",
        adoption="at least 5 percentage points and paired bootstrap 95% lower bound > 0; complete integrity",
        scientific_boundary_audit="private-state projection perturbation every ten decisions",
    )
    atomic(out / "config.json", config)
    atomic(out / "seeds.json", seed_panels())
    atomic(out / "reservation.json", exclusions)
    atomic(out / "historical_diagnosis.json", hist)
    pins = sources()
    artifacts = {
        name: digest(out / name)
        for name in ("config.json", "seeds.json", "reservation.json", "historical_diagnosis.json", "correctness.json")
    }
    frozen = dict(
        sources=pins,
        artifacts=artifacts,
        historical=hist_pins,
        compiled=compiled,
        checkpoint_sha256=checkpoint["sha256"],
    )
    frozen["id"] = fingerprint(frozen)
    atomic(out / "frozen.json", frozen)
    with tarfile.open(out / "frozen_runtime.tar.gz", "w:gz") as tar:
        for name in pins:
            tar.add(ROOT / name, arcname=name)
    print(f"Frozen {len(pins)} source files and 360 new seeds. Clock has not started.", flush=True)


def verify(out):
    frozen = read(out / "frozen.json")
    if fingerprint({k: v for k, v in frozen.items() if k != "id"}) != frozen["id"]:
        raise ValueError("Invalid freeze fingerprint")
    if sources() != frozen["sources"]:
        raise ValueError("Frozen source or compiled module changed")
    for name, sha in frozen["artifacts"].items():
        if digest(out / name) != sha:
            raise ValueError(f"Frozen configuration changed: {name}")
    for name, sha in frozen["historical"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Historical confirmation changed")
    config = read(out / "config.json")
    if digest(ROOT / config["checkpoint"]["path"]) != frozen["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    return frozen, config


def clock_start(out, now=None):
    path = out / "clock.json"
    if path.exists():
        clock = read(path)
        if clock["deadline_unix"] != clock["started_unix"] + 86400:
            raise ValueError("Invalid original wall clock")
        return clock
    now = time.time() if now is None else now
    clock = dict(started_unix=now, deadline_unix=now + 86400)
    atomic(path, clock)
    return clock


def job_identity(stage, arm, seed, frozen_id, config):
    profile = dict(samples=4, seconds=3.0) if arm == "baseline" else PROFILES[arm]
    return dict(
        stage=stage,
        arm=arm,
        seed=seed,
        frozen_id=frozen_id,
        planner=asdict(PlannerConfig(mode="off" if arm == "baseline" else "hand", **profile)),
        checkpoint_sha256=config["checkpoint"]["sha256"],
        sample=False,
    )


def validate_row(row, identity, deadline=None):
    if row.get("identity") != identity or row.get("identity_sha256") != fingerprint(identity):
        raise ValueError("Mismatched game configuration")
    if (
        row.get("complete") is not True
        or type(row.get("won")) is not bool
        or row.get("seed") != identity["seed"]
        or row.get("planner_config") != identity["planner"]
        or row.get("sample") is not False
        or row.get("mode") != identity["planner"]["mode"]
        or row.get("illegal_actions") != 0
        or row.get("boundary_violations") != 0
        or row.get("interpretation") != "public-information-v2"
        or row.get("boundary_audits", 0) < 1
    ):
        raise ValueError("Game integrity requirements failed")
    if deadline is not None and row.get("finished_unix", float("inf")) > deadline:
        raise ValueError("Game completed after the original deadline")
    if len(row["decision_seconds"]) != row["steps"] or sum(row["sample_counts"].values()) != row["searches"]:
        raise ValueError("Inconsistent compact search accounting")
    if any(int(k) > identity["planner"]["samples"] for k in row["sample_counts"]):
        raise ValueError("Too many completed worlds")
    return row


def read_panel(out, stage, arm, seeds, frozen_id, config, deadline):
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seed outcomes")
    folder = out / stage / arm
    expected = {f"{s}.json" for s in seeds}
    actual = {p.name for p in folder.glob("*.json") if not p.name.endswith((".error.json", ".timing.json"))}
    if actual != expected:
        raise ValueError(f"Incomplete or duplicate panel {stage}/{arm}: {len(actual)}/{len(expected)}")
    return [
        validate_row(read(folder / f"{s}.json"), job_identity(stage, arm, s, frozen_id, config), deadline)
        for s in seeds
    ]


def worker(job):
    from pylatro_agent.hybrid_game import play_game

    path = Path(job["path"])
    trace_path = path.with_suffix(".trace.jsonl.gz")
    tmp_trace = trace_path.with_name(trace_path.name + f".{os.getpid()}.tmp")
    identity = job["identity"]
    started = time.time()
    try:

        def execute(sink=None):
            p = identity["planner"]
            return play_game(
                job["checkpoint"],
                identity["seed"],
                mode=p["mode"],
                seconds=p["seconds"],
                search_samples=p["samples"],
                deadline=job["deadline"],
                boundary_audit=True,
                trace_sink=sink,
            )

        if identity["stage"] == "diagnostic":
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(tmp_trace, "wt", compresslevel=3) as f:
                result = execute(lambda row: f.write(json.dumps(row, default=native, allow_nan=False) + "\n"))
            os.replace(tmp_trace, trace_path)
            result["trace_sha256"] = digest(trace_path)
        else:
            result = execute()
        result.update(
            identity=identity, identity_sha256=fingerprint(identity), started_unix=started, finished_unix=time.time()
        )
        validate_row(result, identity, job["deadline"])
        # Pilot timing contains no win/ante/outcome or trajectory fields.
        if identity["stage"] == "pilot":
            timing = {
                k: result[k]
                for k in (
                    "seconds",
                    "steps",
                    "decision_seconds",
                    "searches",
                    "changes",
                    "fallbacks",
                    "sample_counts",
                    "search_stops",
                    "boundary_audits",
                )
            }
            timing.update(identity=identity, outcome_sha256=fingerprint(result), complete=True)
            atomic(path.with_suffix(".timing.json"), timing)
        atomic(path, result)
        return dict(complete=True, seed=identity["seed"])
    except Exception:
        atomic(
            path.with_suffix(".error.json"),
            dict(identity=identity, started_unix=started, finished_unix=time.time(), traceback=traceback.format_exc()),
        )
        return dict(complete=False, seed=identity["seed"])


def reevaluate(job):
    from pylatro import load_game_data

    path = Path(job["path"])
    try:
        trace = Path(job["trace"])
        info = inspect_trace(trace)
        first = info.pop("first_eligible")
        results = []
        if first is not None:
            public = restore_snapshot(first["snapshot"])
            data = load_game_data()
            for name, profile in PROFILES.items():
                for start in (0, 10000):
                    if time.time() >= job["deadline"]:
                        raise TimeoutError("Original deadline reached during sensitivity diagnostics")
                    policy = HybridPolicy(
                        lambda s: first["proposal"], data, PlannerConfig(mode="hand", sample_start=start, **profile)
                    )
                    decision = policy.decide(public, trace=True)
                    record = decision.trace
                    record.pop("snapshot")
                    results.append(dict(profile=name, **record))
        if time.time() >= job["deadline"]:
            raise TimeoutError("Original deadline reached during sensitivity diagnostics")
        atomic(
            path,
            dict(
                identity=job["identity"],
                complete=True,
                trace_sha256=digest(trace),
                first_eligible=first,
                final_blind=info,
                reevaluations=results,
                finished_unix=time.time(),
            ),
        )
        return dict(complete=True)
    except Exception:
        atomic(path.with_suffix(".error.json"), dict(identity=job["identity"], traceback=traceback.format_exc()))
        return dict(complete=False)


def pool_jobs(out, stage, jobs, deadline, fn=worker):
    if not jobs:
        return
    print(f"Stage {stage}: {len(jobs)} pending units", flush=True)
    atomic(out / "progress.json", dict(stage=stage, pending=len(jobs), completed_now=0, time=time.time()))
    ctx = multiprocessing.get_context("spawn")
    failed = False
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=ctx) as pool:
        queue, active = iter(jobs), {}

        def submit():
            if not failed and time.time() < deadline and (job := next(queue, None)) is not None:
                active[pool.submit(fn, job)] = job

        for _ in range(4):
            submit()
        finished, last_print = 0, time.time()
        while active:
            done, _ = concurrent.futures.wait(active, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
            if time.time() >= deadline:
                # Workers also check between decisions. Kill a hung C-level call at the hard cap.
                for process in pool._processes.values():
                    process.terminate()
                raise TimeoutError("Original 24-hour gameplay deadline reached")
            for future in done:
                result = future.result()
                del active[future]
                finished += 1
                failed |= not result["complete"]
                submit()
            if done:
                atomic(
                    out / "progress.json",
                    dict(
                        stage=stage,
                        pending=len(jobs) - finished,
                        completed_now=finished,
                        error=failed,
                        time=time.time(),
                    ),
                )
            if failed or time.time() - last_print >= 600:
                print(f"{stage}: {finished}/{len(jobs)} pending units finished; error={failed}", flush=True)
                last_print = time.time()
        if failed:
            raise RuntimeError(f"{stage} has an integrity/execution error; panel cannot advance")
        if finished != len(jobs):
            raise TimeoutError("Original deadline left missing games; panel incomplete")
    print(f"Stage {stage}: complete", flush=True)


def play_panel(out, stage, arms, seeds, frozen, config, deadline):
    jobs = []
    for seed in seeds:
        for arm in arms:
            path = out / stage / arm / f"{seed}.json"
            identity = job_identity(stage, arm, seed, frozen["id"], config)
            if path.with_suffix(".error.json").exists():
                raise ValueError(f"Recorded execution error requires investigation: {path}")
            if path.exists():
                if stage == "pilot" and not (out / "profile_lock.json").exists():
                    # Validate sealed file by bytes, without reading its outcome.
                    timing = read(path.with_suffix(".timing.json"))
                    if timing["identity"] != identity or timing["outcome_sha256"] != digest(path):
                        raise ValueError("Sealed pilot identity/hash mismatch")
                else:
                    row = validate_row(read(path), identity, deadline)
                    if stage == "diagnostic" and digest(path.with_suffix(".trace.jsonl.gz")) != row["trace_sha256"]:
                        raise ValueError("Diagnostic trace changed")
                continue
            jobs.append(
                dict(
                    identity=identity,
                    path=str(path),
                    checkpoint=str(ROOT / config["checkpoint"]["path"]),
                    deadline=deadline,
                )
            )
    pool_jobs(out, stage, jobs, deadline)
    if stage != "pilot":
        return {a: read_panel(out, stage, a, seeds, frozen["id"], config, deadline) for a in arms}
    return None


def runtime_selection(timings, remaining):
    if set(timings) != set(PROFILES) or any(len(r) != 8 for r in timings.values()):
        raise ValueError("Runtime selection requires all eight pilot seeds in every profile")
    means = {a: float(np.mean([r["seconds"] for r in rows])) for a, rows in timings.items()}
    projections = {}
    for high in ("high64", "high32"):
        worker_seconds = 96 * (means["incumbent"] + means["moderate"] + means[high])
        worker_seconds += 256 * (means["incumbent"] + max(means["moderate"], means[high]))
        tail = 2 * max(r["seconds"] for a in ("incumbent", "moderate", high) for r in timings[a])
        projections[high] = dict(worker_seconds=worker_seconds, wall_with_margin=1.25 * (worker_seconds / 4 + tail))
    selected = next((h for h in ("high64", "high32") if projections[h]["wall_with_margin"] <= remaining), None)
    return dict(
        high=selected,
        mean_game_seconds=means,
        projections=projections,
        remaining_seconds=remaining,
        margin=1.25,
        outcome_fields_used=False,
    )


def paired(base, candidate):
    if not base or [r["seed"] for r in base] != [r["seed"] for r in candidate]:
        raise ValueError("Identical complete paired seeds required")
    if len({r["seed"] for r in base}) != len(base) or not all(r["complete"] for r in base + candidate):
        raise ValueError("Duplicate or incomplete paired games")
    diff = np.array([int(c["won"]) - int(b["won"]) for b, c in zip(base, candidate, strict=True)])
    boot = diff[np.random.default_rng(1729).integers(0, len(diff), (10000, len(diff)))].mean(axis=1) * 100
    return dict(
        points=float(diff.mean() * 100),
        interval95=np.quantile(boot, [0.025, 0.975]).tolist(),
        gained=int((diff == 1).sum()),
        lost=int((diff == -1).sum()),
        games=len(base),
    )


def wilson(wins, count):
    z, p = 1.959963984540054, wins / count
    center = (p + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * ((p * (1 - p) / count + z * z / (4 * count * count)) ** 0.5) / (1 + z * z / count)
    return [100 * (center - radius), 100 * (center + radius)]


def summary(rows):
    latency = [t for r in rows for t in r["decision_seconds"]]
    stops, counts = Counter(), Counter()
    for r in rows:
        stops.update(r["search_stops"])
        counts.update(r["sample_counts"])
    wins = sum(r["won"] for r in rows)
    return dict(
        games=len(rows),
        wins=wins,
        rate=100 * wins / len(rows),
        interval95=wilson(wins, len(rows)),
        changes=sum(r["changes"] for r in rows),
        searches=sum(r["searches"] for r in rows),
        fallbacks=sum(r["fallbacks"] for r in rows),
        completed_samples=dict(counts),
        search_stops=dict(stops),
        paired_sample_sets=sum(r["paired_sample_sets"] for r in rows),
        worker_hours=sum(r["seconds"] for r in rows) / 3600,
        mean_decision_seconds=float(np.mean(latency)),
        p95_decision_seconds=float(np.quantile(latency, 0.95)),
        max_decision_seconds=max(latency),
        boundary_audits=sum(r["boundary_audits"] for r in rows),
    )


def select_challenger(panels):
    base = panels["incumbent"]
    candidates = []
    for arm, rows in panels.items():
        if arm == "incumbent":
            continue
        paired(base, rows)
        gain = sum(r["won"] for r in rows) - sum(r["won"] for r in base)
        if gain * 100 >= 5 * len(base):
            candidates.append(arm)
    return min(
        candidates,
        key=lambda a: (
            -sum(r["won"] for r in panels[a]),
            np.mean([t for r in panels[a] for t in r["decision_seconds"]]),
            PROFILES[a]["samples"],
        ),
        default=None,
    )


def adoption(base, candidate):
    if len(base) != 256 or len(candidate) != 256:
        raise ValueError("Confirmation needs all 256 pairs")
    stats = paired(base, candidate)
    gain = sum(r["won"] for r in candidate) - sum(r["won"] for r in base)
    return bool(gain * 100 >= 5 * 256 and stats["interval95"][0] > 0), stats


def diagnosis(out, panels, hist):
    pairs = {r["seed"]: r for r in hist["pairs"]}
    observations = []
    for arm, rows in panels.items():
        for row in rows:
            seed = row["seed"]
            path = out / "diagnostic" / arm / f"{seed}.trace.jsonl.gz"
            info = inspect_trace(path)
            info.pop("first_eligible")
            observations.append(
                dict(
                    seed=seed,
                    arm=arm,
                    historical_group=pairs[seed]["group"],
                    won=row["won"],
                    replay_differences=replay_difference(pairs[seed][arm], row),
                    **info,
                )
            )
    sensitivity = []
    for seed in sorted({r["seed"] for r in observations}):
        result = read(out / "sensitivity" / f"{seed}.json")
        values = {(r["profile"], r["config"]["sample_start"]): r for r in result["reevaluations"]}
        for profile in PROFILES:
            if (profile, 0) not in values:
                continue
            a, b = values[profile, 0], values[profile, 10000]
            sensitivity.append(
                dict(
                    seed=seed,
                    profile=profile,
                    same_candidates=a["candidates"] == b["candidates"],
                    selected_primary=a["selection"],
                    selected_separate=b["selection"],
                    selection_changed=a["selection"] != b["selection"],
                    primary_samples=len(a["completed_scores"]),
                    separate_samples=len(b["completed_scores"]),
                    primary_stop=a["stopping_reason"],
                    separate_stop=b["stopping_reason"],
                    both_full=len(a["completed_scores"]) == len(b["completed_scores"]) == PROFILES[profile]["samples"],
                )
            )
    result = dict(
        historical_counts=hist["counts"],
        observations=observations,
        sampling_sensitivity=sensitivity,
        limits=[
            "The 64 seeds are outcome-stratified diagnostics, not a win-rate evaluation panel.",
            "Historical compact records cannot establish exact trajectory reproduction; compare available fields.",
            "Timing truncation and sampling variation are reported separately; neither proves a strategic cause.",
            "Entry roster, hand levels, unused resources and score deficits are observations, not causal effects.",
            "Continuation and longer-term strategy weaknesses remain hypotheses absent direct intervention.",
            "Failure to find a rescue does not establish that a seed was unwinnable.",
        ],
    )
    atomic(out / "loss_diagnosis.json", result)
    return result


def run(out):
    frozen, config = verify(out)
    if (out / "verdict.json").exists() and read(out / "verdict.json")["verdict"] in {"adopt", "retain"}:
        report(out)
        return
    times = clock_start(out)
    deadline = times["deadline_unix"]
    if time.time() >= deadline:
        raise TimeoutError("The original 24-hour deadline has expired")
    hist = read(out / "historical_diagnosis.json")
    diagnostic_seeds = [s for v in hist["diagnostic_subset"].values() for s in v]
    panels = play_panel(out, "diagnostic", ["baseline", "incumbent"], diagnostic_seeds, frozen, config, deadline)
    jobs = []
    for seed in diagnostic_seeds:
        path = out / "sensitivity" / f"{seed}.json"
        trace = out / "diagnostic/incumbent" / f"{seed}.trace.jsonl.gz"
        identity = dict(
            seed=seed, frozen_id=frozen["id"], trace_sha256=digest(trace), profiles=PROFILES, batches=[0, 10000]
        )
        if path.with_suffix(".error.json").exists():
            raise ValueError("Recorded sensitivity error requires investigation")
        if path.exists():
            r = read(path)
            if r["identity"] != identity or not r["complete"] or r["finished_unix"] > deadline:
                raise ValueError("Mismatched resumed sensitivity analysis")
        else:
            jobs.append(dict(path=str(path), trace=str(trace), identity=identity, deadline=deadline))
    pool_jobs(out, "sensitivity", jobs, deadline, reevaluate)
    diagnosis(out, panels, hist)
    seeds = read(out / "seeds.json")
    play_panel(out, "pilot", list(PROFILES), seeds["pilot"], frozen, config, deadline)
    timings = {}
    for arm in PROFILES:
        timings[arm] = []
        for seed in seeds["pilot"]:
            path = out / "pilot" / arm / f"{seed}.timing.json"
            r = read(path)
            if r["identity"] != job_identity("pilot", arm, seed, frozen["id"], config):
                raise ValueError("Pilot timing identity mismatch")
            if r["outcome_sha256"] != digest(path.with_name(f"{seed}.json")):
                raise ValueError("Pilot sealed result changed")
            timings[arm].append(r)
    lock = out / "profile_lock.json"
    if not lock.exists():
        immutable(
            lock,
            dict(frozen_id=frozen["id"], locked_unix=time.time(), **runtime_selection(timings, deadline - time.time())),
        )
    selected = read(lock)
    expected_selection = runtime_selection(timings, selected["remaining_seconds"])
    if any(selected.get(k) != v for k, v in expected_selection.items()):
        raise ValueError("Timing-only profile lock was modified")
    if selected["frozen_id"] != frozen["id"]:
        raise ValueError("Profile lock identity mismatch")
    # Pilot wins are first opened after the scientific profiles are locked, solely for integrity validation.
    for arm in PROFILES:
        read_panel(out, "pilot", arm, seeds["pilot"], frozen["id"], config, deadline)
    if selected["high"] is None:
        atomic(
            out / "verdict.json",
            dict(
                verdict="incomplete",
                reason="infeasible under remaining 24-hour budget",
                elapsed_seconds=time.time() - times["started_unix"],
            ),
        )
        report(out)
        return
    print(f"Scientific profiles locked: incumbent, moderate, {selected['high']}", flush=True)
    arms = ["incumbent", "moderate", selected["high"]]
    development = play_panel(out, "development", arms, seeds["development"], frozen, config, deadline)
    challenger = select_challenger(development)
    selection = dict(
        challenger=challenger,
        frozen_id=frozen["id"],
        summaries={a: summary(r) for a, r in development.items()},
        paired={a: paired(development["incumbent"], development[a]) for a in arms[1:]},
        outcomes={
            a: {r["seed"]: digest(out / "development" / a / f"{r['seed']}.json") for r in rows}
            for a, rows in development.items()
        },
    )
    immutable(out / "challenger_lock.json", selection)
    if challenger is None:
        verdict = dict(verdict="retain", reason="Neither challenger gained five percentage points in development")
    else:
        print(f"Challenger locked for untouched confirmation: {challenger}", flush=True)
        confirmation = play_panel(
            out, "confirmation", ["incumbent", challenger], seeds["confirmation"], frozen, config, deadline
        )
        passed, stats = adoption(confirmation["incumbent"], confirmation[challenger])
        atomic(
            out / "confirmation_result.json",
            dict(
                challenger=challenger,
                paired=stats,
                summaries={a: summary(r) for a, r in confirmation.items()},
                passed=passed,
            ),
        )
        verdict = dict(
            verdict="adopt" if passed else "retain",
            reason="Fixed confirmation criteria evaluated",
            challenger=challenger,
            paired=stats,
        )
    verify(out)
    verdict.update(elapsed_seconds=time.time() - times["started_unix"], frozen_id=frozen["id"])
    atomic(out / "verdict.json", verdict)
    outcome_pins = {
        str(p.relative_to(out)): digest(p)
        for stage in ("diagnostic", "sensitivity", "pilot", "development", "confirmation")
        for p in (out / stage).rglob("*.json")
    }
    atomic(
        out / "final_verification.json",
        dict(
            sources_match=True,
            checkpoint_matches=True,
            outcome_hashes=outcome_pins,
            elapsed_seconds=verdict["elapsed_seconds"],
            original_deadline_unix=deadline,
            errors=[str(p.relative_to(out)) for p in out.rglob("*.error.json")],
        ),
    )
    report(out)


def report(out):
    verdict = (
        read(out / "verdict.json")
        if (out / "verdict.json").exists()
        else dict(verdict="incomplete", reason="Study in progress")
    )
    lines = [
        "# Search scaling and loss diagnosis",
        "",
        f"**Verdict: {verdict['verdict'].upper()}** — {verdict['reason']}.",
        "",
        "PPO replica 1 remains frozen. Blue Deck, White Stake, Ante 8; greedy CPU FP32; at most four workers "
        "and one Torch thread each. Public-information-v2, six candidate cap, existing continuation and scoring. "
        "Shop and consumable policy behavior is unchanged.",
        "",
    ]
    if (out / "clock.json").exists():
        c = read(out / "clock.json")
        elapsed = verdict.get("elapsed_seconds", time.time() - c["started_unix"])
        lines += [
            f"Original gameplay clock: {elapsed / 3600:.3f} / 24 hours, including diagnostics, pilot and interruptions. "  # noqa: E501
            "Correctness tests and implementation preceded this clock.",
            "",
        ]
    if (out / "historical_diagnosis.json").exists():
        hist = read(out / "historical_diagnosis.json")
        lines += [
            "## Historical confirmation and descriptive diagnosis",
            "",
            "All 400 historical pairs: 253/400 incumbent wins (63.25%; Wilson 95% "
            f"[{wilson(253, 400)[0]:.2f}, {wilson(253, 400)[1]:.2f}]%), versus 200/400 baseline wins "
            f"(50.00%; Wilson 95% [{wilson(200, 400)[0]:.2f}, {wilson(200, 400)[1]:.2f}]%). "
            "The original paired gain was +13.25 points (95% paired bootstrap [+8.00, +18.50]).",
            "",
            f"Strata: {hist['counts']}. Loss ante counts: {hist['loss_antes']}.",
            "",
            "Detailed records: [historical_diagnosis.json](historical_diagnosis.json). Fixed diagnostic subset: "
            "all 32 regressions, then SHA-256 ascending 16 shared losses, eight rescues and eight shared wins.",
            "",
        ]
    if (out / "loss_diagnosis.json").exists():
        d = read(out / "loss_diagnosis.json")
        lines += [
            "| Replay arm | Games | Outcome differences from history | Any recorded difference | Losses | Median loss score / target | Losses with unused discards / consumables |",  # noqa: E501
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm in ("baseline", "incumbent"):
            rows = [r for r in d["observations"] if r["arm"] == arm]
            losses = [r for r in rows if not r["won"] and r["final"]]
            fractions = [r["final"]["score_fraction"] for r in losses]
            lines.append(
                f"| {arm} | {len(rows)} | {sum('won' in r['replay_differences'] for r in rows)} | "
                f"{sum(bool(r['replay_differences']) for r in rows)} | {len(losses)} | "
                f"{float(np.median(fractions)) if fractions else 0:.3f} | "
                f"{sum(r['final']['discards'] > 0 for r in losses)} / {sum(bool(r['final']['consumables']) for r in losses)} |"  # noqa: E501
            )
        lines += [
            "",
            "Entry rosters, hand levels, boss restrictions, public resources, score deficits and final-blind "
            "overrides are recorded per replay in [loss_diagnosis.json](loss_diagnosis.json); compressed public "
            "decision traces are under `diagnostic/`. These strata are deliberately selected, so their replay "
            "win rate is not a population estimate.",
            "",
            "### Separate-batch sampling sensitivity",
            "",
            "The first eligible hand decision in each hybrid replay's final blind was reevaluated at all four "
            "fixed profiles, using indices 0 onward and a separate batch starting at 10,000. These are public "
            "simulations under the existing continuation, not counterfactual live-game wins.",
            "",
            "| Profile | Decisions | Different choice | Both batches fully completed | Different choice among both full |",  # noqa: E501
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for arm in PROFILES:
            rows = [r for r in d["sampling_sensitivity"] if r["profile"] == arm]
            full = [r for r in rows if r["both_full"] and r["same_candidates"]]
            lines.append(
                f"| {arm} | {len(rows)} | {sum(r['selection_changed'] for r in rows)} | {len(full)} | "
                f"{sum(r['selection_changed'] for r in full)} |"
            )
        lines += [
            "",
            "### Observed loss resources and entry strength",
            "",
            "| Replay arm | Median entry Joker count | Median dollars left | Score below half target | Score at least 90% of target | Final-blind overrides |",  # noqa: E501
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm in ("baseline", "incumbent"):
            losses = [r for r in d["observations"] if r["arm"] == arm and not r["won"] and r["final"]]
            if losses:
                lines.append(
                    f"| {arm} | {float(np.median([len(r['entry']['jokers']) for r in losses])):.1f} | "
                    f"{float(np.median([r['final']['dollars'] for r in losses])):.1f} | "
                    f"{sum(r['final']['score_fraction'] < 0.5 for r in losses)} | "
                    f"{sum(r['final']['score_fraction'] >= 0.9 for r in losses)} | "
                    f"{sum(r['overrides'] for r in losses)} |"
                )
        losses = [r for r in d["observations"] if r["arm"] == "incumbent" and not r["won"] and r["entry"]]
        bosses = Counter((r["entry"]["blind"] or {}).get("name", "unknown") for r in losses)
        lines += [
            "",
            f"Hybrid replay losses by final blind: {dict(bosses)}. "
            "Entry roster size is only a strength proxy; full Joker properties and hand levels are retained. "
            "Unused resources and score shortfalls identify decisions to investigate, but do not prove "
            "that a different use would have rescued the game.",
            "",
            *[f"- {limit}" for limit in d["limits"]],
            "",
        ]
    if (out / "profile_lock.json").exists():
        lock = read(out / "profile_lock.json")
        lines += [
            "## Timing-only pilot",
            "",
            f"Selected high profile: **{lock['high']}**. Eight common non-evaluation seeds "
            "per profile; pilot wins were sealed until this choice was locked and never selected a profile.",
            "",
            "| Profile | Worlds | Limit seconds | Mean measured game seconds |",
            "| --- | ---: | ---: | ---: |",
        ]
        for a, p in PROFILES.items():
            lines.append(f"| {a} | {p['samples']} | {p['seconds']} | {lock['mean_game_seconds'][a]:.3f} |")
        lines += [
            "",
            f"Remaining hours at selection: {lock['remaining_seconds'] / 3600:.3f}. "
            "Projected development plus worst-case confirmation, including 25% margin and scheduling tails: "
            + "; ".join(f"{a}: {p['wall_with_margin'] / 3600:.3f} hours" for a, p in lock["projections"].items())
            + ".",
            "",
        ]
    for stage, file in (
        ("Development (96 common seeds)", "challenger_lock.json"),
        ("Confirmation (256 untouched paired seeds)", "confirmation_result.json"),
    ):
        if not (out / file).exists():
            continue
        result = read(out / file)
        lines += [
            f"## {stage}",
            "",
            "| Arm | Wins | Rate and Wilson 95% | Overrides | Searches | Completed worlds | Deadline stops | Fallbacks | Decision mean / p95 / max seconds | Worker hours |",  # noqa: E501
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
        for arm, r in result["summaries"].items():
            lines.append(
                f"| {arm} | {r['wins']}/{r['games']} | {r['rate']:.2f}% [{r['interval95'][0]:.2f}, {r['interval95'][1]:.2f}] | "  # noqa: E501
                f"{r['changes']} | {r['searches']} | {r['paired_sample_sets']} | {r['search_stops'].get('deadline', 0)} | "  # noqa: E501
                f"{r['fallbacks']} | {r['mean_decision_seconds']:.3f} / {r['p95_decision_seconds']:.3f} / {r['max_decision_seconds']:.3f} | {r['worker_hours']:.3f} |"  # noqa: E501
            )
        stats = result["paired"] if file == "challenger_lock.json" else {result["challenger"]: result["paired"]}
        lines += [
            "",
            "| Challenger minus contemporaneous incumbent | Gain points | Paired bootstrap 95% | Gained / lost games |",
            "| --- | ---: | --- | ---: |",
        ]
        for arm, r in stats.items():
            lines.append(
                f"| {arm} | {r['points']:+.2f} | [{r['interval95'][0]:+.2f}, {r['interval95'][1]:+.2f}] | {r['gained']} / {r['lost']} |"  # noqa: E501
            )
        lines += [
            "",
            f"Complete sample-count histograms and stopping reasons are published for every arm in [{file}]({file}).",
            "",
        ]
    lines += [
        "## Interpretation and reproducibility",
        "",
        "Advancement requires at least five percentage points on development. Adoption additionally requires "
        "all 256 confirmation pairs, a gain of at least five points and a strictly positive paired 95% lower bound. "
        "Paired bootstrap: 10,000 game resamples, seed 1729. Missing/interrupted games are incomplete, never losses; "
        "seeds and the original deadline are never replaced.",
        "",
        "Frozen source/checkpoint hashes, configuration, exclusion audit, seed reservations and runtime archive "
        "are in this directory. Scientific games retain compact outcomes and compute summaries only. "
        "Diagnostics include serialization overhead in worker time, outside planning limits. "
        "Longer-term strategy and continuation weaknesses are hypotheses; sampling sensitivity is reported separately "
        "from observed resource and scoring deficits. This study tests progress toward 90%; it cannot establish a 90% ceiling.",  # noqa: E501
        "",
    ]
    if verdict["verdict"] == "adopt":
        p = PROFILES[verdict["challenger"]]
        lines += [
            "Recommended command after successful confirmation:",
            "",
            "```sh",
            ".venv/bin/python play.py --checkpoint incumbents/ppo_replica1/checkpoint.pt "
            f"--public-information --hybrid hand --search-samples {p['samples']} --planning-budget {p['seconds']} "
            "--device cpu --games 100 --seed 0",
            "```",
            "",
        ]
    else:
        lines += ["The existing four-world recommended player command remains unchanged.", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(f"Report: {out / 'report.md'}; verdict={verdict['verdict']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "status"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output.resolve()
    if out == HISTORICAL or HISTORICAL in out.parents:
        parser.error("Use a separate study directory")
    if args.command == "status":
        for name in ("clock.json", "progress.json", "profile_lock.json", "verdict.json"):
            if (out / name).exists():
                print(name, encode(read(out / name)))
        return
    if args.command == "report":
        report(out)
        return
    with exclusive(out):
        if args.command == "prepare":
            prepare(out)
        else:
            try:
                run(out)
            except Exception as exc:
                atomic(out / "run_error.json", dict(traceback=traceback.format_exc(), time=time.time()))
                atomic(out / "verdict.json", dict(verdict="incomplete", reason=str(exc)))
                report(out)
                raise


if __name__ == "__main__":
    main()
