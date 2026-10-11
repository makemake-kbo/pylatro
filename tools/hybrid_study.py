"""Fixed public hybrid architecture study. Resume only completed whole games.

Commands: prepare -> pilot -> run -> report. No outcome-selected parameter
sweeps. Errors, missing games and deadline expiries make the fixed panel
incomplete; they are never counted as losses or replaced with new seeds.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "runs/public_hybrid_20261002"
MODELS = ("ppo_replica0", "ppo_replica1", "supervised")
MODES = ("off", "hand", "shop", "both")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def source_hashes():
    paths = [p for p in (ROOT / "src").rglob("*") if p.suffix in {".py", ".json", ".so"}]
    paths += [Path(__file__), ROOT / "play.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def prepare(out):
    if (out / "seeds.json").exists():
        raise RuntimeError("Seed reservation already exists")
    # A fresh nonnumeric namespace is disjoint from the historical decimal
    # seed ranges. Check all source/metadata paths for prior namespace use,
    # without parsing or opening reserved outcome files.
    namespace = "PUBLIC-HYBRID-20261002-V3"
    reservations = []
    for p in (ROOT / "eval").rglob("*"):
        if (
            p.is_file()
            and p.suffix in {".json", ".md"}
            and any(
                word in p.name.lower()
                for word in ("seed", "manifest", "protocol", "config", "generation", "reservation", "exclusion")
            )
        ):
            if namespace.encode() in p.read_bytes():
                raise RuntimeError(f"Namespace already present: {p}")
            reservations.append(dict(path=str(p.relative_to(ROOT)), sha256=digest(p)))
    seeds = {
        stage: [f"{namespace}-{stage}-{i:04d}" for i in range(count)]
        for stage, count in (
            ("pilot", 16),
            ("baseline", 128),
            ("development", 256),
            ("confirmation", 400),
            ("sampled", 128),
        )
    }
    assert len({s for values in seeds.values() for s in values}) == sum(map(len, seeds.values()))
    registry = json.loads((ROOT / "incumbents/registry.json").read_text())
    atomic(out / "seeds.json", seeds)
    atomic(
        out / "reservation.json",
        dict(
            namespace=namespace,
            historical_metadata=reservations,
            original_corpus=json.loads((ROOT / "datasets/registry.json").read_text())["original_5000"]["path"],
            rationale=(
                "Fresh string namespace; historical integer ranges cannot overlap. Reserved outcomes remain unopened."
            ),
            seed_sha256=digest(out / "seeds.json"),
        ),
    )
    atomic(
        out / "config.json",
        dict(
            models=registry,
            cpu_threads=1,
            workers=4,
            precision="fp32",
            deck="b_blue",
            stake=1,
            win_ante=8,
            temperature=1,
            planning_seconds=3,
            samples=4,
            hand_candidates=6,
            shop_candidates=4,
            hand_steps=60,
            shop_steps=100,
            greedy_primary=True,
            maximum_wall_seconds=48 * 3600,
            bootstrap_seed=1729,
            bootstrap_draws=10000,
            advancement_points=5,
            baseline_tie_order=list(MODELS),
            assisted_tie="fewer phases, then latency",
        ),
    )
    print("Reserved all five disjoint seed panels", flush=True)


def worker(job):
    from pylatro_agent.hybrid_game import play_game

    started = time.time()
    path = Path(job["path"])
    try:
        result = play_game(
            job["checkpoint"],
            job["seed"],
            mode=job["mode"],
            sample=job["sample"],
            deadline=job["deadline"],
            boundary_audit=job["stage"] == "pilot",
        )
        result.update(model=job["model"], stage=job["stage"], source_id=job["source_id"], started_unix=started)
        atomic(path, result)
        return {"complete": True, "path": str(path)}
    except Exception:
        atomic(path.with_suffix(".error.json"), dict(job=job, traceback=traceback.format_exc(), started_unix=started))
        return {"complete": False, "path": str(path)}


def read_panel(out, stage, model, mode, seeds):
    rows = []
    for seed in seeds:
        path = out / stage / model / mode / f"{seed}.json"
        if not path.exists():
            raise ValueError(f"Missing completed game: {path}")
        row = json.loads(path.read_text())
        if not row.get("complete") or row["seed"] != seed or row["mode"] != mode or row["model"] != model:
            raise ValueError(f"Invalid completed game: {path}")
        rows.append(row)
    if len({row["seed"] for row in rows}) != len(seeds):
        raise ValueError("Duplicate game seeds")
    return rows


def panel(out, stage, arms, seeds, deadline, source_id):
    config = json.loads((out / "config.json").read_text())
    jobs = []
    for model, mode in arms:
        for seed in seeds:
            path = out / stage / model / mode / f"{seed}.json"
            if path.exists():
                row = read_panel(out, stage, model, mode, [seed])[0]
                if row["source_id"] != source_id:
                    raise RuntimeError("Cannot mix source revisions within a panel")
                continue
            if path.with_suffix(".error.json").exists():
                raise RuntimeError(f"Recorded game error requires investigation: {path}")
            jobs.append(
                dict(
                    stage=stage,
                    model=model,
                    mode=mode,
                    seed=seed,
                    checkpoint=str(ROOT / config["models"][model]["path"]),
                    sample=stage == "sampled",
                    path=str(path),
                    deadline=deadline,
                    source_id=source_id,
                )
            )
    # Interleave arms and seeds so resumed partial panels are easy to audit.
    jobs.sort(key=lambda j: (j["seed"], j["model"], j["mode"]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        active = {}
        queue = iter(jobs)
        failed = False

        def submit():
            if failed or time.time() >= deadline:
                return
            job = next(queue, None)
            if job is not None:
                active[pool.submit(worker, job)] = job

        for _ in range(4):
            submit()
        finished = 0
        while active:
            done, _ = concurrent.futures.wait(active, timeout=30, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                result = future.result()
                del active[future]
                finished += 1
                failed |= not result["complete"]
                print(f"{stage}: {finished}/{len(jobs)} pending games completed; errors={failed}", flush=True)
                submit()
        if failed:
            raise RuntimeError(f"{stage} contains an execution error; no result may advance")
    return {f"{model}/{mode}": read_panel(out, stage, model, mode, seeds) for model, mode in arms}


def paired(base, candidate):
    if len(base) != len(candidate) or [r["seed"] for r in base] != [r["seed"] for r in candidate]:
        raise ValueError("Paired statistics require identical complete seed lists")
    difference = np.array([int(c["won"]) - int(b["won"]) for b, c in zip(base, candidate, strict=True)])
    rng = np.random.default_rng(1729)
    boot = difference[rng.integers(0, len(difference), (10000, len(difference)))].mean(axis=1) * 100
    return dict(
        points=float(difference.mean() * 100),
        interval95=list(np.quantile(boot, [0.025, 0.975])),
        gained=int((difference == 1).sum()),
        lost=int((difference == -1).sum()),
        games=len(base),
    )


def summary(rows):
    timings = [v for r in rows for v in r["decision_seconds"]]
    return dict(
        games=len(rows),
        wins=sum(r["won"] for r in rows),
        decisions=sum(r["steps"] for r in rows),
        changes=sum(r["changes"] for r in rows),
        searches=sum(r["searches"] for r in rows),
        fallbacks=sum(r["fallbacks"] for r in rows),
        completed_sample_sets=sum(r["paired_sample_sets"] for r in rows),
        worker_seconds=sum(r["seconds"] for r in rows),
        mean_decision_seconds=float(np.mean(timings)),
        p95_decision_seconds=float(np.quantile(timings, 0.95)),
        max_decision_seconds=max(timings),
        illegal_actions=sum(r["illegal_actions"] for r in rows),
        boundary_audits=sum(r.get("boundary_audits", 0) for r in rows),
    )


def pilot(out):
    tests = [
        "tests/test_public_hybrid.py",
        "tests/test_hybrid_study.py",
        "tests/test_action_masking.py",
        "tests/test_joker_layout.py",
        "tests/test_deck_exhaustion.py",
    ]
    with (out / "correctness.log").open("w") as log:
        check = subprocess.run(
            [sys.executable, "-m", "pytest", *tests, "-q"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
        )
    atomic(
        out / "correctness.json",
        dict(command=tests, exit_code=check.returncode, log_sha256=digest(out / "correctness.log")),
    )
    if check.returncode:
        raise RuntimeError("Correctness checks failed before pilot")
    seeds = json.loads((out / "seeds.json").read_text())
    pins = source_hashes()
    source_id = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    atomic(out / "pilot_sources.json", pins)
    clock = out / "clock.json"
    if not clock.exists():
        atomic(clock, dict(started_unix=time.time(), deadline_unix=time.time() + 48 * 3600))
    times = json.loads(clock.read_text())
    panels = panel(
        out, "pilot", [("ppo_replica0", "both")], seeds["pilot"], technical_deadline(times), source_id
    )
    technical = summary(panels["ppo_replica0/both"])
    technical.pop("wins")  # Pilot outcomes do not select search settings.
    # Conservative projection uses the slow combined arm for every assisted
    # game and confirmation, including the full optional sampled panel.
    per_game = technical["worker_seconds"] / 16
    projection = per_game * (128 * 3 + 256 * 4 + 400 * 2 + 128 * 2) / 4
    technical.update(
        projected_study_seconds=projection,
        complete=True,
        qualifies=projection < 44 * 3600 and technical["illegal_actions"] == 0,
        source_id=source_id,
    )
    atomic(out / "pilot_technical.json", technical)
    print(json.dumps(technical, indent=2), flush=True)


def select_assisted(panels, model):
    base = panels[f"{model}/off"]
    candidates = []
    for mode in MODES[1:]:
        rows = panels[f"{model}/{mode}"]
        if paired(base, rows)["points"] >= 5:
            candidates.append((mode, rows))
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            -sum(r["won"] for r in item[1]),
            2 if item[0] == "both" else 1,
            summary(item[1])["mean_decision_seconds"],
        ),
    )[0]


def technical_deadline(clock):
    # Correctness restarts may reserve a new technical phase explicitly, but
    # cannot extend the original overall clock or the architecture deadline.
    return min(clock["deadline_unix"], clock.get("technical_started_unix", clock["started_unix"]) + 4 * 3600)


def run(out):
    atomic(out / "owner.json", dict(pid=os.getpid(), started_unix=time.time(), command="run"))
    technical = json.loads((out / "pilot_technical.json").read_text())
    if not technical["qualifies"]:
        raise RuntimeError("Technical pilot does not qualify")
    pins = source_hashes()
    if pins != json.loads((out / "pilot_sources.json").read_text()):
        raise RuntimeError("Source changed after pilot; rerun correctness and technical pilot before science")
    frozen = out / "frozen.json"
    if not frozen.exists():
        atomic(
            frozen,
            dict(
                source=pins,
                source_id=technical["source_id"],
                seeds_sha256=digest(out / "seeds.json"),
                config_sha256=digest(out / "config.json"),
                frozen_unix=time.time(),
            ),
        )
    manifest = json.loads(frozen.read_text())
    if (
        manifest["source"] != pins
        or digest(out / "seeds.json") != manifest["seeds_sha256"]
        or digest(out / "config.json") != manifest["config_sha256"]
    ):
        raise RuntimeError("Frozen study inputs changed")
    config = json.loads((out / "config.json").read_text())
    for model in config["models"].values():
        assert digest(ROOT / model["path"]) == model["sha256"]
    seeds = json.loads((out / "seeds.json").read_text())
    clock = json.loads((out / "clock.json").read_text())
    source_id = manifest["source_id"]
    baseline = panel(
        out,
        "baseline",
        [(model, "off") for model in MODELS],
        seeds["baseline"],
        technical_deadline(clock),
        source_id,
    )
    selected = max(MODELS, key=lambda model: sum(row["won"] for row in baseline[f"{model}/off"]))
    atomic(
        out / "baseline_selection.json",
        dict(selected=selected, arms={key: summary(rows) for key, rows in baseline.items()}),
    )
    development = panel(
        out,
        "development",
        [(selected, mode) for mode in MODES],
        seeds["development"],
        min(clock["deadline_unix"], clock["started_unix"] + 24 * 3600),
        source_id,
    )
    chosen = select_assisted(development, selected)
    atomic(
        out / "architecture_selection.json",
        dict(
            selected=chosen,
            baseline=selected,
            arms={key: summary(rows) for key, rows in development.items()},
            paired={
                mode: paired(development[f"{selected}/off"], development[f"{selected}/{mode}"]) for mode in MODES[1:]
            },
        ),
    )
    if chosen is None:
        atomic(
            out / "verdict.json",
            dict(verdict="STOP", reason="No assisted arm gained at least five greedy percentage points", complete=True),
        )
        return
    confirmation = panel(
        out,
        "confirmation",
        [(selected, "off"), (selected, chosen)],
        seeds["confirmation"],
        clock["deadline_unix"],
        source_id,
    )
    effect = paired(confirmation[f"{selected}/off"], confirmation[f"{selected}/{chosen}"])
    success = effect["points"] >= 5 and effect["interval95"][0] > 0
    atomic(
        out / "confirmation_result.json",
        dict(paired=effect, success=success, arms={key: summary(rows) for key, rows in confirmation.items()}),
    )
    sampled = panel(
        out, "sampled", [(selected, "off"), (selected, chosen)], seeds["sampled"], clock["deadline_unix"], source_id
    )
    atomic(
        out / "sampled_result.json",
        dict(
            paired=paired(sampled[f"{selected}/off"], sampled[f"{selected}/{chosen}"]),
            arms={key: summary(rows) for key, rows in sampled.items()},
            secondary=True,
        ),
    )
    atomic(
        out / "verdict.json",
        dict(
            verdict="KEEP" if success else "STOP",
            complete=True,
            reason="Fixed paired greedy confirmation criteria",
            selected=chosen,
            baseline=selected,
            paired=effect,
        ),
    )


def report(out):
    verdict = (
        json.loads((out / "verdict.json").read_text())
        if (out / "verdict.json").exists()
        else dict(verdict="INCOMPLETE")
    )
    lines = [
        "# Public-information hybrid gameplay study",
        "",
        f"**Verdict: {verdict['verdict']}.**",
        "",
        "Blue Deck, White Stake, Ante 8; CPU FP32; temperature 1; frozen neural weights. "
        "Public-information-v2 changes observation interpretation, so historical win rates are reference results.",
        "",
        "The fixed study uses 16 technical pilot games, 128 baseline seeds, 256 architecture seeds, "
        "400 untouched confirmation seeds if qualified, and 128 secondary sampled seeds. "
        "The 48-hour wall cap includes interruptions. Missing or failed games make a panel incomplete.",
        "",
    ]
    for name in (
        "pilot_technical",
        "baseline_selection",
        "architecture_selection",
        "confirmation_result",
        "sampled_result",
    ):
        p = out / f"{name}.json"
        if p.exists():
            result = json.loads(p.read_text())
            lines.extend(
                [name.replace("_", " ").capitalize() + ":", "", "```json", json.dumps(result, indent=2), "```", ""]
            )
    counts = {}
    for stage in ("pilot", "baseline", "development", "confirmation", "sampled"):
        counts[stage] = dict(
            completed=len([p for p in (out / stage).rglob("*.json") if not p.name.endswith(".error.json")]),
            errors=len(list((out / stage).rglob("*.error.json"))),
        )
    lines.extend(
        [
            "Accounting: " + json.dumps(counts),
            "",
            "Search diagnostics report decision changes, equally completed sample sets, "
            "deadline fallbacks, and latency. "
            "No teacher-agreement or forecast-error gate selects an arm. "
            "A keep verdict demonstrates stronger hybrid play; it does not mean unchanged weights learned a strategy.",
            "",
        ]
    )
    target = ROOT / "docs/transfer/public_hybrid_20261002.md"
    target.write_text("\n".join(lines))
    print(target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "pilot", "run", "report"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    try:
        globals()[args.command](args.output)
    except Exception:
        if args.command in {"pilot", "run"}:
            atomic(
                args.output / "incomplete.json",
                dict(command=args.command, traceback=traceback.format_exc(), time=time.time()),
            )
        raise
    finally:
        if args.command in {"pilot", "run", "report"}:
            report(args.output)
