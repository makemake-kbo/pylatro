#!/usr/bin/env python3
"""Verify and report completed second-study artifacts without inventing missing results."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

from pylatro_agent.learning_bottleneck import (
    ARMS,
    CHECKPOINT,
    CONFIRM_SEEDS,
    DEV_SEEDS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    memorization_name,
    read_json,
    stage_limits,
    summarize_rows,
)
from pylatro_agent.transfer import check_seed_separation, paired_statistics, sha256, summarize, write_json


def percentage(value):
    return f"{100 * value:.2f}%"


def delta(result):
    low, high = result["paired_ci95"]
    return f"{100 * result['sampled_or_greedy_delta']:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}]"


def phase_oracle_results(root, *, verify_traces=False, expected_seeds=None, write_analysis=True):
    from pylatro_agent.constants import SubPhase
    from pylatro_agent.phase_oracle import ORACLE_ARMS, ORACLE_SEEDS, teacher_controls_phase

    folder = root / "phase_oracle_cpu"
    seeds = list(ORACLE_SEEDS if expected_seeds is None else expected_seeds)
    if not (folder / "manifest.json").exists():
        return None
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "complete":
        return None
    if manifest["device"] != "cpu" or manifest["accelerator_seconds"] != 0:
        raise ValueError("Phase replacement was not recorded as CPU-only")
    if manifest["seeds"] != seeds or not manifest["never_train_on_seeds"]:
        raise ValueError("Phase-oracle seed reservation changed")
    check_seed_separation(seeds, [entry["path"] for entry in manifest["seed_separation"]])
    if sha256(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
        raise ValueError("Phase-oracle source checkpoint changed")
    rows = [read_json(path) for path in sorted((folder / "games").glob("*.json"))]
    expected = {
        (arm, seed, sampling)
        for seed in seeds
        for arm in ORACLE_ARMS
        for sampling in ((None,) if arm == "teacher_all" else (0, 1, 2))
    }
    actual = [(row["arm"], row["seed"], row["sampling_seed"]) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate phase-oracle comparison matrix")
    if any(row["stalled"] or row["censored"] for row in rows):
        raise ValueError("Phase-oracle panel contains incomplete games")
    if verify_traces:
        for row in rows:
            trace_path = folder / "games" / f"{row['arm']}_{row['seed']}_sample{row['sampling_seed']}.jsonl.gz"
            if sha256(trace_path) != row["trace_sha256"]:
                raise ValueError("Phase-oracle trace changed")
            teacher_steps = replacements = count = 0
            with gzip.open(trace_path, "rt") as stream:
                for count, line in enumerate(stream, start=1):
                    step = json.loads(line)
                    if step["seed"] != row["seed"] or step["step"] != count - 1:
                        raise ValueError("Missing phase-oracle decision identifier")
                    controlled = teacher_controls_phase(row["arm"], SubPhase(step["phase"]))
                    if (step["source"] == "teacher") != controlled:
                        raise ValueError("Phase-oracle intervention crossed its declared phase boundary")
                    if not controlled and step["action"] != step["student_proposal"]:
                        raise ValueError("Uncontrolled phase did not execute the student action")
                    teacher_steps += int(controlled)
                    replacements += int(
                        controlled
                        and step["student_proposal"] is not None
                        and step["action"] != step["student_proposal"]
                    )
            if (count, teacher_steps, replacements) != (
                row["steps"],
                row["teacher_steps"],
                row["teacher_replacements"],
            ):
                raise ValueError("Phase-oracle decision denominator mismatch")
    # Canonical seed order makes absolute bootstrap intervals reproducible from the per-game files.
    groups = {
        arm: sorted((row for row in rows if row["arm"] == arm), key=lambda row: (row["seed"], row["sampling_seed"]))
        for arm in ORACLE_ARMS
    }
    summaries = {arm: summarize(values) for arm, values in groups.items()}
    comparisons = {arm: paired_statistics(groups[arm], groups["student"]) for arm in ORACLE_ARMS[1:-1]}
    teachers = [dict(row, sampling_seed=sampling) for row in groups["teacher_all"] for sampling in (0, 1, 2)]
    comparisons["teacher_all"] = paired_statistics(teachers, groups["student"])
    if comparisons != read_json(folder / "results.json")["comparisons"]:
        raise ValueError("Phase-oracle paired statistics differ from raw-game recomputation")
    conditional = {
        "teacher_shop_given_teacher_hands": paired_statistics(groups["hand_shop"], groups["hand"]),
        "teacher_hands_given_teacher_shop": paired_statistics(groups["hand_shop"], groups["shop"]),
        "hand_shop_minus_full_teacher": paired_statistics(groups["hand_shop"], teachers),
    }
    result = dict(
        manifest=manifest,
        summaries=summaries,
        comparisons=comparisons,
        secondary_conditional_comparisons=conditional,
        ordering="Canonical game-seed order for absolute bootstrap intervals; original summary retained",
    )
    if write_analysis:
        write_json(folder / "analysis_canonical.json", result)
    return result


def sale_oracle_results(root, *, verify_traces=False):
    from pylatro_agent.constants import SubPhase
    from pylatro_agent.phase_oracle import ORACLE_SEEDS, policy_seed
    from pylatro_agent.sale_oracle import SALE_ARMS, replace_shop_action

    folder = root / "sale_oracle_cpu"
    if not (folder / "results.json").exists():
        return None
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "complete":
        return None
    baseline = root / "phase_oracle_cpu"
    for name in ("manifest", "results"):
        if sha256(baseline / f"{name}.json") != manifest[f"baseline_{name}_sha256"]:
            raise ValueError("Sale-oracle matched baseline changed")
    if manifest["device"] != "cpu" or manifest["accelerator_seconds"] != 0 or not manifest["post_hoc"]:
        raise ValueError("Sale-oracle exploratory CPU provenance changed")
    if manifest["seeds"] != list(ORACLE_SEEDS) or not manifest["never_train_on_seeds"]:
        raise ValueError("Sale-oracle seed reservation changed")
    check_seed_separation(ORACLE_SEEDS, [entry["path"] for entry in manifest["seed_separation"]])
    if sha256(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
        raise ValueError("Sale-oracle checkpoint changed")
    rows = [read_json(path) for path in sorted((folder / "games").glob("*.json"))]
    expected = {(arm, seed, sample) for arm in SALE_ARMS for seed in ORACLE_SEEDS for sample in (0, 1, 2)}
    actual = [(r["arm"], r["seed"], r["sampling_seed"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate sale-oracle comparison matrix")
    for row in rows:
        wrong_seed = row["policy_rng_seed"] != policy_seed(row["seed"], row["sampling_seed"])
        if row["stalled"] or row["censored"] or wrong_seed:
            raise ValueError("Sale-oracle outcome or sampling provenance mismatch")
        if not verify_traces:
            continue
        path = folder / "games" / f"{row['arm']}_{row['seed']}_sample{row['sampling_seed']}.jsonl.gz"
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("Sale-oracle trace changed")
        base_name = f"student_{row['seed']}_sample{row['sampling_seed']}.jsonl.gz"
        with gzip.open(baseline / "games" / base_name, "rt") as stream:
            base_steps = [json.loads(line) for line in stream]
        diverged = False
        count = teacher_steps = replacements = 0
        with gzip.open(path, "rt") as stream:
            for count, line in enumerate(stream, start=1):
                step = json.loads(line)
                if step["seed"] != row["seed"] or step["step"] != count - 1:
                    raise ValueError("Sale-oracle decision identifier mismatch")
                controlled = replace_shop_action(row["arm"], SubPhase(step["phase"]), step["teacher_proposal"])
                expected_action = step["teacher_proposal"] if controlled else step["student_proposal"]
                if (step["source"] == "teacher") != controlled or step["action"] != expected_action:
                    raise ValueError("Sale-oracle intervention crossed its declared boundary")
                if not diverged:
                    if count > len(base_steps):
                        raise ValueError("Sale-oracle prefix outlived its identical student trajectory")
                    base_step = base_steps[count - 1]
                    if any(step[key] != base_step[key] for key in ("phase", "ante", "student_proposal")):
                        raise ValueError("Teacher queries changed the common-prefix student sampling")
                    diverged = step["action"] != base_step["action"]
                teacher_steps += int(controlled)
                replacements += int(controlled and step["action"] != step["student_proposal"])
        if (count, teacher_steps, replacements) != (row["steps"], row["teacher_steps"], row["teacher_replacements"]):
            raise ValueError("Sale-oracle decision denominator mismatch")
        if not diverged and count != len(base_steps):
            raise ValueError("Sale-oracle identical trajectory ended at a different step")
    groups = {
        arm: sorted([r for r in rows if r["arm"] == arm], key=lambda r: (r["seed"], r["sampling_seed"]))
        for arm in SALE_ARMS
    }
    references = {
        arm: [read_json(path) for path in sorted((baseline / "games").glob(f"{arm}_*.json"))]
        for arm in ("student", "shop")
    }
    comparisons = {
        arm: {reference: paired_statistics(values, base) for reference, base in references.items()}
        for arm, values in groups.items()
    }
    summaries = {arm: summarize(values) for arm, values in groups.items()}
    saved = read_json(folder / "results.json")
    if comparisons != saved["comparisons"] or summaries != saved["summaries"]:
        raise ValueError("Sale-oracle statistics differ from raw-game recomputation")
    return dict(manifest=manifest, summaries=summaries, comparisons=comparisons)


def sale_head_results(root):
    folder = root / "sale_head_cpu"
    if not (folder / "manifest.json").exists():
        return None
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "complete" or manifest["device"] != "cpu" or manifest["accelerator_seconds"] != 0:
        raise ValueError("Sale-head CPU diagnostic did not complete")
    for path, digest in (
        (manifest["checkpoint"], manifest["checkpoint_sha256"]),
        (manifest["audit"], manifest["audit_sha256"]),
        (folder / "predictions.jsonl", manifest["predictions_sha256"]),
    ):
        if sha256(path) != digest:
            raise ValueError("Sale-head diagnostic source or predictions changed")
    rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
    audit = [json.loads(line) for line in Path(manifest["audit"]).read_text().splitlines()]
    expected = {(r["seed"], r["step"]) for r in audit if r["family"] == "shop_sell_joker"}
    if len(rows) != len(expected) or {(r["seed"], r["step"]) for r in rows} != expected:
        raise ValueError("Sale-head diagnostic omitted or duplicated labels")
    for row in rows:
        if abs(row["probability"] - row["macro_probability"] * row["conditional_probability"]) > 1e-6:
            raise ValueError("Sale-head probability decomposition mismatch")
        if abs(row["nll"] - row["mps_nll"]) >= 5e-4:
            raise ValueError("Sale-head CPU/MPS probability mismatch")
    summary = read_json(folder / "summary.json")
    for label, values in (("overall", rows), ("SHOP", [r for r in rows if r["phase"] == 2])):
        if summary[label]["decisions"] != len(values):
            raise ValueError("Sale-head metric denominator mismatch")
        for key, value in summary[label]["metrics"].items():
            if not math.isclose(sum(row[key] for row in values) / len(values), value, abs_tol=1e-12):
                raise ValueError("Sale-head summary differs from raw predictions")
    return dict(manifest=manifest, summary=summary)


def sale_choice_results(root, *, verify_traces=False):
    import numpy as np

    from pylatro_agent.action import ActionType, decode_action
    from pylatro_agent.constants import SubPhase
    from pylatro_agent.phase_oracle import ORACLE_SEEDS, policy_seed
    from pylatro_agent.sale_choice_oracle import replace_shop_action

    folder = root / "sale_choice_oracle_cpu"
    if not (folder / "results.json").exists():
        return None
    manifest = read_json(folder / "manifest.json")
    if manifest["status"] != "complete":
        return None
    for directory, prefix in (("phase_oracle_cpu", "baseline"), ("sale_oracle_cpu", "teacher_sale")):
        for name in ("manifest", "results"):
            if sha256(root / directory / f"{name}.json") != manifest[f"{prefix}_{name}_sha256"]:
                raise ValueError("Sale-choice baseline changed")
    if manifest["device"] != "cpu" or manifest["accelerator_seconds"] != 0 or not manifest["post_hoc"]:
        raise ValueError("Sale-choice exploratory CPU provenance mismatch")
    check_seed_separation(ORACLE_SEEDS, [entry["path"] for entry in manifest["seed_separation"]])
    if sha256(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
        raise ValueError("Sale-choice checkpoint changed")
    arm = "teacher_sale_student_choice"
    rows = [read_json(path) for path in sorted((folder / "games").glob("*.json"))]
    expected = {(seed, sample) for seed in ORACLE_SEEDS for sample in (0, 1, 2)}
    if len(rows) != len(expected) or {(r["seed"], r["sampling_seed"]) for r in rows} != expected:
        raise ValueError("Incomplete or duplicate sale-choice comparison matrix")
    for row in rows:
        choice_seed = int(np.random.SeedSequence([row["seed"], row["sampling_seed"], 1729, 731]).generate_state(1)[0])
        if (
            row["arm"] != arm
            or row["stalled"]
            or row["censored"]
            or row["policy_rng_seed"] != policy_seed(row["seed"], row["sampling_seed"])
            or row["choice_rng_seed"] != choice_seed
        ):
            raise ValueError("Sale-choice outcome or RNG provenance mismatch")
        if not verify_traces:
            continue
        suffix = f"{row['seed']}_sample{row['sampling_seed']}.jsonl.gz"
        path = folder / "games" / f"{arm}_{suffix}"
        if sha256(path) != row["trace_sha256"]:
            raise ValueError("Sale-choice trace changed")
        with gzip.open(root / "sale_oracle_cpu/games" / f"teacher_sale_{suffix}", "rt") as stream:
            base_steps = [json.loads(line) for line in stream]
        diverged = False
        count = teacher_steps = replacements = 0
        with gzip.open(path, "rt") as stream:
            for count, line in enumerate(stream, start=1):
                step = json.loads(line)
                if step["seed"] != row["seed"] or step["step"] != count - 1:
                    raise ValueError("Sale-choice decision identifier mismatch")
                controlled = replace_shop_action(arm, SubPhase(step["phase"]), step["teacher_proposal"])
                if (step["source"] == "teacher_timing_student_choice") != controlled:
                    raise ValueError("Sale-choice intervention crossed its declared boundary")
                if controlled and decode_action(step["action"]).action_type != ActionType.SHOP_SELL_JOKER:
                    raise ValueError("Sale-choice intervention did not execute a Joker sale")
                if not controlled and step["action"] != step["student_proposal"]:
                    raise ValueError("Uncontrolled sale-choice action changed")
                if not diverged:
                    if count > len(base_steps):
                        raise ValueError("Sale-choice common prefix exceeded its baseline")
                    base_step = base_steps[count - 1]
                    if any(
                        step[key] != base_step[key] for key in ("phase", "ante", "student_proposal", "teacher_proposal")
                    ):
                        raise ValueError("Sale-choice RNG stream changed the common-prefix policy or teacher")
                    diverged = step["action"] != base_step["action"]
                teacher_steps += int(controlled)
                replacements += int(controlled and step["action"] != step["student_proposal"])
        if (count, teacher_steps, replacements) != (row["steps"], row["teacher_steps"], row["teacher_replacements"]):
            raise ValueError("Sale-choice decision denominator mismatch")
        if not diverged and count != len(base_steps):
            raise ValueError("Sale-choice identical trajectory ended at a different step")
    references = {
        label: [read_json(p) for p in (root / directory / "games").glob(f"{label}_*.json")]
        for label, directory in (
            ("student", "phase_oracle_cpu"),
            ("shop", "phase_oracle_cpu"),
            ("teacher_sale", "sale_oracle_cpu"),
        )
    }
    comparison = {label: paired_statistics(rows, values) for label, values in references.items()}
    rows.sort(key=lambda r: (r["seed"], r["sampling_seed"]))
    summary = summarize(rows)
    saved = read_json(folder / "results.json")
    if saved["comparisons"][arm] != comparison or saved["summaries"][arm] != summary:
        raise ValueError("Sale-choice statistics differ from raw-game recomputation")
    return dict(manifest=manifest, summary=summary, comparisons=comparison)


def verification(root, *, require_complete):
    errors = []
    offline_validations = []
    originals = read_json(root / "original_hashes.json")
    for path, digest in originals.items():
        if sha256(path) != digest:
            errors.append(f"Original artifact changed: {path}")
    manifest = read_json(root / "manifest.json")
    paths = [entry["path"] for entry in manifest["seed_separation"]]
    check_seed_separation(TRAIN_SEEDS + VALIDATION_SEEDS + DEV_SEEDS + CONFIRM_SEEDS, paths)
    datasets = {}
    for name, seeds in (("training", TRAIN_SEEDS), ("validation", VALIDATION_SEEDS)):
        summary_path = root / "data" / name / "summary.json"
        if not summary_path.exists():
            if require_complete:
                errors.append(f"Missing {name} dataset")
            continue
        summary = read_json(summary_path)
        for file, digest in summary["file_hashes"].items():
            if sha256(summary_path.parent / file) != digest:
                errors.append(f"Generated data changed: {file}")
        games = [json.loads(line) for line in (summary_path.parent / "games.jsonl").read_text().splitlines()]
        if [row["seed"] for row in games] != list(seeds):
            errors.append(f"Missing/duplicate {name} seeds")
        datasets[name] = dict(games=len(games), decisions=sum(row["steps"] for row in games))
    trials = []
    training_sources = None
    for seed in (0, 1):
        ordering = set()
        for arm, (mixture, weight) in ARMS.items():
            path = root / "training" / f"{arm}_seed{seed}"
            if not (path / "manifest.json").exists():
                if require_complete:
                    errors.append(f"Missing trial {arm}/{seed}")
                continue
            trial = read_json(path / "manifest.json")
            if trial["status"] == "running" and not require_complete:
                continue
            if trial["source_checkpoint_sha256"] != manifest["checkpoint_sha256"]:
                errors.append(f"Trial {arm}/{seed}: different initial checkpoint")
            expected = dict(
                status="complete",
                updates_completed=750,
                examples_processed=240000,
                physical_batch=320,
                precision="fp32",
                training_seed=seed,
                hand_weight=weight,
            )
            for key, value in expected.items():
                if trial[key] != value:
                    errors.append(f"Trial {arm}/{seed}: {key} mismatch")
            config = trial["configuration"]
            expected_config = dict(
                lr=3e-5,
                weight_decay=0.01,
                warmup_steps=100,
                hand_ar_mixture_eps=mixture,
                deck_key="b_blue",
                stake=1,
                action_entropy_coeff=0.001,
                value_loss_coeff=0.5,
                outcome_loss_coeff=0.1,
                outcome_weight_beta=1.5,
            )
            for key, value in expected_config.items():
                if config[key] != value:
                    errors.append(f"Trial {arm}/{seed}: config {key} mismatch")
            if config["num_games"] != len(TRAIN_SEEDS) or Path(config["data_path"]) != root / "data/training":
                errors.append(f"Trial {arm}/{seed}: incorrect effective data source")
            if not set(VALIDATION_SEEDS + DEV_SEEDS + CONFIRM_SEEDS) <= set(config["excluded_seeds"]):
                errors.append(f"Trial {arm}/{seed}: evaluation exclusions missing from configuration")
            if not (path / "provenance.json").exists():
                errors.append(f"Missing trial provenance {arm}/{seed}")
            else:
                provenance = read_json(path / "provenance.json")
                if sha256(path / "checkpoint.pt") != provenance["checkpoint_sha256"]:
                    errors.append(f"Trial checkpoint changed: {arm}/{seed}")
                sources = {key: value for key, value in provenance["source_hashes"].items() if key.startswith("src/")}
                if training_sources is None:
                    training_sources = sources
                elif any(sources.get(key) != value for key, value in training_sources.items()):
                    errors.append(f"Core source changed across matched trials: {arm}/{seed}")
            ordering.add(trial["hf_order_sha256"])
            trials.append(dict(arm=arm, seed=seed, examples=trial["examples_processed"]))
        if len(ordering) > 1:
            errors.append(f"Unmatched training ordering for seed {seed}")
    evaluations = []
    for path in sorted((root / "evaluations").glob("*/*/manifest.json")):
        run = read_json(path)
        if run["status"] == "running" and not require_complete:
            continue
        if run["status"] != "complete":
            errors.append(f"Incomplete evaluation {path}")
            continue
        if sha256(run["checkpoint"]) != run["checkpoint_sha256"]:
            errors.append(f"Evaluation checkpoint changed {path}")
        if run["deck"] != "b_blue" or run["stake"] != 1 or run["win_ante"] != 8:
            errors.append(f"Reference deck/stake/ante mismatch {path}")
        if run["batch_size"] != 32 or run["workers"] != 4:
            errors.append(f"Evaluation batching/worker mismatch {path}")
        rows = [json.loads(line) for line in (path.parent / "outcomes.jsonl").read_text().splitlines()]
        if sorted(row["seed"] for row in rows) != sorted(run["seeds"]):
            errors.append(f"Incorrect evaluation seed denominator {path}")
        if any(row.get("stalled", False) or row.get("censored", False) for row in rows):
            errors.append(f"Stalled/censored evaluation {path}")
        actual = summarize(rows)
        stored = read_json(path.parent / "summary.json")
        if actual != stored:
            errors.append(f"Evaluation statistics changed {path}")
        evaluations.append(dict(path=str(path), games=len(rows), wins=actual["wins"]))
    ledger = read_json(root / "compute.json")
    elapsed = sum(row["elapsed_seconds"] for row in ledger["jobs"])
    limits = stage_limits(root)
    stage_seconds = {
        stage: sum(row["elapsed_seconds"] for row in ledger["jobs"] if row["stage"] == stage) for stage in limits
    }
    for stage, seconds in stage_seconds.items():
        if seconds > limits[stage]:
            errors.append(f"Accelerator allocation exceeded for {stage}")
    if elapsed > 32400:
        errors.append("Nine-hour accelerator budget exceeded")
    if require_complete and any(row["status"] == "running" for row in ledger["jobs"]):
        errors.append("An accelerator job is still running")
    regression_path = root / "regression_verification.json"
    if regression_path.exists():
        regression = read_json(regression_path)
        for suite in [regression, *regression.get("additional_suites", [])]:
            if suite["status"] != "passed" or sha256(suite["log"]) != suite["log_sha256"]:
                errors.append("Regression test evidence is missing or changed")
    if require_complete:
        reference_plan = root / "incumbent_reference_plan.json"
        if reference_plan.exists():
            reference_result = root / "incumbent_reference_comparison.json"
            if not reference_result.exists():
                errors.append("Missing supplementary original-reference budget decision")
            else:
                reference = read_json(reference_result)
                plan = read_json(reference_plan)
                if reference["plan_sha256"] != sha256(reference_plan):
                    errors.append("Supplementary original-reference protocol changed")
                if reference["status"] == "skipped_budget":
                    if reference["remaining_seconds"] >= plan["budget_gate_seconds"]:
                        errors.append("Supplementary reference was skipped despite its admission budget")
                elif reference["status"] == "complete":
                    admitted = reference["remaining_seconds_at_admission"]
                    chosen = plan
                    if plan.get("larger_panel") and admitted >= plan["larger_panel"]["budget_gate_seconds"]:
                        chosen = plan["larger_panel"]
                    if reference["seeds"] != chosen["seeds"] or admitted < chosen["budget_gate_seconds"]:
                        errors.append("Supplementary original-reference panel violates its time-only gate")
                    expected_seeds = set(reference["seeds"])
                    for mode, files in (("sampled", ("sample0", "sample1", "sample2")), ("greedy", ("greedy",))):
                        original = [
                            json.loads(line)
                            for name in files
                            for line in (root / "evaluations" / reference["evaluation_name"] / name / "outcomes.jsonl")
                            .read_text()
                            .splitlines()
                        ]
                        for arm in ("control", reference["selected"]):
                            candidate = [
                                dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                                for seed in (0, 1)
                                for name in files
                                for row in map(
                                    json.loads,
                                    (root / "evaluations" / f"confirm_{arm}_seed{seed}" / name / "outcomes.jsonl")
                                    .read_text()
                                    .splitlines(),
                                )
                                if row["seed"] in expected_seeds
                            ]
                            baseline = [
                                dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                                for seed in (0, 1)
                                for row in original
                            ]
                            if paired_statistics(candidate, baseline) != reference["comparisons"][arm][mode]:
                                errors.append(f"Supplementary original-reference statistics changed: {arm}/{mode}")
                else:
                    errors.append("Unknown supplementary original-reference status")
        if (root / "sale_choice_oracle_cpu").exists():
            try:
                if sale_choice_results(root, verify_traces=True) is None:
                    errors.append("Sale-choice CPU intervention is incomplete")
            except (ValueError, KeyError) as exc:
                errors.append(f"Sale-choice verification: {exc}")
        if (root / "sale_head_cpu").exists():
            try:
                if sale_head_results(root) is None:
                    errors.append("Sale-head CPU diagnostic is incomplete")
            except (ValueError, KeyError) as exc:
                errors.append(f"Sale-head verification: {exc}")
        references = {f"original_m{mixture}": CHECKPOINT for mixture in (0.5, 1.0)}
        references.update(
            {
                f"{arm}_seed{seed}": root / "training" / f"{arm}_seed{seed}" / "checkpoint.pt"
                for arm in ARMS
                for seed in (0, 1)
            }
        )
        validation_games = [
            json.loads(line) for line in (root / "data/validation/games.jsonl").read_text().splitlines()
        ]
        expected_ids = {(row["seed"], step) for row in validation_games for step in range(row["steps"])}
        for label, checkpoint in references.items():
            folder = root / "full_validation" / label
            if not all((folder / name).exists() for name in ("manifest.json", "summary.json", "predictions.jsonl")):
                errors.append(f"Missing complete validation policy {label}")
                continue
            prediction = read_json(folder / "manifest.json")
            provenance = dict(
                status="complete",
                checkpoint_sha256=sha256(checkpoint),
                validation_data_sha256=sha256(root / "data/validation/summary.json"),
                study_manifest_sha256=sha256(root / "manifest.json"),
                decisions=len(expected_ids),
                temperature=1.0,
                detailed=True,
                policy_mode="evaluation",
            )
            if any(prediction.get(key) != value for key, value in provenance.items()):
                errors.append(f"Validation provenance mismatch {label}")
            mixture = (
                float(label.removeprefix("original_m"))
                if label.startswith("original_m")
                else ARMS[label.rsplit("_seed", 1)[0]][0]
            )
            overrides = {"hand_ar_mixture_eps": mixture} if label.startswith("original_m") else {}
            if (
                prediction["agent_config"]["hand_ar_mixture_eps"] != mixture
                or prediction["explicit_overrides"] != overrides
            ):
                errors.append(f"Validation mixture or override mismatch {label}")
            rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text().splitlines()]
            ids = [(row["seed"], row["step"]) for row in rows]
            if len(ids) != len(expected_ids) or set(ids) != expected_ids:
                errors.append(f"Incomplete or duplicate validation decisions {label}")
            if any(not math.isfinite(row["nll"]) or not 0 <= row["probability"] <= 1 for row in rows):
                errors.append(f"Invalid validation probability {label}")
            saved = read_json(folder / "summary.json")
            if saved["strata"] != summarize_rows(rows):
                errors.append(f"Validation statistics differ from raw decisions {label}")
            id_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
            if prediction["record_ids_sha256"] != id_hash or saved["record_ids_sha256"] != id_hash:
                errors.append(f"Validation decision digest mismatch {label}")
            offline_validations.append(dict(policy=label, decisions=len(rows)))
        if (root / "sale_oracle_cpu").exists():
            try:
                if sale_oracle_results(root, verify_traces=True) is None:
                    errors.append("Exploratory CPU sale replacements are incomplete")
            except (ValueError, KeyError) as exc:
                errors.append(f"Sale-oracle verification: {exc}")
        if (root / "phase_oracle_cpu").exists():
            try:
                if phase_oracle_results(root, verify_traces=True) is None:
                    errors.append("Exploratory CPU phase replacements are incomplete")
            except (ValueError, KeyError) as exc:
                errors.append(f"Phase-oracle verification: {exc}")
        for marker in (
            "replay_complete.json",
            "offline_final_complete.json",
            "development_complete.json",
            "confirmation_complete.json",
        ):
            if not (root / marker).exists():
                errors.append(f"Missing completion marker: {marker}")
        replay_path = root / "current_replay.json"
        if replay_path.exists():
            replays = read_json(replay_path)
            if len(replays) != 20 or len({row["seed"] for row in replays}) != 20:
                errors.append("Fresh recorded-action replay panel is incomplete or duplicated")
            if sum(row["recorded_won"] for row in replays) != 10:
                errors.append("Fresh replay is not balanced between wins and losses")
            for row in replays:
                if any(value is not None for key, value in row.items() if key.startswith("first_")):
                    errors.append(f"Fresh replay diverged at seed {row['seed']}")
                if row["replayed_steps"] != row["recorded_steps"] or not row["env_terminal"]:
                    errors.append(f"Fresh replay ended early at seed {row['seed']}")
        else:
            errors.append("Missing recorded-action replay outcomes")
        if (root / "selection.json").exists() and (root / "confirmation_complete.json").exists():
            from continue_transfer import acceptance

            selected = read_json(root / "selection.json")["arm"]
            matrix = {}
            for arm in ("control", selected):
                matrix[arm] = {}
                for seed in (0, 1):
                    sampled, greedy = [], []
                    for sampling in (0, 1, 2, None):
                        mode = "greedy" if sampling is None else f"sample{sampling}"
                        path = root / "evaluations" / f"confirm_{arm}_seed{seed}" / mode / "outcomes.jsonl"
                        if not path.exists():
                            errors.append(f"Missing confirmation panel {path}")
                            continue
                        rows = [json.loads(line) for line in path.read_text().splitlines()]
                        if sorted(row["seed"] for row in rows) != list(CONFIRM_SEEDS):
                            errors.append(f"Incomplete confirmation panel {path}")
                        (greedy if sampling is None else sampled).extend(rows)
                    matrix[arm][seed] = (sampled, greedy)
            if not errors:
                actual = acceptance(
                    {i: matrix[selected][i][0] for i in (0, 1)},
                    {i: matrix["control"][i][0] for i in (0, 1)},
                    {i: matrix[selected][i][1] for i in (0, 1)},
                    {i: matrix["control"][i][1] for i in (0, 1)},
                    confirmation=True,
                )
                if actual != read_json(root / "confirmation_comparison.json"):
                    errors.append("Stored acceptance differs from raw-game recomputation")
            if ARMS[selected][0] != 0.5:
                for mode in ("sample0", "sample1", "sample2", "greedy"):
                    path = root / "evaluations/confirm_original_same_mixture" / mode / "outcomes.jsonl"
                    if not path.exists():
                        errors.append(f"Missing original-routing confirmation panel {mode}")
                    elif sorted(json.loads(line)["seed"] for line in path.read_text().splitlines()) != list(
                        CONFIRM_SEEDS
                    ):
                        errors.append(f"Incomplete original-routing confirmation panel {mode}")
    result = dict(
        passed=not errors,
        errors=errors,
        original_artifacts=len(originals),
        datasets=datasets,
        trials=trials,
        matched_core_source_files=len(training_sources or {}),
        evaluations=evaluations,
        offline_validation_policies=offline_validations,
        elapsed_seconds=elapsed,
        stage_seconds=stage_seconds,
        stage_limits=limits,
        examples_processed=sum(row["examples"] for row in trials),
        games=sum(row["games"] for row in evaluations),
    )
    write_json(root / ("verification_final.json" if require_complete else "verification_progress.json"), result)
    return result


def report(root, destination):
    lines = [
        "# Learning bottleneck investigation",
        "",
        "**Status and interpretation**",
        "",
        "This follow-up reserves nine additional MPS hours. PPO remains stopped. Original HF data, checkpoints, "
        "and the first transfer report are preserved. Memorization and offline agreement are diagnostics; "
        "gameplay improvements require independent confirmation.",
        "The preceding investigation, including historical replay, matched teacher evaluation, "
        "critic removal, and student-state corrections, is retained in [the first report](report.md). "
        "Its candidate-coverage figures predate the mask correction below.",
        "",
    ]
    ledger = read_json(root / "compute.json")
    elapsed = sum(
        time.time() - j["started_unix"] if j["status"] == "running" else j["elapsed_seconds"] for j in ledger["jobs"]
    )
    lines += [
        f"Accounted accelerator time: **{elapsed / 3600:.3f} / 9 hours**, including a two-minute overhead reserve.",
        "",
    ]
    results = dict(compute=ledger, compute_elapsed_seconds=elapsed, configurations=read_json(root / "manifest.json"))
    phase = phase_oracle_results(root)
    if phase is not None:
        results["phase_oracle"] = phase
    sale = sale_oracle_results(root)
    if sale is not None:
        results["sale_oracle"] = sale
    sale_head = sale_head_results(root)
    if sale_head is not None:
        results["sale_head"] = sale_head
    sale_choice = sale_choice_results(root)
    if sale_choice is not None:
        results["sale_choice"] = sale_choice
    if (root / "budget_adjustments.json").exists():
        results["budget_adjustments"] = read_json(root / "budget_adjustments.json")
    lines += ["| Accelerator stage | Hours used | Hours allocated |", "| --- | ---: | ---: |"]
    for stage, limit in stage_limits(root).items():
        used = sum(
            time.time() - j["started_unix"] if j["status"] == "running" else j["elapsed_seconds"]
            for j in ledger["jobs"]
            if j["stage"] == stage
        )
        lines.append(f"| {stage} | {used / 3600:.3f} | {limit / 3600:.3f} |")
    lines += [
        "",
        "Stage reallocations, if any, are retained in `budget_adjustments.json`; the total remains nine hours.",
        "",
    ]
    confirmation = root / "confirmation_comparison.json"
    if confirmation.exists():
        comparison = read_json(confirmation)
        results["confirmation"] = comparison
        lines += [
            f"Confirmation status: **{comparison['status']}**. Sampled candidate versus control: "
            f"**{delta(comparison['paired']['sampled'])}**. Greedy: {delta(comparison['paired']['greedy'])}.",
            "",
            "Sampled differences for training seeds 0 and 1: "
            + ", ".join(f"{100 * value:+.2f} pp" for value in comparison["sampled_training_replicate_deltas"])
            + ".",
            "",
        ]
        if comparison["paired"]["sampled"]["paired_ci95"][1] < 0.05:
            lines += [
                "The sampled interval's upper bound is below the predeclared five-point improvement target "
                "for these continuation settings and two training replicates.",
                "",
            ]
    else:
        lines += ["Confirmation has not completed; no improvement is validated.", ""]
    reference_path = root / "incumbent_reference_comparison.json"
    if reference_path.exists():
        reference = read_json(reference_path)
        if reference["status"] == "complete":
            contrast = reference["comparisons"][reference["selected"]]
            lines += [
                f"Supplementary comparison with the original checkpoint on {len(reference['seeds'])} seeds: "
                f"selected continuation sampled {delta(contrast['sampled'])}; "
                f"greedy {delta(contrast['greedy'])}. Full outcomes and both continuation contrasts appear below.",
                "",
            ]
    lines += ["**Cause and evidence**", "", "| Finding | Evidence | Interpretation |", "| --- | --- | --- |"]
    if phase is not None:
        summaries = phase["summaries"]
        lines.append(
            "| Hand and shop policies both limit gameplay | On the same CPU seed panel, student wins "
            f"{percentage(summaries['student']['win_rate'])}; teacher hand replacement "
            f"{percentage(summaries['hand']['win_rate'])}; teacher shop replacement "
            f"{percentage(summaries['shop']['win_rate'])}; both "
            f"{percentage(summaries['hand_shop']['win_rate'])}. Joint gain: "
            f"{delta(phase['comparisons']['hand_shop'])}. "
            "| Controlled phase replacements substantially improve wins. Neither alone closes the gap; "
            "this does not distinguish poor imitation from unavailable inputs. |"
        )
    if sale is not None:
        lines.append(
            "| Shop-sale decisions contribute to lost games | Replacing only teacher-prescribed shop Joker "
            f"sales gives {delta(sale['comparisons']['teacher_sale']['student'])} versus the sampled student. "
            "| Exploratory controlled evidence on 50 exposed seeds. The intervention includes sale timing "
            "and Joker selection, uses full-state teacher information, and changes later states. |"
        )
    if sale_choice is not None:
        lines.append(
            "| Conditional Joker choice matters | With the teacher's sale-timing rule in both policies, "
            "student conditional Joker choice minus teacher choice gives "
            f"{delta(sale_choice['comparisons']['teacher_sale'])}. "
            "| Exploratory controlled evidence for the value of Joker selection in this setting. "
            "It does not prove that a particular architecture change will learn the teacher choice. |"
        )
    audit = root / "audit/mixture0.5/summary.json"
    if audit.exists():
        a = read_json(audit)
        results["hf_audit"] = a
        play = a["strata"]["family/play_subset"]["metrics"]
        discard = a["strata"]["family/discard_subset"]["metrics"]
        lines += [
            "| Coverage diagnostic defect | Valid candidates cover "
            f"{percentage(play['candidate_covered']['decision'])} "
            f"of teacher plays and {percentage(discard['candidate_covered']['decision'])} of discards. "
            "The previous metric ignored invalid slots. | Confirmed metric defect; "
            "corrected with a failing/passing regression. |",
            "| Fixed mixture probability cap | With candidates present and the teacher action absent, "
            "its conditional probability is at most 0.5 and NLL is at least log(2). | Mathematical limitation of the "
            "configured policy. Does not establish that alternative actions lose games. |",
        ]
        hand_coverage = a["strata"]["overall"]["metrics"]["candidate_covered"]["decision"]
        lines.append(
            "| Average teacher-probability ceiling on audited hands | Valid candidate coverage is "
            f"{percentage(hand_coverage)}. With candidates present on every audited hand, the .5 mixture limits "
            f"mean conditional teacher-action probability to at most {percentage(0.5 + 0.5 * hand_coverage)}. "
            "| A probability ceiling on these labels, not a bound on greedy agreement or game wins. |"
        )
        covered = a["strata"]["hand/covered"]["metrics"]
        uncovered = a["strata"]["hand/uncovered"]["metrics"]
        lines.append(
            "| Component specialization | AR conditional teacher probability is "
            f"{percentage(covered['ar_probability']['decision'])} on candidate-covered labels versus "
            f"{percentage(uncovered['ar_probability']['decision'])} on uncovered labels. "
            "Mean AR gradient responsibility on covered labels is "
            f"{percentage(covered['ar_responsibility']['decision'])}. "
            "| The pretrained AR component has not learned all covered labels well; changing only inference routing "
            "exposes this weakness. |"
        )
        sales = a["strata"]["family/shop_sell_joker"]
        lines.append(
            "| Rare shop actions are poorly imitated | On "
            f"{sales['decisions']:,} HF teacher Joker sales, exact agreement is "
            f"{percentage(sales['metrics']['accuracy']['decision'])} and teacher-action probability is "
            f"{percentage(sales['metrics']['probability']['decision'])}. "
            "| Aggregate shop accuracy masks this weakness. The sale-conditioned CPU intervention below "
            "tests its gameplay consequences; accuracy alone is not causal evidence. |"
        )
    collisions = root / "audit/collisions.json"
    if collisions.exists():
        c = read_json(collisions)
        lines.append(
            f"| Conflicting identical observations | {len(c['conflicting'])} conflicts among {c['records']:,} "
            "audited decisions. | Absence of exact collisions does not prove observation sufficiency. |"
        )
    if (root / "replay_complete.json").exists():
        replays = read_json(root / "current_replay.json")
        results["current_rule_replay"] = replays
        steps = sum(row["replayed_steps"] for row in replays)
        lines.append(
            f"| Current-rule pipeline equivalence | {len(replays)} complete games and {steps:,} decisions "
            "replayed through FastRunner and BalatroEnv, including saved observations and rewards. "
            "| No divergence in the selected balanced panel; this is bounded evidence, not a proof for all states. |"
        )
    gradient = root / "gradients_v2.json"
    if gradient.exists():
        results["gradients"] = read_json(gradient)
        lines.append(
            "| Gradient flow | Component gradients and actual parameter updates recorded for backbone, "
            "candidate heads, AR heads, and critic. | A 32-state local diagnostic; "
            "not proof about the entire training history. |"
        )
    device = root / "cpu_mps_gradients.json"
    if device.exists():
        results["device_gradients"] = read_json(device)
        lines.append(
            f"| CPU/MPS gradient comparison | Passed: {results['device_gradients']['passed']}; relative L2 difference "
            f"{results['device_gradients']['gradient_relative_l2']:.3g}. "
            "| Eight fixed observations, dropout disabled. |"
        )
    lines += [
        "",
        "**Bounded memorization**",
        "",
        "Same 320 distinct HF observations: 160 plays and 160 discards. "
        "Full model, 300 updates, FP32, batch 320, LR 3e-4, 20-update warmup, no dropout, critic, entropy, "
        "weight decay, or outcome weighting. These observations are not held out.",
        "",
        "| Initialization | AR mixture | Accuracy | Teacher probability | NLL | NLL above mixture floor |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    results["memorization"] = {}
    for initialization in ("epoch7", "random"):
        for mixture in (0.5, 1.0):
            key = memorization_name(initialization, mixture)
            path = root / key / "step300/summary.json"
            if not (root / key / "complete.json").exists():
                continue
            summary = read_json(path)
            results["memorization"][key] = summary
            m = summary["strata"]["overall"]["metrics"]
            lines.append(
                f"| {initialization} | {mixture} | {percentage(m['accuracy']['decision'])} | "
                f"{percentage(m['probability']['decision'])} | {m['nll']['decision']:.4f} | "
                f"{m['excess_nll']['decision']:.4f} |"
            )
    pure_ar_controls = [
        results["memorization"].get(memorization_name(initialization, 1.0)) for initialization in ("epoch7", "random")
    ]
    if all(
        summary is not None
        and summary["strata"]["overall"]["metrics"]["accuracy"]["decision"] >= 0.99
        and summary["strata"]["overall"]["metrics"]["probability"]["decision"] > 0.95
        for summary in pure_ar_controls
    ):
        lines += [
            "",
            "Both pure-AR controls crossed the prespecified memorization thresholds: at least 99% "
            "accuracy and more than 95% teacher-action probability. The conditional free-logit/CPU-MPS "
            "failure investigation was therefore unnecessary. The model and optimizer can fit this small panel; "
            "this does not establish sufficient capacity or effective optimization on the full task.",
        ]
    free = root / "free_output_expressivity.json"
    cross = root / "memorization_cross_eval.json"
    if (root / "memorize_epoch7_mixture0.5/complete.json").exists():
        lines += [
            "",
            "The initial pretrained memorization runs retained attention dropout .1 while standard dropout "
            "was disabled. They are preserved under their original names, excluded from this controlled table, and "
            "superseded by the `dropout0_v2` runs. A regression verifies deterministic training-mode forwards with "
            "attention and standard dropout both disabled; gradient computation remains enabled.",
        ]
    if cross.exists():
        results["memorization_same_weights_routing"] = read_json(cross)
        lines += [
            "",
            "Same-weights routing counterfactuals on the same memorization observations:",
            "",
            "| Initialization / training mixture | Evaluation mixture | Accuracy | Teacher probability |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, value in results["memorization_same_weights_routing"].items():
            m = value["summary"]["strata"]["overall"]["metrics"]
            lines.append(
                f"| {name} | {value['evaluated_mixture']} | {percentage(m['accuracy']['decision'])} | "
                f"{percentage(m['probability']['decision'])} |"
            )
        lines += [
            "",
            "For a hand-action family, the fixed mixture is "
            "`p(action) = (1-epsilon) * p_candidate(action) + epsilon * p_AR(action)`. "
            "A label outside the valid candidate set receives probability only from AR. "
            "For covered labels, the AR gradient responsibility is "
            "`epsilon * p_AR(label) / p(label)`, so a 50/50 mixture does not imply equal useful "
            "imitation gradients for both components. Greedy agreement can remain high while "
            "sampling frequently chooses other actions. These measurements establish the probability "
            "effect on the selected states; gameplay confirmation tests its practical impact.",
        ]
    if free.exists():
        results["free_outputs"] = read_json(free)
        m = results["free_outputs"]["curves"][-1]
        lines += [
            "",
            f"Independent per-observation pure-AR logits: {percentage(m['accuracy'])} accuracy, "
            f"{percentage(m['probability'])} teacher probability after {m['update']} updates. "
            "This removes the encoder and shared mapping; it is an expressivity diagnostic, not a usable policy.",
        ]
    lines += [
        "",
        "**Fresh data and matched training**",
        "",
        "Training seeds 4000000-4000999; offline validation 4100000-4100199; development 120000-120199; "
        "confirmation 130000-130399. Every panel is checked against prior training manifests; old evaluation "
        "panels remain excluded. Teacher settings: current pinned sources, Blue, White Stake, Ante 8, search shops, "
        "scaler growth enabled, no blind rollout, gamma .997, reward snapshot version 3.",
        "",
        "Four arms cross mixture {0.5,1.0} with play/discard BC weight {1,2}. Both training seeds start from "
        "epoch 7 and use identical data order, 750 updates, physical batch 320, FP32, AdamW LR 3e-5, "
        "weight decay .01, 100-update warmup and cosine decay. Dropout .1, critic coefficients .5/.1, entropy .001, "
        "outcome weighting beta 1.5 and clipping [.25,4] stay fixed. Effective BC weights are renormalized.",
        "The loader shuffles Parquet row groups and then decisions within each group to preserve read locality. "
        "These new shards contain one game each, so batches can contain relatively few distinct games. "
        "This ordering is held fixed across arms; broadly mixed minibatches are not tested here. "
        "The continuation length is explicitly 750 updates; inherited epoch-budget metadata is not used by this loop.",
        "",
        "Intermediate offline curves use the same fixed panel of up to 16 decisions per game: "
        "100 training games and all 200 "
        "validation games. Diagnostics preserve training RNG and mode. They are recorded at updates 0/250/500/750.",
        "",
    ]
    for name in ("training", "validation"):
        path = root / "data" / name / "summary.json"
        if path.exists():
            d = read_json(path)
            lines.append(
                f"{name.capitalize()} dataset: {d['records']:,} decisions, {d['wins']}/{d['games']} teacher wins. "
                "This generation win rate is not a matched gameplay contrast."
            )
            if name == "training":
                lines.append(
                    f"Each arm processes 240,000 examples, {240000 / d['records']:.2f} times "
                    "the dataset's record count. "
                    "This is a short continuation; flattening late in its cosine schedule "
                    "does not establish convergence."
                )
    if (root / "generation_complete.json").exists():
        seconds = read_json(root / "generation_complete.json")["elapsed_seconds"]
        lines += [
            "",
            f"CPU-only generation took {seconds / 3600:.2f} wall-clock hours; it is excluded from MPS accounting.",
        ]
    if (root / "lightweight_summary_recovery.json").exists():
        results["diagnostic_summary_recovery"] = read_json(root / "lightweight_summary_recovery.json")
        lines += [
            "",
            "A lightweight-summary defect stopped the first continuation attempt before update 1: "
            "the summary expected candidate details omitted by the lightweight observer. A failing/passing "
            "regression fixed the summary. The zero-update attempt is preserved under `failures/`, "
            "its 10.9 seconds are charged, and the complete matched runs start again from the original checkpoint.",
        ]
    results["learning_curves"] = {}
    for arm in ARMS:
        for seed in (0, 1):
            name = f"{arm}_seed{seed}"
            folder = root / "training" / name / "offline"
            curves = {}
            for update in (0, 250, 500, 750):
                paths = {panel: folder / f"{panel}_{update}/summary.json" for panel in ("training", "validation")}
                if all(path.exists() for path in paths.values()):
                    curves[update] = {panel: read_json(path) for panel, path in paths.items()}
            if curves:
                results["learning_curves"][name] = curves
    if results["learning_curves"]:
        lines += [
            "",
            "Fixed-panel learning curves (initial → latest completed diagnostic):",
            "",
            "| Arm / replicate | Latest update | Training NLL | Validation NLL | Validation play accuracy | "
            "Validation discard accuracy | Validation teacher-play probability |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, curve in results["learning_curves"].items():
            update = max(curve)

            cells = []
            for panel, family, metric in (
                ("training", "overall", "nll"),
                ("validation", "overall", "nll"),
                ("validation", "family/play_subset", "accuracy"),
                ("validation", "family/discard_subset", "accuracy"),
                ("validation", "family/play_subset", "probability"),
            ):
                first = curve[0][panel]["strata"][family]["metrics"][metric]["decision"]
                last = curve[update][panel]["strata"][family]["metrics"][metric]["decision"]
                cells.append(
                    f"{percentage(first)} → {percentage(last)}"
                    if metric in ("accuracy", "probability")
                    else f"{first:.3f} → {last:.3f}"
                )
            lines.append(f"| {name} | {update} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "NLL averages negative log teacher-action probability; the probability column averages the "
            "probability itself. Reducing a few extremely low probabilities can improve NLL while reducing "
            "the arithmetic mean probability. Greedy agreement only requires the teacher action to rank "
            "first. Neither metric alone establishes that sampled actions or complete games improved.",
            "",
        ]
    results["full_validation"] = {}
    for path in sorted((root / "full_validation").glob("*/summary.json")):
        results["full_validation"][path.parent.name] = read_json(path)
    if results["full_validation"]:
        lines += [
            "",
            "**Every-decision offline validation**",
            "",
            "All decisions in the 200 fresh validation games, with no training on those seeds. "
            "Accuracy and NLL below are decision-weighted; the structured summaries also retain game-weighted "
            "results and action-family/ante/outcome strata. Differences between mixtures include their "
            "probability parameterization.",
            "",
            "| Policy | Overall accuracy | Play accuracy / NLL | Discard accuracy / NLL | Hand teacher probability |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, summary in results["full_validation"].items():
            strata = summary["strata"]
            overall = strata["overall"]["metrics"]
            play = strata["family/play_subset"]["metrics"]
            discard = strata["family/discard_subset"]["metrics"]
            hand_strata = [strata[f"family/{family}"] for family in ("play_subset", "discard_subset")]
            hand_probability = sum(s["decisions"] * s["metrics"]["probability"]["decision"] for s in hand_strata) / sum(
                s["decisions"] for s in hand_strata
            )
            lines.append(
                f"| {name} | {percentage(overall['accuracy']['decision'])} | "
                f"{percentage(play['accuracy']['decision'])} / {play['nll']['decision']:.3f} | "
                f"{percentage(discard['accuracy']['decision'])} / {discard['nll']['decision']:.3f} | "
                f"{percentage(hand_probability)} |"
            )
        lines += [
            "",
            "Teacher-action probability on hand decisions, split by valid candidate coverage:",
            "",
            "| Policy | Covered labels | Uncovered labels |",
            "| --- | ---: | ---: |",
        ]
        for name, summary in results["full_validation"].items():
            covered = summary["strata"]["hand/covered"]["metrics"]["probability"]["decision"]
            uncovered = summary["strata"]["hand/uncovered"]["metrics"]["probability"]["decision"]
            lines.append(f"| {name} | {percentage(covered)} | {percentage(uncovered)} |")
        if all(name in results["full_validation"] for name in ("original_m1.0", "pure_ar_seed0", "pure_ar_seed1")):
            original = results["full_validation"]["original_m1.0"]["strata"]
            changes = {}
            for label in ("hand/covered", "hand/uncovered", "family/play_subset"):
                changes[label] = {}
                for metric in ("probability", "accuracy", "nll"):
                    before = original[label]["metrics"][metric]["decision"]
                    after = (
                        sum(
                            results["full_validation"][f"pure_ar_seed{seed}"]["strata"][label]["metrics"][metric][
                                "decision"
                            ]
                            for seed in (0, 1)
                        )
                        / 2
                    )
                    changes[label][metric] = dict(before=before, mean_after=after)
            results["autoregressive_validation_changes"] = changes
            covered = changes["hand/covered"]["probability"]
            uncovered = changes["hand/uncovered"]["probability"]
            lines += [
                "",
                "Relative to the original checkpoint using autoregressive routing, the two pure-AR "
                "continuations raise mean teacher-action probability on covered hand labels from "
                f"{percentage(covered['before'])} to {percentage(covered['mean_after'])}, while lowering it "
                f"on uncovered labels from {percentage(uncovered['before'])} to "
                f"{percentage(uncovered['mean_after'])}. This is a redistribution of prediction quality "
                "across label groups; the reduction in large NLL errors does not imply uniformly sharper "
                "teacher-action probabilities. The sampled-game comparisons determine its gameplay value.",
            ]
        lines += [
            "",
            "Shop action-family agreement on the same complete validation panel. The sale family can also "
            "occur outside the shop phase:",
            "",
            "| Policy | Buy | Leave | Reroll | Sell Joker | Teacher-sale probability |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, summary in results["full_validation"].items():
            strata = summary["strata"]
            cells = [
                percentage(strata[f"family/{family}"]["metrics"]["accuracy"]["decision"])
                for family in ("shop_buy", "shop_leave", "shop_reroll", "shop_sell_joker")
            ]
            cells.append(percentage(strata["family/shop_sell_joker"]["metrics"]["probability"]["decision"]))
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "These are exact teacher-action matches. Different Joker sales can be equally useful; weak sale "
            "agreement alone does not establish a causal contribution to lost games.",
        ]
    lines += [
        "",
        "**Gameplay comparisons**",
        "",
        "Primary comparisons use temperature 1.0, Blue Deck, White Stake, "
        "Ante 8, batch 32, four workers, and sampling seeds 0/1/2. The inference-only temperature-.5 sweep is "
        "separate. Sampling and training repeats are averaged within game seed before 10,000 paired bootstrap "
        "resamples. Two training replicates do not estimate uncertainty across a broad population of training runs. "
        "A win advances the engine to ante 9, so mean final ante can exceed 8.",
        "",
    ]
    for key, title in (("inference_comparisons", "Inference-only screen"), ("screen_comparisons", "Training screen")):
        path = root / f"{key}.json"
        if not path.exists():
            continue
        results[key] = read_json(path)
        lines += [
            f"**{title}**",
            "",
            "| Treatment versus control | Sampled difference | Greedy difference |",
            "| --- | ---: | ---: |",
        ]
        for arm, comparison in results[key].items():
            comparison = comparison.get("paired", comparison)
            lines.append(f"| {arm} | {delta(comparison['sampled'])} | {delta(comparison['greedy'])} |")
        lines.append("")
    for key in ("development_comparison", "confirmation_training_beyond_routing"):
        path = root / f"{key}.json"
        if path.exists():
            results[key] = read_json(path)
            comparison = results[key].get("paired", results[key])
            lines += [f"{key}: sampled {delta(comparison['sampled'])}; greedy {delta(comparison['greedy'])}.", ""]
    if (root / "confirmation_complete.json").exists():
        selected = read_json(root / "selection.json")["arm"]
        results["confirmation_panels"] = {}
        for arm in ("control", selected):
            for seed in (0, 1):
                for mode in ("sampled", "greedy"):
                    folders = [f"sample{i}" for i in (0, 1, 2)] if mode == "sampled" else ["greedy"]
                    rows = []
                    for folder in folders:
                        path = root / "evaluations" / f"confirm_{arm}_seed{seed}" / folder / "outcomes.jsonl"
                        rows.extend(json.loads(line) for line in path.read_text().splitlines())
                    results["confirmation_panels"][f"{arm}/seed{seed}/{mode}"] = summarize(rows)
        original = root / "evaluations/confirm_original_same_mixture"
        if original.exists():
            for mode in ("sampled", "greedy"):
                folders = [f"sample{i}" for i in (0, 1, 2)] if mode == "sampled" else ["greedy"]
                rows = []
                for folder in folders:
                    rows.extend(
                        json.loads(line) for line in (original / folder / "outcomes.jsonl").read_text().splitlines()
                    )
                results["confirmation_panels"][f"original/same_mixture/{mode}"] = summarize(rows)
        lines += [
            "**Confirmation outcomes**",
            "",
            "| Policy / replicate / mode | Wins / games | Win rate [cluster 95% CI] | Mean final ante | Stalls |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, summary in results["confirmation_panels"].items():
            low, high = summary["win_rate_cluster_ci95"]
            lines.append(
                f"| {name} | {summary['wins']}/{summary['games']} | {percentage(summary['win_rate'])} "
                f"[{percentage(low)}, {percentage(high)}] | {summary['mean_final_ante']:.3f} | {summary['stalls']} |"
            )
        lines += [
            "",
            "Ante-clear rates use all games in each row as their denominator:",
            "",
            "| Policy / replicate / mode | " + " | ".join(f"Ante {a}" for a in range(1, 9)) + " |",
            "| --- | " + " | ".join(["---:"] * 8) + " |",
        ]
        for name, summary in results["confirmation_panels"].items():
            lines.append(
                f"| {name} | " + " | ".join(percentage(summary["ante_clear_rates"][str(a)]) for a in range(1, 9)) + " |"
            )
        lines.append("")
    reference_path = root / "incumbent_reference_comparison.json"
    if reference_path.exists():
        reference = read_json(reference_path)
        results["incumbent_reference"] = reference
        lines += ["", "**Supplementary original-reference comparison**", ""]
        if reference["status"] == "skipped_budget":
            lines += [
                "The planned check against the original mixed policy was skipped because the "
                "remaining accelerator time did not meet its admission margin. The primary 400-seed "
                "candidate-versus-continuation-control comparison is reported separately.",
                "",
            ]
        else:
            lines += [
                f"On {len(reference['seeds'])} confirmation seeds, the original checkpoint uses mixture .5, "
                "temperature 1, three sampling repeats, and greedy evaluation. The training replicates "
                "share that original baseline; the bootstrap unit remains the game seed. This supplementary "
                "comparison and its time-only choice of 200 or 400 seeds were planned during development "
                "and do not alter primary acceptance.",
                "",
                "| Continuation versus original reference | Sampled difference | Greedy difference |",
                "| --- | ---: | ---: |",
            ]
            for arm, comparison in reference["comparisons"].items():
                lines.append(f"| {arm} | {delta(comparison['sampled'])} | {delta(comparison['greedy'])} |")
            lines.append("")
            results["original_reference_panels"] = {}
            lines += [
                "| Original reference mode | Wins / games | Win rate [cluster 95% CI] | Mean final ante |",
                "| --- | ---: | ---: | ---: |",
            ]
            for mode in ("sampled", "greedy"):
                folders = [f"sample{i}" for i in (0, 1, 2)] if mode == "sampled" else ["greedy"]
                rows = []
                for folder in folders:
                    path = root / "evaluations" / reference["evaluation_name"] / folder / "outcomes.jsonl"
                    rows.extend(json.loads(line) for line in path.read_text().splitlines())
                summary = summarize(rows)
                results["original_reference_panels"][mode] = summary
                low, high = summary["win_rate_cluster_ci95"]
                lines.append(
                    f"| {mode} | {summary['wins']}/{summary['games']} | {percentage(summary['win_rate'])} "
                    f"[{percentage(low)}, {percentage(high)}] | {summary['mean_final_ante']:.3f} |"
                )
            lines.append("")
    if phase is not None:
        lines += [
            "**Exploratory phase replacements (CPU only)**",
            "",
            "Original epoch 7, Blue/White/Ante 8, mixture .5, temperature 1.0, CPU batch 1, "
            "50 separate seeds 5000000-5000049 and sampling repeats 0/1/2. The full teacher runs once per seed. "
            "Each hybrid draws the student proposal before replacing it. Hand replacement covers every "
            "CHOOSE_ACTION decision, including consumables and reordering; shop replacement covers SHOP, "
            "while booster packs remain with the student. Compare arms within this experiment, since its "
            "batching and per-game sampling streams differ from the primary MPS evaluations.",
            "",
            "| Controller | Wins / games | Win rate [cluster 95% CI] | Mean final ante | Paired gain over student |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for arm, summary in phase["summaries"].items():
            low, high = summary["win_rate_cluster_ci95"]
            difference = delta(phase["comparisons"][arm]) if arm != "student" else "reference"
            lines.append(
                f"| {arm} | {summary['wins']}/{summary['games']} | {percentage(summary['win_rate'])} "
                f"[{percentage(low)}, {percentage(high)}] | {summary['mean_final_ante']:.3f} | {difference} |"
            )
        conditional = phase["secondary_conditional_comparisons"]
        lines += [
            "",
            "Exploratory conditional contrasts on the same games: adding teacher shop decisions when "
            f"hands already use the teacher yields {delta(conditional['teacher_shop_given_teacher_hands'])}; "
            "adding teacher hand decisions when shops already use the teacher yields "
            f"{delta(conditional['teacher_hands_given_teacher_shop'])}. "
            f"Both phases versus full teacher: {delta(conditional['hand_shop_minus_full_teacher'])}.",
        ]
        lines += [
            "",
            "Teacher queries use isolated copies and check that the live state and RNG stay unchanged. "
            "The teacher sees full simulator state, so a successful phase replacement cannot separate poor "
            "imitation from omitted observation inputs. Replacing a policy also changes later visited states. "
            "ShopSearch uses public, seed-independent deck samples; exact hand scoring can use copied run RNG state. "
            "The contribution of that stochastic information was not isolated. "
            "These interventions localize gameplay bottlenecks; they do not validate a learned candidate or "
            "supply training labels. All traces and canonical-order statistics are preserved.",
            "",
            f"CPU diagnostic wall time: {phase['manifest']['elapsed_seconds'] / 3600:.2f} hours; "
            "zero accelerator workload. These seeds remain excluded from training.",
            "",
        ]
    if sale is not None:
        phase_note = "The sale action family can occur outside SHOP; this intervention is restricted to SHOP."
        if (root / "sale_phase_audit.json").exists():
            sale_phases = read_json(root / "sale_phase_audit.json")
            for path, digest in sale_phases["sources"].items():
                if sha256(path) != digest:
                    raise ValueError("Sale phase audit source changed")
            results["sale_phase_audit"] = sale_phases
            phase_note = (
                "The sale family spans phases: 1,059 of the audited sales occur in SHOP (5.85% agreement), "
                "148 in booster packs, and 30 during hand play. Phase fields were joined from the prior audit "
                "only after all 44,348 decision identifiers, action families, accuracies, and NLLs matched exactly. "
                "The intervention below is restricted to SHOP."
            )
        lines += [
            "**Exploratory Joker-sale follow-up**",
            "",
            "The HF audit found 6.22% exact agreement on 1,237 teacher Joker sales, compared with about 94% "
            "on buying, leaving, and rerolling. This motivated a post-hoc CPU experiment on the same "
            "50 exploratory seeds and three sampling repeats. In each shop state, an isolated teacher "
            "proposal determines which intervention applies: replace only teacher-prescribed Joker sales, "
            "or replace only other teacher shop actions. All other decisions use the sampled student. "
            "The student proposal is drawn before replacement. These interventions do not select a model.",
            "",
            phase_note,
            "",
            "| Teacher replacements | Wins / 150 | Paired gain over student | Difference from full shop oracle |",
            "| --- | ---: | ---: | ---: |",
        ]
        for arm, summary in sale["summaries"].items():
            lines.append(
                f"| {arm} | {summary['wins']} / {summary['games']} | "
                f"{delta(sale['comparisons'][arm]['student'])} | {delta(sale['comparisons'][arm]['shop'])} |"
            )
        lines += [
            "",
            "The partitions depend on the teacher proposal at each visited state; their win-rate effects "
            "are not additive. Teacher-sale replacement does not suppress a student sale when the teacher "
            "would take a different action. The teacher retains full-state information. This is an "
            "exploratory follow-up with exposed seeds, not independent confirmation or a learned improvement.",
            "",
            f"Additional CPU wall time: {sale['manifest']['elapsed_seconds'] / 3600:.2f} hours; zero MPS workload.",
            "",
        ]
    if sale_head is not None:
        head = sale_head["summary"]["SHOP"]
        metrics = head["metrics"]
        lines += [
            "**Where sale probability is lost**",
            "",
            f"A CPU pass over all {sale_head['manifest']['decisions']:,} audited sale labels decomposes "
            "the complete-action probability into choosing the sale family and choosing the Joker slot. "
            f"On the {head['decisions']:,} SHOP labels, mean sale-family probability is "
            f"{percentage(metrics['macro_probability'])}; conditional teacher-Joker probability is "
            f"{percentage(metrics['conditional_probability'])}, versus "
            f"{percentage(head['legal_uniform_teacher_probability'])} for a uniform legal-Joker choice. "
            f"The resulting mean complete teacher-action probability is {percentage(metrics['probability'])}. "
            f"Forced-family Joker accuracy is {percentage(metrics['conditional_accuracy'])}.",
            "",
            "The current sale head predicts fixed slot logits from the pooled state; buying scores individual "
            "item-token embeddings. A Joker-token-conditioned sale head is therefore a specific architecture "
            "hypothesis, not an established fix or evidence that v13 observations are insufficient. "
            "CPU/MPS NLL differences on these labels are at most "
            f"{sale_head['manifest']['max_cpu_mps_nll_difference']:.2g}; teacher-agreement indicators match.",
            "",
        ]
    if sale_choice is not None:
        summary = sale_choice["summary"]
        lines += [
            "**Sale timing versus Joker choice**",
            "",
            "A further post-hoc CPU panel uses the teacher's SHOP sale-timing rule but samples which Joker to "
            "sell from the student's conditional head. An independent target-selection RNG leaves the main "
            "student sampling stream unchanged. The control is the completed teacher-sale intervention "
            "with both teacher timing and teacher Joker choice, on the same 50 seeds and three repeats.",
            "",
            f"Teacher timing with student Joker choice wins **{summary['wins']}/{summary['games']}** "
            f"({percentage(summary['win_rate'])}). Paired difference versus the original student: "
            f"**{delta(sale_choice['comparisons']['student'])}**. Student Joker choice minus teacher Joker "
            "choice, with both using the teacher's sale-timing rule: "
            f"**{delta(sale_choice['comparisons']['teacher_sale'])}**.",
            "",
            "This tests conditional Joker choice within a teacher-timing policy. It does not isolate a "
            "specific observation input; changing a sale also changes subsequent states. A confidence "
            "interval containing zero would not establish equivalence between the choice policies. "
            "The exploratory intervals are not adjusted for the sequence of follow-up comparisons.",
            "",
        ]
    if (root / "regression_verification.json").exists():
        regression = read_json(root / "regression_verification.json")
        results["regression_tests"] = regression
        lines += [
            "**Verification**",
            "",
            f"{regression.get('distinct_tests_passed', regression['tests_passed'])} distinct relevant "
            "regression tests passed across the recorded suites. "
            "Coverage includes deck propagation, checkpoint/schema restoration, observations and replay, "
            "policy probabilities and legality, snapshot/RNG isolation, label provenance, paired statistics, "
            "budget accounting, phase intervention boundaries, and the confirmed diagnostic defects.",
            "",
        ]
    if (root / "acceptance_infrastructure_fix.json").exists():
        guard = read_json(root / "acceptance_infrastructure_fix.json")
        results["acceptance_infrastructure_fix"] = guard
        lines += [
            "The acceptance helper previously relied on the artifact verifier to reject stalled or censored "
            "games. A failing/passing regression now enforces rejection in the helper itself. "
            f"The healthy comparison across {guard['healthy_games']} recorded outcomes is unchanged. "
            "This is a verification fix and does not change policy behavior or explain poor gameplay.",
            "",
        ]
    if (root / "verification_final.json").exists():
        verification_result = read_json(root / "verification_final.json")
        results["artifact_verification"] = verification_result
        lines += [
            f"Final artifact verification passed: **{verification_result['passed']}**. "
            f"It checks {verification_result['original_artifacts']} preserved original artifacts, "
            "generated data, matched training configurations and sources, raw game panels, "
            "phase traces, paired acceptance, and the nine-hour budget.",
            "",
        ]
    lines += [
        "**Acceptance, limitations, and artifacts**",
        "",
        "Validation requires a sampled gain of at least five percentage points, paired 95% interval above zero, "
        "positive sampled gains in both training replicates, greedy interval ruling out a regression beyond five "
        "points, and no new legality/nonfinite/infrastructure failures. Lower loss alone never validates gameplay.",
        "",
        "The training factorial isolates mixture and hand weighting on shared current-rule data; it does not "
        "isolate data regeneration itself. Memorizing 320 states does not establish capacity or convergence "
        "on the full task. "
        "HF observations retain historical-rule limitations. `macro_accuracy` measures the family of the joint "
        "greedy action; `marginal_macro_accuracy`, where present, measures the macro head's own argmax.",
        "",
        f"Full artifacts and reproducible per-game results: `{root}`. Commands: "
        "`tools/learning_bottleneck.py --help`, `tools/run_learning_bottleneck.py --help`, and "
        "`tools/report_learning_bottleneck.py --require-complete`. PPO remains stopped.",
    ]
    if (root / "confirmation_complete.json").exists():
        selected = read_json(root / "selection.json")["arm"]
        confirmed = read_json(root / "confirmation_comparison.json")
        recommendation = dict(
            confirmation_status=confirmed["status"],
            selected_arm=selected,
            experiment=(
                "Compare the current pooled-slot Joker-sale head against a residual score conditioned on each "
                "Joker token and the global state. Initialize the added output to zero so initial policy "
                "probabilities match, and record the architecture variant explicitly in checkpoint metadata."
            ),
            controls=(
                "Use pinned current-rule records, unchanged v13 observations, two matched training seeds, "
                "identical data ordering, objectives, optimizer settings, and update counts. Keep the rest "
                "of the policy trainable under the same settings in both arms."
            ),
            measurements=(
                "Track conditional Joker-selection probability and sale decisions on all visited states, "
                "including excess sales when the teacher would choose another action. Evaluate sampled "
                "end-to-end games as the primary outcome; teacher-hand-assisted games can diagnose the "
                "shop contribution. Reserve new confirmation seeds because this study's panels are exposed."
            ),
            status="Recommended next experiment; not run or validated in this investigation",
        )
        results["next_training_recommendation"] = recommendation
        lines += [
            "",
            "**Next training recommendation**",
            "",
            (
                f"The {selected} continuation meets the specified confirmation criteria and can serve as "
                "the experimental baseline for the next supervised comparison."
                if confirmed["status"] == "validated"
                else "No continuation in this study has met every validation criterion. Preserve the original "
                "checkpoint as the reference; do not promote a trial based on its lower imitation loss."
            ),
            "",
            recommendation["experiment"],
            recommendation["controls"],
            "",
            recommendation["measurements"],
            "",
            "The rationale is the controlled shop-sale gain, the timing-versus-choice intervention, "
            "and the near-uniform conditional Joker choice, "
            "together with the existing difference between pooled sale logits and token-based buy logits. "
            "This proposes a test of a specific architecture hypothesis. The present experiments do not "
            "establish convergence on the full dataset, identify an indispensable missing feature, or "
            "justify starting PPO.",
        ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    write_json(destination.with_suffix(".json"), results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/learning_bottleneck_20260913"))
    parser.add_argument("--output", type=Path, default=Path("docs/transfer/learning_bottleneck.md"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.verify or args.require_complete:
        result = verification(args.root, require_complete=args.require_complete)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    report(args.root, args.output)


if __name__ == "__main__":
    main()
