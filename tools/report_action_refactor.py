#!/usr/bin/env python3
"""Recompute completed action-refactor results from raw games and report confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_refactor_panels import compare

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.transfer import sha256, summarize


def percentage(value):
    return f"{100 * value:.2f}%"


def difference(result):
    low, high = result["paired_ci95"]
    return f"{100 * result['sampled_or_greedy_delta']:+.2f} pp [{100 * low:+.2f}, {100 * high:+.2f}]"


def load_verified_panel(path):
    saved = read_json(path)
    protocol = saved["protocol"]
    name = path.name.removesuffix("_comparisons.json")
    groups, physical, evidence = {}, {}, []
    for key in ("control", 0, 1):
        label = "control" if key == "control" else f"seed{key}"
        modes = {}
        for sampling in (0, 1, 2, None):
            mode = "greedy" if sampling is None else f"sample{sampling}"
            directory = path.parent / "evaluations" / f"{name}_{label}" / mode
            manifest = read_json(directory / "manifest.json")
            expected = dict(
                status="complete",
                seeds=protocol["seeds"],
                sampling_seed=sampling,
                greedy=sampling is None,
                checkpoint_sha256=protocol["checkpoint_hashes"][label],
                deck="b_blue",
                stake=1,
                win_ante=8,
                temperature=1.0,
                batch_size=32,
                workers=4,
                device=protocol["device"],
            )
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Evaluation configuration mismatch: {directory}")
            if manifest["agent_config"]["hand_ar_mixture_eps"] != 0.5:
                raise ValueError(f"Evaluation mixture mismatch: {directory}")
            if sha256(manifest["checkpoint"]) != manifest["checkpoint_sha256"]:
                raise ValueError(f"Evaluated checkpoint changed: {directory}")
            outcomes_path = directory / "outcomes.jsonl"
            rows = [json.loads(line) for line in outcomes_path.read_text().splitlines()]
            if len(rows) != len(protocol["seeds"]) or {r["seed"] for r in rows} != set(protocol["seeds"]):
                raise ValueError(f"Incomplete or duplicate seed panel: {directory}")
            if any(r.get("stalled") or r.get("censored") or r["sampling_seed"] != sampling for r in rows):
                raise ValueError(f"Incomplete game or wrong sampling repeat: {directory}")
            if summarize(rows) != read_json(directory / "summary.json"):
                raise ValueError(f"Saved summary differs from raw games: {directory}")
            modes[sampling] = rows
            evidence.append(
                dict(
                    path=str(directory),
                    manifest_sha256=sha256(directory / "manifest.json"),
                    outcomes_sha256=sha256(outcomes_path),
                    games=len(rows),
                    elapsed_seconds=manifest["elapsed_seconds"],
                )
            )
        groups[key] = ([row for sampling in (0, 1, 2) for row in modes[sampling]], modes[None])
        physical[label] = {
            mode: summarize(rows) for mode, rows in zip(("sampled", "greedy"), groups[key], strict=True)
        }
    recomputed = json.loads(json.dumps(compare(groups)))
    if recomputed != saved["results"]:
        raise ValueError(f"Paired comparisons differ from raw-game recomputation: {path}")
    return dict(
        **saved,
        physical_arms=physical,
        evaluation_files=evidence,
        comparison_path=str(path),
        comparison_sha256=sha256(path),
        physical_games=sum(item["games"] for item in evidence),
        evaluation_job_seconds=sum(item["elapsed_seconds"] for item in evidence),
    )


def compile_results(root):
    status = read_json(root / "confirmation_status.json")
    verification = read_json(root / "verification_final.json")
    if status["status"] != "complete" or not verification["passed"] or not verification["final"]:
        raise ValueError("A final report requires completed confirmation and a passing final artifact audit")
    selection = read_json(root / "candidate_selection.json")
    representative = read_json(root / "representative_checkpoint_rule.json")
    if representative["representative_training_seed"] != 0 or not representative["recorded_before_confirmation"]:
        raise ValueError("The pre-confirmation representative checkpoint rule changed")
    confirmation_path = Path(status["comparison"])
    if sha256(confirmation_path) != status["comparison_sha256"]:
        raise ValueError("Completed confirmation comparison changed")
    confirmation = load_verified_panel(confirmation_path)
    if confirmation["protocol"]["checkpoint_hashes"] != selection["checkpoint_hashes"]:
        raise ValueError("Confirmation did not evaluate the frozen checkpoints")
    if confirmation["protocol"]["seeds"] != read_json(root / "manifest.json")["confirmation_seeds"]:
        raise ValueError("Confirmation did not use the entire reserved panel")
    if len(confirmation["protocol"]["seeds"]) != 400:
        raise ValueError("Confirmation requires 400 game seeds")
    if confirmation["validated"] != all(confirmation["acceptance"].values()):
        raise ValueError("Confirmation acceptance flags disagree")
    development = {}
    for family, name in selection["source_panels"].items():
        path = Path(name)
        if sha256(path) != selection["development_comparison_hashes"][family]:
            raise ValueError("A development result changed after selection")
        development[family] = load_verified_panel(path)
    family = selection["family"]
    prefix = "entity_sale_residual_v1" if family == "sale" else "entity_sale_subset_residual_v1"
    checkpoint = root / f"{prefix}_seed0" / "checkpoint.pt"
    if sha256(checkpoint) != selection["checkpoint_hashes"]["seed0"]:
        raise ValueError("Representative checkpoint differs from the frozen selection")
    ledger = read_json(root / "compute.json")
    previous = read_json(Path("eval/learning_bottleneck_20260913/compute.json"))
    status_name = "validated" if confirmation["validated"] else "inconclusive"
    if not confirmation["validated"]:
        low, high = confirmation["results"]["sampled"]["paired"]["paired_ci95"]
        if high < 0:
            status_name = "negative"
        elif low > 0 and confirmation["acceptance"]["greedy_rules_out_five_point_regression"]:
            status_name = "promising"
    return dict(
        status=status_name,
        selected_family=family,
        representative_checkpoint=str(checkpoint),
        representative_checkpoint_sha256=sha256(checkpoint),
        representative_rule=representative,
        selection=selection,
        development=development,
        confirmation=confirmation,
        verification_sha256=sha256(root / "verification_final.json"),
        original_artifacts_verified=verification["original_artifacts"],
        training_examples_processed=verification["training_examples_processed"],
        post_hoc_control_examples_processed=verification["post_hoc_control_examples_processed"],
        accelerator_seconds_this_work=ledger["elapsed_seconds"],
        accelerator_seconds_latest_allowance=previous["elapsed_seconds"] + ledger["elapsed_seconds"],
        accelerator_seconds_remaining=ledger["limit_seconds"] - ledger["elapsed_seconds"],
        source_hashes={str(Path(__file__).relative_to(Path.cwd())): sha256(__file__)},
    )


def render(result):
    confirmation = result["confirmation"]
    lines = [
        f"Confirmation result: **{result['status']}**. Selected family: `{result['selected_family']}`.",
        "",
        "These results are recomputed from complete raw outcomes and checked against the frozen selection. "
        "The [methods and causal evidence](action_refactor.md) describe the architecture and exploratory studies.",
        "",
        "All full panels use CPU, Blue Deck, White Stake, Ante 8, temperature 1, mixture .5, batch 32, "
        "four engine workers and FP32. Sampling repeats are 0/1/2. Both training replicas are evaluated. "
        "Each candidate uses the same sampler implementation as its matched control. "
        "Intervals resample game seeds after averaging sampling and training repeats; the shared control "
        "is not counted as two independent sets of games.",
        "",
        "| Panel | Sampled control | Sampled candidate | Paired difference, 95% CI | "
        "Greedy control | Greedy candidate | Paired difference, 95% CI |",
        "| --- | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    panels = [(f"Development: {family}, 200 seeds", panel) for family, panel in result["development"].items()]
    panels.append(("Confirmation: 400 seeds", confirmation))
    for name, panel in panels:
        sampled, greedy = (panel["results"][mode] for mode in ("sampled", "greedy"))
        lines.append(
            f"| {name} | {percentage(sampled['control']['win_rate'])} | "
            f"{percentage(sampled['candidate']['win_rate'])} | {difference(sampled['paired'])} | "
            f"{percentage(greedy['control']['win_rate'])} | {percentage(greedy['candidate']['win_rate'])} | "
            f"{difference(greedy['paired'])} |"
        )
    lines += [
        "",
        "Confirmation counts below are physical games. The combined candidate has two trained replicas; "
        "the control's games are reused only to form paired comparisons.",
        "",
        "| Confirmation arm | Sampled wins / games | Sampled win rate, 95% CI | Mean final ante | "
        "Greedy wins / games | Greedy win rate, 95% CI | Mean final ante |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for name, metrics in confirmation["physical_arms"].items():
        values = []
        for mode in ("sampled", "greedy"):
            metric = metrics[mode]
            low, high = metric["win_rate_cluster_ci95"]
            values.append(
                f"{metric['wins']}/{metric['games']} | {percentage(metric['win_rate'])} "
                f"[{percentage(low)}, {percentage(high)}] | {metric['mean_final_ante']:.3f}"
            )
        lines.append(f"| {name} | {' | '.join(values)} |")
    lines += ["", "| Training replicate | Sampled paired difference, 95% CI | Greedy paired difference, 95% CI |",
              "| --- | --- | --- |"]
    for seed, metrics in confirmation["results"]["per_replicate"].items():
        lines.append(f"| {seed} | {difference(metrics['sampled'])} | {difference(metrics['greedy'])} |")
    lines += ["", "| Ante cleared | Sampled control | Sampled candidates | Greedy control | Greedy candidates |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for ante in range(1, 9):
        values = [
            percentage(confirmation["results"][mode][arm]["ante_clear_rates"][str(ante)])
            for mode in ("sampled", "greedy") for arm in ("control", "candidate")
        ]
        lines.append(f"| {ante} | {' | '.join(values)} |")
    lines += ["", "| Pre-recorded acceptance requirement | Passed |", "| --- | --- |"]
    for name, passed in confirmation["acceptance"].items():
        lines.append(f"| {name.replace('_', ' ')} | {'Yes' if passed else 'No'} |")
    lines += [
        "",
        f"Confirmation completed {confirmation['physical_games']:,} physical games without stalled or censored "
        "trajectories. The evaluator checks action legality and finite selected log probabilities and critic outputs; "
        "an exception prevents a complete panel and validation.",
        "",
        f"The preselected representative is `{result['representative_checkpoint']}` "
        f"(SHA-256 `{result['representative_checkpoint_sha256']}`). Training seed 0 was chosen as the "
        "representative before confirmation; both replicas above remain part of the evidence.",
        "",
        f"The final audit verified {result['original_artifacts_verified']} protected original artifacts, "
        f"all adapter/cache/checkpoint hashes and {result['training_examples_processed']:,} CPU training examples. "
        f"Separate post-hoc CPU controls processed {result['post_hoc_control_examples_processed']:,} examples. "
        f"This architecture work used {result['accelerator_seconds_this_work'] / 60:.2f} MPS minutes; "
        f"cumulative use of the latest nine-hour allowance was "
        f"{result['accelerator_seconds_latest_allowance'] / 3600:.4f} hours, leaving "
        f"{result['accelerator_seconds_remaining']:.2f} seconds. Evaluation durations are recorded per job; "
        "their sum is not elapsed machine wall time when CPU jobs overlap.",
        "",
        "Recompute with `tools/verify_action_refactor.py --final`, then `tools/report_action_refactor.py`. "
        "Structured results, exact counts, source hashes and outcome hashes accompany this document in "
        "`action_refactor_results.json`. PPO remains stopped.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--output", type=Path, default=Path("docs/transfer/action_refactor_results.md"))
    args = parser.parse_args()
    result = compile_results(args.root)
    args.output.write_text(render(result))
    atomic_json(args.output.with_suffix(".json"), result)
    summary = {key: result[key] for key in ("status", "selected_family", "representative_checkpoint")}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
