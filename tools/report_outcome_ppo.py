#!/usr/bin/env python3
"""Verify raw outcome panels, compare training replicas, and freeze confirmation inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT, outcome_reward
from pylatro_agent.transfer import paired_statistics, sha256, summarize

if __package__:
    from .outcome_resource_plan import RESOURCE_PLAN, cpu_confirmation_plan
else:
    from outcome_resource_plan import RESOURCE_PLAN, cpu_confirmation_plan


def compare_replicates(candidates, control):
    if set(candidates) != {0, 1}:
        raise ValueError("Both training replicas are required")
    result = dict(per_replicate={})
    for seed, candidate in candidates.items():
        result["per_replicate"][seed] = {
            mode: paired_statistics(candidate[mode], control[mode]) for mode in ("sampled", "greedy")
        }
    for mode in ("sampled", "greedy"):
        pooled = [dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                  for seed in (0, 1) for row in candidates[seed][mode]]
        reference = [dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                     for seed in (0, 1) for row in control[mode]]
        result[mode] = dict(candidate=summarize(pooled), control=summarize(control[mode]),
                            paired=paired_statistics(pooled, reference))
    return result


def compare_matched_training(candidates, controls):
    """Compare an intervention with its trained control for each training replica."""
    if set(candidates) != {0, 1} or set(controls) != {0, 1}:
        raise ValueError("Both matched training replicas are required")
    result = dict(per_replicate={})
    for seed in (0, 1):
        result["per_replicate"][seed] = {
            mode: paired_statistics(candidates[seed][mode], controls[seed][mode])
            for mode in ("sampled", "greedy")
        }
    for mode in ("sampled", "greedy"):
        pooled = [dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                  for seed in (0, 1) for row in candidates[seed][mode]]
        reference = [dict(row, sampling_seed=f"train{seed}/sample{row['sampling_seed']}")
                     for seed in (0, 1) for row in controls[seed][mode]]
        result[mode] = dict(candidate=summarize(pooled), control=summarize(reference),
                            paired=paired_statistics(pooled, reference))
    return result


def acceptance_rules(results, *, independent_confirmation):
    sampled, greedy = results["sampled"]["paired"], results["greedy"]["paired"]
    return dict(
        independent_confirmation=independent_confirmation,
        sampled_gain_at_least_five_points=sampled["sampled_or_greedy_delta"] >= .05,
        sampled_paired_ci_excludes_zero=sampled["paired_ci95"][0] > 0,
        both_replicates_positive=all(
            item["sampled"]["sampled_or_greedy_delta"] > 0 for item in results["per_replicate"].values()),
        greedy_rules_out_five_point_regression=greedy["paired_ci95"][0] >= -.05,
        no_infrastructure_failures=True,
    )


def frozen_checkpoints(selection):
    """Include separately frozen, equally trained controls without duplicating the baseline."""
    paths, hashes = dict(selection["checkpoints"]), dict(selection["checkpoint_hashes"])
    if set(paths) != {"control", "seed0", "seed1"} or set(hashes) != set(paths):
        raise ValueError("Confirmation requires one initial control and both trained candidates")
    matched = selection.get("matched_training_control")
    if matched:
        expected = {"matched_control_seed0", "matched_control_seed1"}
        if set(matched["checkpoints"]) != expected or set(matched["checkpoint_hashes"]) != expected:
            raise ValueError("Both matched trained controls must be frozen")
        if matched["updates"] != selection["updates"]:
            raise ValueError("Matched controls require equal training update counts")
        paths.update(matched["checkpoints"])
        hashes.update(matched["checkpoint_hashes"])
    for name, path in paths.items():
        if sha256(path) != hashes[name]:
            raise ValueError(f"A frozen confirmation checkpoint changed: {name}")
    return paths, hashes


def compare_frozen_training_control(root):
    """Secondary confirmation contrast: the selected treatment versus equal extra PPO training."""
    selection = read_json(root / "candidate_selection.json")
    matched = selection["matched_training_control"]
    _, frozen = frozen_checkpoints(selection)
    seeds = read_json(root / "manifest.json")["confirmation_seeds"]
    candidates, controls, evidence = {}, {}, {}
    for seed in (0, 1):
        for recipe, group, key in (
            (selection["recipe"], candidates, f"seed{seed}"),
            (matched["recipe"], controls, f"matched_control_seed{seed}"),
        ):
            name = f"{recipe}_seed{seed}_u{selection['updates']}_confirmation400"
            group[seed], evidence[key] = verified_panel(root / "evaluations" / name, seeds,
                                                      selection["confirmation_device"])
            if evidence[key]["checkpoint_sha256"] != frozen[key]:
                raise ValueError("Matched-control confirmation differs from frozen checkpoints")
    results = compare_matched_training(candidates, controls)
    acceptance = acceptance_rules(results, independent_confirmation=True)
    return dict(
        stage="confirmation", recipe=selection["recipe"], control_recipe=matched["recipe"],
        updates=selection["updates"], results=results, evidence=evidence, acceptance=acceptance,
        validated=all(acceptance.values()), physical_games=4 * len(seeds) * 4,
        interpretation="Secondary prespecified contrast against the equally trained original-critic continuation. "
                       "A gain over the initial actor alone does not establish a benefit from the treatment. "
                       "Sampling repeats and both fixed training replicas are grouped by game seed.",
    )


def compare_interventions(root, base, updates, critic_suffix="complete"):
    if base not in ("gae100", "gae097"):
        raise ValueError("Interventions require the selected original-critic GAE base")
    seeds = read_json(root / "manifest.json")["development_seeds"]
    panels, evidence = {}, {}
    for recipe in (base, f"{base}_{critic_suffix}", f"{base}_lr3e5"):
        panels[recipe], evidence[recipe] = {}, {}
        for seed in (0, 1):
            name = f"{recipe}_seed{seed}_u{updates}_dev200"
            panels[recipe][seed], evidence[recipe][seed] = verified_panel(root / "evaluations" / name, seeds)
    return dict(
        stage="development", base=base, updates=updates, evidence=evidence,
        comparisons={recipe: compare_matched_training(panels[recipe], panels[base])
                     for recipe in (f"{base}_{critic_suffix}", f"{base}_lr3e5")},
        interpretation="Each intervention is paired with the original-critic continuation at the same update "
                       "and training seed. Average both training replicas and sampling repeats within game seed "
                       "before bootstrapping. These development comparisons are exploratory, not confirmation.",
    )


def compare_continuation(root, recipe, updates_from, updates_to):
    if not 0 < updates_from < updates_to:
        raise ValueError("Continuation requires increasing positive update counts")
    seeds = read_json(root / "manifest.json")["development_seeds"]
    panels, evidence = {}, {}
    for updates in (updates_from, updates_to):
        panels[updates], evidence[updates] = {}, {}
        for seed in (0, 1):
            path = root / "evaluations" / f"{recipe}_seed{seed}_u{updates}_dev200"
            panels[updates][seed], evidence[updates][seed] = verified_panel(path, seeds)
    return dict(
        recipe=recipe, updates_from=updates_from, updates_to=updates_to, stage="development",
        results=compare_matched_training(panels[updates_to], panels[updates_from]), evidence=evidence,
        interpretation="Exploratory paired continuation comparison, both training replicas and all three sampling "
                       "repeats grouped by game seed. Later checkpoints include declared process restarts; this is "
                       "not an uninterrupted-training comparison. No confirmation games used.",
    )


def verified_panel(path, seeds, device="cpu"):
    modes, evidence = {}, []
    checkpoint_hash = None
    for sampling in (0, 1, 2, None):
        mode = "greedy" if sampling is None else f"sample{sampling}"
        directory = path / mode
        metadata = read_json(directory / "manifest.json")
        expected = dict(status="complete", seeds=seeds, sampling_seed=sampling, greedy=sampling is None,
                        deck="b_blue", stake=1, win_ante=8, temperature=1.0, batch_size=32, workers=4,
                        device=device, reward_config=asdict(outcome_reward()))
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Mismatched evaluation configuration: {directory}")
        if metadata["agent_config"]["hand_ar_mixture_eps"] != 0.5:
            raise ValueError("Evaluation mixture changed")
        checkpoint_hash = checkpoint_hash or metadata["checkpoint_sha256"]
        if metadata["checkpoint_sha256"] != checkpoint_hash or sha256(metadata["checkpoint"]) != checkpoint_hash:
            raise ValueError("Evaluated checkpoint changed between modes or since evaluation")
        outcomes = directory / "outcomes.jsonl"
        rows = [json.loads(line) for line in outcomes.read_text().splitlines()]
        if len(rows) != len(seeds) or {row["seed"] for row in rows} != set(seeds):
            raise ValueError("Missing, duplicate or unexpected game seeds")
        if any(row.get("sampling_seed") != sampling or row.get("stalled") or row.get("censored") for row in rows):
            raise ValueError("Wrong sampling repeat or incomplete trajectory")
        if summarize(rows) != read_json(directory / "summary.json"):
            raise ValueError("Saved summary differs from raw outcomes")
        modes[sampling] = rows
        evidence.append(dict(path=str(directory), manifest_sha256=sha256(directory / "manifest.json"),
                             outcomes_sha256=sha256(outcomes), games=len(rows),
                             elapsed_seconds=metadata["elapsed_seconds"]))
    return dict(sampled=[row for sampling in (0, 1, 2) for row in modes[sampling]], greedy=modes[None]), dict(
        checkpoint=metadata["checkpoint"], checkpoint_sha256=checkpoint_hash, files=evidence,
    )


def compare_panel(root, recipe, updates, stage):
    protocol = read_json(root / "manifest.json")
    seeds = protocol[f"{stage}_seeds"]
    device = "cpu" if stage == "development" else read_json(root / "candidate_selection.json")["confirmation_device"]
    suffix = "dev200" if stage == "development" else "confirmation400"
    control, control_evidence = verified_panel(root / "evaluations" / f"baseline_{suffix}", seeds, device)
    candidates, evidence, physical = {}, {"control": control_evidence}, {}
    for seed in (0, 1):
        name = f"{recipe}_seed{seed}_u{updates}_{suffix}"
        candidates[seed], evidence[f"seed{seed}"] = verified_panel(root / "evaluations" / name, seeds, device)
        physical[f"seed{seed}"] = {mode: summarize(rows) for mode, rows in candidates[seed].items()}
    physical["control"] = {mode: summarize(rows) for mode, rows in control.items()}
    hashes = {name: item["checkpoint_sha256"] for name, item in evidence.items()}
    if stage == "confirmation":
        selection = read_json(root / "candidate_selection.json")
        if selection["recipe"] != recipe or selection["updates"] != updates or hashes != selection["checkpoint_hashes"]:
            raise ValueError("Confirmation differs from the frozen candidate selection")
    results = compare_replicates(candidates, control)
    sampled = results["sampled"]["paired"]
    acceptance = acceptance_rules(results, independent_confirmation=stage == "confirmation")
    return dict(
        stage=stage, recipe=recipe, updates=updates, game_seeds=seeds, device=device,
        results=results, physical_arms=physical,
        evidence=evidence, checkpoint_hashes=hashes, acceptance=acceptance, validated=all(acceptance.values()),
        substantial_step=all(acceptance.values()) and sampled["sampled_or_greedy_delta"] >= .10,
        physical_games=sum(len(rows) for rows in control.values())
                       + sum(len(rows) for candidate in candidates.values() for rows in candidate.values()),
        interpretation=("Game-seed bootstrap averages the three sampling repeats and two fixed training replicas "
                        "before resampling. The same control games are reused only for alignment; their physical "
                        "count is not doubled. Two replicas do not estimate training-seed variability precisely. "
                        "An improvement below majority-winning play remains an intermediate result."),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--recipe", choices=("gae100", "gae097", "gae100_lr3e5", "gae097_lr3e5",
                                             "gae100_terminal", "gae097_terminal",
                                             "gae100_terminal_lr3e5", "gae097_terminal_lr3e5",
                                             "gae100_complete", "gae097_complete"), required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--from-updates", type=int,
                        help="Compare this recipe's later development checkpoint with both earlier replicas")
    parser.add_argument("--stage", choices=("development", "confirmation"), default="development")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--selection-reason")
    parser.add_argument("--confirmation-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--matched-control-recipe", choices=("gae100", "gae097"),
                        help="Also freeze both equally trained original-critic controls for confirmation")
    parser.add_argument("--interventions", action="store_true",
                        help="Compare the critic and LR arms with their matched trained controls")
    parser.add_argument("--critic-suffix", choices=("complete", "terminal"), default="complete")
    args = parser.parse_args()
    if args.from_updates is not None:
        if args.stage != "development" or args.freeze or args.interventions or args.matched_control_recipe:
            parser.error("Continuation comparisons use development games and cannot freeze a candidate")
        result = compare_continuation(args.root, args.recipe, args.from_updates, args.updates)
        path = args.root / f"continuation_{args.recipe}_u{args.from_updates}_to{args.updates}.json"
        if path.exists():
            previous = read_json(path)
            # Preserve older diagnostic wording and artifact hashes while
            # requiring the measurements and their raw provenance to agree.
            for key in ("results", "evidence"):
                if previous[key] != json.loads(json.dumps(result[key])):
                    raise ValueError("Existing continuation evidence differs from the raw panels")
        else:
            atomic_json(path, result)
        print(json.dumps(result["results"], indent=2))
        return
    if args.interventions:
        if args.stage != "development" or args.freeze:
            parser.error("Intervention diagnostics use development games and cannot freeze a candidate")
        result = compare_interventions(args.root, args.recipe, args.updates, args.critic_suffix)
        atomic_json(args.root / f"intervention_comparison_u{args.updates}.json", result)
        print(json.dumps(result["comparisons"], indent=2))
        return
    if args.freeze and (args.stage != "development" or not args.selection_reason):
        parser.error("Freeze requires development results and an explicit selection reason")
    if args.matched_control_recipe and (
            not args.freeze or not args.recipe.startswith(args.matched_control_recipe + "_")):
        parser.error("Matched controls require freezing a declared intervention from that base recipe")
    result = compare_panel(args.root, args.recipe, args.updates, args.stage)
    path = args.root / f"{args.stage}_{args.recipe}_u{args.updates}_comparison.json"
    atomic_json(path, result)
    if args.freeze:
        selection_path = args.root / "candidate_selection.json"
        if selection_path.exists():
            raise ValueError("A candidate selection already exists; do not change it after confirmation begins")
        resource_evidence = None
        parallel_modes = 2
        if (args.root / RESOURCE_PLAN).exists():
            allocation, resource_evidence = cpu_confirmation_plan(args.root)
            if args.confirmation_device != allocation["confirmation_device"]:
                raise ValueError("Confirmation must use the prospectively declared CPU allocation")
            parallel_modes = allocation["parallel_modes"]
        selection = dict(
            resource_allocation=resource_evidence, confirmation_parallel_modes=parallel_modes,
            recipe=args.recipe, updates=args.updates, checkpoint_hashes=result["checkpoint_hashes"],
            checkpoints={key: item["checkpoint"] for key, item in result["evidence"].items()},
            development_comparison=str(path), development_comparison_sha256=sha256(path),
            reason=args.selection_reason, frozen_utc=datetime.now(UTC).isoformat(),
            representative_training_seed=0, baseline_shared_across_replicates=True,
            confirmation_device=args.confirmation_device,
            study_manifest_sha256=sha256(args.root / "manifest.json"),
        )
        if args.matched_control_recipe:
            control_result = compare_panel(args.root, args.matched_control_recipe, args.updates, "development")
            control_path = args.root / f"development_{args.matched_control_recipe}_u{args.updates}_comparison.json"
            atomic_json(control_path, control_result)
            selection["matched_training_control"] = dict(
                recipe=args.matched_control_recipe, updates=args.updates,
                checkpoints={f"matched_control_seed{seed}": control_result["evidence"][f"seed{seed}"]["checkpoint"]
                             for seed in (0, 1)},
                checkpoint_hashes={f"matched_control_seed{seed}": control_result["checkpoint_hashes"][f"seed{seed}"]
                                   for seed in (0, 1)},
                development_comparison=str(control_path), development_comparison_sha256=sha256(control_path),
                contrast="Secondary: isolate the selected treatment from equal additional PPO training",
            )
        frozen_checkpoints(selection)
        atomic_json(selection_path, selection)
    if args.stage == "confirmation" and read_json(
            args.root / "candidate_selection.json").get("matched_training_control"):
        matched_result = compare_frozen_training_control(args.root)
        atomic_json(args.root / "confirmation_matched_control_comparison.json", matched_result)
    print(json.dumps({key: result[key] for key in ("stage", "recipe", "updates", "results", "acceptance")}, indent=2))


if __name__ == "__main__":
    main()
