#!/usr/bin/env python3
"""Render verified outcome-study comparisons as a reviewable Markdown artifact."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from pylatro_agent.learning_bottleneck import read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256


def percentage(value):
    return f"{100 * value:.2f}%"


def paired(value):
    low, high = value["paired_ci95"]
    return f"{100 * value['sampled_or_greedy_delta']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]"


def render(root, output):
    lines = []

    def paragraph(text):
        lines.extend([text, ""])

    def link(path, label):
        return f"[{label}]({os.path.relpath(path, output.parent)})"

    def table(columns, rows):
        lines.extend(["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
        lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        lines.append("")

    selection_path = root / "candidate_selection.json"
    selection = read_json(selection_path) if selection_path.exists() else None
    confirmation = None
    if selection:
        path = root / f"confirmation_{selection['recipe']}_u{selection['updates']}_comparison.json"
        if path.exists():
            confirmation = read_json(path)
    verification_path = root / "verification_final.json"
    verified = verification_path.exists() and read_json(verification_path)["passed"]
    if confirmation:
        results = confirmation["results"]
        delta = results["sampled"]["paired"]
        if confirmation["validated"]:
            verdict = "validated improvement"
        elif delta["paired_ci95"][1] <= 0:
            verdict = "negative result"
        elif delta["paired_ci95"][0] > 0:
            verdict = "promising result that does not pass all acceptance criteria"
        else:
            verdict = "inconclusive result"
        paragraph(f"The selected outcome-only PPO recipe `{selection['recipe']}` at update {selection['updates']} "
                  f"has a **{verdict}**. Its two replicas average "
                  f"{percentage(results['sampled']['candidate']['win_rate'])} sampled wins on 400 confirmation seeds, "
                  f"versus {percentage(results['sampled']['control']['win_rate'])} for the initial actor. "
                  f"The paired change is {paired(delta)} percentage points (95% interval).")
        if results["sampled"]["candidate"]["win_rate"] < .5:
            paragraph("The selected model still loses most sampled games. This is an intermediate result, "
                      "not evidence that the task is solved or that its current win rate is a ceiling.")
        if not verified:
            paragraph("**Final artifact and compute verification is still pending.**")
        figure = Path("docs/transfer/assets/outcome_confirmation.png")
        if figure.exists():
            paragraph("!" + link(figure, "Sampled confirmation rates and paired confidence intervals"))
    else:
        paragraph("**Study in progress. No candidate has completed independent confirmation.** "
                  "The tables below contain completed measurements only; development gains are exploratory.")

    ledger = read_json(root / "compute.json")
    live = sum(time.time() - job["started_unix"] for job in ledger["jobs"] if job["status"] == "running")
    paragraph(f"The study has charged {(ledger['elapsed_seconds'] + live) / 3600:.3f} of the "
              f"{ledger['limit_seconds'] / 3600:.0f} additional MPS hours. "
              "The ledger includes failed jobs and CPU work inside an exclusive accelerator job. "
              "Separate CPU diagnostics and evaluations do not consume that allowance. "
              + link(root / "compute.json", "Compute ledger") + ".")
    resource_plan = root / "resource_allocation_v2.json"
    if resource_plan.exists():
        paragraph("Final confirmation is allocated to CPU for every arm. This releases MPS time for explicitly "
                  "declared matched continuations while retaining five accelerator minutes for backend checks. "
                  "The 12-hour cap and full confirmation panel remain fixed. "
                  + link(resource_plan, "Prospective resource allocation") + ".")
    paragraph("PPO uses ten points for an Ante-8 win and zero otherwise, with discount one. "
              "It receives no teacher labels, teacher action replacements, imitation loss, self-imitation, "
              "or strategic reward bonuses. Existing pretrained weights, candidates, risk inputs and automatic "
              "Joker ordering remain part of the reference interface. "
              + link(Path("docs/transfer/outcome_ppo.md"), "Methods, decision history and limitations") + ".")

    paragraph("Established mechanisms and the limits of the evidence:")
    critic_effect = "Gameplay benefit requires the matched PPO contrast."
    intervention_path = root / "intervention_comparison_u30.json"
    if intervention_path.exists():
        intervention = read_json(intervention_path)
        critic = intervention["comparisons"][intervention["base"] + "_complete"]
        critic_effect = ("The complete-critic bundle changes sampled wins by "
                         + paired(critic["sampled"]["paired"]) + " points and greedy wins by "
                         + paired(critic["greedy"]["paired"]) + " points in matched development play. "
                         "The bundle does not isolate its individual components.")
    table(["Finding", "Evidence", "What it establishes"], [
        ["Completion-selected critic labels",
         link(root / "labels_gae100_seed0_u20.json", "Exact rollout reconstruction")
         + "; " + link(root / "label_selection_cpu_control/manifest.json", "two-replica CPU control"),
         "Winning initial states receive fewer labels. Holding representations and optimizer ordering fixed, "
         "this selection makes initial forecasts too pessimistic. " + critic_effect],
        ["Critic calibration at transfer", link(root / "critic_seed0/manifest.json", "Complete student trajectories"),
         "The original head forecasts about 83% wins against 16% observed on critic-validation games. "
         "Private-head fitting corrects calibration; both PPO pilots use it, so its gameplay effect is not isolated."],
        ["Return-loss objective mismatch", link(root / "return_loss_audit.json", "Production Huber-loss probe"),
         "For binary 0/10 returns with 20% wins, the Huber optimum is .25 rather than the mean 2.0. "
         "Live residuals are smaller; this alone does not establish a large gameplay effect."],
        ["Remaining hand and shop errors", link(root / "phase_oracle_cpu/results.json", "Matched phase interventions"),
         "Replacing either phase helps; replacing both reaches 69.3% on 50 diagnostic seeds. "
         "The teacher has full simulator access, so this does not prove current student inputs are sufficient."],
        ["Shared critic interference", link(root / "gradient_audit_initial/result.json", "Frozen gradient audit"),
         "Weighted critic/actor backbone gradient ratio .145, cosine -.070 on the probe. "
         "This does not support a claim that the critic overwhelms the shared encoder."],
        ["Long-term credit", link(root / "credit_audit.json", "Complete-game trace audit"),
         "Lambda .97 and 1 disagree in sign on 27.7% of decisions. Actual PPO bootstraps fragments; "
         "a realized outcome does not establish whether each preceding action was good."],
        ["Mixture is a reference distribution",
         link(root / "mixture_capacity_audit.json", "Production normalizer probe"),
         "Learned hand scores reweight the mixture before final normalization. An action absent from candidates "
         "can rise from .1 to .9 probability at mixture .5. The distribution layer has no fixed .5 cap; "
         "learned representations and finite exploration remain separate questions."],
        ["Checkpoint and reservation defects", link(root / "reservation_before.log", "Failing reservation regression")
         + "; " + link(root / "mps_rng_hardware/result.json", "MPS hardware reproducer"),
         "Complete seed reservations and checkpoint configuration were fixed before this PPO study. "
         "The separate MPS RNG fix reproduces all 512 test draws exactly, versus 31 matches before. "
         "This establishes RNG persistence, with no claim of a win-rate gain."],
    ])
    diagnostic_figure = Path("docs/transfer/assets/outcome_diagnostics.png")
    if diagnostic_figure.exists():
        paragraph("!" + link(diagnostic_figure, "Phase-policy headroom and controlled critic-label bias"))

    paragraph("Development uses CPU, 200 game seeds, three sampling repeats and two training replicas. "
              "Intervals resample game seeds after averaging repeats and replicas. "
              "The initial control has 600 physical sampled games and is not counted twice.")
    rows = []
    for path in sorted(root.glob("development_*_u*_comparison.json")):
        data = read_json(path)
        results = data["results"]
        sample, greedy = results["sampled"], results["greedy"]
        rows.append([link(path, data["recipe"]), data["updates"],
                     f"{sample['candidate']['wins']}/{sample['candidate']['games']} "
                     f"({percentage(sample['candidate']['win_rate'])})", paired(sample["paired"]),
                     percentage(greedy["candidate"]["win_rate"]), paired(greedy["paired"])])
    table(["Recipe", "Updates", "Sampled wins", "Sampled change [95% CI], pp", "Greedy wins",
           "Greedy change [95% CI], pp"], rows)
    paragraph("The seed-zero lambda-one pilot alone has a two-update runtime-probe restart. "
              "Its initial GAE comparison is therefore not a pure trace-length intervention. "
              "Seed-one initial legs are uninterrupted. Later treatment arms share their replica's restart boundary. "
              "The original checkpoints omit MPS RNG state, so their stream restarts cannot be undone retroactively.")
    gae_contrast = root / "initial_gae_contrast.json"
    if gae_contrast.exists():
        item = read_json(gae_contrast)["results"]
        paragraph("The direct replicated lambda-.97 versus lambda-one sampled difference is "
                  + paired(item["sampled"]["paired"]) + " points. This does not establish a superior GAE setting. "
                  + link(gae_contrast, "Matched initial recipe comparison") + ".")

    continuation = []
    for path in sorted(root.glob("continuation_gae*_u*_to*.json")):
        data = read_json(path)
        if "updates_from" not in data:
            continue
        item = data["results"]
        continuation.append([
            link(path, data["recipe"]), f"{data['updates_from']} to {data['updates_to']}",
            paired(item["sampled"]["paired"]), paired(item["greedy"]["paired"]),
            ", ".join(f"{seed}: {100 * replica['sampled']['sampled_or_greedy_delta']:+.2f}"
                      for seed, replica in item["per_replicate"].items()),
        ])
    if continuation:
        paragraph("Additional PPO is compared with the same recipe's earlier checkpoints on the same development "
                  "games. Later checkpoints include the declared process restarts. These paired changes describe "
                  "learning within the measured budget; they do not establish an asymptotic plateau.")
        table(["Recipe", "Updates", "Sampled change [95% CI], pp", "Greedy change [95% CI], pp",
               "Sampled changes by training seed, pp"], continuation)

    for path in sorted(root.glob("intervention_comparison_u*.json")):
        data = read_json(path)
        paragraph(f"The update-{data['updates']} interventions are paired against equally trained `{data['base']}` "
                  "controls. These are development diagnostics.")
        table(["Intervention", "Sampled change [95% CI], pp", "Greedy change [95% CI], pp",
               "Sampled changes by training seed, pp"], [
            [recipe, paired(item["sampled"]["paired"]), paired(item["greedy"]["paired"]),
             ", ".join(f"{seed}: {100 * replica['sampled']['sampled_or_greedy_delta']:+.2f}"
                       for seed, replica in item["per_replicate"].items())]
            for recipe, item in data["comparisons"].items()
        ])

    calibration = []
    for path in sorted(root.glob("labels_gae*_u*.json")):
        audit = read_json(path)
        forecast = {name: item for name, item in audit["forecast_panels"].items() if name != "baseline_dev200"}
        for name, item in forecast.items():
            calibration.append([link(path, name), percentage(item["mean_start_win_forecast"]),
                                percentage(item["observed_win_rate"]), f"{item['start_win_brier']:.4f}",
                                str(audit.get("outcome_labels_used_in_ppo_loss", True))])
    if calibration:
        paragraph("Critic forecasts on student-played development games are descriptive. Each policy has its own "
                  "realized outcomes; lower calibration error alone does not demonstrate better gameplay. "
                  "The final column shows whether rollout outcome labels actually train the critic.")
        table(["Checkpoint", "Initial win forecast", "Observed sampled wins", "Start Brier loss",
               "Uses rollout outcome labels"], calibration)

    phase_path = root / "phase_oracle_cpu/results.json"
    if phase_path.exists():
        phase = read_json(phase_path)
        paragraph("Teacher phase replacements are a separate CPU diagnosis on 50 reserved seeds. "
                  "They never supply PPO labels or select the candidate. "
                  "Student and partial-replacement arms have three repeats; the full teacher is deterministic.")
        table(["Diagnostic arm", "Wins/games", "Win rate", "Mean final ante"], [
            [name, f"{item['wins']}/{item['games']}", percentage(item["win_rate"]), f"{item['mean_final_ante']:.3f}"]
            for name, item in phase["summaries"].items()
        ])

    if confirmation:
        paragraph(f"Independent confirmation uses {confirmation['device'].upper()} for every arm, "
                  "batch 32 and four engine workers. All 400 seeds and all four modes are evaluated:")
        rows = []
        for name, arm in confirmation["physical_arms"].items():
            for mode, item in arm.items():
                interval = item["win_rate_cluster_ci95"]
                rows.append([name, mode, f"{item['wins']}/{item['games']}", percentage(item["win_rate"]),
                             f"[{percentage(interval[0])}, {percentage(interval[1])}]",
                             f"{item['mean_final_ante']:.3f}", item["stalls"], item["censored"]])
        table(["Arm", "Mode", "Wins/games", "Win rate", "95% seed interval", "Mean final ante", "Stalls",
               "Censored"], rows)
        paragraph("Final ante preserves the simulator's terminal convention: a win can advance the state to ante nine. "
                  "Ante-clear rates are reported separately.")
        table(["Arm/mode", *[f"Clear {ante}" for ante in range(1, 9)]], [
            [f"{name}/{mode}", *[percentage(item["ante_clear_rates"][str(ante)]) for ante in range(1, 9)]]
            for name, arm in confirmation["physical_arms"].items() for mode, item in arm.items()
        ])
        table(["Acceptance condition against initial actor", "Passed"], [
            [key.replace("_", " "), str(value)] for key, value in confirmation["acceptance"].items()
        ])
        matched_path = root / "confirmation_matched_control_comparison.json"
        if matched_path.exists():
            data = read_json(matched_path)
            paragraph("The secondary confirmation contrast tests the selected treatment against equal additional PPO "
                      f"training (`{data['control_recipe']}`). "
                      f"It {'passes' if data['validated'] else 'does not pass'} all improvement criteria. "
                      + link(matched_path, "Raw paired results and acceptance checks") + ".")
            table(["Mode", "Selected rate", "Trained-control rate", "Change [95% CI], pp"], [
                [mode, percentage(data["results"][mode]["candidate"]["win_rate"]),
                 percentage(data["results"][mode]["control"]["win_rate"]), paired(data["results"][mode]["paired"])]
                for mode in ("sampled", "greedy")
            ])
        teacher_path = root / "teacher_confirmation400/comparison.json"
        if teacher_path.exists():
            teacher = read_json(teacher_path)
            reference = teacher["teacher"]
            paragraph("After candidate confirmation, the separately evaluated V68 teacher wins "
                      f"{reference['wins']}/{reference['games']} games "
                      f"({percentage(reference['win_rate'])}) on the same 400 seeds. "
                      "Its search/scaler settings come from the original dataset's generation metadata. "
                      "The selected student's sampled difference from this teacher is "
                      + paired(teacher["paired"]) + " percentage points. The teacher receives full simulator "
                      "state and scoring probes; this measures a remaining policy gap without proving that "
                      "student inputs are sufficient. These games supply no PPO labels and do not change "
                      "candidate selection or acceptance. " + link(teacher_path, "Matched teacher benchmark") + ".")

    paragraph("Pilots use Blue Deck, White Stake, ante eight, v13 observations, FP32, mixture .5, temperature one, "
              "32 environments and 256 steps per rollout (8,192 transitions). PPO makes two passes with logical "
              "batch 640 and physical batch 320, clip .1, entropy coefficient .001, and Adam LR 1e-5 unless explicitly "
              "testing 3e-5. The complete-episode treatment uses a zero return residual, freezes critic parameters "
              "during actor updates, and applies 30 private outcome-tower replay updates of 256 rows instead of "
              "the reference's masked PPO outcome updates plus four replay updates. This is a declared bundle of "
              "value definition, label source, gradient scope and replay exposure.")
    rows = []
    for path in sorted(root.glob("trials/*/recipe.json")):
        recipe = read_json(path)
        legs = [read_json(item) for item in path.parent.glob("leg_*/manifest.json")]
        complete = [item for item in legs if item["status"] == "complete"]
        updates = sum(item["update_count"] - item["start_update"] for item in complete)
        config = recipe["ppo_config"]
        rows.append([path.parent.name, updates, updates * config["num_envs"] * config["rollout_length"],
                     f"{sum(item['elapsed_seconds'] for item in complete) / 3600:.3f}",
                     ", ".join(f"{item['start_update']}-{item['update_count']}" for item in
                               sorted(complete, key=lambda item: item["start_update"]))])
    paragraph("Physical training below counts only updates executed within each trial directory. "
              "Inherited parent updates are not charged again to branches. Complete episode rates during training "
              "are descriptive because the policy changes while collecting them.")
    table(["Trial", "Executed updates", "Collected transitions", "Leg wall hours", "Completed update intervals"], rows)
    paragraph("Regression tests cover deck propagation, strict configuration restoration, complete seed exclusion, "
              "pure-outcome rewards, legal policy paths, paired seed statistics, branch optimizer preservation, "
              "critic isolation, teacher-free PPO execution, confirmation freezing and source archives. "
              "The source archive cannot bypass active-study pinning: it is accepted only after a final audit of "
              "the still-pinned workspace, with unchanged compute and training artifacts. "
              + link(Path("tests/test_complete_episode_critic.py"), "Critic intervention tests") + "; "
              + link(Path("tests/test_outcome_confirmation.py"), "confirmation regression") + "; "
              + link(Path("tests/test_outcome_source_archive.py"), "source archive regression") + ".")
    backend_path = root / "subset_backend_verification.json"
    if backend_path.exists():
        backend = read_json(backend_path)
        paragraph(f"The selected checkpoint's CPU/MPS probe passes on {backend['observation_count']} observations "
                  f"and a separate {backend['gameplay_games']}-game sampled smoke. "
                  "The supplemental states cover play, discard and multi-Joker sales; supervised and PPO "
                  "likelihoods are also compared. These bounded checks establish implementation agreement, "
                  "not gameplay strength. "
                  + link(backend_path, "Backend probabilities, gradients and provenance") + ".")
    if verified:
        audit = read_json(verification_path)
        paragraph(f"The final study audit passes, including {audit['original_artifacts_verified']} preserved original "
                  "artifacts, completed training trajectories, learning curves, source hashes and compute accounting. "
                  + link(verification_path, "Final artifact audit") + ".")
    conclusion_path = root / "conclusion.json"
    if conclusion_path.exists():
        conclusion = read_json(conclusion_path)
        for path, digest in conclusion["evidence_sha256"].items():
            if sha256(path) != digest:
                raise ValueError("The final recommendation's supporting evidence changed")
        for text in conclusion["next_training_recommendation"]:
            paragraph(text)
    paragraph("Uncertainty remains about learning beyond this update budget, variability across additional training "
              "seeds, the separate contributions inside the complete-critic treatment, and information available to "
              "the teacher that the student cannot observe. Two training replicas and one confirmation seed panel "
              "do not settle these questions. No new observation features are inserted into v13.")
    paragraph("Recreate the numeric tables with `.venv/bin/python tools/render_outcome_report.py`. "
              "The strict evaluation and paired-report commands, frozen checkpoint hashes, immutable game rows, "
              "TensorBoard curves and exact reference source are retained under " + link(root, str(root)) + ". "
              "The source archive contains compiled simulator artifacts as well as Python code; environment and "
              "replay resets across PPO legs remain explicit limitations to whole-process exact replay.")
    portable = Path("docs/transfer/data/outcome_confirmation_manifest.json")
    if portable.exists():
        paragraph("The committed " + link(portable.parent / "outcome_confirmation_games.jsonl", "portable game rows")
                  + " and " + link(portable, "configuration, hashes and results")
                  + " also reproduce every confirmation and teacher statistic without the original checkpoints "
                  "or full logs: `.venv/bin/python tools/export_outcome_confirmation.py --verify-only`. "
                  "Each physical initial-control and teacher game appears once.")
    paragraph("Standalone PNG and SVG figures use `.venv/bin/python tools/plot_outcome_study.py "
              "--kind diagnostics` or `--kind confirmation`. "
              "Runtime and plotting package versions are recorded in "
              + link(root / "environment.json", "environment.json") + " and "
              + link(root / "plotting_environment.json", "plotting_environment.json") + ".")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("docs/transfer/outcome_ppo_results.md"))
    args = parser.parse_args()
    render(args.root, args.output)
    print(args.output)
