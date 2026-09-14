#!/usr/bin/env python3
"""Standalone figures from completed outcome-study measurements (no model execution)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.outcome_study import ROOT
from pylatro_agent.transfer import sha256


def point_interval(axis, row, point, bounds, *, color):
    axis.hlines(row, 100 * bounds[0], 100 * bounds[1], color=color, linewidth=1.5)
    axis.vlines(100 * np.asarray(bounds), row - .065, row + .065, color=color, linewidth=1)
    axis.plot(100 * point, row, "o", color=color, markersize=7)


def diagnostics(root, pyplot):
    phase_path = root / "phase_oracle_cpu/results.json"
    control_path = root / "label_selection_cpu_control/manifest.json"
    phase, control = read_json(phase_path), read_json(control_path)
    if control["status"] != "complete":
        raise ValueError("The label-selection control must finish before plotting")
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(13.5, 4.7), layout="constrained")
    labels = ["Starting student", "Teacher: hand decisions", "Teacher: shop decisions",
              "Teacher: hands and shops", "Teacher: all phases"]
    for row, name in enumerate(("student", "hand", "shop", "hand_shop", "teacher_all")):
        value = phase["summaries"][name]
        point_interval(left, row, value["win_rate"], value["win_rate_cluster_ci95"],
                       color="#60656c" if name == "student" else "#235789")
        left.text(98, row, f"{value['wins']}/{value['games']}", ha="right", va="center", fontsize=9)
    left.set_yticks(range(5), labels)
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.axvline(50, color="#c9cdd2", linestyle=":", zorder=0)
    left.set_xlabel("Observed wins (%) with 95% game-seed intervals")
    left.set_title("Both decision phases leave substantial headroom", loc="left", fontsize=12, pad=23)
    left.text(.98, 1.025, "wins / games", transform=left.transAxes, ha="right", fontsize=9, color="#60656c")

    actual = None
    for seed, offset, color, marker in ((0, -.065, "#235789", "o"), (1, .065, "#b06e1f", "s")):
        forecasts = []
        for arm in ("full", "completion", "random"):
            rows = control["results"][str(seed)]["arms"][arm]["curve"]
            values = next(row["validation"] for row in rows if row["update"] == 100)
            forecasts.append(100 * values["start_predicted_win_rate"])
            observed = values["start_observed_win_rate"]
            if actual is not None and observed != actual:
                raise ValueError("Critic controls must share the same validation outcomes")
            actual = observed
        right.scatter(np.arange(3) + offset, forecasts, color=color, marker=marker, s=55,
                      label=f"Training seed {seed}", zorder=3)
    right.axhline(100 * actual, color="#60656c", linestyle="--", linewidth=1,
                  label=f"Observed wins: {100 * actual:.1f}% (21/128)")
    right.set_xticks(range(3), ["Full labels", "Completion\nselected", "Random mask\n(same count)"])
    right.set_xlim(-.45, 2.45)
    right.set_ylim(0, 30)
    right.set_ylabel("Mean initial win forecast (%)")
    right.set_title("Selective labels make the critic too pessimistic", loc="left", fontsize=12, pad=23)
    right.legend(frameon=False, fontsize=9, loc="upper right")
    figure.suptitle("Outcome-only PPO: diagnostic evidence", fontsize=15, fontweight="bold")
    figure.text(.02, -.04,
                "Left: starting actor on 50 seeds; three repeats (full teacher: one). "
                "Teacher has full simulator access; replacements never train PPO.\n"
                "Right: frozen representations and matched critic fitting, after 100 further updates. "
                "This isolates calibration effects; gameplay is measured separately.", fontsize=9)
    return figure, [phase_path, control_path]


def confirmation(root, pyplot):
    selection_path = root / "candidate_selection.json"
    selection = read_json(selection_path)
    result_path = root / f"confirmation_{selection['recipe']}_u{selection['updates']}_comparison.json"
    result = read_json(result_path)
    if result["checkpoint_hashes"] != selection["checkpoint_hashes"] or result["stage"] != "confirmation":
        raise ValueError("Confirmation differs from the frozen model selection")
    sources = [selection_path, result_path]
    sampled = result["results"]["sampled"]
    absolute = [("Initial policy", sampled["control"], "#60656c"),
                ("Selected PPO (two seeds)", sampled["candidate"], "#235789")]
    differences = [("PPO mean vs initial", sampled["paired"], "#235789")]
    for seed in (0, 1):
        differences.append((f"PPO seed {seed} vs initial",
                            result["results"]["per_replicate"][str(seed)]["sampled"], "#5682aa"))
    if selection.get("matched_training_control"):
        path = root / "confirmation_matched_control_comparison.json"
        matched = read_json(path)
        sources.append(path)
        absolute.append(("Matched PPO control (two seeds)", matched["results"]["sampled"]["control"], "#b06e1f"))
        differences.append(("PPO mean vs matched control", matched["results"]["sampled"]["paired"], "#b06e1f"))
    teacher_path = root / "teacher_confirmation400/comparison.json"
    if teacher_path.exists():
        teacher = read_json(teacher_path)
        sources.append(teacher_path)
        absolute.append(("V68 teacher (full simulator state)", teacher["teacher"], "#80589c"))
    figure, (left, right) = pyplot.subplots(1, 2, figsize=(14, 4.3), layout="constrained")
    for row, (_, value, color) in enumerate(absolute):
        point_interval(left, row, value["win_rate"], value["win_rate_cluster_ci95"], color=color)
        left.text(98, row, f"{value['wins']}/{value['games']}", ha="right", va="center", fontsize=9)
    left.set_yticks(range(len(absolute)), [item[0] for item in absolute])
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.axvline(50, color="#c9cdd2", linestyle=":", zorder=0)
    left.set_xlabel("Sampled wins (%)")
    left.set_title("Absolute strength", loc="left", fontsize=12, pad=18)
    left.text(.98, 1.025, "wins / physical games", transform=left.transAxes, ha="right", fontsize=9)
    for row, (_, value, color) in enumerate(differences):
        point_interval(right, row, value["sampled_or_greedy_delta"], value["paired_ci95"], color=color)
    right.set_yticks(range(len(differences)), [item[0] for item in differences])
    right.invert_yaxis()
    right.axvline(0, color="#60656c", linewidth=1, zorder=0)
    right.axvline(5, color="#c9cdd2", linestyle=":", zorder=0)
    right.set_xlabel("Paired change (percentage points)")
    right.set_title("Changes with 95% game-seed intervals", loc="left", fontsize=12, pad=18)
    figure.suptitle(f"Sampled confirmation: {selection['recipe']}, update {selection['updates']}",
                   fontsize=15, fontweight="bold")
    figure.text(.02, -.045,
                f"400 fresh seeds, three sampling repeats, two fixed training replicas; "
                f"all arms use {result['device'].upper()}. Average repeats within game seed before bootstrapping.\n"
                "The initial control's physical count is not doubled. Full acceptance also requires the "
                "greedy guardrail and no infrastructure failures; see the paired report.\n"
                "The teacher has privileged simulator access. Intervals condition on the fixed trained replicas.",
                fontsize=9)
    return figure, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--kind", choices=("diagnostics", "confirmation"), default="diagnostics")
    parser.add_argument("--output-directory", type=Path, default=Path("docs/transfer/assets"))
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(args.root.resolve() / "plotting_cache"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    pyplot.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                            "axes.spines.top": False, "axes.spines.right": False,
                            "svg.hashsalt": "outcome_ppo_20260913"})
    figure, sources = (diagnostics if args.kind == "diagnostics" else confirmation)(args.root, pyplot)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    files = []
    for suffix in ("png", "svg"):
        path = args.output_directory / f"outcome_{args.kind}.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white",
                       **({"metadata": {"Date": None}} if suffix == "svg" else {}))
        files.append(path)
    pyplot.close(figure)
    atomic_json(args.root / f"figure_{args.kind}.json", dict(
        sources={str(path): sha256(path) for path in sources},
        figures={str(path): sha256(path) for path in files},
        script_sha256=sha256(__file__), matplotlib_version=matplotlib.__version__,
        model_execution=False, accelerator_seconds=0,
    ))
    print("\n".join(map(str, files)))


if __name__ == "__main__":
    main()
