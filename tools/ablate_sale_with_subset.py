#!/usr/bin/env python3
"""One-replicate development ablation: trained hand head with and without sale residual."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from evaluate_action_heads import run_panel

from pylatro_agent.learning_bottleneck import atomic_json, read_json
from pylatro_agent.transfer import paired_statistics, sha256, summarize


def existing_panel(folder):
    results = []
    for modes in (("sample0", "sample1", "sample2"), ("greedy",)):
        rows = []
        for mode in modes:
            path = folder / mode
            if read_json(path / "manifest.json")["status"] != "complete":
                raise ValueError("Matched comparison panel is incomplete")
            rows.extend(json.loads(line) for line in (path / "outcomes.jsonl").read_text().splitlines())
        summarize(rows)
        results.append(rows)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    args = parser.parse_args()
    root = args.root
    if not (root / "mps_development_combined_20_comparisons.json").exists():
        raise ValueError("Complete the combined screen before this controlled ablation")
    torch.set_num_threads(4)
    seeds = read_json(root / "manifest.json")["development_seeds"][:20]
    hand_checkpoint = root / "subset_residual_v1_seed0" / "checkpoint.pt"
    combined_checkpoint = root / "entity_sale_subset_residual_v1_seed0" / "checkpoint.pt"
    hand_manifest, combined_manifest = (
        read_json(p.parent / "complete.json") for p in (hand_checkpoint, combined_checkpoint)
    )
    if hand_manifest["subset_adapter_sha256"] != combined_manifest["subset_adapter_sha256"]:
        raise ValueError("Sale ablation does not share the same trained hand adapter")
    original, combined = (
        existing_panel(root / "evaluations" / f"mps_development_combined_20_{name}") for name in ("control", "seed0")
    )
    hand = run_panel(root, "mps_development_subset_ablation_seed0_20", hand_checkpoint, seeds, torch.device("mps"))
    report = dict(
        scope="Exploratory 20-seed, one-training-replicate ablation; not confirmation",
        subset_checkpoint_sha256=sha256(hand_checkpoint),
        combined_checkpoint_sha256=sha256(combined_checkpoint),
        hand_adapter_sha256=hand_manifest["subset_adapter_sha256"],
        subset_versus_neutral={
            mode: paired_statistics(hand[i], original[i]) for i, mode in enumerate(("sampled", "greedy"))
        },
        added_sale_residual={
            mode: paired_statistics(combined[i], hand[i]) for i, mode in enumerate(("sampled", "greedy"))
        },
        subset_outcomes={mode: summarize(hand[i]) for i, mode in enumerate(("sampled", "greedy"))},
        interpretation=(
            "The hand adapter, encoder, original heads, temperature and sampler are held fixed in the sale ablation"
        ),
    )
    atomic_json(root / "subset_sale_ablation.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k in ("subset_versus_neutral", "added_sale_residual")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
