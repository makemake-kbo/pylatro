#!/usr/bin/env python3
"""Controlled checkpoint/deck/mixture development screen; no PPO, no confirmation access."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from pylatro import load_game_data
from pylatro_agent.transfer import (
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    SAMPLING_SEEDS,
    check_seed_separation,
    evaluate_artifact,
    load_policy,
    sha256,
    source_hashes,
    summarize,
    write_json,
)
from pylatro_agent.vocab import build_vocab


def load_rows(path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete experiment: {path}")
    return [json.loads(line) for line in (path / "outcomes.jsonl").read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-hours", type=float, default=3.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    data = load_game_data()
    vocab = build_vocab(data)
    device = torch.device(args.device)
    start = time.monotonic()
    manifests = [
        args.dataset / "metadata/generation.json",
        args.dataset / "metadata/games.jsonl",
        Path("docs/heuristic/pretraining_v68_seeds.json"),
    ]
    separation = check_seed_separation(DEVELOPMENT_SEEDS + CONFIRMATION_SEEDS, manifests)
    source = source_hashes()
    data_hashes = {str(p): sha256(p) for p in sorted(args.dataset.rglob("*")) if p.is_file()}
    ledger = []

    def evaluate(checkpoint, seeds, mode, sample, label, *, deck="b_blue", mixture=0.5, kind="fresh_start"):
        if time.monotonic() - start > args.max_hours * 3600 - 240:
            raise TimeoutError("Diagnostic compute allocation exhausted; confirmation reserve preserved")
        model, reward, _ = load_policy(checkpoint, vocab, device, mixture=mixture)
        began = time.monotonic()
        rows = evaluate_artifact(
            model,
            reward,
            data,
            vocab,
            seeds,
            device,
            args.output_dir / label,
            greedy=mode == "greedy",
            sampling_seed=sample,
            deck=deck,
            kind=kind,
            metadata={
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "source_hashes": source,
                "data_hashes": data_hashes,
                "seed_separation": separation,
                "explicit_overrides": {"hand_ar_mixture_eps": mixture, "deck": deck},
            },
        )
        ledger.append({"label": label, "elapsed_seconds": time.monotonic() - began})
        write_json(
            args.output_dir / "compute.json",
            {"stage": "diagnostics", "runs": ledger, "total_seconds": time.monotonic() - start},
        )
        print(label, summarize(rows), flush=True)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()
        return rows

    screening = []
    for epoch in range(1, 11):
        checkpoint = args.checkpoint_dir / f"supervised_epoch{epoch}.pt"
        sampled = []
        for sample in SAMPLING_SEEDS:
            sampled.extend(
                evaluate(checkpoint, DEVELOPMENT_SEEDS[:50], "sampled", sample, f"epoch{epoch}_sample{sample}_dev50")
            )
        greedy = evaluate(checkpoint, DEVELOPMENT_SEEDS[:50], "greedy", 0, f"epoch{epoch}_greedy_dev50")
        screening.append({"epoch": epoch, "sampled": summarize(sampled), "greedy": summarize(greedy)})
        write_json(args.output_dir / "screening.json", screening)
    strongest = sorted(
        screening, key=lambda r: (r["sampled"]["win_rate"], r["sampled"]["mean_final_ante"]), reverse=True
    )[:2]
    expanded = []
    for row in strongest:
        epoch = row["epoch"]
        checkpoint = args.checkpoint_dir / f"supervised_epoch{epoch}.pt"
        sampled = []
        for sample in SAMPLING_SEEDS:
            sampled.extend(
                evaluate(checkpoint, DEVELOPMENT_SEEDS, "sampled", sample, f"epoch{epoch}_sample{sample}_dev200")
            )
        greedy = evaluate(checkpoint, DEVELOPMENT_SEEDS, "greedy", 0, f"epoch{epoch}_greedy_dev200")
        expanded.append({"epoch": epoch, "sampled": summarize(sampled), "greedy": summarize(greedy)})
        write_json(args.output_dir / "expanded.json", expanded)
    selected = max(expanded, key=lambda r: (r["sampled"]["win_rate"], r["sampled"]["mean_final_ante"]))
    write_json(
        args.output_dir / "selection.json",
        {
            "epoch": selected["epoch"],
            "development_result": selected,
            "checkpoint": str(args.checkpoint_dir / f"supervised_epoch{selected['epoch']}.pt"),
            "rule": "highest sampled win rate, then mean final ante on full development panel",
        },
    )
    factorial = []
    checkpoint = args.checkpoint_dir / "supervised_epoch10.pt"
    for deck in ("b_red", "b_blue"):
        for mixture in (0.1, 0.5):
            sampled = []
            for sample in SAMPLING_SEEDS:
                sampled.extend(
                    evaluate(
                        checkpoint,
                        DEVELOPMENT_SEEDS[:100],
                        "sampled",
                        sample,
                        f"factorial_{deck}_m{mixture}_sample{sample}",
                        deck=deck,
                        mixture=mixture,
                    )
                )
            greedy = evaluate(
                checkpoint,
                DEVELOPMENT_SEEDS[:100],
                "greedy",
                0,
                f"factorial_{deck}_m{mixture}_greedy",
                deck=deck,
                mixture=mixture,
            )
            factorial.append(
                {"deck": deck, "mixture": mixture, "sampled": summarize(sampled), "greedy": summarize(greedy)}
            )
            write_json(args.output_dir / "factorial.json", factorial)
    manifest = json.loads((args.output_dir.parent / "diagnostics_v2/manifest.json").read_text())
    for sample in SAMPLING_SEEDS:
        evaluate(
            checkpoint,
            manifest["offline_seeds"],
            "sampled",
            sample,
            f"epoch10_sample{sample}_demonstration200",
            kind="demonstration_seed",
        )
    evaluate(
        checkpoint, manifest["offline_seeds"], "greedy", 0, "epoch10_greedy_demonstration200", kind="demonstration_seed"
    )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
