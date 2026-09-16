"""Audit stored forecast probabilities without model inference or training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[1]
    root = repo / "eval/complete_cohort_long_20260915/evaluations"
    inputs = []
    results = []
    for endpoint in (0, 20, 40, 60):
        forecasts = clipped = games = 0
        minimum = 1.0
        for replica in (0, 1):
            path = root / f"update{endpoint}" / f"seed{replica}" / "sampled" / "outcomes.jsonl"
            inputs.append({"path": str(path.relative_to(repo)), "sha256": sha256(path)})
            seeds = set()
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["seed"] in seeds:
                    raise ValueError(f"Duplicate seed in {path}: {row['seed']}")
                seeds.add(row["seed"])
                if row["stalled"] or row["censored"]:
                    raise ValueError(f"Incomplete outcome in {path}: {row['seed']}")
                if row["sampling_seed"] != 0 or len(row["critic_forecasts"]) != row["steps"]:
                    raise ValueError(f"Unexpected sampling repeat or missing forecast in {path}: {row['seed']}")
                target = 8 if row["won"] else int(row["final_ante"]) - 1
                if not 0 <= target <= 8:
                    raise ValueError(f"Invalid outcome: {target}")
                games += 1
                for forecast in row["critic_forecasts"]:
                    probability = forecast["outcome_probabilities"][target]
                    if not 0 <= probability <= 1:
                        raise ValueError(f"Invalid probability: {probability}")
                    forecasts += 1
                    clipped += probability <= 1e-7
                    minimum = min(minimum, probability)
            if seeds != set(range(190000, 190050)):
                raise ValueError(f"Unexpected seed panel in {path}")
        results.append(
            {
                "additional_updates": endpoint,
                "games": games,
                "forecasts": forecasts,
                "selected_probability_at_or_below_nll_floor": clipped,
                "minimum_selected_probability": minimum,
            }
        )
    source_paths = (
        Path(__file__).resolve(),
        repo / "src/pylatro_agent/value_head.py",
        repo / "src/pylatro_agent/survival.py",
    )
    result = {
        "date": "2026-09-16",
        "scope": "Stored sampled forecasts; no inference, training, or accelerator use.",
        "limitations": [
            "Does not audit training minibatches.",
            "Does not recover individual pre-sigmoid hazard saturation.",
            "Repeated forecasts within a game are correlated, not independent outcomes.",
        ],
        "nll_probability_floor": 1e-7,
        "game_seeds": list(range(190000, 190050)),
        "training_replicas": [0, 1],
        "sampling_seeds": [0],
        "results": results,
        "total_forecasts": sum(item["forecasts"] for item in results),
        "total_at_or_below_floor": sum(item["selected_probability_at_or_below_nll_floor"] for item in results),
        "inputs": inputs,
        "sources": [{"path": str(path.relative_to(repo)), "sha256": sha256(path)} for path in source_paths],
    }
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("results", "total_forecasts", "total_at_or_below_floor")}, indent=2))


if __name__ == "__main__":
    main()
