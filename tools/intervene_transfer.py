#!/usr/bin/env python3
"""One-action interventions at uniformly sampled student disagreements, with paired continuation RNG."""

import argparse
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from pylatro import load_game_data
from pylatro_agent.action import ActionType, decode_action
from pylatro_agent.hand_candidates import generate_hand_candidates
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.joker_features import JOKER_EXTRA_FIELDS
from pylatro_agent.subset_actions import subset_index, subset_indices
from pylatro_agent.tokenizer import sign_log
from pylatro_agent.training.ppo_observations import _single_obs_to_batch
from pylatro_agent.training.ppo_policy import _grammar_distribution
from pylatro_agent.transfer import load_policy, source_hashes, write_json
from pylatro_agent.transfer_diagnostics import clone_diagnostic_env, live_state_digest
from pylatro_agent.vocab import build_vocab


def continuation(model, env, first_action, device, seed):
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    start_ante = env.state.round_resets.ante
    start_blind = env.state.blind_on_deck
    blind_survived = False
    with torch.inference_mode():
        action = first_action
        steps = 0
        while True:
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            blind_survived |= (
                bool(info["won"])
                or info["ante"] > start_ante
                or (env.state.blind_on_deck != start_blind and not terminated and not truncated)
            )
            if terminated or truncated:
                return {
                    "won": bool(info["won"]),
                    "max_ante": info["ante"],
                    "blind_survived": blind_survived,
                    "stalled": bool(info.get("stalled")),
                    "steps": steps,
                }
            dist, _ = _grammar_distribution(model, _single_obs_to_batch(obs, device))
            action = dist.sample().item()


def audit_inputs(state, probes):
    """Describe observed lossy encodings; this alone makes no causal claim."""
    candidates = [p for p in probes if p["structural_estimate"] is not None]
    for probe in candidates:
        probe["encoded_structural_score"] = int(min(max(sign_log(probe["structural_estimate"]) * 10, -255), 255))
    collisions = sum(
        a["encoded_structural_score"] == b["encoded_structural_score"] and a["exact_score"] != b["exact_score"]
        for i, a in enumerate(candidates)
        for b in candidates[i + 1 :]
    )
    cards = list(state.hand_cards) + list(state.draw_pile) + list(state.discard_pile)
    return {
        "candidate_score_token_collisions_with_distinct_exact_scores": collisions,
        "structural_and_exact_argmax_differ": (
            max(candidates, key=lambda p: p["structural_estimate"])["action"]
            != max(candidates, key=lambda p: p["exact_score"])["action"]
        )
        if candidates
        else None,
        "card_permanent_bonus_not_exactly_recoverable_from_bonus_token": sum(
            min(card.perma_bonus // 5, 31) * 5 != card.perma_bonus for card in cards
        ),
        "joker_extra_keys_without_dedicated_numeric_feature": sorted(
            {
                key
                for joker in state.jokers
                if isinstance(joker.extra, dict)
                for key in joker.extra
                if key not in JOKER_EXTRA_FIELDS
            }
        ),
        "caveat": (
            "Other observation fields/history may supply information; "
            "these counts do not prove state aliasing or causality"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    data = load_game_data()
    model, _, _ = load_policy(args.checkpoint, build_vocab(data), device, mixture=0.5)
    snapshots = json.loads(args.snapshots.read_text())[:100]
    started = time.monotonic()
    results = []
    with (args.output_dir / "interventions.jsonl").open("x", buffering=1) as stream:
        for row in snapshots:
            env = pickle.loads(gzip.decompress(Path(row["path"]).read_bytes()))
            before = live_state_digest(env)
            probes = []
            oracle_action = None
            if decode_action(row["student_action"]).action_type == ActionType.PLAY_SUBSET:
                teacher = HeuristicAgent(shop_policy="search", grow_scalers=True)
                candidates, _ = generate_hand_candidates(env.state)
                for candidate in candidates:
                    action = subset_index(candidate.indices)
                    from pylatro_agent.action import ActionRange

                    action += int(ActionRange.PLAY_SUBSET_START)
                    if not env.action_masks()[action]:
                        continue
                    score = teacher._estimate_hand_score_compute(env.state, candidate.indices)
                    probes.append(
                        {
                            "action": action,
                            "raw_score_proxy": candidate.raw_score,
                            "structural_estimate": candidate.estimated_score,
                            "exact_score": score,
                        }
                    )
                if probes:
                    oracle_action = max(probes, key=lambda p: p["exact_score"])["action"]
                for name in ("student_action", "teacher_action"):
                    action = row[name]
                    decoded = decode_action(action)
                    if decoded.action_type == ActionType.PLAY_SUBSET and not any(p["action"] == action for p in probes):
                        indices = subset_indices(decoded.index)
                        probes.append(
                            {
                                "action": action,
                                "raw_score_proxy": None,
                                "structural_estimate": None,
                                "exact_score": teacher._estimate_hand_score_compute(env.state, indices),
                                "extra_probe": name,
                            }
                        )
            if live_state_digest(env) != before:
                raise RuntimeError("Scoring probe mutated the live state or its RNG")
            interventions = {"student": row["student_action"], "teacher": row["teacher_action"]}
            if oracle_action is not None:
                interventions["exact_candidate_oracle"] = oracle_action
            result = {
                "seed": row["seed"],
                "step": row["step"],
                "phase": row["phase"],
                "ante": row["ante"],
                "first_actions": interventions,
                "exact_score_probes": probes,
                "input_audit": audit_inputs(env.state, probes),
                "outcomes": {},
            }
            for name, action in interventions.items():
                clone = clone_diagnostic_env(env)
                result["outcomes"][name] = continuation(model, clone, action, device, row["continuation_sampling_seed"])
                clone.close()
            env.close()
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            results.append(result)
            print(f"Interventions: {len(results)}/{len(snapshots)}", flush=True)
    summary = {}
    from pylatro_agent.transfer import _bootstrap

    for intervention in ("teacher", "exact_candidate_oracle"):
        paired = [row for row in results if intervention in row["outcomes"]]
        summary[intervention] = {"games": len(paired)}
        for metric in ("won", "max_ante", "blind_survived"):
            differences = [
                float(r["outcomes"][intervention][metric]) - float(r["outcomes"]["student"][metric]) for r in paired
            ]
            summary[intervention][metric] = (
                {"mean_difference": float(np.mean(differences)), "paired_ci95": _bootstrap(differences)}
                if differences
                else None
            )
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "manifest.json",
        {
            "source_hashes": source_hashes(),
            "elapsed_seconds": time.monotonic() - started,
            "snapshots": str(args.snapshots),
            "continuation": "same sampling seed, batch 1 in both branches",
            "interpretation": (
                "Exploratory state/action intervention; an oracle policy change is not a trained new-input experiment"
            ),
        },
    )


if __name__ == "__main__":
    main()
