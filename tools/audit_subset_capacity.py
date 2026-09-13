#!/usr/bin/env python3
"""Find learned teacher probabilities that the original fixed mixture cannot express."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path

import torch
from subset_head_experiment import hand_indices

from pylatro import load_game_data
from pylatro_agent.action import ActionType
from pylatro_agent.action_grammar import ACTION_TYPE_TO_GRAMMAR_INDEX, ActionGrammarDistribution, _masked_log_softmax
from pylatro_agent.constants import ActionRange
from pylatro_agent.learning_bottleneck import CHECKPOINT, atomic_json, read_json
from pylatro_agent.subset_residual import SubsetResidualHead
from pylatro_agent.training.model_generate import load_records
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import load_policy, sha256, source_hashes
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    args = parser.parse_args()
    root = args.root
    marker = root / "subset_capacity_audit.json"
    if marker.exists():
        raise FileExistsError("Capacity audit already exists")
    torch.set_num_threads(4)
    if sha256(CHECKPOINT) != read_json(root / "manifest.json")["checkpoint_sha256"]:
        raise ValueError("Source checkpoint changed")
    folder = Path("eval/learning_bottleneck_20260913/data/validation")
    for name, expected in read_json(folder / "summary.json")["file_hashes"].items():
        if sha256(folder / name) != expected:
            raise ValueError("Validation data changed")
    model, reward, _ = load_policy(CHECKPOINT, build_vocab(load_game_data()), "cpu", mixture=0.5)
    records = load_records(folder, reward_config=reward)
    indices = hand_indices(records)
    heads, adapter_hashes = [], []
    for seed in (0, 1):
        path = root / f"subset_residual_v1_seed{seed}" / "adapter_best.pt"
        if sha256(path) != read_json(path.parent / "adapter_complete.json")["adapter_sha256"]:
            raise ValueError("Trained subset adapter changed")
        head = SubsetResidualHead(model.config.d_model).eval()
        head.load_state_dict(torch.load(path, weights_only=True))
        heads.append(head)
        adapter_hashes.append(sha256(path))
    captured = {}

    def state_hook(module, inputs):
        captured["state"] = inputs[0]

    def hand_hook(module, inputs):
        captured["cards"], captured["present"] = module._gather_hand_slots(*inputs[:4])

    handles = [
        model.action_grammar_head.shop_global_head.register_forward_pre_hook(state_hook),
        model.action_grammar_head.register_forward_pre_hook(hand_hook),
    ]
    values, identifiers = [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(indices), 320):
                rows = [copy.deepcopy(records[index]) for index in indices[start : start + 320]]
                batch = _collate_batch(rows, torch.device("cpu"), SupervisedConfig(reward_config=reward))
                original, _ = _grammar_distribution(model, batch)
                play = batch["actions"] < int(ActionRange.DISCARD_SUBSET_START)
                macro_indices = torch.where(
                    play,
                    ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET],
                    ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.DISCARD_SUBSET],
                )
                macro_logp = _masked_log_softmax(original.output.macro_logits, original.macro_mask).gather(
                    1, macro_indices[:, None]
                )[:, 0]
                has_candidates, covered = (
                    torch.zeros(len(rows), dtype=torch.bool),
                    torch.zeros(len(rows), dtype=torch.bool),
                )
                for is_play in (True, False):
                    selected = play if is_play else ~play
                    _, valid, mapped = original._candidate_distribution(is_play=is_play)
                    has_candidates |= selected & valid.any(-1)
                    covered |= selected & ((mapped == batch["actions"][:, None]) & valid).any(-1)
                constrained = has_candidates & ~covered
                old_probability = (original.log_prob(batch["actions"]) - macro_logp).exp()
                if (old_probability[constrained] > 0.500001).any():
                    raise ValueError("Original probabilities violate the stated fixed-mixture bound")
                probabilities = []
                for head in heads:
                    output = replace(
                        original.output,
                        hand_subset_energies=head.score_available(
                            captured["state"], captured["cards"], captured["present"]
                        ),
                    )
                    distribution = ActionGrammarDistribution(
                        output, batch["action_mask"], tokens=batch["tokens"], hand_ar_mixture_eps=0.5
                    )
                    probabilities.append((distribution.log_prob(batch["actions"]) - macro_logp).exp())
                values.append(torch.stack((play.float(), constrained.float(), old_probability, *probabilities), -1))
                identifiers.extend((row["seed"], row["step"], row["action"]) for row in rows)
                print("audited", min(start + 320, len(indices)), len(indices), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    values = torch.cat(values)
    if not torch.isfinite(values).all() or len({item[:2] for item in identifiers}) != len(indices):
        raise ValueError("Invalid audit values or duplicate decisions")
    result = dict(
        decisions=len(indices),
        source_checkpoint_sha256=sha256(CHECKPOINT),
        device="cpu",
        mixture=0.5,
        temperature=1.0,
        data_scope="current-rule validation",
        validation_manifest_sha256=sha256(folder / "summary.json"),
        adapter_hashes=adapter_hashes,
        probability_threshold=0.5001,
        source_hashes=source_hashes(),
        strata={},
        witnesses=[],
        interpretation=(
            "Representational witnesses; isolating the ceiling's gameplay contribution "
            "requires a separate intervention"
        ),
    )
    for label, selected in (
        ("all", torch.ones(len(values), dtype=torch.bool)),
        ("play", values[:, 0] == 1),
        ("discard", values[:, 0] == 0),
    ):
        constrained = selected & (values[:, 1] == 1)

        averages = values[constrained, 2:].mean(0).tolist() if constrained.any() else [None, None, None]

        result["strata"][label] = dict(
            decisions=int(selected.sum()),
            constrained_decisions=int(constrained.sum()),
            constrained_teacher_probability_original=averages[0],
            per_replicate=[
                dict(
                    training_seed=seed,
                    constrained_teacher_probability=averages[1 + seed],
                    exceeds_original_ceiling=int((constrained & (values[:, 3 + seed] > 0.5001)).sum()),
                )
                for seed in (0, 1)
            ],
        )
    constrained = values[:, 1] == 1
    witness_indices = torch.where(constrained & (values[:, 3:] > 0.5001).all(-1))[0]
    for index in witness_indices[:20].tolist():
        seed, step, action = identifiers[index]
        result["witnesses"].append(
            dict(
                seed=seed,
                step=step,
                action=action,
                old_probability=float(values[index, 2]),
                new_probabilities=values[index, 3:].tolist(),
            )
        )
    atomic_json(marker, result)
    print(result["strata"], flush=True)


if __name__ == "__main__":
    main()
