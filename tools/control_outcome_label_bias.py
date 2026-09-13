#!/usr/bin/env python3
"""CPU control: full, completion-selected, and randomly missing outcome labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pylatro_agent.checkpoint import load_checkpoint_payload
from pylatro_agent.learning_bottleneck import atomic_json
from pylatro_agent.outcome_study import ROOT, critic_from_pooled, critic_metrics
from pylatro_agent.transfer import sha256
from pylatro_agent.value_head import ValueHead, outcome_nll

if __package__:
    from .outcome_ppo_study import load_cache
else:
    from outcome_ppo_study import load_cache


def completion_masks(dataset, seed):
    rng = np.random.default_rng(seed)
    seeds = dataset["seed"].numpy()
    steps = dataset["step"].numpy()
    mask = np.zeros(len(seeds), dtype=bool)
    offsets = {}
    for game_seed in np.unique(seeds):
        selected = seeds == game_seed
        offset = int(rng.integers(256))
        offsets[int(game_seed)] = offset
        length = int(selected.sum())
        if not np.array_equal(np.sort(steps[selected]), np.arange(length)):
            raise ValueError("Each game requires all ordered decision identifiers")
        mask[selected] = (offset + steps[selected]) // 256 == (offset + length - 1) // 256
    random_mask = np.zeros(len(seeds), dtype=bool)
    random_mask[rng.choice(len(seeds), size=int(mask.sum()), replace=False)] = True
    return {"full": torch.ones(len(seeds), dtype=torch.bool),
            "completion": torch.from_numpy(mask), "random": torch.from_numpy(random_mask)}, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(4)
    destination = args.root / "label_selection_cpu_control"
    destination.mkdir(exist_ok=False)
    data = load_cache(args.root)
    records = data["training"]
    population = len(records["x"])
    started = time.monotonic()
    report = dict(status="running", device="cpu", accelerator_seconds=0, training_replicates=[0, 1],
                  physical_batch=320, updates=1000, learning_rate=3e-4, weight_decay=.01,
                  actor_or_encoder_updates=False, results={}, source_sha256=sha256(__file__),
                  data_manifest_sha256=sha256(args.root / "critic_games/manifest.json"),
                  interpretation="The same frozen student-policy representations, initial critic, minibatch order and "
                                 "optimizer are used within each replicate. Only outcome-label availability changes. "
                                 "Independent uniform episode offsets model steady-state 256-step rollout boundaries. "
                                 "The random mask retains the same number of labels without conditioning on length. "
                                 "All validation games remain complete. Fixed update 100 is the prespecified short "
                                 "endpoint from the earlier critic warmup; update 1000 tests continued fitting. "
                                 "This isolates a critic-training mechanism, not a gameplay improvement.")
    atomic_json(destination / "manifest.json", report)
    for seed in (0, 1):
        source = args.root / f"critic_seed{seed}/checkpoint.pt"
        payload = load_checkpoint_payload(source, "cpu")
        head = ValueHead(payload["agent_config"]["d_model"], win_only=True)
        head.load_state_dict({key.removeprefix("value_head."): value for key, value in payload["state_dict"].items()
                              if key.startswith("value_head.")})
        masks, offsets = completion_masks(records, 1729 + seed)
        # Physical exposure and ordering are shared, including masked-out rows.
        order = torch.randint(population, (1000, 320), generator=torch.Generator().manual_seed(seed))
        ordering_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
        replicate = dict(initial_checkpoint_sha256=sha256(source), ordering_sha256=ordering_hash,
                          episode_offsets=offsets, arms={})
        for name, mask in masks.items():
            current = copy.deepcopy(head)
            for parameter in current.parameters():
                parameter.requires_grad_(False)
            parameters = [parameter for module in (current.outcome_proj, current.ante_survival)
                          for parameter in module.parameters()]
            for parameter in parameters:
                parameter.requires_grad_(True)
            optimizer = torch.optim.AdamW(parameters, lr=3e-4, weight_decay=.01)
            curve = []
            used_labels = 0
            for update in range(1001):
                if update in (0, 100, 300, 600, 1000):
                    metrics = critic_metrics(current, data["validation"])
                    curve.append(dict(update=update, valid_training_examples=used_labels, validation=metrics))
                    print(seed, name, update, json.dumps(metrics), flush=True)
                if update == 1000:
                    break
                indices = order[update]
                batch = {key: value[indices] for key, value in records.items()}
                available = mask[indices]
                used_labels += int(available.sum())
                values = critic_from_pooled(current, batch["x"], batch["ante"])
                loss = outcome_nll(values["outcome_probabilities"], batch["target"], available)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
            starts = (records["step"] == 0) & mask
            replica_path = destination / f"seed{seed}_{name}.pt"
            torch.save(current.state_dict(), replica_path)
            replicate["arms"][name] = dict(
                population_labels=int(mask.sum()), population_rows=population,
                initial_states_labeled=int(starts.sum()),
                labeled_initial_win_rate=float((records["target"][starts] == 8).float().mean()),
                mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                physical_examples=320000, valid_examples=used_labels, curve=curve,
                critic_state_sha256=sha256(replica_path),
            )
        report["results"][seed] = replicate
        atomic_json(destination / "manifest.json", report)
    report.update(status="complete", elapsed_seconds=time.monotonic() - started)
    atomic_json(destination / "manifest.json", report)
    print("Label-selection CPU controls complete", flush=True)


if __name__ == "__main__":
    main()
