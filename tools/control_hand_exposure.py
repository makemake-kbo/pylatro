#!/usr/bin/env python3
"""Offline CPU control: fit the original AR hand heads with matched conditional exposure.

This post-hoc diagnostic cannot select the confirmation candidate. It holds the
encoder, records, example order, optimizer settings and update count fixed while
training the original count/card decoder with pure-AR conditional likelihood.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from pylatro import load_game_data
from pylatro_agent.action_grammar import _all_subset_log_probs, _hand_subset_bits, _hand_subset_sizes
from pylatro_agent.constants import MAX_HAND_SIZE
from pylatro_agent.learning_bottleneck import CHECKPOINT, atomic_json, read_json
from pylatro_agent.subset_actions import HAND_SUBSETS
from pylatro_agent.subset_residual import subset_support
from pylatro_agent.transfer import load_policy, sha256, source_hashes
from pylatro_agent.vocab import build_vocab

SNAPSHOTS = (0, 100, 300, 1000, 2000)


class OriginalHandDecoder(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.hand_count_head = copy.deepcopy(original.hand_count_head)
        self.hand_state_proj = copy.deepcopy(original.hand_state_proj)
        self.hand_card_proj = copy.deepcopy(original.hand_card_proj)
        self.hand_card_head = copy.deepcopy(original.hand_card_head)

    def predict(self, width, batch):
        state, cards = batch["state"], batch["cards"]
        rows = torch.arange(len(state))
        counts = self.hand_count_head(state).view(-1, 2, 5)[rows, batch["family"]]
        hidden = self.hand_card_proj(cards) + self.hand_state_proj(state)[:, None]
        logits = self.hand_card_head(hidden).permute(0, 2, 1)[rows, batch["family"]]
        logits = F.pad(logits, (0, MAX_HAND_SIZE - width), value=-1e8)
        support, _ = subset_support(width)
        valid = torch.zeros((len(state), len(HAND_SUBSETS)), dtype=torch.bool)
        valid[:, support] = batch["valid"]
        return _all_subset_log_probs(
            counts, logits, valid,
            sizes=_hand_subset_sizes(torch.device("cpu")), subset_bits=_hand_subset_bits(torch.device("cpu")),
        )[:, support]


def metrics(head, data):
    values, identities, families = [], [], []
    with torch.no_grad():
        for width, bucket in sorted(data.items()):
            for start in range(0, len(bucket["labels"]), 320):
                batch = {k: v[start : start + 320] for k, v in bucket.items()}
                logp = head.predict(width, batch)
                if not torch.allclose((logp.exp() * batch["valid"]).sum(-1), torch.ones(len(logp)), atol=2e-5):
                    raise ValueError("Control hand distribution is not normalized")
                target = logp.gather(1, batch["labels"][:, None])[:, 0]
                values.append(torch.stack((-target, target.exp(), (logp.argmax(-1) == batch["labels"]).float()), -1))
                identities.append(batch["ids"])
                families.append(batch["family"])
    values, identities, families = torch.cat(values), torch.cat(identities), torch.cat(families)
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite control metrics")
    result = {}
    names = ("conditional_nll", "conditional_teacher_probability", "conditional_accuracy")
    for label, mask in (
        ("all", torch.ones(len(values), dtype=torch.bool)), ("play", families == 0), ("discard", families == 1),
    ):
        selected, seeds = values[mask], identities[mask, 0]
        means = torch.stack([selected[seeds == seed].mean(0) for seed in seeds.unique()])
        result[label] = dict(
            decisions=len(selected), games=len(means),
            decision_weighted=dict(zip(names, selected.mean(0).tolist(), strict=True)),
            game_weighted=dict(zip(names, means.mean(0).tolist(), strict=True)),
        )
    return result


def train(root, seed, *, updates=2000):
    destination = root / "hand_exposure_control" / f"pure_ar_seed{seed}"
    if (destination / "complete.json").exists():
        return
    destination.mkdir(parents=True, exist_ok=False)
    info = read_json(root / "subset_cache_complete.json")
    if sha256(CHECKPOINT) != info["source_checkpoint_sha256"]:
        raise ValueError("Original checkpoint changed")
    for split, digest in info["hashes"].items():
        if sha256(root / f"subset_{split}.pt") != digest:
            raise ValueError("Shared hand cache changed")
    training, validation = (
        torch.load(root / f"subset_{split}.pt", weights_only=True) for split in ("training", "validation")
    )
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    original, _, _ = load_policy(CHECKPOINT, build_vocab(load_game_data()), torch.device("cpu"), mixture=1.0)
    head = OriginalHandDecoder(original.action_grammar_head)
    del original
    protocol = dict(
        post_hoc=True, use_for_confirmation_selection=False, device="cpu", accelerator_seconds=0,
        source_checkpoint_sha256=sha256(CHECKPOINT), cache_hashes=info["hashes"],
        source_hashes=source_hashes(), training_seed=seed, updates=updates, batch_size=320,
        lr=3e-4, weight_decay=0.01, warmup=100, schedule="linear warmup then cosine",
        loss="unit conditional teacher action NLL; pure AR mixture 1.0; no macro or critic loss",
        trainable_parameters=sum(p.numel() for p in head.parameters()),
        frozen="cached original encoder/global state; all non-hand heads",
        comparison="subset residual: identical records, order, conditional exposure and optimizer settings",
        limits="Different function class, initialization, parameter count and reference mixture; offline only",
        snapshots=SNAPSHOTS, selection="minimum validation NLL; earlier update breaks ties",
    )
    atomic_json(destination / "manifest.json", protocol)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    index = torch.cat([
        torch.stack((torch.full((len(b["labels"]),), w), torch.arange(len(b["labels"]))), -1)
        for w, b in sorted(training.items())
    ])
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(index), generator=generator)
    cursor, best, started = 0, (float("inf"), -1), time.monotonic()
    ordering = hashlib.sha256()
    with (destination / "learning.jsonl").open("x") as stream:
        for update in range(updates + 1):
            if update in SNAPSHOTS or update == updates:
                validation_metrics = metrics(head, validation)
                row = dict(update=update, validation=validation_metrics, elapsed_cpu_seconds=time.monotonic() - started)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                torch.save(head.state_dict(), destination / f"head_step{update}.pt")
                key = (validation_metrics["all"]["decision_weighted"]["conditional_nll"], update)
                if key < best:
                    best = key
                    torch.save(head.state_dict(), destination / "head_best.pt")
                print("original AR control", seed, update, validation_metrics["all"]["decision_weighted"], flush=True)
            if update == updates:
                break
            if cursor + 320 > len(order):
                order = torch.cat((order[cursor:], torch.randperm(len(index), generator=generator)))
                cursor = 0
            selected = order[cursor : cursor + 320]
            cursor += 320
            ordering.update(selected.numpy().tobytes())
            rows = index[selected]
            lr = 3e-4 * min(1.0, (update + 1) / 100) * (
                1.0 if update < 100 else 0.5 * (1 + math.cos(math.pi * (update - 100) / 1900))
            )
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for width_tensor in rows[:, 0].unique():
                width = int(width_tensor)
                indices = rows[rows[:, 0] == width, 1]
                batch = {k: v[indices] for k, v in training[width].items()}
                logp = head.predict(width, batch)
                loss = -logp.gather(1, batch["labels"][:, None]).sum() / 320
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite original-hand control loss")
                loss.backward()
                loss_value += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if (update + 1) % 100 == 0:
                print("update", seed, update + 1, loss_value, time.monotonic() - started, flush=True)
    if updates == 2000:
        expected_order = read_json(root / f"subset_residual_v1_seed{seed}" / "adapter_complete.json")["ordering_sha256"]
        if ordering.hexdigest() != expected_order:
            raise ValueError("Original hand and subset residual exposure ordering differs")
    atomic_json(destination / "complete.json", dict(
        best_validation_update=best[1], best_validation_nll=best[0], updates=updates,
        examples_processed=updates * 320, ordering_sha256=ordering.hexdigest(),
        elapsed_cpu_seconds=time.monotonic() - started, accelerator_seconds=0,
        head_sha256=sha256(destination / "head_best.pt"), manifest_sha256=sha256(destination / "manifest.json"),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/action_refactor_20260913"))
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    train(args.root, args.seed)


if __name__ == "__main__":
    main()
