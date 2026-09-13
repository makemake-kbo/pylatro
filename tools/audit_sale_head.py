#!/usr/bin/env python3
"""CPU-only decomposition of Joker-sale probabilities on the fixed HF audit."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from learning_bottleneck import hf_selection

from pylatro import load_game_data
from pylatro_agent.action import ActionRange, ActionType
from pylatro_agent.action_grammar import ACTION_TYPE_TO_GRAMMAR_INDEX
from pylatro_agent.learning_bottleneck import CHECKPOINT
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import load_policy, sha256, source_hashes, write_json
from pylatro_agent.vocab import build_vocab


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("eval/learning_bottleneck_20260913"))
    args = parser.parse_args()
    output = args.root / "sale_head_cpu"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device("cpu")
    model, reward, _ = load_policy(CHECKPOINT, build_vocab(load_game_data()), device, mixture=0.5)
    audit_path = args.root / "audit/mixture0.5/predictions.jsonl"
    audit = {(r["seed"], r["step"]): r for r in map(json.loads, audit_path.read_text().splitlines())}
    lo, hi = int(ActionRange.SHOP_SELL_JOKER_START), int(ActionRange.SHOP_SELL_JOKER_END)
    records = [r for r in hf_selection(reward) if lo <= int(r["action"]) <= hi]
    macro = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_SELL_JOKER]
    rows = []
    with torch.inference_mode(), (output / "predictions.jsonl").open("x") as stream:
        for start in range(0, len(records), 320):
            chunk = records[start : start + 320]
            batch = _collate_batch(chunk, device, SupervisedConfig())
            dist, _ = _grammar_distribution(model, batch)
            mask = batch["action_mask"][:, lo : hi + 1].bool()
            conditional = torch.softmax(dist.output.shop_sell_joker_logits.masked_fill(~mask, -1e8), -1)
            indexes = batch["actions"] - lo
            conditional_p = conditional.gather(1, indexes[:, None]).squeeze(1)
            macro_p = dist.action_type_probs[:, macro]
            logp = dist.log_prob(batch["actions"])
            assert torch.allclose(logp, macro_p.log() + conditional_p.log(), atol=2e-5, rtol=1e-5)
            entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(-1)
            predicted = dist.mode()
            for i, record in enumerate(chunk):
                key = int(record["seed"]), int(record["step"])
                old = audit[key]
                row = dict(
                    seed=key[0],
                    step=key[1],
                    phase=int(record["obs"]["scalars"][7]),
                    legal_jokers=int(mask[i].sum()),
                    target_slot=int(indexes[i]),
                    macro_probability=float(macro_p[i]),
                    conditional_probability=float(conditional_p[i]),
                    marginal_macro_accuracy=float(dist.action_type_probs[i].argmax() == macro),
                    conditional_accuracy=float(conditional[i].argmax() == indexes[i]),
                    conditional_entropy=float(entropy[i]),
                    probability=float(logp[i].exp()),
                    nll=-float(logp[i]),
                    joint_accuracy=float(predicted[i] == batch["actions"][i]),
                    mps_nll=old["nll"],
                    mps_joint_accuracy=old["accuracy"],
                )
                assert abs(row["nll"] - row["mps_nll"]) < 5e-4
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
    metrics = [k for k in rows[0] if k not in ("seed", "step", "phase", "target_slot")]
    groups = {"overall": rows, "SHOP": [r for r in rows if r["phase"] == 2]}
    summary = {}
    for label, group in groups.items():
        summary[label] = dict(
            decisions=len(group),
            metrics={k: float(np.mean([r[k] for r in group])) for k in metrics},
            legal_uniform_teacher_probability=float(np.mean([1 / r["legal_jokers"] for r in group])),
            target_slot_counts={str(i): sum(r["target_slot"] == i for r in group) for i in range(hi - lo + 1)},
        )
    write_json(output / "summary.json", summary)
    write_json(
        output / "manifest.json",
        dict(
            status="complete",
            device="cpu",
            accelerator_seconds=0,
            checkpoint=str(CHECKPOINT),
            checkpoint_sha256=sha256(CHECKPOINT),
            audit=str(audit_path),
            audit_sha256=sha256(audit_path),
            source_hashes=source_hashes(),
            predictions_sha256=sha256(output / "predictions.jsonl"),
            decisions=len(rows),
            max_cpu_mps_nll_difference=max(abs(r["nll"] - r["mps_nll"]) for r in rows),
            joint_accuracy_device_disagreements=sum(r["joint_accuracy"] != r["mps_joint_accuracy"] for r in rows),
            elapsed_seconds=time.monotonic() - started,
            interpretation=(
                "Offline HF training-data diagnostic. Decomposes macro and conditional slot probabilities; "
                "does not establish that a particular head redesign improves gameplay."
            ),
        ),
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
