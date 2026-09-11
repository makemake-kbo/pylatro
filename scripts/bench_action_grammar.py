"""Isolated CUDA decoder benchmark against an explicitly supplied frozen source.

Run with PYTHONPATH pointing at either compatible package; no checkpoint or
training files are read or modified. Timings include Python/launch overhead.
The old entropy is NOT a correctness reference: exactness has separate tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch

from pylatro_agent import constants as c
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.subset_actions import HAND_SUBSETS, consumable_subset_index, subset_index


def load_grammar(label, path):
    name = "pylatro_agent.benchmark_" + label
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_case(module, batch_size, width, device):
    torch.manual_seed(42)
    shapes = {
        "macro_logits": (module.NUM_GRAMMAR_ACTIONS,),
        "hand_count_logits": (2, 5),
        "hand_card_logits": (2, c.MAX_HAND_SIZE),
        "candidate_play_logits": (c.HAND_CANDIDATE_MAX,),
        "candidate_discard_logits": (c.HAND_CANDIDATE_MAX,),
        "consumable_slot_logits": (3, c.MAX_CONSUMABLE_SLOTS),
        "consumable_count_logits": (c.MAX_CONSUMABLE_SLOTS, c.MAX_CONSUMABLE_HAND_TARGETS),
        "consumable_card_logits": (c.MAX_CONSUMABLE_SLOTS, c.MAX_HAND_SIZE),
        "consumable_joker_logits": (c.MAX_CONSUMABLE_SLOTS, c.MAX_JOKER_SLOTS),
        "shop_buy_logits": (c.MAX_SHOP_ITEMS,),
        "shop_sell_joker_logits": (c.MAX_JOKER_SLOTS,),
        "shop_sell_consumable_logits": (c.MAX_CONSUMABLE_SLOTS,),
        "pack_claim_logits": (c.MAX_PACK_CARDS,),
    }
    # The production grammar receives FP32 logits even with BF16 network/state.
    output = module.ActionGrammarOutput(
        **{name: torch.randn(batch_size, *shape, device=device, requires_grad=True) for name, shape in shapes.items()}
    )
    mask = torch.zeros(batch_size, c.NUM_ACTIONS, device=device)
    tokens = torch.zeros(batch_size, c.MAX_SEQ_LEN, c.TOKEN_DIM, dtype=torch.long, device=device)
    legal_subsets = [i for i, cards in enumerate(HAND_SUBSETS) if max(cards) < width]
    for row in range(batch_size):
        if row % 4 == 3:
            mask[
                row, [int(c.ActionRange.SHOP_LEAVE), int(c.ActionRange.SHOP_REROLL), int(c.ActionRange.SHOP_BUY_START)]
            ] = 1
        else:
            for family in (ActionType.PLAY_SUBSET, ActionType.DISCARD_SUBSET):
                mask[row, [encode_action(family, i) for i in legal_subsets]] = 1
            if row % 4 == 0:
                for cards in ([0], [1], [0, 1], [1, 2]):
                    mask[
                        row, encode_action(ActionType.USE_CONSUMABLE_HAND_SUBSET, 0, consumable_subset_index(cards))
                    ] = 1
    # Unique proposals; duplicate aggregation correctness has a separate test.
    with torch.no_grad():
        output.candidate_play_logits[:, 8:] = -1e8
        output.candidate_discard_logits[:, 8:] = -1e8
    for slot in range(8):
        cards = HAND_SUBSETS[legal_subsets[slot]]
        tokens[:, c.HAND_CANDIDATE_START + slot, 5 : 5 + len(cards)] = torch.tensor(
            [card + 1 for card in cards],
            device=device,
        )
    actions = torch.tensor(
        [
            int(c.ActionRange.SHOP_LEAVE)
            if row % 4 == 3
            else encode_action(ActionType.PLAY_SUBSET, subset_index([0, 1]))
            for row in range(batch_size)
        ],
        device=device,
    )
    return output, mask, tokens, actions


def benchmark(module, batch_size, width, device, repeats):
    output, mask, tokens, actions = make_case(module, batch_size, width, device)

    def distribution():
        return module.ActionGrammarDistribution(output, mask, tokens=tokens, hand_ar_mixture_eps=0.1)

    def mode():
        with torch.inference_mode():
            return distribution().mode()

    def entropy_backward():
        for name in output.__dataclass_fields__:
            getattr(output, name).grad = None
        entropy = distribution().entropy()
        (-0.01 * entropy.mean()).backward()
        return entropy.detach()

    # Warm both caches in training mode before inference, for old-source parity.
    entropy_backward()
    result = {}
    for name, fn in (("mode", mode), ("entropy_backward", entropy_backward)):
        fn()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            value = fn()
            torch.cuda.synchronize(device)
            times.append(1000 * (time.perf_counter() - start))
        assert torch.isfinite(value).all()
        result[name + "_median_ms"] = statistics.median(times)
        result[name + "_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    with torch.no_grad():
        dist = distribution()
        result["mode_actions"] = dist.mode().cpu().tolist()
        result["selected_log_probs"] = dist.log_prob(actions).cpu().tolist()
        result["entropy_mean"] = float(dist.entropy().mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32, 160])
    parser.add_argument("--hand-size", type=int, default=8, choices=range(3, 17))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This benchmark requires the requested remote CUDA GPU")
    torch.cuda.set_device(device)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda_device": str(device),
                "cuda_reported_name": torch.cuda.get_device_name(device),
                "note": "Check native NVML identity separately; host wrappers may spoof this name.",
            }
        ),
        flush=True,
    )
    modules = {
        label: load_grammar(label, path) for label, path in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    for size in args.batch_sizes:
        results = {}
        for label, module in modules.items():
            path = args.baseline if label == "baseline" else args.candidate
            results[label] = benchmark(module, size, args.hand_size, device, args.repeats)
            print(
                json.dumps(
                    {
                        "label": label,
                        "batch_size": size,
                        "hand_size": args.hand_size,
                        "repeats": args.repeats,
                        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        **results[label],
                    }
                ),
                flush=True,
            )
        torch.testing.assert_close(
            torch.tensor(results["baseline"]["selected_log_probs"]),
            torch.tensor(results["candidate"]["selected_log_probs"]),
            atol=2e-6,
            rtol=2e-6,
        )
        assert results["baseline"]["mode_actions"] == results["candidate"]["mode_actions"]
        print(
            json.dumps({"batch_size": size, "parity": "selected log probabilities and exact modes match"}), flush=True
        )


if __name__ == "__main__":
    main()
