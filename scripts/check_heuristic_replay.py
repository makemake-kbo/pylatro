#!/usr/bin/env python3
"""Compare heuristic evaluation and recorded training decisions on full games."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from importlib import import_module
from itertools import zip_longest
from pathlib import Path

from pylatro import load_game_data
from pylatro_agent.constants import SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_blind_rollout import BlindRollout
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_generate import _run_game_single_pass
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.vocab import build_vocab


class RecordedAgent(HeuristicAgent):
    def __init__(self, shop_policy, grow_scalers):
        super().__init__(shop_policy=shop_policy, grow_scalers=grow_scalers)
        self.decisions = []

    def select_action(self, state, sub_phase, mask, **kwargs):
        action = super().select_action(state, sub_phase, mask, **kwargs)
        if not mask[action]:
            raise RuntimeError(f"Illegal action {action} in {sub_phase}")
        self.decisions.append({
            "action": int(action), "ante": state.round_resets.ante, "phase": str(sub_phase),
            "dollars": state.dollars, "hands": state.current_round.hands_left,
            "discards": state.current_round.discards_left,
            "jokers": self._joker_keys_sig(state), "levels": self._hands_sig(state),
            "consumables": [c.center_key for c in state.consumables],
            "hand": [(c.front_key, c.center_key, c.seal, c.edition_key, c.perma_bonus,
                      c.debuff, c.face_down, c.forced_selection) for c in state.hand_cards],
        })
        return action


def check(seed, data, tokenizer, args):
    started = time.monotonic()
    fast = RecordedAgent(args.shop_policy, args.grow_scalers)
    search = (
        BlindRollout(repeat=args.blind_rollout != "opening", all_blinds=args.blind_rollout == "all",
                     confirm_early=args.confirm_early)
        if args.blind_rollout else None
    )
    runner = FastRunner(seed, data, deck_key=args.deck, stake=args.stake, win_ante=8, raise_errors=True)
    while not runner.done:
        action = fast.select_action(
            runner.state, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score,
        )
        if search is not None and runner.sub_phase == SubPhase.CHOOSE_ACTION:
            action = search.select(fast, runner.state, runner.compute_mask(), action, runner.round_score)
        fast.decisions[-1]["action"] = int(action)
        runner.step(action)
    recorded = RecordedAgent(args.shop_policy, args.grow_scalers)
    records, _ante, won = _run_game_single_pass(
        seed, data, tokenizer, recorded, 0.997, deck_key=args.deck, stake=args.stake,
        blind_rollout=args.blind_rollout, blind_confirm_early=args.confirm_early,
    )
    for decision, record in zip(recorded.decisions, records, strict=True):
        decision["action"] = int(record["action"])
        assert record["heuristic_blind_rollout"] == args.blind_rollout
        assert record["heuristic_blind_confirm_early"] == args.confirm_early
    first = next((i for i, (a, b) in enumerate(zip_longest(fast.decisions, recorded.decisions)) if a != b), None)
    result = {
        "seed": seed, "match": first is None and runner.won == won,
        "fast_steps": len(fast.decisions), "recorded_steps": len(recorded.decisions),
        "training_records": len(records), "fast_won": runner.won, "recorded_won": won,
        "first_difference": first, "seconds": round(time.monotonic() - started, 2),
    }
    if first is not None:
        result["fast_difference"] = fast.decisions[first] if first < len(fast.decisions) else None
        result["recorded_difference"] = recorded.decisions[first] if first < len(recorded.decisions) else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 11])
    parser.add_argument("--deck", default="b_blue")
    parser.add_argument("--stake", type=int, default=1)
    parser.add_argument("--shop-policy", choices=["legacy", "search", "rollout"], default="search")
    parser.add_argument("--grow-scalers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blind-rollout", nargs="?", const="opening", choices=["opening", "adaptive", "all"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-early", action="store_true")
    args = parser.parse_args()
    if args.confirm_early and not args.blind_rollout:
        parser.error("--confirm-early requires --blind-rollout")
    # Keep both paths on the same imported implementation during experiments.
    for name in ("heuristic_draw", "heuristic_growth", "heuristic_shop_search",
                 "heuristic_shop_rollout", "heuristic_simulation", "heuristic_continuation",
                 "heuristic_blind_rollout"):
        import_module(f"pylatro_agent.{name}")
    data = load_game_data()
    tokenizer = Tokenizer(vocab=build_vocab(data))
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    manifest = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for package in ("pylatro", "pylatro_agent", "pylatro_cli")
        for path in (root / "src" / package).rglob("*")
        if path.suffix in {".py", ".so", ".json"}
    }
    manifest[str(Path(__file__).resolve().relative_to(root))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for seed in args.seeds:
        result = check(seed, data, tokenizer, args)
        results.append(result)
        args.output.write_text(json.dumps({
            "deck": args.deck, "stake": args.stake, "shop_policy": args.shop_policy,
            "grow_scalers": args.grow_scalers, "blind_rollout": args.blind_rollout,
            "blind_confirm_early": args.confirm_early, "results": results,
        }, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if not all(result["match"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
