#!/usr/bin/env python3
"""Reproducible full-game heuristic evaluation; all attempted seeds count."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from importlib import import_module
from pathlib import Path

from pylatro import load_game_data
from pylatro_agent.action import decode_action
from pylatro_agent.constants import SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner


def run_game(
    seed, deck, stake, trace=False, shop_policy="legacy", grow_scalers=False,
    blind_rollout=False, confirm_early=False, rollout_min_ante=1, vagabond_policy=False,
    investment_policy=False, economy_policy=False,
):
    started = time.monotonic()
    runner = FastRunner(seed, load_game_data(), deck_key=deck, stake=stake, win_ante=8, raise_errors=True)
    from pylatro_agent.heuristic_vagabond import VagabondAgent
    from pylatro_agent.heuristic_investment import InvestmentAgent
    from pylatro_agent.heuristic_economy import EconomyAgent

    if sum((investment_policy, vagabond_policy, economy_policy)) > 1:
        raise ValueError('Economy, Investment and Vagabond actor variants cannot be combined')
    agent_type = (EconomyAgent if economy_policy else InvestmentAgent if investment_policy
                  else VagabondAgent if vagabond_policy else HeuristicAgent)
    agent = agent_type(shop_policy=shop_policy, grow_scalers=grow_scalers)
    from pylatro_agent.heuristic_blind_rollout import BlindRollout

    blind_search = (
        BlindRollout(repeat=blind_rollout in {"adaptive", "all"}, all_blinds=blind_rollout == "all",
                     confirm_early=confirm_early, min_ante=rollout_min_ante)
        if blind_rollout else None
    )
    events = []
    steps = 0
    last = {}
    while not runner.done:
        state = runner.state
        mask = runner.compute_mask()
        action = agent.select_action(state, runner.sub_phase, mask, round_score=runner.round_score)
        if blind_search is not None and runner.sub_phase == SubPhase.CHOOSE_ACTION:
            action = blind_search.select(agent, state, mask, action, runner.round_score)
        if not mask[action]:
            raise RuntimeError(f"Illegal action {action} for seed {seed}")
        last = dict(
            ante=state.round_resets.ante,
            blind=state.blind_on_deck,
            boss=state.round_resets.blind_choices.get("Boss"),
            dollars=state.dollars,
            jokers=[
                dict(key=j.center_key, mult=j.mult, chips=j.t_chips, xmult=j.x_mult, extra=deepcopy(j.extra))
                for j in state.jokers
            ],
            levels={k: v["level"] for k, v in state.hands.items() if v["level"] > 1},
            score=runner.round_score,
            target=agent._get_blind_target(state),
            hands=state.current_round.hands_left,
            discards=state.current_round.discards_left,
        )
        if trace:
            events.append(
                dict(
                    **last,
                    phase=str(runner.sub_phase),
                    action=str(decode_action(action)),
                    cards=[f"{c.suit[0]}{c.rank}:{c.center_key}:{c.seal}" for c in state.hand_cards],
                    consumables=[c.center_key for c in state.consumables],
                    tarots_used=state.consumeable_usage_total.get("tarot", 0),
                    tags=list(state.tags),
                    blind_tags=dict(state.round_resets.blind_tags),
                    investment_forecast=(
                        getattr(agent, "_investment_decision", None)
                        if runner.sub_phase == SubPhase.BLIND_SELECT else None
                    ),
                    economy_forecast=(
                        getattr(agent, "_economy_decision", None)
                        if runner.sub_phase == SubPhase.BLIND_SELECT else None
                    ),
                    shop=[i.center_key for i in state.shop.cards],
                    offers=[
                        {"key": i.center_key, "cost": i.cost}
                        for i in (*state.shop.cards, *state.shop.vouchers, *state.shop.boosters)
                    ],
                    pack=[c.center_key for c in state.pack.cards] if state.pack else [],
                    shop_forecast=(
                        getattr(agent._shop_search, "last_decision", None)
                        if runner.sub_phase == SubPhase.SHOP else None
                    ),
                    blind_forecast=(
                        blind_search.last_decision
                        if blind_search is not None and runner.sub_phase == SubPhase.CHOOSE_ACTION else None
                    ),
                )
            )
        runner.step(action)
        steps += 1
    last["score_before_final_action"] = last.get("score", 0)
    last["score"] = runner.round_score
    return dict(
        seed=seed,
        won=runner.won,
        phase=str(runner.phase),
        steps=steps,
        seconds=round(time.monotonic() - started, 2),
        final=last,
        events=events,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=0)
    seed_group.add_argument("--seed-file", type=Path, help="JSON list of distinct integer seeds")
    parser.add_argument("--games", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--deck", default="b_blue")
    parser.add_argument("--stake", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shop-policy", choices=["legacy", "search", "rollout"], default="legacy")
    parser.add_argument("--grow-scalers", action="store_true")
    parser.add_argument("--vagabond-policy", action="store_true",
                        help="Use the experimental low-cash Tarot-generation hand policy")
    parser.add_argument("--investment-policy", action="store_true",
                        help="Use the experimental Investment Tag skip policy")
    parser.add_argument("--economy-policy", action="store_true",
                        help="Use the experimental Economy Tag skip policy")
    parser.add_argument(
        "--blind-rollout", nargs="?", const="opening", choices=["opening", "adaptive", "all"],
        help="Search boss openings, all boss decisions (adaptive), or all blinds",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--confirm-early", action="store_true", help="Confirm proposed Ante 1-2 rollout changes")
    parser.add_argument("--rollout-min-ante", type=int, choices=range(1, 9), default=1,
                        help="First ante where blind rollout may change the baseline action")
    args = parser.parse_args()
    if sum((args.investment_policy, args.vagabond_policy, args.economy_policy)) > 1:
        parser.error("--economy-policy, --investment-policy and --vagabond-policy cannot be combined")
    if args.confirm_early and not args.blind_rollout:
        parser.error("--confirm-early requires --blind-rollout")
    if args.rollout_min_ante != 1 and not args.blind_rollout:
        parser.error("--rollout-min-ante requires --blind-rollout")
    if args.seed_file is not None:
        seeds = json.loads(args.seed_file.read_text())
        if (
            not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            parser.error("--seed-file must contain a nonempty JSON list of distinct integer seeds")
        if args.games is not None and args.games != len(seeds):
            parser.error("--games must match the number of seeds in --seed-file")
        args.games = len(seeds)
    else:
        args.games = 100 if args.games is None else args.games
        if args.games <= 0:
            parser.error("--games must be positive")
        seeds = list(range(args.seed, args.seed + args.games))
    # Forked workers inherit every policy module before later experiments can
    # edit a lazily imported implementation in the shared workspace.
    for module in (
        "heuristic_draw", "heuristic_growth", "heuristic_shop_search",
        "heuristic_shop_rollout", "heuristic_simulation", "heuristic_continuation",
        "heuristic_blind_rollout", "heuristic_vagabond", "heuristic_stencil",
        "heuristic_shop_choice_rollout",
        "heuristic_shop_capacity_choice",
        "heuristic_investment",
        "heuristic_economy",
    ):
        import_module(f"pylatro_agent.{module}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Freeze the same game definitions in forked workers and record the pool
    # boundary explicitly. A code-only archive cannot reproduce unlock flags.
    data = load_game_data()
    locked_jokers = sorted(
        key for key, center in data.centers.items()
        if center.get("set") == "Joker" and not center.get("unlocked", True)
        and center.get("rarity") != 4
    )
    # Record source/binary provenance before workers start, not after edits
    # from a subsequent experiment may have changed files on disk.
    root = Path(__file__).resolve().parents[1]
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for package in ("pylatro", "pylatro_agent", "pylatro_cli")
        for p in (root / "src" / package).rglob("*")
        if p.suffix in {".py", ".so", ".json"}
    }
    script_path = Path(__file__).resolve()
    manifest[str(script_path.relative_to(root))] = hashlib.sha256(script_path.read_bytes()).hexdigest()
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    profile = dict(
        deck=args.deck, stake=args.stake, win_ante=8,
        shop_policy=args.shop_policy, grow_scalers=args.grow_scalers,
        vagabond_policy=args.vagabond_policy,
        investment_policy=args.investment_policy,
        economy_policy=args.economy_policy,
        blind_rollout=args.blind_rollout,
        blind_confirm_early=args.confirm_early,
        blind_rollout_min_ante=args.rollout_min_ante if args.blind_rollout else None,
        first_seed=args.seed if args.seed_file is None else None, games=args.games,
        seeds=seeds,
        teacher_information="full_simulator_state",
        joker_pool="bundled_unlock_flags",
        excluded_locked_jokers=locked_jokers,
        game_data_sha256=manifest["src/pylatro/game_data.json"],
    )
    args.output.with_suffix(".profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    with tarfile.open(args.output.with_suffix(".sources.tar.gz"), "w:gz") as archive:
        for name in manifest:
            if name.endswith((".py", ".json")):
                archive.add(root / name, arcname=name)
    results = []
    started = time.monotonic()
    with args.output.open("w") as output, ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [
            pool.submit(
                run_game, s, args.deck, args.stake, args.trace,
                args.shop_policy, args.grow_scalers, args.blind_rollout, args.confirm_early,
                args.rollout_min_ante,
                args.vagabond_policy,
                args.investment_policy,
                args.economy_policy,
            )
            for s in seeds
        ]
        for future in as_completed(pending):
            result = future.result()
            output.write(json.dumps(result) + "\n")
            output.flush()
            results.append(result)
            wins = sum(r["won"] for r in results)
            print(
                f"{len(results)}/{args.games}: {wins} wins ({wins / len(results):.1%}), "
                f"seed {result['seed']} {'WIN' if result['won'] else 'LOSS'} "
                f"ante {result['final']['ante']}, {time.monotonic() - started:.0f}s",
                flush=True,
            )
    import pylatro_agent.heuristic as heuristic

    # Two-sided 95% Wilson interval; fresh held-out trials are still required
    # after tuning. Reaching Ante 8 alone never counts as a win.
    z = 1.959963984540054
    rate = wins / args.games
    denominator = 1 + z * z / args.games
    midpoint = (rate + z * z / (2 * args.games)) / denominator
    radius = z * ((rate * (1 - rate) / args.games + z * z / (4 * args.games**2)) ** 0.5) / denominator
    summary = dict(
        games=args.games,
        wins=wins,
        win_rate=wins / args.games,
        deck=args.deck,
        stake=args.stake,
        win_ante=8,
        first_seed=profile["first_seed"],
        shop_policy=args.shop_policy,
        grow_scalers=args.grow_scalers,
        heuristic_module=heuristic.__file__,
        heuristic_sha256=manifest[str(Path(heuristic.__file__).relative_to(root))],
        profile=profile,
        wilson_95=[midpoint - radius, midpoint + radius],
        seconds=time.monotonic() - started,
    )
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
