"""One resumable study unit: a whole game with independently owned policy RNG."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import asdict

from pylatro import load_game_data
from pylatro_agent.hybrid import HybridPolicy, PlannerConfig
from pylatro_agent.public_runtime import PublicNeuralPolicy, PublicRunner
from pylatro_agent.public_state import OBSERVATION_INTERPRETATION
from pylatro_cli.controller import GamePhase

_POLICIES = {}


def play_game(
    checkpoint,
    seed,
    *,
    mode="off",
    seconds=3.0,
    search_samples=4,
    sample=False,
    temperature=1.0,
    win_ante=8,
    deadline=None,
    boundary_audit=False,
    trace_sink=None,
):
    started = time.monotonic()
    policy_seed = int.from_bytes(hashlib.sha256(f"HYBRID_POLICY_STREAM_0:{seed}".encode()).digest()[:8], "big")
    key = (str(checkpoint), sample, temperature)
    if key not in _POLICIES:
        data = load_game_data()
        _POLICIES[key] = PublicNeuralPolicy(checkpoint, data, sample=sample, temperature=temperature)
    base = _POLICIES[key]
    base.reset_rng(policy_seed)
    runner = PublicRunner(
        seed, base.data, deck_key="b_blue", stake=1, win_ante=win_ante, max_steps=10000, raise_errors=True
    )
    config = PlannerConfig(mode=mode, seconds=seconds, samples=search_samples)
    policy = HybridPolicy(base, base.data, config)
    changes = search = fallback = pending = samples = 0
    elapsed, planning, candidates = [], [], []
    fallbacks = Counter()
    stops, sample_counts = Counter(), Counter()
    steps = audits = 0
    while not runner.done:
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Study wall deadline reached before game completion")
        snapshot = runner.snapshot()
        if boundary_audit and steps % 10 == 0:
            from copy import deepcopy

            from pylatro.rng import PseudorandomState
            from pylatro_agent.public_state import PublicGameSnapshot

            changed = deepcopy(runner.state, {id(runner.state.data): runner.state.data})
            changed.seed = "AUDIT_PRIVATE_SEED"
            changed.pseudorandom = PseudorandomState(changed.seed)
            changed.draw_pile.reverse()
            changed.deck_cards.reverse()
            changed.current_voucher = "v_antimatter"
            if snapshot.values["joker_order_unknown"]:
                changed.jokers.reverse()
                changed.joker_keys = [j.center_key for j in changed.jokers]
            for card in changed.deck_cards:
                card.reward_uid += 100000
            alternate = PublicGameSnapshot.capture(changed, runner.sub_phase, runner.round_score, runner.history)
            if alternate != snapshot:
                raise RuntimeError("Information-boundary violation in gameplay audit")
            audits += 1
        decision = policy.decide(snapshot, trace=trace_sink is not None)
        if not runner.compute_mask()[decision.action]:
            raise RuntimeError(f"Illegal executed action {decision.action} at decision {steps}")
        runner.step(decision.action)
        if trace_sink is not None:
            from pylatro_agent.search_diagnostics import public_resources

            after = runner.snapshot()
            trace_sink(
                dict(
                    step=steps,
                    **decision.trace,
                    public_resources=public_resources(snapshot, base.data),
                    observed_transition=dict(
                        snapshot=after.values,
                        snapshot_key=after.key,
                        resources=public_resources(after, base.data),
                        done=bool(runner.done),
                        won=bool(runner.won),
                    ),
                )
            )
        d = decision.diagnostics
        changes += int(d["changed"])
        search += int(d["candidates"] > 0 or d["fallback"] is not None)
        if d["candidates"] > 0 or d["fallback"] is not None:
            stops[d["search_stop"] or "completed"] += 1
            sample_counts[str(d["samples"])] += 1
        fallback += int(d["fallback"] is not None)
        samples += d["samples"]
        pending += int(d.get("pending_executed", False))
        fallbacks[str(d["fallback"])] += 1
        elapsed.append(d["seconds"])
        planning.append(d.get("planning_seconds", 0))
        candidates.append(d["candidates"])
        steps += 1
    if runner.phase not in (GamePhase.GAME_OVER, GamePhase.GAME_WON):
        raise RuntimeError("Game hit an execution cap without a terminal outcome")
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Study wall deadline reached during the final decision")
    return dict(
        seed=seed,
        mode=mode,
        sample=sample,
        won=bool(runner.won),
        ante=runner.max_ante,
        steps=steps,
        seconds=time.monotonic() - started,
        changes=changes,
        searches=search,
        fallbacks=fallback,
        fallback_reasons=dict(fallbacks),
        paired_sample_sets=samples,
        pending_purchases=pending,
        decision_seconds=elapsed,
        planning_seconds=planning,
        candidates=candidates,
        illegal_actions=0,
        complete=True,
        interpretation=OBSERVATION_INTERPRETATION,
        policy_rng_stream=0,
        boundary_audits=audits,
        planner_config=asdict(config),
        search_stops=dict(stops),
        sample_counts=dict(sample_counts),
        boundary_violations=0,
    )
