"""Frozen neural proposals with bounded, paired public-world tactical search."""

from __future__ import annotations

import math
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.constants import SubPhase
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.heuristic_blind_rollout import candidate_actions
from pylatro_agent.heuristic_shop_search import ShopSearch
from pylatro_agent.public_runtime import PublicHeuristic, PublicRunner, addressable_hand
from pylatro_agent.public_state import OBSERVATION_INTERPRETATION, canonical, public_mask, simulation_state


class PlanningDeadline(Exception):
    pass


@contextmanager
def bounded(seconds):
    """Interrupt candidate generation as well as continuations on POSIX."""
    if seconds <= 0:
        raise PlanningDeadline
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Bounded search must run in a worker's main thread")
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0]:
        raise RuntimeError("Cannot replace an existing process deadline")

    def expired(signum, frame):
        raise PlanningDeadline

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass(frozen=True)
class PlannerConfig:
    mode: str = "both"
    seconds: float = 3.0
    samples: int = 4
    hand_candidates: int = 6
    shop_candidates: int = 4
    hand_steps: int = 60
    shop_steps: int = 100
    sample_start: int = 0

    def __post_init__(self):
        if self.mode not in {"off", "hand", "shop", "both"} or not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("Invalid hybrid mode or planning budget")
        if type(self.samples) is not int or self.samples not in {4, 16, 32, 64}:
            raise ValueError("Search samples must be 4, 16, 32, or 64")
        if type(self.sample_start) is not int or self.sample_start < 0:
            raise ValueError("Sample start must be a nonnegative integer")
        if (self.hand_candidates, self.shop_candidates, self.hand_steps, self.shop_steps) != (
            6,
            4,
            60,
            100,
        ):
            raise ValueError("This architecture has one fixed search configuration")


@dataclass(frozen=True)
class PendingPurchase:
    slot: int
    offer: str
    shop_context: tuple


@dataclass
class Decision:
    action: int
    proposal: int
    pending_purchase: PendingPurchase | None = None
    diagnostics: dict = field(default_factory=dict)
    trace: dict | None = None


@dataclass(frozen=True)
class Plan:
    actions: tuple[int, ...]
    utility: float = 0.0


def shop_context(state):
    return state.round_resets.ante, state.round, state.blind_on_deck


class HybridPolicy:
    def __init__(self, base_policy, data, config=None):
        self.base = base_policy
        self.data = data
        self.config = config or PlannerConfig()
        self.pending = None
        self.cache = {}
        self.continuation = PublicHeuristic(data)

    def _hand_plans(self, snapshot, proposal):
        state = simulation_state(snapshot, self.data)
        mask = public_mask(state, snapshot.sub_phase)
        agent = HeuristicAgent(grow_scalers=True)
        heuristic = self.continuation(snapshot)
        actions = [proposal]
        if heuristic not in actions and (AR.PLAY_SUBSET_START <= heuristic <= AR.DISCARD_SUBSET_END):
            actions.append(heuristic)
        # Reserve space for a discard even when the structural play shortlist
        # is full, and preserve both legal families.
        with addressable_hand(state):
            found = candidate_actions(agent, state, mask, proposal)
        discards = [a for a in found if AR.DISCARD_SUBSET_START <= a <= AR.DISCARD_SUBSET_END]
        if not discards:
            discards = [int(a) for a in mask.nonzero()[0] if AR.DISCARD_SUBSET_START <= a <= AR.DISCARD_SUBSET_END][:1]
        plays = [a for a in found if AR.PLAY_SUBSET_START <= a <= AR.PLAY_SUBSET_END]
        for action in discards[:1] + plays[:1] + found:
            if action not in actions and mask[action]:
                actions.append(action)
        return [Plan((a,)) for a in actions[: self.config.hand_candidates]]

    def _shop_plans(self, snapshot, proposal):
        state = simulation_state(snapshot, self.data)
        mask = public_mask(state, SubPhase.SHOP)
        agent = HeuristicAgent(grow_scalers=True)
        valuation = ShopSearch()
        baseline, base_value = valuation.purchase_output(state, agent)
        hands = valuation.round_hands(state, boss=state.blind_on_deck == "Boss")
        target = valuation.target(state, agent)
        safe = baseline * max(1, hands - 1) >= target * (
            1.3 if len(state.jokers) < 2 or state.round_resets.ante >= 5 else 1.1
        )
        base_income = valuation.income_value(state) if safe else 0
        plans = [Plan((proposal,))]
        if proposal != AR.SHOP_LEAVE:
            plans.append(Plan((int(AR.SHOP_LEAVE),)))
        purchases = []
        for slot, item in enumerate(state.shop.cards):
            if item.card_type != "Joker":
                continue
            buy = int(AR.SHOP_BUY_START + slot)
            sales = [None] + [
                int(a) for a in mask.nonzero()[0] if AR.SHOP_SELL_JOKER_START <= a <= AR.SHOP_SELL_JOKER_END
            ]
            for sale in sales:
                trial = PublicRunner.from_snapshot(snapshot, self.data)
                sequence = (buy,) if sale is None else (sale, buy)
                valid = True
                for action in sequence:
                    if not trial.compute_mask()[action]:
                        valid = False
                        break
                    trial.step(action)
                if not valid:
                    continue
                # Actual transitions include sale triggers and Negative capacity.
                _, value = valuation.purchase_output(trial.state, agent)
                income = valuation.income_value(trial.state) if safe else 0
                spent = max(0, state.dollars - trial.state.dollars)
                penalty = (0.03 if safe else 0.012) * spent
                if safe and state.round_resets.ante < 7:
                    penalty += (
                        0.004
                        * max(0, min(state.dollars, 25) - max(0, trial.state.dollars))
                        * (8 - state.round_resets.ante)
                    )
                purchases.append(Plan(sequence, value - base_value + income - base_income - penalty))
        for i, control in enumerate(plans):
            match = next((p for p in purchases if p.actions == control.actions), None)
            if match is not None:
                plans[i] = match
        for plan in sorted(purchases, key=lambda p: p.utility, reverse=True):
            if plan.actions not in {p.actions for p in plans}:
                plans.append(plan)
            if len(plans) == self.config.shop_candidates:
                break
        return plans

    def _trial(self, snapshot, plan, sample, hand):
        key = (snapshot.key, plan.actions, sample, canonical(asdict(self.config)))
        if key in self.cache:
            return self.cache[key]
        runner = PublicRunner.from_snapshot(snapshot, self.data, sample)
        entered = hand
        remaining_resources = 0

        def advance(action):
            nonlocal remaining_resources
            if runner.sub_phase == SubPhase.CHOOSE_ACTION:
                remaining_resources = (
                    runner.state.current_round.hands_left
                    + runner.state.current_round.discards_left
                    - int(AR.PLAY_SUBSET_START <= action <= AR.DISCARD_SUBSET_END)
                )
            runner.step(action)

        for action in plan.actions:
            if not runner.compute_mask()[action]:
                raise RuntimeError("A shortlisted plan became illegal in a paired world")
            advance(action)
        entered |= runner.sub_phase == SubPhase.CHOOSE_ACTION
        cap = self.config.hand_steps if hand else self.config.shop_steps
        complete = False
        for _ in range(cap + 1):
            if runner.done or (entered and runner.sub_phase == SubPhase.SHOP):
                complete = True
                break
            if _ == cap:
                break
            # The continuation receives a projection, not runner.state. This
            # scrubs the simulated world's future draws and RNG too.
            choice = self.continuation(runner.snapshot())
            advance(choice)
            entered |= runner.sub_phase == SubPhase.CHOOSE_ACTION
        clear = int(runner.won or (complete and entered and not runner.done and runner.sub_phase == SubPhase.SHOP))
        if hand:
            target = max(1, runner._ctrl.blind_target())
            score = (
                clear,
                runner.state.dollars if clear else 0,
                remaining_resources if clear else 0,
                min(1, runner.round_score / target) if not clear else 0,
            )
        else:
            # The existing portfolio utility breaks completion ties. Do not
            # count its value/income terms again at the end of the rollout.
            score = (clear, plan.utility)
        result = (score, complete)
        if len(self.cache) >= 4096:
            self.cache.clear()
        self.cache[key] = result
        return result

    def decide(self, snapshot, *, trace=False):
        started = time.monotonic()
        proposal = int(self.base(snapshot))  # exactly one policy RNG advance
        state = simulation_state(snapshot, self.data)
        mask = public_mask(state, snapshot.sub_phase)
        if not mask[proposal]:
            raise RuntimeError("Illegal base proposal")
        diagnostics = dict(
            mode=self.config.mode,
            samples=0,
            candidates=0,
            changed=False,
            fallback=None,
            search_stop=None,
            pending_cancelled=False,
            interpretation=OBSERVATION_INTERPRETATION,
            requested_samples=self.config.samples,
        )
        if self.pending is not None:
            pending, self.pending = self.pending, None
            action = int(AR.SHOP_BUY_START + pending.slot)
            if (
                snapshot.sub_phase == SubPhase.SHOP
                and shop_context(state) == pending.shop_context
                and pending.slot < len(state.shop.cards)
                and canonical(asdict(state.shop.cards[pending.slot])) == pending.offer
                and mask[action]
            ):
                diagnostics.update(
                    pending_executed=True, changed=action != proposal, seconds=time.monotonic() - started
                )
                result = Decision(action, proposal, diagnostics=diagnostics)
                if trace:
                    result.trace = self._trace(snapshot, result, [], [])
                return result
            diagnostics["pending_cancelled"] = True
        hand = snapshot.sub_phase == SubPhase.CHOOSE_ACTION and self.config.mode in {"hand", "both"}
        shop = snapshot.sub_phase == SubPhase.SHOP and self.config.mode in {"shop", "both"}
        hand &= AR.PLAY_SUBSET_START <= proposal <= AR.DISCARD_SUBSET_END
        selected = Plan((proposal,))
        plans, completed = [], []
        planning_start = time.monotonic()
        if hand or shop:
            try:
                with bounded(self.config.seconds):
                    plans = self._hand_plans(snapshot, proposal) if hand else self._shop_plans(snapshot, proposal)
                    diagnostics["candidates"] = len(plans)
                    for sample in range(self.config.sample_start, self.config.sample_start + self.config.samples):
                        paired = [self._trial(snapshot, plan, sample, hand) for plan in plans]
                        if not all(done for _, done in paired):
                            diagnostics["search_stop"] = "continuation_cap"
                            break
                        completed.append([score for score, _ in paired])
            except PlanningDeadline:
                diagnostics["search_stop"] = "deadline"
            if completed:
                values = [
                    tuple(sum(row[i][j] for row in completed) / len(completed) for j in range(len(completed[0][i])))
                    for i in range(len(plans))
                ]
                # Python max keeps the first (model) plan on a tie.
                selected = plans[max(range(len(plans)), key=lambda i: values[i])]
                diagnostics["values"] = values
                diagnostics["samples"] = len(completed)
            else:
                diagnostics["fallback"] = diagnostics["search_stop"] or "insufficient_search"
        if len(selected.actions) == 2:
            slot = selected.actions[1] - AR.SHOP_BUY_START
            self.pending = PendingPurchase(slot, canonical(asdict(state.shop.cards[slot])), shop_context(state))
        diagnostics.update(
            changed=selected.actions[0] != proposal,
            seconds=time.monotonic() - started,
            planning_seconds=time.monotonic() - planning_start,
        )
        result = Decision(selected.actions[0], proposal, self.pending, diagnostics)
        if trace:
            # Construct/serialize detailed public records only after all timers.
            result.trace = self._trace(snapshot, result, plans, completed)
        return result

    def _trace(self, snapshot, decision, plans, completed):
        return dict(
            snapshot=snapshot.values,
            snapshot_key=snapshot.key,
            config=asdict(self.config),
            proposal=decision.proposal,
            candidates=[asdict(plan) for plan in plans],
            selection=decision.action,
            completed_scores=completed,
            sample_indices=list(range(self.config.sample_start, self.config.sample_start + len(completed))),
            stopping_reason=decision.diagnostics["search_stop"] or ("completed" if completed else "not_eligible"),
            diagnostics=decision.diagnostics.copy(),
        )
