"""Deterministic, objective-aware joker ordering applied by the environment harness.

Joker order only affects scoring, and the result of any given order is exactly
computable with the engine's own scorer. The harness therefore owns ordering:
before each played hand the environment permutes the roster into the best
arrangement for that concrete play. The policy has no reorder actions -
MOVE_JOKER was removed from the action space - so the model spends its capacity
on decisions that are not machine-checkable.

Two objectives exist, because joker order moves both chips and cash (copy
jokers next to an economy joker duplicate its payout):

* ``SCORE`` - maximize chips. The default.
* ``MONEY`` - maximize dollars earned by the hand.

Which one applies is *computed, not predicted*. Because every candidate order
is simulated, the exact chip total of each is known before committing, so
"does this play still clear the blind" is a fact rather than a forecast. Money
is therefore taken only from orders that still clear the remaining target, and
the score-maximizing order is used whenever no order clears. This is the whole
reason the rule does not consult the learned critic: an ordering that trades
chips for cash on a critic's say-so would (a) hand a control input to the least
reliable component in the system, and (b) make the transition function depend
on weights that change every update, so two identical (state, action) pairs
would no longer behave identically across a training run.

Candidate orders are score-relevant permutations only (canonical
non-xmult-before-xmult, plus Blueprint/Brainstorm placements), never all 8!
arrangements.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import TYPE_CHECKING

from pylatro.flow import play_cards

if TYPE_CHECKING:
    from pylatro.models import RunState

from .constants import MAX_JOKER_SLOTS

_MAX_ORDER_CANDIDATES = 16


class OrderObjective(StrEnum):
    """What the harness optimized the roster order for on a given play."""

    SCORE = "score"
    MONEY = "money"


@dataclass(frozen=True)
class OrderDecision:
    """Outcome of one ordering decision, including what it passed up.

    ``order`` is None when the current roster is already the chosen one, so
    callers can skip the mutation. ``dollars_gained`` and ``chips_forgone`` are
    measured against the pure score-maximizing order, which makes the trade the
    harness made visible to diagnostics and to the policy's observation.
    """

    order: tuple[int, ...] | None
    objective: OrderObjective
    chips: int
    dollars: int
    chips_forgone: int
    dollars_gained: int
    clears: bool

    @property
    def changed(self) -> bool:
        return self.order is not None


NO_ORDER_DECISION = OrderDecision(
    order=None,
    objective=OrderObjective.SCORE,
    chips=0,
    dollars=0,
    chips_forgone=0,
    dollars_gained=0,
    clears=False,
)

# Lazy singleton: the xmult/center predicates live on HeuristicAgent (they are
# also used by its shop logic) and carry no per-instance state we depend on.
_AGENT = None


def _agent():
    global _AGENT
    if _AGENT is None:
        from .heuristic import HeuristicAgent

        _AGENT = HeuristicAgent()
    return _AGENT


def _copy_joker_keys() -> frozenset[str]:
    from .heuristic import _COPY_JOKER_KEYS

    return _COPY_JOKER_KEYS


def _retrigger_joker_keys() -> frozenset[str]:
    from .heuristic import _RETRIGGER_JOKER_KEYS

    return _RETRIGGER_JOKER_KEYS


def order_candidates(state: RunState, count: int) -> list[tuple[int, ...]]:
    """Build score-relevant orders without permuting every joker."""
    agent = _agent()
    copy_keys = _copy_joker_keys()
    current = tuple(range(count))
    active_copies = [
        index
        for index, joker in enumerate(state.jokers[:count])
        if not joker.debuff and joker.center_key in copy_keys
    ]
    noncopies = [index for index in current if index not in active_copies]
    canonical_noncopies = sorted(
        noncopies,
        key=lambda index: agent._is_xmult_joker(state, state.jokers[index]),
    )

    candidates = [current]
    seen = {current}

    def add(order: tuple[int, ...]) -> bool:
        if order in seen:
            return True
        if len(candidates) >= _MAX_ORDER_CANDIDATES:
            return False
        seen.add(order)
        candidates.append(order)
        return True

    if not active_copies:
        add(tuple(canonical_noncopies))
        return candidates

    blueprints = [index for index in active_copies if state.jokers[index].center_key == "j_blueprint"]
    brainstorms = [index for index in active_copies if state.jokers[index].center_key == "j_brainstorm"]
    compatible_targets = [
        index
        for index in canonical_noncopies
        if state.data.centers.get(state.jokers[index].center_key, {}).get("blueprint_compat")
        and not state.jokers[index].debuff
    ]
    retrigger_keys = _retrigger_joker_keys()
    compatible_targets.sort(
        key=lambda index: (
            state.jokers[index].center_key not in retrigger_keys,
            not agent._is_xmult_joker(state, state.jokers[index]),
        ),
    )

    if not compatible_targets:
        add(tuple(canonical_noncopies + active_copies))
        return candidates

    blueprint_assignments = product(compatible_targets, repeat=len(blueprints)) if blueprints else [()]
    brainstorm_targets: list[int | None] = compatible_targets if brainstorms else [None]

    for blueprint_targets in blueprint_assignments:
        grouped_blueprints: dict[int, list[int]] = {target: [] for target in compatible_targets}
        for blueprint, target in zip(blueprints, blueprint_targets, strict=True):
            grouped_blueprints[target].append(blueprint)

        for brainstorm_target in brainstorm_targets:
            ordered_targets = list(canonical_noncopies)
            if brainstorm_target is not None:
                ordered_targets.remove(brainstorm_target)
                ordered_targets.insert(0, brainstorm_target)

            blocks = [[*grouped_blueprints.get(target, []), target] for target in ordered_targets]
            if brainstorms:
                if agent._is_xmult_joker(state, state.jokers[brainstorm_target]):
                    blocks.append(list(brainstorms))
                else:
                    insert_at = next(
                        (
                            index
                            for index, block in enumerate(blocks)
                            if agent._is_xmult_joker(
                                state,
                                state.jokers[block[len(block) - 1]],
                            )
                        ),
                        len(blocks),
                    )
                    blocks.insert(insert_at, list(brainstorms))

            order = tuple(index for block in blocks for index in block)
            if not add(order):
                return candidates

    return candidates


def _score_order(state: RunState, hand_indices: tuple[int, ...], order: tuple[int, ...]) -> tuple[int, int]:
    """Exactly score the selected cards with the roster in ``order``.

    Returns ``(chips, dollars)``. The probe runs on a deepcopy that excludes
    only the shared immutable game data, so the live engine RNG, hand, and
    round state are untouched.
    """
    trial = deepcopy(state, {id(state.data): state.data})
    trial.jokers[: len(order)] = [trial.jokers[index] for index in order]
    trial.joker_keys[: len(order)] = [trial.joker_keys[index] for index in order]
    score = play_cards(trial, list(hand_indices)).score
    return int(score.total), int(score.dollars)


def _ordering_can_matter(state: RunState, count: int) -> bool:
    agent = _agent()
    copy_keys = _copy_joker_keys()
    has_copy_joker = any(
        not joker.debuff and joker.center_key in copy_keys for joker in state.jokers[:count]
    )
    has_movable_xmult = any(agent._is_xmult_joker(state, joker) for joker in state.jokers[:count])
    return has_copy_joker or has_movable_xmult


def plan_joker_order(
    state: RunState,
    hand_indices: tuple[int, ...],
    *,
    remaining_target: float | None = None,
) -> OrderDecision:
    """Choose the roster order for this play and report the trade it made.

    ``remaining_target`` is the chips still needed to clear the current blind.
    When it is None the objective is always SCORE: without it, whether a
    candidate clears is unknown, and guessing is exactly the failure mode this
    design avoids.

    Money is taken only from orders that still clear, so the rule can never
    cost a clear that the score order would have achieved. Ties keep the
    current order so the roster stays visually stable.
    """
    count = min(len(state.jokers), MAX_JOKER_SLOTS)
    if count < 2 or not hand_indices or not _ordering_can_matter(state, count):
        return NO_ORDER_DECISION

    candidates = order_candidates(state, count)
    if len(candidates) < 2:
        return NO_ORDER_DECISION

    current = candidates[0]
    scored = [(order, *_score_order(state, hand_indices, order)) for order in candidates]

    # Best pure-score order. Ties resolve to the earliest candidate, and the
    # current order is always candidate 0, so a tie keeps the roster as-is.
    best_score_order, best_chips, best_score_dollars = max(scored, key=lambda row: row[1])

    objective = OrderObjective.SCORE
    chosen_order, chosen_chips, chosen_dollars = best_score_order, best_chips, best_score_dollars

    clears = remaining_target is not None and best_chips >= float(remaining_target)
    if clears:
        # Every order that still clears is safe; among those, take the cash.
        eligible = [row for row in scored if row[1] >= float(remaining_target)]
        money_order, money_chips, money_dollars = max(eligible, key=lambda row: (row[2], row[1]))
        if money_dollars > best_score_dollars:
            objective = OrderObjective.MONEY
            chosen_order, chosen_chips, chosen_dollars = money_order, money_chips, money_dollars

    return OrderDecision(
        order=None if chosen_order == current else chosen_order,
        objective=objective,
        chips=chosen_chips,
        dollars=chosen_dollars,
        chips_forgone=max(best_chips - chosen_chips, 0),
        dollars_gained=max(chosen_dollars - best_score_dollars, 0),
        clears=clears,
    )


def apply_best_joker_order(
    state: RunState,
    hand_indices: tuple[int, ...],
    *,
    remaining_target: float | None = None,
) -> OrderDecision:
    """Permute ``state.jokers``/``state.joker_keys`` into the chosen play order.

    ``joker_keys`` is an index-aligned engine cache and must be reordered
    together with the live instances.
    """
    decision = plan_joker_order(state, hand_indices, remaining_target=remaining_target)
    order = decision.order
    if order is not None:
        state.jokers[: len(order)] = [state.jokers[index] for index in order]
        state.joker_keys[: len(order)] = [state.joker_keys[index] for index in order]
    return decision


def best_joker_order(
    state: RunState,
    hand_indices: tuple[int, ...],
    *,
    remaining_target: float | None = None,
) -> tuple[int, ...] | None:
    """Return the chosen roster permutation, or None to keep the current one."""
    return plan_joker_order(state, hand_indices, remaining_target=remaining_target).order
