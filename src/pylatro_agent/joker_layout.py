"""Deterministic joker ordering applied by the environment harness.

Joker order only affects scoring, and the best order for a concrete play is
exactly computable with the engine's own scorer. The harness therefore owns
ordering: before each played hand the environment permutes the roster into the
best-scoring arrangement. The policy has no reorder actions - MOVE_JOKER was
removed from the action space - so the model spends its capacity on decisions
that are not machine-checkable.

Candidate orders are score-relevant permutations only (canonical
non-xmult-before-xmult, plus Blueprint/Brainstorm placements), mirroring the
retired heuristic reorder planner rather than permuting all 8! arrangements.
Each candidate is scored by replaying the selected cards on a copied state, so
the choice can never disagree with real scoring.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING

from pylatro.flow import play_cards

if TYPE_CHECKING:
    from pylatro.models import RunState

from .constants import MAX_JOKER_SLOTS

_MAX_ORDER_CANDIDATES = 16

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


def _score_order(state: RunState, hand_indices: tuple[int, ...], order: tuple[int, ...]) -> int:
    """Exactly score the selected cards with the roster in ``order``."""
    trial = deepcopy(state, {id(state.data): state.data})
    trial.jokers[: len(order)] = [trial.jokers[index] for index in order]
    trial.joker_keys[: len(order)] = [trial.joker_keys[index] for index in order]
    return play_cards(trial, list(hand_indices)).score.total


def best_joker_order(state: RunState, hand_indices: tuple[int, ...]) -> tuple[int, ...] | None:
    """Return the best-scoring roster permutation for this play, or None.

    None means the current order is already best (or ordering cannot matter:
    fewer than two jokers, no active xmult and no copy joker, or no cards).
    Ties keep the current order so the roster stays visually stable.
    """
    count = min(len(state.jokers), MAX_JOKER_SLOTS)
    if count < 2 or not hand_indices:
        return None
    agent = _agent()
    copy_keys = _copy_joker_keys()
    has_copy_joker = any(
        not joker.debuff and joker.center_key in copy_keys for joker in state.jokers[:count]
    )
    has_movable_xmult = any(agent._is_xmult_joker(state, joker) for joker in state.jokers[:count])
    if not has_copy_joker and not has_movable_xmult:
        return None

    candidates = order_candidates(state, count)
    if len(candidates) < 2:
        return None
    current = candidates[0]
    best_order = current
    best_score = _score_order(state, hand_indices, current)
    for order in candidates[1:]:
        score = _score_order(state, hand_indices, order)
        if score > best_score:
            best_score = score
            best_order = order
    if best_order == current:
        return None
    return best_order


def apply_best_joker_order(state: RunState, hand_indices: tuple[int, ...]) -> bool:
    """Permute ``state.jokers``/``state.joker_keys`` into the best play order.

    Returns True when the roster changed. ``joker_keys`` is an index-aligned
    engine cache and must be reordered together with the live instances.
    """
    order = best_joker_order(state, hand_indices)
    if order is None:
        return False
    state.jokers[: len(order)] = [state.jokers[index] for index in order]
    state.joker_keys[: len(order)] = [state.joker_keys[index] for index in order]
    return True
