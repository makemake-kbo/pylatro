"""Engine-exact play scores that never read the run's own random stream.

The engine is the only complete scoring model: conditional Jokers, boss
blinds, enhancements, held-card effects and retriggers all change what a
hand banks. ``neutral_play_scores`` asks it directly, with the Joker order the
runner would apply before the play.

Scoring a copy that carries the live random stream would report the roll the
real play is about to get, which no player can know. The probe therefore
reports the guaranteed score: chance-gated effects (Lucky cards, Bloodstone,
Space Joker, ...) do not fire, so a hand that "clears" here clears whatever
the dice do. A single lucky hit can multiply a small hand twenty-fold; a
sampled roll would make the number that swing instead of a property of the
hand. Misprint has no chance gate, so its mult is pinned to the middle of its
range. The probe also carries its own stream, so nothing here can depend on
the run's.

``exact_play_candidates`` builds the list of plays the model is shown from
those scores. The structural generator proposes scoring cores (the pair, the
flush). What a play banks also depends on the cards that leave the hand with
it: extra cards cost nothing when the core already scores, redraw more, and
change every held-card effect (Blackboard, Raised Fist, Steel, Baron). So the
best few cores are also offered with the extra cards that bank the most.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

from pylatro.flow import play_cards
from pylatro.rng import PseudorandomState
from pylatro.scoring import get_poker_hand_info

from .constants import MAX_HAND_SIZE
from .hand_candidates import HandCandidate
from .heuristic_simulation import copy_for_scoring
from .joker_layout import NO_ORDER_DECISION, plan_joker_order

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pylatro.models import RunState

# How many of the best-scoring cores are also offered padded. Three puts the
# best legal play in the list for 94% of hand decisions (72% unpadded) at
# roughly 70 extra engine evaluations.
PADDED_CORES = 3
MAX_PLAY_SIZE = 5


def neutral_play_scores(state: RunState, plays: Iterable[tuple[int, ...]]) -> list[int]:
    """Chips each play in ``plays`` (hand-card indices) is certain to bank right now."""
    plays = [tuple(indices) for indices in plays]
    if state.current_round.hands_left <= 0:
        # Terminal hand observations still list candidates; none can be played.
        return [0] * len(plays)
    probe = copy_for_scoring(state)
    probe.pseudorandom = PseudorandomState("probe")
    probe.probabilities = {**probe.probabilities, "normal": 0}
    for joker in probe.jokers:
        if joker.center_key == "j_misprint" and isinstance(joker.extra, dict):
            middle = (int(joker.extra.get("min", 0)) + int(joker.extra.get("max", 0))) // 2
            joker.extra = {**joker.extra, "min": middle, "max": middle}
    scores = []
    for indices in plays:
        if not indices:
            scores.append(0)
            continue
        decision = plan_joker_order(probe, indices)
        if decision is not NO_ORDER_DECISION:
            scores.append(int(decision.chips))
        else:
            scores.append(int(play_cards(copy_for_scoring(probe), list(indices)).score.total))
    return scores


def exact_play_candidates(
    state: RunState, candidates: Sequence[HandCandidate], limit: int
) -> list[tuple[HandCandidate, int]]:
    """Up to ``limit`` plays with their guaranteed scores, best first.

    ``candidates`` are the structural proposals. Each of the ``PADDED_CORES``
    best is extended with every choice of extra cards, and the extension that
    banks the most is added (ties go to the play that sheds more cards). The
    result is ordered by score, so the first entry is the strongest play found.
    """
    plays = [tuple(candidate.indices) for candidate in candidates]
    scores = dict(zip(plays, neutral_play_scores(state, plays), strict=True))
    by_indices = dict(zip(plays, candidates, strict=True))

    def strength(indices: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        return scores[indices], len(indices), indices

    if state.current_round.hands_left > 0:
        hand_size = min(len(state.hand_cards), MAX_HAND_SIZE)  # later cards have no play action
        cores = [
            indices for indices in sorted(by_indices, key=strength, reverse=True)[:PADDED_CORES]
            if len(indices) < MAX_PLAY_SIZE
        ]
        extensions = [
            [
                tuple(sorted((*core, *extra)))
                for count in range(1, min(MAX_PLAY_SIZE - len(core), hand_size - len(core)) + 1)
                for extra in combinations([i for i in range(hand_size) if i not in core], count)
            ]
            for core in cores
        ]
        unscored = sorted({indices for group in extensions for indices in group if indices not in scores})
        scores.update(zip(unscored, neutral_play_scores(state, unscored), strict=True))
        for core, group in zip(cores, extensions, strict=True):
            best = max((core, *group), key=strength)
            if best not in by_indices:
                hand_name = get_poker_hand_info(state, [state.hand_cards[i] for i in best])[0]
                by_indices[best] = HandCandidate(
                    kind="play", indices=best, hand_name=hand_name, estimated_score=float(scores[best])
                )
    ranked = sorted(by_indices, key=strength, reverse=True)[:limit]
    return [(by_indices[indices], scores[indices]) for indices in ranked]
