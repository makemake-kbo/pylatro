"""Use spare safe hands to develop permanent scoring jokers."""

from copy import deepcopy
from itertools import combinations

from pylatro import get_poker_hand_info
from pylatro.runtime import sell_joker
from pylatro.scoring import RANK_TO_NOMINAL

from .constants import MAX_JOKER_SLOTS, ActionRange
from .heuristic_simulation import copy_for_scoring
from .subset_actions import subset_index


def burnt_discard(agent, state, mask):
    """Spend Burnt Joker's first discard on the established hand plan."""
    if (
        state.current_round.discards_used > 0 or state.current_round.discards_left <= 0
        or state.current_round.hands_left <= 1
        or not any(j.center_key == "j_burnt" and not j.debuff for j in state.jokers)
        or (state.round_resets.ante >= 8 and state.blind_on_deck == "Boss")
    ):
        return None
    main = agent._get_main_hand_type(state)
    eligible = [
        i for i, card in enumerate(state.hand_cards)
        if not card.forced_selection and card.seal != "Blue" and card.center_key != "m_steel"
    ]
    probe = copy_for_scoring(state)
    for size in range(1, min(5, len(eligible)) + 1):
        candidates = []
        for indices in combinations(eligible, size):
            action = ActionRange.DISCARD_SUBSET_START + subset_index(indices)
            if not mask[action] or agent._quick_hand_quality(probe, [probe.hand_cards[i] for i in indices]) != main:
                continue
            cost = sum(
                100 * (state.hand_cards[i].center_key != "c_base")
                + 20 * bool(state.hand_cards[i].edition_key)
                + RANK_TO_NOMINAL.get(state.hand_cards[i].rank, 0)
                for i in indices
            )
            candidates.append((cost, action))
        if candidates:
            return min(candidates)[1]
    return None


def pad_scoring_hand(agent, state, mask, play):
    """Redraw more cards without sacrificing the selected hand's score."""
    best = (agent._estimate_hand_score(state, play), len(play), play)
    optional = [i for i in range(len(state.hand_cards)) if i not in play]
    for count in range(1, min(5 - len(play), len(optional)) + 1):
        for extra in combinations(optional, count):
            padded = tuple(sorted((*play, *extra)))
            if not mask[ActionRange.PLAY_SUBSET_START + subset_index(padded)]:
                continue
            candidate = (agent._estimate_hand_score(state, padded), len(padded), padded)
            if candidate > best:
                best = candidate
    return best[2]


def verdant_leaf_sale(agent, state, mask):
    if (
        state.blind_on_deck != "Boss" or state.blind_disabled
        or state.round_resets.blind_choices.get("Boss") != "bl_final_leaf"
    ):
        return None
    current = tuple(sorted(agent._cached_best_hand(state, state.hand_cards)))
    baseline = agent._estimate_hand_score(state, current)
    best = (baseline * 1.05, None)
    for index in range(min(len(state.jokers), MAX_JOKER_SLOTS)):
        action = ActionRange.SHOP_SELL_JOKER_START + index
        if not mask[action]:
            continue
        trial = deepcopy(state, {id(state.data): state.data})
        sell_joker(trial, index)
        indices = tuple(sorted(agent._cached_best_hand(trial, trial.hand_cards)))
        score = agent._estimate_hand_score(trial, indices)
        if score > best[0]:
            best = (score, action)
    return best[1]


def income_discard(agent, state, mask, finisher, remaining, score):
    """Collect Mail-In Rebate while keeping an already sufficient scoring hand."""
    if state.current_round.discards_left <= 0 or score < remaining * 1.2:
        return None
    if state.round_resets.ante >= 8 and state.blind_on_deck == "Boss":
        return None
    keys = {j.center_key for j in state.jokers if not j.debuff}
    if "j_mail" not in keys or keys & {
        "j_green_joker", "j_banner", "j_baron", "j_shoot_the_moon", "j_mime",
        "j_raised_fist", "j_blackboard", "j_ramen",
    }:
        return None
    _, _, _, scoring = get_poker_hand_info(state, [state.hand_cards[i] for i in finisher])
    protected = {c.reward_uid for c in scoring}
    rank = state.current_round.mail_card.get("rank")
    rank = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J", "10": "T"}.get(rank, rank)
    indices = tuple(
        i for i, card in enumerate(state.hand_cards)
        if card.rank == rank and not card.face_down and not card.debuff
        and not card.forced_selection and card.reward_uid not in protected
        and card.seal != "Blue" and card.center_key != "m_steel"
    )[:5]
    if indices and agent._discard_mask_ok(state, mask, indices):
        return ActionRange.DISCARD_SUBSET_START + subset_index(indices)
    return None


def growth_play(agent, state, mask, finisher, remaining, score):
    """Grow a scaler while holding the cards that can already finish the blind.

    This only uses the visible hand. It neither looks at future draws nor
    assumes that a sacrificed scoring hand will be replaced by another one.
    """
    if state.blind_on_deck == "Boss" or state.round_resets.ante > 6:
        return None
    spare_hand_floor = 2 if state.dollars >= 25 and state.round_resets.ante <= 5 else 4
    if state.current_round.hands_left < spare_hand_floor or score < remaining * 1.35:
        return None
    if state.dollars < 10 and state.current_round.hands_played > 0:
        return None
    keys = {j.center_key for j in state.jokers if not j.debuff}
    scalers = keys & {"j_green_joker", "j_square", "j_ride_the_bus", "j_wee"}
    if not scalers or keys & {"j_baron", "j_shoot_the_moon", "j_mime", "j_raised_fist", "j_blackboard"}:
        return None
    cards = [state.hand_cards[i] for i in finisher]
    _, _, _, scoring = get_poker_hand_info(state, cards)
    if "j_ride_the_bus" in keys and any(c.rank in {"J", "Q", "K"} for c in scoring):
        return None
    protected = {c.reward_uid for c in scoring}
    eligible = [
        i
        for i, c in enumerate(state.hand_cards)
        if c.reward_uid not in protected
        and not c.forced_selection
        and c.seal != "Blue"
        and c.center_key not in {"m_glass", "m_steel"}
    ]
    sizes = (4,) if scalers == {"j_square"} else ((1, 4) if "j_square" in scalers else (1,))
    best = None
    for size in sizes:
        for indices in combinations(eligible, size):
            action = ActionRange.PLAY_SUBSET_START + subset_index(indices)
            if not mask[action]:
                continue
            _, _, _, growing = get_poker_hand_info(state, [state.hand_cards[i] for i in indices])
            growth = int("j_green_joker" in keys)
            growth += int("j_square" in keys and size == 4)
            growth += int("j_wee" in keys and any(c.rank == "2" and not c.debuff for c in growing))
            if "j_ride_the_bus" in keys:
                if any(c.rank in {"J", "Q", "K"} for c in growing):
                    continue
                growth += 1
            if not growth:
                continue
            value = agent._estimate_hand_score(state, indices)
            if value >= remaining * 0.8:
                continue
            candidate = (growth, -size, -value, action)
            if best is None or candidate > best:
                best = candidate
    return best[-1] if best is not None else None
