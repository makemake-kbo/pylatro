"""Plan discards: complements of a draw plan, with the chance the redraw completes it."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from math import comb

import pytest

from pylatro import load_game_data
from pylatro_agent.constants import HAND_CANDIDATE_START, MAX_PLAY_CANDIDATES, SubPhase, TokenType
from pylatro_agent.discard_plans import PlanKind, hit_chance, plan_discards
from pylatro_agent.hand_candidates import generate_hand_candidates
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.play_score import exact_play_candidates
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def data():
    return load_game_data()


def _hand_runner(data, seed: int = 5) -> FastRunner:
    runner = FastRunner(seed, data, max_steps=600, win_ante=8, deck_key="b_red", stake=1)
    agent = HeuristicAgent()
    while not runner.done:
        if runner.sub_phase == SubPhase.CHOOSE_ACTION and runner.state.current_round.discards_left > 0:
            return runner
        runner.step(
            agent.select_action(runner.state, runner.sub_phase, runner.compute_mask(), round_score=runner.round_score)
        )
    raise AssertionError("no hand decision with discards left")


def _set_hand(state, cards: list[tuple[str, str]]) -> None:
    template = state.hand_cards[0]
    hand = []
    for rank, suit in cards:
        card = deepcopy(template)
        card.rank, card.suit = rank, suit
        card.reward_uid = template.reward_uid * 100 + len(hand)
        card.center_key, card.seal, card.edition_key = "c_base", None, None
        card.debuff = card.forced_selection = card.face_down = False
        hand.append(card)
    state.hand_cards = hand


def _plays(state):
    return [
        candidate
        for candidate, _ in exact_play_candidates(state, generate_hand_candidates(state)[0], MAX_PLAY_CANDIDATES)
    ]


def test_hit_chance_is_the_hypergeometric_tail() -> None:
    assert hit_chance(9, 40, 3, 0) == 1.0
    assert hit_chance(9, 40, 3, 4) == 0.0
    assert hit_chance(9, 40, 3, 1) == pytest.approx(1 - comb(31, 3) / comb(40, 3))
    exact = sum(comb(9, k) * comb(31, 3 - k) for k in (2, 3)) / comb(40, 3)
    assert hit_chance(9, 40, 3, 2) == pytest.approx(exact)
    # Brute force on a tiny pile: outs are cards 0..2 of six, two draws, one needed.
    hits = [len({0, 1, 2} & set(draw)) >= 1 for draw in combinations(range(6), 2)]
    assert hit_chance(3, 6, 2, 1) == pytest.approx(sum(hits) / len(hits))


def test_a_four_card_flush_discards_the_rest_with_the_suit_count_of_the_pile(data) -> None:
    state = _hand_runner(data).state
    _set_hand(
        state,
        [
            ("2", "Hearts"),
            ("7", "Hearts"),
            ("9", "Hearts"),
            ("K", "Hearts"),
            ("3", "Spades"),
            ("5", "Clubs"),
            ("8", "Diamonds"),
            ("J", "Spades"),
        ],
    )
    plans = plan_discards(state, _plays(state))
    flush = [(candidate, chance) for candidate, kind, chance in plans if kind == PlanKind.SUIT]
    assert len(flush) == 1
    candidate, chance = flush[0]
    assert candidate.indices == (4, 5, 6, 7) and candidate.hand_name == "Flush"
    hearts = sum(card.suit == "Hearts" for card in state.draw_pile)
    assert chance == pytest.approx(hit_chance(hearts, len(state.draw_pile), 4, 1))
    # Plans that complete something come before kept plays, most likely first.
    kinds = [kind for _, kind, _ in plans]
    assert kinds.index(PlanKind.KEEP_PLAY) > kinds.index(PlanKind.SUIT)
    chances = [chance for _, kind, chance in plans if kind != PlanKind.KEEP_PLAY]
    assert chances == sorted(chances, reverse=True)


def test_an_open_straight_window_counts_the_missing_rank(data) -> None:
    state = _hand_runner(data).state
    _set_hand(
        state,
        [
            ("5", "Hearts"),
            ("6", "Spades"),
            ("7", "Clubs"),
            ("8", "Diamonds"),
            ("2", "Spades"),
            ("2", "Clubs"),
            ("K", "Diamonds"),
            ("Q", "Spades"),
        ],
    )
    plans = {
        candidate.indices: (candidate, kind, chance) for candidate, kind, chance in plan_discards(state, _plays(state))
    }
    straight = [entry for entry in plans.values() if entry[1] == PlanKind.STRAIGHT]
    assert {entry[0].indices for entry in straight} == {(4, 5, 6, 7)}
    # Two windows (4-8 and 5-9) keep the same four cards; the likelier one names the discard.
    pile = state.draw_pile
    fours, nines = sum(c.rank == "4" for c in pile), sum(c.rank == "9" for c in pile)
    assert straight[0][2] == pytest.approx(hit_chance(max(fours, nines), len(pile), 4, 1))
    pair = [entry for entry in plans.values() if entry[1] == PlanKind.RANK]
    assert pair and pair[0][0].hand_name == "Three of a Kind" and 4 not in pair[0][0].indices


def test_long_complements_drop_the_most_useful_cards_and_keep_blue_seals(data) -> None:
    state = _hand_runner(data).state
    _set_hand(
        state,
        [
            ("A", "Hearts"),
            ("A", "Spades"),
            ("2", "Clubs"),
            ("4", "Diamonds"),
            ("6", "Clubs"),
            ("8", "Diamonds"),
            ("T", "Clubs"),
            ("Q", "Diamonds"),
        ],
    )
    state.hand_cards[2].seal = "Blue"
    for candidate, kind, _ in plan_discards(state, _plays(state)):
        assert 1 <= len(candidate.indices) <= 5 and 2 not in candidate.indices
        if kind == PlanKind.RANK:
            assert candidate.indices == (3, 4, 5, 6, 7)


def test_no_plan_discards_without_discards_left(data) -> None:
    state = _hand_runner(data).state
    state.current_round.discards_left = 0
    assert plan_discards(state, _plays(state)) == []


def test_plan_discards_lead_the_discard_tokens(data) -> None:
    runner = _hand_runner(data)
    state = runner.state
    raw = Tokenizer(vocab=build_vocab(data)).tokenize(
        state, SubPhase.CHOOSE_ACTION, round_score=0, history=runner.history
    )
    plays = _plays(state)
    plans = plan_discards(state, plays)
    assert plans
    first = HAND_CANDIDATE_START + len(plays)
    for slot, (candidate, kind, chance) in enumerate(plans[:16]):
        token = raw.tokens[first + slot]
        assert raw.token_types[first + slot] == TokenType.HAND_CANDIDATE
        assert token[0] == 2
        assert tuple(int(v) - 1 for v in token[5:10] if v > 0) == candidate.indices
        assert (token[12], token[13]) == (int(kind), round(chance * 100))
    assert all(raw.tokens[HAND_CANDIDATE_START + slot, 12] == 0 for slot in range(len(plays)))
