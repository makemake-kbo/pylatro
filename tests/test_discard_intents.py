"""Discard intents (v18): the policy picks a plan and a count, code picks the cards."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from math import comb

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.action_grammar import ActionGrammarDistribution
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import HAND_CANDIDATE_START, MAX_PLAY_CANDIDATES, ActionRange, SubPhase, TokenType
from pylatro_agent.discard_plans import PlanKind, _redraw_count, discard_intents, hit_chance
from pylatro_agent.hand_candidates import generate_hand_candidates
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.play_score import exact_play_candidates
from pylatro_agent.subset_actions import HAND_SUBSETS
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


FLUSH_DRAW = [
    ("2", "Hearts"),
    ("7", "Hearts"),
    ("9", "Hearts"),
    ("K", "Hearts"),
    ("3", "Spades"),
    ("5", "Clubs"),
    ("8", "Diamonds"),
    ("J", "Spades"),
]


def _intents(state):
    return discard_intents(state, _plays(state))


def test_a_flush_intent_lists_the_other_cards_with_a_chance_per_count(data) -> None:
    state = _hand_runner(data).state
    _set_hand(state, FLUSH_DRAW)
    suit = [(candidate, odds) for candidate, kind, odds in _intents(state) if kind == PlanKind.SUIT]
    assert len(suit) == 1
    candidate, odds = suit[0]
    assert sorted(candidate.indices) == [4, 5, 6, 7] and candidate.hand_name == "Flush"
    hearts = sum(card.suit == "Hearts" for card in state.draw_pile)
    pile = len(state.draw_pile)
    for k in range(1, 5):
        assert odds[k - 1] == pytest.approx(hit_chance(hearts, pile, _redraw_count(state, k), 1))
    assert odds[4] == 0.0  # only four cards to discard
    assert list(odds[:4]) == sorted(odds[:4])  # more cards out, more draws


def test_forced_cards_lead_every_intent_and_purple_seals_lead_theirs(data) -> None:
    state = _hand_runner(data).state
    _set_hand(state, FLUSH_DRAW)
    state.hand_cards[6].forced_selection = True
    state.hand_cards[1].seal = "Purple"
    intents = _intents(state)
    assert intents and all(candidate.indices[0] == 6 for candidate, _, _ in intents)
    purple = [candidate for candidate, kind, _ in intents if kind == PlanKind.PURPLE]
    assert len(purple) == 1 and purple[0].indices[:2] == (6, 1)
    # Kept plays come last, after the plans that can complete something.
    kinds = [kind for _, kind, _ in intents]
    assert kinds.index(PlanKind.KEEP_PLAY) > kinds.index(PlanKind.PURPLE) > kinds.index(PlanKind.SUIT)


def test_blue_seals_are_never_listed(data) -> None:
    state = _hand_runner(data).state
    _set_hand(state, FLUSH_DRAW)
    state.hand_cards[5].seal = "Blue"
    assert all(5 not in candidate.indices for candidate, _, _ in _intents(state))


def _observation(data, runner):
    raw = Tokenizer(vocab=build_vocab(data)).tokenize(
        runner.state,
        SubPhase.CHOOSE_ACTION,
        round_score=0,
        history=runner.history,
        action_mask=runner.compute_mask(),
    )
    return raw


def test_intent_tokens_keep_the_discard_order_and_the_chances(data) -> None:
    runner = _hand_runner(data)
    _set_hand(runner.state, FLUSH_DRAW)
    raw = _observation(data, runner)
    intents = _intents(runner.state)
    first = HAND_CANDIDATE_START + len(_plays(runner.state))
    for slot, (candidate, kind, odds) in enumerate(intents):
        token = raw.tokens[first + slot]
        assert raw.token_types[first + slot] == TokenType.HAND_CANDIDATE and token[0] == 2
        assert tuple(int(v) - 1 for v in token[5:10] if v > 0) == candidate.indices
        assert token[12] == int(kind)
        assert list(token[13:18]) == [round(chance * 100) for chance in odds]
    assert raw.token_types[first + len(intents)] != TokenType.HAND_CANDIDATE or raw.tokens[first + len(intents), 0] != 2


def test_the_policy_spreads_each_intent_over_its_prefixes(data) -> None:
    runner = _hand_runner(data)
    _set_hand(runner.state, FLUSH_DRAW)
    raw = _observation(data, runner)
    torch.manual_seed(0)
    config = AgentConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        action_head_variant="entity_sale_subset_residual_v1",
        hand_ar_mixture_eps=0.5,
    )
    model = BalatroAgent(config, build_vocab(data)).eval()
    inputs = [torch.as_tensor(np.asarray(getattr(raw, name)))[None].long() for name in ("tokens", "token_types")] + [
        torch.as_tensor(raw.scalars)[None].float(),
        torch.as_tensor(raw.attention_mask)[None].long(),
        torch.as_tensor(raw.action_mask)[None].float(),
    ]
    with torch.no_grad():
        dist, _ = model(*inputs)
    assert isinstance(dist, ActionGrammarDistribution)
    logits, valid, actions = dist._candidate_distribution(is_play=False)
    intents = _intents(runner.state)
    first = len(_plays(runner.state))
    start = int(ActionRange.DISCARD_SUBSET_START)
    for slot, (candidate, _, _) in enumerate(intents):
        for k in range(1, 6):
            flat = (first + slot) * 5 + (k - 1)
            if k <= len(candidate.indices):
                assert bool(valid[0, flat])
                assert tuple(HAND_SUBSETS[int(actions[0, flat]) - start]) == tuple(sorted(candidate.indices[:k]))
            else:
                assert not bool(valid[0, flat])
    # p(intent) * p(k | intent): the counts of one intent share its softmax mass.
    probs = torch.softmax(logits.masked_fill(~valid, -1e8), dim=-1)[0].view(-1, 5)
    plan_probs = torch.softmax(dist.output.candidate_discard_logits[0], dim=-1)
    for slot in range(len(intents)):
        assert probs[first + slot].sum() == pytest.approx(float(plan_probs[first + slot]), rel=1e-4)
    assert not bool(valid.view(-1, 5)[:first].any())  # play slots propose no discard
    # Every legal discard keeps a finite probability through the card-by-card fallback.
    discard_logp = dist._corrected_hand_log_probs(is_play=False)[0]
    legal = dist._hand_valid_subset_mask(False)[0]
    assert torch.isfinite(discard_logp[legal]).all() and discard_logp[legal].max() > -1e7


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
    intents = _intents(state)
    straight = [(candidate, odds) for candidate, kind, odds in intents if kind == PlanKind.STRAIGHT]
    # Two windows (4-8 and 5-9) keep the same four cards; the likelier one names the intent.
    assert len(straight) == 1 and sorted(straight[0][0].indices) == [4, 5, 6, 7]
    pile = state.draw_pile
    outs = max(sum(c.rank == "4" for c in pile), sum(c.rank == "9" for c in pile))
    assert straight[0][1][3] == pytest.approx(hit_chance(outs, len(pile), _redraw_count(state, 4), 1))
    pair = [candidate for candidate, kind, _ in intents if kind == PlanKind.RANK]
    assert pair and pair[0].hand_name == "Three of a Kind" and not {4, 5} & set(pair[0].indices)


def test_no_intents_without_discards_left(data) -> None:
    state = _hand_runner(data).state
    state.current_round.discards_left = 0
    assert _intents(state) == []
