"""Play candidates carry engine scores that do not depend on the run's dice."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations

import pytest

from pylatro import load_game_data
from pylatro.instances import add_joker, remove_joker
from pylatro.rng import PseudorandomState
from pylatro_agent.constants import HAND_CANDIDATE_START, MAX_PLAY_CANDIDATES, SubPhase, TokenType
from pylatro_agent.hand_candidates import generate_hand_candidates
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.play_score import exact_play_candidates, neutral_play_scores
from pylatro_agent.tokenizer import Tokenizer, sign_log
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def data():
    return load_game_data()


def _hand_runner(data, seed: int = 3, min_jokers: int = 2) -> FastRunner:
    """A teacher game advanced to a hand decision with some Jokers owned."""
    runner = FastRunner(seed, data, max_steps=600, win_ante=8, deck_key="b_blue", stake=1)
    agent = HeuristicAgent()
    while not runner.done:
        if runner.sub_phase == SubPhase.CHOOSE_ACTION and len(runner.state.jokers) >= min_jokers:
            return runner
        mask = runner.compute_mask()
        runner.step(agent.select_action(runner.state, runner.sub_phase, mask, round_score=runner.round_score))
    raise AssertionError("no hand decision reached")


def _stream(state) -> tuple:
    rng = state.pseudorandom
    return dict(rng.values), rng.last_seed, rng.draws_since_seed


def test_scores_are_the_same_under_any_live_random_stream(data) -> None:
    state = _hand_runner(data).state
    for card in state.hand_cards:
        card.center_key = "m_lucky"  # every scored card now rolls for +20 Mult
    plays = [candidate.indices for candidate in generate_hand_candidates(state)[0]]
    assert plays

    before = _stream(state)
    scores = neutral_play_scores(state, plays)
    assert _stream(state) == before  # the probe draws nothing from the run

    teacher = HeuristicAgent()
    live_probe_varies = False
    for seed in ("A", "B", "C", "D", "E", "F"):
        other = deepcopy(state)
        other.pseudorandom = PseudorandomState(seed)
        assert neutral_play_scores(other, plays) == scores
        live = [teacher._estimate_hand_score_compute(other, indices) for indices in plays]
        assert all(guaranteed <= rolled for guaranteed, rolled in zip(scores, live, strict=True))
        live_probe_varies |= live != scores
    assert live_probe_varies  # the test would be vacuous if luck never mattered here


def test_play_candidate_tokens_hold_score_and_remaining_blind_coverage(data) -> None:
    runner = _hand_runner(data)
    state = runner.state
    tokenizer = Tokenizer(vocab=build_vocab(data))
    target = float(tokenizer._blind_target(state))
    round_score = int(target * 0.6)
    raw = tokenizer.tokenize(state, SubPhase.CHOOSE_ACTION, round_score=round_score, history=runner.history)

    structural, discards = generate_hand_candidates(state)
    plays = exact_play_candidates(state, structural, MAX_PLAY_CANDIDATES)
    assert plays and plays[0][1] > 0
    for slot, (candidate, score) in enumerate(plays):
        token = raw.tokens[HAND_CANDIDATE_START + slot]
        assert raw.token_types[HAND_CANDIDATE_START + slot] == TokenType.HAND_CANDIDATE
        assert (token[0], token[2]) == (1, len(candidate.indices))
        assert token[3] == int(min(sign_log(score) * 10.0, 255.0))
        assert token[4] == int(min(score / (target - round_score) * 10.0, 255.0))
        assert tuple(int(value) - 1 for value in token[5:10] if value > 0) == candidate.indices
        assert token[10] == slot + 1
    if discards:
        assert raw.tokens[HAND_CANDIDATE_START + len(plays), 0] == 2


def _short_core_runner(data) -> FastRunner:
    """A hand decision whose best structural candidate plays fewer than five cards."""
    for seed in range(3, 40):
        runner = _hand_runner(data, seed=seed)
        structural = generate_hand_candidates(runner.state)[0]
        scores = neutral_play_scores(runner.state, [candidate.indices for candidate in structural])
        best = max(zip(scores, structural, strict=True), key=lambda pair: pair[0])
        if len(best[1].indices) <= 3 and len(runner.state.hand_cards) >= 8:
            return runner
    raise AssertionError("no short best play found")


def test_candidates_are_ordered_by_score_and_keep_the_structural_plays(data) -> None:
    state = _short_core_runner(data).state
    structural = generate_hand_candidates(state)[0]
    plays = exact_play_candidates(state, structural, MAX_PLAY_CANDIDATES)

    indices = [candidate.indices for candidate, _ in plays]
    scores = [score for _, score in plays]
    assert scores == sorted(scores, reverse=True)
    assert len(set(indices)) == len(indices)
    assert scores == neutral_play_scores(state, indices)
    if len(plays) < MAX_PLAY_CANDIDATES:
        assert {candidate.indices for candidate in structural} <= set(indices)
    assert all(candidate.kind == "play" and candidate.hand_name for candidate, _ in plays)
    assert len(exact_play_candidates(state, structural, 2)) == 2
    assert exact_play_candidates(state, structural, 2) == plays[:2]


def test_padding_that_changes_held_cards_is_offered_with_its_score(data) -> None:
    state = _short_core_runner(data).state
    for joker in list(state.jokers):
        remove_joker(state, joker)
    add_joker(state, "j_raised_fist")  # adds twice the rank of the lowest card left in hand to Mult
    structural = generate_hand_candidates(state)[0]
    cores = [candidate.indices for candidate in structural]
    structural_best = max(neutral_play_scores(state, cores))

    best, score = exact_play_candidates(state, structural, MAX_PLAY_CANDIDATES)[0]

    # Shedding the low cards alongside the scoring cards raises what Raised Fist adds.
    assert score > structural_best
    assert best.indices not in cores and len(best.indices) == 5
    padded_cores = [core for core in cores if set(core) < set(best.indices)]
    assert padded_cores
    core = min(padded_cores, key=len)
    rest = [i for i in range(len(state.hand_cards)) if i not in core]
    every_padding = [
        tuple(sorted((*core, *extra)))
        for count in range(1, 6 - len(core))
        for extra in combinations(rest, count)
    ]
    assert score == max(neutral_play_scores(state, every_padding))


def test_no_padding_is_searched_once_the_round_is_out_of_hands(data) -> None:
    state = _short_core_runner(data).state
    structural = generate_hand_candidates(state)[0]
    state.current_round.hands_left = 0
    plays = exact_play_candidates(state, structural, MAX_PLAY_CANDIDATES)
    assert {candidate.indices for candidate, _ in plays} == {candidate.indices for candidate in structural}
    assert all(score == 0 for _, score in plays)
