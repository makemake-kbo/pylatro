"""Tests for the tokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from pylatro import create_run_state, load_game_data, select_blind, start_blind
from pylatro_agent.constants import (
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    MAX_HAND_SIZE,
    MAX_SEQ_LEN,
    SCALAR_DIM,
    TOKEN_DIM,
    SubPhase,
    TokenType,
)
from pylatro_agent.tokenizer import Tokenizer, sign_log
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def vocab(game_data):
    return build_vocab(game_data)


@pytest.fixture
def run_state(game_data):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    select_blind(state, "Small")
    start_blind(state, "Small")
    return state


def test_sign_log():
    assert sign_log(0) == 0.0
    assert sign_log(1) == pytest.approx(0.6931, abs=0.01)
    assert sign_log(-1) == pytest.approx(-0.6931, abs=0.01)
    assert sign_log(100) > 0


def test_tokenize_shape(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    assert obs.tokens.shape == (MAX_SEQ_LEN, TOKEN_DIM)
    assert obs.token_types.shape == (MAX_SEQ_LEN,)
    assert obs.scalars.shape == (SCALAR_DIM,)
    assert obs.attention_mask.shape == (MAX_SEQ_LEN,)
    assert obs.selected_cards.shape == (MAX_HAND_SIZE,)


def test_tokenize_has_tokens(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    # Should have OBJ token
    assert obs.token_types[0] == TokenType.OBJ
    assert obs.attention_mask[0] == 1

    # Should have META tokens
    for i in range(1, 10):
        assert obs.token_types[i] == TokenType.META
        assert obs.attention_mask[i] == 1

    # Should have some DECK tokens (hand cards at minimum)
    deck_mask = obs.token_types[10:72] == TokenType.DECK
    assert deck_mask.sum() > 0, "Should have deck card tokens"


def test_tokenize_total_within_limit(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    active_tokens = (obs.attention_mask == 1).sum()
    assert active_tokens <= MAX_SEQ_LEN


def test_tokenize_choose_action_includes_hand_context_tokens(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    assert (obs.token_types == TokenType.HAND_LEVEL).sum() > 0
    assert (obs.token_types == TokenType.HAND_CANDIDATE).sum() > 0


def test_tokenize_blind_select_tokens(game_data, vocab):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(state, SubPhase.BLIND_SELECT)

    # Should have blind select tokens
    blind_types = obs.token_types[99:102]
    assert (blind_types == TokenType.BLIND_SELECT).sum() > 0


def test_tokenize_includes_round_score_context(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION, round_score=125)
    blind_target = tok._blind_target(run_state)

    assert obs.scalars[8] == pytest.approx(sign_log(125.0))
    assert obs.scalars[9] == pytest.approx(sign_log(max(blind_target - 125.0, 0.0)))
    assert obs.scalars[10] == pytest.approx(min(125.0 / max(blind_target, 1.0), 1.0))


def test_hand_candidates_emitted_in_choose_action(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = candidate_mask.sum()
    assert n_candidates > 0, "Hand candidates should be emitted during CHOOSE_ACTION"

    for i in range(HAND_CANDIDATE_START, HAND_CANDIDATE_START + n_candidates):
        assert obs.token_types[i] == TokenType.HAND_CANDIDATE
        assert obs.attention_mask[i] == 1


def test_hand_candidates_not_emitted_in_other_phases(run_state, vocab):
    tok = Tokenizer(vocab=vocab)

    for phase in (SubPhase.SHOP, SubPhase.BLIND_SELECT, SubPhase.BOOSTER_PACK):
        obs = tok.tokenize(run_state, phase)
        n = (obs.token_types == TokenType.HAND_CANDIDATE).sum()
        assert n == 0, f"Hand candidates should not be emitted during {phase}"


def test_hand_candidates_have_valid_kind(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    assert n_candidates > 0

    for i in range(n_candidates):
        kind = obs.tokens[HAND_CANDIDATE_START + i, 0]
        assert kind in (1, 2), f"Candidate {i} kind should be 1 (play) or 2 (discard), got {kind}"


def test_hand_candidates_have_valid_size(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    hand_size = len(run_state.hand_cards)

    for i in range(n_candidates):
        size = obs.tokens[HAND_CANDIDATE_START + i, 2]
        assert 1 <= size <= min(5, hand_size), f"Candidate {i} size {size} out of range"


def test_hand_candidates_card_indices_valid(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    hand_size = len(run_state.hand_cards)

    for i in range(n_candidates):
        size = obs.tokens[HAND_CANDIDATE_START + i, 2]
        for j in range(size):
            card_idx = obs.tokens[HAND_CANDIDATE_START + i, 5 + j] - 1
            assert 0 <= card_idx < hand_size, (
                f"Candidate {i} card slot {j} = {card_idx} out of range for hand size {hand_size}"
            )


def test_hand_candidates_rank_slot_assigned(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())

    rank_slots = set()
    for i in range(n_candidates):
        rank_slot = obs.tokens[HAND_CANDIDATE_START + i, 10]
        assert rank_slot >= 1, f"Candidate {i} rank_slot should be >= 1"
        rank_slots.add(rank_slot)
    assert len(rank_slots) == n_candidates, "Each candidate should have a unique rank_slot"


def test_hand_candidates_within_max(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    assert n_candidates <= HAND_CANDIDATE_MAX


def test_hand_candidates_play_before_discard(run_state, vocab):
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(run_state, SubPhase.CHOOSE_ACTION)

    candidate_mask = obs.token_types == TokenType.HAND_CANDIDATE
    n_candidates = int(candidate_mask.sum())
    if n_candidates < 2:
        pytest.skip("Need at least 2 candidates")

    first_kind = obs.tokens[HAND_CANDIDATE_START, 0]
    assert first_kind == 1, "First candidates should be play candidates (kind=1)"
