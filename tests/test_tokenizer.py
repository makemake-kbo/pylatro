"""Tests for the tokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from pylatro import create_run_state, load_game_data, select_blind, start_blind
from pylatro_agent.constants import MAX_SEQ_LEN, TOKEN_DIM, SubPhase, TokenType
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
    assert obs.attention_mask.shape == (MAX_SEQ_LEN,)
    assert obs.selected_cards.shape == (12,)


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


def test_tokenize_blind_select_tokens(game_data, vocab):
    state = create_run_state("test_seed", 1, "b_red", data=game_data)
    tok = Tokenizer(vocab=vocab)
    obs = tok.tokenize(state, SubPhase.BLIND_SELECT)

    # Should have blind select tokens
    blind_types = obs.token_types[99:102]
    assert (blind_types == TokenType.BLIND_SELECT).sum() > 0
