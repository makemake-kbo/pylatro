"""Embedding contracts that keep entity identity and card references learnable."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.constants import (
    DECK_MAX,
    DECK_START,
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    HAND_LEVEL_MAX,
    HAND_LEVEL_START,
    JOKER_MAX,
    JOKER_START,
    MAX_SEQ_LEN,
    SCALAR_DIM,
    SHOP_START,
    TOKEN_DIM,
    SubPhase,
    TokenType,
)
from pylatro_agent.embeddings import ContentEmbeddingLayer
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_generate import _build_obs
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.vocab import build_vocab

D_MODEL = 64


@pytest.fixture(scope="module")
def vocab():
    return build_vocab(load_game_data())


@pytest.fixture()
def layer(vocab) -> ContentEmbeddingLayer:
    torch.manual_seed(0)
    return ContentEmbeddingLayer(vocab, D_MODEL)


def test_hand_cards_and_candidates_share_one_slot_row(layer: ContentEmbeddingLayer) -> None:
    """A candidate referencing hand card i must carry exactly that card's slot vector."""
    slot = 4
    cards = torch.zeros(1, 2, TOKEN_DIM, dtype=torch.long)
    cards[0, 0, 11] = slot  # hand card (location 0) in tokenizer slot 4
    plain = cards.clone()
    plain[0, 0, 11] = 0
    with torch.no_grad():
        # deck_emb applies a final linear map; compare inside it by differencing.
        delta = layer.deck_emb(cards)[0, 0] - layer.deck_emb(plain)[0, 0]
        expected = layer.deck_emb.proj.weight @ (layer.hand_slot_emb.weight[slot + 1] - layer.hand_slot_emb.weight[1])
    assert torch.allclose(delta, expected, atol=1e-5)

    candidate = torch.zeros(1, 1, TOKEN_DIM, dtype=torch.long)
    candidate[0, 0, 5] = slot + 1  # tokenizer writes 1-based references, 0 = no card
    empty = torch.zeros(1, 1, TOKEN_DIM, dtype=torch.long)
    with torch.no_grad():
        reference = layer.hand_candidate_emb(candidate)[0, 0] - layer.hand_candidate_emb(empty)[0, 0]
    assert torch.allclose(reference, layer.hand_slot_emb.weight[slot + 1], atol=1e-5)


def test_cards_outside_the_hand_never_use_a_hand_slot_row(layer: ContentEmbeddingLayer) -> None:
    cards = torch.zeros(1, 3, TOKEN_DIM, dtype=torch.long)
    cards[0, :, 5] = torch.tensor([1, 2, 3])  # draw pile, discard pile, history
    moved = cards.clone()
    moved[0, :, 11] = 7  # a stale slot value must not matter away from the hand
    with torch.no_grad():
        assert torch.equal(layer.deck_emb(cards), layer.deck_emb(moved))


def test_candidate_padding_does_not_contribute_a_slot_vector(layer: ContentEmbeddingLayer) -> None:
    with torch.no_grad():
        layer.hand_slot_emb.weight[0].fill_(100.0)
        one_card = torch.zeros(1, 1, TOKEN_DIM, dtype=torch.long)
        one_card[0, 0, 5] = 3
        out = layer.hand_candidate_emb(one_card)
    assert out.abs().max() < 50.0


def test_shop_items_reuse_owned_identity_tables(layer: ContentEmbeddingLayer) -> None:
    shop = layer.shop_emb
    assert shop.joker_id_emb is layer.joker_emb.id_emb
    assert shop.consumable_id_emb is layer.consumable_emb.id_emb
    assert shop.voucher_id_emb is layer.voucher_emb.emb

    item_id = 5
    tokens = torch.zeros(1, 4, TOKEN_DIM, dtype=torch.long)
    tokens[0, :, 0] = item_id
    tokens[0, :, 1] = torch.tensor([1, 2, 3, 4])  # Joker, consumable, voucher, booster
    with torch.no_grad():
        identity = shop._item_identity(tokens[:, :, 0], tokens[:, :, 1])[0]
    assert torch.equal(identity[0], layer.joker_emb.id_emb.weight[item_id])
    assert torch.equal(identity[1], layer.consumable_emb.id_emb.weight[item_id])
    assert torch.equal(identity[2], layer.voucher_emb.emb.weight[item_id])
    assert torch.equal(identity[3], shop.item_emb.weight[item_id])
    # The same id in different spaces must no longer be one shared vector.
    assert not torch.equal(identity[0], identity[1])


def test_shared_tables_are_single_parameters(layer: ContentEmbeddingLayer) -> None:
    """Tied tables appear under several names but must be optimized once."""
    parameters = list(layer.parameters())
    assert len({id(p) for p in parameters}) == len(parameters)
    assert sum(p is layer.joker_emb.id_emb.weight for p in parameters) == 1


def test_no_token_type_dominates_at_initialization(vocab, layer: ContentEmbeddingLayer) -> None:
    """Raw-magnitude numeric fields must not swamp categorical identity.

    Uses real mid-game hand states: scaled Jokers, levelled hands and scored
    candidates are exactly where unnormalized projections used to be 10-50x
    larger than every other token.
    """
    data = load_game_data()
    tokenizer = Tokenizer(vocab=vocab)
    from pylatro_agent.heuristic import HeuristicAgent

    observations = []
    for seed in (3, 4):
        runner = FastRunner(seed, data, max_steps=400, win_ante=8, deck_key="b_blue", stake=1)
        agent = HeuristicAgent()
        while not runner.done and len(observations) < 40 * (seed - 2):
            obs = _build_obs(runner, tokenizer)
            if runner.sub_phase == SubPhase.CHOOSE_ACTION and len(runner.state.jokers) >= 2:
                observations.append(obs)
            mask = runner.compute_mask()
            runner.step(agent.select_action(runner.state, runner.sub_phase, mask, round_score=runner.round_score))
    assert len(observations) >= 20

    tokens = torch.tensor(np.array([o["tokens"] for o in observations]), dtype=torch.long)
    token_types = torch.tensor(np.array([o["token_types"] for o in observations]), dtype=torch.long)
    scalars = torch.tensor(np.array([o["scalars"] for o in observations]), dtype=torch.float32)
    active = torch.tensor(np.array([o["attention_mask"] for o in observations])) > 0
    assert tokens.shape[1:] == (MAX_SEQ_LEN, TOKEN_DIM) and scalars.shape[1] == SCALAR_DIM
    with torch.no_grad():
        norms = layer(tokens, token_types, scalars).norm(dim=-1)

    def median_norm(start: int, count: int, token_type: TokenType) -> float:
        region = slice(start, start + count)
        mask = active[:, region] & (token_types[:, region] == int(token_type))
        assert mask.any(), token_type
        return float(norms[:, region][mask].median())

    card = median_norm(DECK_START, DECK_MAX, TokenType.DECK)
    for start, count, token_type in (
        (JOKER_START, JOKER_MAX, TokenType.JOKER),
        (HAND_LEVEL_START, HAND_LEVEL_MAX, TokenType.HAND_LEVEL),
        (HAND_CANDIDATE_START, HAND_CANDIDATE_MAX, TokenType.HAND_CANDIDATE),
    ):
        ratio = median_norm(start, count, token_type) / card
        assert 0.5 < ratio < 2.5, (token_type, ratio)
    assert SHOP_START < HAND_LEVEL_START  # layout sanity for the regions used above
