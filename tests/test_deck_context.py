import torch

from pylatro import load_game_data
from pylatro_agent.constants import DECK_SCALAR_INDEX, SCALAR_DIM, STAKE_SCALAR_INDEX
from pylatro_agent.embeddings import MetaEmbedding
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_generate import _build_obs, _deck_options
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.vocab import DECK_TO_ID, build_vocab


def test_observation_exposes_deck_and_stake():
    data = load_game_data()
    tokenizer = Tokenizer(vocab=build_vocab(data))
    for deck, stake in (("b_blue", 1), ("b_green", 3)):
        runner = FastRunner(7, data, max_steps=10, win_ante=8, deck_key=deck, stake=stake)
        scalars = _build_obs(runner, tokenizer)["scalars"]
        assert len(scalars) == SCALAR_DIM
        assert scalars[DECK_SCALAR_INDEX] == DECK_TO_ID[deck]
        assert scalars[STAKE_SCALAR_INDEX] == stake


def test_deck_and_stake_embeddings_start_as_no_op():
    meta = MetaEmbedding(16)
    tokens = torch.zeros(2, 9, 4, dtype=torch.long)
    scalars = torch.zeros(2, SCALAR_DIM)
    scalars[:, 2] = 1
    scalars[:, 22] = 8
    other = scalars.clone()
    other[:, DECK_SCALAR_INDEX] = DECK_TO_ID["b_black"]
    other[:, STAKE_SCALAR_INDEX] = 5
    assert torch.equal(meta(tokens, scalars), meta(tokens, other))


def test_deck_mix_cycles_by_seed():
    options = {"deck_key": ("b_red", "b_blue", "b_green"), "stake": 1}
    assert [_deck_options(options, seed)["deck_key"] for seed in range(4)] == ["b_red", "b_blue", "b_green", "b_red"]
    assert _deck_options({"deck_key": "b_blue"}, 5)["deck_key"] == "b_blue"
