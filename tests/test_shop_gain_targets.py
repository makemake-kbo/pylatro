"""The teacher's shop-search gains reach training as targets, never as inputs."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.constants import (
    JOKER_MAX,
    JOKER_START,
    MAX_SEQ_LEN,
    MAX_SHOP_ITEMS,
    SHOP_START,
    SubPhase,
    TokenType,
)
from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.reward import DEFAULT_REWARD_CONFIG
from pylatro_agent.shop_gain_targets import SHOP_GAIN_SLOTS, shop_gain_loss, shop_gain_targets
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.fast_generate import _run_game_single_pass
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.training.model_generate import load_records, save_records
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def data():
    return load_game_data()


@pytest.fixture(scope="module")
def records(data):
    teacher = HeuristicAgent(shop_policy="search", grow_scalers=True)
    tokenizer = Tokenizer(vocab=build_vocab(data))
    game, _ante, _won = _run_game_single_pass(
        11, data, tokenizer, teacher, 0.997, win_ante=8, reward_config=DEFAULT_REWARD_CONFIG, deck_key="b_blue"
    )
    return game


def test_every_teacher_sale_is_the_best_positive_sell_target(data) -> None:
    runner = FastRunner(7_100_000, data, max_steps=2048, win_ante=8, deck_key="b_blue", stake=1)
    teacher = HeuristicAgent(shop_policy="search", grow_scalers=True)
    sales = 0
    while not runner.done:
        mask = runner.compute_mask()
        action = teacher.select_action(runner.state, runner.sub_phase, mask, round_score=runner.round_score)
        targets = shop_gain_targets(teacher, runner.state, runner.sub_phase, mask)
        assert (targets is None) == (runner.sub_phase != SubPhase.SHOP)
        if targets is not None and AR.SHOP_SELL_JOKER_START <= action < AR.SHOP_SELL_JOKER_START + JOKER_MAX:
            sell = np.where(np.isnan(targets[:JOKER_MAX]), -np.inf, targets[:JOKER_MAX])
            assert int(sell.argmax()) == action - AR.SHOP_SELL_JOKER_START
            assert sell.max() > 0.03
            sales += 1
        runner.step(action)
    assert sales > 0
    assert shop_gain_targets(HeuristicAgent(), runner.state, SubPhase.SHOP, mask) is None  # no portfolio search


def test_targets_are_recorded_beside_observations_and_survive_parquet(records, tmp_path) -> None:
    shops = [r for r in records if r["shop_gain_mask"] == 1.0]
    others = [r for r in records if r["shop_gain_mask"] == 0.0]
    assert shops and others
    assert all(len(r["shop_gain_targets"]) == SHOP_GAIN_SLOTS for r in records)
    assert all(all(math.isnan(v) for v in r["shop_gain_targets"]) for r in others)
    assert any(not math.isnan(v) for r in shops for v in r["shop_gain_targets"])
    assert all("shop_gain_targets" not in r["obs"] for r in records)

    path = tmp_path / "game.parquet"
    save_records(records, path, reward_config=DEFAULT_REWARD_CONFIG)
    loaded = load_records(path, reward_config=DEFAULT_REWARD_CONFIG)
    batch = _collate_batch([loaded[i] for i in range(len(loaded))], torch.device("cpu"), SupervisedConfig())
    expected = torch.tensor([r["shop_gain_targets"] for r in records], dtype=torch.float32)
    assert torch.equal(torch.isnan(batch["shop_gain_targets"]), torch.isnan(expected))
    assert torch.allclose(batch["shop_gain_targets"].nan_to_num(), expected.nan_to_num())
    assert batch["shop_gain_mask"].tolist() == [r["shop_gain_mask"] for r in records]


def test_records_without_targets_train_nothing_in_the_gain_head(records) -> None:
    bare = [{k: v for k, v in r.items() if not k.startswith("shop_gain")} for r in records[:8]]
    batch = _collate_batch(bare, torch.device("cpu"), SupervisedConfig())
    prediction = torch.randn(len(bare), SHOP_GAIN_SLOTS, 2)
    loss = shop_gain_loss(prediction, batch["shop_gain_targets"], batch["shop_gain_mask"], batch["token_types"])
    assert float(loss) == 0.0


def test_loss_scopes_to_real_tokens_of_labelled_shops() -> None:
    token_types = torch.zeros(2, MAX_SEQ_LEN, dtype=torch.long)
    token_types[:, JOKER_START : JOKER_START + 2] = int(TokenType.JOKER)
    token_types[:, SHOP_START : SHOP_START + 3] = int(TokenType.SHOP)
    targets = torch.full((2, SHOP_GAIN_SLOTS), float("nan"))
    targets[0, 1] = 0.2  # selling the second Joker gains 0.2
    targets[0, JOKER_MAX + 2] = 5.0  # clamped: 5.0 * 10 > 10
    targets[1, 0] = 0.4  # row 1 is unlabelled; must be ignored
    mask = torch.tensor([1.0, 0.0])

    prediction = torch.zeros(2, SHOP_GAIN_SLOTS, 2, requires_grad=True)
    loss = shop_gain_loss(prediction, targets, mask, token_types)
    loss.backward()
    grad = prediction.grad
    assert grad[1].abs().sum() == 0  # unlabelled row
    assert grad[0, 2:JOKER_MAX].abs().sum() == 0  # empty Joker slots
    assert grad[0, JOKER_MAX + 3 :].abs().sum() == 0  # empty shop slots
    assert (grad[0, :2, 0] != 0).all() and (grad[0, JOKER_MAX : JOKER_MAX + 3, 0] != 0).all()  # presence on real tokens
    assert grad[0, 0, 1] == 0 and grad[0, 1, 1] != 0 and grad[0, JOKER_MAX + 2, 1] != 0  # gain only where tried

    exact = torch.zeros(2, SHOP_GAIN_SLOTS, 2)
    exact[0, 1, 1], exact[0, JOKER_MAX + 2, 1] = 2.0, 10.0
    assert float(shop_gain_loss(exact, targets, mask, token_types)) == pytest.approx(math.log(2.0))


def test_agent_predicts_a_gain_for_every_joker_and_shop_slot(data, records) -> None:
    torch.manual_seed(0)
    agent = BalatroAgent(AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64), build_vocab(data)).eval()
    batch = _collate_batch(records[:4], torch.device("cpu"), SupervisedConfig())
    from pylatro_agent.training.ppo_policy import _grammar_distribution

    with torch.no_grad():
        _dist, values = _grammar_distribution(agent, batch)
    assert values["shop_gain_prediction"].shape == (4, JOKER_MAX + MAX_SHOP_ITEMS, 2)
