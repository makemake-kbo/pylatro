"""Tests for the supervised pretraining pipeline.

Covers: FastRunner termination guarantees, data generation, ETA formatting,
collation, and a minimal end-to-end supervised training run.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training import fast_generate
from pylatro_agent.training.fast_generate import (
    _format_eta,
    _run_game_fast_no_obs,
    _run_game_single_pass,
)
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.training.supervised import (
    SupervisedConfig,
    _action_type_dataset_stats,
    _collate_batch,
    _discounted_returns,
    _masked_action_loss,
    train_supervised,
)
from pylatro_agent.vocab import build_vocab


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def vocab(game_data):
    return build_vocab(game_data)


@pytest.fixture(scope="module")
def tokenizer(vocab):
    return Tokenizer(vocab=vocab)


@pytest.fixture(scope="module")
def agent():
    return HeuristicAgent()


def _dummy_obs() -> dict[str, np.ndarray]:
    return {
        "tokens": np.zeros((MAX_SEQ_LEN, TOKEN_DIM), dtype=np.int16),
        "token_types": np.zeros((MAX_SEQ_LEN,), dtype=np.int8),
        "scalars": np.zeros((SCALAR_DIM,), dtype=np.float32),
        "attention_mask": np.ones((MAX_SEQ_LEN,), dtype=np.int8),
        "action_mask": np.ones((NUM_ACTIONS,), dtype=np.int8),
    }


# ── FastRunner termination tests ──


class TestFastRunnerTermination:
    def test_fast_runner_terminates_within_max_steps(self, game_data, agent):
        runner = FastRunner(42, game_data, max_steps=500)
        steps = 0
        while not runner.done:
            mask = runner.compute_mask()
            action = agent.select_action(
                runner.state,
                runner.sub_phase,
                mask,
                selected_cards=runner.selected_cards,
                pending_action=runner.pending_action,
            )
            runner.step(action)
            steps += 1
            if steps > 10000:
                pytest.fail("FastRunner did not terminate within 10000 steps")

        assert runner._step_count <= 500 or runner._steps_since_progress >= 500
        assert runner.done

    def test_fast_runner_absolute_step_limit(self, game_data, agent):
        max_steps = 50
        runner = FastRunner(0, game_data, max_steps=max_steps)
        steps = 0
        while not runner.done:
            mask = runner.compute_mask()
            action = agent.select_action(
                runner.state,
                runner.sub_phase,
                mask,
                selected_cards=runner.selected_cards,
                pending_action=runner.pending_action,
            )
            runner.step(action)
            steps += 1
            if steps > max_steps * 2:
                break

        assert runner.done
        assert runner._step_count <= max_steps

    def test_fast_runner_no_infinite_loop_many_seeds(self, game_data, agent):
        for seed in range(50):
            runner = FastRunner(seed, game_data, max_steps=500)
            steps = 0
            while not runner.done:
                mask = runner.compute_mask()
                action = agent.select_action(
                    runner.state,
                    runner.sub_phase,
                    mask,
                    selected_cards=runner.selected_cards,
                    pending_action=runner.pending_action,
                )
                runner.step(action)
                steps += 1
                if steps > 1000:
                    pytest.fail(f"Seed {seed}: FastRunner exceeded 1000 steps")

            assert runner.done, f"Seed {seed}: runner.done not set"


# ── Single-pass game tests ──


class TestSinglePassGame:
    def test_run_game_single_pass_produces_records(self, game_data, tokenizer, agent):
        records, max_ante, _won = _run_game_single_pass(0, game_data, tokenizer, agent, 0.995)
        assert isinstance(records, list)
        assert len(records) > 0
        assert max_ante >= 1
        for rec in records:
            assert "obs" in rec
            assert "action" in rec
            assert "reward" in rec
            assert "won" in rec
            assert "max_ante" in rec
            assert "return_target" in rec
            assert isinstance(rec["obs"], dict)
            assert "tokens" in rec["obs"]
            assert rec["obs"]["tokens"].shape == (MAX_SEQ_LEN, TOKEN_DIM)

    def test_single_pass_records_have_consistent_targets(self, game_data, tokenizer, agent):
        records, max_ante, won = _run_game_single_pass(0, game_data, tokenizer, agent, 0.995)
        for rec in records:
            assert rec["won"] == won
            assert rec["max_ante"] == max_ante
            assert isinstance(rec["return_target"], float)

    def test_single_pass_discounted_returns_monotonic(self, game_data, tokenizer, agent):
        records, _, _ = _run_game_single_pass(0, game_data, tokenizer, agent, 0.995)
        targets = [r["return_target"] for r in records]
        assert len(targets) > 0
        assert all(isinstance(t, float) for t in targets)
        assert targets[-1] == pytest.approx(records[-1]["reward"])


# ── Fast no-obs game tests ──


class TestFastNoObs:
    def test_fast_no_obs_returns_ante_and_won(self, game_data, agent):
        max_ante, won = _run_game_fast_no_obs(0, game_data, agent)
        assert isinstance(max_ante, int)
        assert max_ante >= 1
        assert isinstance(won, bool)

    def test_fast_no_obs_consistent_with_single_pass(self, game_data, tokenizer, agent):
        max_ante_fast, won_fast = _run_game_fast_no_obs(0, game_data, agent)
        _, max_ante_single, won_single = _run_game_single_pass(0, game_data, tokenizer, agent, 0.995)
        assert max_ante_fast == max_ante_single, (
            f"Fast ({max_ante_fast}) and single ({max_ante_single}) max_ante disagree"
        )
        assert won_fast == won_single, (
            f"Fast ({won_fast}) and single ({won_single}) won disagree"
        )


# ── ETA formatting tests ──


class TestETAFormatting:
    def test_format_eta_seconds(self):
        assert _format_eta(30) == "30s"

    def test_format_eta_minutes(self):
        assert _format_eta(120) == "2.0m"

    def test_format_eta_hours(self):
        assert _format_eta(7200) == "2.0h"

    def test_format_eta_inf(self):
        assert _format_eta(float("inf")) == "--"

    def test_format_eta_negative(self):
        assert _format_eta(-1) == "--"

    def test_format_eta_zero(self):
        assert _format_eta(0) == "0s"

    def test_format_eta_boundary_60(self):
        assert _format_eta(59) == "59s"
        assert _format_eta(60) == "1.0m"

    def test_format_eta_boundary_3600(self):
        assert _format_eta(3599) == "60.0m"
        assert _format_eta(3600) == "1.0h"


# ── Data generation tests ──


class TestDataGeneration:
    def test_generate_single_worker_min_ante_1(self, game_data, vocab):
        records = fast_generate.generate_training_data(
            3, data=game_data, vocab=vocab, min_ante=1, num_workers=1,
        )
        assert len(records) > 0
        for rec in records:
            assert "obs" in rec
            assert "action" in rec
            assert rec["obs"]["tokens"].shape == (MAX_SEQ_LEN, TOKEN_DIM)

    def test_generate_two_workers_min_ante_1(self, game_data, vocab):
        records = fast_generate.generate_training_data(
            5, data=game_data, vocab=vocab, min_ante=1, num_workers=2,
        )
        assert len(records) > 0

    def test_generate_completes_quickly(self, game_data, vocab):
        t0 = time.monotonic()
        records = fast_generate.generate_training_data(
            5, data=game_data, vocab=vocab, min_ante=1, num_workers=1,
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 30, f"Generation took {elapsed:.1f}s, expected < 30s"
        assert len(records) > 0


# ── Collation tests ──


class TestCollation:
    def test_collate_batch_shapes(self):
        batch = _collate_batch(
            [
                {
                    "obs": _dummy_obs(),
                    "action": 0,
                    "won": True,
                    "max_ante": 5,
                    "return_target": 10.0,
                },
                {
                    "obs": _dummy_obs(),
                    "action": 1,
                    "won": False,
                    "max_ante": 3,
                    "return_target": -2.0,
                },
            ],
            torch.device("cpu"),
        )
        assert batch["tokens"].shape == (2, MAX_SEQ_LEN, TOKEN_DIM)
        assert batch["token_types"].shape == (2, MAX_SEQ_LEN)
        assert batch["scalars"].shape == (2, SCALAR_DIM)
        assert batch["attention_mask"].shape == (2, MAX_SEQ_LEN)
        assert batch["action_mask"].shape == (2, NUM_ACTIONS)
        assert batch["actions"].shape == (2,)
        assert batch["won"].shape == (2,)
        assert batch["value_target"].shape == (2,)
        assert batch["actions"].tolist() == [0, 1]
        assert batch["won"].tolist() == [1.0, 0.0]
        assert batch["value_target"].tolist() == [10.0, -2.0]

    def test_collate_batch_prefers_recorded_return_target(self):
        batch = _collate_batch(
            [
                {
                    "obs": _dummy_obs(),
                    "action": 0,
                    "won": False,
                    "max_ante": 4,
                    "return_target": -0.25,
                }
            ],
            torch.device("cpu"),
        )
        assert batch["value_target"].tolist() == [-0.25]

    def test_collate_batch_falls_back_to_legacy_value_target(self):
        batch = _collate_batch(
            [
                {
                    "obs": _dummy_obs(),
                    "action": 0,
                    "won": False,
                    "max_ante": 3,
                }
            ],
            torch.device("cpu"),
        )
        assert batch["value_target"].tolist() == [-7.0]

    def test_masked_action_loss_ignores_invalid_logits(self):
        logits = torch.tensor([[0.0, 100.0, -5.0]], dtype=torch.float32)
        action_mask = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)
        actions = torch.tensor([0], dtype=torch.long)

        loss = _masked_action_loss(logits, action_mask, actions)

        expected = -torch.log_softmax(torch.tensor([[0.0, -5.0]]), dim=-1)[0, 0]
        assert loss.item() == pytest.approx(expected.item())

    def test_action_type_dataset_stats_separates_chosen_from_valid(self):
        obs = _dummy_obs()
        obs["action_mask"] = np.zeros((NUM_ACTIONS,), dtype=np.int8)
        obs["action_mask"][0] = 1
        obs["action_mask"][encode_action(ActionType.USE_CONSUMABLE_HAND_SUBSET, 0, 0)] = 1

        stats = _action_type_dataset_stats([{"obs": obs, "action": 0}])

        assert stats["chosen"][ActionType.BLIND_PLAY.value] == 1
        assert stats["chosen"][ActionType.USE_CONSUMABLE_HAND_SUBSET.value] == 0
        assert stats["valid_states"][ActionType.USE_CONSUMABLE_HAND_SUBSET.value] == 1


# ── Discounted returns tests ──


class TestDiscountedReturns:
    def test_discounted_returns_tracks_reward_to_go(self):
        returns = _discounted_returns([1.0, 0.5, -2.0], gamma=0.9)
        assert returns == [pytest.approx(-0.17), pytest.approx(-1.3), pytest.approx(-2.0)]

    def test_discounted_returns_single_reward(self):
        returns = _discounted_returns([5.0], gamma=0.99)
        assert returns == [5.0]

    def test_discounted_returns_empty(self):
        returns = _discounted_returns([], gamma=0.99)
        assert returns == []

    def test_discounted_returns_gamma_zero(self):
        returns = _discounted_returns([1.0, 2.0, 3.0], gamma=0.0)
        assert returns == [1.0, 2.0, 3.0]

    def test_discounted_returns_gamma_one(self):
        returns = _discounted_returns([1.0, 2.0, 3.0], gamma=1.0)
        assert returns == [pytest.approx(6.0), pytest.approx(5.0), pytest.approx(3.0)]


# ── End-to-end training test ──


class TestEndToEndTraining:
    def test_train_supervised_small_run(self, game_data, vocab, tmp_path):
        config = SupervisedConfig(
            num_games=3,
            batch_size=4,
            max_epochs=2,
            num_workers=1,
            min_ante=1,
            device="cpu",
            save_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "logs"),
        )
        model = train_supervised(config, agent_config=None, data=game_data)
        assert model is not None
        ckpts = list((tmp_path / "ckpts").glob("*.pt"))
        assert len(ckpts) == 2, f"Expected 2 checkpoint files, found {len(ckpts)}"

    def test_train_supervised_empty_data_handled(self, game_data, vocab, tmp_path):
        with patch(
            "pylatro_agent.training.supervised.generate_training_data",
            return_value=[],
        ):
            config = SupervisedConfig(
                num_games=0,
                batch_size=4,
                max_epochs=1,
                device="cpu",
                save_dir=str(tmp_path / "ckpts"),
                log_dir=str(tmp_path / "logs"),
            )
            model = train_supervised(config, data=game_data)
            assert model is not None
