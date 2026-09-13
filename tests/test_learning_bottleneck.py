import json

import numpy as np
import pytest
import torch


def test_offline_candidate_coverage_excludes_invalid_slots(monkeypatch, tmp_path):
    from pylatro_agent import transfer_diagnostics as diagnostics
    from pylatro_agent.action import ActionRange

    action = int(ActionRange.PLAY_SUBSET_START)

    class Distribution:
        def log_prob(self, actions):
            return torch.zeros(1)

        def mode(self):
            return torch.tensor([action])

        def _candidate_distribution(self, *, is_play):
            return torch.zeros(1, 1), torch.zeros(1, 1, dtype=torch.bool), torch.tensor([[action]])

    monkeypatch.setattr(diagnostics, "_collate_batch", lambda *args: {"actions": torch.tensor([action])})
    monkeypatch.setattr(diagnostics, "_grammar_distribution", lambda *args: (Distribution(), {}))
    row = dict(action=action, seed=1, step=0, won=False, obs={"scalars": np.zeros(20)})
    diagnostics.offline_predictions(torch.nn.Identity(), [row], torch.device("cpu"), tmp_path)
    result = json.loads((tmp_path / "predictions.jsonl").read_text())
    assert result["candidate_covered"] == 0.0


@pytest.mark.parametrize("mixture", [0.0, 0.5, 1.0])
def test_noncandidate_probability_ceiling_and_no_candidate_fallback(mixture):
    from test_action_grammar import _blank_output

    from pylatro_agent.action import ActionType, encode_action
    from pylatro_agent.action_grammar import ActionGrammarDistribution
    from pylatro_agent.constants import HAND_CANDIDATE_START, MAX_SEQ_LEN, NUM_ACTIONS, TOKEN_DIM
    from pylatro_agent.learning_bottleneck import hand_details
    from pylatro_agent.subset_actions import subset_index

    output = _blank_output(1)
    tokens = torch.zeros(1, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    tokens[0, HAND_CANDIDATE_START, 5] = 1
    output.candidate_play_logits[0, 0] = 0
    output.hand_count_logits[0, 0, 0] = 80
    output.hand_card_logits[0, 0, 1] = 80
    candidate = encode_action(ActionType.PLAY_SUBSET, subset_index([0]))
    target = encode_action(ActionType.PLAY_SUBSET, subset_index([1]))
    mask = torch.zeros(1, NUM_ACTIONS)
    mask[0, [candidate, target]] = 1
    actions = torch.tensor([target])
    dist = ActionGrammarDistribution(output, mask, tokens=tokens, hand_ar_mixture_eps=mixture)
    assert dist.log_prob(actions).exp().item() == pytest.approx(mixture, abs=1e-6)
    detail = hand_details(dist, actions)
    assert detail["candidate_covered"].item() == 0
    assert detail["has_candidates"].item() == 1
    if mixture > 0:
        assert detail["nll_floor"].item() == pytest.approx(-np.log(mixture))
    output.candidate_play_logits.fill_(-1e8)
    fallback = ActionGrammarDistribution(output, mask, tokens=tokens, hand_ar_mixture_eps=mixture)
    assert fallback.log_prob(actions).exp().item() == pytest.approx(1.0)
    assert hand_details(fallback, actions)["nll_floor"].item() == 0


def test_hand_loss_weights_renormalize_and_leave_other_terms_unchanged(monkeypatch):
    from pylatro_agent import transfer_training as training
    from pylatro_agent.action import ActionRange
    from pylatro_agent.training.supervised import SupervisedConfig

    parameter = torch.tensor([-1.0, -3.0], requires_grad=True)

    class Distribution:
        def log_prob(self, actions):
            return parameter

        def entropy(self):
            return parameter * 0 + 0.2

    monkeypatch.setattr(training, "_grammar_distribution", lambda *args: (Distribution(), {}))
    batch = {
        "actions": torch.tensor([int(ActionRange.BLIND_PLAY), int(ActionRange.PLAY_SUBSET_START)]),
        "bc_weight": torch.tensor([2.0, 3.0]),
    }
    config = SupervisedConfig(action_entropy_coeff=0.001)
    original, _ = training.continuation_loss(None, batch, config, actor_only=True)
    weighted, metrics = training.continuation_loss(None, batch, config, actor_only=True, hand_weight=2)
    assert original.item() == pytest.approx(11 / 5 - 0.0002)
    assert weighted.item() == pytest.approx(20 / 8 - 0.0002)
    weighted.backward()
    torch.testing.assert_close(parameter.grad, torch.tensor([-2 / 8, -6 / 8]))
    assert metrics["critic_loss"].item() == 0
    for invalid in (0, -1, float("nan")):
        with pytest.raises(ValueError):
            training.continuation_loss(None, batch, config, actor_only=True, hand_weight=invalid)


def test_budget_preserves_confirmation_and_accounts_failure(tmp_path, monkeypatch):
    from pylatro_agent.learning_bottleneck import accelerator_job, atomic_json, read_json

    monkeypatch.setattr(torch.mps, "synchronize", lambda: None)
    with (
        pytest.raises(RuntimeError, match="probe failure"),
        accelerator_job(tmp_path, "failed_probe", "diagnostics", 1),
    ):
        raise RuntimeError("probe failure")
    ledger = read_json(tmp_path / "compute.json")
    assert next(job for job in ledger["jobs"] if job["name"] == "failed_probe")["status"] == "failed"
    assert ledger["elapsed_seconds"] >= 0
    ledger["jobs"].append(dict(name="prior", status="complete", stage="training", elapsed_seconds=23000))
    atomic_json(tmp_path / "compute.json", ledger)
    with pytest.raises(TimeoutError, match="20%"), accelerator_job(tmp_path, "too_large", "development", 500):
        pytest.fail("Reserve was consumed")


def test_diagnostic_budget_can_only_borrow_finished_training_surplus(tmp_path):
    from pylatro_agent.learning_bottleneck import (
        ARMS,
        atomic_json,
        reserve_diagnostic_completion,
        stage_limits,
    )

    jobs = [dict(name="diagnostics", stage="diagnostics", elapsed_seconds=5000, status="complete")]
    atomic_json(tmp_path / "compute.json", dict(jobs=jobs))
    with pytest.raises(TimeoutError, match="completed training"):
        reserve_diagnostic_completion(tmp_path, 500)
    jobs.extend(
        dict(name=f"train/{arm}/{seed}", stage="training", elapsed_seconds=1500, status="complete")
        for arm in ARMS
        for seed in (0, 1)
    )
    atomic_json(tmp_path / "compute.json", dict(jobs=jobs))
    reserve_diagnostic_completion(tmp_path, 500)
    limits = stage_limits(tmp_path)
    assert sum(limits.values()) == 32400
    assert limits["confirmation"] == 9000
    assert limits["diagnostics"] == 5605
    assert limits["training"] == 12395
    with pytest.raises(TimeoutError, match="completed training"):
        reserve_diagnostic_completion(tmp_path, 2000)


def test_evaluation_can_only_use_completed_stage_surplus(tmp_path):
    from pylatro_agent.learning_bottleneck import (
        ARMS,
        atomic_json,
        reserve_evaluation_capacity,
        stage_limits,
    )

    jobs = [dict(name="development", stage="development", elapsed_seconds=5350, status="complete")]
    atomic_json(tmp_path / "compute.json", dict(jobs=jobs))
    with pytest.raises(TimeoutError, match="completed-stage"):
        reserve_evaluation_capacity(tmp_path, "development", 100)
    jobs.extend(
        dict(name=f"train/{arm}/{seed}", stage="training", elapsed_seconds=1500, status="complete")
        for arm in ARMS
        for seed in (0, 1)
    )
    atomic_json(tmp_path / "compute.json", dict(jobs=jobs))
    reserve_evaluation_capacity(tmp_path, "development", 100)
    limits = stage_limits(tmp_path)
    assert limits["development"] == 5475
    assert limits["confirmation"] == 9000
    assert sum(limits.values()) == 32400
    # The substantial unused diagnostic allocation remains protected until its work completes.
    with pytest.raises(TimeoutError, match="completed-stage"):
        reserve_evaluation_capacity(tmp_path, "development", 1000)
    atomic_json(tmp_path / "offline_final_complete.json", {})
    reserve_evaluation_capacity(tmp_path, "development", 1000)
    assert stage_limits(tmp_path)["confirmation"] == 9000
    jobs.append(dict(name="confirmation", stage="confirmation", elapsed_seconds=8950, status="complete"))
    atomic_json(tmp_path / "compute.json", dict(jobs=jobs))
    atomic_json(tmp_path / "development_complete.json", {})
    reserve_evaluation_capacity(tmp_path, "confirmation", 100)
    assert stage_limits(tmp_path)["confirmation"] == 9075
    assert sum(stage_limits(tmp_path).values()) == 32400


def test_lightweight_learning_summary_does_not_require_candidate_details():
    from pylatro_agent.learning_bottleneck import summarize_rows

    row = dict(
        seed=1, step=2, family="play_subset", ante=1, won=False, nll=0.2, probability=0.8, accuracy=1.0, is_hand=1.0
    )
    summary = summarize_rows([row])
    assert summary["overall"]["decisions"] == 1
    assert summary["family/play_subset"]["metrics"]["accuracy"] == {"decision": 1.0, "game": 1.0}
    assert "hand/covered" not in summary


def test_observation_collisions_include_history_masks_and_scalars():
    from pylatro_agent.learning_bottleneck import collision_audit

    rows = [
        dict(
            seed=1,
            step=i,
            action=i,
            obs={"tokens": np.array([1]), "history_events": np.array([i % 2]), "action_mask": np.array([1, 1])},
        )
        for i in range(3)
    ]
    result = collision_audit(rows)
    assert result["unique_observations"] == 2
    assert len(result["conflicting"]) == 1
    assert result["conflicting"][0]["labels"] == [(1, 0, 0), (1, 2, 2)]


def test_memorization_disables_attention_dropout_as_well_as_modules():
    from types import SimpleNamespace

    from pylatro_agent.backbone import TransformerBackbone
    from pylatro_agent.learning_bottleneck import disable_dropout

    model = TransformerBackbone(n_layers=1, d_model=8, n_heads=2, d_ff=16, dropout=0.7)
    model.config = SimpleNamespace(dropout=0.7)
    disable_dropout(model)
    model.train()
    assert model.layers[0].attn.dropout == 0.0
    x = torch.randn(2, 4, 8)
    before = torch.get_rng_state()
    first = model(x)
    second = model(x)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    first.sum().backward()
    assert model.layers[0].attn.in_proj_weight.grad is not None


def test_intermediate_measurements_preserve_training_rng_and_configuration(tmp_path):
    import random
    from dataclasses import asdict

    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.checkpoint import load_checkpoint_payload, save_checkpoint
    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, reward_checkpoint_metadata
    from pylatro_agent.training.supervised import SupervisedConfig
    from pylatro_agent.transfer_training import train_continuation
    from pylatro_agent.vocab import build_vocab

    torch.set_num_threads(1)
    vocab = build_vocab(load_game_data())
    config = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.1, hand_ar_mixture_eps=0.5)
    model = BalatroAgent(config, vocab)
    source = tmp_path / "source.pt"
    save_checkpoint(
        model,
        source,
        extra={
            "agent_config": asdict(config),
            "supervised_config": asdict(SupervisedConfig()),
            **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG),
        },
    )
    env = BalatroEnv(seed=1, enable_teacher=False, deck_key="b_blue")
    obs, _ = env.reset()
    action = int(np.flatnonzero(obs["action_mask"])[0])

    class Records(list):
        def shuffled_indices(self):
            return np.random.permutation(len(self))

    records = Records(
        [dict(obs=obs, action=action, seed=1, step=i, won=False, max_ante=1, return_target=0.0) for i in range(4)]
    )

    def observer(model, update, destination):
        random.random()
        np.random.random(17)
        torch.rand(19)
        model.eval()

    checkpoints = []
    for label, callback in (("plain", None), ("observed", observer)):
        checkpoints.append(
            train_continuation(
                source,
                records,
                vocab,
                torch.device("cpu"),
                tmp_path / label,
                arm="control",
                seed=0,
                updates=2,
                batch_size=2,
                mixture=1.0,
                hand_weight=2.0,
                intermediate_steps=(0, 1, 2),
                observer=callback,
                reserved_seeds=(120000, 130000),
                data_path=tmp_path / "fixture_records",
                num_games=1,
            )
        )
    payloads = [load_checkpoint_payload(path, "cpu") for path in checkpoints]
    for key, value in payloads[0]["state_dict"].items():
        torch.testing.assert_close(value, payloads[1]["state_dict"][key], rtol=0, atol=0)
    assert payloads[1]["agent_config"]["hand_ar_mixture_eps"] == 1.0
    assert payloads[1]["supervised_config"]["hand_ar_mixture_eps"] == 1.0
    assert payloads[1]["continuation"]["hand_weight"] == 2.0
    assert payloads[1]["supervised_config"]["data_path"] == str(tmp_path / "fixture_records")
    assert payloads[1]["supervised_config"]["num_games"] == 1
    assert {120000, 130000} <= set(payloads[1]["supervised_config"]["excluded_seeds"])
    assert {120000, 130000} <= set(payloads[1]["training_reserved_seeds"])
    env.close()
