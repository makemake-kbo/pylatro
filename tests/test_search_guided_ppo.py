"""Public collection, independent labels, shared optimizer, and study gates."""

from __future__ import annotations

import copy
import os
import pickle
import time
from dataclasses import asdict, fields

import numpy as np
import pytest
import test_complete_cohort as fixtures
import torch
from test_public_hybrid import hand_state, private_mutation, snapshot

from pylatro_agent.checkpoint import capture_rng_states, load_checkpoint_payload, restore_rng_states
from pylatro_agent.constants import ActionRange as AR
from pylatro_agent.constants import SubPhase
from pylatro_agent.hybrid import HybridPolicy, Plan
from pylatro_agent.public_runtime import PublicHeuristic, PublicRunner
from pylatro_agent.public_state import PublicGameSnapshot, observation
from pylatro_agent.search_ppo_study import (
    assert_separation,
    check_deadline,
    immutable,
    metadata_integers,
    metadata_name,
    paired_claim,
    read,
    save_torch,
    seed_panels,
    select_duration,
    start_clock,
)
from pylatro_agent.tokenizer import Tokenizer
from pylatro_agent.training.cohort import CohortBuffer
from pylatro_agent.training.evaluation_envs import EvaluationEnvs
from pylatro_agent.training.ppo_observations import _obs_dicts_to_batch
from pylatro_agent.training.ppo_optimization import _make_policy_optimizer, _run_ppo_update
from pylatro_agent.training.ppo_policy import _grammar_distribution
from pylatro_agent.training.public_training import PublicTrainingEnv, observation_dict, selection_key
from pylatro_agent.training.search_guidance import RECIPE, SearchGuidance, filter_correction, teach
from pylatro_agent.training.search_ppo import initial_checkpoint, make_config, validate_resume
from pylatro_agent.vocab import build_vocab
from tools.train_outcome_ppo import same_state_tree

cpu_threads = fixtures.cpu_threads
game_data = fixtures.game_data
observation_fixture = fixtures.observation


def test_public_adapter_actions_observations_rewards_and_selection(game_data):
    seed = 931071
    tokenizer = Tokenizer(build_vocab(game_data))
    env = PublicTrainingEnv(seed, game_data, build_vocab(game_data), capture_states=True)
    obs, _ = env.reset(seed=seed)
    player = PublicRunner(seed, game_data, deck_key="b_blue", stake=1, raise_errors=True)
    heuristic = PublicHeuristic(game_data)
    expected = {"early": [], "late": []}
    rewards = []
    for decision in range(2048):
        public = player.snapshot()
        player_obs = observation(public, game_data, tokenizer)
        for f in fields(player_obs):
            np.testing.assert_array_equal(obs[f.name], getattr(player_obs, f.name))
        assert env.runner.snapshot() == public
        ante = public.values["round_resets"]["ante"]
        if public.sub_phase == SubPhase.CHOOSE_ACTION:
            expected["early" if ante <= 3 else "late"].append(decision)
        action = heuristic(public)
        player.step(action)
        obs, reward, terminal, truncated, info = env.step(action)
        rewards.append(reward)
        assert not truncated
        assert env.runner.snapshot() == player.snapshot()
        if terminal:
            assert sum(rewards) == 10 * info["won"]
            break
    else:
        pytest.fail("Expected whole game completion")
    selected = env.selected_public_states()
    selected_steps = {r["decision"] for r in selected}
    assert selected_steps == {
        s for group in expected.values() for s in sorted(group, key=lambda d: selection_key(seed, d))[:2]
    }
    encoded = repr(selected)
    for private in ("reward_uid", "pseudorandom", "draw_pile", "PRIVATE_"):
        assert private not in encoded


@pytest.mark.parametrize("boss", ["bl_hook", "bl_final_acorn", "bl_psychic", "bl_mark", "bl_water"])
def test_adapter_concealment_and_boss_observation_equivalence(boss, game_data):
    state = hand_state(boss)
    state.hand_cards[0].face_down = True
    public = snapshot(state)
    hidden = private_mutation(state)
    tokenizer = Tokenizer(build_vocab(game_data))
    a = observation_dict(public, game_data, tokenizer)
    b = observation_dict(snapshot(hidden), game_data, tokenizer)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    runner = PublicRunner.from_snapshot(public, game_data)
    np.testing.assert_array_equal(a["action_mask"], runner.compute_mask())


def test_public_worker_adapter_retains_states_and_exhaustion_is_censored(game_data):
    envs = EvaluationEnvs(
        0,
        environment_class=PublicTrainingEnv,
        data=game_data,
        vocab=build_vocab(game_data),
        capture_states=True,
        max_steps=1,
    )
    try:
        obs, _ = envs.reset(67134)
        action = int(np.flatnonzero(obs["action_mask"])[0])
        _, reward, terminal, truncated, info, _ = envs.step_training([(67134, action)])[67134]
        assert not terminal and truncated and info["stalled"] and reward == 0
        assert envs.public_states() == []
    finally:
        envs.close()


def batches(values):
    return [dict(complete=True, samples=16, values=copy.deepcopy(values)) for _ in range(2)]


def test_filter_rejects_incomplete_tied_illegal_unsupported_and_disagreeing():
    mask = np.ones(5)
    values = [[0.5, 4, 1, 0], [0.625, 3, 1, 0], [0.55, 10, 2, 0]]
    target, reason = filter_correction(0, [0, 1, 2], batches(values), mask)
    assert target == {"target": 1, "weight": 0.125} and reason == "accepted"
    changed = batches(values)
    changed[1]["samples"] = 15
    assert filter_correction(0, [0, 1, 2], changed, mask)[1] == "incomplete"
    changed = batches(values)
    changed[1]["values"][2] = [0.75, 1, 1, 0]
    assert filter_correction(0, [0, 1, 2], changed, mask)[1] == "disagreement"
    tied = batches([values[0], values[1], values[1]])
    assert filter_correction(0, [0, 1, 2], tied, mask)[1] == "tied"
    mask[1] = 0
    assert filter_correction(0, [0, 1, 2], batches(values), mask)[1] == "illegal"
    assert filter_correction(0, [0, 1, 1], batches(values), mask)[1] == "unsupported"
    weak = batches([values[0], [0.5625, 999, 999, 0]])
    assert filter_correction(0, [0, 2], weak, np.ones(5))[1] == "small_clear_gain"


def label_record(game_data):
    public = snapshot(hand_state())
    mask = observation(public, game_data, Tokenizer(build_vocab(game_data))).action_mask
    actions = [int(a) for a in np.flatnonzero(mask) if AR.PLAY_SUBSET_START <= a <= AR.DISCARD_SUBSET_END][:2]
    return dict(
        seed=12,
        decision=1,
        ante=1,
        snapshot=public.values,
        proposal=actions[0],
        target=actions[1],
        weight=0.125,
        accepted=True,
    ), actions


def test_two_batches_use_identical_candidates_disjoint_worlds_and_no_rng(monkeypatch, game_data):
    row, actions = label_record(game_data)
    generated, trials = [], []

    def plans(self, state, proposal):
        generated.append(proposal)
        return [Plan((a,)) for a in actions]

    def trial(self, state, plan, sample, hand):
        trials.append((sample, plan.actions[0]))
        return ((int(plan.actions[0] == actions[1]), 0, 0, 0), True)

    monkeypatch.setattr(HybridPolicy, "_hand_plans", plans)
    monkeypatch.setattr(HybridPolicy, "_trial", trial)
    before = capture_rng_states()
    result = teach(row, game_data)
    after = capture_rng_states()
    assert same_state_tree(before, after)
    assert result["accepted"] and result["weight"] == 1
    assert generated == [actions[0]]
    assert trials == [
        (sample, action) for start in (0, 10000) for sample in range(start, start + 16) for action in actions
    ]


def test_labels_route_gradients_to_actor_and_encoder_but_not_critic(game_data):
    model = fixtures.tiny_model(game_data).eval()
    config = fixtures.cohort_config()
    row, _ = label_record(game_data)
    guide = SearchGuidance([row], game_data, Tokenizer(build_vocab(game_data)), np.random.RandomState(6))
    before = capture_rng_states()
    loss = guide.loss(model, config, torch.device("cpu"))
    assert torch.isfinite(loss)
    loss.backward()
    assert same_state_tree(before, capture_rng_states())
    assert guide.draws == 64
    assert all(p.grad is None for p in model.value_head.parameters())
    nonzero = [name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert any(name.startswith("backbone.") for name in nonzero)
    assert any("action" in name or "policy" in name or "hand" in name for name in nonzero)


@pytest.mark.parametrize("empty", [True, False])
def test_zero_guidance_is_bit_exact_ppo_and_does_not_change_collection(empty, game_data, observation_fixture):
    config = fixtures.cohort_config()
    model, episodes, _ = fixtures.collected(observation_fixture, game_data, config=config)
    before_episodes = pickle.dumps(episodes)
    buffer = CohortBuffer(episodes, config)
    row, _ = label_record(game_data)
    guide = SearchGuidance(
        [] if empty else [row],
        game_data,
        Tokenizer(build_vocab(game_data)),
        np.random.RandomState(7),
        coefficient=0 if not empty else 0.02,
    )
    state = capture_rng_states()
    outcomes = []
    for auxiliary in (None, guide):
        clone = copy.deepcopy(model)
        optimizer = _make_policy_optimizer(clone.parameters(), config.lr)
        restore_rng_states(state)
        stats = _run_ppo_update(
            clone,
            optimizer,
            buffer,
            config.entropy_coeff,
            config,
            2,
            2,
            torch.device("cpu"),
            False,
            search_guidance=auxiliary,
        )
        outcomes.append((clone.state_dict(), optimizer.state_dict(), asdict(stats), capture_rng_states()))
    assert same_state_tree(outcomes[0], outcomes[1])
    assert before_episodes == pickle.dumps(episodes)
    assert guide.draws == 0


def test_guidance_shares_logical_steps_and_keeps_ppo_buffer_intact(game_data, observation_fixture):
    config = fixtures.cohort_config()
    model, episodes, _ = fixtures.collected(observation_fixture, game_data, config=config)
    buffer = CohortBuffer(episodes, config)
    before = copy.deepcopy(buffer.arrays)
    row, _ = label_record(game_data)
    guide = SearchGuidance([row], game_data, Tokenizer(build_vocab(game_data)), np.random.RandomState(7))
    optimizer = _make_policy_optimizer(model.parameters(), config.lr)
    stats = _run_ppo_update(
        model, optimizer, buffer, config.entropy_coeff, config, 2, 2, torch.device("cpu"), False, search_guidance=guide
    )
    assert guide.draws == 4 * 64
    assert stats.ppo_samples_processed == 16
    for key in before:
        np.testing.assert_array_equal(before[key], buffer.arrays[key])
    assert all(int(s["step"]) == 4 for s in optimizer.state.values())


def test_seed_reservations_atomic_commits_clock_duration_and_bootstrap(tmp_path):
    panels = seed_panels()
    exclusions = [panels["pilot"][0]]
    fresh = seed_panels(exclusions)
    assert fresh == seed_panels(exclusions)
    assert_separation(fresh, exclusions)
    with pytest.raises(ValueError):
        assert_separation(dict(a=[1], b=[1]))
    with pytest.raises(ValueError):
        assert_separation(dict(a=[1]), [1])
    assert {23, 42, 77} <= metadata_integers(b'{"invalidated_seeds":[23,42],"unused":"77"}')
    assert metadata_name("invalidated/reservation.json")
    assert not metadata_name("reserved/outcomes.jsonl")
    assert not metadata_name("games.json")
    immutable(tmp_path / "frozen.json", dict(a=np.int64(4)))
    immutable(tmp_path / "frozen.json", dict(a=4))
    with pytest.raises(ValueError):
        immutable(tmp_path / "frozen.json", dict(a=5))
    assert read(tmp_path / "frozen.json") == {"a": 4}
    assert not list(tmp_path.glob("*.tmp"))
    clock = start_clock(tmp_path, now=100)
    assert start_clock(tmp_path, now=200) == clock
    assert clock["deadline"] == 86500
    with pytest.raises(TimeoutError):
        check_deadline(time.time() - 1)
    timing = dict(
        ordinary_update=1,
        guided_update=1,
        greedy_game=1,
        sampled_game=1,
        assisted_game=1,
        label_state=1,
        diagnostic_inference_state=1,
    )
    choice = select_duration(timing, 1e9)
    assert choice["updates"] == 20
    assert select_duration(timing, 1)["updates"] is None
    with pytest.raises(ValueError):
        select_duration({**timing, "wins": 10}, 1e9)
    claim = paired_claim(np.ones((2, 256)), np.zeros((2, 256)))
    assert claim["qualifies"] and claim["ci95"] == [1, 1]
    assert not paired_claim(np.zeros((2, 256)), np.zeros((2, 256)))["qualifies"]


def test_atomic_failure_and_deadline_tampering_preserve_evidence(tmp_path, monkeypatch):
    from pylatro_agent import search_ppo_study as study

    study.atomic(tmp_path / "evidence.json", {"complete": False})
    original = study.os.replace

    def fail(*args):
        raise OSError("interrupted before commit")

    monkeypatch.setattr(study.os, "replace", fail)
    with pytest.raises(OSError):
        study.atomic(tmp_path / "evidence.json", {"complete": True})
    assert study.read(tmp_path / "evidence.json") == {"complete": False}
    monkeypatch.setattr(study.os, "replace", original)
    clock = start_clock(tmp_path, now=100)
    study.atomic(tmp_path / "clock.json", {**clock, "started": 200, "deadline": 86600})
    with pytest.raises(ValueError, match="Frozen artifact"):
        start_clock(tmp_path)


@pytest.mark.skipif(os.environ.get("SEARCH_PPO_ACTUAL_MPS_TEST") != "1", reason="Explicit MPS resume preflight")
def test_mps_complete_update_resume_matches_uninterrupted_training(
    tmp_path, monkeypatch, game_data, observation_fixture
):
    from pylatro_agent.checkpoint import save_ppo_checkpoint
    from pylatro_agent.training import search_ppo as training
    from pylatro_agent.training.search_ppo import from_dict, train_to

    config = from_dict(asdict(fixtures.cohort_config()))
    config.device = "mps"
    config.search_guidance = RECIPE.metadata()
    model = fixtures.tiny_model(game_data)
    optimizer = _make_policy_optimizer(model.parameters(), config.lr)
    source = save_ppo_checkpoint(
        model,
        tmp_path / "parent.pt",
        optimizer=optimizer,
        update_count=0,
        total_steps=0,
        planned_updates=2,
        entropy_coeff=config.entropy_coeff,
        entropy_signal_ema=None,
        lr=config.lr,
        agent_config=model.config,
        reward_config=config.reward_config,
        extra=dict(cohort_state={"slot_counters": [0, 0]}, cohort_ppo_config=asdict(config)),
    )
    row, _ = label_record(game_data)

    class Envs(fixtures.ScriptedEnvs):
        def public_states(self):
            return [row]

    monkeypatch.setattr(training, "EvaluationEnvs", lambda *a, **kw: Envs(observation_fixture))
    identity = dict(arm="guided", replica="test")
    kwargs = dict(
        data=game_data,
        vocab=build_vocab(game_data),
        deadline=time.time() + 120,
        teach_records=lambda states, deadline: [
            {**r, "target": row["target"], "accepted": True, "weight": 0.125, "reason": "accepted", "seconds": 0}
            for r in states
        ],
    )
    continuous = train_to(source, tmp_path / "continuous", config, identity, 2, **kwargs)
    train_to(source, tmp_path / "resumed", config, identity, 1, **kwargs)
    # A failed attempt is preserved, and a checkpoint remains the commit point.
    abandoned = tmp_path / "resumed/cohorts/update_2"
    abandoned.mkdir()
    (abandoned / "incomplete.txt").write_text("unfinished")
    resumed = train_to(source, tmp_path / "resumed", config, identity, 2, **kwargs)
    assert (tmp_path / "resumed/cohorts/update_2_attempt1/incomplete.txt").exists()
    a, b = load_checkpoint_payload(continuous, "cpu"), load_checkpoint_payload(resumed, "cpu")
    # Metal reductions may differ by a few ULPs between independent runs.
    # RNG/counters are exact; parameters and Adam moments must agree tightly.
    for key in ("state_dict", "optimizer_state_dict"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-6, atol=1e-7)
    for key in ("rng_states", "update_count", "total_steps"):
        assert same_state_tree(a[key], b[key]), key
    assert same_state_tree(a["search_ppo_study"]["guidance_rng"], b["search_ppo_study"]["guidance_rng"])
    assert a["search_ppo_study"]["optimizer_examples"] == 32
    # Recover metrics after an interruption between checkpoint and metrics commits.
    (tmp_path / "resumed/cohorts/update_2/update.json").unlink()
    train_to(source, tmp_path / "resumed", config, identity, 2, **kwargs)
    assert (tmp_path / "resumed/cohorts/update_2/update.json").exists()


@pytest.mark.skipif(os.environ.get("SEARCH_PPO_ACTUAL_MPS_TEST") != "1", reason="Explicit actual-model MPS preflight")
def test_actual_model_cpu_mps_likelihood_gradients_rng_optimizer_and_resume(tmp_path, game_data):
    from pathlib import Path

    from pylatro_agent.agent import AgentConfig, BalatroAgent

    assert torch.backends.mps.is_available()
    parent_path = Path("incumbents/ppo_replica1/checkpoint.pt")
    parent = load_checkpoint_payload(parent_path, "cpu")
    config = make_config(parent, guided=True, seeds=seed_panels()["pilot"], rng_seed=613, updates=1)
    initial = initial_checkpoint(parent_path, tmp_path / "initial.pt", config, {"test": True})
    assert same_state_tree(initial["state_dict"], parent["state_dict"])
    assert same_state_tree(initial["optimizer_state_dict"], parent["optimizer_state_dict"])
    validate_resume(load_checkpoint_payload(tmp_path / "initial.pt", "cpu"), {"test": True}, config)
    changed = copy.deepcopy(config)
    changed.observation_mode = "legacy"
    with pytest.raises(ValueError):
        validate_resume(initial, {"test": True}, changed)
    changed = copy.deepcopy(config)
    changed.search_guidance["coefficient"] = 0.03
    with pytest.raises(ValueError):
        validate_resume(initial, {"test": True}, changed)
    rng = capture_rng_states()
    expected = torch.rand(8, device="mps").cpu()
    restore_rng_states(rng)
    torch.testing.assert_close(torch.rand(8, device="mps").cpu(), expected, rtol=0, atol=0)
    model = BalatroAgent(AgentConfig(**parent["agent_config"]), build_vocab(game_data)).float().eval()
    model.load_state_dict(parent["state_dict"])
    tokenizer = Tokenizer(build_vocab(game_data))
    env = PublicTrainingEnv(44332, game_data, build_vocab(game_data))
    obs, _ = env.reset(seed=44332)
    _row, _ = label_record(game_data)
    observations = [
        obs,
        observation_dict(PublicGameSnapshot.capture(hand_state(), SubPhase.CHOOSE_ACTION), game_data, tokenizer),
    ]
    outputs, gradients = [], []
    for device in (torch.device("cpu"), torch.device("mps")):
        model.to(device)
        model.zero_grad(set_to_none=True)
        batch = _obs_dicts_to_batch(observations, device)
        dist, values = _grammar_distribution(model, batch, temperature=1.0)
        actions = dist.mode()
        logs = dist.log_prob(actions)
        loss = -logs.mean() - values["outcome_probabilities"][:, -1].clamp_min(1e-7).log().mean()
        assert torch.isfinite(loss)
        loss.backward()
        outputs.append((actions.cpu(), logs.detach().cpu(), values["expected_return"].detach().cpu()))
        gradients.append({n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None})
    for a, b in zip(outputs[0], outputs[1], strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-4)
    difference = sum((gradients[0][k] - gradients[1][k]).square().sum().item() for k in gradients[0])
    norm = sum(g.square().sum().item() for g in gradients[0].values())
    assert (difference / norm) ** 0.5 < 0.003
    from pylatro_agent.training.ppo_optimization import _restore_policy_optimizer_state

    optimizer = _make_policy_optimizer(model.parameters(), config.lr)
    _restore_policy_optimizer_state(optimizer, initial, config, torch.device("mps"))
    assert same_state_tree(
        {k: v.cpu() if torch.is_tensor(v) else v for k, v in next(iter(optimizer.state.values())).items()},
        next(iter(initial["optimizer_state_dict"]["state"].values())),
    )
    # Resume restores the exact next CPU and MPS streams, and independent labels.
    save_torch(tmp_path / "resume.pt", initial)
    restored = load_checkpoint_payload(tmp_path / "resume.pt", "cpu")
    validate_resume(restored, {"test": True}, config)
    assert same_state_tree(restored["search_ppo_study"]["guidance_rng"], initial["search_ppo_study"]["guidance_rng"])
    with pytest.raises(FileExistsError):
        save_torch(tmp_path / "resume.pt", initial)
