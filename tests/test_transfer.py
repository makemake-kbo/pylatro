import pytest

from pylatro_agent.eval import SeedOutcome, paired_win_rate_delta


def test_pairing_rejects_missing_and_duplicate_seeds():
    rows = [SeedOutcome(1, True, 8, 0), SeedOutcome(2, False, 2, 0)]
    with pytest.raises(ValueError, match=r"same|missing"):
        paired_win_rate_delta(rows, rows[:1])
    with pytest.raises(ValueError, match=r"duplicate|distinct"):
        paired_win_rate_delta(rows + rows[:1], rows)


def test_configuration_is_strict_and_mixture_override_is_explicit():
    from dataclasses import asdict

    from pylatro_agent.agent import AgentConfig
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, reward_checkpoint_metadata
    from pylatro_agent.transfer import restore_configuration

    payload = {
        "agent_config": asdict(AgentConfig(hand_ar_mixture_eps=0.5)),
        **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG),
    }
    original, reward = restore_configuration(payload)
    changed, _ = restore_configuration(payload, mixture=0.1)
    assert original.hand_ar_mixture_eps == 0.5
    assert changed.hand_ar_mixture_eps == 0.1
    assert reward == DEFAULT_REWARD_CONFIG
    assert payload["agent_config"]["hand_ar_mixture_eps"] == 0.5
    payload["agent_config"]["unknown"] = 1
    with pytest.raises(ValueError, match="configuration fields"):
        restore_configuration(payload)
    del payload["agent_config"]["unknown"]
    del payload["reward_config"]["gamma"]
    with pytest.raises(ValueError, match="configuration fields"):
        restore_configuration(payload)


def test_statistics_cluster_repeats_and_validate_denominators():
    from pylatro_agent.transfer import paired_statistics, summarize

    rows = [
        dict(
            seed=s,
            sampling_seed=r,
            won=(s == 1),
            max_ante=8 if s == 1 else 1,
            cleared_antes=list(range(1, 9)) if s == 1 else [],
        )
        for s in (1, 2)
        for r in (0, 1, 2)
    ]
    summary = summarize(rows)
    assert summary["game_seeds"] == 2
    assert summary["games"] == 6
    assert summary["wins"] == 3
    assert summary["win_rate"] == 0.5
    assert summary["ante_clear_rates"]["8"] == 0.5
    assert summary["win_rate_cluster_ci95"] == [0, 1]
    assert paired_statistics(rows, rows)["paired_ci95"] == [0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + rows[:1])
    with pytest.raises(ValueError, match="missing"):
        summarize(rows[:-1])
    with pytest.raises(ValueError, match="same"):
        paired_statistics(rows, [dict(row, seed=row["seed"] + 10) for row in rows])


def test_seed_separation_checks_each_manifest(tmp_path):
    import json

    from pylatro_agent.transfer import CONFIRMATION_SEEDS, DEVELOPMENT_SEEDS, check_seed_separation

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps({"seeds": [2000000]}))
    b.write_text(json.dumps({"seeds": [110100]}))
    check_seed_separation(DEVELOPMENT_SEEDS, [a, b])
    with pytest.raises(ValueError, match="overlap"):
        check_seed_separation(CONFIRMATION_SEEDS, [a, b])
    with pytest.raises(ValueError, match="distinct"):
        check_seed_separation([1, 1], [a])


def test_demonstration_panel_requires_training_membership(tmp_path):
    from pylatro_agent.transfer import check_demonstration_seeds

    manifest = tmp_path / "training.json"
    manifest.write_text('{"seeds": [1, 2]}')
    assert check_demonstration_seeds([1, 2], [manifest])[0]["sha256"]
    for seeds in ([1, 3], [1, 1], []):
        with pytest.raises(ValueError, match="Demonstration evaluation"):
            check_demonstration_seeds(seeds, [manifest])


def test_paired_zero_gain_is_exact_and_does_not_depend_on_seed_order():
    from pylatro_agent.transfer import paired_statistics

    positive = {0, 12, 13, 21, 23, 27, 35, 50, 51, 55, 64, 77, 84, 91, 92}
    negative = {4, 26, 36, 49, 56, 65, 68, 70, 73, 75, 81, 86, 95, 97, 98}
    candidate = [dict(seed=s, sampling_seed=r, won=s in positive and r == 0) for s in range(100) for r in range(3)]
    control = [dict(seed=s, sampling_seed=r, won=s in negative and r == 0) for s in range(100) for r in range(3)]
    assert paired_statistics(candidate, control)["sampled_or_greedy_delta"] == 0
    assert paired_statistics(list(reversed(candidate)), control)["sampled_or_greedy_delta"] == 0


def test_policy_paths_preserve_observations_and_probabilities():
    import numpy as np
    import torch

    from pylatro import load_game_data
    from pylatro_agent.action import ActionRange
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.training.ppo_observations import _single_obs_to_batch
    from pylatro_agent.training.ppo_policy import _grammar_distribution as ppo_distribution
    from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
    from pylatro_agent.vocab import build_vocab

    torch.set_num_threads(1)
    data = load_game_data()
    vocab = build_vocab(data)
    env = BalatroEnv(seed=1729, data=data, vocab=vocab, enable_teacher=False, deck_key="b_blue")
    obs, _ = env.reset()
    obs, *_ = env.step(int(ActionRange.BLIND_PLAY))
    row = dict(obs=obs, action=int(np.flatnonzero(obs["action_mask"])[0]), return_target=0.0, won=False, max_ante=1)
    batch = _collate_batch([row], torch.device("cpu"), SupervisedConfig())
    online = _single_obs_to_batch(obs, torch.device("cpu"))
    for key in obs:
        torch.testing.assert_close(batch[key], online[key], check_dtype=False, rtol=0, atol=0)
    for mixture in (0.1, 0.5):
        model = BalatroAgent(
            AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, hand_ar_mixture_eps=mixture), vocab
        ).eval()
        with torch.inference_mode():
            supervised, _ = _grammar_distribution(model, batch)
            evaluation, _ = ppo_distribution(model, online, temperature=1.0)
            torch.testing.assert_close(supervised.log_prob(batch["actions"]), evaluation.log_prob(batch["actions"]))
            assert supervised.mode().item() == evaluation.mode().item()
            for _ in range(5):
                assert obs["action_mask"][evaluation.sample().item()]
    env.close()


def test_any_phase_diagnostic_copy_and_teacher_scoring_preserve_live_rng():
    import pickle

    from pylatro_agent.action import ActionRange
    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.heuristic import HeuristicAgent
    from pylatro_agent.transfer_diagnostics import clone_diagnostic_env, isolated_teacher_action, live_state_digest

    env = BalatroEnv(seed=1729, deck_key="b_blue", enable_teacher=False)
    env.reset()
    env.step(int(ActionRange.BLIND_PLAY))
    before = live_state_digest(env)
    rng_before = pickle.dumps(env.state.pseudorandom)
    teacher = HeuristicAgent(shop_policy="search", grow_scalers=True)
    teacher._estimate_hand_score_compute(env.state, (0,))
    assert live_state_digest(env) == before
    teacher.select_action(env.state, env._sub_phase, env.action_masks(), round_score=env._controller.round_score)
    assert live_state_digest(env) == before
    action = isolated_teacher_action(env)
    assert env.action_masks()[action]
    clone = clone_diagnostic_env(env)
    clone.step(action)
    assert live_state_digest(env) == before
    assert pickle.dumps(env.state.pseudorandom) == rng_before
    env.close()
    clone.close()


def test_correction_provenance_unit_weight_and_no_critic_targets():
    import torch

    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.training.supervised import SupervisedConfig
    from pylatro_agent.transfer_training import continuation_batch, correction_record

    env = BalatroEnv(enable_teacher=False)
    obs, _ = env.reset(seed=3000000)
    action = int(obs["action_mask"].nonzero()[0][0])
    correction = correction_record(
        obs,
        action,
        seed=3000000,
        step=0,
        student_action=action,
        checkpoint_sha256="checkpoint",
        teacher_sources="sources",
    )
    assert not {"won", "return_target", "terminal_outcome_target", "max_ante", "reward"} & correction.keys()
    batch = continuation_batch([correction], torch.device("cpu"), SupervisedConfig())
    assert batch["bc_weight"].item() == 1
    assert batch["terminal_outcome_mask"].item() == 0
    assert not batch["critic_mask"].item()
    with pytest.raises(ValueError, match="outcome/critic"):
        continuation_batch([{**correction, "won": True}], torch.device("cpu"), SupervisedConfig())
    with pytest.raises(ValueError, match="Evaluation seeds"):
        correction_record(
            obs,
            action,
            seed=100000,
            step=0,
            student_action=action,
            checkpoint_sha256="checkpoint",
            teacher_sources="sources",
        )
    env.close()


@pytest.mark.parametrize("deck", ["b_blue", "b_red"])
def test_supervised_cli_deck_propagation(monkeypatch, deck):
    import runpy
    import sys

    from pylatro_agent.training import supervised

    captured = []
    monkeypatch.setattr(supervised, "train_supervised", lambda config, **kwargs: captured.append(config))
    monkeypatch.setattr(sys, "argv", ["train.py", "supervised", "--device", "cpu", "--deck", deck])
    runpy.run_path("train.py", run_name="__main__")
    assert captured[0].deck_key == deck


def test_collection_workers_label_and_snapshot_the_same_state(tmp_path):
    import gzip
    import pickle

    from pylatro import load_game_data
    from pylatro_agent.transfer_collection import CollectionEnvs
    from pylatro_agent.transfer_diagnostics import isolated_teacher_action
    from pylatro_agent.vocab import build_vocab

    data = load_game_data()
    envs = CollectionEnvs(1, data=data, vocab=build_vocab(data), deck_key="b_blue", enable_teacher=False)
    try:
        obs, _ = envs.reset(3000000)
        teacher = envs.command("query_teacher", [3000000])[3000000]
        assert obs["action_mask"][teacher]
        path = tmp_path / "snapshot.pkl.gz"
        envs.command("save_diagnostic_snapshot", [(3000000, str(path))], snapshot=True)
        env = pickle.loads(gzip.decompress(path.read_bytes()))
        assert isolated_teacher_action(env) == teacher
        env.close()
    finally:
        envs.close()


def test_fast_reward_snapshot_matches_env_at_boss_transition():
    from pylatro import load_game_data
    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, state_potential
    from pylatro_agent.training.fast_generate import _capture_info
    from pylatro_agent.training.fast_runner import FastRunner
    from pylatro_agent.vocab import build_vocab

    # HF seed 2045118, first 14 recorded actions. Clearing Big exposes Pillar;
    # omitting boss_key overestimates safety and changes the economy potential.
    actions = [0, 9788, 2663, 2519, 17297, 13771, 17296, 13771, 17320, 0, 2519, 2519, 5600, 2519]
    data = load_game_data()
    runner = FastRunner(2045118, data, deck_key="b_blue", raise_errors=True)
    env = BalatroEnv(seed=2045118, data=data, vocab=build_vocab(data), deck_key="b_blue", enable_teacher=False)
    env.reset()
    try:
        for action in actions:
            assert runner.compute_mask()[action] and env.action_masks()[action]
            runner.step(action)
            env.step(action)
        fast = _capture_info(runner)
        gym = env._capture_state_info()
        assert state_potential(fast, DEFAULT_REWARD_CONFIG) == pytest.approx(
            state_potential(gym, DEFAULT_REWARD_CONFIG), abs=1e-12
        )
    finally:
        env.close()


def test_collection_records_load_without_outcome_targets(tmp_path):
    import torch

    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG
    from pylatro_agent.training.model_generate import load_records
    from pylatro_agent.transfer_collection import collect_corrections
    from pylatro_agent.vocab import build_vocab

    torch.set_num_threads(1)
    torch.manual_seed(0)
    data = load_game_data()
    vocab = build_vocab(data)
    model = BalatroAgent(AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32), vocab)
    source = tmp_path / "source.pt"
    source.write_bytes(b"test provenance")
    output = tmp_path / "corrections"
    rows = collect_corrections(
        model,
        DEFAULT_REWARD_CONFIG,
        data,
        vocab,
        source,
        torch.device("cpu"),
        output,
        seeds=[3000000],
        workers=0,
        batch_size=1,
        max_idle_steps=2,
    )
    loaded = load_records(output, reward_config=DEFAULT_REWARD_CONFIG)
    assert len(loaded) == len(rows) > 0
    assert all(row["bc_weight"] == 1 and "won" not in row and "return_target" not in row for row in loaded)


def test_actor_only_has_no_critic_gradient_and_control_does():
    import torch

    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.training.supervised import SupervisedConfig
    from pylatro_agent.transfer_training import continuation_batch, continuation_loss
    from pylatro_agent.vocab import build_vocab

    torch.set_num_threads(1)
    data = load_game_data()
    vocab = build_vocab(data)
    env = BalatroEnv(seed=3000000, data=data, vocab=vocab, enable_teacher=False)
    obs, _ = env.reset()
    row = dict(obs=obs, action=0, return_target=2.0, won=True, max_ante=8)
    config = SupervisedConfig()
    batch = continuation_batch([row], torch.device("cpu"), config)
    model = BalatroAgent(AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32), vocab)
    loss, _ = continuation_loss(model, batch, config, actor_only=True)
    loss.backward()
    assert all(p.grad is None for p in model.value_head.parameters())
    model.zero_grad(set_to_none=True)
    loss, _ = continuation_loss(model, batch, config, actor_only=False)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value_head.parameters())
    env.close()


def test_shared_seed_api_restores_sampling_rng_and_forwards_deck(monkeypatch):
    import torch

    from pylatro_agent.eval import evaluate_on_seeds
    from pylatro_agent.training import ppo_evaluation

    captured = []

    def evaluate(*args, **kwargs):
        captured.append(kwargs)
        kwargs["outcomes_out"].append(dict(seed=1, won=False, max_ante=1, round_score=0))

    monkeypatch.setattr(ppo_evaluation, "evaluate_model", evaluate)
    for greedy in (True, False):
        result = evaluate_on_seeds(
            None, None, None, [1], torch.device("cpu"), greedy=greedy, deck_key="b_blue", sampling_seed=2
        )
        assert result[0].seed == 1
    assert all(row["deck_key"] == "b_blue" and row["sampling_seed"] == 2 for row in captured)
    assert [row["greedy"] for row in captured] == [True, False]


def test_acceptance_requires_confirmation_and_greedy_guardrail():
    import runpy

    acceptance = runpy.run_path("tools/continue_transfer.py")["acceptance"]

    def panel(won, sampled):
        return [
            dict(seed=s, sampling_seed=r, won=won, max_ante=8 if won else 1)
            for s in (1, 2)
            for r in ((0, 1, 2) if sampled else (None,))
        ]

    candidates = {seed: panel(True, True) for seed in (0, 1)}
    controls = {seed: panel(False, True) for seed in (0, 1)}
    greedy = {seed: panel(True, False) for seed in (0, 1)}
    assert acceptance(candidates, controls, greedy, greedy)["status"] == "development_only"
    assert acceptance(candidates, controls, greedy, greedy, confirmation=True)["status"] == "validated"
    loss = {seed: panel(False, False) for seed in (0, 1)}
    assert acceptance(candidates, controls, loss, greedy, confirmation=True)["status"] != "validated"
    candidates[1] = controls[1]
    assert acceptance(candidates, controls, greedy, greedy, confirmation=True)["status"] != "validated"


def test_acceptance_rejects_stalled_or_censored_games():
    import runpy

    acceptance = runpy.run_path("tools/continue_transfer.py")["acceptance"]
    for failure in ("stalled", "censored"):
        candidate = {
            seed: [dict(seed=game, sampling_seed=0, won=True, max_ante=9) for game in (1, 2)] for seed in (0, 1)
        }
        control = {
            seed: [dict(seed=game, sampling_seed=0, won=False, max_ante=1) for game in (1, 2)] for seed in (0, 1)
        }
        greedy = {
            seed: [dict(seed=game, sampling_seed=None, won=True, max_ante=9) for game in (1, 2)] for seed in (0, 1)
        }
        candidate[0][0].update({failure: True, "won": False, "max_ante": 1})
        with pytest.raises(ValueError, match="stalled or censored"):
            acceptance(candidate, control, greedy, greedy, confirmation=True)


def test_small_continuation_saves_strict_reloadable_checkpoint(tmp_path):
    from dataclasses import asdict

    import torch

    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.checkpoint import save_checkpoint
    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, reward_checkpoint_metadata
    from pylatro_agent.training.model_generate import load_records, save_records
    from pylatro_agent.training.supervised import SupervisedConfig
    from pylatro_agent.transfer import load_policy
    from pylatro_agent.transfer_training import train_continuation
    from pylatro_agent.vocab import build_vocab

    data = load_game_data()
    vocab = build_vocab(data)
    config = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32)
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
    env = BalatroEnv(seed=3000000, data=data, vocab=vocab, enable_teacher=False)
    obs, _ = env.reset()
    row = dict(obs=obs, action=0, return_target=2.0, won=True, max_ante=8, seed=3000000)
    path = tmp_path / "records.parquet"
    save_records([row] * 4, path, reward_config=DEFAULT_REWARD_CONFIG)
    records = load_records(path, reward_config=DEFAULT_REWARD_CONFIG)
    output = train_continuation(
        source,
        records,
        vocab,
        torch.device("cpu"),
        tmp_path / "trial",
        arm="actor_only",
        seed=0,
        updates=2,
        batch_size=2,
    )
    _, _, metadata = load_policy(output, vocab, torch.device("cpu"))
    assert metadata["continuation"]["status"] == "complete"
    assert metadata["supervised_config"]["deck_key"] == "b_blue"
    env.close()


def test_acceptance_includes_exact_five_point_gain_despite_roundoff():
    import runpy

    acceptance = runpy.run_path("tools/continue_transfer.py")["acceptance"]
    winning = {8, 9, 27, 36, 48, 50, 58, 62, 63, 65, 66, 71, 75, 76, 96}
    candidate = {
        seed: [
            dict(seed=g, sampling_seed=r, won=g in winning and r == 0, max_ante=2) for g in range(100) for r in range(3)
        ]
        for seed in (0, 1)
    }
    control = {seed: [dict(row, won=False) for row in rows] for seed, rows in candidate.items()}
    greedy = {seed: [dict(seed=g, sampling_seed=None, won=False, max_ante=1) for g in range(100)] for seed in (0, 1)}
    result = acceptance(candidate, control, greedy, greedy, confirmation=True)
    # Exactly 15 added wins / 300 runs per replica, irrespective of seed positions.
    assert result["paired"]["sampled"]["sampled_or_greedy_delta"] == pytest.approx(0.05)
    assert result["status"] == "validated"


def test_intervention_runner_scores_and_replays_isolated_branches(tmp_path, monkeypatch):
    import gzip
    import json
    import pickle
    import runpy
    import sys
    from dataclasses import asdict

    import torch

    from pylatro import load_game_data
    from pylatro_agent.action import ActionRange
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.checkpoint import save_checkpoint
    from pylatro_agent.env import BalatroEnv
    from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, reward_checkpoint_metadata
    from pylatro_agent.transfer_diagnostics import isolated_teacher_action
    from pylatro_agent.vocab import build_vocab

    torch.set_num_threads(1)
    data = load_game_data()
    vocab = build_vocab(data)
    config = AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        BalatroAgent(config, vocab),
        checkpoint,
        extra={"agent_config": asdict(config), **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG)},
    )
    env = BalatroEnv(seed=3000000, data=data, vocab=vocab, enable_teacher=False, max_steps=2)
    env.reset()
    env.step(int(ActionRange.BLIND_PLAY))
    snapshot = tmp_path / "snapshot.pkl.gz"
    snapshot.write_bytes(gzip.compress(pickle.dumps(env)))
    snapshots = tmp_path / "snapshots.json"
    snapshots.write_text(
        json.dumps(
            [
                dict(
                    seed=3000000,
                    step=1,
                    phase=1,
                    ante=1,
                    path=str(snapshot),
                    student_action=int(ActionRange.PLAY_SUBSET_START),
                    teacher_action=isolated_teacher_action(env),
                    continuation_sampling_seed=3500000,
                )
            ]
        )
    )
    output = tmp_path / "intervention"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "intervene_transfer.py",
            "--checkpoint",
            str(checkpoint),
            "--snapshots",
            str(snapshots),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ],
    )
    runpy.run_path("tools/intervene_transfer.py", run_name="__main__")
    row = json.loads((output / "interventions.jsonl").read_text())
    assert set(row["outcomes"]) == {"student", "teacher", "exact_candidate_oracle"}
    assert row["exact_score_probes"]
    assert all(result["steps"] > 0 for result in row["outcomes"].values())
    env.close()
