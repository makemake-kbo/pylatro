"""DAgger from V68: labels, execution mixing, shards, fine-tuning, panel and resume."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.checkpoint import save_checkpoint
from pylatro_agent.constants import NUM_ACTIONS
from pylatro_agent.dagger import (
    DaggerConfig,
    SeedPlan,
    ShardPool,
    collate,
    collect_round,
    compact_observation,
    dagger_loss,
    hashed_seeds,
    load_exclusions,
    make_envs,
    play,
    read_shard,
    run,
    teacher_label,
    train_round,
    write_shard,
)
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.reward import RewardConfig, reward_checkpoint_metadata
from pylatro_agent.training.public_training import PublicTrainingEnv
from pylatro_agent.vocab import build_vocab

# Ante-1 games keep every collected round to a few dozen decisions.
TINY = DaggerConfig(
    rounds=2,
    games_per_round=2,
    panel_games=2,
    win_ante=1,
    workers=0,
    updates_per_round=3,
    batch_size=8,
    micro_batch_size=4,
    shard_rows=16,
    pool_shards=2,
    warmup_updates=2,
    inference_batch=4,
    device="cpu",
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def game_data():
    return load_game_data()


@pytest.fixture(scope="module")
def vocab(game_data):
    return build_vocab(game_data)


def tiny_model(vocab, seed=0):
    torch.manual_seed(seed)
    config = AgentConfig(
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        hand_ar_mixture_eps=0.5,
        win_only_value=True,
        return_value_mode="terminal_only",
        action_head_variant="entity_sale_subset_residual_v1",
    )
    return BalatroAgent(config, vocab).eval()


def tiny_checkpoint(path, vocab):
    model = tiny_model(vocab)
    reward = RewardConfig(objective="milestone", milestone_reward_budget=0.0, gamma=1.0)
    save_checkpoint(model, path, extra={"agent_config": asdict(model.config), **reward_checkpoint_metadata(reward)})
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_config_validation_and_beta_schedule():
    config = DaggerConfig(beta_initial=0.5, beta_decay=0.5)
    assert [config.beta(r) for r in (1, 2, 3)] == [0.5, 0.25, 0.125]
    assert DaggerConfig.from_json(json.loads(json.dumps(config.to_json()))) == config
    for bad in (
        {"teacher_view": "oracle"},
        {"student_action": "argmax"},
        {"panel_modes": ()},
        {"batch_size": 10, "micro_batch_size": 4},
        {"beta_initial": 1.5},
    ):
        with pytest.raises(ValueError):
            DaggerConfig(**bad)


def test_seed_plan_is_disjoint_persistent_and_avoids_exclusions(tmp_path):
    excluded = set(hashed_seeds("panel", 3))
    (tmp_path / "excluded.json").write_text(json.dumps({"evaluation": sorted(excluded), "training": []}))
    assert load_exclusions([tmp_path / "excluded.json"]) == excluded
    plan = SeedPlan(tmp_path / "seeds.json", TINY, excluded)
    first, second = plan.round(1), plan.round(2)
    every = plan.panel + first + second
    assert len(set(every)) == len(every) and not set(every) & excluded
    assert all(type(s) is int and 1_000_000_000 <= s < 2**31 for s in every)
    reloaded = SeedPlan(tmp_path / "seeds.json", TINY)
    assert reloaded.panel == plan.panel and reloaded.round(1) == first and reloaded.round(2) == second
    assert reloaded.exclusions == excluded
    with pytest.raises(ValueError, match="panel size"):
        SeedPlan(tmp_path / "seeds.json", replace(TINY, panel_games=3))
    with pytest.raises(ValueError, match="collision"):
        SeedPlan(tmp_path / "seeds.json", TINY, {first[0]})
    assert SeedPlan(tmp_path / "seeds.json", TINY).exclusions == excluded


@pytest.mark.parametrize("view", ["full", "public"])
def test_teacher_label_is_legal_deterministic_and_leaves_the_game_unchanged(view, game_data, vocab):
    env = PublicTrainingEnv(4_242_001, game_data, vocab)
    obs, _ = env.reset(seed=4_242_001)
    rng = np.random.default_rng(0)
    for _ in range(12):  # Reach the hand and shop phases, not just blind selection.
        before = env.runner.snapshot().payload
        label = teacher_label(env, HeuristicAgent(shop_policy="search", grow_scalers=True), view)
        again = teacher_label(env, HeuristicAgent(shop_policy="search", grow_scalers=True), view)
        assert env.runner.snapshot().payload == before
        assert label["action"] == again["action"] and "error" not in label
        assert obs["action_mask"][label["action"]]
        obs, _, terminal, truncated, _ = env.step(int(rng.choice(np.flatnonzero(obs["action_mask"]))))
        if terminal or truncated:
            break


def test_shards_round_trip_and_pool_joins_outcomes(tmp_path, game_data, vocab):
    env = PublicTrainingEnv(4_242_002, game_data, vocab)
    obs, _ = env.reset(seed=4_242_002)
    rng = np.random.default_rng(0)
    rows, originals = [], []
    for step in range(10):
        action = int(rng.choice(np.flatnonzero(obs["action_mask"])))
        seed = 11 if step < 6 else 12
        rows.append(
            dict(
                obs=compact_observation(obs),
                label=action,
                executed=action,
                student=action,
                greedy=action,
                by_teacher=False,
                seed=seed,
                step=step,
            )
        )
        originals.append(obs)
        obs, _, terminal, truncated, _ = env.step(action)
        if terminal or truncated:
            obs, _ = env.reset(seed=4_242_102 + step)
    write_shard(tmp_path / "a.npz", rows[:6])
    write_shard(tmp_path / "b.npz", rows[6:])
    batch = collate({**read_shard(tmp_path / "a.npz"), "outcome_class": np.zeros(6), "outcome_mask": np.ones(6)}, "cpu")
    for i in range(6):
        for key in ("tokens", "scalars", "history_cards"):
            np.testing.assert_array_equal(batch[key][i].numpy(), originals[i][key])
        np.testing.assert_array_equal(batch["action_mask"][i].numpy(), originals[i]["action_mask"])
    assert batch["action_mask"].shape[1] == NUM_ACTIONS
    outcomes = {
        11: {"won": True, "final_ante": 2, "stalled": False},
        12: {"won": False, "final_ante": 3, "stalled": True},
    }
    pool = ShardPool(
        [(tmp_path / "a.npz", 6), (tmp_path / "b.npz", 4)], outcomes, pool_shards=2, rng=np.random.default_rng(0)
    )
    drawn = pool.next_batch(10)
    assert sorted(drawn["step"].tolist()) == list(range(10))
    won = drawn["seed"] == 11
    assert (drawn["outcome_class"][won] == 8).all() and (drawn["outcome_mask"][won] == 1).all()
    assert (drawn["outcome_mask"][~won] == 0).all()


def test_beta_one_executes_the_teacher_and_beta_zero_the_student(game_data, vocab):
    model = tiny_model(vocab)
    for beta in (1.0, 0.0):
        decisions = []

        def record(seed, step, obs, *, outcome=None, decisions=decisions, **decision):
            if outcome is None:
                decisions.append(decision)

        envs = make_envs(TINY, game_data, vocab)
        try:
            outcomes = play(
                model,
                envs,
                [4_242_003],
                TINY,
                device=torch.device("cpu"),
                mode="sample",
                beta=beta,
                rng=np.random.default_rng(0),
                teacher_view="full",
                on_decision=record,
            )
        finally:
            envs.close()
        assert outcomes[0]["steps"] == len(decisions) > 0
        if beta == 1.0:
            assert all(d["by_teacher"] and d["executed"] == d["label"]["action"] for d in decisions)
        else:
            assert not any(d["by_teacher"] for d in decisions)
            assert all(d["executed"] == d["student"] for d in decisions)


def test_fine_tuning_on_collected_labels_lowers_teacher_nll(tmp_path, game_data, vocab):
    model = tiny_model(vocab)
    config = replace(TINY, lr=3e-3, warmup_updates=0, updates_per_round=25, batch_size=16, micro_batch_size=8)
    manifest = collect_round(
        model,
        config,
        [4_242_004, 4_242_005],
        tmp_path / "collect",
        round_index=1,
        device=torch.device("cpu"),
        data=game_data,
        vocab=vocab,
        log=lambda _message: None,
    )
    assert manifest["labelled"] == manifest["decisions"] == sum(s["rows"] for s in manifest["shards"])
    outcomes = {
        o["seed"]: o for o in map(json.loads, (tmp_path / "collect" / "outcomes.jsonl").read_text().splitlines())
    }
    shards = [(tmp_path / "collect" / s["file"], s["rows"]) for s in manifest["shards"]]

    def nll():
        pool = ShardPool(shards, outcomes, pool_shards=len(shards), rng=np.random.default_rng(0))
        batch = collate(pool.next_batch(manifest["labelled"]), "cpu")
        with torch.no_grad():
            return dagger_loss(model, batch, 0.0)[1]["nll"]

    before = nll()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    pool = ShardPool(shards, outcomes, pool_shards=len(shards), rng=np.random.default_rng(1))
    updates = train_round(
        model, optimizer, pool, config, torch.device("cpu"), completed_updates=0, log_path=tmp_path / "train.jsonl"
    )
    assert updates == 25
    assert nll() < before
    assert json.loads((tmp_path / "train.jsonl").read_text().splitlines()[-1])["update"] == 25


@pytest.mark.slow
def test_run_resumes_interrupted_rounds_and_reuses_finished_work(tmp_path, vocab):
    import tools.dagger as cli

    start = tiny_checkpoint(tmp_path / "start.pt", vocab)
    out = tmp_path / "run"
    curve = run(start, out, TINY, log=lambda _message: None)
    assert [row["round"] for row in curve] == [0, 1, 2]
    assert all(row["panel_greedy"]["games"] == 2 for row in curve)
    assert curve[2]["panel_greedy"]["win_rate_delta_ci95"] is not None
    first = digest(out / "round_001" / "checkpoint.pt")
    shard = digest(out / "round_002" / "collect" / "shard_00000.npz")

    # An interrupted second round: its collection never finished.
    for name in ("round_002/checkpoint.pt", "round_002/collect/manifest.json", "round_002/panel_greedy_summary.json"):
        (out / name).unlink()
    # Worker processes must replay exactly what the in-process collection produced.
    curve = run(start, out, replace(TINY, workers=1), log=lambda _message: None)
    assert digest(out / "round_001" / "checkpoint.pt") == first
    assert digest(out / "round_002" / "collect" / "shard_00000.npz") == shard
    assert "panel_greedy" in curve[2]

    with pytest.raises(ValueError, match="different settings"):
        run(start, out, replace(TINY, games_per_round=3), log=lambda _message: None)
    other = tiny_checkpoint(tmp_path / "other.pt", build_vocab(load_game_data()))
    torch.save({**torch.load(other, weights_only=False), "note": "different"}, other)
    with pytest.raises(ValueError, match="starting checkpoint"):
        run(other, out, TINY, log=lambda _message: None)

    # The CLI resumes from the saved configuration without repeating its flags.
    cli.main(["run", "--checkpoint", str(start), "--out", str(out)])
    cli.main(["report", "--out", str(out)])
