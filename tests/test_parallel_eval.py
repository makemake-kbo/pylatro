"""CPU workers must preserve complete evaluation results and always shut down."""

import multiprocessing as mp

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.training.evaluation_envs import EvaluationEnvs
from pylatro_agent.training.ppo_evaluation import run_seed_evaluation
from pylatro_agent.vocab import build_vocab


@pytest.mark.parametrize("greedy", [True, False])
def test_parallel_evaluation_matches_local(greedy):
    data = load_game_data()
    vocab = build_vocab(data)
    torch.manual_seed(0)
    model = BalatroAgent(AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64), vocab)
    children_before = {p.pid for p in mp.active_children()}

    def evaluate(workers):
        torch.manual_seed(123)
        rows = run_seed_evaluation(
            model,
            data,
            vocab,
            [10000, 10001, 10002, 10003, 10004],
            torch.device("cpu"),
            max_no_progress_steps=8,
            win_ante=2,
            batch_size=3,
            record_critic=True,
            greedy=greedy,
            eval_workers=workers,
        )
        rng = torch.get_rng_state().clone()
        return [{k: v for k, v in row.items() if not k.endswith("_seconds")} for row in rows], rng

    local, local_rng = evaluate(0)
    remote, remote_rng = evaluate(2)
    assert remote == local
    torch.testing.assert_close(remote_rng, local_rng, atol=0, rtol=0)
    assert model.training
    assert {p.pid for p in mp.active_children()} == children_before


def test_worker_error_and_death_propagate_and_cleanup():
    data = load_game_data()
    vocab = build_vocab(data)
    children_before = {p.pid for p in mp.active_children()}
    pool = EvaluationEnvs(1, data=data, vocab=vocab, enable_teacher=False)
    try:
        pool.reset(1701)
        # A nonexistent seed raises in the worker and must reach the caller.
        pool.connections[0].send(("step", [(999, 0)]))
        with pytest.raises(RuntimeError, match="KeyError"):
            pool._receive(0)
    finally:
        pool.close()
    pool = EvaluationEnvs(1, data=data, vocab=vocab, enable_teacher=False)
    try:
        pool.reset(1701)
        pool.processes[0].terminate()
        pool.processes[0].join(timeout=10)
        with pytest.raises(RuntimeError, match=r"exited|disconnected"):
            pool._receive(0)
    finally:
        pool.close()
    assert {p.pid for p in mp.active_children()} == children_before


def test_callback_error_closes_workers_and_restores_model():
    data = load_game_data()
    vocab = build_vocab(data)
    torch.manual_seed(0)
    model = BalatroAgent(AgentConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64), vocab)
    children_before = {p.pid for p in mp.active_children()}
    threads_before = torch.get_num_threads()

    def fail(_row):
        assert torch.get_num_threads() == 2
        raise ValueError("callback failed")

    with pytest.raises(ValueError, match="callback failed"):
        run_seed_evaluation(
            model,
            data,
            vocab,
            [10000, 10001],
            torch.device("cpu"),
            max_no_progress_steps=4,
            batch_size=2,
            eval_workers=2,
            eval_cpu_threads=2,
            on_result=fail,
        )
    assert model.training
    assert torch.get_num_threads() == threads_before
    assert {p.pid for p in mp.active_children()} == children_before
