"""CPU evaluation tuning is scoped, including failures and nested contexts."""

from types import SimpleNamespace

import pytest
import torch

from pylatro_agent.training.ppo_evaluation import _evaluation_threads


@pytest.mark.parametrize(("width", "available", "expected"), [(32, 16, 1), (384, 16, 8), (384, 2, 2)])
def test_auto_threads_respect_model_size_and_affinity(monkeypatch, width, available, expected):
    import pylatro_agent.training.ppo_evaluation as evaluation

    monkeypatch.setattr(evaluation.os, "sched_getaffinity", lambda _: set(range(available)), raising=False)
    previous = torch.get_num_threads()
    model = SimpleNamespace(config=SimpleNamespace(d_model=width))
    with _evaluation_threads(model, torch.device("cpu"), 0):
        assert torch.get_num_threads() == expected
    assert torch.get_num_threads() == previous


def test_nested_thread_overrides_restore_after_error():
    previous = torch.get_num_threads()
    with pytest.raises(RuntimeError, match="intentional"), _evaluation_threads(None, torch.device("cpu"), 2):
        assert torch.get_num_threads() == 2
        with _evaluation_threads(None, torch.device("cpu"), 1):
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 2
        raise RuntimeError("intentional")
    assert torch.get_num_threads() == previous


@pytest.mark.parametrize(("device", "requested"), [("cuda", 8), ("mps", 8), ("cpu", None)])
def test_gpu_and_unspecified_threads_preserve_caller(device, requested):
    previous = torch.get_num_threads()
    with _evaluation_threads(None, torch.device(device), requested):
        assert torch.get_num_threads() == previous
    assert torch.get_num_threads() == previous


def test_negative_eval_threads_rejected():
    with pytest.raises(ValueError, match="eval_cpu_threads"), _evaluation_threads(None, torch.device("cpu"), -1):
        pass
