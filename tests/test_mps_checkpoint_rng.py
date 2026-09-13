import pytest
import torch

from pylatro_agent.checkpoint import capture_rng_states, restore_rng_states


def test_checkpoint_round_trip_preserves_accelerator_sampling_stream(monkeypatch, tmp_path):
    # Use a separate CPU generator as a deterministic stand-in for MPS. This
    # reproduces missing backend state without requiring a GPU in routine tests.
    generator = torch.Generator().manual_seed(1729)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", generator.get_state)
    monkeypatch.setattr(torch.mps, "set_rng_state", generator.set_state)
    state = capture_rng_states()
    assert "torch_mps" in state
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    expected = torch.multinomial(torch.ones(32, 8), 1, generator=generator)
    torch.rand(100, generator=generator)
    restore_rng_states(torch.load(path, map_location="cpu", weights_only=False))
    actual = torch.multinomial(torch.ones(32, 8), 1, generator=generator)
    assert torch.equal(actual, expected)


def test_unavailable_backend_and_legacy_checkpoint_do_not_require_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    def unavailable(*args):
        pytest.fail("An unavailable MPS backend must not be accessed")

    monkeypatch.setattr(torch.mps, "get_rng_state", unavailable)
    monkeypatch.setattr(torch.mps, "set_rng_state", unavailable)
    assert "torch_mps" not in capture_rng_states()
    restore_rng_states({"torch_mps": torch.zeros(8, dtype=torch.uint8)})
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    restore_rng_states({"torch_cpu": torch.get_rng_state()})
