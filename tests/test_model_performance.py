"""Reference parity for optimized kernels, independent of timing/hardware speed."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from pylatro_agent import action_grammar as grammar
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.backbone import PreNormTransformerLayer
from pylatro_agent.constants import MAX_HAND_SIZE, MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, ActionRange
from pylatro_agent.subset_actions import (
    CONSUMABLE_HAND_SUBSET_BITS,
    CONSUMABLE_HAND_SUBSET_SIZES,
    HAND_SUBSET_BITS,
    HAND_SUBSET_SIZES,
)
from pylatro_agent.training.ppo_observations import _ObsBuffer, _observation_buffer_batch

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def reference_next_slot_mask(valid, counts, prefix, last, bits, sizes, slot_bits):
    valid = valid & sizes[None].eq(counts[:, None])
    valid = valid & (bits[None] & prefix[:, None]).eq(prefix[:, None])
    return torch.stack(
        [
            (valid & (bits[None] & ((1 << (slot + 1)) - 1)).eq((prefix | slot_bits[slot])[:, None])).any(-1)
            & (last < slot)
            for slot in range(MAX_HAND_SIZE)
        ],
        dim=-1,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("consumable", [False, True])
def test_next_slot_reduction_matches_reference(device, consumable):
    rng = np.random.default_rng(937)
    bits_np = CONSUMABLE_HAND_SUBSET_BITS if consumable else HAND_SUBSET_BITS
    sizes_np = CONSUMABLE_HAND_SUBSET_SIZES if consumable else HAND_SUBSET_SIZES
    bits = torch.tensor(bits_np.astype(np.int64), device=device)
    sizes = torch.tensor(sizes_np.astype(np.int64), device=device)
    slot_bits = grammar._slot_bits(torch.device(device))
    rows = 64
    masks = rng.random((rows, len(bits_np))) < 0.3
    masks[0] = False
    masks[1] = True
    valid = torch.tensor(masks, device=device)
    for step in range(6):
        prefixes, lasts, counts = [], [], []
        for _ in range(rows):
            subset = int(rng.choice(bits_np))
            slots = [slot for slot in range(MAX_HAND_SIZE) if subset & (1 << slot)]
            selected = slots[:step]
            prefixes.append(sum(1 << slot for slot in selected))
            lasts.append(selected[-1] if selected else -1)
            # Also exercise impossible target sizes and exhausted prefixes.
            counts.append(int(rng.integers(0, 7)))
        args = [valid, *[torch.tensor(x, device=device) for x in (counts, prefixes, lasts)], bits, sizes, slot_bits]
        torch.testing.assert_close(grammar._next_slot_mask(*args), reference_next_slot_mask(*args), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_grammar_sampling_scoring_and_gradients_match_reference(device, monkeypatch):
    torch.manual_seed(109)
    head = grammar.ActionGrammarHead(8).to(device)
    mask = torch.zeros(4, NUM_ACTIONS, device=device)
    mask[0, ActionRange.PLAY_SUBSET_START : ActionRange.PLAY_SUBSET_END + 1] = 1
    mask[1, ActionRange.DISCARD_SUBSET_START : ActionRange.DISCARD_SUBSET_END + 1] = 1
    mask[2] = (torch.rand(NUM_ACTIONS, device=device) < 0.1).float()
    # The last row deliberately exercises the all-masked sampling fallback.
    inputs = (
        torch.randn(4, MAX_SEQ_LEN, 8, device=device),
        torch.ones(4, MAX_SEQ_LEN, device=device),
        torch.zeros(4, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long, device=device),
        torch.zeros(4, MAX_SEQ_LEN, dtype=torch.long, device=device),
        torch.zeros(4, SCALAR_DIM, device=device),
    )

    def run():
        torch.manual_seed(319)
        head.zero_grad(set_to_none=True)
        dist = grammar.ActionGrammarDistribution(head(*inputs), mask, hand_ar_mixture_eps=1)
        sampled = dist.sample()
        log_probs, entropy = dist.log_prob(sampled), dist.entropy()
        (log_probs + 0.01 * entropy).sum().backward()
        grads = [p.grad.detach().clone() if p.grad is not None else None for p in head.parameters()]
        return (sampled, log_probs.detach(), entropy.detach()), grads

    actual, actual_grads = run()
    monkeypatch.setattr(grammar, "_next_slot_mask", reference_next_slot_mask)
    expected, expected_grads = run()
    for value, ref_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, ref_value, rtol=0, atol=0)
    for value, ref_value in zip(actual_grads, expected_grads, strict=True):
        if value is None:
            assert ref_value is None
        else:
            torch.testing.assert_close(value, ref_value, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("padding", [False, True])
def test_attention_output_and_gradients_match_weight_materializing_reference(device, padding):
    torch.manual_seed(73)
    layer = PreNormTransformerLayer(32, 4, 64, dropout=0).to(device).eval()
    reference = copy.deepcopy(layer)
    x = torch.randn(3, 17, 32, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    mask = (
        torch.arange(17, device=device)[None] >= torch.tensor([1, 9, 17], device=device)[:, None] if padding else None
    )
    normed = reference.norm1(ref_x)
    attn, _ = reference.attn(normed, normed, normed, key_padding_mask=mask, need_weights=True)
    expected = ref_x + reference.dropout(attn)
    expected = expected + reference.ffn(reference.norm2(expected))
    actual = layer(x, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(x.grad, ref_x.grad, atol=5e-6, rtol=3e-5)
    assert layer.state_dict().keys() == reference.state_dict().keys()
    for (_, parameter), (_, ref_parameter) in zip(layer.named_parameters(), reference.named_parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, ref_parameter.grad, atol=2e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_observation_staging_reuses_storage_and_preserves_snapshots(device):
    buffer = _ObsBuffer(2, torch.device(device))
    tensors = _observation_buffer_batch(buffer)
    pointers = {key: value.data_ptr() for key, value in tensors.items()}
    snapshots = []
    for value in (1, 2, 7):
        source = {key: np.full_like(array, value) for key, array in buffer.as_numpy_dict().items()}
        buffer.update(source)
        # Clone on the same stream, then immediately reuse staging storage.
        # CUDA source mutation must wait for the preceding H2D transfers.
        snapshots.append({key: tensor.clone() for key, tensor in tensors.items()})
        for key, array in buffer.as_numpy_dict().items():
            np.testing.assert_array_equal(array, source[key])
        for array in source.values():
            array.fill(99)  # Buffer owns its copy, not the caller's arrays.
    for expected, snapshot in zip((1, 2, 7), snapshots, strict=True):
        for tensor in snapshot.values():
            torch.testing.assert_close(tensor, torch.full_like(tensor, expected), rtol=0, atol=0)
    assert {key: value.data_ptr() for key, value in tensors.items()} == pointers
    if device == "cpu":
        for key, host in buffer._host_tensors.items():
            assert host.data_ptr() == tensors[key].data_ptr()
    else:
        assert all(host.is_pinned() for host in buffer._host_tensors.values())


def test_precision_config_validation_and_cpu_fallback():
    from pylatro_agent.precision import backbone_autocast
    from pylatro_agent.training.ppo_config import PPOConfig, _validate_ppo_config

    with pytest.raises(ValueError, match="precision"):
        AgentConfig(precision="fp16")
    with pytest.raises(ValueError, match="precision"):
        _validate_ppo_config(PPOConfig(precision="fp16"))
    with backbone_autocast("bf16", torch.device("cpu")):
        assert not torch.is_autocast_enabled("cpu")


def test_bf16_requires_native_support(monkeypatch):
    from pylatro_agent import precision

    monkeypatch.setattr(precision, "_supports_native_bf16", lambda index: False)
    with pytest.raises(RuntimeError, match="native CUDA BF16"):
        precision.backbone_autocast("bf16", torch.device("cuda:0"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 validation")
@pytest.mark.parametrize("training", [False, True])
def test_bf16_trunk_keeps_rl_heads_gradients_and_adam_fp32(training):
    from pylatro import load_game_data
    from pylatro_agent.vocab import build_vocab

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("Native CUDA BF16 unavailable")
    torch.manual_seed(923)
    model = (
        BalatroAgent(
            AgentConfig(d_model=32, n_layers=2, n_heads=4, d_ff=64, precision="bf16"),
            build_vocab(load_game_data()),
        )
        .cuda()
        .train(training)
    )
    staged = _ObsBuffer(4, torch.device("cuda"))
    batch = _observation_buffer_batch(staged)
    batch["attention_mask"].fill_(1)
    batch["scalars"][:, 2] = torch.tensor([1, 5, 6, 8], device="cuda")
    batch["scalars"][:, 22] = 8
    batch["action_mask"].fill_(1)
    dtypes = []
    hook = model.backbone.layers[0].ffn[0].register_forward_hook(lambda _, args, out: dtypes.append(out.dtype))
    with torch.no_grad():
        dist, _ = model(**batch)
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions)
    dist, values = model(**batch)
    hook.remove()
    assert dtypes == [torch.bfloat16, torch.bfloat16]
    log_probs = dist.log_prob(actions)
    if not training:  # Dropout deliberately changes log probabilities during BC.
        torch.testing.assert_close(log_probs, old_log_probs, atol=1e-4, rtol=1e-4)
    assert log_probs.dtype == dist.entropy().dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in values.values())
    loss = -log_probs.mean() + values["expected_return"].square().mean()
    loss.backward()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.step()
    for parameter in model.parameters():
        assert parameter.dtype == torch.float32 and torch.isfinite(parameter).all()
        if parameter.grad is not None:
            assert parameter.grad.dtype == torch.float32 and torch.isfinite(parameter.grad).all()
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.dtype == torch.float32 and torch.isfinite(value).all()
