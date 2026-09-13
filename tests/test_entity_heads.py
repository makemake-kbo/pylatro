from dataclasses import asdict

import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.architecture_transfer import add_sale_residual
from pylatro_agent.checkpoint import save_checkpoint
from pylatro_agent.constants import MAX_JOKER_SLOTS, MAX_SEQ_LEN, SCALAR_DIM, TOKEN_DIM
from pylatro_agent.entity_heads import SaleResidualHead
from pylatro_agent.reward import DEFAULT_REWARD_CONFIG, reward_checkpoint_metadata
from pylatro_agent.transfer import load_policy, restore_configuration
from pylatro_agent.vocab import build_vocab


@pytest.mark.parametrize("variant", ["pooled_sale_residual_v1", "entity_sale_residual_v1"])
def test_transfer_preserves_all_component_logits_and_source_weights(variant):
    torch.manual_seed(1)
    vocab = build_vocab(load_game_data())
    source = BalatroAgent(AgentConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, dropout=0), vocab)
    target, migration = add_sale_residual(source, vocab, variant)
    inputs = (
        torch.randn(3, MAX_SEQ_LEN, 32),
        torch.ones(3, MAX_SEQ_LEN),
        torch.zeros(3, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long),
        torch.zeros(3, MAX_SEQ_LEN, dtype=torch.long),
        torch.randn(3, SCALAR_DIM),
    )
    before = source.action_grammar_head(*inputs)
    after = target.action_grammar_head(*inputs)
    for key in before.__dataclass_fields__:
        if getattr(before, key) is None:
            assert getattr(after, key) is None
        else:
            torch.testing.assert_close(getattr(before, key), getattr(after, key), rtol=0, atol=0)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
    assert migration["observation_semantics_changed"] is False


def test_entity_residual_scores_travel_with_token_features():
    torch.manual_seed(8)
    head = SaleResidualHead(32, entity_conditioned=True)
    torch.nn.init.normal_(head.net[-1].weight)
    state, jokers = torch.randn(2, 128), torch.randn(2, MAX_JOKER_SLOTS, 32)
    permutation = torch.randperm(MAX_JOKER_SLOTS)
    torch.testing.assert_close(head(state, jokers[:, permutation]), head(state, jokers)[:, permutation])
    loss = head(state, jokers).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_sale_controls_match_parameter_capacity():
    heads = [SaleResidualHead(384, entity_conditioned=value) for value in (False, True)]
    counts = [sum(p.numel() for p in head.parameters()) for head in heads]
    assert abs(counts[0] - counts[1]) / max(counts) < 0.001


def test_known_archived_config_migrates_but_other_missing_fields_do_not():
    payload = {"agent_config": asdict(AgentConfig()), **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG)}
    del payload["agent_config"]["action_head_variant"]
    config, _ = restore_configuration(payload)
    assert config.action_head_variant == "legacy_v1"
    assert "action_head_variant" not in payload["agent_config"]
    del payload["agent_config"]["d_model"]
    with pytest.raises(ValueError, match="configuration fields"):
        restore_configuration(payload)
    with pytest.raises(ValueError, match="Unknown action head"):
        AgentConfig(action_head_variant="unregistered")


@pytest.mark.parametrize("variant", ["pooled_sale_residual_v1", "entity_sale_residual_v1"])
def test_checkpoint_roundtrip_requires_variant_metadata(tmp_path, variant):
    vocab = build_vocab(load_game_data())
    config = AgentConfig(d_model=32, n_layers=1, n_heads=4, d_ff=64, action_head_variant=variant)
    model = BalatroAgent(config, vocab)
    extra = {"agent_config": asdict(config), **reward_checkpoint_metadata(DEFAULT_REWARD_CONFIG)}
    path = save_checkpoint(model, tmp_path / "variant.pt", extra=extra)
    restored, _, _ = load_policy(path, vocab, "cpu")
    assert restored.config.action_head_variant == variant
    del extra["agent_config"]["action_head_variant"]
    invalid = save_checkpoint(model, tmp_path / "missing_variant.pt", extra=extra)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_policy(invalid, vocab, "cpu")
