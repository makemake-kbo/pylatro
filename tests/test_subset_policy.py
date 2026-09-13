from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from pylatro import load_game_data
from pylatro_agent.action_grammar import ActionGrammarDistribution, ActionGrammarHead
from pylatro_agent.agent import AgentConfig, BalatroAgent
from pylatro_agent.architecture_transfer import add_sale_residual, add_subset_residual
from pylatro_agent.checkpoint import save_checkpoint
from pylatro_agent.constants import MAX_SEQ_LEN, NUM_ACTIONS, SCALAR_DIM, TOKEN_DIM, ActionRange
from pylatro_agent.env import BalatroEnv
from pylatro_agent.learning_bottleneck import hand_details
from pylatro_agent.reward import reward_checkpoint_metadata
from pylatro_agent.subset_actions import HAND_SUBSETS, subset_index
from pylatro_agent.training.ppo_policy import _grammar_distribution as ppo_distribution
from pylatro_agent.training.supervised import SupervisedConfig, _collate_batch, _grammar_distribution
from pylatro_agent.transfer import load_policy
from pylatro_agent.vocab import build_vocab


@pytest.mark.parametrize("with_sale", [False, True])
def test_neutral_subset_migration_preserves_policy_and_checkpoint_paths(tmp_path, with_sale):
    torch.manual_seed(982)
    torch.set_num_threads(1)
    data = load_game_data()
    vocab = build_vocab(data)
    source = BalatroAgent(
        AgentConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0, hand_ar_mixture_eps=0.5), vocab
    ).eval()
    if with_sale:
        source, _ = add_sale_residual(source, vocab, "entity_sale_residual_v1")
        torch.nn.init.normal_(source.action_grammar_head.sale_residual.net[-1].weight)
        source.eval()
    model, migration = add_subset_residual(source, vocab)
    model.eval()
    env = BalatroEnv(seed=1729, data=data, vocab=vocab, enable_teacher=False, deck_key="b_blue")
    obs, _ = env.reset()
    obs, *_ = env.step(int(ActionRange.BLIND_PLAY))
    rows = [dict(obs=obs, action=int(np.flatnonzero(obs["action_mask"])[0]), return_target=0.0, won=False, max_ante=1)]
    batch = _collate_batch(rows, torch.device("cpu"), SupervisedConfig())
    before, _ = _grammar_distribution(source, batch)
    after, _ = _grammar_distribution(model, batch)
    for family in (True, False):
        torch.testing.assert_close(
            after._hand_probabilities(is_play=family), before._hand_probabilities(is_play=family), atol=1e-7, rtol=2e-5
        )
    torch.testing.assert_close(before.entropy(), after.entropy(), atol=2e-6, rtol=2e-5)
    assert torch.equal(before.mode(), after.mode())
    assert migration["observation_semantics_changed"] is False
    for key, value in source.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, atol=0, rtol=0)
    # Exercise nonzero corrections through the ordinary and gathered paths.
    torch.nn.init.normal_(model.action_grammar_head.subset_residual.score[-1].weight, std=0.2)
    ordinary, _ = _grammar_distribution(model, batch)
    gathered, _ = _grammar_distribution(torch.nn.DataParallel(model), batch)
    rollout, _ = ppo_distribution(model, batch)
    gathered_rollout, _ = ppo_distribution(torch.nn.DataParallel(model), batch)
    for alternative in (gathered, rollout, gathered_rollout):
        torch.testing.assert_close(ordinary.log_prob(batch["actions"]), alternative.log_prob(batch["actions"]))
        torch.testing.assert_close(ordinary.entropy(), alternative.entropy())
        assert torch.equal(ordinary.mode(), alternative.mode())
    (-ordinary.log_prob(batch["actions"]).mean() + .01 * ordinary.entropy().mean()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters())
    assert model.action_grammar_head.subset_residual.score[-1].weight.grad.abs().sum() > 0
    path = save_checkpoint(
        model,
        tmp_path / "subset.pt",
        extra={
            "agent_config": asdict(model.config),
            **reward_checkpoint_metadata(env._reward_config),
        },
    )
    restored, _, _ = load_policy(path, vocab, "cpu")
    reloaded, _ = _grammar_distribution(restored, batch)
    torch.testing.assert_close(ordinary.log_prob(batch["actions"]), reloaded.log_prob(batch["actions"]))
    with pytest.raises(ValueError, match="positive AR mixture"):
        ActionGrammarDistribution(ordinary.output, batch["action_mask"], hand_ar_mixture_eps=0.0)
    with pytest.raises(ValueError, match="Legacy mixture ceilings"):
        hand_details(ordinary, batch["actions"])
    env.close()


def test_subset_sampling_mode_entropy_and_likelihood_agree_on_atomic_actions():
    torch.manual_seed(410)
    tokens = torch.zeros(1, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    output = ActionGrammarHead(8)(
        torch.randn(1, MAX_SEQ_LEN, 8),
        torch.ones(1, MAX_SEQ_LEN),
        tokens,
        torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long),
        torch.zeros(1, SCALAR_DIM),
    )
    output.hand_card_logits = torch.randn_like(output.hand_card_logits)
    energy = torch.randn(1, 2, len(HAND_SUBSETS), requires_grad=True)
    output = replace(output, hand_subset_energies=energy)
    actions = [int(ActionRange.PLAY_SUBSET_START) + subset_index(cards) for cards in ([0], [0, 1], [0, 2])]
    actions += [int(ActionRange.DISCARD_SUBSET_START) + subset_index(cards) for cards in ([0, 1], [1, 2])]
    actions += [int(ActionRange.SHOP_LEAVE)]
    mask = torch.zeros(1, NUM_ACTIONS)
    mask[0, actions] = 1
    dist = ActionGrammarDistribution(
        output, mask, tokens=tokens, hand_ar_mixture_eps=0.5, temperature=torch.tensor([0.8])
    )
    probs = torch.stack([dist.log_prob(torch.tensor([a])).exp()[0] for a in actions])
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))
    entropy = -(probs * probs.log()).sum()
    torch.testing.assert_close(dist.entropy().sum(), entropy, atol=2e-6, rtol=2e-5)
    expected_gradient = torch.autograd.grad(entropy, energy, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(dist.entropy().sum(), energy)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=2e-5)
    assert dist.mode().item() == actions[int(probs.argmax())]
    counts = {action: 0 for action in actions}
    with torch.no_grad():
        for _ in range(1024):
            selected = dist.sample().item()
            assert selected in counts
            counts[selected] += 1
    observed = torch.tensor([counts[action] / 1024 for action in actions])
    tolerance = 6 * (probs.detach() * (1 - probs.detach()) / 1024).sqrt() + 0.005
    assert ((observed - probs.detach()).abs() < tolerance).all()


def test_subset_metadata_and_energy_shape_reject_incompatible_architectures():
    with pytest.raises(ValueError, match="positive AR mixture"):
        AgentConfig(action_head_variant="subset_residual_v1", hand_ar_mixture_eps=0.0)
    tokens = torch.zeros(1, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    output = ActionGrammarHead(8)(
        torch.zeros(1, MAX_SEQ_LEN, 8),
        torch.ones(1, MAX_SEQ_LEN),
        tokens,
        torch.zeros(1, MAX_SEQ_LEN, dtype=torch.long),
        torch.zeros(1, SCALAR_DIM),
    )
    output.hand_subset_energies = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match="energy shape"):
        ActionGrammarDistribution(output, torch.ones(1, NUM_ACTIONS), hand_ar_mixture_eps=0.5)
