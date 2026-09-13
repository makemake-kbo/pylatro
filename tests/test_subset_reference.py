import math

import pytest
import torch

from pylatro_agent.action_grammar import ActionGrammarDistribution, ActionGrammarHead
from pylatro_agent.constants import (
    HAND_CANDIDATE_START,
    MAX_SEQ_LEN,
    NUM_ACTIONS,
    SCALAR_DIM,
    TOKEN_DIM,
    ActionRange,
)
from pylatro_agent.subset_actions import subset_index
from pylatro_agent.subset_residual import reference_hand_log_probs


def make_distribution(eps, *, temperature=1.0):
    torch.manual_seed(97)
    tokens = torch.zeros(3, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long)
    # Duplicate proposals must sum as atomic probability, not double-count NLL.
    tokens[0, HAND_CANDIDATE_START : HAND_CANDIDATE_START + 2, 5:7] = torch.tensor([1, 2])
    tokens[0, HAND_CANDIDATE_START : HAND_CANDIDATE_START + 2, 0] = 1
    output = ActionGrammarHead(8)(
        torch.randn(3, MAX_SEQ_LEN, 8),
        torch.ones(3, MAX_SEQ_LEN),
        tokens,
        torch.zeros(3, MAX_SEQ_LEN, dtype=torch.long),
        torch.zeros(3, SCALAR_DIM),
    )
    output.hand_card_logits = torch.randn_like(output.hand_card_logits, requires_grad=True)
    mask = torch.zeros(3, NUM_ACTIONS)
    subsets = [subset_index(cards) for cards in ([0, 1], [0, 2], [1], [0, 1, 2])]
    mask[0:2, [int(ActionRange.PLAY_SUBSET_START) + i for i in subsets]] = 1
    mask[2, int(ActionRange.SHOP_LEAVE)] = 1
    return ActionGrammarDistribution(
        output, mask, tokens=tokens, hand_ar_mixture_eps=eps, temperature=temperature
    ), subsets


@pytest.mark.parametrize("eps", [0.1, 0.5, 1.0])
def test_reference_matches_original_probability_and_gradient_with_duplicate_proposals(eps):
    dist, subsets = make_distribution(eps, temperature=torch.tensor([0.7, 1.0, 1.3]))
    reference = reference_hand_log_probs(dist, is_play=True)
    valid = dist._hand_valid_subset_mask(True)
    torch.testing.assert_close(reference.exp() * valid, dist._hand_probabilities(is_play=True), rtol=2e-5, atol=1e-7)
    actions = torch.tensor([int(ActionRange.PLAY_SUBSET_START) + subsets[1]] * 2 + [int(ActionRange.SHOP_LEAVE)])
    expected = dist.log_prob(actions)[:2].sum()
    actual = reference[:2, subsets[1]].sum()
    gradient = torch.autograd.grad(actual, dist.output.hand_card_logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected, dist.output.hand_card_logits)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-5, atol=1e-6)
    assert torch.isfinite(reference).all()


def test_reference_retains_finite_logs_for_extremely_unlikely_legal_actions():
    dist, subsets = make_distribution(0.5)
    with torch.no_grad():
        dist.output.hand_card_logits[:, 0, :3] = torch.tensor([1000.0, -1000.0, 0.0])
    logp = reference_hand_log_probs(dist, is_play=True)
    assert logp[1, subsets[0]] < -1000
    assert torch.isfinite(logp).all()
    # An empty candidate component falls back to the full AR policy.
    assert math.isclose(float(logp[1].detach().exp().sum()), 1.0, abs_tol=2e-5)


def test_reference_rejects_candidate_only_policy_without_full_support():
    dist, _ = make_distribution(0.0)
    with pytest.raises(ValueError, match="positive AR mixture"):
        reference_hand_log_probs(dist, is_play=True)
