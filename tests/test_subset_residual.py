import math

import torch

from pylatro_agent.subset_residual import SubsetResidualHead, corrected_subset_log_probs, subset_support


def test_zero_energy_preserves_policy_and_invalid_rows_have_no_gradient():
    torch.manual_seed(3)
    raw = torch.randn(3, 6)
    valid = torch.tensor([[True] * 6, [True, True, False, False, False, False], [False] * 6])
    base = raw.masked_fill(~valid, -1e8).log_softmax(-1)
    energies = torch.zeros_like(base, requires_grad=True)
    corrected = corrected_subset_log_probs(base, energies, valid)
    torch.testing.assert_close(corrected.exp() * valid, base.exp() * valid, atol=1e-7, rtol=1e-6)
    corrected[:, 0].sum().backward()
    assert torch.isfinite(corrected).all() and torch.isfinite(energies.grad).all()
    assert not energies.grad[~valid].any()


def test_energy_correction_can_overcome_reference_half_mass():
    base = torch.tensor([[0.5, 0.5]]).log()
    result = corrected_subset_log_probs(
        base, torch.tensor([[0.0, math.log(9)]]), torch.ones_like(base, dtype=torch.bool)
    )
    torch.testing.assert_close(result.exp(), torch.tensor([[0.1, 0.9]]))


def test_subset_scores_depend_on_entities_not_membership_column_order():
    torch.manual_seed(9)
    head = SubsetResidualHead(8, state_dim=12, hidden=16)
    torch.nn.init.normal_(head.score[-1].weight)
    cards, state = torch.randn(2, 5, 8), torch.randn(2, 12)
    membership = torch.tensor([[1, 1, 0, 0, 0], [1, 0, 0, 1, 1]], dtype=torch.float32)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(
        head(state, cards, membership), head(state, cards[:, permutation], membership[:, permutation])
    )


def test_joint_subset_head_learns_correlations_static_card_odds_cannot():
    # Four legal pairs: (0,2), (0,3), (1,2), (1,3). Static card logits
    # necessarily factor first-card and second-card probabilities. Thus the
    # two rows share the same second-card odds, and a 50/50 diagonal target
    # has best expected NLL log(4), despite target entropy being log(2).
    torch.manual_seed(11)
    card_logits = torch.randn(4)
    legacy = (card_logits[:2].log_softmax(-1)[:, None] + card_logits[2:].log_softmax(-1)[None]).flatten()
    torch.testing.assert_close(legacy[0] - legacy[1], legacy[2] - legacy[3])
    target = torch.tensor([[0.5, 0.0, 0.0, 0.5]])
    assert -(legacy * target[0]).sum() >= math.log(4) - 1e-6

    membership = torch.tensor([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    head = SubsetResidualHead(4, state_dim=4, hidden=16)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.02)
    base = torch.full((1, 4), -math.log(4))
    state, cards = torch.zeros(1, 4), torch.eye(4)[None]
    valid = torch.ones_like(base, dtype=torch.bool)
    for _ in range(300):
        logp = corrected_subset_log_probs(base, head(state, cards, membership)[:, 0], valid)
        loss = -(target * logp).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    learned = corrected_subset_log_probs(base, head(state, cards, membership)[:, 0], valid).exp()
    assert learned[0, [0, 3]].sum() > 0.99
    assert -(target * learned.log()).sum() < math.log(2) + 0.02


def test_available_scoring_preserves_sparse_slots_and_inference_cache_allows_backward():
    subset_support.cache_clear()
    with torch.inference_mode():
        indices, membership = subset_support(4)
    assert not torch.is_inference(indices) and not torch.is_inference(membership)
    head = SubsetResidualHead(8, state_dim=12, hidden=16)
    torch.nn.init.normal_(head.score[-1].weight)
    state, cards = torch.randn(3, 12), torch.randn(3, 4, 8)
    present = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]])
    dense = head.score_available(state, cards, present)
    direct = head(state[:1], cards[:1], membership)
    torch.testing.assert_close(dense[0][:, indices], direct[0])
    assert not dense[1].any()
    dense.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
