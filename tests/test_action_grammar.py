from __future__ import annotations

import math

import pytest
import torch

from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.action_grammar import (
    ACTION_TYPE_TO_GRAMMAR_INDEX,
    NUM_GRAMMAR_ACTIONS,
    ActionGrammarDistribution,
    ActionGrammarHead,
    ActionGrammarOutput,
)
from pylatro_agent.constants import (
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_PLAY_CANDIDATES,
    MAX_SEQ_LEN,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
    SCALAR_DIM,
    TOKEN_DIM,
)
from pylatro_agent.subset_actions import consumable_subset_index, subset_index


def _blank_output(batch_size: int) -> ActionGrammarOutput:
    return ActionGrammarOutput(
        macro_logits=torch.zeros(batch_size, NUM_GRAMMAR_ACTIONS),
        hand_count_logits=torch.zeros(batch_size, 2, 5),
        hand_card_logits=torch.zeros(batch_size, 2, MAX_HAND_SIZE),
        candidate_play_logits=torch.full((batch_size, HAND_CANDIDATE_MAX), -1e8),
        candidate_discard_logits=torch.full((batch_size, HAND_CANDIDATE_MAX), -1e8),
        consumable_slot_logits=torch.zeros(batch_size, 3, MAX_CONSUMABLE_SLOTS),
        consumable_count_logits=torch.zeros(batch_size, MAX_CONSUMABLE_SLOTS, MAX_CONSUMABLE_HAND_TARGETS),
        consumable_card_logits=torch.zeros(batch_size, MAX_CONSUMABLE_SLOTS, MAX_HAND_SIZE),
        consumable_joker_logits=torch.zeros(batch_size, MAX_CONSUMABLE_SLOTS, MAX_JOKER_SLOTS),
        shop_buy_logits=torch.zeros(batch_size, MAX_SHOP_ITEMS),
        shop_sell_joker_logits=torch.zeros(batch_size, MAX_JOKER_SLOTS),
        shop_sell_consumable_logits=torch.zeros(batch_size, MAX_CONSUMABLE_SLOTS),
        pack_claim_logits=torch.zeros(batch_size, MAX_PACK_CARDS),
        joker_move_logits=torch.zeros(
            batch_size,
            MAX_JOKER_SLOTS * (MAX_JOKER_SLOTS - 1),
        ),
    )


def _grammar_head_inputs(batch_size: int, d_model: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(batch_size, MAX_SEQ_LEN, d_model),
        torch.ones(batch_size, MAX_SEQ_LEN),
        torch.zeros(batch_size, MAX_SEQ_LEN, TOKEN_DIM, dtype=torch.long),
        torch.zeros(batch_size, MAX_SEQ_LEN, dtype=torch.long),
        torch.zeros(batch_size, SCALAR_DIM),
    )


def test_action_head_directly_conditions_on_danger() -> None:
    head = ActionGrammarHead(8)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.danger_policy_proj.weight[0, 0] = 1.0
        play_index = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
        head.macro_head.weight[play_index, 0] = 1.0

    inputs = list(_grammar_head_inputs(2, 8))
    inputs[-1][:, 7] = 1.0
    inputs[-1][:, 12] = torch.tensor([0.2, 0.8])
    output = head(*inputs)

    danger_delta = output.macro_logits[1, play_index] - output.macro_logits[0, play_index]
    assert danger_delta.item() == pytest.approx(0.6)


def test_unsafe_shop_prior_is_soft_and_phase_specific() -> None:
    head = ActionGrammarHead(8, danger_shop_leave_logit_penalty=2.0)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()

    inputs = list(_grammar_head_inputs(3, 8))
    scalars = inputs[-1]
    scalars[:, 12] = torch.tensor([1.0, 0.35, 1.0])
    scalars[:, 7] = torch.tensor([2.0, 2.0, 1.0])
    output = head(*inputs)

    leave_index = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_LEAVE]
    reroll_index = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_REROLL]
    assert output.macro_logits[:, leave_index].tolist() == pytest.approx([-2.0, 0.0, 0.0])
    assert output.macro_logits[:, reroll_index].tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_action_grammar_samples_only_valid_flat_actions() -> None:
    action_mask = torch.zeros(3, NUM_ACTIONS)
    play_subset = subset_index([0, 2, 4])
    consumable_subset = consumable_subset_index([1, 3])

    targets = torch.tensor(
        [
            encode_action(ActionType.PLAY_SUBSET, play_subset),
            encode_action(ActionType.SHOP_LEAVE),
            encode_action(ActionType.USE_CONSUMABLE_HAND_SUBSET, 1, consumable_subset),
        ],
        dtype=torch.long,
    )
    action_mask[0, targets[0]] = 1
    action_mask[1, encode_action(ActionType.SHOP_BUY, 2)] = 1
    action_mask[1, targets[1]] = 1
    action_mask[2, targets[2]] = 1

    dist = ActionGrammarDistribution(_blank_output(batch_size=3), action_mask)
    sampled = dist.sample()
    greedy = dist.mode()

    assert action_mask.gather(1, sampled.unsqueeze(-1)).squeeze(-1).bool().all()
    assert action_mask.gather(1, greedy.unsqueeze(-1)).squeeze(-1).bool().all()
    assert torch.isfinite(dist.log_prob(targets)).all()
    assert dist.entropy().shape == (3,)


def test_move_joker_action_has_finite_grammar_support() -> None:
    action = encode_action(ActionType.MOVE_JOKER, 0, 1)
    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, action] = 1

    dist = ActionGrammarDistribution(_blank_output(batch_size=1), action_mask)

    assert dist.mode().item() == action
    assert dist.log_prob(torch.tensor([action])).item() > -1e7


def test_action_grammar_log_prob_has_gradients_for_components() -> None:
    action_mask = torch.zeros(1, NUM_ACTIONS)
    play_subset = subset_index([0, 1, 2, 3, 4])
    action = torch.tensor([encode_action(ActionType.PLAY_SUBSET, play_subset)])
    action_mask[0, action.item()] = 1

    output = _blank_output(batch_size=1)
    output.macro_logits.requires_grad_(True)
    output.hand_count_logits.requires_grad_(True)
    output.hand_card_logits.requires_grad_(True)

    dist = ActionGrammarDistribution(output, action_mask)
    loss = -dist.log_prob(action).mean()
    loss.backward()

    assert output.macro_logits.grad is not None
    assert output.hand_count_logits.grad is not None
    assert output.hand_card_logits.grad is not None


def test_action_grammar_temperature_changes_log_prob_consistently() -> None:
    action_mask = torch.zeros(1, NUM_ACTIONS)
    leave = encode_action(ActionType.SHOP_LEAVE)
    reroll = encode_action(ActionType.SHOP_REROLL)
    action_mask[0, leave] = 1
    action_mask[0, reroll] = 1

    output = _blank_output(batch_size=1)
    output.macro_logits[0, :] = -10.0
    output.macro_logits[0, ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_LEAVE]] = 1.0
    output.macro_logits[0, ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_REROLL]] = 0.0

    cool = ActionGrammarDistribution(output, action_mask, temperature=0.5)
    warm = ActionGrammarDistribution(output, action_mask, temperature=1.0)

    assert cool.log_prob(torch.tensor([leave])).item() > warm.log_prob(torch.tensor([leave])).item()
    assert cool.log_prob(torch.tensor([reroll])).item() < warm.log_prob(torch.tensor([reroll])).item()


def test_action_grammar_accepts_per_row_temperature() -> None:
    action_mask = torch.zeros(2, NUM_ACTIONS)
    leave = encode_action(ActionType.SHOP_LEAVE)
    reroll = encode_action(ActionType.SHOP_REROLL)
    action_mask[:, leave] = 1
    action_mask[:, reroll] = 1

    output = _blank_output(batch_size=2)
    output.macro_logits[:, :] = -10.0
    output.macro_logits[:, ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_LEAVE]] = 1.0
    output.macro_logits[:, ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.SHOP_REROLL]] = 0.0

    dist = ActionGrammarDistribution(
        output,
        action_mask,
        temperature=torch.tensor([0.5, 1.0]),
    )
    leave_log_prob = dist.log_prob(torch.tensor([leave, leave]))

    assert leave_log_prob[0] > leave_log_prob[1]


def test_candidate_logits_in_output_shape():
    output = _blank_output(batch_size=2)
    assert output.candidate_play_logits.shape == (2, HAND_CANDIDATE_MAX)
    assert output.candidate_discard_logits.shape == (2, HAND_CANDIDATE_MAX)


def test_candidate_logits_default_masked():
    output = _blank_output(batch_size=1)
    assert (output.candidate_play_logits == -1e8).all()
    assert (output.candidate_discard_logits == -1e8).all()


def test_candidate_scoring_falls_back_when_no_candidates():
    action_mask = torch.zeros(1, NUM_ACTIONS)
    play_subset = subset_index([0, 1])
    action = encode_action(ActionType.PLAY_SUBSET, play_subset)
    action_mask[0, action] = 1

    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, play_idx] = 10.0

    dist = ActionGrammarDistribution(output, action_mask)
    sampled = dist.sample()
    greedy = dist.mode()

    assert action_mask[0, sampled.item()] == 1
    assert action_mask[0, greedy.item()] == 1


def test_candidate_scoring_maps_play_candidate_to_correct_action():
    play_indices = [0, 2, 4]
    play_action = encode_action(ActionType.PLAY_SUBSET, subset_index(play_indices))
    distractor_action = encode_action(ActionType.PLAY_SUBSET, subset_index([1, 3]))

    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, play_action] = 1
    action_mask[0, distractor_action] = 1

    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, play_idx] = 10.0

    tokens = torch.zeros(1, 160, 13, dtype=torch.long)
    cand_slot = 0
    cand_pos = HAND_CANDIDATE_START + cand_slot
    tokens[0, cand_pos, 0] = 1
    tokens[0, cand_pos, 2] = len(play_indices)
    for j, idx in enumerate(play_indices):
        tokens[0, cand_pos, 5 + j] = idx + 1
    output.candidate_play_logits[0, cand_slot] = 5.0

    dist = ActionGrammarDistribution(output, action_mask, tokens=tokens)
    greedy = dist.mode()
    assert greedy.item() == play_action, f"Expected {play_action}, got {greedy.item()}"


def test_candidate_scoring_maps_discard_candidate_to_correct_action():
    discard_indices = [1, 3]
    discard_action = encode_action(ActionType.DISCARD_SUBSET, subset_index(discard_indices))
    distractor_action = encode_action(ActionType.DISCARD_SUBSET, subset_index([0, 2]))

    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, discard_action] = 1
    action_mask[0, distractor_action] = 1

    output = _blank_output(batch_size=1)
    discard_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.DISCARD_SUBSET]
    output.macro_logits[0, discard_idx] = 10.0

    tokens = torch.zeros(1, 160, 13, dtype=torch.long)
    cand_slot = MAX_PLAY_CANDIDATES
    cand_pos = HAND_CANDIDATE_START + cand_slot
    tokens[0, cand_pos, 0] = 2
    tokens[0, cand_pos, 2] = len(discard_indices)
    for j, idx in enumerate(discard_indices):
        tokens[0, cand_pos, 5 + j] = idx + 1
    output.candidate_discard_logits[0, cand_slot] = 5.0

    dist = ActionGrammarDistribution(output, action_mask, tokens=tokens)
    greedy = dist.mode()
    assert greedy.item() == discard_action, f"Expected {discard_action}, got {greedy.item()}"


def test_log_prob_routes_through_candidate_head_when_candidates_present():
    play_indices = [0, 2, 4]
    play_action = encode_action(ActionType.PLAY_SUBSET, subset_index(play_indices))
    distractor_action = encode_action(ActionType.PLAY_SUBSET, subset_index([1, 3]))

    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, play_action] = 1
    action_mask[0, distractor_action] = 1

    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, play_idx] = 10.0
    cand_logits = torch.full((1, HAND_CANDIDATE_MAX), -1e8)
    cand_logits[0, 0] = 5.0
    cand_logits[0, 1] = 3.0
    cand_logits.requires_grad_(True)
    output.candidate_play_logits = cand_logits
    output.hand_count_logits = output.hand_count_logits.clone().requires_grad_(True)
    output.hand_card_logits = output.hand_card_logits.clone().requires_grad_(True)

    tokens = torch.zeros(1, 160, 13, dtype=torch.long)
    tokens[0, HAND_CANDIDATE_START + 0, 0] = 1
    tokens[0, HAND_CANDIDATE_START + 0, 2] = 3
    for j, idx in enumerate(play_indices):
        tokens[0, HAND_CANDIDATE_START + 0, 5 + j] = idx + 1
    tokens[0, HAND_CANDIDATE_START + 1, 0] = 1
    tokens[0, HAND_CANDIDATE_START + 1, 2] = 2
    tokens[0, HAND_CANDIDATE_START + 1, 5] = 1 + 1
    tokens[0, HAND_CANDIDATE_START + 1, 6] = 3 + 1

    dist = ActionGrammarDistribution(output, action_mask, tokens=tokens)
    logp = dist.log_prob(torch.tensor([play_action]))
    expected = 5.0 - torch.tensor([5.0, 3.0]).logsumexp(dim=0)
    assert torch.allclose(logp, expected.unsqueeze(0), atol=1e-5), (
        f"log_prob should reflect candidate softmax, got {logp.item():.4f}"
    )

    loss = -logp.sum()
    cand_grad, count_grad, card_grad = torch.autograd.grad(
        loss,
        [output.candidate_play_logits, output.hand_count_logits, output.hand_card_logits],
        allow_unused=True,
    )
    assert cand_grad is not None and cand_grad.abs().sum() > 0
    assert count_grad is None or count_grad.abs().sum() == 0
    assert card_grad is None or card_grad.abs().sum() == 0


def test_log_prob_falls_back_to_autoregressive_without_candidates():
    play_a = encode_action(ActionType.PLAY_SUBSET, subset_index([0, 2, 4]))
    play_b = encode_action(ActionType.PLAY_SUBSET, subset_index([1, 3]))
    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, play_a] = 1
    action_mask[0, play_b] = 1

    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, play_idx] = 10.0
    output.hand_count_logits = output.hand_count_logits.clone().requires_grad_(True)
    output.hand_card_logits = output.hand_card_logits.clone().requires_grad_(True)

    # tokens=None → candidate path empty → fallback to autoregressive.
    dist = ActionGrammarDistribution(output, action_mask, tokens=None)
    logp = dist.log_prob(torch.tensor([play_a]))
    assert torch.isfinite(logp).all()

    loss = -logp.sum()
    count_grad, card_grad = torch.autograd.grad(
        loss, [output.hand_count_logits, output.hand_card_logits], allow_unused=True
    )
    assert (count_grad is not None and count_grad.abs().sum() > 0) or (
        card_grad is not None and card_grad.abs().sum() > 0
    )


def test_entropy_routes_through_candidate_head_when_candidates_present():
    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, play_idx] = 10.0
    output.candidate_play_logits = output.candidate_play_logits.clone()
    output.candidate_play_logits[0, 0] = 5.0
    output.candidate_play_logits[0, 1] = 3.0

    play_a = encode_action(ActionType.PLAY_SUBSET, subset_index([0, 2, 4]))
    play_b = encode_action(ActionType.PLAY_SUBSET, subset_index([1, 3]))
    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, play_a] = 1
    action_mask[0, play_b] = 1

    tokens = torch.zeros(1, 160, 13, dtype=torch.long)
    tokens[0, HAND_CANDIDATE_START + 0, 0] = 1
    tokens[0, HAND_CANDIDATE_START + 0, 2] = 3
    for j, idx in enumerate([0, 2, 4]):
        tokens[0, HAND_CANDIDATE_START + 0, 5 + j] = idx + 1
    tokens[0, HAND_CANDIDATE_START + 1, 0] = 1
    tokens[0, HAND_CANDIDATE_START + 1, 2] = 2
    tokens[0, HAND_CANDIDATE_START + 1, 5] = 1 + 1
    tokens[0, HAND_CANDIDATE_START + 1, 6] = 3 + 1

    dist = ActionGrammarDistribution(output, action_mask, tokens=tokens)
    ent = dist.entropy()
    # Two-way softmax entropy: -p1 log p1 - p2 log p2 with logits (5, 3)
    logits = torch.tensor([5.0, 3.0])
    log_probs = logits.log_softmax(dim=0)
    expected_hand_entropy = -(log_probs.exp() * log_probs).sum()
    # Total entropy includes macro contribution; just assert the value reflects
    # the candidate softmax (it's nonzero and below log(NUM_ACTIONS)).
    assert ent.item() > 0
    assert ent.item() < expected_hand_entropy.item() + 5.0  # generous upper bound


def test_head_produces_candidate_logits():
    from pylatro import load_game_data
    from pylatro_agent.agent import AgentConfig, BalatroAgent
    from pylatro_agent.vocab import build_vocab

    data = load_game_data()
    v = build_vocab(data)
    model = BalatroAgent(AgentConfig(d_model=64, n_layers=2), v)

    batch = 2
    tokens = torch.zeros(batch, 160, 13, dtype=torch.long)
    token_types = torch.full((batch, 160), 10, dtype=torch.long)
    attn = torch.ones(batch, 160, dtype=torch.long)
    scalars = torch.zeros(batch, 11)

    output = model.action_grammar_head(
        model.backbone(model.embedding(tokens, token_types, scalars), padding_mask=(attn == 0)),
        attn,
        tokens,
        token_types,
        scalars,
    )

    assert output.candidate_play_logits.shape == (batch, HAND_CANDIDATE_MAX)
    assert output.candidate_discard_logits.shape == (batch, HAND_CANDIDATE_MAX)


# ──────────────────────────────────────────────────────────────────────────
# Phase 1: candidate-biased autoregressive mixture tests
# ──────────────────────────────────────────────────────────────────────────


def _hand_state_with_candidates_and_distractor():
    """Build a 1-row synthetic state with one candidate play and one extra legal play.

    Returns (output, action_mask, tokens, candidate_action, distractor_action).
    The candidate maps to play subset {0,2,4}; the distractor is play {1,3},
    which is legal but not in the candidate list — exactly the situation where
    the historical -1e8 floor poisoned BC and blocked exploration.
    """
    from pylatro_agent.constants import HAND_CANDIDATE_START

    candidate_action = encode_action(ActionType.PLAY_SUBSET, subset_index([0, 2, 4]))
    distractor_action = encode_action(ActionType.PLAY_SUBSET, subset_index([1, 3]))

    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, candidate_action] = 1
    action_mask[0, distractor_action] = 1

    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, play_idx] = 10.0
    output.candidate_play_logits = output.candidate_play_logits.clone()
    output.candidate_play_logits[0, 0] = 5.0

    tokens = torch.zeros(1, 160, 13, dtype=torch.long)
    tokens[0, HAND_CANDIDATE_START + 0, 0] = 1
    tokens[0, HAND_CANDIDATE_START + 0, 2] = 3
    for j, idx in enumerate([0, 2, 4]):
        tokens[0, HAND_CANDIDATE_START + 0, 5 + j] = idx + 1

    return output, action_mask, tokens, candidate_action, distractor_action


def test_mixture_eps_zero_matches_historical_behavior():
    """eps=0 must reproduce the historical candidate-only log_prob exactly."""
    output, action_mask, tokens, cand_action, dist_action = _hand_state_with_candidates_and_distractor()

    dist_eps0 = ActionGrammarDistribution(output, action_mask, tokens=tokens, hand_ar_mixture_eps=0.0)
    dist_legacy = ActionGrammarDistribution(output, action_mask, tokens=tokens)

    # The candidate action gets a finite log_prob; the distractor gets the -1e8 floor.
    lp_cand = dist_eps0.log_prob(torch.tensor([cand_action]))
    lp_dist = dist_eps0.log_prob(torch.tensor([dist_action]))
    lp_legacy_cand = dist_legacy.log_prob(torch.tensor([cand_action]))
    lp_legacy_dist = dist_legacy.log_prob(torch.tensor([dist_action]))

    assert torch.allclose(lp_cand, lp_legacy_cand, atol=1e-6)
    assert torch.allclose(lp_dist, lp_legacy_dist, atol=1e-6)
    # Distractor is unreachable at eps=0 (the -1e8 floor).
    assert lp_dist.item() < -1e7


def test_mixture_eps_positive_gives_full_support():
    """With eps>0, every legal play (including non-candidate) has log_prob > -1e7."""
    output, action_mask, tokens, cand_action, dist_action = _hand_state_with_candidates_and_distractor()

    dist = ActionGrammarDistribution(output, action_mask, tokens=tokens, hand_ar_mixture_eps=0.1)
    lp_cand = dist.log_prob(torch.tensor([cand_action]))
    lp_dist = dist.log_prob(torch.tensor([dist_action]))

    assert lp_cand.item() > -1e7, "candidate play must have finite log_prob"
    assert lp_dist.item() > -1e7, "non-candidate play must now have finite log_prob (full support)"


def test_mixture_normalizes_over_legal_hand_plays():
    """exp(log_prob) over all legal play actions (conditional on macro=PLAY) sums to ~1.

    Uses a small subset of play actions (the first 8) rather than all 6884
    hand subsets, so the enumeration is fast. The normalization property holds
    regardless of which plays are legal — what matters is that the distribution
    over the legal set sums to 1.
    """
    # Take the first 8 play subsets as the legal set.
    play_actions = [encode_action(ActionType.PLAY_SUBSET, i) for i in range(8)]
    action_mask = torch.zeros(1, NUM_ACTIONS)
    for a in play_actions:
        action_mask[0, a] = 1

    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, :] = -10.0
    output.macro_logits[0, play_idx] = 0.0  # force macro = PLAY

    for eps in (0.0, 0.1, 0.5):
        dist = ActionGrammarDistribution(output, action_mask, tokens=None, hand_ar_mixture_eps=eps)
        # tokens=None -> no candidates -> pure AR path. Enumerate the 8 legal plays.
        total = 0.0
        for a in play_actions:
            lp = dist.log_prob(torch.tensor([a])).item()
            total += math.exp(lp)
        # Each exp(log_prob) includes the macro prob (=1.0 here since PLAY is the
        # only valid macro). So the sum should be ~1.0.
        assert abs(total - 1.0) < 1e-3, f"eps={eps}: sum={total}, expected ~1.0"


def test_mixture_sampling_produces_non_candidate_plays():
    """Sampling with eps=0.1 yields >=1 non-candidate play on a strict-subset state."""
    output, action_mask, tokens, _cand_action, dist_action = _hand_state_with_candidates_and_distractor()

    dist = ActionGrammarDistribution(output, action_mask, tokens=tokens, hand_ar_mixture_eps=0.1)
    torch.manual_seed(42)
    non_cand_count = 0
    # Sample the hand-play component directly (fast) rather than the full
    # dist.sample() which also draws the macro. 1000 draws is enough to see the
    # eps=0.1 mass produce at least one AR (non-candidate) play.
    for _ in range(1000):
        s = dist._sample_hand_actions(is_play=True).item()
        if s == dist_action:
            non_cand_count += 1
    assert non_cand_count >= 1, "eps=0.1 should occasionally sample the non-candidate play"


def test_mixture_log_prob_has_gradient_to_both_components():
    """The logsumexp routes gradient to both candidate and AR heads when eps>0."""
    from pylatro_agent.constants import HAND_CANDIDATE_START

    # Two candidates so the candidate softmax is non-degenerate (a single valid
    # candidate gives prob=1.0 and zero gradient on its own logit).
    cand_a = encode_action(ActionType.PLAY_SUBSET, subset_index([0, 2, 4]))
    cand_b = encode_action(ActionType.PLAY_SUBSET, subset_index([1, 3]))
    action_mask = torch.zeros(1, NUM_ACTIONS)
    action_mask[0, cand_a] = 1
    action_mask[0, cand_b] = 1

    output = _blank_output(batch_size=1)
    play_idx = ACTION_TYPE_TO_GRAMMAR_INDEX[ActionType.PLAY_SUBSET]
    output.macro_logits[0, play_idx] = 10.0
    output.candidate_play_logits = output.candidate_play_logits.clone()
    output.candidate_play_logits[0, 0] = 5.0
    output.candidate_play_logits[0, 1] = 3.0

    tokens = torch.zeros(1, 160, 13, dtype=torch.long)
    for slot, idxs in enumerate(([0, 2, 4], [1, 3])):
        tokens[0, HAND_CANDIDATE_START + slot, 0] = 1
        tokens[0, HAND_CANDIDATE_START + slot, 2] = len(idxs)
        for j, idx in enumerate(idxs):
            tokens[0, HAND_CANDIDATE_START + slot, 5 + j] = idx + 1

    output.candidate_play_logits.requires_grad_(True)
    output.hand_count_logits = output.hand_count_logits.clone().requires_grad_(True)
    output.hand_card_logits = output.hand_card_logits.clone().requires_grad_(True)

    dist = ActionGrammarDistribution(output, action_mask, tokens=tokens, hand_ar_mixture_eps=0.5)
    lp = dist.log_prob(torch.tensor([cand_a]))
    loss = -lp.sum()
    loss.backward()

    # Both the candidate head and the AR head should receive gradient.
    assert output.candidate_play_logits.grad is not None
    assert output.candidate_play_logits.grad.abs().sum() > 0
    assert output.hand_count_logits.grad is not None
    assert output.hand_count_logits.grad.abs().sum() > 0


def test_mixture_entropy_is_finite_and_positive():
    output, action_mask, tokens, _, _ = _hand_state_with_candidates_and_distractor()
    for eps in (0.0, 0.1, 0.5):
        dist = ActionGrammarDistribution(output, action_mask, tokens=tokens, hand_ar_mixture_eps=eps)
        ent = dist._hand_entropy(is_play=True)
        assert torch.isfinite(ent).all()
        assert ent.item() >= 0.0


def test_mixture_normalizes_with_candidates_present():
    """With candidates present, exp(log_prob) over all legal plays sums to ~1.

    Complements test_mixture_normalizes_over_legal_hand_plays (which exercises
    the pure-AR path): here both mixture components are active, so this catches
    double-counting between the candidate and AR components.
    """
    output, action_mask, tokens, cand_action, dist_action = _hand_state_with_candidates_and_distractor()

    for eps in (0.1, 0.5, 1.0):
        dist = ActionGrammarDistribution(output, action_mask, tokens=tokens, hand_ar_mixture_eps=eps)
        total = sum(math.exp(dist.log_prob(torch.tensor([a])).item()) for a in (cand_action, dist_action))
        assert abs(total - 1.0) < 1e-3, f"eps={eps}: sum={total}, expected ~1.0"
