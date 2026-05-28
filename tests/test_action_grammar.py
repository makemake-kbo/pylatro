from __future__ import annotations

import torch

from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.action_grammar import (
    ACTION_TYPE_TO_GRAMMAR_INDEX,
    NUM_GRAMMAR_ACTIONS,
    ActionGrammarDistribution,
    ActionGrammarOutput,
)
from pylatro_agent.constants import (
    HAND_CANDIDATE_MAX,
    HAND_CANDIDATE_START,
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_DISCARD_CANDIDATES,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_PLAY_CANDIDATES,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
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
    )


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
        attn, tokens, token_types, scalars,
    )

    assert output.candidate_play_logits.shape == (batch, HAND_CANDIDATE_MAX)
    assert output.candidate_discard_logits.shape == (batch, HAND_CANDIDATE_MAX)
