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
    MAX_CONSUMABLE_HAND_TARGETS,
    MAX_CONSUMABLE_SLOTS,
    MAX_HAND_SIZE,
    MAX_JOKER_SLOTS,
    MAX_PACK_CARDS,
    MAX_SHOP_ITEMS,
    NUM_ACTIONS,
)
from pylatro_agent.subset_actions import consumable_subset_index, subset_index


def _blank_output(batch_size: int) -> ActionGrammarOutput:
    return ActionGrammarOutput(
        macro_logits=torch.zeros(batch_size, NUM_GRAMMAR_ACTIONS),
        hand_count_logits=torch.zeros(batch_size, 2, 5),
        hand_card_logits=torch.zeros(batch_size, 2, MAX_HAND_SIZE),
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
