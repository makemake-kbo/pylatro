import numpy as np
import pytest
import torch

from pylatro_agent.constants import NUM_ACTIONS, ActionRange
from pylatro_agent.sale_choice_oracle import sample_joker_sale


def test_conditional_sale_sampling_is_legal_and_preserves_policy_rng():
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    lo, hi = int(ActionRange.SHOP_SELL_JOKER_START), int(ActionRange.SHOP_SELL_JOKER_END)
    mask[lo + 2] = True
    logits = torch.arange(hi - lo + 1, dtype=torch.float32)
    before = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(1729)
    assert sample_joker_sale(logits, mask, generator) == lo + 2
    assert torch.equal(before, torch.get_rng_state())
    mask[:] = False
    with pytest.raises(ValueError, match="No legal"):
        sample_joker_sale(logits, mask, generator)
