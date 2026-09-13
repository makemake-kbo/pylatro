import pytest

from pylatro_agent.action import ActionType, encode_action
from pylatro_agent.constants import SubPhase
from pylatro_agent.sale_oracle import replace_shop_action


@pytest.mark.parametrize("family", [ActionType.SHOP_SELL_JOKER, ActionType.SHOP_BUY, ActionType.SHOP_LEAVE])
def test_shop_partition_is_complementary_and_never_controls_other_phases(family):
    action = encode_action(family)
    sale = family == ActionType.SHOP_SELL_JOKER
    assert replace_shop_action("teacher_sale", SubPhase.SHOP, action) == sale
    assert replace_shop_action("teacher_other_shop", SubPhase.SHOP, action) != sale
    for phase in SubPhase:
        if phase != SubPhase.SHOP:
            assert not replace_shop_action("teacher_sale", phase, None)
            assert not replace_shop_action("teacher_other_shop", phase, None)


def test_shop_partition_rejects_missing_teacher_or_unknown_arm():
    with pytest.raises(ValueError, match="proposal"):
        replace_shop_action("teacher_sale", SubPhase.SHOP, None)
    with pytest.raises(ValueError, match="Unknown"):
        replace_shop_action("sale", SubPhase.SHOP, 0)
