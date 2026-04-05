from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.constants import ActionRange, SubPhase
from pylatro_cli.controller import GamePhase

data = load_game_data()
agent = HeuristicAgent()


def action_name(aid):
    AR = ActionRange
    if aid == AR.BLIND_PLAY:
        return "BLIND_PLAY"
    if aid == AR.PLAY_HAND:
        return "PLAY_HAND"
    if aid == AR.DISCARD:
        return "DISCARD"
    if aid == AR.USE_CONSUMABLE:
        return "USE_CONSUMABLE"
    if aid == AR.SELECT_CONFIRM:
        return "SELECT_CONFIRM"
    if aid == AR.SHOP_LEAVE:
        return "SHOP_LEAVE"
    if aid == AR.SHOP_REROLL:
        return "SHOP_REROLL"
    if aid == AR.CONSUMABLE_CONFIRM:
        return "CONS_CONFIRM"
    if aid == AR.CONSUMABLE_CANCEL:
        return "CONS_CANCEL"
    if aid == AR.PACK_SKIP:
        return "PACK_SKIP"
    if AR.TOGGLE_CARD_START <= aid <= AR.TOGGLE_CARD_END:
        return "TOGGLE_" + str(aid - AR.TOGGLE_CARD_START)
    if AR.SHOP_BUY_START <= aid <= AR.SHOP_BUY_END:
        return "BUY_" + str(aid - AR.SHOP_BUY_START)
    if AR.CONSUMABLE_SLOT_START <= aid <= AR.CONSUMABLE_SLOT_END:
        return "CONS_SLOT_" + str(aid - AR.CONSUMABLE_SLOT_START)
    return str(aid)


for seed in range(3):
    print("=" * 60)
    print("Seed", seed)
    runner = FastRunner(seed, data)
    step = 0
    ante = 1
    while not runner.done:
        s = runner.state
        sp = runner.sub_phase
        mask = runner.compute_mask()
        action = agent.select_action(
            s,
            sp,
            mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
            pending_consumable_slot=runner.pending_consumable_slot,
        )
        aname = action_name(action)
        ante_now = s.round_resets.ante

        if sp == SubPhase.SHOP:
            jokers = [s.data.centers[j.center_key]["name"] for j in s.jokers]
            items = []
            for it in list(s.shop.cards) + list(s.shop.vouchers) + list(s.shop.boosters):
                c = s.data.centers.get(it.center_key, {})
                items.append(c.get("name", "?") + " $" + str(it.cost))
            if action >= ActionRange.SHOP_BUY_START and action <= ActionRange.SHOP_BUY_END:
                idx = action - ActionRange.SHOP_BUY_START
                all_items = list(s.shop.cards) + list(s.shop.vouchers) + list(s.shop.boosters)
                bought = (
                    s.data.centers.get(all_items[idx].center_key, {}).get("name", "?") if idx < len(all_items) else "?"
                )
                print(f"  SHOP {aname}: buying {bought} | jokers={jokers} $={s.dollars} shop={items}")
            elif action == ActionRange.SHOP_REROLL:
                print(f"  SHOP REROLL | jokers={jokers} $={s.dollars}")
            elif action == ActionRange.SHOP_LEAVE:
                print(f"  SHOP LEAVE | jokers={jokers} $={s.dollars}")
            else:
                print(f"  SHOP {aname} | jokers={jokers} $={s.dollars} shop={items}")

        elif sp == SubPhase.BLIND_SELECT:
            blind = s.blind_on_deck or "?"
            print(f"  Ante {ante_now} BLIND: {blind} -> {aname}")

        elif sp == SubPhase.CHOOSE_ACTION:
            pass

        elif sp == SubPhase.SELECT_CARDS:
            if action == ActionRange.SELECT_CONFIRM and runner.pending_action == "play":
                hand = s.hand_cards
                sel_ranks = [hand[i].rank for i in sorted(runner.selected_cards) if i < len(hand)]
                hl = s.current_round.hands_left
                print(f"  PLAY {sel_ranks} (hands_left={hl})")

        runner.step(action)
        step += 1

        if runner.blind_just_beaten:
            print(f"  BLIND BEATEN! score={runner.round_score}")
        if s.round_resets.ante > ante_now:
            print(f"  *** Reached ante {s.round_resets.ante}")

    print(f"  DONE: max_ante={runner.max_ante} won={runner.won}")
