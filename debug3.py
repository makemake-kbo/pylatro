from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_agent.constants import ActionRange, SubPhase

data = load_game_data()
agent = HeuristicAgent()
AR = ActionRange


def aname(aid):
    if aid == AR.BLIND_PLAY:
        return "BLIND_PLAY"
    if aid == AR.PLAY_HAND:
        return "PLAY_HAND"
    if aid == AR.DISCARD:
        return "DISCARD"
    if aid == AR.USE_CONSUMABLE:
        return "USE_CONSUMABLE"
    if aid == AR.SELECT_CONFIRM:
        return "CONFIRM"
    if aid == AR.SHOP_LEAVE:
        return "LEAVE"
    if aid == AR.SHOP_REROLL:
        return "REROLL"
    if aid == AR.CONSUMABLE_CONFIRM:
        return "CONS_CONF"
    if aid == AR.CONSUMABLE_CANCEL:
        return "CONS_CANCEL"
    if aid == AR.PACK_SKIP:
        return "SKIP"
    if AR.TOGGLE_CARD_START <= aid <= AR.TOGGLE_CARD_END:
        return "TOG" + str(aid - AR.TOGGLE_CARD_START)
    if AR.SHOP_BUY_START <= aid <= AR.SHOP_BUY_END:
        return "BUY" + str(aid - AR.SHOP_BUY_START)
    if AR.CONSUMABLE_SLOT_START <= aid <= AR.CONSUMABLE_SLOT_END:
        return "SLOT" + str(aid - AR.CONSUMABLE_SLOT_START)
    if AR.PACK_CLAIM_START <= aid <= AR.PACK_CLAIM_END:
        return "CLAIM" + str(aid - AR.PACK_CLAIM_START)
    return "?" + str(aid)


for seed in range(3):
    print(f"\n=== Seed {seed} ===")
    runner = FastRunner(seed, data)
    steps = 0
    ante = 1

    while not runner.done and steps < 200:
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

        if sp == SubPhase.CHOOSE_ACTION:
            hand = [c.rank + c.suit[0] for c in s.hand_cards]
            print(
                f"  [{s.round_resets.ante}:{s.blind_on_deck}] CHOOSE hand={hand} h={s.current_round.hands_left} d={s.current_round.discards_left} -> {aname(action)}"
            )
        elif sp == SubPhase.SELECT_CARDS:
            sel = sorted(runner.selected_cards)
            sel_r = [s.hand_cards[i].rank for i in sel if i < len(s.hand_cards)]
            print(f"    SELECT {sel}({sel_r}) {runner.pending_action} -> {aname(action)}")
        elif sp == SubPhase.SHOP:
            jk = [s.data.centers[j.center_key]["name"] for j in s.jokers]
            items = []
            for item in list(s.shop.cards) + list(s.shop.vouchers) + list(s.shop.boosters):
                c = s.data.centers.get(item.center_key, {})
                items.append(f"{c.get('name', '?')}(${item.cost})")
            print(f"  SHOP jokers={jk} ${s.dollars} items={items} -> {aname(action)}")
        elif sp == SubPhase.BOOSTER_PACK:
            pack = s.pack
            pn = [s.data.centers[c.center_key]["name"] for c in pack.cards] if pack and pack.cards else []
            print(f"  PACK {pn} -> {aname(action)}")
        elif sp == SubPhase.CONSUMABLE_TARGET:
            cn = [s.data.centers[c.center_key]["name"] for c in s.consumables]
            print(f"  CONS slot={runner.pending_consumable_slot} cons={cn} -> {aname(action)}")

        runner.step(action)
        steps += 1

        if runner.blind_just_beaten:
            print(f"  >>> BEATEN!")

    print(f"  RESULT: max_ante={runner.max_ante} won={runner.won} steps={steps}")
    jk = [s.data.centers[j.center_key]["name"] for j in runner.state.jokers]
    cn = [s.data.centers[c.center_key]["name"] for c in runner.state.consumables]
    played = {k: v for k, v in runner.state.hands.items() if v.get("played", 0) > 0}
    print(f"  jokers={jk} cons={cn} ${s.dollars} played={played}")
