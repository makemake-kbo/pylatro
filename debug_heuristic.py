from pylatro import load_game_data
from pylatro_agent.heuristic import HeuristicAgent
from pylatro_agent.training.fast_runner import FastRunner
from pylatro_cli.controller import GamePhase

data = load_game_data()
agent = HeuristicAgent()

for seed in range(5):
    runner = FastRunner(seed, data)
    steps = 0
    ante = runner.state.round_resets.ante
    phase = runner.phase
    print(f"\n=== Seed {seed} ===")

    while not runner.done and steps < 100:
        mask = runner.compute_mask()
        s = runner.state
        sp = runner.sub_phase
        hl = s.current_round.hands_left
        dl = s.current_round.discards_left
        action_type = "?"

        if sp.name == "BLIND_SELECT":
            blind = s.blind_on_deck or "?"
            action_type = f"blind={blind} ante={s.round_resets.ante}"
        elif sp.name == "CHOOSE_ACTION":
            hand = s.hand_cards
            ranks = [c.rank for c in hand]
            action_type = f"hand={ranks} hands={hl} discards={dl}"
        elif sp.name == "SHOP":
            jokers = [s.data.centers[j.center_key]["name"] for j in s.jokers]
            shop_items = []
            for item in list(s.shop.cards) + list(s.shop.vouchers) + list(s.shop.boosters):
                c = s.data.centers.get(item.center_key, {})
                shop_items.append(f"{c.get('name', '?')}(${item.cost})")
            action_type = f"jokers={jokers} shop={shop_items} $={s.dollars}"
        elif sp.name == "SELECT_CARDS":
            sel = runner.selected_cards
            hand = s.hand_cards
            sel_ranks = [hand[i].rank for i in sel if i < len(hand)]
            action_type = f"selected={sel_ranks} pending={runner.pending_action}"
        elif sp.name == "CONSUMABLE_TARGET":
            action_type = f"slot={runner.pending_consumable_slot}"
        elif sp.name == "BOOSTER_PACK":
            pack = s.pack
            if pack:
                pack_items = [s.data.centers[c.center_key]["name"] for c in pack.cards] if pack.cards else []
                action_type = f"pack={pack_items}"
            else:
                action_type = "no pack"

        action = agent.select_action(
            s,
            sp,
            mask,
            selected_cards=runner.selected_cards,
            pending_action=runner.pending_action,
            pending_consumable_slot=runner.pending_consumable_slot,
        )

        old_phase = runner.phase
        runner.step(action)
        steps += 1

        if runner.phase != old_phase:
            print(f"  Phase: {old_phase.name} -> {runner.phase.name}")
        if s.round_resets.ante != ante:
            ante = s.round_resets.ante
            print(f"  *** Reached ante {ante}")

    print(f"  Done: max_ante={runner.max_ante}, won={runner.won}, phase={runner.phase.name}")
