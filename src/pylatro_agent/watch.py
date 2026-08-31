"""Interactive step-by-step viewer for a trained policy checkpoint.

``uv run pylatro --model checkpoint.pt`` loads a :class:`BalatroAgent`
checkpoint and plays one seeded run, pausing before each decision so you can
watch what the policy does: the live game state, the action it picks, the
greedy alternatives it weighed, and the critic's value estimate.

The whole feature lives in this one module. It depends only on the agent /
env stack (torch, BalatroEnv), never on the Textual CLI, so it can be lifted
out or changed without touching the terminal UI. The CLI entry point does
nothing but forward ``--model`` here.
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .action import DecodedAction
    from .env import BalatroEnv

# Balatro stake order (1-indexed). White is the base stake.
STAKE_NAMES: dict[int, str] = {
    1: "White",
    2: "Red",
    3: "Green",
    4: "Black",
    5: "Blue",
    6: "Purple",
    7: "Orange",
    8: "Gold",
}
STAKE_KEYS: dict[str, int] = {name.lower(): num for num, name in STAKE_NAMES.items()}

SUIT_SYMBOL: dict[str, str] = {
    "Spades": "♠",
    "Hearts": "♥",
    "Diamonds": "♦",
    "Clubs": "♣",
}

# Short tags for card enhancements / editions / seals so the hand stays compact.
_ENHANCEMENT_SHORT: dict[str, str] = {
    "m_bonus": "bonus",
    "m_mult": "mult",
    "m_wild": "wild",
    "m_glass": "glass",
    "m_steel": "steel",
    "m_stone": "stone",
    "m_gold": "gold",
    "m_lucky": "lucky",
}


# ── Deck / stake resolution ──────────────────────────────────────────────


def _deck_aliases(data) -> dict[str, str]:
    """Map friendly deck names (and raw keys) to center keys.

    ``red`` -> ``b_red``, ``Red Deck`` -> ``b_red``, ``b_red`` -> ``b_red``.
    """
    aliases: dict[str, str] = {}
    for key, center in data.centers.items():
        if not isinstance(center, dict) or center.get("set") != "Back":
            continue
        aliases[key.lower()] = key
        name = center.get("name")
        if name:
            aliases[name.lower()] = key
            aliases[name.lower().removesuffix(" deck").strip()] = key
    return aliases


def _resolve_deck(data, raw: str) -> str:
    aliases = _deck_aliases(data)
    key = aliases.get(raw.strip().lower())
    if key is None:
        options = sorted({k for k in aliases if not k.startswith("b_")})
        raise SystemExit(f"Unknown deck {raw!r}. Options: {', '.join(options)}")
    return key


def _resolve_stake(raw: str) -> int:
    raw = raw.strip().lower()
    if raw.isdigit():
        num = int(raw)
        if num not in STAKE_NAMES:
            raise SystemExit(f"Stake must be 1-8, got {num}.")
        return num
    num = STAKE_KEYS.get(raw)
    if num is None:
        raise SystemExit(
            f"Unknown stake {raw!r}. Options: {', '.join(n.lower() for n in STAKE_NAMES.values())} (or 1-8)."
        )
    return num


# ── Model loading ────────────────────────────────────────────────────────


def _infer_agent_config(payload: dict, device):
    """Reconstruct an AgentConfig from a checkpoint.

    Prefers the stored ``agent_config`` snapshot. For v8 checkpoints without
    one, it infers the transformer dimensions from tensor shapes.
    """
    from .agent import AgentConfig

    valid = {f.name for f in __import__("dataclasses").fields(AgentConfig)}
    stored = payload.get("agent_config")
    if stored:
        return AgentConfig(**{k: v for k, v in stored.items() if k in valid})

    sd = payload["state_dict"]
    layer_idxs = [int(k.split(".")[2]) for k in sd if k.startswith("backbone.layers.") and k.split(".")[2].isdigit()]
    n_layers = (max(layer_idxs) + 1) if layer_idxs else AgentConfig.n_layers
    d_model = int(sd["value_head.ante_survival.weight"].shape[1])
    d_ff = int(sd["backbone.layers.0.ffn.0.weight"].shape[0])
    return AgentConfig(d_model=d_model, n_layers=n_layers, d_ff=d_ff)


def _load_model(model_path: str, vocab, device):
    from .agent import BalatroAgent
    from .checkpoint import load_checkpoint_payload

    payload = load_checkpoint_payload(model_path, device)
    config = _infer_agent_config(payload, device)
    model = BalatroAgent(config, vocab).to(device)
    state_dict = payload["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, config


def _pick_device(requested: str | None):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cpu")


# ── State / action rendering ─────────────────────────────────────────────


def _center_name(data, key: str) -> str:
    center = data.centers.get(key, {})
    return center.get("name", key) if isinstance(center, dict) else key


def _card_str(card) -> str:
    rank = "10" if card.rank == "T" else card.rank
    suit = SUIT_SYMBOL.get(card.suit, card.suit[:1])
    text = f"{rank}{suit}"
    tags: list[str] = []
    if card.center_key and card.center_key != "c_base":
        tags.append(_ENHANCEMENT_SHORT.get(card.center_key, card.center_key))
    if card.seal:
        tags.append(f"{card.seal.lower()}-seal")
    if card.edition_key:
        tags.append(card.edition_key.removeprefix("e_"))
    if card.debuff:
        tags.append("debuffed")
    if tags:
        text += f"({'/'.join(tags)})"
    return text


def _hand_str(cards) -> str:
    return "  ".join(f"[{i}]{_card_str(c)}" for i, c in enumerate(cards))


def _shop_item_str(data, item) -> str:
    name = _center_name(data, item.center_key)
    return f"{name} (${item.cost})"


def _render_state(env: BalatroEnv, info: dict) -> str:
    from .constants import SubPhase

    state = env.state
    data = state.data
    lines: list[str] = []

    blind = state.round_resets.blind
    blind_name = blind.get("name", "?") if blind else "-"
    target = info.get("blind_target", 0)
    lines.append(
        f"Ante {info.get('ante', '?')}  |  Blind: {blind_name}"
        f"  |  Score {info.get('round_score', 0)}/{target}"
        f"  |  Hands {info.get('hands_left', 0)}  Discards {info.get('discards_left', 0)}"
        f"  |  ${info.get('dollars', 0)}"
    )

    if state.jokers:
        jokers = ", ".join(_center_name(data, j.center_key) for j in state.jokers)
        lines.append(f"Jokers: {jokers}")
    if state.consumables:
        cons = ", ".join(f"[{i}]{_center_name(data, c.center_key)}" for i, c in enumerate(state.consumables))
        lines.append(f"Consumables: {cons}")

    sub_phase = env._sub_phase
    if sub_phase == SubPhase.CHOOSE_ACTION:
        lines.append(f"Hand: {_hand_str(state.hand_cards)}")
    elif sub_phase == SubPhase.SHOP:
        shop = state.shop
        items = (
            [f"buy: {_shop_item_str(data, c)}" for c in shop.cards]
            + [f"voucher: {_shop_item_str(data, v)}" for v in shop.vouchers]
            + [f"pack: {_shop_item_str(data, b)}" for b in shop.boosters]
        )
        lines.append("Shop: " + (" | ".join(items) if items else "(empty)"))
        lines.append(f"Reroll cost: ${state.current_round.reroll_cost}")
    elif sub_phase == SubPhase.BOOSTER_PACK and state.pack is not None:
        pack_cards = ", ".join(_center_name(data, c.center_key) for c in state.pack.cards)
        lines.append(f"Pack ({state.pack.state_name}): {pack_cards}")
        lines.append(f"Choices remaining: {state.pack.choices_remaining}")
    elif sub_phase == SubPhase.BLIND_SELECT:
        lines.append(f"Blind on deck: {state.blind_on_deck or '?'}")

    return "\n".join(lines)


def _describe_action(env: BalatroEnv, action_id: int) -> str:
    """One-line human description of what ``action_id`` does in this state."""
    from .action import ActionType, decode_action
    from .subset_actions import consumable_subset_indices, subset_indices

    state = env.state
    data = state.data
    decoded: DecodedAction = decode_action(action_id)
    at = decoded.action_type

    def cards_for(indices) -> str:
        return " ".join(_card_str(state.hand_cards[i]) for i in indices if i < len(state.hand_cards))

    if at == ActionType.BLIND_PLAY:
        return f"Play blind ({state.blind_on_deck or 'Small'})"
    if at == ActionType.BLIND_SKIP:
        return "Skip blind"
    if at == ActionType.BLIND_REROLL:
        return "Reroll boss blind"
    if at == ActionType.PLAY_SUBSET:
        indices = subset_indices(decoded.index)
        hand_info = env._controller.hand_evaluation(list(indices))
        name = hand_info[1] if hand_info else "?"
        return f"Play {cards_for(indices)}  → {name}"
    if at == ActionType.DISCARD_SUBSET:
        indices = subset_indices(decoded.index)
        return f"Discard {cards_for(indices)}"
    if at == ActionType.USE_CONSUMABLE_NO_TARGET:
        return f"Use {_slot_consumable(state, data, decoded.index)}"
    if at == ActionType.USE_CONSUMABLE_HAND_SUBSET:
        targets = consumable_subset_indices(decoded.detail)
        return f"Use {_slot_consumable(state, data, decoded.index)} on {cards_for(targets)}"
    if at == ActionType.USE_CONSUMABLE_JOKER:
        joker = (
            _center_name(data, state.jokers[decoded.detail].center_key) if decoded.detail < len(state.jokers) else "?"
        )
        return f"Use {_slot_consumable(state, data, decoded.index)} on {joker}"
    if at == ActionType.SHOP_BUY:
        return f"Buy {_shop_buy_label(state, data, decoded.index)}"
    if at == ActionType.SHOP_REROLL:
        return f"Reroll shop (${state.current_round.reroll_cost})"
    if at == ActionType.SHOP_SELL_JOKER:
        joker = _center_name(data, state.jokers[decoded.index].center_key) if decoded.index < len(state.jokers) else "?"
        return f"Sell joker {joker}"
    if at == ActionType.SHOP_SELL_CONSUMABLE:
        return f"Sell {_slot_consumable(state, data, decoded.index)}"
    if at == ActionType.SHOP_LEAVE:
        return "Leave shop"
    if at == ActionType.PACK_CLAIM:
        card = (
            _center_name(data, state.pack.cards[decoded.index].center_key)
            if state.pack and decoded.index < len(state.pack.cards)
            else "?"
        )
        return f"Claim {card} from pack"
    if at == ActionType.PACK_SKIP:
        return "Skip pack"
    return str(at)


def _slot_consumable(state, data, slot: int) -> str:
    if slot < len(state.consumables):
        return _center_name(data, state.consumables[slot].center_key)
    return f"consumable[{slot}]"


def _shop_buy_label(state, data, idx: int) -> str:
    n_cards = len(state.shop.cards)
    n_vouchers = len(state.shop.vouchers)
    if idx < n_cards:
        return _center_name(data, state.shop.cards[idx].center_key)
    if idx < n_cards + n_vouchers:
        return _center_name(data, state.shop.vouchers[idx - n_cards].center_key)
    booster_idx = idx - n_cards - n_vouchers
    if booster_idx < len(state.shop.boosters):
        return _center_name(data, state.shop.boosters[booster_idx].center_key)
    return f"shop[{idx}]"


# ── Policy inspection ────────────────────────────────────────────────────


def _macro_type_probs(dist) -> list[tuple[str, float]]:
    """Probability the policy assigns to each high-level action type."""
    from .action_grammar import ACTION_TYPE_TO_GRAMMAR_INDEX

    probs = dist.action_type_probs[0].tolist()
    index_to_type = {idx: at for at, idx in ACTION_TYPE_TO_GRAMMAR_INDEX.items()}
    rows = [(index_to_type[idx].value, p) for idx, p in enumerate(probs) if idx in index_to_type and p > 1e-4]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _topk_actions(model, batch, valid_ids, k: int, temperature: float):
    """Return [(action_id, prob), ...] for the k most likely legal actions.

    Evaluates the grammar log-prob of every legal action by replaying the
    single observation as a batch (one row per candidate action). This is the
    same distribution the policy acts under, so the ranking is faithful.
    """
    import torch

    from .training.ppo import _grammar_distribution

    n = len(valid_ids)
    big = {key: value.repeat(n, *([1] * (value.dim() - 1))) for key, value in batch.items()}
    dist, _ = _grammar_distribution(model, big, temperature=temperature)
    ids = torch.tensor(valid_ids, dtype=torch.long, device=batch["tokens"].device)
    log_probs = dist.log_prob(ids)
    probs = log_probs.exp().tolist()
    ranked = sorted(zip(valid_ids, probs, strict=True), key=lambda r: r[1], reverse=True)
    return ranked[:k]


# ── Main loop ────────────────────────────────────────────────────────────


def _play(args) -> int:
    import numpy as np
    import torch

    from pylatro import load_game_data

    from .env import BalatroEnv
    from .training.ppo import _grammar_distribution, _single_obs_to_batch
    from .vocab import build_vocab

    data = load_game_data()
    vocab = build_vocab(data)
    device = _pick_device(args.device)

    deck_key = _resolve_deck(data, args.deck)
    stake = _resolve_stake(args.stake)

    print(f"Loading model: {args.model}")
    model, config = _load_model(args.model, vocab, device)
    print(
        f"  d_model={config.d_model} n_layers={config.n_layers} "
        f"critic=conditional_hazard_residual params={model.count_parameters():,}"
    )
    print(
        f"Deck: {_center_name(data, deck_key)} ({deck_key})  |  "
        f"Stake: {STAKE_NAMES[stake]} ({stake})  |  Seed: {args.seed if args.seed is not None else 'random'}  |  "
        f"Selection: {'sample' if args.sample else 'greedy'} (T={args.temperature})"
    )
    print("\nControls: [Enter] step  |  a = auto-run  |  q = quit\n")

    env = BalatroEnv(
        seed=args.seed,
        data=data,
        vocab=vocab,
        stake=stake,
        deck_key=deck_key,
        win_ante=args.win_ante,
        max_steps=args.max_steps,
        enable_teacher=False,
    )
    obs, info = env.reset()

    auto = args.auto
    step = 0
    done = False
    result_info: dict = info

    with torch.inference_mode():
        while not done:
            step += 1
            batch = _single_obs_to_batch(obs, device)
            dist, value_dict = _grammar_distribution(model, batch, temperature=args.temperature)
            chosen = dist.sample().item() if args.sample else dist.mode().item()

            valid_ids = np.flatnonzero(obs["action_mask"]).tolist()
            print("=" * 78)
            print(f"Step {step}")
            print(_render_state(env, env._capture_state_info()))
            print("-" * 78)

            win_prob = float(value_dict["win_prob"][0])
            expected = float(value_dict["expected_score"][0])
            print(f"Critic: win_prob={win_prob:.1%}  expected_return={expected:+.2f}")

            macro = _macro_type_probs(dist)
            if len(macro) > 1:
                summary = "  ".join(f"{name} {p:.0%}" for name, p in macro[:5])
                print(f"Action type: {summary}")

            if len(valid_ids) > 1:
                topk = _topk_actions(model, batch, valid_ids, args.topk, args.temperature)
                print(f"Top {len(topk)} of {len(valid_ids)} legal actions:")
                for action_id, prob in topk:
                    marker = ">" if action_id == chosen else " "
                    print(f"  {marker} {prob:6.1%}  {_describe_action(env, action_id)}")
            else:
                print(f"Only legal action: {_describe_action(env, chosen)}")

            print(f"\n  → MODEL PLAYS: {_describe_action(env, chosen)}")

            del batch, dist, value_dict

            if not auto:
                cmd = input("").strip().lower()
                if cmd == "q":
                    print("Quit.")
                    return 0
                if cmd == "a":
                    auto = True
            else:
                time.sleep(args.delay)

            obs, _reward, terminated, truncated, result_info = env.step(chosen)
            done = terminated or truncated

    print("=" * 78)
    won = bool(result_info.get("won", False))
    ante = int(result_info.get("ante", 1))
    print(f"GAME OVER  |  {'WON' if won else 'LOST'}  |  reached ante {ante}  |  {step} steps")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pylatro --model",
        description="Watch a trained pylatro policy play a run, one decision at a time.",
    )
    parser.add_argument("--model", "-m", required=True, help="Path to a checkpoint (.pt).")
    parser.add_argument("--seed", type=int, default=None, help="Run seed (default: random).")
    parser.add_argument("--deck", default="red", help="Deck name or key (default: red).")
    parser.add_argument("--stake", default="white", help="Stake name or 1-8 (default: white).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature for the policy.")
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of greedy argmax.")
    parser.add_argument("--topk", type=int, default=5, help="How many alternative actions to show.")
    parser.add_argument("--auto", action="store_true", help="Auto-advance without waiting for input.")
    parser.add_argument("--delay", type=float, default=0.8, help="Seconds between auto steps.")
    parser.add_argument("--win-ante", type=int, default=None, help="Override victory ante.")
    parser.add_argument("--max-steps", type=int, default=2000, help="No-progress step cap before truncation.")
    parser.add_argument("--device", default=None, help="torch device (default: cpu).")
    args = parser.parse_args(argv)

    try:
        return _play(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
