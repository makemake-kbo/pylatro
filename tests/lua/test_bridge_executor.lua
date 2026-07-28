local function node(config, children)
    return {
        config = config or {},
        children = children or {},
        states = {visible = true},
    }
end

local hand_card = {wire_id = "hand:1", children = {}}
local shop_card = {wire_id = "shop:1", children = {}}
local pack_card = {wire_id = "pack:1", children = {}}
local owned_joker = {wire_id = "joker:1", children = {}}
local owned_consumable = {wire_id = "consumable:1", children = {}}

local function owned_area(cards)
    return {
        cards = cards,
        highlighted = {},
        add_to_highlighted = function(self, card)
            table.insert(self.highlighted, card)
            card.children.use_button = {
                UIRoot = node({
                    button = "sell_card",
                    func = "can_sell_card",
                    ref_table = card,
                }),
            }
        end,
        unhighlight_all = function(self)
            self.highlighted = {}
        end,
    }
end

local play_button = node({id = "play_button", func = "can_play"})
local discard_button = node({id = "discard_button", func = "can_discard"})

G = {
    I = {UIBOX = {}},
    FUNCS = {},
    buttons = {
        get_UIE_by_ID = function(_, id)
            if id == "play_button" then return play_button end
            if id == "discard_button" then return discard_button end
            return nil
        end,
    },
    hand = {
        cards = {hand_card},
        highlighted = {},
        add_to_highlighted = function(self, card)
            table.insert(self.highlighted, card)
        end,
        unhighlight_all = function(self)
            self.highlighted = {}
        end,
    },
    jokers = owned_area({owned_joker}),
    consumeables = owned_area({owned_consumable}),
    shop_jokers = {cards = {shop_card}},
    pack_cards = {
        cards = {pack_card},
        highlighted = {},
        add_to_highlighted = function(self, card)
            table.insert(self.highlighted, card)
            card.children.use_button = {
                UIRoot = node({
                    button = "use_card",
                    func = "can_select_card",
                    ref_table = card,
                }),
            }
        end,
        unhighlight_all = function(self)
            self.highlighted = {}
        end,
    },
}

PYLATRO_BRIDGE = {
    serializer = {
        card_id = function(card) return card.wire_id end,
    },
}

local calls = {
    play = 0,
    discard = 0,
    cash_out = 0,
    buy = 0,
    pack = 0,
    sell = 0,
}
G.FUNCS.can_play = function(button)
    button.config.button = #G.hand.highlighted > 0
        and "play_cards_from_highlighted"
        or nil
end
G.FUNCS.can_discard = function(button)
    button.config.button = #G.hand.highlighted > 0
        and "discard_cards_from_highlighted"
        or nil
end
G.FUNCS.play_cards_from_highlighted = function(button)
    assert(button == play_button)
    calls.play = calls.play + 1
end
G.FUNCS.discard_cards_from_highlighted = function(button)
    assert(button == discard_button)
    calls.discard = calls.discard + 1
end

local Executor = assert(loadfile("mods/pylatro_bridge/executor.lua"))()

local ok, err = Executor.execute({type = "play", card_ids = {"hand:1"}})
assert(ok, err)
assert(calls.play == 1)

ok, err = Executor.execute({type = "discard", card_ids = {"hand:1"}})
assert(ok, err)
assert(calls.discard == 1)

local cash_out_button = node({button = "cash_out"})
G.round_eval = {UIRoot = node({}, {cash_out_button})}
G.FUNCS.cash_out = function(button)
    assert(button == cash_out_button)
    calls.cash_out = calls.cash_out + 1
end
ok, err = Executor.cash_out()
assert(ok, err)
assert(calls.cash_out == 1)

local buy_button = node({button = "buy_from_shop", ref_table = shop_card})
shop_card.children.buy_button = {UIRoot = buy_button}
G.FUNCS.buy_from_shop = function(button)
    assert(button == buy_button)
    calls.buy = calls.buy + 1
end
ok, err = Executor.execute({type = "shop_buy", item_id = "shop:1"})
assert(ok, err)
assert(calls.buy == 1)

local leave_button = node({id = "next_round_button", button = "toggle_shop"})
G.shop = {
    get_UIE_by_ID = function(_, id)
        if id == "next_round_button" then return leave_button end
    end,
}
G.FUNCS.toggle_shop = function(button)
    assert(button == leave_button)
end
ok, err = Executor.execute({type = "shop_leave"})
assert(ok, err)

G.FUNCS.can_select_card = function(button)
    button.config.button = "use_card"
end
G.FUNCS.use_card = function(button)
    assert(button.config.ref_table == pack_card)
    calls.pack = calls.pack + 1
end
ok, err = Executor.execute({type = "pack_claim", item_id = "pack:1"})
assert(ok, err)
assert(calls.pack == 1)

G.FUNCS.can_sell_card = function(button)
    button.config.button = "sell_card"
end
G.FUNCS.sell_card = function(button)
    assert(
        button.config.ref_table == owned_joker
        or button.config.ref_table == owned_consumable
    )
    calls.sell = calls.sell + 1
end
ok, err = Executor.execute({type = "shop_sell", item_id = "joker:1"})
assert(ok, err)
ok, err = Executor.execute({type = "shop_sell", item_id = "consumable:1"})
assert(ok, err)
assert(calls.sell == 2)

print("bridge executor mock tests passed")
