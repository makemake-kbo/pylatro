local card_1 = {wire_id = "hand:1", STATIONARY = true}
local card_2 = {wire_id = "hand:2", STATIONARY = true}
local play_button = {}
local discard_button = {}

G = {
    GAME = {STOP_USE = 0},
    CONTROLLER = {locked = false, locks = {}},
    E_MANAGER = {queues = {base = {}}},
    STATE_COMPLETE = false,
    blind_select = {},
    hand = {cards = {card_1}},
    buttons = {
        get_UIE_by_ID = function(_, id)
            if id == "play_button" then return play_button end
            if id == "discard_button" then return discard_button end
            return nil
        end,
    },
}

SMODS = {cards_to_draw = 0, draw_queued = nil}
PYLATRO_BRIDGE = {
    serializer = {
        card_id = function(card) return card.wire_id end,
    },
    executor = {
        callback_available = function() return true end,
    },
}

local Readiness = assert(loadfile("mods/pylatro_bridge/readiness.lua"))()
local state = {}

-- Unrelated UI events may live in the base queue indefinitely and must not
-- prevent the bridge's first blind-selection request.
G.E_MANAGER.queues.base = {{}}
G.STATE_COMPLETE = true
assert(Readiness.ready("blind_select", 0, state))

G.STATE_COMPLETE = false
assert(not Readiness.ready("hand_play", 0, state))
G.STATE_COMPLETE = true

SMODS.cards_to_draw = 2
assert(not Readiness.ready("hand_play", 0, state))
SMODS.cards_to_draw = 0

card_1.STATIONARY = false
assert(not Readiness.ready("hand_play", 0.40, state))
card_1.STATIONARY = true

assert(not Readiness.ready("hand_play", 0.40, state))
assert(not Readiness.ready("hand_play", 0.59, state))
assert(Readiness.ready("hand_play", 0.60, state))

-- Any newly drawn card resets the stable-hand timer.
table.insert(G.hand.cards, card_2)
assert(not Readiness.ready("hand_play", 0.61, state))
assert(Readiness.ready("hand_play", 1.21, state))

G.CONTROLLER.locks.use = true
assert(not Readiness.ready("hand_play", 1.22, state))
-- A stale transition lock must not suppress an otherwise stable hand forever.
assert(Readiness.ready("hand_play", 1.46, state))
G.CONTROLLER.locks.use = nil

SMODS.cards_to_draw = 1
assert(Readiness.ready("hand_play", 1.47, state))
SMODS.cards_to_draw = 0

-- A shop is not actionable until vanilla's delayed opening event has filled
-- its inventory. This prevents leaving while that event still references it.
local next_round_button = {}
G.shop = {
    get_UIE_by_ID = function(_, id)
        if id == "next_round_button" then return next_round_button end
    end,
}
G.shop_jokers = {cards = {}}
G.shop_vouchers = {cards = {}}
G.shop_booster = {cards = {}}
assert(not Readiness.ready("shop", 1.48, state))
table.insert(G.shop_jokers.cards, {wire_id = "shop:1"})
assert(Readiness.ready("shop", 1.49, state))

-- Booster contents must be complete and stable before inference.
G.pack_cards = {cards = {{wire_id = "pack:1"}}}
assert(not Readiness.ready("booster_pack", 1.50, state))
assert(not Readiness.ready("booster_pack", 1.74, state))
assert(Readiness.ready("booster_pack", 1.75, state))

-- Leaving hand play clears the settle state for the next draw.
assert(Readiness.ready(nil, 1.76, state))
assert(state.hand_signature == nil)
assert(state.hand_stable_since == nil)

print("bridge readiness mock tests passed")
