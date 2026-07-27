local card_1 = {wire_id = "hand:1"}
local card_2 = {wire_id = "hand:2"}
local play_button = {}
local discard_button = {}

G = {
    GAME = {STOP_USE = 0},
    CONTROLLER = {locked = false, locks = {}},
    E_MANAGER = {queues = {base = {}}},
    STATE_COMPLETE = false,
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
}

local Readiness = assert(loadfile("mods/pylatro_bridge/readiness.lua"))()
local state = {}

assert(not Readiness.ready("hand_play", 0, state))
G.STATE_COMPLETE = true

-- The real EventManager field is queues.base. Pending draw animation events
-- must block a decision even after Balatro enters SELECTING_HAND.
G.E_MANAGER.queues.base = {{}}
assert(not Readiness.ready("hand_play", 0, state))
G.E_MANAGER.queues.base = {}

SMODS.cards_to_draw = 2
assert(not Readiness.ready("hand_play", 0, state))
SMODS.cards_to_draw = 0

assert(not Readiness.ready("hand_play", 1.00, state))
assert(not Readiness.ready("hand_play", 1.24, state))
assert(Readiness.ready("hand_play", 1.25, state))

-- Any newly drawn card resets the stable-hand timer.
table.insert(G.hand.cards, card_2)
assert(not Readiness.ready("hand_play", 1.26, state))
assert(Readiness.ready("hand_play", 1.51, state))

G.CONTROLLER.locks.use = true
assert(not Readiness.ready("hand_play", 2.00, state))
G.CONTROLLER.locks.use = nil

-- Leaving hand play clears the settle state for the next draw.
assert(Readiness.ready(nil, 2.00, state))
assert(state.hand_signature == nil)
assert(state.hand_stable_since == nil)

print("bridge readiness mock tests passed")
