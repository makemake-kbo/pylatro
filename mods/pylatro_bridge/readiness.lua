local Readiness = {}

local HAND_SETTLE_SECONDS = 0.25

local function base_queue_empty()
    local queues = G.E_MANAGER and G.E_MANAGER.queues
    return not queues or not queues.base or #queues.base == 0
end

local function controller_locks_clear()
    if G.CONTROLLER.locked then return false end
    for _, locked in pairs(G.CONTROLLER.locks or {}) do
        if locked then return false end
    end
    return true
end

local function hand_signature()
    local parts = {}
    for _, card in ipairs(G.hand and G.hand.cards or {}) do
        table.insert(parts, PYLATRO_BRIDGE.serializer.card_id(card))
    end
    return table.concat(parts, "|")
end

local function hand_buttons_live()
    if not G.buttons or type(G.buttons.get_UIE_by_ID) ~= "function" then
        return false
    end
    return G.buttons:get_UIE_by_ID("play_button") ~= nil
        and G.buttons:get_UIE_by_ID("discard_button") ~= nil
end

function Readiness.ready(current_phase, now, state)
    if current_phase ~= "hand_play" then
        state.hand_signature = nil
        state.hand_stable_since = nil
    end

    if not G or not G.GAME or not G.CONTROLLER then return false end
    if not controller_locks_clear() then return false end
    if G.GAME.STOP_USE and G.GAME.STOP_USE > 0 then return false end
    if not base_queue_empty() then return false end

    if current_phase == "hand_play" then
        if not G.STATE_COMPLETE then return false end
        if not hand_buttons_live() then return false end
        if (SMODS.cards_to_draw or 0) > 0 or SMODS.draw_queued then return false end

        local signature = hand_signature()
        if state.hand_signature ~= signature then
            state.hand_signature = signature
            state.hand_stable_since = now
            return false
        end
        if not state.hand_stable_since
            or now - state.hand_stable_since < HAND_SETTLE_SECONDS then
            return false
        end
    end
    return true
end

return Readiness
