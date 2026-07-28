local Readiness = {}

local HAND_SETTLE_SECONDS = 0.60
local TRANSITION_GRACE_SECONDS = 0.85
local PACK_SETTLE_SECONDS = 0.25

local function controller_locks_clear()
    if G.CONTROLLER.locked then return false end
    for _, locked in pairs(G.CONTROLLER.locks or {}) do
        if locked then return false end
    end
    return true
end

local function hand_cards_stationary()
    for _, card in ipairs(G.hand and G.hand.cards or {}) do
        if card.STATIONARY ~= true then return false end
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

local function area_signature(area)
    local parts = {}
    for _, card in ipairs(area and area.cards or {}) do
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
    if state.readiness_phase ~= current_phase then
        state.readiness_phase = current_phase
        state.pack_signature = nil
        state.pack_stable_since = nil
    end

    if current_phase ~= "hand_play" then
        state.hand_signature = nil
        state.hand_stable_since = nil
    end

    if not G or not G.GAME or not G.CONTROLLER then
        return false, "game objects unavailable"
    end

    if current_phase == "hand_play" then
        if not G.STATE_COMPLETE then return false, "state incomplete" end
        if not hand_buttons_live() then return false, "hand buttons unavailable" end

        local signature = hand_signature()
        if state.hand_signature ~= signature then
            state.hand_signature = signature
            state.hand_stable_since = now
            return false, "hand changed"
        end
        if not state.hand_stable_since
            or now - state.hand_stable_since < HAND_SETTLE_SECONDS then
            return false, "hand settling"
        end

        -- Steamodded and controller flags are useful while cards are being
        -- drawn, but third-party UI/animation mods can leave one truthy after
        -- the visible hand has finished. Bound their veto so a stale flag
        -- cannot disable the bridge for the rest of the run. The ordered card
        -- signature above still resets for every newly emplaced card.
        local stable_for = now - state.hand_stable_since
        if stable_for < TRANSITION_GRACE_SECONDS then
            if (SMODS.cards_to_draw or 0) > 0 or SMODS.draw_queued then
                return false, "draw pending"
            end
            if not hand_cards_stationary() then
                return false, "cards moving"
            end
            if not controller_locks_clear() then
                return false, "controller locked"
            end
            if G.GAME.STOP_USE and G.GAME.STOP_USE > 0 then
                return false, "use transition"
            end
        end
    elseif current_phase == "blind_select" then
        if not G.STATE_COMPLETE then return false, "state incomplete" end
        if not G.blind_select then return false, "blind UI unavailable" end
        if state.executor
            and not state.executor.callback_available({"select_blind"}) then
            return false, "blind button unavailable"
        end
    elseif current_phase == "shop" then
        if not G.STATE_COMPLETE then return false, "state incomplete" end
        if not G.shop or G.shop.REMOVED then return false, "shop UI unavailable" end
        if type(G.shop.get_UIE_by_ID) ~= "function"
            or not G.shop:get_UIE_by_ID("next_round_button") then
            return false, "shop leave button unavailable"
        end

        -- update_shop marks the state complete before its delayed opening
        -- event has populated the card areas. Leaving during that interval
        -- removes G.shop while the event still references it and crashes the
        -- game. A new shop object is ready only after vanilla has populated at
        -- least one of its guaranteed inventory areas.
        if state.initialized_shop ~= G.shop then
            local inventory_count = #(G.shop_jokers and G.shop_jokers.cards or {})
                + #(G.shop_vouchers and G.shop_vouchers.cards or {})
                + #(G.shop_booster and G.shop_booster.cards or {})
            if inventory_count == 0 then return false, "shop inventory initializing" end
            state.initialized_shop = G.shop
        end
    elseif current_phase == "booster_pack" then
        if not G.STATE_COMPLETE then return false, "state incomplete" end
        if not G.pack_cards or #(G.pack_cards.cards or {}) == 0 then
            return false, "pack contents unavailable"
        end
        local signature = area_signature(G.pack_cards)
        if state.pack_signature ~= signature then
            state.pack_signature = signature
            state.pack_stable_since = now
            return false, "pack contents changed"
        end
        if not state.pack_stable_since
            or now - state.pack_stable_since < PACK_SETTLE_SECONDS then
            return false, "pack contents settling"
        end
    end
    return true, nil
end

return Readiness
