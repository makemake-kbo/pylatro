local Executor = {}

local function walk(node, visit, seen)
    if type(node) ~= "table" then return nil end
    seen = seen or {}
    if seen[node] then return nil end
    seen[node] = true
    local result = visit(node)
    if result then return result end

    -- UIBox instances keep the executable UIElement tree under UIRoot. The
    -- `definition` field is only the construction schema and must not be
    -- passed to G.FUNCS callbacks.
    local ui_root = rawget(node, "UIRoot")
    if type(ui_root) == "table" then
        local found = walk(ui_root, visit, seen)
        if found then return found end
    end

    for _, key in ipairs({"children", "nodes"}) do
        local children = rawget(node, key)
        if type(children) == "table" then
            for _, child in pairs(children) do
                local found = walk(child, visit, seen)
                if found then return found end
            end
        end
    end

    -- Some UI elements embed another UIBox in config.object.
    local config = rawget(node, "config")
    local object = type(config) == "table" and rawget(config, "object")
    if type(object) == "table" then
        local found = walk(object, visit, seen)
        if found then return found end
    end
    return nil
end

local function roots()
    local result = {}
    local seen = {}
    local function append(root)
        if type(root) == "table" and not seen[root] then
            seen[root] = true
            table.insert(result, root)
        end
    end

    -- Named live UI boxes cover vanilla decision screens.
    if G.buttons then append(G.buttons) end
    if G.ROOM then table.insert(result, G.ROOM) end
    if G.HUD then append(G.HUD) end
    if G.HUD_blind then append(G.HUD_blind) end
    if G.blind_select then append(G.blind_select) end
    if G.shop then append(G.shop) end
    if G.booster_pack then append(G.booster_pack) end
    if G.round_eval then append(G.round_eval) end
    if G.OVERLAY_MENU then append(G.OVERLAY_MENU) end

    -- Card-local buy/use/sell buttons are separate UIBox instances. G.I.UIBOX
    -- is Balatro's registry of currently live boxes and catches those without
    -- relying on their visual nesting.
    for _, root in pairs(G.I and G.I.UIBOX or {}) do append(root) end
    return result
end

local function node_refers_to(node, ref)
    if not ref then return true end
    local config = node.config or {}
    return config.ref_table == ref
        or config.card == ref
        or node.card == ref
        or node == ref
end

local function find_callback(name, ref)
    if ref then
        local found = walk(ref, function(node)
            local config = node.config or {}
            if config.button == name and node_refers_to(node, ref)
                and config.disable_button ~= true and config.disabled ~= true
                and node.REMOVED ~= true
                and (not node.states or node.states.visible ~= false) then
                return node
            end
        end)
        if found then return found end
    end

    for _, root in ipairs(roots()) do
        local found = walk(root, function(node)
            local config = node.config or {}
            if config.button == name and node_refers_to(node, ref)
                and config.disable_button ~= true and config.disabled ~= true
                and node.REMOVED ~= true
                and (not node.states or node.states.visible ~= false) then
                return node
            end
        end)
        if found then return found end
    end
    return nil
end

local function invoke(names, ref)
    for _, name in ipairs(names) do
        local node = find_callback(name, ref)
        if node then
            -- Card buttons are sometimes created and invoked in this same
            -- update. Run Balatro's own gate once so can_select_card,
            -- can_buy, and related checks decide whether the callback is
            -- actually live before execution.
            local config = node.config or {}
            local gate = config.func
            if gate and G.FUNCS and type(G.FUNCS[gate]) == "function" then
                G.FUNCS[gate](node)
            end
            if config.button == name
                and config.disable_button ~= true and config.disabled ~= true
                and G.FUNCS and type(G.FUNCS[name]) == "function" then
                G.FUNCS[name](node)
                return true
            end
        end
    end
    return false, "no enabled UI callback: " .. table.concat(names, "/")
end

function Executor.callback_available(names, ref)
    for _, name in ipairs(names) do
        if find_callback(name, ref) then return true end
    end
    return false
end

local function invoke_hand_button(action_type)
    local callback = action_type == "play"
        and "play_cards_from_highlighted"
        or "discard_cards_from_highlighted"
    local button_id = action_type == "play" and "play_button" or "discard_button"

    if not G.buttons or type(G.buttons.get_UIE_by_ID) ~= "function" then
        return false, "hand button UI is not live: " .. button_id
    end
    local node = G.buttons:get_UIE_by_ID(button_id)
    if not node or type(node.config) ~= "table" then
        return false, "hand button UI node is absent: " .. button_id
    end

    -- Highlighting and the HTTP callback happen in the same update. Refresh
    -- Balatro's own can_play/can_discard gate immediately instead of waiting a
    -- frame for UIElement:update to set config.button.
    local gate = node.config.func
    if gate and G.FUNCS and type(G.FUNCS[gate]) == "function" then
        G.FUNCS[gate](node)
    end
    if node.config.button ~= callback then
        return false, "hand button is disabled after live legality check: " .. button_id
    end
    if not G.FUNCS or type(G.FUNCS[callback]) ~= "function" then
        return false, "Balatro callback is unavailable: " .. callback
    end
    G.FUNCS[callback](node)
    return true
end

local function invoke_shop_leave()
    if not G.shop or G.shop.REMOVED
        or type(G.shop.get_UIE_by_ID) ~= "function" then
        return false, "active shop UI is unavailable"
    end
    local node = G.shop:get_UIE_by_ID("next_round_button")
    if not node or type(node.config) ~= "table"
        or node.config.button ~= "toggle_shop"
        or node.config.disable_button == true or node.config.disabled == true then
        return false, "active shop leave button is unavailable"
    end
    if not G.FUNCS or type(G.FUNCS.toggle_shop) ~= "function" then
        return false, "Balatro callback is unavailable: toggle_shop"
    end
    G.FUNCS.toggle_shop(node)
    return true
end

local function find_card(id)
    -- Use names rather than a table containing optional nil values: ipairs
    -- stops at the first hole, which could hide every shop or pack area.
    for _, area_name in ipairs({
        "hand", "deck", "discard", "jokers", "consumeables",
        "shop_jokers", "shop_vouchers", "shop_booster", "pack_cards",
    }) do
        local area = G[area_name]
        for _, card in ipairs(area and area.cards or {}) do
            if PYLATRO_BRIDGE.serializer.card_id(card) == id then return card, area end
        end
    end
    return nil
end

local function clear_highlight(area)
    if not area then return end
    if area.unhighlight_all then
        area:unhighlight_all()
    else
        area.highlighted = {}
    end
end

local function highlight_ids(ids, expected_area)
    clear_highlight(expected_area)
    for _, id in ipairs(ids or {}) do
        local card, area = find_card(id)
        if not card or area ~= expected_area then return false, "card is absent or in the wrong area: " .. id end
        if expected_area.add_to_highlighted then
            expected_area:add_to_highlighted(card, true)
        else
            table.insert(expected_area.highlighted, card)
        end
    end

    local highlighted = {}
    for _, card in ipairs(expected_area and expected_area.highlighted or {}) do
        highlighted[PYLATRO_BRIDGE.serializer.card_id(card)] = true
    end
    for _, id in ipairs(ids or {}) do
        if not highlighted[id] then
            return false, "Balatro refused to highlight card: " .. id
        end
    end
    return true
end

local function validate_ids(ids)
    for _, id in ipairs(ids or {}) do
        if not find_card(id) then return false, "object is no longer live: " .. id end
    end
    return true
end

function Executor.execute(action)
    if type(action) ~= "table" or type(action.type) ~= "string" then
        return false, "malformed semantic action"
    end
    local ok, err = validate_ids(action.card_ids)
    if not ok then return false, err end
    ok, err = validate_ids(action.joker_ids)
    if not ok then return false, err end

    if action.type == "blind_play" then
        return invoke({"select_blind"})
    elseif action.type == "blind_skip" then
        return invoke({"skip_blind"})
    elseif action.type == "blind_reroll" then
        return invoke({"reroll_boss"})
    elseif action.type == "play" or action.type == "discard" then
        ok, err = highlight_ids(action.card_ids, G.hand)
        if not ok then return false, err end
        return invoke_hand_button(action.type)
    elseif action.type == "shop_buy" then
        local card = find_card(action.item_id)
        if not card then return false, "shop item is no longer live" end
        return invoke({"buy_from_shop", "use_card"}, card)
    elseif action.type == "shop_reroll" then
        return invoke({"reroll_shop"})
    elseif action.type == "shop_sell" then
        local card, area = find_card(action.item_id)
        if not card then return false, "owned item is no longer live" end
        if area ~= G.jokers and area ~= G.consumeables then
            return false, "shop sell target is not an owned joker or consumable"
        end
        -- Owned Sell controls are instantiated by Card:highlight().
        ok, err = highlight_ids({action.item_id}, area)
        if not ok then return false, err end
        return invoke({"sell_card"}, card)
    elseif action.type == "shop_leave" then
        return invoke_shop_leave()
    elseif action.type == "use_consumable" then
        local card = find_card(action.consumable_id)
        if not card then return false, "consumable is no longer live" end
        ok, err = highlight_ids(action.card_ids or {}, G.hand)
        if not ok then return false, err end
        if action.joker_ids and #action.joker_ids > 0 then
            ok, err = highlight_ids(action.joker_ids, G.jokers)
            if not ok then return false, err end
        end
        return invoke({"use_card"}, card)
    elseif action.type == "pack_claim" then
        local card = find_card(action.item_id)
        if not card then return false, "pack card is no longer live" end
        -- Pack Select/Use controls are instantiated by Card:highlight().
        -- Highlight the claimed card first, just as a player click does.
        ok, err = highlight_ids({action.item_id}, G.pack_cards)
        if not ok then return false, err end
        ok, err = highlight_ids(action.card_ids or {}, G.hand)
        if not ok then return false, err end
        if action.joker_ids and #action.joker_ids > 0 then
            ok, err = highlight_ids(action.joker_ids, G.jokers)
            if not ok then return false, err end
        end
        return invoke({"use_card"}, card)
    elseif action.type == "pack_skip" then
        return invoke({"skip_booster"})
    end
    return false, "unknown action type: " .. action.type
end

function Executor.cash_out()
    return invoke({"cash_out"})
end

return Executor
