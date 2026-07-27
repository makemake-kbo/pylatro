local Executor = {}

local function walk(node, visit, seen)
    if type(node) ~= "table" then return nil end
    seen = seen or {}
    if seen[node] then return nil end
    seen[node] = true
    local result = visit(node)
    if result then return result end
    for _, key in ipairs({"children", "nodes", "UIBox", "definition"}) do
        local child = rawget(node, key)
        if type(child) == "table" then
            for _, nested in pairs(child) do
                local found = walk(nested, visit, seen)
                if found then return found end
            end
        end
    end
    return nil
end

local function roots()
    local result = {}
    if G.ROOM then table.insert(result, G.ROOM) end
    if G.HUD then table.insert(result, G.HUD) end
    if G.blind_select then table.insert(result, G.blind_select) end
    if G.shop then table.insert(result, G.shop) end
    if G.booster_pack then table.insert(result, G.booster_pack) end
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
    for _, root in ipairs(roots()) do
        local found = walk(root, function(node)
            local config = node.config or {}
            if config.button == name and node_refers_to(node, ref)
                and config.disable_button ~= true and config.disabled ~= true then
                return node
            end
        end)
        if found then return found end
    end
    if ref and ref.children then
        return walk(ref.children, function(node)
            local config = node.config or {}
            if config.button == name and config.disable_button ~= true
                and config.disabled ~= true then return node end
        end)
    end
    return nil
end

local function invoke(names, ref)
    for _, name in ipairs(names) do
        local node = find_callback(name, ref)
        if node and G.FUNCS and type(G.FUNCS[name]) == "function" then
            G.FUNCS[name](node)
            return true
        end
    end
    return false, "no enabled UI callback: " .. table.concat(names, "/")
end

local function find_card(id)
    for _, area in ipairs({
        G.hand, G.deck, G.discard, G.jokers, G.consumeables,
        G.shop_jokers, G.shop_vouchers, G.shop_booster, G.pack_cards,
    }) do
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
        return invoke(action.type == "play"
            and {"play_cards_from_highlighted"}
            or {"discard_cards_from_highlighted"})
    elseif action.type == "shop_buy" then
        local card = find_card(action.item_id)
        if not card then return false, "shop item is no longer live" end
        return invoke({"buy_from_shop", "use_card"}, card)
    elseif action.type == "shop_reroll" then
        return invoke({"reroll_shop"})
    elseif action.type == "shop_sell" then
        local card = find_card(action.item_id)
        if not card then return false, "owned item is no longer live" end
        return invoke({"sell_card"}, card)
    elseif action.type == "shop_leave" then
        return invoke({"toggle_shop", "next_round"})
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
