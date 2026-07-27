local Serializer = {}

local function value_or(value, fallback)
    if value == nil then return fallback end
    return value
end

local function card_id(card)
    if not card then return "" end
    return "card:" .. tostring(card.unique_val or card.ID or card)
end

local RANK_SHORT = {
    Ace = "A", Jack = "J", Queen = "Q", King = "K",
}

local function front_key(card)
    if card.config and card.config.card_key then return card.config.card_key end
    if card.config and card.config.card and card.config.card.key then
        return card.config.card.key
    end
    local suit = card.base and card.base.suit or "Spades"
    local value = card.base and card.base.value or "Ace"
    return string.sub(suit, 1, 1) .. "_" .. value_or(RANK_SHORT[value], value)
end

local function center_key(card)
    if card.config and card.config.center and card.config.center.key then
        return card.config.center.key
    end
    return "c_base"
end

local function edition_key(card)
    if not card.edition then return nil end
    for _, key in ipairs({"negative", "polychrome", "holo", "foil"}) do
        if card.edition[key] then return key end
    end
    return card.edition.key
end

local function serialize_playing_card(card)
    local ability = card.ability or {}
    local base = card.base or {}
    return {
        id = card_id(card),
        front_key = front_key(card),
        suit = value_or(base.suit, "Spades"),
        rank = value_or(RANK_SHORT[base.value], value_or(base.value, "A")),
        center_key = center_key(card),
        edition = edition_key(card),
        seal = card.seal,
        perma_bonus = value_or(ability.perma_bonus, 0),
        debuff = card.debuff == true,
        face_down = card.facing == "back",
        forced_selection = ability.forced_selection == true,
        times_played = value_or(card.times_played, 0),
    }
end

local DYNAMIC_JOKER_FIELDS = {
    "mult", "h_mult", "h_x_mult", "h_dollars", "p_dollars", "t_mult",
    "t_chips", "Xmult", "h_size", "d_size", "extra", "extra_value", "type",
    "hands_played_at_create", "invis_rounds", "caino_xmult",
    "yorick_discards", "loyalty_remaining", "driver_tally", "stone_tally",
    "steel_tally", "to_do_poker_hand", "blueprint_compat", "money", "nine_tally",
}

local function serialize_joker(card)
    local ability = card.ability or {}
    local result = {
        id = card_id(card),
        center_key = center_key(card),
        edition = edition_key(card),
        eternal = ability.eternal == true,
        perishable = ability.perishable == true,
        perish_tally = ability.perish_tally,
        rental = ability.rental == true,
        debuff = card.debuff == true,
        sell_cost = value_or(card.sell_cost, 1),
    }
    for _, key in ipairs(DYNAMIC_JOKER_FIELDS) do
        local wire_key = key == "Xmult" and "x_mult" or key
        local value = ability[key]
        if value ~= nil and type(value) ~= "function" and type(value) ~= "userdata" then
            result[wire_key] = value
        end
    end
    return result
end

local function serialize_consumable(card)
    local ability = card.ability or {}
    return {
        id = card_id(card),
        center_key = center_key(card),
        edition = edition_key(card),
        extra_value = value_or(ability.extra_value, 0),
        sell_cost = value_or(card.sell_cost, 1),
    }
end

local function serialize_shop_card(card)
    local center = card.config and card.config.center or {}
    local set_name = center.set or (card.ability and card.ability.set) or ""
    return {
        id = card_id(card),
        center_key = center_key(card),
        card_type = set_name,
        cost = value_or(card.cost, 0),
        base_cost = value_or(card.base_cost, value_or(card.cost, 0)),
        front_key = (set_name == "Default" or set_name == "Enhanced") and front_key(card) or nil,
        edition = edition_key(card),
        seal = card.seal,
        eternal = card.ability and card.ability.eternal == true,
        perishable = card.ability and card.ability.perishable == true,
        rental = card.ability and card.ability.rental == true,
        shop_voucher = set_name == "Voucher",
    }
end

local function serialize_area(area, serializer)
    local result = {}
    if not area or not area.cards then return result end
    for _, card in ipairs(area.cards) do
        table.insert(result, serializer(card))
    end
    return result
end

local function unsupported_center(card, result)
    local center = card and card.config and card.config.center
    if not center or not center.mod then return end
    local mod_id = center.mod.id or center.mod.name or tostring(center.mod)
    if mod_id ~= "Steamodded" and mod_id ~= "pylatro_bridge" then
        table.insert(result, center.key or ("mod:" .. tostring(mod_id)))
    end
end

local function scan_area(area, result)
    if not area or not area.cards then return end
    for _, card in ipairs(area.cards) do unsupported_center(card, result) end
end

local function blind_type()
    if not G.GAME then return "Small" end
    return G.GAME.blind_on_deck or "Small"
end

local function blind_snapshot()
    local blind = G.GAME and G.GAME.blind
    if not blind then return {} end
    local config = blind.config and blind.config.blind or {}
    return {
        key = config.key or blind.config_blind_key,
        name = blind.name or config.name,
        chips = value_or(blind.chips, 0),
        mult = value_or(config.mult, 1),
        debuff = config.debuff or {},
        boss = config.boss,
    }
end

local function used_vouchers()
    local result = {}
    for key, enabled in pairs(G.GAME.used_vouchers or {}) do
        if enabled then result[key] = true end
    end
    return result
end

local function hand_levels()
    local result = {}
    for name, hand in pairs(G.GAME.hands or {}) do
        result[name] = {
            level = value_or(hand.level, 1),
            chips = value_or(hand.chips, 0),
            mult = value_or(hand.mult, 0),
            played = value_or(hand.played, 0),
            visible = hand.visible ~= false,
        }
    end
    return result
end

local function tags()
    local result = {}
    for _, tag in ipairs(G.GAME.tags or {}) do
        table.insert(result, tag.key or (tag.config and tag.config.tag and tag.config.tag.key) or tostring(tag))
    end
    return result
end

local function blind_tags()
    local result = {}
    local resets = G.GAME.round_resets or {}
    for key, tag in pairs(resets.blind_tags or {}) do
        result[key] = type(tag) == "table" and (tag.key or tostring(tag)) or tag
    end
    return result
end

function Serializer.card_id(card)
    return card_id(card)
end

function Serializer.snapshot(phase)
    local game = G.GAME
    local round = game.current_round or {}
    local resets = game.round_resets or {}
    local unsupported = {}
    for _, area in ipairs({
        G.hand, G.deck, G.discard, G.jokers, G.consumeables,
        G.shop_jokers, G.shop_vouchers, G.shop_booster, G.pack_cards,
    }) do
        scan_area(area, unsupported)
    end

    local shop = {
        cards = serialize_area(G.shop_jokers, serialize_shop_card),
        vouchers = serialize_area(G.shop_vouchers, serialize_shop_card),
        boosters = serialize_area(G.shop_booster, serialize_shop_card),
    }
    local pack = nil
    if phase == "booster_pack" then
        pack = {
            booster_key = game.pack_key or (game.pack and game.pack.key) or "",
            state_name = tostring(G.STATE),
            choices_remaining = value_or(game.pack_choices, 1),
            cards = serialize_area(G.pack_cards, serialize_shop_card),
        }
    end

    local deck_key = game.selected_back and game.selected_back.effect
        and game.selected_back.effect.center and game.selected_back.effect.center.key
    deck_key = deck_key or (game.selected_back and game.selected_back.key) or "b_red"

    return {
        seed = tostring(game.pseudorandom and game.pseudorandom.seed or game.seeded or game.seed or ""),
        stake = value_or(game.stake, 1),
        deck_key = deck_key,
        dollars = value_or(game.dollars, 0),
        bankrupt_at = value_or(game.bankrupt_at, 0),
        interest_cap = value_or(game.interest_cap, 25),
        ante = value_or(resets.ante, 1),
        blind_ante = value_or(resets.blind_ante, value_or(resets.ante, 1)),
        round = value_or(game.round, 0),
        round_score = value_or(round.chips, 0),
        hands_left = value_or(round.hands_left, 0),
        discards_left = value_or(round.discards_left, 0),
        hands_played = value_or(round.hands_played, 0),
        discards_used = value_or(round.discards_used, 0),
        hand_size = G.hand and G.hand.config.card_limit or 8,
        joker_slots = G.jokers and G.jokers.config.card_limit or 5,
        consumable_slots = G.consumeables and G.consumeables.config.card_limit or 2,
        reroll_cost = value_or(round.reroll_cost, 5),
        free_rerolls = value_or(round.free_rerolls, 0),
        skips = value_or(game.skips, 0),
        blind_on_deck = blind_type(),
        blind_disabled = game.blind and game.blind.disabled == true,
        blind_triggered = game.blind_triggered == true,
        boss_rerolled = resets.boss_rerolled == true,
        blind = blind_snapshot(),
        blind_choices = resets.blind_choices or {},
        blind_states = resets.blind_states or {},
        blind_tags = blind_tags(),
        tags = tags(),
        cards = {
            hand = serialize_area(G.hand, serialize_playing_card),
            draw = serialize_area(G.deck, serialize_playing_card),
            discard = serialize_area(G.discard, serialize_playing_card),
            deck = serialize_area(G.playing_cards and {cards = G.playing_cards} or nil, serialize_playing_card),
        },
        jokers = serialize_area(G.jokers, serialize_joker),
        consumables = serialize_area(G.consumeables, serialize_consumable),
        vouchers = used_vouchers(),
        hands = hand_levels(),
        shop = shop,
        pack = pack,
        won = game.won == true
            or (G.STATES.GAME_WON ~= nil and G.STATE == G.STATES.GAME_WON),
        game_over = G.STATE == G.STATES.GAME_OVER,
        unsupported = unsupported,
    }
end

function Serializer.legality(phase)
    local legal = {
        blind_play = false, blind_skip = false, blind_reroll = false,
        play = false, discard = false,
        play_card_ids = {}, discard_card_ids = {}, use_consumable_ids = {},
        shop_buy_ids = {}, shop_reroll = false,
        shop_sell_joker_ids = {}, shop_sell_consumable_ids = {}, shop_leave = false,
        pack_claim_ids = {}, pack_skip = false,
    }
    if phase == "blind_select" then
        legal.blind_play = true
        legal.blind_skip = blind_type() ~= "Boss"
        legal.blind_reroll = blind_type() == "Boss"
            and value_or(G.GAME.dollars, 0) >= 10
            and not (G.GAME.round_resets and G.GAME.round_resets.boss_rerolled)
    elseif phase == "hand_play" then
        legal.play = value_or(G.GAME.current_round.hands_left, 0) > 0
        legal.discard = value_or(G.GAME.current_round.discards_left, 0) > 0
        for _, card in ipairs(G.hand and G.hand.cards or {}) do
            table.insert(legal.play_card_ids, card_id(card))
            table.insert(legal.discard_card_ids, card_id(card))
        end
        for _, card in ipairs(G.consumeables and G.consumeables.cards or {}) do
            if not card.debuff then table.insert(legal.use_consumable_ids, card_id(card)) end
        end
    elseif phase == "shop" then
        for _, area in ipairs({G.shop_jokers, G.shop_vouchers, G.shop_booster}) do
            for _, card in ipairs(area and area.cards or {}) do
                if card.cost <= value_or(G.GAME.dollars, 0) then
                    table.insert(legal.shop_buy_ids, card_id(card))
                end
            end
        end
        legal.shop_reroll = value_or(G.GAME.current_round.reroll_cost, 5)
            <= value_or(G.GAME.dollars, 0)
        for _, card in ipairs(G.jokers and G.jokers.cards or {}) do
            if not (card.ability and card.ability.eternal) then
                table.insert(legal.shop_sell_joker_ids, card_id(card))
            end
        end
        for _, card in ipairs(G.consumeables and G.consumeables.cards or {}) do
            table.insert(legal.shop_sell_consumable_ids, card_id(card))
        end
        legal.shop_leave = true
    elseif phase == "booster_pack" then
        for _, card in ipairs(G.pack_cards and G.pack_cards.cards or {}) do
            if not card.debuff then table.insert(legal.pack_claim_ids, card_id(card)) end
        end
        legal.pack_skip = true
    end
    return legal
end

function Serializer.fingerprint(phase)
    local function canonical(value, seen)
        local kind = type(value)
        if kind == "nil" then return "null" end
        if kind == "boolean" or kind == "number" or kind == "string" then
            return kind .. ":" .. tostring(value)
        end
        if kind ~= "table" then return kind end
        seen = seen or {}
        if seen[value] then return "<cycle>" end
        seen[value] = true
        local keys = {}
        for key in pairs(value) do table.insert(keys, key) end
        table.sort(keys, function(left, right)
            return type(left) .. ":" .. tostring(left) < type(right) .. ":" .. tostring(right)
        end)
        local parts = {"{"}
        for _, key in ipairs(keys) do
            table.insert(parts, canonical(key, seen))
            table.insert(parts, "=")
            table.insert(parts, canonical(value[key], seen))
            table.insert(parts, ";")
        end
        table.insert(parts, "}")
        seen[value] = nil
        return table.concat(parts)
    end

    local serialized = canonical({
        phase = phase,
        game_state = G.STATE,
        state = Serializer.snapshot(phase),
        legal = Serializer.legality(phase),
    })
    local hash = 5381
    for index = 1, #serialized do
        hash = (hash * 33 + string.byte(serialized, index)) % 2147483647
    end
    return tostring(hash) .. ":" .. tostring(#serialized)
end

return Serializer
