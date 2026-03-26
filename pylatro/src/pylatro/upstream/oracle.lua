-- Oracle: self-contained Lua module for Balatro parity testing
-- Uses the original game data tables and RNG functions.
-- math.randomseed/math.random are overridden by Python to use LuaJIT.

oracle = {}

-- Global stubs (minimal LOVE2D replacements)
G = {}
HEX = function(v) return v end
localize = function(v)
    if type(v) == 'table' then return v.key or v[1] or '' end
    return v
end

-- RNG functions (extracted from misc_functions.lua lines 253-320)
function pseudohash(str)
    local num = 1
    for i = #str, 1, -1 do
        num = ((1.1239285023 / num) * string.byte(str, i) * math.pi + math.pi * i) % 1
    end
    return num
end

function pseudoseed(key, state)
    state = state or (G.GAME and G.GAME.pseudorandom) or {}
    local seed_str = state.seed or ''
    if not state.values then state.values = {} end

    if not state.values[key] then
        state.values[key] = pseudohash(key .. seed_str)
    end

    state.values[key] = math.abs(tonumber(string.format("%.13f", (2.134453429141 + state.values[key] * 1.72431234) % 1)))
    return (state.values[key] + (state.hashed_seed or 0.5)) / 2
end

function pseudorandom(seed, min, max)
    if type(seed) == 'string' then seed = pseudoseed(seed) end
    math.randomseed(seed)
    if min and max then return math.random(min, max)
    else return math.random() end
end

function pseudorandom_element(_t, seed)
    if seed then math.randomseed(seed) end
    local keys = {}
    for k, v in pairs(_t) do
        keys[#keys + 1] = { k = k, v = v }
    end
    if keys[1] and keys[1].v and type(keys[1].v) == 'table' and keys[1].v.sort_id then
        table.sort(keys, function(a, b) return a.v.sort_id < b.v.sort_id end)
    else
        -- Check if all keys are sequential integers (array-style table)
        local all_numeric = true
        for _, entry in ipairs(keys) do
            if type(entry.k) ~= "number" then all_numeric = false; break end
        end
        if all_numeric then
            table.sort(keys, function(a, b) return a.k < b.k end)
        else
            table.sort(keys, function(a, b) return tostring(a.k) < tostring(b.k) end)
        end
    end
    local idx = math.random(#keys)
    local key = keys[idx].k
    return _t[key], key
end

-- Fisher-Yates shuffle (from misc_functions.lua)
local function pseudoshuffle(list, seed)
    if seed then math.randomseed(seed) end
    for i = #list, 2, -1 do
        local j = math.random(i)
        list[i], list[j] = list[j], list[i]
    end
    return list
end

-- Data loading: extract table definitions from game.lua
local function extract_table(source, assignment)
    local start = source:find(assignment, 1, true)
    if not start then return nil end
    local brace_start = source:find("{", start, true)
    if not brace_start then return nil end
    local depth = 0
    local in_single = false
    local in_double = false
    local in_comment = false
    local prev = ""
    for i = brace_start, #source do
        local c = source:sub(i, i)
        local next_c = source:sub(i+1, i+1)
        if in_comment then
            if c == "\n" then in_comment = false end
        elseif in_single then
            if c == "'" and prev ~= "\\" then in_single = false end
        elseif in_double then
            if c == '"' and prev ~= "\\" then in_double = false end
        else
            if c == "-" and next_c == "-" then in_comment = true
            elseif c == "'" then in_single = true
            elseif c == '"' then in_double = true
            elseif c == "{" then depth = depth + 1
            elseif c == "}" then
                depth = depth - 1
                if depth == 0 then
                    return source:sub(brace_start, i)
                end
            end
        end
        prev = c
    end
    return nil
end

local function load_game_data()
    local f = io.open(VENDOR_PATH .. "/game.lua", "r")
    if not f then error("Cannot open " .. VENDOR_PATH .. "/game.lua") end
    local source = f:read("*a")
    f:close()

    local assignments = {
        blinds = "self.P_BLINDS =",
        centers = "self.P_CENTERS =",
        cards = "self.P_CARDS =",
        hands = "        hands = {",
    }

    local data = {}
    for name, assignment in pairs(assignments) do
        local table_src = extract_table(source, assignment)
        if table_src then
            local fn, err = load("return " .. table_src)
            if fn then
                local ok, val = pcall(fn)
                if ok then data[name] = val end
            end
        end
    end
    return data
end

-- Card data: suit map and rank order
local suit_map = { S = "Spades", H = "Hearts", D = "Diamonds", C = "Clubs" }
local rank_order = { "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A" }

-- Boss selection (must match Python's get_new_boss in pool.py)
local function get_new_boss(state, data)
    local eligible = {}
    local a = math.max(1, state.round_resets.ante)
    for k, blind in pairs(data.blinds) do
        local boss = blind.boss
        if boss then
            if (not boss.showdown and boss.min <= a and (a % state.win_ante ~= 0 or state.round_resets.ante < 2))
               or (boss.showdown and a % state.win_ante == 0 and state.round_resets.ante >= 2) then
                eligible[k] = true
            end
        end
    end
    -- Filter banned_keys
    for k in pairs(eligible) do
        if state.banned_keys and state.banned_keys[k] then eligible[k] = nil end
    end
    -- Track usage - bosses_used is pre-initialized with 0 for all bosses
    local min_use = 100
    for k, _ in pairs(eligible) do
        local uses = state.bosses_used[k] or 0
        eligible[k] = uses
        if uses < min_use then min_use = uses end
    end
    -- Filter to min usage
    local filtered = {}
    for k, v in pairs(eligible) do
        if v == min_use then filtered[k] = v end
    end
    local _, key = pseudorandom_element(filtered, pseudoseed("boss", state.pseudorandom))
    if key then
        state.bosses_used[key] = (state.bosses_used[key] or 0) + 1
    end
    return key or "bl_hook"
end

-- Pool building helpers (must match Python's pool.py logic)
local function build_center_pools(data)
    data.center_pools = {
        Booster = {}, Default = {}, Enhanced = {}, Edition = {},
        Joker = {}, Tarot = {}, Planet = {}, Tarot_Planet = {},
        Spectral = {}, Consumeables = {}, Voucher = {}, Back = {},
        Tag = {}, Seal = {}, Stake = {}, Demo = {},
    }
    data.joker_rarity_pools = { {}, {}, {}, {} }

    for k, v in pairs(data.centers) do
        v.key = k
        local set = v.set
        if set == "Joker" then
            data.center_pools.Joker[#data.center_pools.Joker + 1] = v
        end
        if set and v.demo and v.pos then
            data.center_pools.Demo[#data.center_pools.Demo + 1] = v
        end
        if not v.wip then
            if set and set ~= "Joker" and not v.skip_pool and not v.omit then
                if data.center_pools[set] then
                    data.center_pools[set][#data.center_pools[set] + 1] = v
                end
            end
            if set == "Tarot" or set == "Planet" then
                data.center_pools.Tarot_Planet[#data.center_pools.Tarot_Planet + 1] = v
            end
            if v.consumeable then
                data.center_pools.Consumeables[#data.center_pools.Consumeables + 1] = v
            end
            if set == "Joker" and v.rarity and not v.demo then
                local r = math.floor(v.rarity)
                if r >= 1 and r <= 4 then
                    data.joker_rarity_pools[r][#data.joker_rarity_pools[r] + 1] = v
                end
            end
        end
    end

    -- Sort pools by order (matches Python's _build_pools)
    local function by_order(a, b) return (a.order or 0) < (b.order or 0) end
    for _, pool_name in ipairs({
        "Joker", "Tarot", "Planet", "Tarot_Planet", "Spectral",
        "Voucher", "Booster", "Consumeables", "Enhanced", "Stake", "Tag", "Seal",
    }) do
        table.sort(data.center_pools[pool_name], by_order)
    end
    for _, pool in ipairs(data.joker_rarity_pools) do
        table.sort(pool, by_order)
    end
end

-- get_current_pool: build filtered pool for a card type (matches Python's pool.py:9-85)
local function get_current_pool(state, card_type, append, rarity, legendary)
    local data = state.data
    local pool_key
    local starting_pool

    if card_type == "Joker" then
        local rarity_roll = rarity or
            pseudorandom("rarity" .. tostring(state.round_resets.ante) .. (append or ""))
        local rarity_index
        if legendary then
            rarity_index = 4
        elseif rarity_roll > 0.95 then
            rarity_index = 3
        elseif rarity_roll > 0.7 then
            rarity_index = 2
        else
            rarity_index = 1
        end
        starting_pool = data.joker_rarity_pools[rarity_index]
        pool_key = "Joker" .. tostring(rarity_index) .. (legendary and "" or (append or ""))
    else
        starting_pool = data.center_pools[card_type]
        pool_key = card_type .. (append or "")
    end

    local pool = {}
    local pool_size = 0
    for _, proto in ipairs(starting_pool) do
        local add = false
        if card_type == "Enhanced" then
            add = true
        elseif card_type == "Tag" then
            add = (not proto.requires or (data.centers[proto.requires] and data.centers[proto.requires].discovered))
                and (not proto.min_ante or proto.min_ante <= state.round_resets.ante)
        elseif not (state.used_jokers[proto.key] and true) and (proto.unlocked ~= false or proto.rarity == 4) then
            if proto.set == "Voucher" then
                if not state.used_vouchers[proto.key] then
                    add = true
                    if proto.requires then
                        if type(proto.requires) == "table" then
                            for _, req in ipairs(proto.requires) do
                                if not state.used_vouchers[req] then add = false end
                            end
                        elseif type(proto.requires) == "string" then
                            if not state.used_vouchers[proto.requires] then add = false end
                        end
                    end
                    for _, voucher in ipairs(state.shop.vouchers) do
                        if voucher.center_key == proto.key then add = false end
                    end
                end
            elseif proto.set == "Planet" then
                local config = type(proto.config) == "table" and proto.config or {}
                if config.softlock then
                    add = state.hands[config.hand_type] and state.hands[config.hand_type].played > 0
                else
                    add = true
                end
            elseif proto.enhancement_gate then
                add = false
                for _, card in ipairs(state.deck_cards) do
                    if card.center_key == proto.enhancement_gate then add = true; break end
                end
            else
                add = true
            end
            if proto.name == "Black Hole" or proto.name == "The Soul" then
                add = false
            end
        end

        if proto.no_pool_flag and state.pool_flags[proto.no_pool_flag] then
            add = false
        end
        if proto.yes_pool_flag and not state.pool_flags[proto.yes_pool_flag] then
            add = false
        end
        if add and not state.banned_keys[proto.key] then
            pool[#pool + 1] = proto.key
            pool_size = pool_size + 1
        else
            pool[#pool + 1] = "UNAVAILABLE"
        end
    end

    if pool_size == 0 then
        local fallback = {
            Tarot = "c_strength", Tarot_Planet = "c_strength",
            Planet = "c_pluto", Spectral = "c_incantation",
            Joker = "j_joker", Voucher = "v_blank", Tag = "tag_handy",
        }
        pool = { fallback[card_type] or "j_joker" }
    end
    local suffix = legendary and "" or tostring(state.round_resets.ante)
    return pool, pool_key .. suffix
end

-- _pick_pool_key: pick a non-UNAVAILABLE key from pool (matches Python's pool.py:88-97)
local function pick_pool_key(state, pool, pool_key)
    local center, _ = pseudorandom_element(pool, pseudoseed(pool_key, state.pseudorandom))
    local reroll = 1
    while center == "UNAVAILABLE" do
        reroll = reroll + 1
        center, _ = pseudorandom_element(pool, pseudoseed(pool_key .. "_resample" .. tostring(reroll), state.pseudorandom))
    end
    return center
end

-- get_next_voucher_key: matches Python's pool.py:100-102
local function get_next_voucher_key(state)
    local pool, pool_key = get_current_pool(state, "Voucher")
    return pick_pool_key(state, pool, pool_key)
end

-- get_next_tag_key: matches Python's pool.py:105-107
local function get_next_tag_key(state, append)
    local pool, pool_key = get_current_pool(state, "Tag", append)
    return pick_pool_key(state, pool, pool_key)
end

-- poll_edition: matches Python's pool.py:173-200
local function poll_edition(state, key, mod, no_negative)
    key = key or "edition_generic"
    mod = mod or 1
    local edition_poll = pseudorandom(pseudoseed(key, state.pseudorandom))
    if edition_poll > 1 - 0.003 * mod and not no_negative then
        return { negative = true }
    end
    if edition_poll > 1 - 0.006 * state.edition_rate * mod then
        return { polychrome = true }
    end
    if edition_poll > 1 - 0.02 * state.edition_rate * mod then
        return { holo = true }
    end
    if edition_poll > 1 - 0.04 * state.edition_rate * mod then
        return { foil = true }
    end
    return nil
end

-- _apply_joker_stickers: matches Python's pool.py:203-233
local function apply_joker_stickers(state, center, source)
    local eternal = false
    local perishable = false
    local rental = false

    if state.modifiers.all_eternal and center.eternal_compat then
        eternal = true
    end

    if source == "shop" or source == "pack" then
        local eternal_key = (source == "pack" and "packetper" or "etperpoll") .. tostring(state.round_resets.ante)
        local eternal_poll = pseudorandom(eternal_key)
        if state.modifiers.enable_eternals_in_shop and eternal_poll > 0.7 and center.eternal_compat and not perishable then
            eternal = true
        elseif state.modifiers.enable_perishables_in_shop and eternal_poll > 0.4 and eternal_poll <= 0.7 and center.perishable_compat and not eternal then
            perishable = true
        end

        local rental_key = (source == "pack" and "packssjr" or "ssjr") .. tostring(state.round_resets.ante)
        if state.modifiers.enable_rentals_in_shop and pseudorandom(rental_key) > 0.7 then
            rental = true
        end
    end

    return eternal, perishable, rental
end

-- _edition_cost helper
local function edition_cost(edition)
    if not edition then return 0 end
    local c = 0
    if edition.holo then c = c + 3 end
    if edition.foil then c = c + 2 end
    if edition.polychrome then c = c + 5 end
    if edition.negative then c = c + 5 end
    return c
end

-- _calculate_cost: matches Python's _helpers.py:33-53
local function calculate_cost(state, center, edition, rental)
    local base_cost = math.max(1, center.cost or 1)
    local cost = math.max(1, math.floor(
        (base_cost + state.inflation + edition_cost(edition) + 0.5) * (100 - state.discount_percent) / 100
    ))
    if center.set == "Booster" and state.modifiers.booster_ante_scaling then
        cost = cost + state.round_resets.ante - 1
    end
    if rental then cost = 1 end
    return cost
end

-- _mark_center_used: matches Python's _helpers.py:29-30
local function mark_center_used(state, center_key)
    state.used_jokers[center_key] = true
end

-- create_card_spec: matches Python's pool.py:236-303
local function create_card_spec(state, card_type, forced_key, append, source, soulable)
    local requested_type = card_type
    local data = state.data

    -- Soul check (only for soulable cards)
    if not forced_key and soulable and not state.banned_keys.c_soul then
        if (card_type == "Tarot" or card_type == "Spectral" or card_type == "Tarot_Planet") then
            if not (state.used_jokers.c_soul) then
                if pseudorandom("soul_" .. card_type .. tostring(state.round_resets.ante)) > 0.997 then
                    forced_key = "c_soul"
                end
            end
        end
        if (card_type == "Planet" or card_type == "Spectral") then
            if not (state.used_jokers.c_black_hole) then
                if pseudorandom("soul_" .. card_type .. tostring(state.round_resets.ante)) > 0.997 then
                    forced_key = "c_black_hole"
                end
            end
        end
    end

    if card_type == "Base" then
        forced_key = "c_base"
    end

    local center_key
    local center
    if forced_key and not state.banned_keys[forced_key] then
        center_key = forced_key
        center = data.centers[center_key]
        if center and center.set and center.set ~= "Default" then
            card_type = center.set
        else
            card_type = requested_type
        end
    else
        local pool, pool_key = get_current_pool(state, card_type, append)
        center_key = pick_pool_key(state, pool, pool_key)
        center = data.centers[center_key]
    end

    local front_key = nil
    if card_type == "Base" or card_type == "Enhanced" then
        _, front_key = pseudorandom_element(
            data.cards,
            pseudoseed("front" .. (append or "") .. tostring(state.round_resets.ante), state.pseudorandom)
        )
    end

    local ed = nil
    local eternal = false
    local perishable = false
    local rental = false
    if card_type == "Joker" then
        eternal, perishable, rental = apply_joker_stickers(state, center, source)
        ed = poll_edition(state, "edi" .. (append or "") .. tostring(state.round_resets.ante))
    end

    mark_center_used(state, center_key)

    return {
        center_key = center_key,
        card_type = card_type,
        cost = calculate_cost(state, center, ed, rental),
        base_cost = math.max(1, center.cost or 1),
        front_key = front_key,
        edition = ed,
        seal = nil,
        eternal = eternal,
        perishable = perishable,
        rental = rental,
    }
end

-- create_shop_card: matches Python's shop.py:19-62
local function create_shop_card(state)
    local total_rate = state.joker_rate + state.tarot_rate + state.planet_rate
        + state.playing_card_rate + state.spectral_rate
    local polled_rate = pseudorandom(pseudoseed("cdt" .. tostring(state.round_resets.ante), state.pseudorandom)) * total_rate
    local running = 0.0

    -- Determine card types in order
    local playing_card_type = "Base"
    if state.used_vouchers.v_illusion and pseudorandom("illusion") > 0.6 then
        playing_card_type = "Enhanced"
    end

    local card_types = {
        { "Joker", state.joker_rate },
        { "Tarot", state.tarot_rate },
        { "Planet", state.planet_rate },
        { playing_card_type, state.playing_card_rate },
        { "Spectral", state.spectral_rate },
    }

    for _, ct in ipairs(card_types) do
        local ctype, value = ct[1], ct[2]
        if running < polled_rate and polled_rate <= running + value then
            local card = create_card_spec(state, ctype, nil, "sho", "shop", false)
            -- Post-creation illusion edition for Base/Enhanced
            if (ctype == "Base" or ctype == "Enhanced") and state.used_vouchers.v_illusion then
                if pseudorandom("illusion") > 0.8 then
                    local edition_poll = pseudorandom("illusion")
                    if edition_poll > 1 - 0.15 then
                        card.edition = { polychrome = true }
                    elseif edition_poll > 0.5 then
                        card.edition = { holo = true }
                    else
                        card.edition = { foil = true }
                    end
                    card.cost = calculate_cost(state, state.data.centers[card.center_key], card.edition, card.rental)
                end
            end
            return card
        end
        running = running + value
    end
    error("Shop card selection failed")
end

-- get_pack: matches Python's pool.py:142-170
local function get_pack(state, key)
    if not state.first_shop_buffoon and not state.banned_keys.p_buffoon_normal_1 then
        state.first_shop_buffoon = true
        return "p_buffoon_normal_" .. tostring(math.random(1, 2))
    end

    local cumulative = 0.0
    for _, proto in ipairs(state.data.center_pools.Booster) do
        if not state.banned_keys[proto.key] then
            cumulative = cumulative + (proto.weight or 1)
        end
    end

    local poll = pseudorandom(pseudoseed((key or "pack_generic") .. tostring(state.round_resets.ante), state.pseudorandom)) * cumulative
    local current = 0.0
    for _, proto in ipairs(state.data.center_pools.Booster) do
        if state.banned_keys[proto.key] then goto continue end
        local weight = proto.weight or 1
        current = current + weight
        if current >= poll then
            return proto.key
        end
        ::continue::
    end
    error("Booster selection failed")
end

function oracle.create_run(seed, stake, deck_key)
    local data = load_game_data()

    local state = {
        seed = seed,
        stake = stake or 1,
        deck_key = deck_key or "b_red",
        dollars = 4,
        ante = 1,
        round = 0,
        hands_played = 0,
        deck_cards = {},
        draw_pile = {},
        hand_cards = {},
        discard_pile = {},
        play_cards = {},
        jokers = {},
        joker_keys = {},
        consumables = {},
        used_vouchers = {},
        used_jokers = {},
        banned_keys = {},
        pool_flags = {},
        tags = {},
        probabilities = { normal = 1 },
        modifiers = {},
        joker_rate = 20,
        tarot_rate = 4,
        planet_rate = 4,
        playing_card_rate = 0,
        spectral_rate = 0,
        edition_rate = 1,
        inflation = 0,
        discount_percent = 0,
        first_shop_buffoon = false,
        current_voucher = nil,
        current_round = {
            hands_left = 4,
            hands_played = 0,
            discards_left = 3,
            discards_used = 0,
            hand_size = 8,
            first_hand_drawn = false,
            reroll_cost = 5,
        },
        round_resets = {
            hands = 4,
            discards = 3,
            reroll_cost = 5,
            ante = 1,
            blind_states = { Small = "Select", Big = "Upcoming", Boss = "Upcoming" },
            blind_choices = { Small = "bl_small", Big = "bl_big" },
            blind_tags = {},
        },
        starting_params = {
            dollars = 4,
            hand_size = 8,
            discards = 3,
            hands = 4,
            reroll_cost = 5,
            joker_slots = 5,
            ante_scaling = 1,
            consumable_slots = 2,
        },
        shop = { joker_max = 2, cards = {}, vouchers = {}, boosters = {} },
        data = data,
        blind_disabled = false,
        blind_triggered = false,
        blind_prepped = false,
        win_ante = 8,
    }

    -- Set up RNG state
    state.pseudorandom = {
        seed = seed,
        hashed_seed = pseudohash(seed),
        values = {},
    }
    G.GAME = state

    -- Apply deck config (must match Python's _apply_deck)
    local deck_center = data.centers and data.centers[state.deck_key]
    if deck_center and deck_center.config then
        local cfg = deck_center.config
        if cfg.discards then
            state.starting_params.discards = state.starting_params.discards + cfg.discards
        end
        if cfg.hands then
            state.starting_params.hands = state.starting_params.hands + cfg.hands
        end
        if cfg.dollars then
            state.starting_params.dollars = state.starting_params.dollars + cfg.dollars
        end
        if cfg.hand_size then
            state.starting_params.hand_size = state.starting_params.hand_size + cfg.hand_size
        end
        if cfg.joker_slot then
            state.starting_params.joker_slots = state.starting_params.joker_slots + cfg.joker_slot
        end
        if cfg.consumable_slot then
            state.starting_params.consumable_slots = state.starting_params.consumable_slots + cfg.consumable_slot
        end
        if cfg.ante_scaling then
            state.starting_params.ante_scaling = cfg.ante_scaling
        end
    end

    -- Apply starting_params to round_resets and state
    state.round_resets.hands = state.starting_params.hands
    state.round_resets.discards = state.starting_params.discards
    state.round_resets.reroll_cost = state.starting_params.reroll_cost
    state.dollars = state.starting_params.dollars
    state.current_round.reroll_cost = state.starting_params.reroll_cost
    state.current_round.discards_left = state.round_resets.discards
    state.current_round.hands_left = state.round_resets.hands
    state.current_round.hand_size = state.starting_params.hand_size

    -- Build starting deck (52 cards, sorted)
    local card_keys = {}
    for _, suit in ipairs({"C", "D", "H", "S"}) do
        for _, rank in ipairs(rank_order) do
            card_keys[#card_keys + 1] = suit .. "_" .. rank
        end
    end
    table.sort(card_keys)

    for _, key in ipairs(card_keys) do
        local suit = key:sub(1, 1)
        local rank = key:sub(3, 3)
        local card = {
            front_key = key,
            suit = suit_map[suit],
            rank = rank,
            center_key = "c_base",
            debuff = false,
            destroyed = false,
            shattered = false,
            played_this_ante = false,
            discarded = false,
            face_down = false,
            forced_selection = false,
            times_played = 0,
            perma_bonus = 0,
        }
        state.deck_cards[#state.deck_cards + 1] = card
    end

    -- Shuffle deck (must match Python's deck shuffle)
    local shuffle_seed = pseudoseed("shuffle", state.pseudorandom)
    local shuffled = {}
    for _, card in ipairs(state.deck_cards) do shuffled[#shuffled + 1] = card end
    pseudoshuffle(shuffled, shuffle_seed)
    state.deck_cards = shuffled
    state.draw_pile = {}
    for _, card in ipairs(shuffled) do state.draw_pile[#state.draw_pile + 1] = card end

    -- Initialize hand levels from data
    if data.hands then
        state.hands = {}
        for name, hand in pairs(data.hands) do
            state.hands[name] = {
                level = hand.level or 1,
                played = hand.played or 0,
                played_this_round = hand.played_this_round or 0,
                visible = hand.visible or false,
                chips = hand.chips or 0,
                mult = hand.mult or 0,
                s_chips = hand.s_chips or 0,
                s_mult = hand.s_mult or 0,
                l_chips = hand.l_chips or 0,
                l_mult = hand.l_mult or 0,
                order = hand.order or 0,
            }
        end
    end

    -- Build center pools (must happen before pool operations)
    build_center_pools(data)

    -- Select boss blind (must match Python's get_new_boss)
    -- Initialize bosses_used with 0 for all boss blinds (like Python's dict comprehension)
    state.bosses_used = {}
    if data.blinds then
        for k, v in pairs(data.blinds) do
            if v.boss then state.bosses_used[k] = 0 end
        end
    end
    state.blind_on_deck = "Small"
    state.skips = 0

    state.round_resets.blind_choices.Boss = get_new_boss(state, data)

    -- Get voucher and tags using proper pool functions (must match Python's create_run_state)
    state.current_voucher = get_next_voucher_key(state)
    state.round_resets.blind_tags.Small = get_next_tag_key(state)
    state.round_resets.blind_tags.Big = get_next_tag_key(state)

    return state
end

-- Nominal value for hand sorting (must match Python's _card_nominal)
local rank_to_nominal = {
    ["2"] = 2, ["3"] = 3, ["4"] = 4, ["5"] = 5, ["6"] = 6,
    ["7"] = 7, ["8"] = 8, ["9"] = 9, ["T"] = 10, ["J"] = 10,
    ["Q"] = 10, ["K"] = 10, ["A"] = 11,
}
local suit_to_nominal = { Diamonds = 0.01, Clubs = 0.02, Hearts = 0.03, Spades = 0.04 }

local function card_nominal(card, centers)
    local base = rank_to_nominal[card.rank] or 0
    local face_nominal = 0
    if card.rank == "J" then face_nominal = 0.1
    elseif card.rank == "Q" then face_nominal = 0.2
    elseif card.rank == "K" then face_nominal = 0.3
    elseif card.rank == "A" then face_nominal = 0.4
    end
    local suit_nom = suit_to_nominal[card.suit] or 0
    local center = centers and centers[card.center_key]
    local suit_mult = 1
    if center and center.effect == "Stone Card" then suit_mult = -1000 end
    return base + suit_nom * suit_mult + suit_nom * 0.0001 * suit_mult + face_nominal
end

local function sort_hand(hand_cards, centers)
    table.sort(hand_cards, function(a, b)
        return card_nominal(a, centers) > card_nominal(b, centers)
    end)
end

function oracle.start_blind(state, blind_type)
    G.GAME = state
    blind_type = blind_type or "Small"
    local data = state.data

    -- select_blind: look up blind from choices
    local blind_key = state.round_resets.blind_choices[blind_type]
    state.round_resets.blind = data.blinds[blind_key]

    -- select_blind: reset current_round fields
    state.round_resets.blind_states[blind_type] = "Current"
    state.shop = { joker_max = 2, cards = {}, vouchers = {}, boosters = {} }
    state.current_round.discards_left = math.max(0, state.round_resets.discards)
    state.current_round.hands_left = math.max(1, state.round_resets.hands)
    state.current_round.hands_played = 0
    state.current_round.discards_used = 0
    state.current_round.reroll_cost_increase = 0
    state.current_round.used_packs = {}
    state.current_round.free_rerolls = 0
    state.current_round.dollars = 0

    -- _reset_for_blind
    local ante = state.round_resets.ante or 1
    local subhash_suffix = "S"
    if blind_type == "Big" then subhash_suffix = "B"
    elseif blind_type ~= "Small" then subhash_suffix = "L"
    end
    state.subhash = tostring(ante) .. subhash_suffix
    state.blind_disabled = false
    state.blind_triggered = false
    state.blind_prepped = false
    state.current_round.first_hand_drawn = false
    state.current_round.hand_size = math.max(0,
        state.starting_params.hand_size + (state.round_resets.temp_handsize or 0))

    local blind = state.round_resets.blind or {}
    local blind_name = blind.name or ""

    if blind_name == "The Water" then
        state.current_round.discards_left = 0
    elseif blind_name == "The Needle" then
        state.current_round.hands_left = 1
    elseif blind_name == "The Manacle" then
        state.current_round.hand_size = math.max(0, state.current_round.hand_size - 1)
    end

    -- Reset card flags + debuff (match Python's _debuff_card)
    for _, card in ipairs(state.deck_cards) do
        card.discarded = false
        card.forced_selection = false
        card.face_down = false
        -- Debuff logic
        if state.blind_disabled then
            card.debuff = false
        elseif blind_name == "Verdant Leaf" then
            card.debuff = true
        else
            local debuff_cfg = blind.debuff or {}
            if type(debuff_cfg) == "table" and debuff_cfg.suit and card.suit == debuff_cfg.suit then
                card.debuff = true
            elseif type(debuff_cfg) == "table" and debuff_cfg.is_face == "face" and (card.rank == "J" or card.rank == "Q" or card.rank == "K") then
                card.debuff = true
            elseif blind_name == "The Pillar" and card.played_this_ante then
                card.debuff = true
            else
                card.debuff = false
            end
        end
    end

    -- apply_setting_blind: no-op for empty joker list

    -- _fold_areas_back_into_deck
    if #state.hand_cards > 0 then
        for _, card in ipairs(state.hand_cards) do
            state.discard_pile[#state.discard_pile + 1] = card
        end
        state.hand_cards = {}
    end
    if #state.play_cards > 0 then
        for _, card in ipairs(state.play_cards) do
            if not card.destroyed and not card.shattered then
                state.discard_pile[#state.discard_pile + 1] = card
            end
        end
        state.play_cards = {}
    end
    if #state.discard_pile > 0 then
        local new_draw = {}
        for _, card in ipairs(state.discard_pile) do
            new_draw[#new_draw + 1] = card
        end
        for _, card in ipairs(state.draw_pile) do
            new_draw[#new_draw + 1] = card
        end
        state.draw_pile = new_draw
        state.discard_pile = {}
    end
    -- Filter out destroyed/shattered
    local filtered = {}
    for _, card in ipairs(state.draw_pile) do
        if not card.destroyed and not card.shattered then
            filtered[#filtered + 1] = card
        end
    end
    state.draw_pile = filtered

    -- Shuffle draw_pile
    local shuffle_seed = pseudoseed("nr" .. tostring(ante), state.pseudorandom)
    pseudoshuffle(state.draw_pile, shuffle_seed)

    -- draw_to_hand
    local hand_size = state.current_round.hand_size
    local hand_space = math.min(#state.draw_pile, math.max(0, hand_size - #state.hand_cards))
    for i = 1, hand_space do
        local card = state.draw_pile[#state.draw_pile]
        state.draw_pile[#state.draw_pile] = nil
        card.discarded = false
        card.forced_selection = false
        card.face_down = false  -- no boss blind flip logic for simple case
        state.hand_cards[#state.hand_cards + 1] = card
    end

    -- Sort hand (matches Python's _sort_hand)
    sort_hand(state.hand_cards, data.centers)

    -- _first_hand_drawn: no-op for empty joker list
    state.current_round.first_hand_drawn = true

    -- _drawn_to_hand: no-op for simple blinds with no jokers

    return state
end

-- Hand evaluation helpers (must match Python's scoring.py)
local rank_to_id = {
    ["2"] = 2, ["3"] = 3, ["4"] = 4, ["5"] = 5, ["6"] = 6,
    ["7"] = 7, ["8"] = 8, ["9"] = 9, ["T"] = 10, ["J"] = 11,
    ["Q"] = 12, ["K"] = 13, ["A"] = 14,
}

local function card_id(card)
    return rank_to_id[card.rank]
end

local function get_x_same(num, hand)
    local vals = {}
    for i = 0, 14 do vals[i] = {} end
    for i = #hand, 1, -1 do
        local curr = { hand[i] }
        for j = 1, #hand do
            if i ~= j and card_id(hand[i]) == card_id(hand[j]) then
                curr[#curr + 1] = hand[j]
            end
        end
        if #curr == num then
            vals[card_id(curr[1])] = curr
        end
    end
    local result = {}
    for i = 14, 0, -1 do
        if #vals[i] > 0 then
            result[#result + 1] = vals[i]
        end
    end
    return result
end

local function get_flush(hand)
    local required = 5
    if #hand > 5 or #hand < required then return {} end
    for _, suit in ipairs({"Spades", "Hearts", "Clubs", "Diamonds"}) do
        local cards = {}
        for _, card in ipairs(hand) do
            if card.suit == suit then
                cards[#cards + 1] = card
            end
        end
        if #cards >= required then
            return { cards }
        end
    end
    return {}
end

local function get_straight(hand)
    local required = 5
    if #hand > 5 or #hand < required then return {} end

    local ids = {}
    for _, card in ipairs(hand) do
        local cid = card_id(card)
        if cid > 1 and cid < 15 then
            if not ids[cid] then ids[cid] = {} end
            ids[cid][#ids[cid] + 1] = card
        end
    end

    local straight_cards = {}
    local straight_length = 0
    local straight = false
    for j = 1, 14 do
        local actual = j == 1 and 14 or j
        if ids[actual] then
            straight_length = straight_length + 1
            for _, c in ipairs(ids[actual]) do
                straight_cards[#straight_cards + 1] = c
            end
        else
            straight_length = 0
            if not straight then
                straight_cards = {}
            end
            if straight then
                break
            end
        end
        if straight_length >= required then
            straight = true
        end
    end
    if not straight then return {} end
    return { straight_cards }
end

local function get_highest(hand, centers)
    if #hand == 0 then return {} end
    local highest = hand[1]
    local highest_nom = card_nominal(highest, centers)
    for i = 2, #hand do
        local nom = card_nominal(hand[i], centers)
        if nom > highest_nom then
            highest = hand[i]
            highest_nom = nom
        end
    end
    return { { highest } }
end

local poker_hand_order = {
    "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
    "Four of a Kind", "Full House", "Flush", "Straight",
    "Three of a Kind", "Two Pair", "Pair", "High Card",
}

local function evaluate_poker_hand(hand, centers)
    local results = {
        ["Flush Five"] = {},
        ["Flush House"] = {},
        ["Five of a Kind"] = {},
        ["Straight Flush"] = {},
        ["Four of a Kind"] = {},
        ["Full House"] = {},
        ["Flush"] = {},
        ["Straight"] = {},
        ["Three of a Kind"] = {},
        ["Two Pair"] = {},
        ["Pair"] = {},
        ["High Card"] = {},
    }

    local _5 = get_x_same(5, hand)
    local _4 = get_x_same(4, hand)
    local _3 = get_x_same(3, hand)
    local _2 = get_x_same(2, hand)
    local _flush = get_flush(hand)
    local _straight = get_straight(hand)
    local _highest = get_highest(hand, centers)

    if #_5 > 0 and #_flush > 0 then
        results["Flush Five"] = _5
    end
    if #_3 > 0 and #_2 > 0 and #_flush > 0 then
        local combined = {}
        for _, c in ipairs(_3[1]) do combined[#combined + 1] = c end
        for _, c in ipairs(_2[1]) do combined[#combined + 1] = c end
        results["Flush House"] = { combined }
    end
    if #_5 > 0 then
        results["Five of a Kind"] = _5
    end
    if #_flush > 0 and #_straight > 0 then
        local flush_cards = {}
        for _, c in ipairs(_flush[1]) do flush_cards[#flush_cards + 1] = c end
        local straight_cards = {}
        for _, c in ipairs(_straight[1]) do straight_cards[#straight_cards + 1] = c end
        local ret = {}
        for _, c in ipairs(flush_cards) do ret[#ret + 1] = c end
        for _, c in ipairs(straight_cards) do
            local found = false
            for _, fc in ipairs(flush_cards) do
                if c == fc then found = true; break end
            end
            if not found then ret[#ret + 1] = c end
        end
        results["Straight Flush"] = { ret }
    end
    if #_4 > 0 then
        results["Four of a Kind"] = _4
    end
    if #_3 > 0 and #_2 > 0 then
        local combined = {}
        for _, c in ipairs(_3[1]) do combined[#combined + 1] = c end
        for _, c in ipairs(_2[1]) do combined[#combined + 1] = c end
        results["Full House"] = { combined }
    end
    if #_flush > 0 then
        results["Flush"] = _flush
    end
    if #_straight > 0 then
        results["Straight"] = _straight
    end
    if #_3 > 0 then
        results["Three of a Kind"] = _3
    end
    if #_2 == 2 or (#_3 == 1 and #_2 == 1) then
        local second_pair = #_2 > 1 and _2[2] or _3[1]
        local combined = {}
        for _, c in ipairs(_2[1]) do combined[#combined + 1] = c end
        for _, c in ipairs(second_pair) do combined[#combined + 1] = c end
        results["Two Pair"] = { combined }
    end
    if #_2 > 0 then
        results["Pair"] = _2
    end
    if #_highest > 0 then
        results["High Card"] = _highest
    end

    -- Backfill lower hands from higher ones (match Python)
    if #results["Five of a Kind"] > 0 then
        local cards = results["Five of a Kind"][1]
        results["Four of a Kind"] = { { cards[1], cards[2], cards[3], cards[4] } }
    end
    if #results["Four of a Kind"] > 0 then
        local cards = results["Four of a Kind"][1]
        results["Three of a Kind"] = { { cards[1], cards[2], cards[3] } }
    end
    if #results["Three of a Kind"] > 0 then
        local cards = results["Three of a Kind"][1]
        results["Pair"] = { { cards[1], cards[2] } }
    end

    return results
end

local function get_poker_hand_info(hand, centers)
    local poker_hands = evaluate_poker_hand(hand, centers)
    local text = "High Card"
    local scoring_hand = #poker_hands["High Card"] > 0 and poker_hands["High Card"][1] or {}

    for _, hand_name in ipairs(poker_hand_order) do
        if #poker_hands[hand_name] > 0 then
            text = hand_name
            scoring_hand = poker_hands[hand_name][1]
            break
        end
    end

    return text, scoring_hand
end

-- Helper: get blind name from state
local function get_blind_name(state)
    local blind = state.round_resets.blind or {}
    return blind.name or ""
end

function oracle.play_hand(state, card_indices)
    G.GAME = state
    local data = state.data

    -- Convert card_indices from Python list/lupa table to a proper Lua array
    local indices = {}
    if type(card_indices) == "table" then
        for _, v in ipairs(card_indices) do
            indices[#indices + 1] = v
        end
    else
        -- lupa userdata: iterate with python protocol
        for v in python.iter(card_indices) do
            indices[#indices + 1] = v
        end
    end

    -- Collect selected cards (before removing from hand, for _press_play)
    local selected = {}
    for _, idx in ipairs(indices) do
        selected[#selected + 1] = state.hand_cards[idx]
    end

    state.current_round.hands_left = math.max(0, state.current_round.hands_left - 1)

    -- _press_play: boss blind pre-scoring effects (runs BEFORE cards removed from hand)
    local blind_name = get_blind_name(state)
    if not state.blind_disabled then
        if blind_name == "The Hook" and #state.hand_cards > 0 then
            local available = {}
            for _, c in ipairs(state.hand_cards) do available[#available + 1] = c end
            for _ = 1, math.min(2, #available) do
                if #available == 0 then break end
                local chosen, idx = pseudorandom_element(available, pseudoseed("hook", state.pseudorandom))
                -- Remove from hand_cards
                for i = #state.hand_cards, 1, -1 do
                    if state.hand_cards[i] == chosen then
                        table.remove(state.hand_cards, i)
                        break
                    end
                end
                chosen.discarded = true
                state.discard_pile[#state.discard_pile + 1] = chosen
                -- Remove from available
                local new_available = {}
                for _, c in ipairs(available) do
                    if c ~= chosen then new_available[#new_available + 1] = c end
                end
                available = new_available
            end
            state.blind_triggered = true
        end
        if blind_name == "The Tooth" then
            for _ = 1, #selected do
                state.dollars = math.max(0, state.dollars - 1)
            end
            state.blind_triggered = true
        end
        if blind_name == "The Fish" or blind_name == "Crimson Heart" then
            state.blind_prepped = true
        end
    end

    -- NOW remove selected cards from hand_cards (after _press_play)
    -- Remove by identity (like Python's _remove_exact), not by index,
    -- because _press_play (e.g. The Hook) may have already removed some
    -- Track which cards were actually still in hand (not already discarded by Hook)
    local actually_played = {}
    for _, card in ipairs(selected) do
        local found = false
        for i = #state.hand_cards, 1, -1 do
            if state.hand_cards[i] == card then
                table.remove(state.hand_cards, i)
                found = true
                break
            end
        end
        if found then
            actually_played[#actually_played + 1] = card
        end
    end

    -- Update card tracking and move to play_cards (only cards not already discarded by Hook)
    for _, card in ipairs(actually_played) do
        card.times_played = card.times_played + 1
        card.played_this_ante = true
        card.discarded = false
        card.forced_selection = false
        state.play_cards[#state.play_cards + 1] = card
    end

    -- Evaluate poker hand
    local play_list = {}
    for _, card in ipairs(state.play_cards) do play_list[#play_list + 1] = card end
    local hand_name, scoring_hand = get_poker_hand_info(play_list, data.centers)

    -- Update hand played counts (match Python's score_hand)
    state.hands[hand_name].played = state.hands[hand_name].played + 1
    state.hands[hand_name].played_this_round = state.hands[hand_name].played_this_round + 1
    state.hands[hand_name].visible = true

    -- Score: base chips/mult from hand level + card chip bonuses
    local hand_chips = state.hands[hand_name].chips
    local mult = state.hands[hand_name].mult

    -- For each scoring card, add rank_to_nominal chip bonus
    for _, card in ipairs(scoring_hand) do
        hand_chips = hand_chips + (rank_to_nominal[card.rank] or 0)
    end

    -- Apply blind modify_hand effects (The Flint)
    if not state.blind_disabled then
        local bn = get_blind_name(state)
        if bn == "The Flint" then
            state.blind_triggered = true
            mult = math.max(math.floor(mult * 0.5 + 0.5), 1)
            hand_chips = math.max(math.floor(hand_chips * 0.5 + 0.5), 0)
        end
        -- The Ox: playing the most played hand sets money to $0
        if bn == "The Ox" then
            state.blind_triggered = false
            if hand_name == (state.current_round.most_played_poker_hand or "High Card") then
                state.blind_triggered = true
                state.dollars = 0
            end
        end
    end

    local total = math.floor(hand_chips * mult)

    -- Increment global counters
    state.hands_played = state.hands_played + 1
    state.current_round.hands_played = state.current_round.hands_played + 1

    -- Move played cards to discard_pile
    while #state.play_cards > 0 do
        local card = table.remove(state.play_cards, 1)
        card.face_down = false
        state.discard_pile[#state.discard_pile + 1] = card
    end

    -- Draw replacement cards from draw_pile to hand_cards (only if hand is empty)
    if #state.hand_cards == 0 and #state.draw_pile > 0 then
        local hand_size = state.current_round.hand_size
        local hand_space = math.min(#state.draw_pile, math.max(0, hand_size - #state.hand_cards))
        for i = 1, hand_space do
            local card = state.draw_pile[#state.draw_pile]
            state.draw_pile[#state.draw_pile] = nil
            card.discarded = false
            card.forced_selection = false
            card.face_down = false
            state.hand_cards[#state.hand_cards + 1] = card
        end

        -- Sort hand after drawing
        sort_hand(state.hand_cards, data.centers)
    end

    return state
end

function oracle.discard(state, card_indices)
    G.GAME = state
    local data = state.data

    -- Convert card_indices from Python list/lupa table to a proper Lua array
    local indices = {}
    if type(card_indices) == "table" then
        for _, v in ipairs(card_indices) do
            indices[#indices + 1] = v
        end
    else
        -- lupa userdata: iterate with python protocol
        for v in python.iter(card_indices) do
            indices[#indices + 1] = v
        end
    end

    -- Move selected cards from hand_cards to discard_pile (1-indexed)
    local selected = {}
    for _, idx in ipairs(indices) do
        selected[#selected + 1] = state.hand_cards[idx]
    end

    -- Sort indices in descending order to remove from end first
    local sorted_indices = {}
    for _, idx in ipairs(indices) do sorted_indices[#sorted_indices + 1] = idx end
    table.sort(sorted_indices, function(a, b) return a > b end)
    for _, idx in ipairs(sorted_indices) do
        table.remove(state.hand_cards, idx)
    end

    -- Mark cards as discarded and move to discard_pile
    for _, card in ipairs(selected) do
        card.discarded = true
        card.face_down = false
        state.discard_pile[#state.discard_pile + 1] = card
    end

    -- Decrement discards_left, increment discards_used
    state.current_round.discards_left = math.max(0, state.current_round.discards_left - 1)
    state.current_round.discards_used = (state.current_round.discards_used or 0) + 1

    -- Draw replacement cards from draw_pile to hand_cards
    local hand_size = state.current_round.hand_size
    local hand_space = math.min(#state.draw_pile, math.max(0, hand_size - #state.hand_cards))
    for i = 1, hand_space do
        local card = state.draw_pile[#state.draw_pile]
        state.draw_pile[#state.draw_pile] = nil
        card.discarded = false
        card.forced_selection = false
        card.face_down = false
        state.hand_cards[#state.hand_cards + 1] = card
    end

    -- Sort hand after drawing
    sort_hand(state.hand_cards, data.centers)

    return state
end

function oracle.cash_out(state)
    G.GAME = state
    local data = state.data

    -- apply_end_of_round: no joker effects for basic parity (no jokers in initial runs)

    -- Reset fields
    state.current_round.jokers_purchased = 0
    state.current_round.discards_left = math.max(0, state.round_resets.discards)
    state.current_round.hands_left = math.max(1, state.round_resets.hands)
    state.shop = { joker_max = 2, cards = {}, vouchers = {}, boosters = {} }
    state.current_round.used_packs = {}

    if state.round_resets.blind_states.Boss == "Defeated" then
        -- Calculate most_played_poker_hand
        local most_played_name = "High Card"
        local most_played_count = 0
        local most_played_order = 0
        for name, hand in pairs(state.hands) do
            local played = hand.played or 0
            local order = hand.order or 0
            if played > most_played_count or (played == most_played_count and order < most_played_order) then
                most_played_name = name
                most_played_count = played
                most_played_order = order
            end
        end
        state.current_round.most_played_poker_hand = most_played_name

        -- Check win condition
        if state.round_resets.ante == state.win_ante then
            state.won = true
        end

        -- Increment ante
        state.round_resets.ante = state.round_resets.ante + 1
        state.round_resets.blind_ante = state.round_resets.ante

        -- Reset played_this_round for all hands
        for _, hand in pairs(state.hands) do
            hand.played_this_round = 0
        end

        -- Reset played_this_ante for all deck_cards
        for _, card in ipairs(state.deck_cards) do
            card.played_this_ante = false
        end

        -- Next voucher (using proper pool function, matches Python)
        state.current_voucher = get_next_voucher_key(state)

        -- Next tags (twice, using proper pool function, matches Python)
        state.round_resets.blind_tags.Small = get_next_tag_key(state)
        state.round_resets.blind_tags.Big = get_next_tag_key(state)
    end

    -- Reset blinds (only when boss defeated)
    if state.round_resets.blind_states.Boss == "Defeated" then
        state.round_resets.blind_states = { Small = "Upcoming", Big = "Upcoming", Boss = "Upcoming" }
        state.blind_on_deck = "Small"
        state.round_resets.boss_rerolled = false

        -- New boss (must match Python's get_new_boss)
        state.round_resets.blind_choices.Boss = get_new_boss(state, data)
    end

    return state
end

function oracle.skip_blind(state)
    G.GAME = state

    local skipped = state.blind_on_deck
    local skip_to = "Big"
    if skipped ~= "Small" then skip_to = "Boss" end

    state.skips = (state.skips or 0) + 1

    -- Apply tag if exists
    if state.round_resets.blind_tags and state.round_resets.blind_tags[skipped] then
        if not state.tags then state.tags = {} end
        state.tags[#state.tags + 1] = state.round_resets.blind_tags[skipped]
    end

    -- Update blind_states
    state.round_resets.blind_states[skipped] = "Skipped"
    state.round_resets.blind_states[skip_to] = "Select"
    state.blind_on_deck = skip_to

    return state
end

function oracle.populate_shop(state)
    G.GAME = state
    local data = state.data

    -- refresh_shop: generate shop cards (matches Python's populate_shop + refresh_shop)
    if #state.shop.cards == 0 then
        for _ = 1, state.shop.joker_max do
            state.shop.cards[#state.shop.cards + 1] = create_shop_card(state)
        end
    end

    -- Voucher (matches Python: if not shop.vouchers and current_voucher)
    if #state.shop.vouchers == 0 and state.current_voucher then
        local voucher = create_card_spec(state, "Voucher", state.current_voucher, nil, nil, false)
        voucher.shop_voucher = true
        state.shop.vouchers = { voucher }
    end

    -- Boosters (matches Python's populate_shop booster logic)
    if #state.shop.boosters == 0 then
        local boosters = {}
        -- Ensure used_packs has 2 entries
        if not state.current_round.used_packs then state.current_round.used_packs = {} end
        while #state.current_round.used_packs < 2 do
            state.current_round.used_packs[#state.current_round.used_packs + 1] = ""
        end
        for index = 1, 2 do
            if state.current_round.used_packs[index] == "" then
                state.current_round.used_packs[index] = get_pack(state, "shop_pack")
            end
            if state.current_round.used_packs[index] ~= "USED" then
                local booster = create_card_spec(state, "Booster", state.current_round.used_packs[index], nil, nil, false)
                booster.booster_pos = index
                boosters[#boosters + 1] = booster
            end
        end
        state.shop.boosters = boosters
    end

    return state
end

function oracle.reroll_shop(state)
    G.GAME = state
    if state.current_round.reroll_cost > 0 then
        state.dollars = state.dollars - state.current_round.reroll_cost
    end
    return state
end

function oracle.buy_card(state, index)
    G.GAME = state
    local data = state.data

    -- Pop card from shop (1-indexed)
    local card = table.remove(state.shop.cards, index)
    if not card then return state end

    state.dollars = state.dollars - card.cost
    local center = data.centers[card.center_key]

    if center.set == "Default" or center.set == "Enhanced" then
        -- Playing card: add to deck_cards
        local suit_letter = card.front_key:sub(1, 1)
        local suit_map = { S = "Spades", H = "Hearts", D = "Diamonds", C = "Clubs" }
        local new_card = {
            front_key = card.front_key,
            suit = suit_map[suit_letter] or "",
            rank = card.front_key:sub(3, 3),
            center_key = card.center_key,
            debuff = false, destroyed = false, shattered = false,
            played_this_ante = false, discarded = false, face_down = false,
            forced_selection = false, times_played = 0, perma_bonus = 0,
        }
        state.deck_cards[#state.deck_cards + 1] = new_card
    elseif center.consumeable then
        -- Consumable: add to consumables
        local cost = card.cost
        local cons = {
            center_key = card.center_key,
            edition = card.edition,
            sell_cost = math.max(1, math.floor(cost / 2)),
        }
        state.consumables[#state.consumables + 1] = cons
    else
        -- Joker: create joker instance and add
        local config = center.config or {}
        local extra = config.extra
        -- Shallow copy if extra is a table
        if type(extra) == "table" then
            local copy = {}
            for k, v in pairs(extra) do copy[k] = v end
            extra = copy
        end

        local cost = card.cost
        local joker = {
            center_key = card.center_key,
            edition = card.edition,
            eternal = card.eternal or false,
            perishable = card.perishable or false,
            rental = card.rental or false,
            mult = tonumber(config.mult) or 0,
            h_mult = tonumber(config.h_mult) or 0,
            h_x_mult = tonumber(config.h_x_mult) or 0,
            h_dollars = tonumber(config.h_dollars) or 0,
            p_dollars = tonumber(config.p_dollars) or 0,
            t_mult = tonumber(config.t_mult) or 0,
            t_chips = tonumber(config.t_chips) or 0,
            x_mult = tonumber(config.Xmult) or 1,
            h_size = tonumber(config.h_size) or 0,
            d_size = tonumber(config.d_size) or 0,
            extra = extra,
            type = tostring(config.type or ""),
            hands_played_at_create = state.hands_played,
            sell_cost = math.max(1, math.floor(cost / 2)),
        }

        state.jokers[#state.jokers + 1] = joker
        if not state.joker_keys then state.joker_keys = {} end
        state.joker_keys[#state.joker_keys + 1] = card.center_key

        -- Apply stat modifiers (matching instances.py add_joker)
        if joker.d_size > 0 then
            state.round_resets.discards = state.round_resets.discards + joker.d_size
            state.current_round.discards_left = state.current_round.discards_left + joker.d_size
        end
        if joker.h_size ~= 0 then
            state.starting_params.hand_size = state.starting_params.hand_size + joker.h_size
            state.current_round.hand_size = state.current_round.hand_size + joker.h_size
        end

        local name = center.name
        if name == "Credit Card" and type(joker.extra) == "number" then
            state.bankrupt_at = (state.bankrupt_at or 0) - joker.extra
        elseif name == "Chaos the Clown" then
            state.current_round.free_rerolls = (state.current_round.free_rerolls or 0) + 1
        elseif name == "Oops! All 6s" then
            for k, v in pairs(state.probabilities) do
                state.probabilities[k] = v * 2
            end
        elseif name == "To the Moon" and type(joker.extra) == "number" then
            state.interest_amount = (state.interest_amount or 5) + joker.extra
        elseif name == "Troubadour" and type(joker.extra) == "table" then
            local h_size = tonumber(joker.extra.h_size) or 0
            local h_plays = tonumber(joker.extra.h_plays) or 0
            state.starting_params.hand_size = state.starting_params.hand_size + h_size
            state.round_resets.hands = state.round_resets.hands + h_plays
            state.current_round.hand_size = state.current_round.hand_size + h_size
        elseif name == "Stuntman" and type(joker.extra) == "table" then
            local h_size = tonumber(joker.extra.h_size) or 0
            state.starting_params.hand_size = state.starting_params.hand_size - h_size
            state.current_round.hand_size = state.current_round.hand_size - h_size
        elseif name == "Turtle Bean" and type(joker.extra) == "table" then
            local h_size = tonumber(joker.extra.h_size) or 0
            state.starting_params.hand_size = state.starting_params.hand_size + h_size
            state.current_round.hand_size = state.current_round.hand_size + h_size
        elseif name == "To Do List" then
            -- RNG call: pseudorandom_element on visible hands (matches Python instances.py)
            local visible_hands = {}
            for hand_name, hand in pairs(state.hands) do
                if hand.visible then
                    visible_hands[#visible_hands + 1] = hand_name
                end
            end
            if #visible_hands > 0 then
                local seed = pseudoseed("to_do", state.pseudorandom)
                local hand_name = pseudorandom_element(visible_hands, seed)
                joker.to_do_poker_hand = hand_name
            end
        end
        if card.edition and type(card.edition) == "table" and card.edition.negative then
            state.starting_params.joker_slots = state.starting_params.joker_slots + 1
        end
    end

    return state
end

function oracle.finish_shop(state)
    G.GAME = state
    -- apply_end_shop is mostly joker effects (Perkeo)
    return state
end

function oracle.use_consumable(state, index, targets)
    G.GAME = state
    local cons = table.remove(state.consumables, index)
    if not cons then return state end

    local center = state.data.centers[cons.center_key]
    if not center then return state end

    -- Track usage (matches _register_consumable_use in consumables.py)
    if center.set == "Planet" or center.set == "Tarot" or center.set == "Spectral" then
        state.last_tarot_planet = cons.center_key
    end
    state.consumeable_usage_total = (state.consumeable_usage_total or 0) + 1

    -- Planet cards: level up the associated hand
    if center.set == "Planet" then
        local config = center.config or {}
        local hand_name = config.hand_type
        if hand_name and state.hands[hand_name] then
            state.hands[hand_name].level = state.hands[hand_name].level + 1
        end
    elseif center.name == "Black Hole" then
        for hand_name, hand in pairs(state.hands) do
            hand.level = hand.level + 1
        end
    end

    return state
end

function oracle.open_pack(state, index)
    G.GAME = state
    local data = state.data

    -- Pop booster from shop (1-indexed)
    local booster = table.remove(state.shop.boosters, index)
    if not booster then return state end

    state.dollars = state.dollars - booster.cost

    -- Mark used_packs slot as USED
    if booster.booster_pos then
        while #state.current_round.used_packs < booster.booster_pos do
            state.current_round.used_packs[#state.current_round.used_packs + 1] = ""
        end
        state.current_round.used_packs[booster.booster_pos] = "USED"
    end

    local center = data.centers[booster.center_key]
    local name = center.name or ""
    local config = type(center.config) == "table" and center.config or {}
    local size = config.extra or 0
    local choices = config.choose or 1

    local cards = {}
    for card_index = 1, size do
        local card
        if name:find("Arcana") then
            if state.used_vouchers.v_omen_globe and pseudorandom("omen_globe") > 0.8 then
                card = create_card_spec(state, "Spectral", nil, "ar2", "pack", true)
            else
                card = create_card_spec(state, "Tarot", nil, "ar1", "pack", true)
            end
        elseif name:find("Celestial") then
            local forced_key = nil
            if state.used_vouchers.v_telescope and card_index == 1 then
                -- Find most played visible hand
                local hand_name = nil
                local hand_tally = 0
                -- Use POKER_HANDS order
                local poker_hands_order = {
                    "Royal Flush", "Straight Flush", "Four of a Kind",
                    "Full House", "Flush", "Straight", "Three of a Kind",
                    "Two Pair", "Pair", "High Card",
                    "Flush Five", "Flush House", "Five of a Kind",
                }
                for _, nk in ipairs(poker_hands_order) do
                    local hand = state.hands[nk]
                    if hand and hand.visible and hand.played > hand_tally then
                        hand_name = nk
                        hand_tally = hand.played
                    end
                end
                if hand_name then
                    for _, proto in ipairs(data.center_pools.Planet) do
                        local pc = type(proto.config) == "table" and proto.config or {}
                        if pc.hand_type == hand_name then
                            forced_key = proto.key
                            break
                        end
                    end
                end
            end
            card = create_card_spec(state, "Planet", forced_key, "pl1", "pack", true)
        elseif name:find("Spectral") then
            card = create_card_spec(state, "Spectral", nil, "spe", "pack", true)
        elseif name:find("Standard") then
            local base_type
            if pseudorandom("stdset" .. tostring(state.round_resets.ante)) > 0.6 then
                base_type = "Enhanced"
            else
                base_type = "Base"
            end
            card = create_card_spec(state, base_type, nil, "sta", "pack", true)
            card.edition = poll_edition(state, "standard_edition" .. tostring(state.round_resets.ante), 2, true)
            local seal_poll = pseudorandom("stdseal" .. tostring(state.round_resets.ante))
            if seal_poll > 1 - 0.02 * 10 then
                local seal_type = pseudorandom("stdsealtype" .. tostring(state.round_resets.ante))
                if seal_type > 0.75 then
                    card.seal = "Red"
                elseif seal_type > 0.5 then
                    card.seal = "Blue"
                elseif seal_type > 0.25 then
                    card.seal = "Gold"
                else
                    card.seal = "Purple"
                end
            end
        elseif name:find("Buffoon") then
            card = create_card_spec(state, "Joker", nil, "buf", "pack", true)
        else
            error("Unknown booster pack: " .. name)
        end

        -- Recalculate cost if edition or rental
        if card.edition or card.rental then
            card.cost = calculate_cost(state, data.centers[card.center_key], card.edition, card.rental)
        end
        cards[#cards + 1] = card
    end

    state.pack = {
        booster_key = booster.center_key,
        cards = cards,
        choices_remaining = choices,
        source_slot = booster.booster_pos,
    }

    -- apply_open_booster: joker effects (no-op if no jokers)
    return state
end

function oracle.claim_card(state, index)
    G.GAME = state
    local data = state.data

    if not state.pack then return state end

    local card = table.remove(state.pack.cards, index)
    if not card then return state end

    local center = data.centers[card.center_key]

    if center.set == "Default" or center.set == "Enhanced" then
        -- Playing card: add to deck_cards and draw_pile
        local suit_letter = card.front_key:sub(1, 1)
        local sm = { S = "Spades", H = "Hearts", D = "Diamonds", C = "Clubs" }
        local new_card = {
            front_key = card.front_key,
            suit = sm[suit_letter] or "",
            rank = card.front_key:sub(3, 3),
            center_key = card.center_key,
            seal = card.seal,
            debuff = false, destroyed = false, shattered = false,
            played_this_ante = false, discarded = false, face_down = false,
            forced_selection = false, times_played = 0, perma_bonus = 0,
        }
        state.deck_cards[#state.deck_cards + 1] = new_card
        state.draw_pile[#state.draw_pile + 1] = new_card
    elseif center.consumeable then
        -- Consumable
        local cons = {
            center_key = card.center_key,
            edition = card.edition,
            sell_cost = math.max(1, math.floor(card.cost / 2)),
        }
        state.consumables[#state.consumables + 1] = cons
    else
        -- Joker
        local config = center.config or {}
        local extra = config.extra
        if type(extra) == "table" then
            local copy = {}
            for k, v in pairs(extra) do copy[k] = v end
            extra = copy
        end

        local joker = {
            center_key = card.center_key,
            edition = card.edition,
            eternal = card.eternal or false,
            perishable = card.perishable or false,
            rental = card.rental or false,
            mult = tonumber(config.mult) or 0,
            h_mult = tonumber(config.h_mult) or 0,
            h_x_mult = tonumber(config.h_x_mult) or 0,
            h_dollars = tonumber(config.h_dollars) or 0,
            p_dollars = tonumber(config.p_dollars) or 0,
            t_mult = tonumber(config.t_mult) or 0,
            t_chips = tonumber(config.t_chips) or 0,
            x_mult = tonumber(config.Xmult) or 1,
            h_size = tonumber(config.h_size) or 0,
            d_size = tonumber(config.d_size) or 0,
            extra = extra,
            type = tostring(config.type or ""),
            hands_played_at_create = state.hands_played,
            sell_cost = math.max(1, math.floor(card.cost / 2)),
        }

        state.jokers[#state.jokers + 1] = joker
        if not state.joker_keys then state.joker_keys = {} end
        state.joker_keys[#state.joker_keys + 1] = card.center_key

        -- Apply stat modifiers (same as buy_card)
        if joker.d_size > 0 then
            state.round_resets.discards = state.round_resets.discards + joker.d_size
            state.current_round.discards_left = state.current_round.discards_left + joker.d_size
        end
        if joker.h_size ~= 0 then
            state.starting_params.hand_size = state.starting_params.hand_size + joker.h_size
            state.current_round.hand_size = state.current_round.hand_size + joker.h_size
        end

        local cname = center.name
        if cname == "Credit Card" and type(joker.extra) == "number" then
            state.bankrupt_at = (state.bankrupt_at or 0) - joker.extra
        elseif cname == "Chaos the Clown" then
            state.current_round.free_rerolls = (state.current_round.free_rerolls or 0) + 1
        elseif cname == "Oops! All 6s" then
            for k, v in pairs(state.probabilities) do
                state.probabilities[k] = v * 2
            end
        elseif cname == "To the Moon" and type(joker.extra) == "number" then
            state.interest_amount = (state.interest_amount or 5) + joker.extra
        elseif cname == "Troubadour" and type(joker.extra) == "table" then
            local h_size = tonumber(joker.extra.h_size) or 0
            local h_plays = tonumber(joker.extra.h_plays) or 0
            state.starting_params.hand_size = state.starting_params.hand_size + h_size
            state.round_resets.hands = state.round_resets.hands + h_plays
            state.current_round.hand_size = state.current_round.hand_size + h_size
        elseif cname == "Stuntman" and type(joker.extra) == "table" then
            local h_size = tonumber(joker.extra.h_size) or 0
            state.starting_params.hand_size = state.starting_params.hand_size - h_size
            state.current_round.hand_size = state.current_round.hand_size - h_size
        elseif cname == "Turtle Bean" and type(joker.extra) == "table" then
            local h_size = tonumber(joker.extra.h_size) or 0
            state.starting_params.hand_size = state.starting_params.hand_size + h_size
            state.current_round.hand_size = state.current_round.hand_size + h_size
        elseif cname == "To Do List" then
            local visible_hands = {}
            for hand_name, hand in pairs(state.hands) do
                if hand.visible then
                    visible_hands[#visible_hands + 1] = hand_name
                end
            end
            if #visible_hands > 0 then
                local seed = pseudoseed("to_do", state.pseudorandom)
                local hand_name = pseudorandom_element(visible_hands, seed)
                joker.to_do_poker_hand = hand_name
            end
        end
        if card.edition and type(card.edition) == "table" and card.edition.negative then
            state.starting_params.joker_slots = state.starting_params.joker_slots + 1
        end
    end

    state.pack.choices_remaining = math.max(0, state.pack.choices_remaining - 1)
    if state.pack.choices_remaining == 0 then
        state.pack = nil
    end

    return state
end

function oracle.close_pack(state)
    G.GAME = state
    state.pack = nil
    return state
end

function oracle.defeat_blind(state)
    G.GAME = state
    local data = state.data

    -- Mark current blind as Defeated and advance to next
    for _, bt in ipairs({"Small", "Big", "Boss"}) do
        if state.round_resets.blind_states[bt] == "Current" then
            state.round_resets.blind_states[bt] = "Defeated"
            if bt == "Small" then
                state.round_resets.blind_states.Big = "Select"
                state.blind_on_deck = "Big"
            elseif bt == "Big" then
                state.round_resets.blind_states.Boss = "Select"
                state.blind_on_deck = "Boss"
            end
            break
        end
    end

    -- Then do cash_out
    return oracle.cash_out(state)
end

function oracle.add_consumable(state, center_key)
    G.GAME = state
    local center = state.data.centers[center_key]
    state.consumables[#state.consumables + 1] = {
        center_key = center_key,
        edition = nil,
        sell_cost = math.max(1, math.floor((center.cost or 0) / 2)),
    }
    return state
end

function oracle.add_joker(state, center_key)
    G.GAME = state
    local center = state.data.centers[center_key]
    local config = center.config or {}
    local extra = config.extra
    if type(extra) == "table" then
        local copy = {}
        for k, v in pairs(extra) do copy[k] = v end
        extra = copy
    end
    local joker = {
        center_key = center_key,
        mult = tonumber(config.mult) or 0,
        x_mult = tonumber(config.Xmult) or 1,
        t_mult = tonumber(config.t_mult) or 0,
        t_chips = tonumber(config.t_chips) or 0,
        extra = extra,
        type = config.type or "",
        h_size = tonumber(config.h_size) or 0,
        d_size = tonumber(config.d_size) or 0,
        sell_cost = math.max(1, math.floor((center.cost or 0) / 2)),
    }
    state.jokers[#state.jokers + 1] = joker
    if not state.joker_keys then state.joker_keys = {} end
    state.joker_keys[#state.joker_keys + 1] = center_key
    -- Apply stat modifiers
    if joker.h_size ~= 0 then
        state.starting_params.hand_size = state.starting_params.hand_size + joker.h_size
        state.current_round.hand_size = state.current_round.hand_size + joker.h_size
    end
    if joker.d_size > 0 then
        state.round_resets.discards = state.round_resets.discards + joker.d_size
        state.current_round.discards_left = state.current_round.discards_left + joker.d_size
    end
    return state
end

return oracle
