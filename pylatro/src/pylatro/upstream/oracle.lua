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
        table.sort(keys, function(a, b) return tostring(a.k) < tostring(b.k) end)
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
        consumables = {},
        used_vouchers = {},
        banned_keys = {},
        pool_flags = {},
        tags = {},
        probabilities = { normal = 1 },
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

    -- Select boss blind (uses RNG like Python does)
    local boss_key = "bl_hook"  -- placeholder
    if data.blinds then
        local boss_pool = {}
        for k, v in pairs(data.blinds) do
            if v.boss then boss_pool[k] = v end
        end
        local _, key = pseudorandom_element(boss_pool, pseudoseed("boss", state.pseudorandom))
        if key then boss_key = key end
    end
    state.round_resets.blind_choices.Boss = boss_key

    -- Advance RNG for voucher and tags (must match Python's create_run_state)
    if data.centers then
        local voucher_pool = {}
        for k, v in pairs(data.centers) do
            if v.set == "Voucher" and not v.requires then
                voucher_pool[k] = v
            end
        end
        if next(voucher_pool) then
            pseudorandom_element(voucher_pool, pseudoseed("Voucher", state.pseudorandom))
        end
    end

    if data.centers then
        local tag_pool = {}
        for k, v in pairs(data.centers) do
            if v.set == "Tag" then
                tag_pool[k] = v
            end
        end
        if next(tag_pool) then
            pseudorandom_element(tag_pool, pseudoseed("Tag", state.pseudorandom))
            pseudorandom_element(tag_pool, pseudoseed("Tag", state.pseudorandom))
        end
    end

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

    -- Reset card flags
    for _, card in ipairs(state.deck_cards) do
        card.discarded = false
        card.forced_selection = false
        card.face_down = false
        card.debuff = false  -- no boss debuff logic for simple case
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

    -- Move selected cards from hand_cards to play_cards (1-indexed)
    -- Collect cards first, then remove (in reverse order to preserve indices)
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

    -- Update card tracking and move to play_cards
    for _, card in ipairs(selected) do
        card.times_played = card.times_played + 1
        card.played_this_ante = true
        card.discarded = false
        card.forced_selection = false
        state.play_cards[#state.play_cards + 1] = card
    end

    -- Decrement hands_left, increment counters
    state.current_round.hands_left = math.max(0, state.current_round.hands_left - 1)

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

return oracle
