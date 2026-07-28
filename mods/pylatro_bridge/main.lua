--- STEAMODDED HEADER
--- MOD_NAME: Pylatro Live Bridge
--- MOD_ID: pylatro_bridge
--- MOD_AUTHOR: [makemake]
--- MOD_DESCRIPTION: Loopback bridge to the Pylatro live agent.
--- PREFIX: pylatro_bridge
--- VERSION: 0.1.7
--- DEPENDENCIES: [Steamodded>=1.0.0~BETA]

local mod = SMODS.current_mod
local https = require "SMODS.https"
local json = require "json"
local Serializer = assert(SMODS.load_file("serializer.lua"))()
local Readiness = assert(SMODS.load_file("readiness.lua"))()

PYLATRO_BRIDGE = {
    serializer = Serializer,
    executor = nil,
    protocol_version = 1,
    endpoint = "http://127.0.0.1:43137/v1/decision",
    session_id = tostring(os.time()) .. "-" .. tostring(math.random(100000, 999999)),
    decision_id = 0,
    in_flight = false,
    last_fingerprint = nil,
    pending_request = nil,
    next_attempt_at = 0,
    failures = 0,
    previous_action = nil,
    connected = false,
    cash_out_started = false,
    action_rejections = 0,
    hand_signature = nil,
    hand_stable_since = nil,
    readiness_reason = nil,
    readiness_log_at = 0,
}
PYLATRO_BRIDGE.executor = assert(SMODS.load_file("executor.lua"))()
local Bridge = PYLATRO_BRIDGE

local function log(level, message)
    local line = "[Pylatro Bridge] " .. message
    if level == "error" and sendErrorMessage then
        sendErrorMessage(message, "PylatroBridge")
    elseif level == "warn" and sendWarnMessage then
        sendWarnMessage(message, "PylatroBridge")
    elseif sendInfoMessage then
        sendInfoMessage(message, "PylatroBridge")
    else
        print(line)
    end
end

local function phase()
    if not G or not G.GAME or not G.STATES then return nil end
    if G.STATE == G.STATES.BLIND_SELECT then return "blind_select" end
    if G.STATE == G.STATES.SELECTING_HAND then return "hand_play" end
    if G.STATE == G.STATES.SHOP then return "shop" end
    if G.STATE == G.STATES.TAROT_PACK
        or G.STATE == G.STATES.PLANET_PACK
        or G.STATE == G.STATES.SPECTRAL_PACK
        or G.STATE == G.STATES.STANDARD_PACK
        or G.STATE == G.STATES.BUFFOON_PACK then
        return "booster_pack"
    end
    if G.STATE == G.STATES.GAME_OVER
        or (G.STATES.GAME_WON ~= nil and G.STATE == G.STATES.GAME_WON) then
        return "terminal"
    end
    return nil
end

local function versions()
    return {
        balatro = tostring(G.VERSION or "unknown"),
        steamodded = tostring(SMODS.version or "unknown"),
        bridge = tostring(mod.version or "0.1.7"),
    }
end

local function fresh_request(current_phase, fingerprint)
    Bridge.decision_id = Bridge.decision_id + 1
    return {
        protocol_version = Bridge.protocol_version,
        session_id = Bridge.session_id,
        decision_id = Bridge.decision_id,
        state_fingerprint = fingerprint,
        phase = current_phase,
        versions = versions(),
        state = Serializer.snapshot(current_phase),
        legal = Serializer.legality(current_phase),
        previous_action = Bridge.previous_action,
    }
end

local function schedule_retry(message)
    Bridge.in_flight = false
    Bridge.failures = math.min(Bridge.failures + 1, 8)
    local delay = math.min(0.25 * (2 ^ (Bridge.failures - 1)), 8)
    Bridge.next_attempt_at = love.timer.getTime() + delay
    if Bridge.connected or Bridge.failures == 1 or Bridge.failures == 4 then
        log("warn", message .. "; retrying in " .. tostring(delay) .. "s")
    end
    Bridge.connected = false
end

local function identity_matches(response, request)
    return response.protocol_version == request.protocol_version
        and response.session_id == request.session_id
        and response.decision_id == request.decision_id
        and response.phase == request.phase
        and response.state_fingerprint == request.state_fingerprint
end

local function handle_response(code, body, _, request)
    Bridge.in_flight = false
    if code ~= 200 then
        schedule_retry("server unavailable (HTTP " .. tostring(code) .. ")")
        return
    end
    local ok, response = pcall(json.decode, body or "")
    if not ok or type(response) ~= "table" then
        schedule_retry("invalid JSON response")
        return
    end
    if not identity_matches(response, request) then
        log("warn", "discarded stale or mismatched response")
        Bridge.pending_request = nil
        Bridge.last_fingerprint = nil
        return
    end
    Bridge.pending_request = nil
    Bridge.failures = 0
    if not Bridge.connected then log("info", "connected to Python live agent") end
    Bridge.connected = true
    if response.wait then return end
    if response.error then
        log("error", "Python rejected decision: " .. tostring(response.error.message))
        return
    end
    if not response.action then
        log("error", "response contained no action")
        return
    end
    local current_phase = phase()
    local current_fingerprint = current_phase and Serializer.fingerprint(current_phase)
    if current_phase ~= request.phase or current_fingerprint ~= request.state_fingerprint then
        log("warn", "state changed during inference; action discarded")
        Bridge.last_fingerprint = nil
        return
    end
    local executed, action_error = Bridge.executor.execute(response.action)
    Bridge.previous_action = {
        decision_id = request.decision_id,
        ok = executed == true,
        error = executed and nil or tostring(action_error),
    }
    if not executed then
        log("warn", "action rejected: " .. tostring(action_error))
        Bridge.action_rejections = math.min(Bridge.action_rejections + 1, 5)
        local delay = math.min(0.25 * (2 ^ (Bridge.action_rejections - 1)), 2)
        Bridge.next_attempt_at = love.timer.getTime() + delay
        Bridge.last_fingerprint = nil
    else
        Bridge.action_rejections = 0
    end
end

local function send_request(request)
    Bridge.in_flight = true
    local ok, encoded = pcall(json.encode, request)
    if not ok then
        Bridge.in_flight = false
        Bridge.pending_request = nil
        log("error", "snapshot serialization failed: " .. tostring(encoded))
        return
    end
    https.asyncRequest(Bridge.endpoint, {
        method = "POST",
        headers = {["Content-Type"] = "application/json"},
        data = encoded,
    }, function(code, body, headers)
        handle_response(code, body, headers, request)
    end)
end

local function update_bridge()
    if Bridge.in_flight then return end
    local now = love.timer.getTime()
    local current_phase = phase()
    local ready, reason = Readiness.ready(current_phase, now, Bridge)
    if not ready then
        if current_phase and (reason ~= Bridge.readiness_reason or now >= Bridge.readiness_log_at) then
            log("info", "waiting in " .. current_phase .. ": " .. tostring(reason))
            Bridge.readiness_reason = reason
            Bridge.readiness_log_at = now + 5
        end
        return
    end
    Bridge.readiness_reason = nil
    if now < Bridge.next_attempt_at then return end
    if not current_phase then
        -- Cash-out is a non-strategic transition. It still goes through the
        -- enabled UI callback and waits for Balatro's own event queue.
        if G.STATE == G.STATES.ROUND_EVAL then
            if not Bridge.cash_out_started then
                local executed = Bridge.executor.cash_out()
                Bridge.cash_out_started = executed == true
            end
        else
            Bridge.cash_out_started = false
        end
        return
    end
    Bridge.cash_out_started = false
    local fingerprint = Serializer.fingerprint(current_phase)
    if Bridge.pending_request then
        if Bridge.pending_request.phase ~= current_phase
            or Bridge.pending_request.state_fingerprint ~= fingerprint then
            log("warn", "state changed while disconnected; old request discarded")
            Bridge.pending_request = nil
            Bridge.last_fingerprint = nil
        else
            send_request(Bridge.pending_request)
            return
        end
    end
    if fingerprint == Bridge.last_fingerprint then return end
    Bridge.last_fingerprint = fingerprint
    Bridge.previous_action = Bridge.previous_action
    Bridge.pending_request = fresh_request(current_phase, fingerprint)
    send_request(Bridge.pending_request)
end

local original_update = Game.update
function Game:update(dt)
    original_update(self, dt)
    local ok, err = pcall(update_bridge)
    if not ok then
        Bridge.in_flight = false
        Bridge.pending_request = nil
        Bridge.next_attempt_at = love.timer.getTime() + 2
        log("error", "bridge update failed without stopping Balatro: " .. tostring(err))
    end
end

log("info", "loaded; start Python with `uv run --extra agent python play.py --live --heuristic`")
