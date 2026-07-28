"""Loopback HTTP server for the Steamodded bridge."""

from __future__ import annotations

import json
import logging
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any, cast

from .protocol import PROTOCOL_VERSION, DecisionRequest, DecisionResponse, ProtocolError

if TYPE_CHECKING:
    from .policy import LivePolicyRunner

logger = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 4 * 1024 * 1024


class DecisionService:
    """Serializes inference, attaches to one run, and caches duplicates."""

    def __init__(self, runner: LivePolicyRunner) -> None:
        self.runner = runner
        self.active_session: str | None = None
        self.last_decision_id = -1
        self.responses: dict[tuple[str, int, str, str], DecisionResponse] = {}
        self.decision_fingerprints: dict[tuple[str, int], tuple[str, str]] = {}
        self.terminal = False
        self._lock = threading.Lock()

    def handle(self, request: DecisionRequest) -> DecisionResponse:
        started = time.perf_counter()
        with self._lock:
            cached = self.responses.get(request.identity)
            if cached is not None:
                logger.debug("Replayed cached decision %d", request.decision_id)
                return cached

            if self.active_session is None:
                self.active_session = request.session_id
                logger.info("Attached to live Balatro session %s", request.session_id)
            elif request.session_id != self.active_session:
                return DecisionResponse.error_response(
                    request,
                    "session_busy",
                    f"server is already attached to session {self.active_session}",
                )

            old_identity = self.decision_fingerprints.get((request.session_id, request.decision_id))
            if old_identity is not None:
                return DecisionResponse.error_response(
                    request,
                    "decision_conflict",
                    "decision id was already used with a different phase or fingerprint",
                )
            if request.decision_id < self.last_decision_id:
                return DecisionResponse.error_response(
                    request, "stale_request", "decision id is older than the latest processed decision"
                )

            self.decision_fingerprints[(request.session_id, request.decision_id)] = (
                request.phase,
                request.state_fingerprint,
            )
            self.last_decision_id = max(self.last_decision_id, request.decision_id)

            previous = request.previous_action
            if previous and not bool(previous.get("ok", True)):
                logger.warning(
                    "Balatro rejected previous action: %s",
                    previous.get("error") or previous.get("message") or "unknown error",
                )

            if request.phase == "terminal":
                state = request.state
                outcome = "win" if state.get("won") else "game over"
                logger.info("Live run ended: %s", outcome)
                response = DecisionResponse.wait_response(request)
                self.terminal = True
            else:
                try:
                    action, selection = self.runner.decide(request)
                    response = DecisionResponse.action_response(request, action)
                    logger.info(
                        "decision=%d phase=%s action=%s value=%s win_p=%s latency_ms=%.1f",
                        request.decision_id,
                        request.phase,
                        action.get("type"),
                        (f"{selection.expected_score:.3f}" if selection.expected_score is not None else "-"),
                        (f"{selection.win_probability:.3f}" if selection.win_probability is not None else "-"),
                        (time.perf_counter() - started) * 1000,
                    )
                except Exception as exc:
                    logger.exception("Live inference rejected decision %d", request.decision_id)
                    response = DecisionResponse.error_response(request, "decision_error", str(exc))
            self.responses[request.identity] = response
            return response


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        server = cast("LiveHTTPServer", self.server)
        if self.path != "/health":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._send(
            HTTPStatus.OK,
            {
                "ok": True,
                "protocol_version": PROTOCOL_VERSION,
                "attached": server.service.active_session is not None,
            },
        )

    def do_POST(self) -> None:
        server = cast("LiveHTTPServer", self.server)
        if self.path != "/v1/decision":
            self._send(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._send(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request_too_large"})
            return
        try:
            raw = json.loads(self.rfile.read(length))
            request = DecisionRequest.from_dict(raw)
            response = server.service.handle(request)
        except (json.JSONDecodeError, UnicodeDecodeError, ProtocolError) as exc:
            self._send(
                HTTPStatus.BAD_REQUEST,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "error": {"code": "invalid_request", "message": str(exc)},
                },
            )
            return
        self._send(HTTPStatus.OK, response.to_dict())

    def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("HTTP " + format, *args)


class LiveHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: DecisionService):
        self.service = service
        super().__init__(address, _Handler)


def serve_live(runner: LivePolicyRunner, host: str = "127.0.0.1", port: int = 43137) -> None:
    if host == "localhost":
        host = "127.0.0.1"
    if host != "127.0.0.1":
        raise ValueError("live mode only binds to 127.0.0.1")
    service = DecisionService(runner)
    server = LiveHTTPServer((host, port), service)
    server.timeout = 0.5
    logger.info("Waiting for Balatro at http://%s:%d/v1/decision", host, port)
    try:
        while not service.terminal:
            server.handle_request()
    except KeyboardInterrupt:
        logger.info("Live server interrupted")
    finally:
        server.server_close()
