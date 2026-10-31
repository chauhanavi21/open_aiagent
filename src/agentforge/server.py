"""Local HTTP API. Loopback only, bearer-token auth, bounded bodies, generic errors."""
from __future__ import annotations

import hmac
import json
import logging
import secrets
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Runtime
from .company import Company
from .webui import APP_JS, INDEX_HTML

MAX_BODY = 64 * 1024
log = logging.getLogger("agentforge.server")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {"level": record.levelname, "msg": record.getMessage()}
        data.update({k: v for k, v in record.__dict__.items() if k in ("method", "path", "status", "brain", "error")})
        return json.dumps(data)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)


def make_server(runtime: Runtime, token: str) -> ThreadingHTTPServer:
    company = Company(runtime.agent, runtime.workspace, runtime.memory)
    token_bytes = token.encode()

    class Handler(BaseHTTPRequestHandler):
        timeout = 30
        server_version = "AgentForge"
        sys_version = ""

        def log_message(self, *args) -> None:  # silence default access log
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
            log.info("request", extra={"method": self.command, "path": self.path, "status": status})

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            return header.startswith("Bearer ") and hmac.compare_digest(header[7:].encode(), token_bytes)

        def _read_json(self) -> dict | None:
            if "application/json" not in self.headers.get("Content-Type", ""):
                return None
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return None
            if not 0 < length <= MAX_BODY:
                return None
            try:
                data = json.loads(self.rfile.read(length))
            except ValueError:
                return None
            return data if isinstance(data, dict) else None

        def _send_static(self, content_type: str, text: str) -> None:
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'self'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok"})
            elif self.path == "/":
                self._send_static("text/html", INDEX_HTML)
            elif self.path == "/app.js":
                self._send_static("text/javascript", APP_JS)
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path not in ("/agent", "/company"):
                return self._send(404, {"error": "not found"})
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            body = self._read_json()
            if body is None:
                return self._send(400, {"error": "invalid request"})
            try:
                if self.path == "/agent":
                    goal = body.get("goal")
                    if not isinstance(goal, str):
                        return self._send(400, {"error": "invalid request"})
                    res = runtime.agent.run(goal)
                    return self._send(200, {"answer": res.answer, "brain": res.brain, "degraded": res.degraded,
                                            "steps": [{"tool": s.tool, "ok": s.ok} for s in res.steps]})
                mission, name = body.get("mission"), body.get("name", "")
                if not isinstance(mission, str) or not isinstance(name, str):
                    return self._send(400, {"error": "invalid request"})
                run = company.run(mission, name)
                self._send(200, {"name": run.name, "folder": run.folder, "report": run.report,
                                 "files": [o.file for o in run.outcomes]})
            except ValueError:
                self._send(400, {"error": "invalid request"})
            except Exception:  # noqa: BLE001 - never leak internals to clients
                log.exception("unhandled error")
                self._send(500, {"error": "internal error"})

    return ThreadingHTTPServer((runtime.config.host, runtime.config.port), Handler)


def new_token() -> str:
    return secrets.token_urlsafe(32)
