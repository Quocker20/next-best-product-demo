"""Web server of the demo: serves index.html and a small JSON API (standard library only).

    python server.py [--port 8765]        # PORT env var is used when --port is absent (hosting platforms)

GET  /              index.html
GET  /api/meta      form fields, model label, sample users
GET  /healthz       liveness probe
POST /api/user      {user_id}                  -> test bookings of a sample user (search fields + actual cluster)
POST /api/recommend {user_id, index, fields, truth} -> top-5 of the main model, ItemKNN, Popularity
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from nbp_demo.engine import Engine

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "index.html"
LOCK = threading.Lock()
_ENGINE: Engine | None = None


def engine() -> Engine:
    """Loaded on first use (also by the serverless entries in `api/`)."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


class Handler(BaseHTTPRequestHandler):
    ROUTE: str | None = None  # set by the serverless entries in api/ (one file per endpoint)

    def _route(self) -> str:
        if self.ROUTE:
            return self.ROUTE
        path = self.path.split("?")[0]
        if "/api/" not in path:
            return path
        return path[path.index("/api/") + 5:].rstrip("/").removesuffix(".json")  # api/meta.json == api/meta

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self._route()
        if path == "meta":
            self._send(200, json.dumps(engine().meta_json()).encode(), "application/json")
        elif path == "/healthz":
            self._send(200, b"ok", "text/plain")
        elif path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fn = {"user": engine().api_user, "recommend": engine().api_recommend}[self._route()]
            with LOCK:
                out = fn(req)
        except Exception as e:  # shown in the page
            out = {"error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out).encode(), "application/json")

    def log_message(self, *a) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    a = ap.parse_args()
    engine()
    print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
