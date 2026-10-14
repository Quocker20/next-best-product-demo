"""Web server of the demo: serves index.html and a small JSON API (standard library only).

    python server.py [--port 8765]        # PORT env var is used when --port is absent (hosting platforms)

GET  /              index.html
GET  /api/meta      form fields, model label, sample users
GET  /healthz       liveness probe
POST /api/user      {user_id}                  -> test bookings of a sample user + fixed top-20 list
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
ENGINE: Engine


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/meta":
            self._send(200, json.dumps(ENGINE.meta_json()).encode(), "application/json")
        elif self.path == "/healthz":
            self._send(200, b"ok", "text/plain")
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fn = {"/api/user": ENGINE.api_user, "/api/recommend": ENGINE.api_recommend}[self.path]
            with LOCK:
                out = fn(req)
        except Exception as e:  # shown in the page
            out = {"error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out).encode(), "application/json")

    def log_message(self, *a) -> None:  # quiet
        pass


def main() -> None:
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    a = ap.parse_args()
    ENGINE = Engine()
    print(f"ready: http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
