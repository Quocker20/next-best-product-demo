"""Vercel serverless entry of /api/user."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import Handler  # noqa: E402


class handler(Handler):  # Vercel looks for a BaseHTTPRequestHandler subclass named `handler`
    ROUTE = "user"
