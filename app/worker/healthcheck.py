"""
app/worker/healthcheck.py
──────────────────────────
HF Spaces (Docker SDK) health-checks whatever's listening on port 7860,
even for a Space that's really just a background worker with no web UI
of its own. This is a deliberately trivial stdlib-only HTTP server (no
FastAPI, no extra imports) that answers 200 on every path so the worker
Space registers as healthy without paying for a second full app import.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"celery worker alive")

    def log_message(self, format, *args):
        pass  # keep the container logs focused on Celery's own output


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 7860), _HealthHandler).serve_forever()
