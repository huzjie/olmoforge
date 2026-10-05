"""Zero-dependency stdlib HTTP server.

Exposes /health and /status so the framework can be probed in a deployment
without Flask/FastAPI. The /status endpoint reports the current config and a
throughput snapshot, which is handy for operator health checks.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._send(200, {"status": "ok", "service": "olmoforge"})
        elif self.path == "/status":
            self._send(200, {"service": "olmoforge", "mode": "training",
                             "experts": self.server.n_experts,
                             "top_k": self.server.top_k})
        else:
            self._send(404, {"error": "not found"})


def create_server(n_experts=384, top_k=6, host="127.0.0.1", port=8899):
    srv = HTTPServer((host, port), _Handler)
    srv.n_experts = n_experts
    srv.top_k = top_k
    return srv


def run_server(n_experts=384, top_k=6, host="127.0.0.1", port=8899):
    srv = create_server(n_experts, top_k, host, port)
    print(f"[serve] olmoforge status server on http://{host}:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
