"""Tiny mock of the ArmInferX /generate endpoints for frontend E2E testing.

Serves on port 8000:
- POST /generate: canned response matching the real API contract, echoing the
  request's engine_id (defaults to llamacpp-optimized) plus token metadata.
- POST /generate/stream: Server-Sent Events — token deltas then a done event
  with the same metadata (mirrors the real streaming endpoint).
- GET /health, /benchmarks/latest, /benchmarks/summary.

Honors CORS so the Vite dev server (localhost:3000) can call it. No model or
ML dependencies required.

Usage:
    python tests/mock_generate_server.py
"""

import json
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

HOST = "127.0.0.1"
PORT = 8000

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}

# Latest simulated benchmark record, refreshed on every /generate call so the
# frontend's /benchmarks/latest refresh flow can be exercised end-to-end.
# NOTE: single-user state; two concurrent /generate calls could interleave the
# response and the record, which is fine for a frontend test double.
_last_record = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003 - keep it quiet, print it
        print("[mock] " + fmt % args)

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _begin_sse(self) -> None:
        """Send the SSE status line and headers exactly once, then bodies."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()

    def _write_sse(self, data: dict) -> None:
        """Write one JSON Server-Sent Event body (headers already sent)."""
        self.wfile.write(b"data: " + json.dumps(data).encode() + b"\n\n")
        self.wfile.flush()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        for key, value in CORS_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"status": "healthy", "service": "mock"})
        elif path == "/benchmarks/latest":
            if _last_record is None:
                self._send(404, {"detail": "No benchmark records found"})
            else:
                self._send(200, _last_record)
        elif path == "/benchmarks/summary":
            total = 1 if _last_record is not None else 0
            r = _last_record or {}
            self._send(
                200,
                {
                    "avg_latency_ms": round(r.get("latency_ms", 0.0), 2),
                    "avg_memory_mb": round(r.get("memory_mb", 0.0), 2),
                    "avg_cpu_percent": round(r.get("cpu_percent", 0.0), 2),
                    "total_runs": total,
                },
            )
        else:
            self._send(404, {"detail": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/generate", "/generate/stream"):
            self._send(404, {"detail": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw or b"{}")
            prompt = str(payload.get("prompt", "")).strip()
        except json.JSONDecodeError:
            self._send(400, {"detail": "invalid JSON body"})
            return
        if not prompt:
            self._send(400, {"detail": "prompt must be a non-empty string"})
            return

        engine_id = str(payload.get("engine_id") or "llamacpp-optimized")
        runtime = "llama.cpp"

        # Simulate inference so the loading state is visible, and report the
        # simulated latency the same way the real backend reports it.
        start = time.perf_counter()
        time.sleep(0.6)
        latency_ms = (time.perf_counter() - start) * 1000
        # Mirror the real backend: every inference also saves a benchmark
        # record that /benchmarks/latest then returns, tagged with the engine.
        # There is no tokenizer in this test double, so the token count is
        # simulated deterministically from the response text.
        response = f"Hello from the mock model! You asked: {prompt[:60]}"
        generated_tokens = len(response.split())
        global _last_record
        _last_record = {
            "prompt": prompt,
            "model": "mock-qwen2.5-3b",
            "response": response,
            "latency_ms": round(latency_ms, 2),
            # Simulated TTFT: a fraction of the total latency (real backend
            # measures it via a streamer as prefill + first decode step).
            "ttft_ms": round(latency_ms * 0.35, 2),
            "memory_mb": 512.42,
            "cpu_percent": 38.7,
            "generated_tokens": generated_tokens,
            "tokens_per_second": round(generated_tokens / (latency_ms / 1000), 2),
            "timestamp": _utc_now_iso(),
            "engine_id": engine_id,
            "runtime": runtime,
        }

        metadata = {
            "status": "success",
            "model": "mock-qwen2.5-3b",
            "response": _last_record["response"],
            "latency_ms": _last_record["latency_ms"],
            "engine_id": engine_id,
            "runtime": runtime,
            "generated_tokens": generated_tokens,
            "tokens_per_second": _last_record["tokens_per_second"],
            "ttft_ms": _last_record["ttft_ms"],
        }

        if path == "/generate/stream":
            self._begin_sse()
            words = response.split()
            for i, word in enumerate(words):
                self._write_sse(
                    {
                        "text": word + (" " if i < len(words) - 1 else ""),
                        "is_first": i == 0,
                        "is_last": i == len(words) - 1,
                    }
                )
                time.sleep(0.05)
            done = {**metadata, "done": True}
            self._write_sse({k: v for k, v in done.items() if k != "status"})
            return

        self._send(200, metadata)


if __name__ == "__main__":
    print(f"[mock] listening on http://{HOST}:{PORT}")
    HTTPServer((HOST, PORT), Handler).serve_forever()
