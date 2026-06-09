"""HTTP server for the Raster2Seq RunPod Pod.

Why a plain HTTP server instead of the runpod SDK?
- RunPod Serverless workers go through the RunPod gateway, which
  we're currently stuck on (template / version state is broken
  and the worker exits with code 1 with no log visibility).
- A Pod is just a single GPU box.  We run a plain HTTP server
  inside the container, expose port 8000, and the Pod's
  built-in port-forwarding gives us a stable public URL.
- This bypasses the runpod SDK entirely.  The client-side
  `cloud.py` auto-detects Pod URLs by checking for a
  non-RunPod host, and uses the same JSON contract.

Contract (matches RunPod /runsync shape so the client doesn't
have to special-case it):

  POST /runsync
  Content-Type: application/json
  Body: {"image_b64": "<base64 PNG>", "checkpoint": "cubicasa5k"}

  200 OK
  Body: <polygon-sequence JSON — same schema as RunPod's `output` field>

  500 on inference error: {"error": "...", "traceback": "..."}
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WORKER_DIR = Path(__file__).parent
PREDICT_SCRIPT = WORKER_DIR / "predict_one.py"

# Mirrors handler.py:ALLOWED_CHECKPOINTS.
ALLOWED_CHECKPOINTS = {
    "cubicasa5k",
    "s3d-bw",
    "raster2graph",
    "raster2graph-512",
    "s3d-density",
}

# Startup self-test (mirrors handler.py:_startup_self_test).
def _startup_self_test() -> None:
    print("[startup] python:", sys.version.split()[0], flush=True)
    print(f"[startup] predict script: {PREDICT_SCRIPT}", flush=True)
    print(f"[startup] PYTHONPATH: {os.environ.get('PYTHONPATH', '<unset>')}", flush=True)
    print(f"[startup] CUDA_HOME: {os.environ.get('CUDA_HOME', '<unset>')}", flush=True)
    try:
        import torch  # noqa: F401
        print(f"[startup] torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}", flush=True)
    except Exception as exc:
        print(f"[startup] torch import FAILED: {exc}", file=sys.stderr, flush=True)
    try:
        from detectron2.data import transforms  # noqa: F401
        print("[startup] detectron2 imports OK", flush=True)
    except Exception as exc:
        print(f"[startup] detectron2 import FAILED: {exc}", file=sys.stderr, flush=True)
    try:
        import MultiScaleDeformableAttention  # noqa: F401
        print("[startup] MSDeformAttn import OK", flush=True)
    except Exception as exc:
        print(f"[startup] MSDeformAttn import FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    try:
        from diff_ras import SoftPolygon  # noqa: F401
        print("[startup] diff_ras import OK", flush=True)
    except Exception as exc:
        print(f"[startup] diff_ras import FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _run_predict(image_path: Path, checkpoint: str, output_path: Path) -> dict:
    """Run predict_one.py as a subprocess and return the parsed JSON."""
    proc = subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            str(image_path),
            checkpoint,
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min cap; cold start can be 60-90s on a Pod
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"predict_one.py exited {proc.returncode}: stderr={proc.stderr[-2000:]!r}"
        )
    if not output_path.exists():
        raise RuntimeError(f"predict_one.py wrote no output. stderr: {proc.stderr[-500:]}")
    return json.loads(output_path.read_text())


class Handler(BaseHTTPRequestHandler):
    """Threaded HTTP handler — one process, one thread per request."""

    def log_message(self, fmt: str, *args) -> None:
        # Send server logs to stderr so they're visible in `docker logs`.
        print(f"[http] {self.address_string()} - {fmt % args}", file=sys.stderr, flush=True)

    def do_GET(self) -> None:
        # Health check for the Pod's port-forwarder.
        if self.path == "/health":
            self._json(200, {"ok": True, "pid": os.getpid()})
        else:
            self._json(404, {"error": "not found; POST to /runsync"})

    def do_POST(self) -> None:
        if self.path != "/runsync":
            self._json(404, {"error": f"unknown route {self.path!r}; use POST /runsync"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                self._json(400, {"error": "empty body"})
                return
            body = self.rfile.read(length)
            payload = json.loads(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"invalid JSON: {exc}"})
            return

        # Accept both {"input": {...}} (RunPod-shaped) and flat {...}.
        if "input" in payload and isinstance(payload["input"], dict):
            payload = payload["input"]
        image_b64 = payload.get("image_b64")
        checkpoint = payload.get("checkpoint", "cubicasa5k")
        if not image_b64:
            self._json(400, {"error": "missing 'image_b64'"})
            return
        if checkpoint not in ALLOWED_CHECKPOINTS:
            self._json(400, {
                "error": f"invalid checkpoint {checkpoint!r}",
                "allowed": sorted(ALLOWED_CHECKPOINTS),
            })
            return

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception as exc:
            self._json(400, {"error": f"invalid base64: {exc}"})
            return

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                img_path = tmp / "input.png"
                img_path.write_bytes(image_bytes)
                out_path = tmp / "pred.json"
                result = _run_predict(img_path, checkpoint, out_path)
            self._json(200, result)
        except subprocess.TimeoutExpired:
            self._json(504, {"error": "inference timed out (>300s)"})
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[http] inference failed: {type(exc).__name__}: {exc}\n{tb}", file=sys.stderr, flush=True)
            self._json(500, {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb[-2000:],
            })

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    _startup_self_test()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"[http] listening on {host}:{port}", flush=True)
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[http] shutting down", flush=True)
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
