"""RunPod serverless handler for Raster2Seq floor plan inference.

RunPod calls `handler(job)` for each request and expects the returned
dict as the result. The client side
(`floorplan_to_3d.r2s_parser.cloud.infer_live`) posts a base64-encoded
image + a checkpoint alias; we run `predict_one.py` and return the
polygon JSON as the result.

Input shape (job["input"]):
    {
        "image_b64": "<base64-encoded PNG bytes>",
        "checkpoint": "cubicasa5k" | "s3d-bw" | "raster2graph" | ...,
    }

Output shape (returned dict):
    On success:  {"polygons": [...], "openings": [...], "checkpoint": "...", "image_sha256": "..."}
    On failure:  {"error": "..."}  -- RunPod marks the job FAILED if we raise

This module is import-safe without `runpod` installed, so it can be
smoke-tested locally with `python handler.py` (which reads from stdin).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

try:
    import runpod  # type: ignore[import-not-found]
except ImportError:
    runpod = None  # allows local smoke-testing without runpod SDK

WORKER_DIR = Path(__file__).parent
PREDICT_SCRIPT = WORKER_DIR / "predict_one.py"

# Mirrors r2s_parser/config.py:CheckpointAlias. Keep in sync.
ALLOWED_CHECKPOINTS = {
    "cubicasa5k",
    "s3d-bw",
    "raster2graph",
    "raster2graph-512",
    "s3d-density",
}


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
        timeout=180,  # 3 min hard cap (cold start can be 30-90s)
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"predict_one.py exited {proc.returncode}: {proc.stderr[-2000:]}"
        )
    if not output_path.exists():
        raise RuntimeError(f"predict_one.py wrote no output. stderr: {proc.stderr[-500:]}")
    return json.loads(output_path.read_text())


def handler(job: dict) -> dict:
    """RunPod handler entrypoint."""
    try:
        job_input = job.get("input", {})
        image_b64 = job_input.get("image_b64")
        checkpoint = job_input.get("checkpoint", "cubicasa5k")

        if not image_b64:
            return {"error": "missing 'image_b64' in job input"}
        if checkpoint not in ALLOWED_CHECKPOINTS:
            return {"error": f"invalid checkpoint {checkpoint!r}; allowed: {sorted(ALLOWED_CHECKPOINTS)}"}

        # Decode the image, write to a tmp file, run inference, return the
        # result dict. RunPod is billed per request so we don't keep any
        # state on disk between calls.
        image_bytes = base64.b64decode(image_b64)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img_path = tmp / "input.png"
            img_path.write_bytes(image_bytes)
            out_path = tmp / "pred.json"
            result = _run_predict(img_path, checkpoint, out_path)
        return result
    except subprocess.TimeoutExpired:
        return {"error": "inference timed out (>180s)"}
    except Exception as exc:  # noqa: BLE001 — RunPod wants any failure surfaced
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }


def _local_smoke_test() -> None:
    """Read a base64 image from stdin and print the handler's output.

    Usage:
        echo '{"image_b64": "...", "checkpoint": "cubicasa5k"}' | python handler.py
    """
    payload = json.loads(sys.stdin.read())
    fake_job = {"id": "local-smoke", "input": payload}
    result = handler(fake_job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if runpod is not None and os.environ.get("RUNPOD_LOCAL") != "1":
        runpod.serverless.start({"handler": handler})
    else:
        _local_smoke_test()
