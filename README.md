# vibe3d-r2s-worker

A [RunPod Serverless](https://docs.runpod.io/serverless/workers/overview)
worker that runs [Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq)
(Cornell, MIT) floor-plan inference. The PoC's `r2s_parser/cloud.py`
posts base64-encoded PNG + a checkpoint alias; this worker returns a
polygon-sequence JSON that the local `r2s_parser/postprocess.py` turns
into a `FloorPlan`.

## Files

- `Dockerfile` — multi-stage build. Stage 1 pulls the runpod/pytorch
  base image; stage 2 clones Raster2Seq, builds the two custom CUDA
  ops (`models/ops` and `diff_ras`), and pre-downloads the
  `cubicasa5k` checkpoint from Hugging Face. Stage 3 strips the
  C++/CUDA toolchain and ships a minimal runtime image.
- `handler.py` — RunPod Serverless handler entrypoint. Decodes the
  base64 image, shells out to `predict_one.py`, returns the result
  dict. Used for serverless endpoints.
- `serve.py` — plain HTTP server (stdlib `http.server`) for RunPod
  Pods. Exposes port 8000; accepts the same `POST /runsync` JSON
  contract as the serverless handler. **Use this for Pods.**
- `predict_one.py` — single-image Raster2Seq inference script. Uses
  the same flags as `tools/predict_cc5k.sh` but takes one image at a
  time. Writes a `pred.json` with the polygon-sequence schema.
- `requirements.txt` — handler-only deps (just `runpod`).

## Deploy to RunPod Serverless (GitHub integration)

1. **Connect GitHub**: in the RunPod console, go to **Settings →
   Connections → GitHub → Connect**. Authorize the OAuth. (Required
   even for public repos.)
2. **Create endpoint**: **Serverless → New Endpoint → Import Git
   Repository → vibe3d-r2s-worker** (or whatever you named it).
3. **Configure**:
   - Branch: `main`
   - Dockerfile Path: `Dockerfile`
   - Endpoint Type: **Queue** (per-second billing; we want this)
   - GPU Configuration: **A10G** or **L4** (24 GB VRAM is plenty for
     the 4 GB Raster2Seq checkpoint; L4 is cheaper at idle)
   - Active Workers: 0 (scaled-to-zero)
   - Max Workers: 1 (raise to 2 if you want to bake the 8-sample
     cache in parallel)
4. **Deploy Endpoint**. The first build takes 5–15 min. Subsequent
   updates push via GitHub releases.

## Deploy to a RunPod Pod (recommended for debugging)

If the serverless endpoint is unhealthy or stuck, a Pod gives you a
real Linux box with GPU and direct terminal access — no gateway
mystery.  Use `serve.py` instead of `handler.py`.

1. **Push a tagged image to Docker Hub** (or any registry RunPod can
   pull from):
   ```bash
   docker build -t yourdockerhub/vibe3d-r2s-worker:latest .
   docker push yourdockerhub/vibe3d-r2s-worker:latest
   ```
2. **Create the Pod** in the RunPod console: **Pods → New Pod →
   Custom Image → `yourdockerhub/vibe3d-r2s-worker:latest`**.
3. **Configure**:
   - GPU: any of the enabled types (RTX A4000 / A4500 / 4000 Ada /
     2000 Ada — all work)
   - Container Disk: 20 GB (image is ~5 GB after pull + 4 GB checkpoint)
   - **Expose HTTP Port: 8000** (RunPod Pods auto-port-forward 8000-9000)
4. **Override the entrypoint** in the Pod's "Advanced" / "Docker
   Command" field:
   ```
   python -u serve.py
   ```
   (This replaces the default `python -u handler.py`.)
5. **Click Deploy**. After ~2-3 min (image pull) + ~30-60s (cold
   start, which compiles nothing on Pods), the Pod is running.
6. **Find the public URL** in the Pod's "Connect" panel — it'll be
   `https://<pod-id>-8000.proxy.runpod.net`.  Set
   `R2S_CLOUD_ENDPOINT` to that URL (no `/v2/<id>` suffix) and the
   client-side `cloud.py` will auto-detect the Pod path and use the
   right JSON contract.

7. **Tail logs** with the Pod's "Logs" tab to see the `[startup]`
   self-test output and any `[http]` request lines.

**Cost**: ~$0.40/hr for an A4000.  Stop the Pod when not in use.

## Cold-start latency

The first request after a scale-to-zero event is **30–90 s** (loads
the ~4 GB checkpoint + warms up the two custom CUDA ops). Subsequent
requests on a warm worker are **5–15 s**. For the Week 1 8-sample
cache bake, the cold start dominates the cost — actual inference is
~5 s per image.

## Local smoke test (no GPU)

Without a CUDA build, `predict_one.py` will fail at `engine.generate`
with a missing-MSDeformAttn error. To verify the handler plumbing
without the model, you can import the module and inspect `handler`:

```bash
python -c "from handler import handler, ALLOWED_CHECKPOINTS; print(ALLOWED_CHECKPOINTS)"
```

The full end-to-end test only works on the deployed RunPod endpoint.

## Endpoint contract

POST `<endpoint>/runsync` (or `/run` for async) with:

```json
{
  "input": {
    "image_b64": "<base64 PNG bytes>",
    "checkpoint": "cubicasa5k"
  }
}
```

Response (200 OK):

```json
{
  "polygons": [
    {"label": "kitchen", "vertices": [[0.1, 0.2], ...]},
    ...
  ],
  "openings": [],
  "checkpoint": "cubicasa5k",
  "image_sha256": "abc123..."
}
```

This is the same schema as `floorplan_to_3d.r2s_parser.cache.PolygonSeq`.

## What this does NOT include

- **Openings (doors/windows) detection.** Raster2Seq v1.0 encodes these
  as polygon edges with semantic class = door/window, but `engine.generate`
  doesn't separate them from room polygons. The client-side postprocess
  currently only emits `polygons`; openings are detected downstream by
  the v3 CV parser as a fallback. Week 2 of v4 will wire proper
  opening extraction.
- **Refinement (test-time).** `--refinement` is supported upstream but
  not enabled here. Adds ~3× inference time for ~5% accuracy.
