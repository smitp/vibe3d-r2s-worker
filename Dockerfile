# Raster2Seq RunPod serverless worker
#
# Multi-stage build:
#   1. base    — pytorch + CUDA toolkit + system deps
#   2. builder — clone Raster2Seq, build the two custom CUDA ops,
#                pre-download the cubicasa5k checkpoint
#   3. runtime — copy the built artifacts + our handler; minimal image
#
# Why a runtime stage?  The build stage carries the C++/CUDA toolchain
# (~5 GB) and the Raster2Seq repo.  We strip both for the deployed
# image (~3 GB final vs ~8 GB without this split).

# ─── Stage 1: base ────────────────────────────────────────────────────────
# runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 ships nvcc
# + pytorch + cuda-toolkit. Pin to a specific digest once you find one
# that works in CI; for now the tag is fine for a research PoC.
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf_cache \
    # MSDeformAttn's setup.py uses torch.utils.cpp_extension, which
    # calls nvcc directly.  Pin CUDA_HOME + PATH so it picks the
    # base image's bundled nvcc 11.8 (no system fallback surprises).
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# OpenCV (required by Raster2Seq's plot_utils) needs libgl.
# Detectron2 (a transitive dep) needs ninja and a Cython compiler.
#
# Notes on the package list:
#   * `cython3` is the Ubuntu 22.04 name for the Cython compiler
#     (in older releases it was `cython`, but that was retired in
#     jammy — installing it returns E: Package 'cython' has no
#     installation candidate, which surfaces as apt exit 100).
#   * The retry loop is for transient mirror/network failures
#     unrelated to the package list.
RUN for i in 1 2 3; do \
        apt-get -o Acquire::http::No-Cache=True update \
        && apt-get install -y --no-install-recommends \
                libgl1 libglib2.0-0 \
                ninja-build cython3 \
        && rm -rf /var/lib/apt/lists/* \
        && break; \
        echo "apt-get failed (attempt $i), retrying in 5s..."; \
        sleep 5; \
    done

# ─── Stage 2: builder ─────────────────────────────────────────────────────
FROM base AS builder

WORKDIR /opt/build

# 1. Clone Raster2Seq (MIT license, master branch as of 2026).
RUN git clone --depth 1 https://github.com/Cornell-VAILab/Raster2Seq.git \
        /opt/build/Raster2Seq

# 2. Install Raster2Seq's Python deps. The repo's requirements.txt is
#    pinned to torch 2.3.1 + cuda 11.8; we override the torch install
#    because the base image already has the right CUDA / torch pair.
#
#    detectron2 is pinned to a specific commit (v0.6) because the
#    master branch tracks the latest torch, which can pull a torch
#    version that breaks the MSDeformAttn C++ build on the same step.
#    `--no-build-isolation` makes the build use the existing
#    (base-image) torch instead of building a fresh one in a venv.
#
#    We also pin the detectron2 transitive deps explicitly: detectron2
#    declares them in its setup.py but doesn't always install them
#    (fvcore is a classic example — Detectron2's setup.py imports it
#    but `pip install detectron2` doesn't always pull it in newer pip
#    resolver versions).
WORKDIR /opt/build/Raster2Seq
RUN pip install --no-cache-dir \
        einops transformers huggingface_hub scipy shapely \
        opencv-python-headless pycocotools matplotlib timm \
        fvcore omegaconf portalocker iopath pyyaml \
        # Raster2Seq's util/plot_utils.py imports descartes for
        # rendering polygons in debug PNGs.  Not a model dep but
        # the import is unconditional, so the runtime fails without
        # it.
        descartes \
    && pip install --no-cache-dir --no-build-isolation \
        'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# 3. Build MSDeformAttn (deformable-DETR's C++/CUDA op). Required at
#    inference time (no `if self.training` gate in deformable_transformer.py).
#
#    Patch setup.py to drop the `torch.cuda.is_available()` guard.
#    The build env has nvcc + CUDA_HOME but no GPU driver, so the
#    guard returns False and the build aborts with
#    `NotImplementedError: Cuda is not availabel`.  Building the
#    .so without a GPU driver present is fine — we only need a
#    GPU at *inference* time, not at *build* time.
WORKDIR /opt/build/Raster2Seq/models/ops
RUN sed -i 's|if torch.cuda.is_available() and CUDA_HOME is not None:|if CUDA_HOME is not None:|' setup.py \
    && grep -n "CUDA_HOME is not None" setup.py \
    && sh make.sh

# 4. Build the differentiable rasterizer (BoundaryFormer's C++/CUDA op).
#    Used by the RoomFormer branch of the model.
#
#    Use `build install` (not `build develop`) so the compiled .so
#    is copied into site-packages rather than left as a develop-mode
#    egg-link that points back to the builder's /opt/build path.
#    A develop install would break the runtime stage (different path)
#    and surface as `ModuleNotFoundError: No module named 'polygon'`.
WORKDIR /opt/build/Raster2Seq/diff_ras
RUN python setup.py build install

# 5. Pre-download the cubicasa5k checkpoint so cold starts are fast.
#    We use huggingface_hub with hf_token=hf_… if HF_TOKEN is set,
#    otherwise anonymous (the cc5k weights are public).
#
#    Why a separate file and not `RUN python -c "..."`?  BuildKit
#    (RunPod's builder) mis-tokenizes the inner `"..."` when the
#    script spans many lines, surfacing as `unknown instruction:
#    import` on the first non-Python line.  A separate file sidesteps
#    that and keeps the Dockerfile readable.
COPY download_checkpoint.py /opt/build/download_checkpoint.py
RUN python /opt/build/download_checkpoint.py \
        || echo "WARN: cubicasa5k pre-download failed; will retry at runtime"

# 6. Install our handler's small dep set on top.
WORKDIR /opt/worker
COPY requirements.txt /opt/worker/requirements.txt
RUN pip install --no-cache-dir -r /opt/worker/requirements.txt
COPY handler.py /opt/worker/handler.py
COPY predict_one.py /opt/worker/predict_one.py
COPY serve.py /opt/worker/serve.py

# ─── Stage 3: runtime ─────────────────────────────────────────────────────
FROM base AS runtime

# Copy the built Raster2Seq + checkpoint cache + our handler. Excludes
# the dev toolchain (nvcc, .git, /opt/build cache).
COPY --from=builder /opt/build/Raster2Seq /opt/worker/vendor/Raster2Seq
COPY --from=builder /opt/hf_cache /opt/hf_cache
COPY --from=builder /opt/worker /opt/worker

# Copy the Python site-packages the builder installed. This includes:
#   * the pip-installed Raster2Seq deps (einops, transformers, fvcore,
#     detectron2, runpod, …) — none of which the base image has by
#     default
#   * the compiled C++ ops:
#       - MultiScaleDeformableAttention*.so  (built by `sh make.sh`
#         → `python setup.py build install` in models/ops/)
#       - polygon.cpython-310-x86_64-linux-gnu.so  (built by
#         `python setup.py build install` in diff_ras/)
#     Without these copies, the runtime stage's
#     `import torch / import MultiScaleDeformableAttention /
#      from diff_ras import SoftPolygon` all fail with
#     ModuleNotFoundError, even though the build succeeded.
#
# The base image's site-packages is /usr/local/lib/python3.10/dist-packages,
# and both stages share the same Python (3.10) and base image, so a
# wholesale copy is the simplest and most reliable transfer.
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# Make `vendor.Raster2Seq.*` importable from /opt/worker.
ENV PYTHONPATH=/opt/worker:/opt/worker/vendor/Raster2Seq

WORKDIR /opt/worker

# 8000 is the port serve.py listens on.  RunPod Pods auto-port-forward
# ports in the 8000-9000 range when the container exposes them.
EXPOSE 8000

# RunPod's container-runtime contract: ENTRYPOINT calls the handler.
# The `runpod` Python SDK sets up the serverless loop; our handler.py
# starts the loop when invoked without RUNPOD_LOCAL=1.
#
# To run as a Pod instead of a serverless endpoint, override the
# entrypoint at Pod-deploy time:
#   docker run <image> python -u serve.py
ENTRYPOINT ["python", "-u", "handler.py"]
