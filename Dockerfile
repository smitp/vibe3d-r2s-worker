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
WORKDIR /opt/build/Raster2Seq
RUN pip install --no-cache-dir \
        einops transformers huggingface_hub scipy shapely \
        opencv-python-headless pycocotools matplotlib timm \
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
WORKDIR /opt/build/Raster2Seq/diff_ras
RUN python setup.py build develop

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

# ─── Stage 3: runtime ─────────────────────────────────────────────────────
FROM base AS runtime

# Copy the built Raster2Seq + checkpoint cache + our handler. Excludes
# the dev toolchain (nvcc, .git, /opt/build cache).
COPY --from=builder /opt/build/Raster2Seq /opt/worker/vendor/Raster2Seq
COPY --from=builder /opt/hf_cache /opt/hf_cache
COPY --from=builder /opt/worker /opt/worker

# Make `vendor.Raster2Seq.*` importable from /opt/worker.
ENV PYTHONPATH=/opt/worker:/opt/worker/vendor/Raster2Seq

WORKDIR /opt/worker

# RunPod's container-runtime contract: ENTRYPOINT calls the handler.
# The `runpod` Python SDK sets up the serverless loop; our handler.py
# starts the loop when invoked without RUNPOD_LOCAL=1.
ENTRYPOINT ["python", "-u", "handler.py"]
