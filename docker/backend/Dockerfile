# =============================================================================
# ArmInferX — Backend Dockerfile (Linux, CPU-only; native ARM64 / aarch64)
# =============================================================================
#
# Builds the FastAPI backend together with the llama.cpp optimized engine
# (llama-cpp-python, compiled from source for aarch64 Linux).
#
# Multi-arch usage:
#   Native ARM64 host:          docker build -f docker/backend/Dockerfile -t arminferx-backend .
#   ARM64 image from x86 host:  docker buildx build --platform linux/arm64 \
#                                 -f docker/backend/Dockerfile -t arminferx-backend .
#
# Model mounting strategy
# -----------------------
# The ~2 GB Q4_K_M GGUF is intentionally NOT baked into the image (it is
# gitignored and would bloat the image). It is bind-mounted at runtime at the
# engine's *default* path, so no engine code or configuration changes:
#
#   -v "$PWD/models/gguf":/app/models/gguf:ro
#
# The backend is copied to /app/backend, so the code's PROJECT_ROOT resolves
# to /app and the engine default resolves to
# /app/models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf — exactly where the mount
# places it.
#
# Lifecycle: no model is loaded at startup (lazy EngineManager); the image
# boots with zero models and loads Q4_K_M on the first /generate request.
#
# Inference behavior is untouched: n_ctx=2048, n_threads=8, n_gpu_layers=0,
# greedy decoding, max_new_tokens=64 for the benchmark. No Arm64 performance
# claim is made here — these are build/run instructions only.
# =============================================================================

# -- Build stage: compilers + llama-cpp-python source build ------------------
# Python 3.11: best aarch64 wheel availability for the pinned API deps and a
# well-supported toolchain for llama-cpp-python 0.3.34's vendored ggml.
FROM python:3.11-slim-bookworm AS build

# Compilers/tools needed to build llama-cpp-python from source on aarch64
# (there is no prebuilt manylinux aarch64 wheel for the pinned 0.3.34).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Reproducible llama.cpp CPU build for ARM64:
#   GGML_NATIVE=OFF  -> portable binary (no -mcpu=native); the image runs on
#                       Graviton 1/2/3, Ampere Altra, Neoverse-N1/N2 and any
#                       other aarch64 core.
#   GGML_OPENMP=ON   -> OpenMP threading on aarch64 Linux (libgomp1 is added
#                       to the runtime stage below).
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=ON"

WORKDIR /deps
COPY backend/requirements.txt /deps/requirements.txt
COPY backend/requirements-ml.txt /deps/requirements-ml.txt

# Dedicated venv shared between the stages.
ENV VIRTUAL_ENV=/opt/armiferx/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 1. API dependencies (pure wheels; cached layer).
RUN pip install --no-cache-dir -r /deps/requirements.txt

# 2. llama.cpp runtime, pinned to the version validated on the development
#    machine (0.3.34) so inference behavior matches. --no-binary forces the
#    source build so the CMAKE_ARGS above actually apply (a prebuilt aarch64
#    wheel would skip the build and could carry different GGML flags).
RUN pip install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.34

# 3. OPTIONAL Transformers FP16 baseline dependencies (torch CPU, transformers,
#    accelerate). OFF by default: the ARM64 deployment is llama.cpp-only, so
#    the image stays lean and the FP16 baseline is never auto-loaded. Enable
#    only on hardware where that baseline is feasible (>= 16 GiB RAM):
#       docker build --build-arg INSTALL_BASELINE_DEPS=1 ...
ARG INSTALL_BASELINE_DEPS="0"
RUN if [ "$INSTALL_BASELINE_DEPS" = "1" ]; then \
        pip install --no-cache-dir -r /deps/requirements-ml.txt; \
    fi

# -- Runtime stage: slim image, no compilers ---------------------------------
FROM python:3.11-slim-bookworm AS runtime

# libgomp1: OpenMP runtime shared library required by the OpenMP-enabled
# llama.cpp build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/armiferx/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=build $VIRTUAL_ENV $VIRTUAL_ENV

# Non-root runtime user (the image never runs as root).
RUN useradd --create-home --uid 1000 arminferx

WORKDIR /app
COPY backend/ /app/backend/
# The STEP 9/11 benchmark driver (composes the existing BenchmarkRunner with
# the exact STEP 9/11 configuration; used by the in-container benchmark
# command — see docs/arm64-deployment.md).
COPY scripts/run_llamacpp_benchmark.py /app/scripts/run_llamacpp_benchmark.py

# Runtime writable area: benchmark records + /generate auto-saved records land
# under /app/results (gitignored; bind-mount to persist on the host if wanted).
RUN mkdir -p /app/models/gguf /app/results \
    && chown -R arminferx:arminferx /app/results

USER arminferx
WORKDIR /app/backend

EXPOSE 8000

# Environment: CPU-only inference; llamacpp-optimized remains the default
# engine. Engine defaults are untouched (n_ctx=2048, n_threads=8,
# n_gpu_layers=0, greedy decoding, max_new_tokens=64 for the benchmark).
ENV ARMINFERX_DEFAULT_ENGINE=llamacpp-optimized \
    ARMINFERX_DEVICE=cpu

# Reproducible startup: nothing is loaded at boot; engines load lazily on the
# first request (the 2 GB Q4_K_M model is loaded once, then reused).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# /health returns 200 immediately (no model is loaded at startup), so this
# checks the API process, not model state.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/opt/armiferx/venv/bin/python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).getcode()==200 else 1)"]
