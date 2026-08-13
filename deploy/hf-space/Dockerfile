# =============================================================================
# ArmInferX — Hugging Face Spaces (Docker) backend image
# =============================================================================
#
# Deploys the FastAPI backend with the llama.cpp Q4_K_M engine to a Hugging
# Face Docker Space so anyone can use the API at a public URL. The image is
# the same CPU-only, llama.cpp-only build as docker/backend/Dockerfile with
# two Spaces-specific differences:
#
#   1. HF Spaces requires the app to listen on port 7860 (app_port in the
#      Space card README.md metadata).
#   2. The ~2 GB GGUF is gitignored and cannot be bind-mounted on Spaces, so
#      it is downloaded from Hugging Face during the image build and
#      SHA-256-verified (download_model.py).
#
# Build context = the Space repo root (this directory synced with backend/):
#   bash sync-backend.sh        # populates ./backend from the project repo
#   docker build -t arminferx-hf-space .
#
# Lifecycle: no model is loaded at startup (lazy EngineManager); Q4_K_M is
# loaded on the first /generate request and cached for the process lifetime.
# =============================================================================

# -- Build stage: compilers + llama-cpp-python source build ------------------
# Python 3.11: best wheel availability for the pinned API deps and a
# well-supported toolchain for llama-cpp-python 0.3.34's vendored ggml.
FROM python:3.11-slim-bookworm AS build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Portable CPU build (same flags as docker/backend/Dockerfile): GGML_NATIVE=OFF
# keeps the binary portable; GGML_OPENMP=ON enables threading (libgomp1 is
# added to the runtime stage below).
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=ON"

WORKDIR /deps
COPY backend/requirements.txt /deps/requirements.txt

ENV VIRTUAL_ENV=/opt/armiferx/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 1. API dependencies (pure wheels; cached layer).
RUN pip install --no-cache-dir -r /deps/requirements.txt

# 2. llama.cpp runtime, pinned to the version validated on the development
#    machine (0.3.34). --no-binary forces the source build so CMAKE_ARGS
#    actually apply.
RUN pip install --no-cache-dir --no-binary llama-cpp-python llama-cpp-python==0.3.34

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
COPY download_model.py /app/download_model.py

# Download + verify the Q4_K_M GGUF at build time so the model is ready when
# the Space starts (first /generate then only pays the ~16 s in-memory load).
RUN mkdir -p /app/models/gguf /app/results \
    && python /app/download_model.py \
    && chown -R arminferx:arminferx /app/models /app/results

USER arminferx
WORKDIR /app/backend

# HF Spaces expects the app on port 7860 (see app_port in README.md metadata).
EXPOSE 7860

# CPU-only inference; llamacpp-optimized remains the default engine. Engine
# defaults are untouched (n_ctx=2048, n_threads=8, n_gpu_layers=0, greedy,
# max_new_tokens=64). Override ARMINFERX_CORS_ORIGINS in the Space's
# "Variables and secrets" settings with the deployed frontend origin, e.g.
#   ARMINFERX_CORS_ORIGINS=https://arminferx.vercel.app
ENV ARMINFERX_DEFAULT_ENGINE=llamacpp-optimized \
    ARMINFERX_DEVICE=cpu \
    ARMINFERX_CORS_ORIGINS=""

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

# /health returns 200 immediately (no model is loaded at startup), so this
# checks the API process, not model state.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["/opt/armiferx/venv/bin/python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).getcode()==200 else 1)"]
