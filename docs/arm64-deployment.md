# ArmInferX — ARM64 Docker Deployment (STEP 14A, preparation only)

This document describes how to run ArmInferX as a **Linux ARM64 CPU-only**
container using the validated Q4_K_M llama.cpp engine.

> **Status: preparation only.** No Arm64 benchmark has been run yet, and no
> Arm64 performance claim is made anywhere in this document or the code.
> All measured Q4_K_M numbers to date come from the Windows development
> laptop (see `docs/optimization.md`). This page only defines how to build,
> start, verify, and benchmark the container on aarch64.

## What is deployed

| Piece | Value |
|---|---|
| Backend | FastAPI + uvicorn, `main:app` on port 8000 |
| Engine | `llamacpp-optimized` (default), runtime `llama.cpp` |
| Model | `models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` (Q4_K_M, CPU-only) |
| Engine config | defaults: `n_ctx=1024`, `n_threads=2`, `n_gpu_layers=0`, greedy decoding, `max_new_tokens=64` (benchmark) — tuned for 1 GB RAM / 2 vCPU |
| Benchmark | the existing STEP 9/11 procedure via `BenchmarkRunner` (same prompt + config) |

## Files involved

| File | Purpose |
|---|---|
| `docker/backend/Dockerfile` | Multi-stage ARM64 build (compilers in build stage only; slim runtime) |
| `docker-compose.yml` | Backend service: build context, model mount, CPU-only env |
| `.dockerignore` | Keeps the build context small; models/results never enter the image |
| `docs/arm64-deployment.md` | This guide |

Deliberately untouched: `BenchmarkConfig`, `BenchmarkRunner`,
`BenchmarkService`, `BenchmarkMetrics`, both inference engines, `main.py`,
FastAPI routes, the Q4_K_M model, and the benchmark methodology.

## Dependency strategy (llama-cpp-python on aarch64)

`llama-cpp-python` is **not** in `backend/requirements.txt` (the Windows dev
venv has it installed manually at `0.3.34`). The image installs it itself:

- Pinned to **`llama-cpp-python==0.3.34`** — the same version validated on the
  development machine, so inference behavior matches.
- Built **from source** with `--no-binary llama-cpp-python` so the `CMAKE_ARGS`
  below actually apply (pip would otherwise install a prebuilt aarch64 wheel
  and silently skip the documented build flags), using `build-essential`,
  `cmake`, and `python3-dev` in the build stage only.
- `CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=ON"`:
  - `GGML_NATIVE=OFF` → portable binary (no `-mcpu=native`); runs on
    Graviton 1/2/3, Ampere Altra, Neoverse-N1/N2, and generic aarch64.
  - `GGML_OPENMP=ON` → OpenMP threading on aarch64 Linux; the runtime stage
    adds `libgomp1`.
- API dependencies come from the pinned `requirements.txt` (pure wheels).

The FP16 Transformers baseline dependencies (torch CPU, transformers,
accelerate) are **opt-in** via the `INSTALL_BASELINE_DEPS` build arg (default
`0`). The ARM64 deployment is llama.cpp-only; the baseline engine code is
unchanged and simply unavailable (clean error) unless those deps are built in.

## Build

Native ARM64 host (build context must be the **repo root** — the image
copies `backend/` and the benchmark driver from it; building with context
`docker/backend` will fail the `COPY`):

```bash
docker build -f docker/backend/Dockerfile -t arminferx-backend .
# or, with compose:
docker compose build backend
```

ARM64 image from an x86 host (QEMU/binfmt emulation via buildx):

```bash
docker buildx create --use   # once, if no builder exists
docker buildx build --platform linux/arm64 \
  -f docker/backend/Dockerfile -t arminferx-backend .
```

## Model strategy

The Q4_K_M GGUF is gitignored and ~470 MB. `docker/backend/Dockerfile`
downloads it from Hugging Face **at build time** and SHA-256-verifies it into
`/app/models/gguf` (this is what the Railway deployment uses — no runtime
mount needed). The read-only bind-mount below is an optional alternative that
keeps the model out of the image; the file is identical either way:

```bash
docker run --rm -d --name arminferx \
  -p 8000:8000 \
  -v "$PWD/models/gguf":/app/models/gguf:ro \
  arminferx-backend
```

`backend/` is copied to `/app/backend/` in the image, so the code's
`PROJECT_ROOT` is `/app` and the engine default resolves to
`/app/models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` — exactly where the
downloader (or the mount) places it. `docker-compose.yml` mounts the same
directory automatically.

## Configuration (environment)

| Variable | Value | Meaning |
|---|---|---|
| `ARMINFERX_DEFAULT_ENGINE` | `llamacpp-optimized` | default engine preserved |
| `ARMINFERX_DEVICE` | `cpu` | CPU-only inference |

Nothing overrides the engine defaults (`n_ctx`, `n_threads`, `n_gpu_layers`,
temperature, `max_new_tokens`). The container starts with **no model loaded**
(lazy `EngineManager`); the Q4_K_M model loads on the first `/generate` and is
then reused.

## Start

```bash
docker compose up --build
# or with docker run (see "Model mounting strategy" above)
```

The API is at `http://localhost:8000`; interactive docs at `/docs`.

## Verification

Health (process liveness; `/health` answers immediately, nothing loaded):

```bash
curl -fsS http://localhost:8000/health
# {"status":"healthy","service":"ArmInferX Backend",...,"loaded_engines":[],...}
```

Container-internal health check (no `curl` in the slim image):

```bash
docker compose exec backend python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

Real inference verification (loads Q4_K_M once; ~0.5–0.8 GB RSS):

```bash
curl -s -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain what an AI inference engine is.","engine_id":"llamacpp-optimized"}'
# -> {"status":"success","engine_id":"llamacpp-optimized","runtime":"llama.cpp",...}
```

The response carries `engine_id`, `runtime`, `model`, `latency_ms`,
`generated_tokens`, `tokens_per_second`, and `ttft_ms`.

## Benchmark (existing BenchmarkRunner, STEP 9/11 procedure)

The image ships the STEP 9 driver (`scripts/run_llamacpp_benchmark.py`),
which composes the **existing** `BenchmarkRunner` with the exact STEP 9/11
configuration (prompt `"Explain what an AI inference engine is."`,
`max_new_tokens=64`, `temperature=None`/greedy, `chat_template=False`,
`warmup=1`, `repeats=5`) — no methodology change:

```bash
docker compose exec backend python /app/scripts/run_llamacpp_benchmark.py
```

It prints per-run metrics + `BenchmarkAggregates` and verifies the persisted
records. Records land in `/app/results/benchmarks/llamacpp-optimized/`
(inside the container). To keep them on the host, add the bind mount
`-v "$PWD/results":/app/results` (commented out in `docker-compose.yml`).
The host `./results` directory must exist and be writable by the container's
uid 1000, otherwise record writes fail (the container runs as a non-root
user).

## Automated STEP 14B smoke test (one-shot script)

`scripts/arm64_deploy_smoke.sh` runs the complete STEP 14B sequence on a real
Linux ARM64 host: environment report → build → image-architecture check →
llama-cpp-python + flag verification → model mount + SHA-256 check → startup →
`/health` before inference (`loaded_engines=[]`) → **two** real `POST
/generate` requests (prompt `"Hello! Who are you?"`, `engine_id
llamacpp-optimized`) → `/health` after inference
(`loaded_engines=["llamacpp-optimized"]`) → engine-config verification
(n_ctx=1024, n_threads=2, n_gpu_layers=0, greedy, max_new_tokens=64). It
stops with a FAIL at the exact stage on any problem and never runs the
5-repeat benchmark.

On the ARM64 host (from the repo root; the ~470 MB GGUF is gitignored, so
copy it there — see the script header for `scp` instructions):

```bash
bash scripts/arm64_deploy_smoke.sh              # full run (build included)
bash scripts/arm64_deploy_smoke.sh --skip-build # reuse an existing image
```

The script asserts the verified Q4_K_M SHA-256
(`74a4da8c…a9db`) on the host **and** inside the container, and confirms the
STEP 14A build flags (`--no-binary`, `-DGGML_NATIVE=OFF`, `-DGGML_OPENMP=ON`)
are still present in the Dockerfile.

## What is deliberately NOT done here

- No Arm64 benchmark run (this is preparation only).
- No FP16-vs-Q4_K_M comparison, no speedup claims (FP16 is infeasible on the
  dev machine and no Arm64 numbers exist).
- No quantization, no new model downloads.
- No change to `BenchmarkConfig`, `BenchmarkRunner`, `BenchmarkService`,
  `BenchmarkMetrics`, the engines, `main.py`, or the routes.
- The Transformers baseline is not auto-loaded anywhere.

## Validation on the dev machine

Docker is not installed on the Windows dev machine, so the deployment
configuration is validated by `tests/test_arm64_deployment.py`
(config/driver/Dockerfile/compose consistency) plus the existing test suite:

```bash
backend/.venv/Scripts/python.exe -m py_compile backend/main.py backend/benchmark/runner.py scripts/run_llamacpp_benchmark.py tests/test_arm64_deployment.py
backend/.venv/Scripts/python.exe tests/test_arm64_deployment.py
```
