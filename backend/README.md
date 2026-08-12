# ArmInferX Backend

FastAPI server for inference orchestration and benchmark management.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-ml.txt
```

## Baseline Model Setup

The baseline is **Qwen2.5-3B-Instruct** (Apache 2.0) — chosen for its quality/size
balance, CPU friendliness, and clean migration path to ONNX Runtime and llama.cpp.

```bash
# 1. Download the model snapshot (resumable) into models/downloaded/
cd ..
backend/.venv/Scripts/python scripts/download_model.py

# 2. Verify the model and tokenizer load and generate on CPU
backend/.venv/Scripts/python scripts/verify_model.py
```

Pass `--repo-id` / `--out-dir` to `scripts/download_model.py` to stage additional
models for benchmarking. On 8 GB laptops the fp16 weights (~6.2 GB) are loaded
with accelerate disk-offload (max 3 GiB RAM), so verification is slow but needs
no GPU. No optimization is applied at this stage.

## Optimized Engine (llama.cpp)

`LlamaCppOptimizedEngine` is **implemented** and registered in the engine
registry as `llamacpp-optimized` (`runtime = llama.cpp`). It runs GGUF models
through llama.cpp (via `llama-cpp-python`) on CPU only (`n_gpu_layers=0`),
exposing the same `InferenceEngine` interface as the transformers baseline.

- **Default optimized model:** Qwen2.5-3B-Instruct **Q4_K_M**
  (`models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf`, single-file GGUF;
  `n_ctx=2048`, `n_threads=8`, greedy decoding).
- **CPU-only inference has been validated successfully** on the Windows
  development machine with llama-cpp-python 0.3.34.
- **The common benchmark layer supports the engine:** `BenchmarkConfig`,
  `BenchmarkRunner` and `EngineResultStore` (`results/benchmarks/<engine_id>/`)
  work with `llamacpp-optimized` unchanged.
- **A Q4_K_M benchmark has been measured on the Windows development machine:**
  1 warmup + 5 timed runs, persisted as JSON records under
  `results/benchmarks/llamacpp-optimized/` (see `docs/optimization.md` for the
  full measured/unmeasured breakdown).

The FP16 transformers baseline could not complete inference on this 7.63 GiB
RAM machine (hardware memory constraint), so **no FP16-vs-Q4_K_M performance
comparison is claimed** — measured results are reported in `docs/optimization.md`.

## Running

```bash
# Development (auto-reload)
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Windows troubleshooting: llama.dll fails to load

Symptom: `import llama_cpp` raises
`RuntimeError: Failed to load shared library '...\llama_cpp\lib\llama.dll': Could not
find module ... (or one of its dependencies)`.

Root cause (seen on this dev machine): the llama.cpp DLLs are built with MSVC
OpenMP and import **`VCOMP140.DLL`** (the VC++ OpenMP runtime). If it is absent
from `C:\Windows\System32` (e.g. the VC++ 2015-2022 redistributable was never
installed or was removed), the DLLs cannot load — even though they are present
and are valid x86-64 binaries.

Fix (no rebuild needed — keeps `llama-cpp-python==0.3.34`):

1. Obtain a **64-bit** `vcomp140.dll` (part of the VC++ redistributable; many
   apps bundle it, e.g. `C:\Program Files\Cisco Packet Tracer ...\bin\` —
   verify with `file` that it is `PE32+ ... x86-64`, not the 32-bit copy in
   `C:\Windows\SysWOW64\`).
2. Copy it next to the package DLLs so `load_shared_library` finds it:
   `cp <path>/vcomp140.dll .venv/Lib/site-packages/llama_cpp/lib/`
3. Verify: `python -c "from llama_cpp import Llama; print('ok')"`

No application code changes are needed for this fix.

Automated alternative: `scripts/fix_llama_dll.py` does steps 1–3 for you — it
checks whether `import llama_cpp` works, locates an x86-64 `vcomp140.dll` on
the machine (System32, Visual Studio folders, or common app `bin/` dirs) and
copies it into the package `lib/` folder. Run from the repo root with the
project venv: `backend\.venv\Scripts\python.exe scripts\fix_llama_dll.py`
(exits 0 when `import llama_cpp` already works; idempotent).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (default/available/loaded engines, loaded model id) |
| `POST` | `/generate` | Generate text: `{"prompt": "...", "engine_id"?: "..."}` → `{"status", "model", "response", "latency_ms", engine_id?, runtime?, generated_tokens?, tokens_per_second?, ttft_ms?}` |
| `POST` | `/generate/stream` | SSE stream of the completion (token deltas + final metadata `done` event); streaming-capable engines only |
| `GET` | `/benchmarks` | All saved benchmark records, oldest first; optional `?engine_id=` filter (empty list if none) |
| `GET` | `/benchmarks/latest` | Most recent benchmark record incl. `ttft_ms`, `generated_tokens`, `tokens_per_second`, engine tag (404 if none) |
| `GET` | `/benchmarks/summary` | Averages: `{"avg_latency_ms", "avg_memory_mb", "avg_cpu_percent", "total_runs"}`; optional `?engine_id=` filter |
| `GET` | `/optimization/report` | Machine-readable optimization evidence report (STEP 11 artifact) |
| `GET` | `/docs` | Auto-generated Swagger UI (schemas, examples, error codes) |
| `GET` | `/redoc` | ReDoc alternative documentation |

## Structure

```
backend/
├── main.py                          # Composition root: app factory, lifespan (loads
│                                   #   the model once), middleware, routes, error mapping
├── api/
│   ├── routes/
│   │   ├── inference/
│   │   │   ├── schemas.py           # Pydantic request/response models (OpenAPI contract)
│   │   │   ├── router.py            # APIRouter: POST /generate (DI + Swagger docs)
│   │   │   ├── model_loader.py      # Loading only: tokenizer + model from disk → InferenceModel
│   │   │   └── inference_service.py # Generation only: validated text generation on a loaded model
│   │   └── benchmarks/
│   │       ├── schemas.py           # BenchmarkRecord / BenchmarkSummary response models
│   │       └── router.py            # APIRouter: GET /benchmarks, /latest, /summary
│   └── utils/
│       ├── exceptions.py            # Shared typed exception hierarchy (ArmInferXError family)
│       ├── logging_config.py        # Centralized logging setup
│       ├── error_handlers.py        # Domain exceptions → JSON error responses
│       └── timing.py                # Stopwatch: baseline wall-clock latency (start/end, ms)
├── engines/
│   ├── base.py                      # InferenceEngine interface + EngineInfo/StreamChunk
│   ├── result.py                    # GenerationResult (shared engine result contract)
│   ├── manager.py                   # EngineManager: lazy load + cache by engine_id (default engine)
│   ├── registry.py                  # engine_id -> engine class; load_engine() uniform entry point
│   ├── transformers_baseline.py     # TransformersBaselineEngine (wraps the baseline service)
│   ├── llamacpp_optimized.py        # LlamaCppOptimizedEngine (llama.cpp / GGUF, CPU-only, Q4_K_M default)
│   └── __init__.py                  # Engine public re-exports
├── benchmark/
│   ├── metrics.py                   # BenchmarkMetrics result object + psutil samplers (memory, CPU)
│   ├── logger.py                    # Dedicated benchmark logging
│   ├── benchmark_service.py         # BenchmarkService: measures a call → BenchmarkMetrics
│   └── storage.py                   # BaselineResultStore: unique JSON records under results/baseline/
├── requirements.txt                 # Pinned HTTP/API dependencies
├── requirements-ml.txt              # Pinned ML runtime deps (CPU torch, transformers, accelerate)
└── .venv/                           # Virtual environment (gitignored)
```

**No model is loaded at startup.** An `EngineManager` (in `app.state.engine_manager`)
loads the requested engine lazily on first use and reuses it for every
subsequent request — the 2 GB Q4_K_M model is loaded once per process, never
per request, and the FP16 Transformers baseline is **never loaded
automatically** (it is infeasible on this machine's 7.63 GiB RAM). The default
engine is `llamacpp-optimized`, overridable via `ARMINFERX_DEFAULT_ENGINE`
(e.g. `transformers-baseline` on hardware where that is feasible).

`POST /generate` accepts an optional `engine_id` (`llamacpp-optimized` or
`transformers-baseline`) resolved through the engine registry; unknown ids get
a clean 400. Responses include engine identity, generated token count,
tokens/second and TTFT when the engine provides them. `POST /generate/stream`
streams the same generation as Server-Sent Events for streaming-capable
engines. Run from this directory: `uvicorn main:app --reload --port 8000`.

`POST /generate` measures **engine inference latency**: wall-clock milliseconds
around the engine's `generate()` call (see `api/utils/timing.py`).

The modular benchmark subsystem (`backend/benchmark/`) measures latency,
time-to-first-token (TTFT), process memory (MB), CPU (%), timestamp,
generated tokens, and tokens-per-second throughput, returning a
`BenchmarkMetrics` object — the foundation for the full benchmark engine.
New metrics (throughput by phase) plug in later via `BenchmarkMetrics.extra`.

TTFT is measured inside the inference service with an instrumentation
streamer attached to `model.generate()`: it records the wall-clock time until
the first output token is produced (prefill + first decode step) and is kept
as a separate metric (`ttft_ms`) from total latency (`latency_ms`). The
streamer only observes tokens, so the generated text and the `/generate`
contract are unchanged.

Every successful `POST /generate` automatically measures the request and
saves a JSON record to `results/baseline/` (gitignored) containing prompt,
model, response, latency, ttft, memory (MB), CPU (%), generated tokens,
tokens per second, and timestamp. Token counting is tokenizer-native:
`generated_tokens` is the number of output token IDs the model emitted, and
`tokens_per_second = generated_tokens / (latency_ms / 1000)` uses the same
inference time stored as `latency_ms`. Filenames are unique (timestamp + UUID
suffix); file-writing errors are logged and never fail the request.

Saved records are exposed read-only through three endpoints (no charts yet):
`GET /benchmarks` (all records), `GET /benchmarks/latest` (404 when empty),
and `GET /benchmarks/summary` (average latency/memory/CPU + run count, all
zeros when empty). They are documented in Swagger UI at `/docs`.

## Files

| File | Purpose |
|------|---------|
| `main.py` | Application entry point, router setup, middleware config |
| `api/routes/inference/schemas.py` | Pydantic request/response models with OpenAPI examples |
| `api/routes/inference/router.py` | `APIRouter` for `POST /generate` (dependency injection, docs) |
| `api/routes/inference/model_loader.py` | Load tokenizer + model from disk into an `InferenceModel` |
| `api/routes/inference/inference_service.py` | Validate and run text generation on the loaded model |
| `engines/` | `InferenceEngine` interface, `TransformersBaselineEngine`, `LlamaCppOptimizedEngine` (llama.cpp / GGUF) |
| `api/utils/` | Shared exceptions, logging, HTTP error handlers, latency Stopwatch |
| `api/routes/benchmarks/schemas.py` | `BenchmarkRecord` / `BenchmarkSummary` Pydantic response models |
| `api/routes/benchmarks/router.py` | `GET /benchmarks`, `/latest`, `/summary` (read-only results API) |
| `benchmark/metrics.py` | `BenchmarkMetrics` object + process memory/CPU samplers (psutil) |
| `benchmark/logger.py` | Dedicated benchmark logging (separate from HTTP loggers) |
| `benchmark/benchmark_service.py` | `BenchmarkService.run()`/`measure()` → measured `BenchmarkMetrics` |
| `benchmark/storage.py` | `BaselineResultStore.save()` → unique JSON file in `results/baseline/` |
| `requirements.txt` | Pinned HTTP/API dependencies |
| `requirements-ml.txt` | Pinned ML runtime deps (CPU torch, transformers, accelerate) |
| `.venv/` | Virtual environment (gitignored) |
