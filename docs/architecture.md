# ArmInferX — Architecture

This document describes how the ArmInferX components fit together: the request
path from the browser to an inference engine, and the supporting benchmark /
reporting data flow. It is a **design map**, not a claim about performance —
no Arm64 or FP16 numbers exist yet (see `docs/optimization.md`).

## System overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Frontend (React 18 + Vite, port 3000)                              │
│    Studio tab            Optimization Dashboard tab                 │
│    └─ chat + engine selector        └─ GET /optimization/report     │
│    └─ SSE reader (streaming)            (fallback: bundled JSON)    │
└───────────────┬──────────────────────────┬──────────────────────────┘
                │ fetch / SSE              │ fetch
                ▼                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  FastAPI backend (uvicorn, port 8000) — no model loaded at startup  │
│                                                                     │
│  app.state.engine_manager : EngineManager                           │
│      resolve(engine_id) → registry lookup (400 on unknown id)       │
│      get(engine_id)      → lazy load once + cache (thread-safe)     │
│                                                                     │
│  POST /generate            POST /generate/stream  (SSE)             │
│      │                          │  per-token {text} events +        │
│      ▼                          ▼  final {done, metadata} event     │
│  engine.generate(prompt)  →  engine.stream_generate(prompt)         │
│      │  (BenchmarkService.measure wraps every call)                 │
│      ▼                                                              │
│  BenchmarkMetrics { latency_ms, ttft_ms, memory_mb, cpu_percent,    │
│                     generated_tokens, tokens_per_second }           │
│      │  auto-saved every request                                    │
│      ▼                                                              │
│  BaselineResultStore → results/baseline/baseline-*.json  (gitignored)│
└─────────────────────────────────────────────────────────────────────┘

Engine layer (backend/engines/)
    ENGINE_REGISTRY = {
        "transformers-baseline": TransformersBaselineEngine,  # FP16, no streaming
        "llamacpp-optimized":    LlamaCppOptimizedEngine,     # Q4_K_M, SSE streaming
    }
        │  both implement the same InferenceEngine interface:
        │  load_model / generate / stream_generate / get_model_info
        ▼
Benchmark layer (backend/benchmark/)
    BenchmarkConfig (deterministic, engine-agnostic)
        → BenchmarkRunner.run(engine, config)
            → warmup (untimed) + repeats (timed) via BenchmarkService.measure
            → EngineResultStore → results/benchmarks/<engine_id>/benchmark-*.json
            → BenchmarkAggregates (mean / median / p90 latency, TTFT, tok/s, mem, CPU)

Report layer (backend/optimization/)
    scripts/run_optimization_report.py
        → GGUF metadata (parsed from file bytes) + footprint + fresh benchmark
        → optimization_report.json  (results/, gitignored; measured/not-measured)
        → GET /optimization/report serves a path-sanitized copy to the dashboard
```

## Component responsibilities

### Frontend (`frontend/src/`)

- `App.jsx` — the Studio: composer, engine selector, streaming chat cards with
  metadata chips (engine, runtime, model, latency, tokens, tokens/sec, TTFT),
  and the "Benchmark · latest run" panel that refreshes after every inference.
- `optimization/OptimizationDashboard.jsx` — renders the evidence report:
  summary metric cards, footprint bars, FP16 feasibility banner,
  measured-vs-not-measured lists, per-run table, configuration grid, limitations.
- `optimization/loadReport.js` — the **single** place that reads the
  optimization report. It normalizes the raw JSON into a view model so
  components never hardcode numbers.

### FastAPI backend (`backend/main.py`)

- `create_app()` wires logging, CORS (3000 / 5173 / 4173), typed error handlers,
  and the three routers (inference, benchmarks, optimization).
- The `lifespan` context builds an `EngineManager` onto `app.state.engine_manager`
  and loads **nothing**. `/health` answers immediately with `loaded_engines: []`
  until the first inference.

### EngineManager (`backend/engines/manager.py`)

The single holder of loaded engine instances:

- **Lazy loading** — `get(engine_id)` constructs the engine through the registry
  on first use only. Startup and `/health` never touch a model.
- **Caching** — loaded engines are kept in `self._loaded` and reused for every
  request; the ~2 GB Q4_K_M model is loaded exactly once per process.
- **Thread safety** — a `threading.Lock` around the load prevents concurrent
  first requests from double-loading a model.
- **Engine selection** — `resolve(None)` falls back to the configured default
  (`ARMINFERX_DEFAULT_ENGINE`, default `llamacpp-optimized`); unknown ids raise
  `UnknownEngineError`, mapped to a clean HTTP 400.

### Engine registry (`backend/engines/registry.py`)

A plain `engine_id → engine class` map plus `load_engine(engine_id, **kwargs)`,
the uniform entry point used by both the HTTP layer and the benchmark driver.

### The two engines

| | `TransformersBaselineEngine` | `LlamaCppOptimizedEngine` |
|---|---|---|
| File | `engines/transformers_baseline.py` | `engines/llamacpp_optimized.py` |
| Model | Qwen2.5-3B-Instruct FP16 | `models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf` |
| Loads | tokenizer + model via `api/routes/inference/model_loader.py` | GGUF via `llama_cpp.Llama`, CPU-only |
| Streaming | `EngineOperationUnsupportedError` | `stream_generate()` → `StreamChunk`s |
| TTFT | internal instrumentation streamer | first streamed token timing |
| Status | not auto-loaded (infeasible on 7.63 GiB) | default engine, validated + benchmarked |

Both return the shared `GenerationResult` (`engines/result.py`) with
tokenizer-native `prompt_tokens` / `generated_tokens` and a `latency_ms` / `ttft_ms`.

### Streaming (`POST /generate/stream`)

- The engine's `stream_generate()` yields `StreamChunk(text, is_first, is_last)`.
- The router wraps them as Server-Sent Events: one `data: {text}` per token,
  then a final `data: {done: true, engine_id, runtime, model, latency_ms,
  generated_tokens, tokens_per_second, ttft_ms}` event.
- The browser `streamGenerate()` in `App.jsx` reads the stream, appends deltas
  live, and fills the metadata chips from the `done` event.
- Streaming failures are emitted as `data: {error}` events — never raw tracebacks.
- Every streamed generation is also measured and auto-saved (same
  `BenchmarkMetrics` path as `/generate`), so the "latest run" panel stays accurate.

### Benchmark persistence

- **Per-request records** — every `/generate` and `/generate/stream` is measured
  by `BenchmarkService` and saved as a unique JSON file under `results/baseline/`
  (`baseline-<utc>-<uuid>.json`, atomic write). Read-only API: `/benchmarks`,
  `/benchmarks/latest`, `/benchmarks/summary`.
- **Benchmark-run records** — `BenchmarkRunner` (used by the scripts and the
  report generator) persists each timed repeat under
  `results/benchmarks/<engine_id>/` (`benchmark-*.json`) and computes
  aggregates. All of `results/` is gitignored.

### Optimization report & dashboard data flow

1. `scripts/run_optimization_report.py` runs the fresh benchmark, parses GGUF
   metadata from file bytes, computes the FP16-vs-Q4_K_M footprint, and writes
   `results/optimization_report.json` with explicit `measured` and
   `not_measured` blocks (FP16 metrics are hard-coded `null` in
   `optimization/report.py` — they can never be fabricated).
2. The dashboard requests `GET /optimization/report` (path fields sanitized
   server-side). If the backend is unreachable, `loadReport.js` falls back to
   the bundled static copy at `frontend/public/optimization-report.json`
   (regenerated by `scripts/_bundle_sanitized_report.py`).
3. `loadReport.js` normalizes everything into one view model; the dashboard
   renders **"Not measured"** for every null value — never zero.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ARMINFERX_DEFAULT_ENGINE` | `llamacpp-optimized` | engine used when a request omits `engine_id` |
| `ARMINFERX_MODEL_DIR` | `models/downloaded/qwen2.5-3b-instruct` | baseline model directory |
| `ARMINFERX_DEVICE` | `cpu` | baseline device |
| `ARMINFERX_DTYPE` | `float16` | baseline dtype |
| `ARMINFERX_MAX_CPU_MEMORY` | `3GiB` | baseline memory cap |

Engine defaults (llama.cpp) are fixed for comparability: `n_ctx=2048`,
`n_threads=8`, `n_gpu_layers=0`, greedy decoding, `max_new_tokens=64`.

## Source of truth

- `backend/main.py` — composition root
- `backend/api/routes/inference/router.py` — `/generate`, `/generate/stream`
- `backend/engines/manager.py` — lazy load, cache, selection
- `backend/benchmark/runner.py` + `config.py` — benchmark procedure
- `backend/optimization/report.py` — measured/not-measured report contract
- `frontend/src/optimization/loadReport.js` — dashboard data mapping
