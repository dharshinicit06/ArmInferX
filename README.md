<div align="center">

# ArmInferX

### AI Inference Optimization Studio for Arm64 Cloud

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-2496ED.svg)](https://www.docker.com/)

A demo-ready inference studio that runs a **quantized (Q4_K_M) llama.cpp** LLM
runtime, measures real inference metrics, and visualizes the evidence in a web
dashboard — all on plain CPU. The optimization story is a measured
**footprint/feasibility** result: the FP16 Transformers baseline could not run
on this 7.63 GiB machine, while Q4_K_M runs and is 69.05% smaller.

</div>

---

## Status

| Step | What | State |
|------|------|-------|
| STEP 9 | Q4_K_M llama.cpp benchmark | ✅ done (measured, persisted) |
| STEP 10A | FP16 baseline feasibility | ⛔ safely aborted — not enough RAM |
| STEP 11 | Optimization evidence report | ✅ done (measured / not-measured separated) |
| STEP 12 | Optimization Dashboard | ✅ done, validated |
| STEP 13 | Engine selection + streaming `/generate/stream` | ✅ done, validated |
| STEP 14A | ARM64 Docker deployment preparation | ✅ done (build config only) |
| STEP 14B | ARM64 deployment smoke run | ⏸ **postponed** — needs a real Linux ARM64 host |
| STEP 15 | Final polish, docs, demo readiness | ✅ done (this state) |

> **Hardware note:** everything was built and measured on a **Windows 11 (AMD64)
> laptop with 7.63 GiB RAM** and **no Docker installed**. All measured numbers
> below are from that machine. No Arm64 numbers exist yet.

---

## Problem Statement

Running LLMs on Arm64 cloud infrastructure (AWS Graviton, Ampere Altra, NVIDIA
Grace) is increasingly cost-effective, but teams have no standardized tooling to:

1. **Measure** real-world CPU inference performance across runtimes and quantizations.
2. **Compare** a baseline (unoptimized FP16 Transformers) pipeline with an
   optimized (quantized llama.cpp) pipeline under an identical procedure.
3. **Visualize** the trade-offs between latency, throughput, memory, and model footprint.

Without this, deployment decisions about models and quantizations are guesses.

## Solution

**ArmInferX** is an end-to-end studio that:

1. Loads an inference **engine** through a pluggable registry
   (`llamacpp-optimized`), lazily and cached per process.
2. Serves text generation over HTTP — plain and **streamed** (SSE) — while
   automatically measuring **latency, TTFT, tokens/sec, memory, and CPU** on
   every request.
3. Runs a **reproducible, engine-agnostic benchmark** (`BenchmarkConfig` +
   `BenchmarkRunner`) that persists timestamped JSON records per engine.
4. Generates a machine-readable **optimization report** that strictly separates
   **measured** facts from **not measured** ones (no fabricated numbers).
5. Visualizes everything in a **React dashboard** — the chat Studio and the
   Optimization Dashboard.

## Key innovation

**Honest, reproducible measurement.** The project's evidence pipeline was built
to never overclaim:

- Every number is either **measured** (Q4_K_M CPU inference on this laptop) or
  explicitly **not measured** (FP16 latency/throughput, any speedup, any Arm64
  number) — enforced by the report schema (`measured` / `not_measured` blocks).
- The benchmark procedure is identical for both engines (`BenchmarkConfig`),
  greedy decoding, `chat_template=False`, 1 warmup + 5 timed runs.
- The FP16 baseline **could not complete inference on this 7.63 GiB machine**
  (STEP 10A watchdog aborted at 159–263 MB free RAM), so **no FP16-vs-Q4_K_M
  performance claim is made** — anywhere.
- The Q4_K_M → llama.cpp story is a **footprint/feasibility** optimization
  validated by real CPU inference, not a claimed speedup.

---

## Architecture

```
Frontend (React + Vite, port 3000)
        │  REST + SSE (fetch / EventSource)
        ▼
FastAPI backend (uvicorn, port 8000)
        │  app.state.engine_manager
        ▼
EngineManager ── resolves engine_id, lazy-loads once, caches, thread-safe
        │
        ▼
Engine Registry (engines/registry.py)
        └── llamacpp-optimized     (llama.cpp / GGUF Q4_K_M, CPU-only, streams)
        │
        ▼
InferenceEngine interface (load_model / generate / stream_generate / get_model_info)
        │
        ▼
Benchmark layer + Result storage
        ├── BenchmarkService → BenchmarkMetrics (latency, TTFT, memory, CPU, tokens)
        ├── BenchmarkRunner → aggregates (mean / median / p90)
        ├── EngineResultStore → results/benchmarks/<engine_id>/  (gitignored)
        └── BaselineResultStore → results/baseline/  (per-request auto-save, gitignored)
```

Full details: **`docs/architecture.md`**.

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + engine state (default, available, loaded) |
| `POST` | `/generate` | Generate: `{prompt, engine_id?}` → text + latency, tokens/sec, TTFT |
| `POST` | `/generate/stream` | SSE token stream + final metadata `done` event |
| `GET` | `/benchmarks` | Saved records, oldest first (`?engine_id=` filter) |
| `GET` | `/benchmarks/latest` | Most recent record (404 if none) |
| `GET` | `/benchmarks/summary` | Average latency / memory / CPU + run count |
| `GET` | `/optimization/report` | STEP 11 evidence report (paths sanitized) |
| `GET` | `/docs` | Swagger UI (OpenAPI) |

## The engine

| | `llamacpp-optimized` |
|---|---|
| Runtime | **llama.cpp** (llama-cpp-python) |
| Model | Qwen2.5-3B-Instruct **Q4_K_M** (GGUF, single file) |
| Device | CPU-only (`n_gpu_layers=0`) |
| Streaming | ✅ SSE streaming |
| Status on this laptop | ✅ validated, benchmarked |

The engine implements the shared `InferenceEngine` interface. The FP16
Transformers baseline was evaluated during STEP 10A and found infeasible on
this 7.63 GiB machine, so it is **not part of the engine registry**; its
storage footprint is still compared against Q4_K_M in the optimization report
(a footprint comparison, not a speed claim). The application starts with
nothing loaded and only loads what a request asks for
(`ARMINFERX_DEFAULT_ENGINE=llamacpp-optimized` by default).

## The optimization story (measured)

```
Qwen2.5-3B-Instruct (FP16, ~6.5 GB on disk)
        │
        ▼  STEP 10A feasibility check on 7.63 GiB laptop
FP16 baseline infeasible — watchdog aborted at 159–263 MB free RAM (safely)
        │
        ▼
Qwen2.5-3B-Instruct Q4_K_M — single-file GGUF, ~2.0 GB on disk
        │
        ▼  llama.cpp CPU runtime (llama-cpp-python 0.3.34, n_gpu_layers=0)
CPU inference validated → 5-run benchmark persisted → evidence report generated
```

**Q4_K_M model facts** (parsed from the actual GGUF bytes, not assumed):

| Fact | Value |
|---|---|
| File | `qwen2.5-3b-instruct-q4_k_m.gguf` (single file) |
| File size | 2,104,932,768 bytes ≈ **2,007.4 MB** |
| Parameters | 3,397,103,616 (3.4B) |
| GGUF version / file type | v3 / `Q4_K_M` (tensor mix: 217 Q4_K, 37 Q6_K, 181 F32) |
| Architecture | qwen2 · model context length 32768 (engine runs `n_ctx=2048`) |
| SHA-256 | `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d` |

### Storage footprint (measured, storage only)

| Model | Size on disk | Runs on this laptop? |
|---|---|---|
| FP16 (2 GGUF shards) | 6,800,646,784 bytes ≈ **6,485.6 MB** | ❌ |
| **Q4_K_M** (1 GGUF file) | 2,104,932,768 bytes ≈ **2,007.4 MB** | ✅ |

**Storage reduction: 69.05%** (4,478.2 MB). This is a *footprint* comparison —
it implies **no** inference speedup. No FP16 latency/throughput exists, so no
percentage performance improvement over FP16 is claimed.

---

## Measured results (canonical, from `results/optimization_report.json`)

Benchmark: engine `llamacpp-optimized`, prompt `"Explain what an AI inference
engine is."` (9 prompt tokens), `max_new_tokens=64`, greedy (`temperature=None`),
`chat_template=False`, 1 warmup + **5 timed runs**, `n_ctx=2048`, `n_threads=8`,
`n_gpu_layers=0`. Hardware: Windows 11 AMD64 laptop, 7.63 GiB RAM, 8 logical / 6
physical CPUs.

| Metric | Value |
|---|---|
| Mean latency | **11,278.62 ms** |
| Median latency | 10,314.56 ms |
| P90 latency | 14,018.02 ms |
| Mean TTFT | **156.5 ms** |
| Mean generated tokens | 64 (max) |
| Mean tokens/sec | **5.77** |
| Peak memory (RSS) | **2,460.32 MB** |
| Mean CPU | 535.82 % (≈5.4 busy cores of 8 logical) |

### Measured vs Not measured (strict boundary)

**✅ MEASURED** (Q4_K_M llama.cpp, Windows laptop)
- CPU inference completed successfully
- Latency, TTFT, tokens/sec, generated tokens
- Memory (peak RSS) and CPU utilization
- Model storage footprint (Q4_K_M and FP16 files)
- Model metadata (parsed from GGUF bytes)

**⛔ NOT MEASURED** (kept `null` in the report by design)
- FP16 inference latency
- FP16 TTFT
- FP16 throughput / tokens/sec
- FP16-vs-Q4_K_M speedup percentage
- Any Arm64 performance number

> Why: the FP16 Transformers baseline completed **no inference** on this
> machine. During the STEP 10A feasibility check, system available RAM dropped
> to **159–263 MB** (below the 300 MB severe-paging alarm) while loading the
> model, and the watchdog aborted the run safely. Fabricating a comparison
> would be dishonest — the dashboard renders these values as **"Not measured"**,
> never as zero.

---

## Feature status

| Feature | State |
|---|---|
| Chat Studio (generate + auto-measured benchmark panel) | ✅ |
| Engine selection + streaming (llama.cpp Q4_K_M) | ✅ |
| Streaming inference (SSE token-by-token) | ✅ |
| Optimization Dashboard (measured evidence, footprint, feasibility) | ✅ |
| Benchmark runner + per-engine persistence | ✅ |
| Optimization report (`GET /optimization/report`) | ✅ |
| ARM64 Docker deployment (build config) | ✅ prepared — not yet run |
| Model upload / registry UI | 📋 future |
| Cost estimator, energy profiling, Arm64 measurements | 📋 future |

## Running the demo (no Docker, no ARM64 needed)

Quick start on the current Windows laptop — full runbook in **`docs/demo-runbook.md`**:

```bash
# Terminal 1 — backend
cd backend
.venv\Scripts\activate
uvicorn main:app --port 8000

# Terminal 2 — frontend
cd frontend
npm install        # first time only
npm run dev        # → http://localhost:3000
```

Then in the browser:
1. **Studio** tab → the **llama.cpp — Q4_K_M** engine is selected by default.
2. Type a prompt (e.g. "Explain what an AI inference engine is.") → watch the
   **streaming** response and the metadata chips (engine, runtime, latency,
   tokens, tokens/sec, TTFT) and the live **Benchmark · latest run** panel.
3. **Optimization Dashboard** tab → measured summary cards, footprint
   comparison, FP16 feasibility banner, run history table, configuration, and
   the measured / not-measured evidence lists.
4. Explain the **"Not measured"** FP16 values (hardware limitation) and the
   **ARM64 deployment preparation** (STEP 14A ready, STEP 14B postponed).

API evidence while the backend is running:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/benchmarks?engine_id=llamacpp-optimized
curl -s http://localhost:8000/optimization/report
```

## Installation (from scratch)

```bash
# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt   # API deps
pip install -r requirements-ml.txt  # optional: torch/transformers (model verify/feasibility scripts)
pip install llama-cpp-python==0.3.34  # llama.cpp runtime (validated version)

# Model — the Q4_K_M GGUF must sit at models/gguf/qwen2.5-3b-instruct-q4_k_m.gguf
# (gitignored; copy it in or fetch it from HuggingFace). The FP16 shards are
# NOT required for the demo — the engine never loads them.

# Frontend
cd ../frontend
npm install
```

## Reproducing the benchmark & report

```bash
# Re-runs the 5-run Q4_K_M benchmark and regenerates the evidence report.
# NOTE: this takes ~10+ minutes on this laptop and adds new records.
backend/.venv/Scripts/python scripts/run_optimization_report.py

# Re-bundle the static copy the dashboard falls back to (path-free):
python scripts/_bundle_sanitized_report.py
```

## Project structure

```
ArmInferX/
├── backend/
│   ├── main.py                      # FastAPI composition root, EngineManager wiring
│   ├── api/routes/                  # inference (generate + stream), benchmarks, optimization
│   ├── engines/                     # InferenceEngine interface, registry, manager,
│   │                                #   llamacpp_optimized
│   ├── benchmark/                   # BenchmarkConfig, BenchmarkRunner, metrics, storage
│   └── optimization/                # report builder, GGUF metadata, footprint
├── benchmark/README.md              # benchmark tooling notes
├── frontend/                        # React + Vite dashboard (Studio + Optimization)
├── scripts/                         # download, verify, feasibility, benchmark, report drivers
├── tests/                           # standalone unittest scripts (fake engines, no models)
├── docs/
│   ├── README.md                    # docs index
│   ├── architecture.md              # architecture deep-dive (STEP 15)
│   ├── demo-runbook.md              # Windows-laptop demo script (STEP 15)
│   ├── final-demo-checklist.md      # 3-5 min hackathon demo sequence (STEP 16)
│   ├── final-validation.md          # STEP 16 final validation report
│   ├── optimization.md              # STEP 11 measured evidence
│   ├── arm64-deployment.md          # STEP 14A deployment preparation
│   └── step14b-runbook.md           # ARM64-host runbook (postponed)
├── docker/backend/Dockerfile        # ARM64 CPU-only image (prepared, not run)
├── docker-compose.yml
├── models/gguf/                     # GGUF files (gitignored)
└── results/                         # benchmark records + report (gitignored)
```

## Future work

- Run the ARM64 deployment smoke test (STEP 14B) on a real Linux ARM64 host
  and publish the first Arm64 numbers.
- Benchmark FP16 on feasible hardware (≥16 GiB RAM) to enable a real
  baseline-vs-optimized comparison.
- ONNX Runtime engine, cost estimator, energy profiling, and model registry UI.

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

**Measured on this laptop. Honest about what isn't.**

</div>
