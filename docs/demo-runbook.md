# ArmInferX — Demo Runbook (Windows laptop, no Docker)

This runbook is the exact script for presenting ArmInferX at a hackathon or
demo on the current **Windows 11 (AMD64) laptop**. It requires **no Docker, no
WSL, no ARM64 hardware, and no FP16 model**. Everything runs with the verified
**Q4_K_M llama.cpp** engine.

> Pre-flight assumptions: the repo is on the laptop, `backend/.venv` exists
> with the pinned deps (incl. `llama-cpp-python==0.3.34`), the Q4_K_M GGUF is at
> `models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` (~469 MB), and `frontend/node_modules`
> is installed. Check with the commands below — the **Quick check** step.

---

## 0 · Quick check (30 seconds)

```bash
# From the repo root — all of these must succeed:
backend/.venv/Scripts/python.exe --version          # 3.13.x
backend/.venv/Scripts/python.exe -c "import llama_cpp; print(llama_cpp.__version__)"   # 0.3.34
ls -la models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf  # ~491,400,032 bytes
ls frontend/node_modules >/dev/null && echo "node_modules OK"
```

If `llama_cpp` is missing: `pip install llama-cpp-python==0.3.34` (in the venv).
If the GGUF is missing: copy it from a backup — it is gitignored, so it is not
part of a fresh clone.

---

## 1 · Start the backend

Open **Terminal 1**:

```bash
cd /d/ArmInferX/backend
.venv\Scripts\activate
uvicorn main:app --port 8000
```

Wait for `Application startup complete.` The important detail for the demo:
**the startup log contains no model-load line** — the engine loads lazily on
the first request.

Sanity check (optional, in a second terminal):

```bash
curl -s http://localhost:8000/health
```

Expect `"default_engine": "llamacpp-optimized"` and **`"loaded_engines": []`** —
proof that nothing was loaded at startup.

## 2 · Start the frontend

Open **Terminal 2**:

```bash
cd /d/ArmInferX/frontend
npm run dev
```

Open **http://localhost:3000** in Chrome. You should see the ArmInferX Studio
with the engine selector and a composer.

---

## 3 · Demo script

### 3a · Show lazy loading → engine selection

- Point at the header status chip: `API · http://localhost:8000`.
- The **Inference Engine** selector shows the registered engine:
  **llama.cpp — Q4_K_M** — CPU-only, ~469 MB model footprint, streaming.
- Say: *"The FP16 Transformers baseline proved infeasible on this 7.63 GiB
  machine during the STEP 10A check, so the registry ships the engine that
  actually runs here — llama.cpp Q4_K_M."*

### 3b · Run a short prompt — show streaming + metadata

- Type: `Explain what an AI inference engine is.` (the exact benchmark prompt —
  bonus: it matches the saved evidence) and press Enter.
- **First request** includes the one-time model load (a couple of seconds on
  this laptop — a machine-dependent estimate, **not** a measured benchmark
  value). Say: *"The ~470 MB model loads once, on demand — nothing was loaded at startup."*
- Watch the response **stream in token-by-token** (SSE) with typing dots.
- After completion, call out the **metadata chips** on the response card:
  **Engine** `llamacpp-optimized`, **Runtime** `llama.cpp`,
  **Latency**, **Tokens**, **Tokens/sec**, **TTFT**.
- Call out the **Benchmark · latest run** panel below the conversation: latency,
  TTFT, generated tokens, tokens/sec, memory (MB), CPU (%), timestamp — the
  live record of the run you just made.

### 3c · Show the benchmark evidence API

In Terminal 2 (or 3):

```bash
curl -s "http://localhost:8000/benchmarks?engine_id=llamacpp-optimized" | head -40
curl -s http://localhost:8000/benchmarks/latest
curl -s http://localhost:8000/benchmarks/summary
```

Show that every generation auto-persists a JSON record with latency / TTFT /
memory / CPU / tokens, and that `/benchmarks/summary` averages them.

### 3d · Show the Optimization Dashboard

- Click the **Optimization Dashboard** tab in the header.
- Walk the sections:
  1. **Header badges** — Runtime `llama.cpp`, Engine `llamacpp-optimized`,
     Model `Qwen2.5-0.5B-Instruct Q4_K_M`, Platform `Windows CPU`,
     Benchmark status **Measured**.
  2. **Optimization Summary** — mean latency 11.28 s, median 10.31 s, P90
     14.02 s, mean TTFT 156.5 ms, 5.77 tok/s, peak memory 2460.3 MB,
     CPU 535.8%, 64 generated tokens.
  3. **Storage Footprint Reduction** — FP16 ~6485.6 MB vs Q4_K_M ~2007.4 MB,
     **69.05%** reduction. Read the note aloud: *"This is storage footprint
     only — it is not a measured inference speedup."*
  4. **FP16 Baseline: Not Feasible** banner — 7.63 GiB total RAM, watchdog
     aborted at 159–263 MB available RAM. FP16 badge: **NOT MEASURED**.
  5. **Measured vs Not Measured** — the two columns side by side.
  6. **Benchmark Run History** — the 5 timed runs table.
  7. **Benchmark Configuration** — warmup 1, runs 5, max tokens 64, greedy,
     no chat template, 1024 ctx, 2 threads, 0 GPU layers.

### 3e · Explain the two honest caveats (rehearse these)

**Why is FP16 marked "Not measured"?**
The Transformers FP16 baseline could not complete inference on this machine:
it has **7.63 GiB total RAM**, and during the STEP 10A feasibility check the
available RAM dropped to **159–263 MB** while loading the ~6.2 GB FP16 weights.
The watchdog aborted the run safely. So there is no FP16 latency, throughput,
or TTFT — and therefore **no speedup percentage is claimed**. The dashboard
renders these as "Not measured" on purpose.

**What about ARM64?**
STEP 14A prepared the ARM64 **Docker deployment** (multi-stage ARM64 image,
llama-cpp-python built from source with `GGML_NATIVE=OFF GGML_OPENMP=ON`, the
Q4_K_M GGUF bind-mounted read-only). STEP 14B — the actual deployment smoke run
on a real Linux ARM64 host — is **intentionally postponed**: there is no ARM64
host available and Docker is not installed on this laptop. No Arm64 performance
numbers exist, and none are claimed.

### 3f · Optional closer — the report artifact

Show the source of truth:

```bash
curl -s http://localhost:8000/optimization/report | backend/.venv/Scripts/python.exe -m json.tool | head -60
```

Highlight the `"measured"` and `"not_measured"` blocks and `"generated_at"`.

---

## 4 · Teardown

- `Ctrl+C` the frontend (`npm run dev`) and the backend (uvicorn).
- Optional: verify no model is resident — the backend process exiting releases
  the ~0.6 GB of loaded model memory.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Frontend shows "Could not reach the backend" | Backend not started or wrong port; start uvicorn on 8000. |
| `/generate` returns 400 "Unknown engine id" | The selector sends `engine_id`; if it drifted, refresh the page. |
| First generation slow (includes one-time load) | Expected: the first request loads Q4_K_M (~0.6 GB RSS for the 0.5B model). Subsequent prompts skip the load (lifecycle difference only — not a measured performance claim). |
| Optimization Dashboard loads from "bundled report copy" | Backend is down; the static `frontend/public/optimization-report.json` is served instead. |
| `npm run dev` fails | `npm install` in `frontend/` first. |

## What this demo deliberately does NOT do

- Loads or benchmarks the FP16 Transformers baseline (removed from the engine
  registry — infeasible on this machine, documented limitation).
- Runs Docker or WSL.
- Makes any Arm64 or FP16-vs-Q4_K_M performance claim.
- Re-runs the 5-repeat benchmark (that takes ~10+ minutes and adds new records).
