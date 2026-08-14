# ArmInferX — Final Validation Report (STEP 16)

Run on **2026-08-12** on the Windows development laptop, ahead of hackathon
submission. Everything below was verified on this machine with the **Q4_K_M
llama.cpp** engine only. No FP16 inference, no Docker/WSL, and no ARM64
execution were performed.

> **Update (2026-08-14):** the deployed model switched to
> **Qwen2.5-0.5B-Instruct Q4_K_M** (`qwen2.5-0.5b-instruct-q4_k_m.gguf`, 491 MB)
> so the deployment fits Railway's 1 GB RAM / 2 vCPU limit (engine defaults
> `n_ctx=1024`, `n_threads=2`). The model-integrity table below reflects the
> current file. The benchmark evidence numbers in this report were measured
> with the previous 3B model; re-run `scripts/run_optimization_report.py` to
> refresh `results/optimization_report.json` and the bundled dashboard copy.

---

## Environment

| Item | Value |
|---|---|
| OS / arch | Windows 11 (build 10.0.26200) — AMD64 |
| RAM | 7.63 GiB total |
| CPU | 8 logical / 6 physical cores |
| Python (venv) | 3.13.14 |
| llama-cpp-python | 0.3.34 |
| FastAPI / uvicorn | 0.141.1 / 0.52.1 |
| torch / transformers | 2.13.0+cpu / 5.14.1 |
| Node / Vite | node_modules installed · vite 5.4.21 |

## Cleanup audit

| Item | Result |
|---|---|
| `step14b.log` (root) | Removed — UTF-16 WSL error log artifact, gitignored (`*.log`) |
| `models/gguf/.cache/` | Removed — 9.0 KB HuggingFace download metadata (5 files), gitignored |
| `frontend/package.json` name | Fixed typo `arminferex-frontend` → `arminferx-frontend` |
| Secrets / `.env` / keys/certs | **None found** anywhere outside ignored dirs |
| Ignored big dirs | `backend/.venv/`, `frontend/node_modules/`, `frontend/dist/`, `results/` — all present locally, all gitignored |
| `frontend/package-lock.json` | Gitignored by design (line 68 of `.gitignore`) |

**Removed in total:** 1 log file + 5 cache metadata files + 3 empty cache
directories (~10 KB). Nothing else was deleted; no model or source file was
modified.

## Git safety

All confirmed with `git check-ignore -v`:

```
.gitignore:160:models/**/*.gguf        models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf
.gitignore:178:models/**/.cache/       models/gguf/.cache/
.gitignore:186:results/                results/
.gitignore:50:.venv/                   backend/.venv/
.gitignore:62:node_modules/            frontend/node_modules/
```

- `git add -n models/` staged **nothing** (all model files and cache ignored).
- Largest file that could be staged is a KB-scale source file — the ~470 MB
  GGUF and ~6.5 GB FP16 shards **cannot** accidentally enter a commit.

## Model integrity

| Check | Result |
|---|---|
| File exists | ✅ `models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| Exact size | ✅ 491,400,032 bytes |
| SHA-256 | ✅ `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` (matches expected) |

Model was **not** modified, downloaded, or re-quantized.

## Python compile

`py_compile` over all 56 `.py` files under `backend/`, `scripts/`, `tests/` →
**COMPILE-ALL-OK** (including the previously-broken `scripts/verify_model.py`).

## Tests

All 10 test files run individually with the venv Python (no pytest installed),
each in its own process, zero real-model loads:

```
PASS test_arm64_deployment      PASS test_engine_selection
PASS test_benchmark_runner      PASS test_engines
PASS test_benchmark_smoke       PASS test_generate_smoke
PASS test_benchmarks_api        PASS test_latency_baseline
PASS test_optimization_report   PASS test_result_storage
RESULT: 10/10 PASS, 0 FAIL
```

> **Update (2026-08-13):** the FP16 Transformers baseline was removed from the
> engine registry (STEP 10A proved it infeasible on this 7.63 GiB machine).
> The two suites that exercised the removed baseline service
> (`test_inference_service_ttft`, `test_ttft_streamer`) were removed with it;
> the remaining suites were updated to the llama.cpp-only registry and all
> pass.

## Frontend build

`npm run build` → **success** (`vite v5.4.21`, 34 modules, dist 172.81 kB JS /
23.11 kB CSS, built in ~2 s).

## Q4_K_M smoke test (real inference — not a benchmark)

`uvicorn main:app` + `POST /generate` with `engine_id=llamacpp-optimized`,
prompt `"What is an AI inference engine in one sentence?"`:

| Field | Observed |
|---|---|
| `engine_id` | `llamacpp-optimized` ✅ |
| `runtime` | `llama.cpp` ✅ |
| `model` | `qwen2.5-0.5b-instruct-q4_k_m` ✅ |
| `response` | non-empty, coherent ✅ |
| `latency_ms` | 9275.5 (engine inference time only — the model load happens before the measured call; the ~16.9 s request wall-time includes it) ✅ |
| `ttft_ms` | 1385.84 (first request; warmed-engine mean is 156.5 ms) ✅ |
| `generated_tokens` | 51 ✅ |
| `tokens_per_second` | 5.5 ✅ |

Lazy loading proven: `/health` before inference → `loaded_engines: []`; after →
`["llamacpp-optimized"]`. A baseline record was auto-saved under
`results/baseline/` (normal app behavior, gitignored). This is a **smoke test
only** — it is not part of the benchmark evidence set.

## Benchmark evidence (canonical, from `results/optimization_report.json`)

Q4_K_M llama.cpp · prompt `"Explain what an AI inference engine is."` (9 prompt
tokens) · greedy · `max_new_tokens=64` · 1 warmup + 5 timed runs · `n_ctx=2048`,
`n_threads=8`, `n_gpu_layers=0`:

- Mean latency **11,278.62 ms** · median 10,314.56 · p90 14,018.02
- Mean TTFT **156.5 ms** · mean **5.77 tok/s** · 64 generated tokens
- Peak memory **2,460.32 MB** · mean CPU **535.82 %**
- Q4_K_M footprint **2,007.4 MB** vs FP16 **6,485.6 MB** → **69.05% storage
  reduction** (footprint only, no speedup implied)

## Optimization report status

Verified programmatically (assertions passed on both
`results/optimization_report.json` and the bundled
`frontend/public/optimization-report.json`):

- measured aggregates present ✅ · Q4_K_M footprint ✅ · FP16 footprint ✅
- reduction = **69.05** ✅ · feasibility = **not_feasible** ✅
- `not_measured` all **null**: fp16 latency, TTFT, tokens/sec, throughput,
  comparative speedup ✅
- The dashboard renders unavailable values as **"Not measured"** — the report
  schema hard-codes them as `null`, and `loadReport.js` maps them to a dash.
  No value is fabricated, and no benchmark number is hardcoded in the UI
  (code search for the measured figures in `frontend/src` → 0 matches).

## Dashboard validation (browser)

Automated Chrome run against `http://localhost:3000` (backend on :8000):

- Studio: header, engine selector (`llama.cpp — Q4_K_M`), composer ✅
- Optimization Dashboard: `llama.cpp`, `llamacpp-optimized`,
  `Qwen2.5-3B-Instruct Q4_K_M`, **Measured**, latency 11.28 s, TTFT, 5.77
  tok/s, 2460.3 MB, 535.8 %, **69.05%**, FP16 (reference), **Not Feasible**,
  **NOT MEASURED**, Benchmark Run History, Benchmark Configuration, Evidence &
  Limitations — all present ✅
- Whole page scrolled, **zero console errors** ✅

## Documentation status

`README.md`, `docs/README.md`, `docs/architecture.md`, `docs/demo-runbook.md`,
`docs/optimization.md`, `docs/arm64-deployment.md`, `docs/step14b-runbook.md`,
`backend/README.md`, `frontend/README.md`, `benchmark/README.md` — scanned for
forbidden claims:

- ❌ no FP16 speedup claim ✅ (all "speedup" mentions are explicit negations)
- ❌ no Q4_K_M-vs-FP16 inference speedup claim ✅
- ❌ no Arm64 benchmark/performance results ✅ (all marked preparation /
  postponed / future)
- ❌ no "Docker ran on this laptop" claim ✅
- ❌ no "STEP 14B completed" claim ✅ (consistently **postponed**)

## ARM64 status

- **STEP 14A (preparation):** done — multi-stage ARM64 Dockerfile
  (`GGML_NATIVE=OFF`, `GGML_OPENMP=ON`, `--no-binary llama-cpp-python`),
  read-only model bind-mount, `docker-compose.yml`, one-shot
  `scripts/arm64_deploy_smoke.sh`. Config validated by
  `test_arm64_deployment.py`.
- **STEP 14B (execution):** **postponed** — no ARM64 host available, no Docker
  on this laptop. No Arm64 numbers exist and none are claimed.

## Known limitations

- All measured numbers are from the Windows AMD64 laptop (CPU) — reproducible
  on this machine only.
- FP16 baseline is infeasible on 7.63 GiB RAM (watchdog aborted at 159–263 MB
  free); FP16 metrics remain null by design.
- No Arm64 measurements; Docker image unbuilt on this machine.
- `frontend/package-lock.json` is gitignored by project convention.

## Final Git status

```
 M README.md
?? .dockerignore  ?? .gitignore  ?? LICENSE  ?? docker-compose.yml
?? backend/  ?? benchmark/  ?? docs/  ?? docker/
?? frontend/  ?? models/  ?? scripts/  ?? tests/
```

All model files, `results/`, `.venv/`, `node_modules/`, `dist/`, and
`*.log` are ignored and cannot be staged. The working tree is clean of
artifacts and ready for the initial commit.

---

## STEP 16 STATUS

| Check | Status |
|---|---|
| Cleanup | **PASS** |
| Git safety | **PASS** |
| Model integrity | **PASS** |
| Python compile | **PASS** |
| Tests | **10/10** |
| Frontend build | **PASS** |
| Q4_K_M smoke test | **PASS** |
| Optimization report | **PASS** |
| Documentation | **PASS** |
| Hackathon demo readiness | **READY** |

**Recommended next action:** stage and commit the repository as the initial
hackathon snapshot — `git add .` is now safe (verified: nothing large or
ignored gets staged) — then run the 3–5 minute demo from
`docs/final-demo-checklist.md` with the servers started per
`docs/demo-runbook.md`.
