# ArmInferX — Final Demo Checklist (3–5 minutes)

A tight, rehearsed sequence for the hackathon demo. Everything runs on the
Windows laptop with the **Q4_K_M llama.cpp** engine — no Docker, no ARM64, no
FP16 model. The two servers must already be running (see
[`demo-runbook.md`](demo-runbook.md) for the start commands).

> **Before you start:** backend on `:8000`, frontend on `:3000`, browser at
> http://localhost:3000. Keep `curl` handy in a third terminal.

---

## A · Project introduction (15 s)
> *"ArmInferX — an AI Inference Optimization Studio for Arm64 cloud. Today I'll
> show a quantized llama.cpp engine running real CPU inference with an honest,
> measurement-first evidence pipeline."*

## B · Problem (20 s)
> *"Teams deploying LLMs to Arm64 cloud hardware can't compare runtimes and
> quantizations with a standardized, reproducible procedure — and dashboards
> usually overclaim performance. We built a studio that measures honestly."*

## C · ArmInferX solution (20 s)
- Point at the header: **Studio** (live inference) + **Optimization Dashboard**
  (evidence). Header chip: `API · http://localhost:8000`.
- *"A FastAPI backend selects from a pluggable engine registry; nothing loads
  at startup. Every generation is auto-measured and persisted."*

## D · Engine selector (15 s)
- Show the selector: **llama.cpp — Q4_K_M** (the registered engine).
- *"The FP16 Transformers baseline was evaluated during the feasibility check
  and proved infeasible on this machine's RAM, so the registry ships the
  engine that actually runs here — llama.cpp Q4_K_M."*

## E · Live llama.cpp inference (25 s)
- Type: `Explain what an AI inference engine is.` → **Generate**.
- *"This is the exact prompt from our saved benchmark. First request also
  includes the one-time ~2 GB model load."*

## F · Streaming response (15 s)
- Watch tokens stream in (SSE). *"Each token arrives as a Server-Sent Event;
  the UI renders it live, then fills in the metadata."*

## G · Benchmark metadata (20 s)
- Point at the response chips: **Engine** `llamacpp-optimized`, **Runtime**
  `llama.cpp`, **Latency**, **Tokens**, **Tokens/sec**, **TTFT**.
- Point at **Benchmark · latest run** panel (memory, CPU, timestamp).
- Optional `curl`:
  ```bash
  curl -s http://localhost:8000/benchmarks?engine_id=llamacpp-optimized
  ```

## H · Optimization Dashboard (30 s)
- Click **Optimization Dashboard**. Show the header badges (runtime, engine,
  model, platform, **Measured**) and the summary cards:
  mean latency **11.28 s**, mean TTFT **156.5 ms**, **5.77 tok/s**,
  peak memory **2460.3 MB**, CPU **535.8%**, 64 generated tokens.

## I · Q4_K_M footprint reduction (20 s)
- Show **Storage Footprint Reduction**: FP16 ~6485.6 MB → Q4_K_M ~2007.4 MB.
- *"**69.05% storage reduction** — and note the disclaimer: this is footprint
  only, it is not a claimed inference speedup."*

## J · FP16 feasibility limitation (20 s)
- Show the **FP16 Baseline: Not Feasible** banner.
- *"The machine has 7.63 GiB RAM; the feasibility watchdog aborted during FP16
  loading when free RAM hit 159–263 MB. So the dashboard shows **Not measured**
  — never zero, never invented."*

## K · ARM64 deployment preparation (15 s)
- *"The ARM64 Docker image is prepared — llama.cpp compiled for aarch64,
  model bind-mounted, smoke script ready. STEP 14B is deliberately postponed
  until a real ARM64 host is available; we make no Arm64 claims yet."*

## L · Honest conclusion (20 s)
- Show **Measured vs Not Measured** and **Evidence & Limitations**.
- *"Everything you saw is measured on this laptop: latency, TTFT, throughput,
  memory, CPU, footprint. What we could not measure — FP16 speedup, Arm64
  numbers — is labeled as such. That's the boundary that makes these numbers
  trustworthy."*

---

## Timing budget
A–D ≈ 1:10 · E–G ≈ 1:00 · H–I ≈ 0:50 · J–L ≈ 0:55 → **~3:55 total**.

## Failure fallbacks
| Failure | Recovery |
|---|---|
| Backend not reachable | Restart uvicorn; dashboard falls back to the bundled report copy |
| Frontend not loading | `npm run dev` in `frontend/`; verify port 3000 |
| First generation stalls | Wait — one-time model load; check CPU activity |
| Judge asks "what about FP16?" | Read the feasibility banner verbatim (RAM limitation, watchdog abort, not measured) |
| Judge asks "Arm64 numbers?" | STEP 14A prepared, STEP 14B postponed pending a real aarch64 host |
