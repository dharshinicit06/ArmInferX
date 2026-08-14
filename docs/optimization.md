# ArmInferX — Optimization Evidence (STEP 11)

This document records the **measured** optimization evidence for the Arm AI
Optimization Challenge. It states facts only: everything here was measured on
this machine or is explicitly labeled as a hardware limitation. No claim of
"X% faster" or "Y× better" is made — the FP16 baseline completed no inference
on this hardware, so no performance comparison between FP16 and Q4_K_M
exists.

## The optimization story

```
Qwen2.5-3B-Instruct (FP16, HF safetensors)
        ↓
large memory footprint (~6.2 GB fp16 weights)
        ↓
FP16 baseline infeasible on this 7.63 GiB RAM laptop
        ↓      (STEP 10A feasibility watchdog aborted safely)
Qwen2.5-0.5B-Instruct Q4_K_M (GGUF)
        ↓
smaller model footprint (storage reduction measured)
        ↓
llama.cpp CPU runtime (llama-cpp-python 0.3.34, CPU-only)
        ↓
successful inference (Q4_K_M)
        ↓
5-run reproducible benchmark (BenchmarkRunner)
```

## Measured facts

| Fact | Value | Source |
|---|---|---|
| Q4_K_M GGUF file size | see `results/optimization_report.json` | file `stat()` |
| FP16 GGUF shards total | see report | file `stat()` |
| Storage reduction (absolute / %) | see report | `optimization/model_footprint.py` |
| Q4_K_M mean latency | see report | `BenchmarkRunner` aggregates |
| Q4_K_M mean TTFT | see report | `BenchmarkRunner` aggregates |
| Q4_K_M mean tokens/sec | see report | `BenchmarkRunner` aggregates |
| Q4_K_M peak memory / mean CPU | see report | `BenchmarkService` samplers |
| Benchmark repetitions | 5 timed + 1 warmup | `BenchmarkConfig` |

Regenerate the machine-readable report (this also re-runs the benchmark):

```bash
backend/.venv/Scripts/python scripts/run_optimization_report.py
```

Output: `results/optimization_report.json` (gitignored).

## Hardware limitation (not a measurement)

The Transformers FP16 baseline **could not run inference** on this machine:

- Total RAM: 7.63 GiB.
- The STEP 10A feasibility check (`scripts/feasibility_baseline.py`) aborted
  during model loading when system available RAM dropped to 159–263 MB
  (severe-paging alarm; the watchdog stopped the run safely).
- Therefore the following were **not measured**: FP16 inference latency,
  FP16 TTFT, FP16 throughput, and any comparative speedup of Q4_K_M over
  FP16. Do not derive a percentage improvement from this document.

## Methodology (reproducibility)

- Engine: `llamacpp-optimized` (`llama.cpp` runtime) loaded via
  `engines.registry.load_engine`.
- Model: `models/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf` (default engine path).
- Engine defaults: `n_ctx=1024`, `n_threads=2`, `n_gpu_layers=0`, greedy
  decoding (`temperature=None` → llama.cpp default 0.0) — tuned for 1 GB RAM /
  2 vCPU hosts such as Railway.
- Procedure: `BenchmarkConfig(prompt="Explain what an AI inference engine is.",
  max_new_tokens=64, temperature=None, chat_template=False, warmup=1,
  repeats=5)` executed by `BenchmarkRunner` → records under
  `results/benchmarks/llamacpp-optimized/`.
- Model metadata (architecture, GGUF version, file type, tensor breakdown,
  parameter count, SHA-256) is parsed from the actual GGUF file bytes by
  `optimization.gguf_metadata` — never assumed from the filename.

## Notes

- GGUF's `general.file_type` enum reports `Q4_K` (code 11); the `_M` variant
  suffix is not part of the enum. The `Q4_K_M` label comes from the filename;
  the per-tensor quantization breakdown comes from the file.
- Results were produced on a Windows development laptop (CPU). They are
  reproducible on this machine and make no claim about Arm64 cloud
  performance (that is a later phase).
