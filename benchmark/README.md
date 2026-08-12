# Benchmark

Inference benchmarking harnesses and profiling tools for Arm64 targets.

## Current status

The modular benchmarking subsystem lives in **`backend/benchmark/`**:

- `metrics.py` — `BenchmarkMetrics` result object (timestamp, latency_ms,
  ttft_ms, memory_mb, cpu_percent, generated_tokens, tokens_per_second) +
  the `compute_tokens_per_second()` derivation + psutil-based system samplers.
- `logger.py` — dedicated benchmark logging.
- `benchmark_service.py` — `BenchmarkService.run(service, prompt)` measures a
  generation and returns a `BenchmarkMetrics` object. `measure(call)` also
  derives `tokens_per_second` from the result's `generated_tokens` and the
  same latency value that gets persisted as `latency_ms`, and surfaces a
  result-side `ttft_ms` (time-to-first-token) separately from total latency.

Token counting is tokenizer-native: the inference service counts the actual
output token IDs emitted by `model.generate()` (no character/word estimation),
and throughput is `generated_tokens / (latency_ms / 1000)`. TTFT is measured
by an instrumentation streamer attached to `model.generate()` (first token
observed = prefill + first decode step) and does not change the generated
output. Design stays modular so more metrics (throughput by phase) can be
added later via `BenchmarkMetrics.extra`.

Current optimization state: the **Q4_K_M llama.cpp engine** is implemented,
validated, and benchmarked on this laptop (see `docs/optimization.md` and the
Optimization Dashboard). The FP16 Transformers baseline is not feasible on
this 7.63 GiB machine — no FP16-vs-Q4_K_M performance comparison is claimed.

Usage from the repo root:

```bash
backend/.venv/Scripts/python -m py_compile backend/benchmark/*.py
backend/.venv/Scripts/python tests/test_benchmark_smoke.py
```

## Future

- Benchmark engine: repeated runs, aggregations (mean/p50/p95), sequence
  length and batch sweeps.
- Comparison of baseline vs. optimized (ONNX, llama.cpp) pipelines.
